"""
build_polytopes_cnn.py

Builds linear polytope constraints (Ax + b <= 0) for CNNs composed of
Conv2d, MaxPool2d, ReLU, Linear, Flatten, and Dropout layers.

Adapts the message-passing / shortcut-weight approach from:
    RoundingErrorEstimation/appmax/neurons.py  (student code, Jiri et al.)
adjusting conventions and interface to match this codebase.

Key ideas
---------
A *message* carries three quantities that evolve as it passes through the
network, for a fixed reference sample x_0:

    sample   : the actual forward pass value (shape matches layer output)
    s_weight : how the *input* maps to the current layer's pre-activation
               values; shape (input_flat_dim, *current_spatial_shape)
    s_bias   : the bias accumulated up to the current layer

At a ReLU, unsaturated neurons (sample >= 0) produce:
    s_weight[i] @ x_flat + s_bias[i] >= 0   → -(Ax + b) <= 0
At a MaxPool2d, non-maximum positions produce:
    s_weight[other] @ x_flat + s_bias[other] <= s_weight[max] @ x_flat + s_bias[max]
    →  (s_weight[other] - s_weight[max]) @ x_flat + (s_bias[other] - s_bias[max]) <= 0

Convention throughout: Ax + b <= 0  (same as build_polytopes.py)

Usage (from project root):
    python -m src.optim.build_polytopes_cnn

Attribution
-----------
The Message / Constraints dataclasses and the collect_* functions are adapted
from RoundingErrorEstimation/appmax/neurons.py (Jiri et al., 2025).
The key logic for Conv2d and MaxPool2d constraints is unchanged; the interface
and output convention differ.
"""

import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================== #
# Message and Constraints (adapted from appmax/neurons.py)                    #
# =========================================================================== #


@dataclass
class _Message:
    """
    Carries the three quantities that propagate through the network.

    Attributes
    ----------
    sample   : current layer output (batch-dim 1, e.g. shape (1, C, H, W))
    s_weight : shortcut-weight tensor; shape (input_flat_dim, *current_shape)
               so that  sample = x_flat @ s_weight.flatten(0) + s_bias  holds
               (but kept in spatial form for Conv2d / MaxPool2d compatibility)
    s_bias   : accumulated bias; same spatial shape as sample
    """

    sample:   torch.Tensor
    s_weight: torch.Tensor
    s_bias:   torch.Tensor

    def __init__(self, sample: torch.Tensor):
        """sample must be a single un-batched tensor, e.g. shape (C, H, W) or (D,)."""
        self.sample   = sample.unsqueeze(0)
        self.s_weight = torch.eye(
            sample.numel(), dtype=sample.dtype, device=sample.device
        ).reshape(-1, *sample.shape)
        self.s_bias   = torch.zeros_like(self.sample)

    def apply(self, module: nn.Module) -> "_Message":
        self.sample   = module(self.sample)
        self.s_weight = module(self.s_weight)
        self.s_bias   = module(self.s_bias)
        return self


@dataclass
class _Constraints:
    """
    Accumulates per-layer polytope constraints.

    U_weight[l], U_bias[l]  →  unsaturated (active ReLU):  Wx + b  >= 0
    S_weight[l], S_bias[l]  →  saturated   (inactive / max pool):  Wx + b <= 0
    """
    U_weight: list = field(default_factory=list)
    U_bias:   list = field(default_factory=list)
    S_weight: list = field(default_factory=list)
    S_bias:   list = field(default_factory=list)


# =========================================================================== #
# Layer-specific collect functions (adapted from appmax/neurons.py)           #
# =========================================================================== #


@torch.no_grad()
def _collect(module: nn.Module, msg: _Message, cst: _Constraints) -> _Message:
    """Pass msg through module and accumulate constraints into cst."""
    match module:
        case nn.Sequential():
            for sub in module:
                msg = _collect(sub, msg, cst)
            return msg
        case nn.ReLU():
            return _collect_relu(msg, cst)
        case nn.Linear():
            return _collect_linear(module, msg)
        case nn.Conv2d():
            return _collect_conv2d(module, msg)
        case nn.MaxPool2d():
            return _collect_max_pool2d(module, msg, cst)
        case nn.Flatten():
            return msg.apply(module)
        case nn.Dropout():
            if module.training:
                warnings.warn("Dropout is in training mode during constraint collection.")
            return msg
    raise NotImplementedError(
        f"_collect is not implemented for '{type(module).__name__}'. "
        "Convert your model to an nn.Sequential of supported layers."
    )


def _collect_relu(msg: _Message, cst: _Constraints) -> _Message:
    unsaturated    = msg.sample >= 0
    unsaturated_sq = unsaturated.squeeze(0)   # drop batch dim for indexing

    # s_weight_T[neuron, input_flat] — rows are per-neuron shortcut weights
    s_weight_T = msg.s_weight.movedim(0, -1)

    cst.U_weight.append(s_weight_T[ unsaturated_sq])
    cst.U_bias.append(  msg.s_bias[ unsaturated].flatten())
    cst.S_weight.append(s_weight_T[~unsaturated_sq])
    cst.S_bias.append(  msg.s_bias[~unsaturated].flatten())

    msg.sample   = F.relu(msg.sample)
    msg.s_weight = msg.s_weight * unsaturated
    msg.s_bias   = msg.s_bias   * unsaturated
    return msg


def _collect_linear(linear: nn.Linear, msg: _Message) -> _Message:
    msg.sample   = linear(msg.sample)
    msg.s_weight = F.linear(msg.s_weight, linear.weight, None)
    msg.s_bias   = linear(msg.s_bias)
    return msg


def _collect_conv2d(conv: nn.Conv2d, msg: _Message) -> _Message:
    msg.sample   = conv(msg.sample)
    msg.s_weight = conv._conv_forward(msg.s_weight, conv.weight, None)
    msg.s_bias   = conv(msg.s_bias)
    return msg


def _batch_channels_take(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select max-pool positions from data using indices (same as appmax version)."""
    B, C     = data.shape[:2]
    flat_idx = indices.flatten(2).expand(B, -1, -1)
    gathered = torch.gather(data.flatten(2), dim=2, index=flat_idx)
    return gathered.reshape(B, C, *indices.shape[2:])


def _collect_max_pool2d(pool: nn.MaxPool2d, msg: _Message, cst: _Constraints) -> _Message:
    if pool.ceil_mode:
        raise NotImplementedError("ceil_mode=True is not supported.")

    _, C, M, N = msg.sample.shape

    msg.sample, idx_max = F.max_pool2d(
        msg.sample,
        pool.kernel_size, pool.stride, pool.padding, pool.dilation,
        return_indices=True,
    )

    # s_weight_T[channel, pixel, input_flat]
    s_weight_T = msg.s_weight.movedim(0, -1).flatten(1, 2)
    # s_bias_sq[channel, pixel]
    s_bias_sq  = msg.s_bias.reshape(C, -1)

    # All pixel indices within each pooling window
    idx_all = F.unfold(
        torch.arange(M * N, device=msg.sample.device).reshape(1, 1, M, N).float(),
        kernel_size=pool.kernel_size, stride=pool.stride,
        padding=pool.padding, dilation=pool.dilation,
    ).long().movedim(1, -1)                  # (1, n_windows, window_cells)
    window_cells = idx_all.shape[-1]

    idx_all_sq = idx_all.flatten().repeat(C)
    idx_max_sq = idx_max.flatten().repeat_interleave(window_cells)
    useful     = idx_max_sq != idx_all_sq    # non-max positions only
    idx_max_sq = idx_max_sq[useful]
    idx_all_sq = idx_all_sq[useful]
    channels   = torch.arange(C, device=msg.sample.device).repeat_interleave(
        idx_max_sq.shape[0] // C
    )

    # other + o_bias <= max + m_bias  →  (other - max)·x + (ob - mb) <= 0
    cst.S_weight.append(
        s_weight_T[channels, idx_all_sq] - s_weight_T[channels, idx_max_sq]
    )
    cst.S_bias.append(
        s_bias_sq[channels, idx_all_sq] - s_bias_sq[channels, idx_max_sq]
    )

    msg.s_weight = _batch_channels_take(msg.s_weight, idx_max)
    msg.s_bias   = _batch_channels_take(msg.s_bias,   idx_max)
    return msg


# =========================================================================== #
# Convert non-sequential CNNs to nn.Sequential                                #
# =========================================================================== #

def _fx_to_sequential(model: nn.Module) -> nn.Sequential:
    """
    Use torch.fx symbolic tracing to convert a sequential-style CNN to nn.Sequential.

    Handles:
    - call_module  : sub-modules (Conv2d, Linear, MaxPool2d, Dropout, …)
    - call_function: F.relu / torch.relu → nn.ReLU()
                     torch.flatten       → nn.Flatten(start_dim)
    - call_method  : .view() / .flatten() / .reshape() → nn.Flatten(1)
                     .relu()                            → nn.ReLU()

    Nodes not mappable to a layer (size(), getitem, arithmetic, …) are silently
    skipped — they must not carry any information that affects the network output
    shape (e.g. pure shape queries used only as view() arguments are fine).
    """
    try:
        traced = torch.fx.symbolic_trace(model)
    except Exception as exc:
        raise RuntimeError(
            f"torch.fx symbolic tracing failed for {type(model).__name__}: {exc}.\n"
            "Either restructure the model as nn.Sequential, or pass a custom "
            "to_seq_fn to build_two_class_cnn_polytopes / build_cnn_all_polytopes."
        ) from exc

    named_mods = dict(model.named_modules())
    layers: list[nn.Module] = []

    for node in traced.graph.nodes:
        if node.op == "call_module":
            layers.append(named_mods[node.target])

        elif node.op == "call_function":
            fn = node.target
            if fn in (F.relu, torch.relu):
                layers.append(nn.ReLU())
            elif fn is torch.flatten:
                start = (node.args[1] if len(node.args) > 1
                         else node.kwargs.get("start_dim", 1))
                layers.append(nn.Flatten(start))
            # Other functions (arithmetic, size queries, …) → skip

        elif node.op == "call_method":
            meth = node.target
            if meth in ("view", "reshape", "flatten"):
                layers.append(nn.Flatten(1))
            elif meth == "relu":
                layers.append(nn.ReLU())
            # Other methods → skip

        # "placeholder", "output", "get_attr" → skip

    return nn.Sequential(*layers)


def model_to_sequential(model: nn.Module) -> nn.Sequential:
    """
    Convert any sequential-style CNN to an nn.Sequential compatible with _collect.

    Tries, in order:
    1. Model is already an nn.Sequential → return as-is.
    2. Model has a `.layers` / `.net` / `.features` / `.model` attribute that is
       an nn.Sequential → return that attribute.
    3. Fall back to torch.fx symbolic tracing via _fx_to_sequential.

    For custom architectures where FX tracing fails (dynamic control flow, etc.),
    pass a custom ``to_seq_fn`` to the public API functions instead.
    """
    if isinstance(model, nn.Sequential):
        return model

    for attr in ("layers", "net", "features", "model"):
        sub = getattr(model, attr, None)
        if isinstance(sub, nn.Sequential):
            return sub

    return _fx_to_sequential(model)


# =========================================================================== #
# Convert _Constraints + s_weight/s_bias → (A, b) in our convention Ax+b<=0  #
# =========================================================================== #

def _constraints_to_Ab(cst: _Constraints,
                        s_weight: torch.Tensor,
                        s_bias:   torch.Tensor,
                        c: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Assemble the full polytope constraint matrix (A, b) in Ax + b <= 0.

    Activation constraints
    ----------------------
    Unsaturated (ReLU active, activation >= 0):
        U_weight[l] @ x + U_bias[l] >= 0
        → A = -cat(U_weight),  b = -cat(U_bias)

    Saturated (ReLU inactive or max-pool non-max, activation <= 0):
        S_weight[l] @ x + S_bias[l] <= 0
        → A =  cat(S_weight),  b =  cat(S_bias)

    Classification constraints
    --------------------------
    s_weight has shape (input_flat_dim, n_classes);
        output[k] = x_flat @ s_weight[:, k] + s_bias[k]
    For class c to win over j:
        output[j] - output[c] <= 0
        → A_cls[j] = (s_weight[:, j] - s_weight[:, c])
          b_cls[j] = (s_bias[j]  - s_bias[c])
    """
    parts_A, parts_b = [], []

    # --- Unsaturated ---
    if cst.U_weight:
        A_U = -torch.cat(cst.U_weight, dim=0)
        b_U = -torch.cat(cst.U_bias,   dim=0)
        parts_A.append(A_U)
        parts_b.append(b_U)

    # --- Saturated ---
    if cst.S_weight:
        A_S = torch.cat(cst.S_weight, dim=0)
        b_S = torch.cat(cst.S_bias,   dim=0)
        parts_A.append(A_S)
        parts_b.append(b_S)

    # --- Classification ---
    # s_weight: (input_flat_dim, n_classes),  s_bias: (1, n_classes)
    sw  = s_weight                            # (input_flat_dim, n_classes)
    sb  = s_bias.squeeze(0)                   # (n_classes,)  — drop batch dim only
    n_classes = sw.shape[1]
    mask = torch.arange(n_classes) != c
    A_cls = (sw[:, mask] - sw[:, c:c+1]).T   # (n_classes-1, input_flat_dim)
    b_cls = (sb[mask]    - sb[c])             # (n_classes-1,)
    parts_A.append(A_cls)
    parts_b.append(b_cls)

    A = torch.cat(parts_A, dim=0)
    b = torch.cat(parts_b, dim=0)
    return A, b


# =========================================================================== #
# Public API                                                                   #
# =========================================================================== #

def build_two_class_cnn_polytopes(
    model: nn.Module,
    qmodel: nn.Module,
    x: torch.Tensor,
    c: int,
    to_seq_fn=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build two polytopes for a CNN, matching the MLP convention of
    build_two_class_polytopes in build_polytopes.py.

    1) correct_polytope (A_correct, b_correct):
        - activation constraints from model AND qmodel
        - model classifies x' as class c

    2) correct_and_qcorrect_polytope (A_both, b_both):
        - same activation constraints as A_correct
        - qmodel also classifies x' as class c

    Convention: Ax + b <= 0

    Parameters
    ----------
    model : nn.Module — full-precision CNN, eval mode
    qmodel : nn.Module — quantized CNN, eval mode
    x : torch.Tensor — shape (1, C, H, W) or (C, H, W)
    c : int — true class label
    to_seq_fn : callable or None

    Returns
    -------
    A_correct, b_correct : torch.Tensor
    A_both, b_both : torch.Tensor
    """
    if to_seq_fn is None:
        to_seq_fn = model_to_sequential

    x_sample = x.squeeze(0) if x.dim() == 4 else x  # (C, H, W)

    # --- Model constraints (computed once) ---
    seq_model = to_seq_fn(model)
    cst_model = _Constraints()
    msg_model = _Message(x_sample)
    with torch.no_grad():
        msg_model = _collect(seq_model, msg_model, cst_model)

    # --- Qmodel constraints (computed once) ---
    seq_q = to_seq_fn(qmodel)
    cst_q = _Constraints()
    msg_q = _Message(x_sample)
    with torch.no_grad():
        msg_q = _collect(seq_q, msg_q, cst_q)

    # --- Combined activation constraints (model + qmodel) ---
    cst_both = _Constraints(
        U_weight = cst_model.U_weight + cst_q.U_weight,
        U_bias   = cst_model.U_bias   + cst_q.U_bias,
        S_weight = cst_model.S_weight + cst_q.S_weight,
        S_bias   = cst_model.S_bias   + cst_q.S_bias,
    )
    A_act, b_act = _activation_constraints_to_Ab(cst_both)

    # --- Classification constraints ---
    A_cls_m, b_cls_m = _class_constraints_to_Ab(msg_model.s_weight, msg_model.s_bias, c)
    A_cls_q, b_cls_q = _class_constraints_to_Ab(msg_q.s_weight,     msg_q.s_bias,     c)

    # A_correct : model act + qmodel act + model cls
    A_correct = torch.cat([A_act, A_cls_m], dim=0)
    b_correct = torch.cat([b_act, b_cls_m], dim=0)

    # A_both : A_correct + qmodel cls
    A_both = torch.cat([A_correct, A_cls_q], dim=0)
    b_both = torch.cat([b_correct, b_cls_q], dim=0)

    return A_correct, b_correct, A_both, b_both


def build_cnn_all_polytopes(
    model: nn.Module,
    qmodels_dict: dict,
    x: torch.Tensor,
    c: int,
    to_seq_fn=None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Build three nested polytopes per quantized model in a single pass,
    computing model shortcuts only once.

    Three polytopes per bit-width, with the containment chain
    A_base ⊇ A_correct ⊇ A_both:

    A_base (model-only reference, shared across all bit-widths):
        model activation + model classification

    A_correct (per bit-width):
        model activation + model classification + qmodel activation
        = A_base + qmodel activation

    A_both (per bit-width):
        model activation + model classification + qmodel activation
        + qmodel classification
        = A_correct + qmodel classification  (differ by exactly 9 constraints)

    Convention: Ax + b <= 0

    Parameters
    ----------
    model : nn.Module
    qmodels_dict : dict {bits (int): qmodel (nn.Module)}
    x : torch.Tensor  — shape (1, C, H, W) or (C, H, W)
    c : int
    to_seq_fn : callable or None

    Returns
    -------
    A_base : torch.Tensor  (n_base, input_flat_dim)
    b_base : torch.Tensor  (n_base,)
    polytopes_dict : dict {bits: (A_correct, b_correct, A_both, b_both)}
    """
    if to_seq_fn is None:
        to_seq_fn = model_to_sequential

    x_sample = x.squeeze(0) if x.dim() == 4 else x  # (C, H, W)

    # --- Model shortcuts — computed ONCE ---
    seq_model = to_seq_fn(model)
    cst_model = _Constraints()
    msg_model = _Message(x_sample)
    with torch.no_grad():
        msg_model = _collect(seq_model, msg_model, cst_model)

    # A_base: model-only (independent of quantization)
    A_base, b_base = _constraints_to_Ab(cst_model, msg_model.s_weight, msg_model.s_bias, c)

    # --- One pair (A_correct, A_both) per bit-width ---
    polytopes_dict = {}
    for bits, qmodel in qmodels_dict.items():
        seq_q = to_seq_fn(qmodel)
        cst_q = _Constraints()
        msg_q = _Message(x_sample)
        with torch.no_grad():
            msg_q = _collect(seq_q, msg_q, cst_q)

        A_act_q, b_act_q = _activation_constraints_to_Ab(cst_q)
        A_cls_q, b_cls_q = _class_constraints_to_Ab(msg_q.s_weight, msg_q.s_bias, c)

        # A_correct: A_base + qmodel activation
        A_correct = torch.cat([A_base, A_act_q], dim=0)
        b_correct = torch.cat([b_base, b_act_q], dim=0)

        # A_both: A_correct + qmodel classification (differ by 9 constraints)
        A_both = torch.cat([A_correct, A_cls_q], dim=0)
        b_both = torch.cat([b_correct, b_cls_q], dim=0)

        polytopes_dict[bits] = (A_correct, b_correct, A_both, b_both)

    return A_base, b_base, polytopes_dict


def build_cnn_all_polytopes_per_class(
    model: nn.Module,
    qmodels_dict: dict,
    x: torch.Tensor,
    c: int,
    to_seq_fn=None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Like build_cnn_all_polytopes, but computes P3(k) for ALL classes k (not just c).

    For each bit-width, returns:
      - A_correct, b_correct  : P2 (same as build_cnn_all_polytopes)
      - per_class_dict        : {k: (A_k, b_k)} where (A_k, b_k) defines
                                P3(k) = A_correct + "qmodel predicts k" constraints

    The existing build_cnn_all_polytopes function is NOT modified.

    Convention: Ax + b <= 0

    Parameters
    ----------
    model : nn.Module
    qmodels_dict : dict {bits (int): qmodel (nn.Module)}
    x : torch.Tensor  — shape (1, C, H, W) or (C, H, W)
    c : int
    to_seq_fn : callable or None

    Returns
    -------
    A_base : torch.Tensor  (n_base, input_flat_dim)
    b_base : torch.Tensor  (n_base,)
    polytopes_dict : dict {bits: (A_correct, b_correct, per_class_dict)}
        per_class_dict : dict {k (int): (A_k, b_k)} for k in range(n_classes)
    """
    if to_seq_fn is None:
        to_seq_fn = model_to_sequential

    x_sample = x.squeeze(0) if x.dim() == 4 else x  # (C, H, W)

    # Model shortcuts — computed ONCE
    seq_model = to_seq_fn(model)
    cst_model = _Constraints()
    msg_model = _Message(x_sample)
    with torch.no_grad():
        msg_model = _collect(seq_model, msg_model, cst_model)

    A_base, b_base = _constraints_to_Ab(cst_model, msg_model.s_weight, msg_model.s_bias, c)

    n_classes = msg_model.s_weight.shape[1]

    polytopes_dict = {}
    for bits, qmodel in qmodels_dict.items():
        seq_q = to_seq_fn(qmodel)
        cst_q = _Constraints()
        msg_q = _Message(x_sample)
        with torch.no_grad():
            msg_q = _collect(seq_q, msg_q, cst_q)

        A_act_q, b_act_q = _activation_constraints_to_Ab(cst_q)

        # A_correct: A_base + qmodel activation  (same as build_cnn_all_polytopes)
        A_correct = torch.cat([A_base, A_act_q], dim=0)
        b_correct = torch.cat([b_base, b_act_q], dim=0)

        # P3(k) for each class k: A_correct + "qmodel predicts k" constraints
        per_class = {}
        for k in range(n_classes):
            A_cls_k, b_cls_k = _class_constraints_to_Ab(msg_q.s_weight, msg_q.s_bias, k)
            per_class[k] = (
                torch.cat([A_correct, A_cls_k], dim=0),
                torch.cat([b_correct, b_cls_k], dim=0),
            )

        polytopes_dict[bits] = (A_correct, b_correct, per_class)

    return A_base, b_base, polytopes_dict


def _activation_constraints_to_Ab(
    cst: _Constraints,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Activation-only part of (A, b) — no classification constraints."""
    parts_A, parts_b = [], []
    if cst.U_weight:
        parts_A.append(-torch.cat(cst.U_weight, dim=0))
        parts_b.append(-torch.cat(cst.U_bias,   dim=0))
    if cst.S_weight:
        parts_A.append( torch.cat(cst.S_weight, dim=0))
        parts_b.append( torch.cat(cst.S_bias,   dim=0))
    if not parts_A:
        # No activation constraints (e.g. linear-only model)
        raise ValueError(
            "_activation_constraints_to_Ab: no activation constraints collected. "
            "The model must contain at least one ReLU or MaxPool2d layer."
        )
    return torch.cat(parts_A, dim=0), torch.cat(parts_b, dim=0)


def _class_constraints_to_Ab(
    s_weight: torch.Tensor,
    s_bias:   torch.Tensor,
    c: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classification-only part of (A, b)."""
    sw      = s_weight                    # (input_flat_dim, n_classes)
    sb      = s_bias.squeeze(0)           # (n_classes,)  — drop batch dim only
    n_cls   = sw.shape[1]
    mask    = torch.arange(n_cls) != c
    A_cls   = (sw[:, mask] - sw[:, c:c+1]).T
    b_cls   = (sb[mask]    - sb[c])
    return A_cls, b_cls


# =========================================================================== #
# Example usage: run from project root:                                       #
# >>> python -m src.optim.build_polytopes_cnn                                 #
# =========================================================================== #

if __name__ == "__main__":

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    import torch
    from src.models.networks import FashionCNN_Small
    from src.quantization.quantize import quantize_model
    from src.optim.build_polytopes import check_polytope_membership

    device = torch.device("cpu")

    # -------------------- #
    # Load model and data  #
    # -------------------- #
    print("\n*** Loading model and dataset... ***\n")

    model_path = "./checkpoints/fashion_cnn_best.pth"
    data_path  = "./data/fashionMNIST_correct_cnn.pt"

    model = FashionCNN_Small()
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    model.to(device).eval()
    qmodel = quantize_model(model, bits=8).to(device).eval()

    dataset = torch.load(data_path, weights_only=False)
    x, c    = dataset[0]
    print(f"Sample shape : {x.shape}  |  label : {c}")
    print(f"Pixel range  : [{x.min().item():.2f}, {x.max().item():.2f}]")

    # --------------------------------------------------------------- #
    # Verify model_to_sequential (torch.fx) matches model output      #
    # --------------------------------------------------------------- #
    print("\n*** Verifying model_to_sequential (FX tracing) matches model output... ***\n")

    seq_fx  = model_to_sequential(model)
    x_batch = x.unsqueeze(0) if x.dim() == 3 else x
    with torch.no_grad():
        out_fx    = seq_fx(x_batch)
        out_model = model(x_batch)

    match = torch.allclose(out_fx, out_model)
    print(f"seq_fx == model : {match}")
    print(f"Sequential layers (FX): {[type(m).__name__ for m in seq_fx]}")

    # ------------------------------------------- #
    # Build correct polytope (full-precision CNN) #
    # ------------------------------------------- #
    print("\n*** Building correct polytope (full-precision CNN)... ***\n")

    A_correct, b_correct, _, _ = build_two_class_cnn_polytopes(model, qmodel, x.unsqueeze(0).to(device), int(c))
    print(f"A_correct shape : {tuple(A_correct.shape)}")
    print(f"b_correct shape : {tuple(b_correct.shape)}")

    x_vec = x.flatten()
    in_correct = check_polytope_membership(A_correct, b_correct, x_vec)
    print(f"x_0 in correct polytope : {in_correct.item()}")

    # ------------------------------------------------------- #
    # Build all polytopes for multiple bit-widths              #
    # ------------------------------------------------------- #
    print("\n*** Building b-approximated polytopes for bits in [4, 8, 16]... ***\n")

    bits_list    = [4, 8, 16]
    qmodels_dict = {b: quantize_model(model, bits=b) for b in bits_list}
    for qm in qmodels_dict.values():
        qm.eval()

    A_base, b_base, polytopes_dict = build_cnn_all_polytopes(
        model, qmodels_dict, x, c
    )
    print(f"A_base shape (from build_cnn_all_polytopes) : {tuple(A_base.shape)}")

    for bits, (A_correct2, b_correct2, A_both, b_both) in polytopes_dict.items():
        in_correct2 = check_polytope_membership(A_correct2, b_correct2, x_vec)
        in_both     = check_polytope_membership(A_both,     b_both,     x_vec)
        print(f"  bits={bits:2d}  A_correct={tuple(A_correct2.shape)}"
              f"  A_both={tuple(A_both.shape)}"
              f"  in_correct={in_correct2.item()}  in_both={in_both.item()}")

    print("\nDone.")
