import numpy as np

import torch
from torch.utils.data import Subset

from src.shortcuts.shortcut_weights import compute_shortcut_weights
from src.shortcuts.shortcut_weights import pack_shortcut_weights


# ---------------------------------- #
# Build polytopes' linear contraints #
# ---------------------------------- #


def build_base_polytope_from_shortcuts(W_l, B_l, m_l):
    """
    Build the base (activation) polytope from precomputed shortcut weights.

    For each hidden neuron i, the activation constraint is:
        - if unsaturated (m_l[i] = True):  -(W_i x + b_i) <= 0  i.e. z_i >= 0
        - if saturated   (m_l[i] = False):  (W_i x + b_i) <= 0  i.e. z_i <= 0

    The output layer is excluded (its constraints come from classification).

    Convention throughout this codebase: Ax + b <= 0 (NOT Ax <= b).

    Parameters
    ----------
    W_l : list of torch.Tensor
        Shortcut weight matrices, one per layer.
    B_l : list of torch.Tensor
        Shortcut bias vectors, one per layer.
    m_l : list of torch.Tensor (bool)
        Unsaturation masks, one per layer (True = unsaturated).

    Returns
    -------
    A : torch.Tensor of shape (n_constraints, input_dim)
    b : torch.Tensor of shape (n_constraints,)
    """

    packed_matrix, packed_mask = pack_shortcut_weights(W_l, B_l, m_l)

    # Remove output layer
    n_out = m_l[-1].numel()
    packed_matrix = packed_matrix[:-n_out]
    packed_mask = packed_mask[:-n_out]

    B = packed_matrix[:, 0]
    W = packed_matrix[:, 1:]

    # sign vector s with the same shape as B, where:
    # s[i] = -1 if packed_mask[i] is True
    # s[i] = +1 if packed_mask[i] is False
    s = torch.where(packed_mask, -torch.ones_like(B), torch.ones_like(B))

    A = s.unsqueeze(1) * W # unsaturated rows multiplied by -1
    b = s * B              # unsaturated rows multiplied by -1

    return A, b


def build_class_constraints_from_shortcuts(W_l, B_l, c):
    """
    Build classification constraints from shortcut weights:

        xi_j - xi_c <= 0  for all j != c

    i.e.
        (w_j - w_c)x + (b_j - b_c) <= 0

    Parameters
    ----------
    W_l : list of torch.Tensor
        Shortcut weights per layer
    B_l : list of torch.Tensor
        Shortcut biases per layer
    c : int
        Target class

    Returns
    -------
    A_class : torch.Tensor (nb_outputs-1, input_dim)
    b_class : torch.Tensor (nb_outputs-1,)
    """

    W_out = W_l[-1]   # (nb_outputs, input_dim)
    B_out = B_l[-1]   # (nb_outputs,)

    nb_outputs = W_out.shape[0]

    # Differences to class c
    W_diff = W_out - W_out[c]
    B_diff = B_out - B_out[c]

    # Remove class c itself
    mask = torch.arange(nb_outputs) != c

    A_class = W_diff[mask]
    b_class = B_diff[mask]

    return A_class, b_class


def build_two_class_polytopes(model, qmodel, x, c):
    """
    Build two polytopes:

    1) correct_polytope:
        - activation constraints (model + qmodel)
        - model predicts class c

    2) correct_and_qcorrect_polytope:
        - same as above
        - qmodel also predicts class c

    No redundant computations.

    Parameters
    ----------
    model : torch.nn.Module
    qmodel : torch.nn.Module
    x : torch.Tensor
    c : int

    Returns
    -------
    A_correct : torch.Tensor
    b_correct : torch.Tensor

    A_both : torch.Tensor
    b_both : torch.Tensor
    """

    # ========================= #
    # 1) Precompute shortcuts   #
    # ========================= #

    W_l, B_l, m_l = compute_shortcut_weights(model, x)
    Wq_l, Bq_l, mq_l = compute_shortcut_weights(qmodel, x)

    # ========================= #
    # 2) Base polytope          #
    # ========================= #

    A1, b1 = build_base_polytope_from_shortcuts(W_l, B_l, m_l)
    A2, b2 = build_base_polytope_from_shortcuts(Wq_l, Bq_l, mq_l)

    A_base = torch.cat([A1, A2], dim=0)
    b_base = torch.cat([b1, b2], dim=0)

    # ============================== #
    # 3) Classification constraints  #
    # ============================== #

    A_class_model, b_class_model = build_class_constraints_from_shortcuts(W_l, B_l, c)
    A_class_qmodel, b_class_qmodel = build_class_constraints_from_shortcuts(Wq_l, Bq_l, c)

    # ============================== #
    # 4) Polytope: model correct     #
    # ============================== #

    A_correct = torch.cat([A_base, A_class_model], dim=0)
    b_correct = torch.cat([b_base, b_class_model], dim=0)

    # ======================================= #
    # 5) Polytope: model AND qmodel correct   #
    # ======================================= #

    A_both = torch.cat([A_correct, A_class_qmodel], dim=0)
    b_both = torch.cat([b_correct, b_class_qmodel], dim=0)

    return A_correct, b_correct, A_both, b_both


def build_all_polytopes(model, qmodels_dict, x, c):
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
    model : torch.nn.Module
    qmodels_dict : dict {bits (int): qmodel (torch.nn.Module)}
    x : torch.Tensor of shape (1, input_dim)
    c : int

    Returns
    -------
    A_base : torch.Tensor of shape (n_base, input_dim)
    b_base : torch.Tensor of shape (n_base,)
    polytopes_dict : dict {bits (int): (A_correct, b_correct, A_both, b_both)}
    """

    # Model shortcuts — computed ONCE
    W_l, B_l, m_l = compute_shortcut_weights(model, x)
    A_act, b_act   = build_base_polytope_from_shortcuts(W_l, B_l, m_l)
    A_cls, b_cls   = build_class_constraints_from_shortcuts(W_l, B_l, c)

    # A_base: model-only (independent of quantization)
    A_base = torch.cat([A_act, A_cls], dim=0)
    b_base = torch.cat([b_act, b_cls], dim=0)

    polytopes_dict = {}
    for bits, qmodel in qmodels_dict.items():
        Wq_l, Bq_l, mq_l = compute_shortcut_weights(qmodel, x)
        A_act_q, b_act_q  = build_base_polytope_from_shortcuts(Wq_l, Bq_l, mq_l)
        A_cls_q, b_cls_q  = build_class_constraints_from_shortcuts(Wq_l, Bq_l, c)

        # A_correct: A_base + qmodel activation
        A_correct = torch.cat([A_base, A_act_q], dim=0)
        b_correct = torch.cat([b_base, b_act_q], dim=0)

        # A_both: A_correct + qmodel classification (differ by 9 constraints)
        A_both = torch.cat([A_correct, A_cls_q], dim=0)
        b_both = torch.cat([b_correct, b_cls_q], dim=0)

        polytopes_dict[bits] = (A_correct, b_correct, A_both, b_both)

    return A_base, b_base, polytopes_dict


def ensure_vector(x):
    """
    Ensure x has shape (input_dim,)
    Accepts (input_dim,) or (1, input_dim)
    """
    if x.dim() == 2:
        if x.shape[0] == 1:
            return x.view(-1)
        else:
            raise ValueError(f"x has shape {x.shape}, expected (1, d) or (d,)")
    elif x.dim() == 1:
        return x
    else:
        raise ValueError(f"x has invalid shape {x.shape}")


def check_polytope_membership(A, b, x, tol=1e-5):
    """
    Check if x satisfies A x + b <= 0
    (and not if A x <= b)
    """

    x_vec = ensure_vector(x)

    lhs = A @ x_vec + b
    return torch.all(lhs <= tol)


def evaluate_polytopes(model, qmodel, dataset, indices):
    """
    Evaluate polytope membership for a set of samples.

    For each sample where model is correct, builds the correct and both
    polytopes and checks whether the sample lies inside each one.

    Parameters
    ----------
    model : torch.nn.Module
    qmodel : torch.nn.Module
    dataset : torch.utils.data.Dataset
    indices : list of int
        Indices into dataset to evaluate.

    Returns
    -------
    dict with keys:
        "total"             : samples where model is correct
        "model_correct"     : same as total
        "qmodel_correct"    : samples where qmodel is also correct
        "correct_polytope_ok" : samples inside correct_polytope
        "both_polytope_ok"    : samples inside both_polytope
    """

    device = next(model.parameters()).device

    subset = Subset(dataset, indices)

    stats = {
        "total": 0,
        "model_correct": 0,
        "qmodel_correct": 0,
        "correct_polytope_ok": 0,
        "both_polytope_ok": 0,
    }

    for i in range(len(subset)):

        x, c = subset[i]

        x_batch = x.flatten().unsqueeze(0).to(device)
        x_vec = x_batch.view(-1)

        with torch.no_grad():
            pred_model = model(x_batch).argmax(dim=1).item()
            pred_qmodel = qmodel(x_batch).argmax(dim=1).item()

        # Skip if model is wrong
        if pred_model != c:
            continue

        stats["model_correct"] += 1
        stats["qmodel_correct"] += (pred_qmodel == c)

        # Build polytopes
        A_correct, b_correct, A_both, b_both = build_two_class_polytopes(
            model, qmodel, x_batch, c
        )

        # Membership
        if check_polytope_membership(A_correct, b_correct, x_vec):
            stats["correct_polytope_ok"] += 1

        if check_polytope_membership(A_both, b_both, x_vec):
            stats["both_polytope_ok"] += 1

        stats["total"] += 1

    return stats


# ================================================= #
# Example usage: 'Play' or run from root directory: #
# >>> python -m src.optim.build_polytopes           #
# ================================================= #

if __name__ == "__main__":

    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    import torch

    from src.models.networks import FashionMLP_Large
    from src.quantization.quantize import quantize_model

    # -------------------- #
    # Load model and data  #
    # -------------------- #
    print("\n\n*** Loading FashionMLP_Large and fashionMNIST_correct_mlp... ***\n")

    fashion_model = FashionMLP_Large()
    fashion_model.load_state_dict(
        torch.load("./checkpoints/fashion_mlp_best.pth",
                   weights_only=True, map_location="cpu")
    )
    fashion_model.eval()

    fashion_dataset = torch.load(
        "./data/fashionMNIST_correct_mlp.pt", weights_only=False
    )
    x_f, c_f  = fashion_dataset[0]
    x_f_batch = x_f.flatten().unsqueeze(0)  # (1, 784)
    x_f_vec   = x_f_batch.view(-1)          # (784,)

    bits_list    = [4, 8, 16]
    qmodels_dict = {b: quantize_model(fashion_model, bits=b) for b in bits_list}
    for qm in qmodels_dict.values():
        qm.eval()

    print(f"Sample shape: {x_f_batch.shape}  label: {int(c_f)}")
    print("Models and dataset loaded.\n")


    # ================================================== #
    # Test build_two_class_polytopes (one bit at a time) #
    # ================================================== #
    print("*** Testing build_two_class_polytopes (FashionMLP_Large)... ***\n")

    for bits in bits_list:
        qmodel = qmodels_dict[bits]
        A_correct, b_correct, A_both, b_both = build_two_class_polytopes(
            fashion_model, qmodel, x_f_batch, int(c_f)
        )
        n_correct = A_correct.shape[0]
        n_both    = A_both.shape[0]
        diff_cb   = n_both - n_correct   # should be 9

        in_correct = check_polytope_membership(A_correct, b_correct, x_f_vec).item()
        in_both    = check_polytope_membership(A_both,    b_both,    x_f_vec).item()

        print(f"  bits={bits:2d}")
        print(f"    A_correct rows : {n_correct}")
        print(f"    A_both rows    : {n_both}  (diff={diff_cb}, should be 9: {diff_cb == 9})")
        print(f"    x_0 in A_correct : {in_correct}")
        print(f"    x_0 in A_both    : {in_both}")


    # ========================================================= #
    # Test build_all_polytopes (FashionMLP_Large, FashionMNIST) #
    # ========================================================= #
    print("\n\n*** Testing build_all_polytopes (FashionMLP_Large)... ***\n")

    A_base, b_base, poly_dict = build_all_polytopes(
        fashion_model, qmodels_dict, x_f_batch, int(c_f)
    )

    print(f"A_base shape : {tuple(A_base.shape)}")
    print(f"x_0 in A_base: {check_polytope_membership(A_base, b_base, x_f_vec).item()}")

    for bits, (A_correct, b_correct, A_both, b_both) in poly_dict.items():
        n_correct = A_correct.shape[0]
        n_both    = A_both.shape[0]
        n_base    = A_base.shape[0]
        diff_cb   = n_both - n_correct   # should be 9

        in_base    = check_polytope_membership(A_base,    b_base,    x_f_vec).item()
        in_correct = check_polytope_membership(A_correct, b_correct, x_f_vec).item()
        in_both    = check_polytope_membership(A_both,    b_both,    x_f_vec).item()

        print(f"\n  bits={bits:2d}")
        print(f"    A_correct rows : {n_correct}  (A_base {n_base} + qmodel_act {n_correct - n_base})")
        print(f"    A_both rows    : {n_both}  (A_correct {n_correct} + {diff_cb} cls constraints)")
        print(f"    A_both - A_correct == 9 : {diff_cb == 9}")
        print(f"    x_0 in A_base    : {in_base}")
        print(f"    x_0 in A_correct : {in_correct}")
        print(f"    x_0 in A_both    : {in_both}")

    print("\nDone.")
