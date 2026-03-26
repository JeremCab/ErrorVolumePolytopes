from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import Subset

from src.shortcuts.shortcut_weights import compute_shortcut_weights
from src.shortcuts.shortcut_weights import pack_shortcut_weights


# ---------------------------------- #
# Build polytopes' linear contraints #
# ---------------------------------- #


def build_base_polytope_from_shortcuts(W_l, B_l, m_l):
    """
    Build base polytope from precomputed shortcut weights.
    (single model)
    TODO explain that A, b represent constraints A @ x + b ≤ 0 (and not A @ x ≤ b)
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

    A = s.unsqueeze(1) * W # unsaturated rows multiplied by -1
    b = s * B              # unsaturated rows multiplied by -1

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

        # 🔴 Skip if model is wrong
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
    from torch.utils.data import Subset

    from data.mnist_data import load_mnist_datasets
    from src.models.networks import SmallMLP
    from src.quantization.quantize import quantize_model
    from src.shortcuts.shortcut_weights import *

    # =============== #
    # Build polytopes #
    # =============== #

    # -------------------- #
    # Load models and data #
    # -------------------- #
    print("\n\n*** Load models and data... ***\n\n")

    model_type = "smallmlp"
    nb_epochs = 10
    bits = 4
    p = 0.85
    device = torch.device("cpu")

    # Model loading
    model_name = f"{model_type}_{nb_epochs}.pth"
    model_path = os.path.join("./checkpoints", model_name)

    # Model
    model = SmallMLP()
    model.load_state_dict(torch.load(model_path)['model_state'])
    model.to(device).eval()

    # qModel
    qmodel = quantize_model(model, bits=bits)
    qmodel.to(device).eval()

    # Dataset (and sample)
    _, test_dataset = load_mnist_datasets()
    x_0, c = test_dataset[123]
    x_0 = x_0.flatten().unsqueeze(0) # shape (1, input_dim)
    print("Sample x_0 shape:", x_0.shape)
    print("Models and dataset have been loaded.")


    # ---------------------------------------------- #
    # Test single sample membership in the polytopes #
    # ---------------------------------------------- #
    print("\n\n*** Testing single sample in the polytopes... ***\n\n")
    
    A_correct, b_correct, A_both, b_both = build_two_class_polytopes(model, qmodel, x_0, c)

    print("A_correct shape:", A_correct.shape)
    print("b_correct shape:", b_correct.shape)
    print("A_both shape:", A_both.shape)
    print("b_both shape:", b_both.shape)

    # test A_correct x <= b_correct
    x_vec = x_0.view(-1)  # shape (784,)
    correct_satisfied = check_polytope_membership(A_correct, b_correct, x_0)
    print("\nDoes x_0 satisfy correct_polytope constraints?", correct_satisfied.item())


    # --------------------------------------------- #
    # Test many samples membership in the polytopes #
    # --------------------------------------------- #
    print("\n\n*** Testing many samples in the polytopes... ***\n\n")

    indices = list(range(1000))  # first 200 samples
    stats = evaluate_polytopes(model, qmodel, test_dataset, indices)

    for k, v in stats.items():
        print(f"{k}: {v}")