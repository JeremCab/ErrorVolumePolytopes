import numpy as np
from scipy.optimize import linprog


# ----------------------------------- #
# Prune redundant polytope contraints #
# ----------------------------------- #

def prune_constraints_Clarkson(A, b, bounds=None, indices_to_check=None, tol=1e-8, verbose=False):
    """
    Supprime les contraintes redondantes d'un polytope Ax + b <= 0 
    de manière séquentielle (Exact).
    """
    if hasattr(A, "detach"):
        A, b = A.detach().cpu().numpy(), b.detach().cpu().numpy()

    n_constraints, d = A.shape
    
    # Sécurité sur les bounds
    if isinstance(bounds, tuple) and len(bounds) == 2 and not isinstance(bounds[0], tuple):
        bounds = [bounds] * d
    elif bounds is None:
        bounds = [(None, None)] * d
    
    # Si on ne donne pas d'indices, on teste TOUT
    if indices_to_check is None:
        indices_to_check = list(range(n_constraints))
    
    active_indices = list(range(n_constraints))
    removed_indices = []

    for idx_to_test in indices_to_check:
        # On ne teste que si elle est encore dans le set actif 
        # (important si indices_to_check contient des doublons)
        if idx_to_test not in active_indices:
            continue

        others = [j for j in active_indices if j != idx_to_test]
        
        # Maximiser a_i * x + b_i
        res = linprog(
            c=-A[idx_to_test],
            A_ub=A[others],
            b_ub=-b[others],
            bounds=bounds,
            method="highs"
        )

        if res.success and (A[idx_to_test] @ res.x + b[idx_to_test] <= tol):
            active_indices.remove(idx_to_test)
            removed_indices.append(idx_to_test)
            if verbose:
                print(f"  → Constraint {idx_to_test} is redundant.")

    A_pruned = A[active_indices]
    b_pruned = b[active_indices]
    ratio_removed = len(removed_indices) / n_constraints
    
    return A_pruned, b_pruned, ratio_removed


# def prune_constraints_RayTracing(A, b, bounds=None, n_rays=1000, tol=1e-8, verbose=False):
#     """
#     Accélère la détection de contraintes indispensables par échantillonnage de rayons,
#     puis utilise Clarkson pour une élimination exacte du reste.
#     """
#     if hasattr(A, "detach"):
#         A, b = A.detach().cpu().numpy(), b.detach().cpu().numpy()
    
#     n_constraints, d = A.shape
    
#     # Normalisation des bounds
#     if isinstance(bounds, tuple) and len(bounds) == 2 and not isinstance(bounds[0], tuple):
#         bounds = [bounds] * d
#     elif bounds is None:
#         bounds = [(None, None)] * d

#     # 1. Trouver un point intérieur x0 (Centre de Chebyshev)
#     c_cheb = np.zeros(d + 1)
#     c_cheb[-1] = -1 
#     norms = np.linalg.norm(A, axis=1, keepdims=True)
#     A_cheb = np.hstack([A, norms])
    
#     res_inner = linprog(c_cheb, A_ub=A_cheb, b_ub=-b, bounds=bounds + [(0, None)], method="highs")
    
#     if not res_inner.success:
#         if verbose: print("Polytope vide ou intérieur non trouvé.")
#         return A, b, 0.0

#     x0 = res_inner.x[:-1]

#     # 2. Ray-Tracing : Marquage des contraintes "frappées"
#     is_essential = np.zeros(n_constraints, dtype=bool)
#     directions = np.random.randn(n_rays, d)
#     directions /= np.linalg.norm(directions, axis=1, keepdims=True)

#     # Calcul vectorisé des intersections pour la vitesse
#     # t = (-b_i - A_i*x0) / (A_i*v)
#     numerator = -b - (A @ x0)
#     for v in directions:
#         denom = A @ v
#         # On ne garde que les intersections devant nous (t > 0)
#         t_values = np.where(denom > 1e-12, numerator / denom, np.inf)
#         idx_hit = np.argmin(t_values)
#         if t_values[idx_hit] != np.inf:
#             is_essential[idx_hit] = True

#     # 3. Clarkson sur le reste
#     must_test = [j for j in range(n_constraints) if not is_essential[j]]
    
#     if verbose:
#         print(f"Ray-tracing found {np.sum(is_essential)} essential faces.")
#         print(f"Running Clarkson on {len(must_test)} candidates...")

#     return prune_constraints_Clarkson(A, b, bounds=bounds, indices_to_check=must_test, tol=tol, verbose=verbose)



def sample_directions(n_rays, d, mode="gaussian"):
    """
    Sample directions on the unit sphere.

    Parameters
    ----------
    n_rays : int
    d : int
    mode : str
        "gaussian"   : random Gaussian directions (normalized)
        "sign"       : random ±1 directions (L∞ corners)
        "axis"       : canonical basis (+/- e_i)

    Returns
    -------
    directions : np.ndarray (n_rays, d)
    """

    if mode == "gaussian":
        directions = np.random.randn(n_rays, d)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    elif mode == "sign":
        directions = np.random.choice([-1.0, 1.0], size=(n_rays, d))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    elif mode == "axis":
        directions = []
        for i in range(d):
            e = np.zeros(d)
            e[i] = 1.0
            directions.append(e)
            directions.append(-e)
        directions = np.array(directions)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return directions


def prune_constraints_RayTracing(
    A,
    b,
    bounds=None,
    n_rays=1000,
    direction_mode="gaussian",
    tol=1e-8,
    verbose=False,
):
    """
    Hybrid redundancy removal:
        1) Ray tracing (fast detection of essential constraints)
        2) Clarkson on remaining candidates

    Polytope:
        A x + b <= 0

    Returns
    -------
    A_pruned, b_pruned
    """

    # ---------------- #
    # Convert to numpy #
    # ---------------- #
    if hasattr(A, "detach"):
        A = A.detach().cpu().numpy()
        b = b.detach().cpu().numpy()

    n_constraints, d = A.shape

    # ---------------- #
    # Bounds handling  #
    # ---------------- #
    if bounds is None:
        bounds = [(None, None)] * d
    elif isinstance(bounds, tuple) and len(bounds) == 2 and not isinstance(bounds[0], tuple):
        bounds = [bounds] * d

    # ------------------------------- #
    # 1. Chebyshev center (interior) #
    # ------------------------------- #
    c_cheb = np.zeros(d + 1)
    c_cheb[-1] = -1.0  # maximize radius

    norms = np.linalg.norm(A, axis=1, keepdims=True)
    A_cheb = np.hstack([A, norms])

    res = linprog(
        c_cheb,
        A_ub=A_cheb,
        b_ub=-b,
        bounds=bounds + [(0, None)],
        method="highs",
    )

    if not res.success:
        if verbose:
            print("⚠️ Polytope empty or no interior found.")
        return A, b

    x0 = res.x[:-1]

    # ------------------------------- #
    # 2. Ray tracing (vectorized)     #
    # ------------------------------- #
    directions = sample_directions(n_rays, d, mode=direction_mode)

    V = directions.T                          # (d, n_rays)
    denom = A @ V                             # (n_constraints, n_rays)

    numerator = (-b - A @ x0)[:, None]        # (n_constraints, 1)

    # Valid intersections:
    # - denom > 0  → ray goes toward constraint
    # - numerator > 0 → constraint is in front of x0
    valid = (denom > 1e-12) & (numerator > 0)

    t_values = np.where(valid, numerator / denom, np.inf)

    # Closest intersection per ray
    idx_hits = np.argmin(t_values, axis=0)
    t_min = t_values[idx_hits, np.arange(t_values.shape[1])]

    is_essential = np.zeros(n_constraints, dtype=bool)

    valid_hits = t_min != np.inf
    is_essential[idx_hits[valid_hits]] = True

    if verbose:
        print(f"Ray tracing: {is_essential.sum()} constraints marked essential.")

    # -------------------------------- #
    # 3. Clarkson on remaining         #
    # -------------------------------- #
    candidates = np.where(~is_essential)[0]

    if verbose:
        print(f"Clarkson on {len(candidates)} remaining constraints.")

    return prune_constraints_Clarkson(
        A,b,bounds=bounds,
        indices_to_check=candidates,
        tol=tol,
        verbose=verbose,
    )



# ================================================= #
# Example usage: 'Play' or run from root directory: #
# >>> python -m src.optim.prune_constraints         #
# ================================================= #
if __name__ == "__main__":

    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    import torch
    from torch.utils.data import Subset

    from data.mnist_data import load_mnist_datasets
    from src.models.networks import SmallMLP
    from src.quantization.quantize import quantize_model
    from src.optim.build_polytopes import build_two_class_polytopes
    

    # =========== #
    # Toy example #
    # =========== #
    print("\n\n*** Toy example... ***\n\n")

    def test_pruning():
        # Définition du carré unité avec redondances
        # Format Ax + b <= 0  =>  Ax <= -b
        # 1. x <= 1       ->  1x + 0y - 1 <= 0
        # 2. -x <= 0      -> -1x + 0y + 0 <= 0
        # 3. y <= 1       ->  0x + 1y - 1 <= 0
        # 4. -y <= 0      ->  0x - 1y + 0 <= 0
        # 5. x <= 2       ->  1x + 0y - 2 <= 0  (REDONDANT - Dominé)
        # 6. x <= 1       ->  1x + 0y - 1 <= 0  (REDONDANT - Doublon)
        # 7. x + y <= 3   ->  1x + 1y - 3 <= 0  (REDONDANT - Combiné)

        A = np.array([
            [1, 0],   # 0: x <= 1
            [-1, 0],  # 1: x >= 0
            [0, 1],   # 2: y <= 1
            [0, -1],  # 3: y >= 0
            [1, 0],   # 4: x <= 2
            [1, 0],   # 5: x <= 1 (doublon)
            [1, 1]    # 6: x+y <= 3
        ], dtype=float)

        b = np.array([-1, 0, -1, 0, -2, -1, -3], dtype=float)

        print(f"Original constraints: {len(A)}")
        
        # On définit les bornes (optionnel ici mais bien pour tester ta fonction)
        bounds = (None, None) 

        # Test avec Clarkson
        A_p, b_p, ratio = prune_constraints_Clarkson(A, b, bounds=bounds, verbose=True)
        # Test avec Ray-Tracing + Clarkson
        A_p2, b_p2, ratio2 = prune_constraints_RayTracing(A, b, bounds=bounds, n_rays=500, verbose=True)

        print("\n--- RESULTS ---")
        print(f"Constraints kept (Clarkson): {len(A_p)}")
        print(f"Constraints removed (Clarkson): {len(A) - len(A_p)}")
        print(f"Ratio removed (Clarkson):: {ratio:.2%}")
        print("\n")
        print(f"Constraints kept (Ray Tracing): {len(A_p2)}")
        print(f"Constraints removed (Ray Tracing): {len(A) - len(A_p2)}")
        print(f"Ratio removed (Ray Tracing):: {ratio2:.2%}")


        # Vérification attendue
        # On doit garder exactement 4 contraintes (les faces du carré).
        if len(A_p) == 4:
            print("\nSUCCESS: The algorithm identified all 3 redundant constraints!")
        else:
            print(f"\nFAILURE: Expected 4 constraints, but got {len(A_p)}.")

    # Lancement du test
    test_pruning()

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

    # ------------------------ #
    # Compute correct polytope #
    # ------------------------ #
    print("\n\n*** Testing single sample in the polytopes... ***\n\n")
    
    A_correct, b_correct, A_both, b_both = build_two_class_polytopes(model, qmodel, x_0, c)

    print("\nBefore pruning:")
    print("A_correct shape:", A_correct.shape)
    print("b_correct shape:", b_correct.shape)

    print("\nPruning constraints with Clarkson method...")
    bounds = (-0.5, 2.9) # bounds for pixel values (after flattening and transformation)
    A_pruned, b_pruned, ratio_removed = prune_constraints_Clarkson(A_correct, 
                                                          b_correct, 
                                                          bounds=bounds, 
                                                          verbose=True)
    print("\nAfter pruning:")
    print("A_pruned shape:", A_pruned.shape)
    print("b_pruned shape:", b_pruned.shape)
    print("Ratio of constraints removed:", ratio_removed)


    print("\nPruning constraints with Ray-Tracing + Clarkson method...")
    bounds = (-0.5, 2.9) # bounds for pixel values (after flattening and transformation)
    A_pruned, b_pruned, ratio_removed = prune_constraints_RayTracing(A_correct, 
                                                                     b_correct, 
                                                                     bounds=bounds, 
                                                                     verbose=True)
    print("\nAfter pruning:")
    print("A_pruned shape:", A_pruned.shape)
    print("b_pruned shape:", b_pruned.shape)
    print("Ratio of constraints removed:", ratio_removed)