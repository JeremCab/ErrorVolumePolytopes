# ========== #
# Librairies #
# ========== #


import torch
import torch.nn as nn


# ================ #
# Shortcut weights #
# ================ #

def get_weights_and_biases(model):
    """
    Extract weights and biases from all Linear layers.

    Parameters
    ----------
    model : torch.nn.Module
        Feedforward neural network.

    Returns
    -------
    weights : list of torch.Tensor
        Weight matrices W_l of shape (out_dim, in_dim).
    biases : list of torch.Tensor
        Bias vectors b_l of shape (out_dim,).
    """

    weights = []
    biases = []

    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            weights.append(layer.weight.detach())
            biases.append(layer.bias.detach())

    return weights, biases


def get_unsaturations(model, x, return_all=False):
    """
    Compute the (un)saturation pattern of a feedforward ReLU network.

    This function performs a forward pass and extracts the unsaturation
    masks: a list of boolean tensors, one per layer. Each mask indicates
    which neurons are unsaturated (pre-activation z >= 0). The final layer
    is treated as always unsaturated (mask of all True). Optionally, it also
    returns the corresponding pre-activations and post-activations for debugging.

    Parameters
    ----------
    model : torch.nn.Module
        Feedforward neural network (MLP with ReLU activations).
    x : torch.Tensor
        Input tensor of shape (1, input_dim) or (input_dim,).
    return_all : bool, optional (default=False)
        If True, also returns pre-activations and activations.
        If False, returns only unsaturation masks (more efficient).

    Returns
    -------
    unsaturations : list of torch.Tensor (bool)
        Boolean masks indicating active neurons (z >= 0) for each layer.
        The final layer mask is all True.

    OR (if return_all=True)

    preactivations : list of torch.Tensor
        Pre-activation values z_l for each layer.
    activations : list of torch.Tensor
        Post-activation values a_l (after ReLU, except final layer).
    unsaturations : list of torch.Tensor (bool)
        Boolean masks (z_l >= 0).
    """

    preactivations = [] if return_all else None
    activations = [] if return_all else None
    unsaturations = []
    hooks = []

    def relu_hook(module, input, output):
        z = input[0].detach().view(-1)

        if return_all:
            a = output.detach().view(-1)
            preactivations.append(z)
            activations.append(a)

        unsaturations.append(z >= 0) # Hidden layer: True if z >= 0, False if z < 0

    def final_hook(module, input, output):
        z = output.detach().view(-1)

        if return_all:
            preactivations.append(z)
            activations.append(z)

        unsaturations.append(torch.ones_like(z, dtype=torch.bool)) # Final layer: all True

    # Register hooks
    for layer in model.modules():
        if isinstance(layer, nn.ReLU):
            hooks.append(layer.register_forward_hook(relu_hook))

    last_layer = None
    for layer in model.modules():
        if len(list(layer.children())) == 0:
            last_layer = layer

    if last_layer is not None and not isinstance(last_layer, nn.ReLU):
        hooks.append(last_layer.register_forward_hook(final_hook))

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    if return_all:
        return preactivations, activations, unsaturations
    else:
        return unsaturations
    

def compute_shortcut_weights(model, x):
    """
    Compute shortcut (effective) weights of a ReLU network in a fixed region.

    The network is locally linear given a fixed activation pattern.
    This function computes the equivalent affine map:

        f_l(x) = W_l^shortcut x + B_l^shortcut

    for each layer l.

    Parameters
    ----------
    model : torch.nn.Module
        Feedforward ReLU network.
    x : torch.Tensor
        Input sample used to determine the activation region.

    Returns
    -------
    W_shortcut_l : list of torch.Tensor
        Effective weight matrices mapping input to each layer.
    B_shortcut_l : list of torch.Tensor
        Effective bias vectors.
    unsaturations_l : list of torch.Tensor (bool)
        Unsaturation masks for each layer (True if z >= 0).
    """

    unsaturations_l = get_unsaturations(model, x, return_all=False)
    weights, biases = get_weights_and_biases(model)

    nb_layers = len(unsaturations_l)

    W_shortcut_l = [weights[0]]
    B_shortcut_l = [biases[0]]

    for l in range(1, nb_layers):

        m_l = unsaturations_l[l-1].float().view(-1)

        W_l = weights[l] @ (m_l[:, None] * W_shortcut_l[-1])
        B_l = biases[l] + weights[l] @ (m_l * B_shortcut_l[-1])

        W_shortcut_l.append(W_l)
        B_shortcut_l.append(B_l)

    return W_shortcut_l, B_shortcut_l, unsaturations_l


def pack_shortcut_weights(W_shortcut_l, B_shortcut_l, unsaturations_l):
    """
    Pack shortcut weights and unsaturations mask into a single augmented matrix.

    For shortcut weights, this function:
      1. Vertically concatenates all W_l into a matrix W_big
      2. Vertically concatenates all B_l into a vector B_big
      3. Concatenates B_big as an extra first column to W_big

    Result is an affine map in augmented form:
        [B_big | W_big]

    For unsaturations mask, this function:
      1. Concatenates all elements of unsaturations_l

    Parameters
    ----------
    W_shortcut_l : list of torch.Tensor
        List of weight matrices W_l of shape (d_l, input_dim)
    B_shortcut_l : list of torch.Tensor
        List of bias vectors B_l of shape (d_l,)
    unsaturations_l : list of torch.Tensor
        List of boolean masks of shape (d_l,)

    Returns
    -------
    packed_matrix : torch.Tensor
        Tensor of shape (sum_l d_l, input_dim + 1)
        representing the affine maps stacked together:
            y = W_big x + B_big
        encoded as:
            [B_big | W_big]
    packed_mask : torch.tensor
        Tensor of shape (sum_l d_l) 
        representing the unsaturation mask of all layers.
    """

    # Stack weights vertically
    W_big = torch.cat(W_shortcut_l, dim=0)  # (sum d_l, input_dim)

    # Stack biases and make column
    B_big = torch.cat([b.view(-1, 1) for b in B_shortcut_l], dim=0)  # (sum d_l, 1)

    # Concatenate horizontally
    packed_matrix = torch.cat([B_big, W_big], dim=1)  # (sum d_l, input_dim + 1)

    packed_mask = torch.cat(unsaturations_l, dim=0).squeeze()  # (sum d_l,)

    return packed_matrix, packed_mask


def test_shortcut_weights(model, x):
    """
    Verify that shortcut weights reproduce layer pre-activations.

    This function compares:
        z_l (true pre-activations)
        vs
        W_l^shortcut x + B_l^shortcut

    Parameters
    ----------
    model : torch.nn.Module
        Feedforward ReLU network.
    x : torch.Tensor
        Input sample.
    """

    preactivations, _, _ = get_unsaturations(model, x, return_all=True)
    W_shortcut_l, B_shortcut_l, _ = compute_shortcut_weights(model, x)

    x_flat = x.view(-1)

    for l in range(len(preactivations)):

        z_true = preactivations[l]
        z_shortcut = W_shortcut_l[l] @ x_flat + B_shortcut_l[l]

        print(f"Layer {l}:")
        print(f"  True preactivation:\t{z_true.shape}")
        print(f"  Shortcut:\t\t{z_shortcut.shape}")
        print(f"  Match:\t\t{torch.allclose(z_true, z_shortcut, atol=1e-6)}\n")



# ================================================== #
# Example usage: run from root directory:            #
# >>> python -m src.shortcuts.shortcut_weights       #
# ================================================== #

if __name__ == '__main__':

    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    
    import torch
    from torchvision import datasets, transforms

    from data.mnist_data import load_mnist_datasets
    from src.models.networks import SmallMLP, FashionMLP_Large
    from src.quantization.quantize import quantize_model

    # ============== #
    # Model and Data #
    # ============== #

    torch.manual_seed(42)
    model = SmallMLP()
    
    _, test_dataset = load_mnist_datasets()
    x_0, c = test_dataset[123]
    x_0 = x_0.flatten().unsqueeze(0) # shape (1, input_dim)


    # =========== #
    # Activations #
    # =========== #    
    
    preactivations, activations, unsaturations = get_unsaturations(model, x_0, return_all=True)
    
    print("\n*** Activations ***\n")
    print([ (activations[i].shape, unsaturations[i].shape) for i in range(len(activations)) ])


    # ================== #
    # Weights and Biases #
    # ================== #
    
    weights, biases = get_weights_and_biases(model)

    print("\n*** Weights and Biases ***\n")
    for i, (W, b) in enumerate(zip(weights, biases)):
        print(f"Layer {i}: W shape = {W.shape}, b shape = {b.shape}")


    # =========================== #
    # Shortcut weights and Biases #
    # =========================== #

    print("\n*** Shortcut Weights and Biases ***\n")

    print("(a) Testing shortcut weights on small MLP\n")

    W_shortcut_l, B_shortcut_l, unsaturations_l = compute_shortcut_weights(model, x_0)
    
    for i, (W, B) in enumerate(zip(W_shortcut_l, B_shortcut_l)):
        print(f"Layer {i}: W_shortcut shape = {W.shape}", 
              f"\tB_shortcut shape = {B.shape}")


    W_shortcut, unsaturation_mask = pack_shortcut_weights(W_shortcut_l, B_shortcut_l, unsaturations_l)
    print(f"\nPacked shortcut weights shape: {W_shortcut.shape}")
    print(f"Packed unsaturation mask shape: {unsaturation_mask.shape}")
    print(unsaturation_mask[-10:], "\n")

    test_shortcut_weights(model, x_0)


    # Test on other MLP #


    print("\n(b) Testing shortcut weights on larger MLP\n")

    x_2 = torch.rand(size=(1, 784))

    def build_deep_mlp(input_dim=784, seed=42):

        torch.manual_seed(seed)

        model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 10)   # final layer (no ReLU)
        )

        return model
        
    model_2 = build_deep_mlp()

    W_shortcut_l2, B_shortcut_l2, unsaturations_l = compute_shortcut_weights(model_2, x_2)

    for i, (W, B) in enumerate(zip(W_shortcut_l2, B_shortcut_l2)):
        print(f"Layer {i}: W_shortcut shape = {W.shape}", 
              f"\tB_shortcut shape = {B.shape}")

    W_shortcut_2, unsaturation_mask_2 = pack_shortcut_weights(W_shortcut_l2, B_shortcut_l2, unsaturations_l)
    print(f"\nPacked shortcut weights shape: {W_shortcut_2.shape}")
    print(f"Packed unsaturation mask shape: {unsaturation_mask_2.shape}")
    print(unsaturation_mask_2[-10:], "\n")

    test_shortcut_weights(model_2, x_2)
