"""Weight quantization utilities.

Provides functions to quantize tensors and PyTorch models to a given number of bits.
"""
import torch
import math


def quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """Uniform symmetric quantization to given number of bits."""
    if bits <= 0:
        raise ValueError("bits must be positive")
    qmin = - (2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    max_val = tensor.abs().max()
    if max_val == 0:
        return tensor.clone()
    scale = max_val / qmax
    q = torch.round(tensor / scale).clamp(qmin, qmax)
    return (q * scale).to(tensor.dtype)


def quantize_model(model: torch.nn.Module, bits: int) -> torch.nn.Module:
    """Return a copy of model where all parameter tensors are quantized to `bits`.

    This performs weight rounding (post-training quantization).
    """
    qmodel = type(model)() if hasattr(type(model), '__call__') else None
    # Safer approach: create deep copy and quantize in-place
    import copy
    qmodel = copy.deepcopy(model)
    for p in qmodel.parameters():
        with torch.no_grad():
            p.copy_(quantize_tensor(p.data, bits))
    return qmodel
