# Why the CNN correct polytope is much narrower than the MLP correct polytope

## Observed values (FashionMNIST, sample 13)

| Model | Mean width | Fraction of hypercube [-1,1]^784 |
|-------|-----------|----------------------------------|
| MLP (FashionMLP_Large) | ~37.3 | ~84% |
| CNN (FashionCNN_Small) | ~1.4  |  ~3% |

For reference, the maximum possible width of the hypercube [-1,1]^784 in a
random unit direction u is approximately 2 × E[‖u‖₁] ≈ 44.6.

## Reason 1 — more constraints (quantitative)

The CNN correct polytope has ~20,685 constraints vs ~3,849 for the MLP.
More halfspace constraints cut the polytope into a smaller region.
This effect alone does not explain the 25× difference in width.

## Reason 2 — MaxPool constraints are geometrically very tight (dominant effect)

A MaxPool2d constraint encodes:

    pixel_other ≤ pixel_max    (for every non-maximum position in each pooling window)

This enforces a **local pixel ordering**: the identity of the maximum pixel in
every 2×2 window must be preserved under perturbation. This is extremely
sensitive — a small change to two nearby pixels can swap their order,
immediately violating the constraint. With thousands of pooling windows, these
local ordering constraints collectively confine the input to a very tight
neighbourhood of x_0.

By contrast, an MLP ReLU constraint is a global halfspace in ℝ^784:

    W_i · x + b_i ≥ 0

A single such constraint rules out only one halfspace of the input cube;
~3,849 of them still leave a large polytope.

## Reason 3 — convolutional shortcut weights are spatially concentrated

For a convolutional ReLU neuron, the shortcut weight s_weight is supported
on a small local patch of the input image (the receptive field). The
corresponding halfspace constraint is therefore geometrically "sharp" in a
specific local direction rather than spread uniformly across all 784 inputs.
Many such concentrated constraints together create a tight region.

## Implication for the paper

The small CNN polytope width means:
- Even mild quantization may push the qmodel's activation region far enough
  to substantially reduce the intersection width (large error).
- Alternatively, if quantization preserves the local pixel ordering (MaxPool
  constraints), the error may be small.
The volume experiment will reveal which regime dominates.
