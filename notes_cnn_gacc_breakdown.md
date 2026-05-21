# CNN GACC breakdown — metric measures non-additivity, not accuracy

## Finding

For CNN (999 samples, fashionMNIST_correct_cnn.pt), GACC **increases** as bit-width decreases — the opposite of what MLP shows. This is not a bug.

## Root cause

V3_c ≈ V2 for CNN at all bit-widths (q-model is ~100% accurate within P2), but ΣV3_k >> V2 due to high non-additivity:

| bits | V3_c/V2 (accuracy in P2) | ΣV3/V2 (non-additivity) | GACC |
|------|--------------------------|-------------------------|------|
| 4    | 0.994                    | 1.721                   | 0.578 |
| 6    | 1.000                    | 1.938                   | 0.516 |
| 8    | 1.000                    | 2.286                   | 0.437 |
| 10   | 1.000                    | 2.361                   | 0.424 |
| 12   | 1.000                    | 2.318                   | 0.431 |
| 16   | 1.000                    | 2.373                   | 0.421 |

For MLP: ΣV3/V2 ≈ 1.005–1.028 (near-additive), so GACC ≈ V3_c/V2 ≈ ACC.

**For CNN: GACC ≈ 1 / (ΣV3/V2) — it measures inverse non-additivity, not accuracy.**

## Why non-additivity increases with bits for CNN

At higher bit-width, Q-CNN more closely tracks the FP CNN's complex decision surface inside P2, creating more competing class regions (mean 2.74 non-zero classes at b=16 vs 1.75 at b=4). Each P3(k) is thin in volume but has large mean-width in 784D (concentration of measure). At b=4, heavy quantization collapses many decision boundaries → fewer competing regions → lower non-additivity → higher GACC.

## Implication for the paper

The mean-width-based GACC is only a valid accuracy proxy when the partition {P3(k)} is nearly additive (holds for MLP, not CNN):
- **Old GACC** (V3_c/V2): biased upward for CNN (stays ≈1.0, masking inaccuracy)
- **New GACC** (V3_c/ΣV3_k): biased downward for CNN (says 0.42–0.58 for a 100%-accurate model)

For CNN, neither GACC variant reliably measures classification accuracy. This is a limitation to address in the paper.

**Why:** In 784D, each thin P3(k) slab "spans" P2 when projected onto most axes, giving it a mean-width comparable to V2 even if its volume is negligible.

## Possible fix — normalisation by mean #nonzero classes

Define a corrected GACC calibrated to b=16 (where Q-CNN ≈ FP CNN, so GACC should be ~1):

1. Compute ν = GACC(b=16) × mean_nz(b=16) ≈ 0.421 × 2.74 ≈ 1.155
2. Define **GACC_new(b) = GACC(b) × (mean_nz(b) / ν)**

This removes the "baseline non-additivity" of the FP CNN's decision structure and measures only quantization-induced degradation. Result for CNN:

| bits | GACC_new |
|------|----------|
| 4    | 0.875    |
| 6    | 0.933    |
| 8    | 0.966    |
| 10   | 0.986    |
| 12   | 0.986    |
| 16   | 1.000    |

This recovers the expected decreasing trend as b decreases. For MLP, mean_nz ≈ 1 at all b, so GACC_new ≈ GACC (unchanged).

**Caveats:**
- ν is data-dependent (estimated from b=16), making GACC_new a *relative* measure, not an absolute accuracy.
- The two networks (MLP vs CNN) use conceptually different normalisations, which is asymmetric.
- To be discussed with Jiri.
