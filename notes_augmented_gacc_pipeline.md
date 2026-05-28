# Augmented GACC pipeline — important notes

## Missing piece: original points T must be included in T*

Jiri's formula (18) evaluates GACC over T* = T ∪ ⋃_{x∈T} T_x, which includes
BOTH the original 200 data points x₀ ∈ T AND the augmented representatives T_x.

Our augmented `.pt` files (`fashionMNIST_augmented_{mlp,cnn}_seed42_walk_aug200.pt`)
contain ONLY the augmented reps — not the original x₀s.

**Consequence:** when computing the new GACC, results must be aggregated from:
1. `run_volumes_v3k` on the 200 original samples (subset of existing 1K results)
2. `run_volumes_v3k` on the augmented dataset (student's cluster run)

New GACC(b) = [Σ_{original 200} V3_c(x,b) + Σ_{augmented} V3_c(y,b)]
            / [Σ_{original 200} ΣV3_k(x,b) + Σ_{augmented} ΣV3_k(y,b)]

The existing 1K-point results cover the 200 original samples already (they are
the first 200 indices of `fashionMNIST_correct_{mlp,cnn}.pt`). So no extra
computation is needed for the original T part — just filter the existing results
to indices 0–199 when aggregating.
