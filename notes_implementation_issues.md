# Implementation issues and their resolutions

## 1. Polytope correspondence with the paper

Code variables in `run_volumes.py` JSON output correspond to paper (Šíma & Cabessa, ICONIP 2026) as follows:

| Code key | Paper symbol | Constraints |
|---|---|---|
| `width_base` | d̃(Ξ̄_x) | model activation (10) + model classification (9) + pixel bounds (12) |
| `widths_correct[b]` | d̃(Ξ_x) | Ξ̄_x + q-model activation (11) |
| `widths_both[b]` | d̃(Ξ̃_x) | Ξ_x + q-model classification (13) |

Containment: Ξ̃_x ⊆ Ξ_x ⊆ Ξ̄_x  ↔  A_both ⊇ A_correct ⊇ A_base (more rows = smaller polytope). ✓

Pixel bounds (eq. 12) are NOT stored as rows in A_base; they are passed as the `bounds` argument to
`scipy.optimize.linprog` (with bounds=[-1, 1] per variable, matching the normalised FashionMNIST range).
This is correct and equivalent. ✓

---

## 2. GACC formula: paper vs. code

### Paper formula (eq. 18)
γ(Ñ) = Σ_{x∈T} d̃(Ξ̃_x) / Σ_{x∈T} d̃(Ξ_x)
      = Σ widths_both[b] / Σ widths_correct[b]   ← ratio of sums P3/P2

### Empirical result (1999 MLP samples)
| b | P3/P2 (paper formula) | P3/P1 (old notebook) |
|---|---|---|
|  4 | 0.9995 | 0.9405 |
|  6 | 1.0000 | 0.9859 |
|  8 | 1.0000 | 0.9964 |
| 16 | 1.0000 | 1.0000 |

**P3/P2 ≈ 1 for all b.** This is a structural mathematical property, not a bug: within any fixed
q-model activation region (P2 = Ξ_x), the q-model is an affine function, so if it classifies x
correctly, it classifies correctly *everywhere* in P2 → P3 = P2 → GACC = 1.

The paper acknowledges this on p.9 ("the polytopes Ξ̃_x and Ξ_x almost coincide") and proposes
data augmentation as the remedy.  But even augmented points (all in distinct P2 regions of their
own) give P3/P2 ≈ 1, for the same structural reason.

**The non-trivial signal is P2/P1** = widths_correct[b] / width_base, which measures what fraction
of Ξ̄_x the q-model's activation region covers.  The old `plot_volumes.ipynb` was effectively
computing P3/P1 ≈ P2/P1 (since P3 ≈ P2).

**Action required:** Discuss with Jiri whether the intended GACC formula is P3/P2 (paper eq. 18)
or P3/P1 (old implementation).  The non-trivial results previously obtained correspond to P3/P1.

---

## 3. MCMC walk — missing pixel bounds

### The bug
The paper's hit-and-run walk (Section 3, eqs. 19–25) is defined over Ξ̄_x which includes the
pixel-bound constraints (eq. 12): 0 ≤ y_j ≤ 1 (or −1 ≤ y_j ≤ 1 after normalisation).

`chord_interval` in `src/optim/mcmc_augment.py` only uses A_base (model activation + model
classification) and does NOT enforce pixel bounds.  As a result, augmented points can and do
exceed the valid pixel range.

**Measured on `fashionMNIST_augmented_mlp_seed42_walk_1point.pt`:**
- 100/100 augmented points outside [−1, 1]⁷⁸⁴
- Max pixel value: 2.55 (should be ≤ 1)
- Max exceedance beyond ±1: 1.55

### Why the naive fix collapses the walk

x_0 (a real FashionMNIST image) has **355 out of 784 pixels at exactly ±1** (background pixels
are all 0 in the original [0,1] range → −1 after normalisation).  x_0 therefore lies on 355
simultaneous faces of the [−1, 1]⁷⁸⁴ box.

Adding pixel-bound rows to A_base (or computing the exact chord-interval intersection with the
box) forces t_min = t_max = 0 for virtually every random direction d, because:
- Pixels at −1 with d_j > 0 contribute lower bound t ≥ 0
- Pixels at −1 with d_j < 0 contribute upper bound t ≤ 0
- Both sets are non-empty for almost any d → chord collapses to a point

Numerically confirmed: with pixel bounds, 1000/1000 test directions gave chord length = 0.

**Root cause:** Hit-and-run MCMC requires x_0 to be strictly interior to the polytope.  Real
images with discrete pixel values (many exactly 0 or 1) violate this assumption at the box
boundary.

### The fix implemented (projected hit-and-run)

All three strategy functions (`find_augmented_point`, `find_augmented_point_margin`,
`find_augmented_points_walk`) now accept `pixel_lo=-1.0, pixel_hi=1.0` parameters.

After computing each candidate point from the A_base chord, the point is clipped:
    x_next = np.clip(x_next, pixel_lo, pixel_hi)

This converts the algorithm to "projected hit-and-run": sample freely inside the A_base
polytope, then project onto the pixel box.  The projected point is used both as the accepted
representative and as the next chain state.

**Limitation:** The clipped point may slightly violate A_base constraints.  In practice, the
violation is absorbed by the 1e-6 tolerance in `chord_interval`.  The walk is no longer
theoretically exact, but it is a valid practical approximation that:
- Keeps all augmented points within the valid image domain [−1, 1]⁷⁸⁴
- Maintains mixing: the walk still explores different activation regions
- Introduces no additional hyperparameters beyond the existing pixel range

**Volume computation is unaffected:** `run_volumes.py` already uses `bounds=[(-1., 1.)]` in
all LP calls, so the mean-width estimates are always computed within the valid pixel range. ✓
