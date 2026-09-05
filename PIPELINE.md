# How the generalized accuracy is computed

A description of what the code does and why, for someone who knows the paper but not the
repository. Notation follows the paper: `P1 = Ξ̄_x` (extended correct polytope, defined by
(9), (10), (12) — the full-precision network alone), `P2 = Ξ_x` (correct polytope, adds
(11), the linearity of the approximated network `Ñ`), `P3^k = Ξ̃_x^k` (the subpolytope of
`P2` where `Ñ` predicts class `k`, adding (13)). Operational commands are in the README;
this file is about the *why*.

---

## 1. What has to be computed

The generalized accuracy (19) is

    γ_T(Ñ) = Σ_x d̃₀(Ξ̃_x^c) / Σ_{x,k} d̃₀(Ξ̃_x^k)

so for every data point we need the modified mean width `d̃₀` of the correct subpolytope
and of **every** other one. By (18), `d̃₀(Ξ) = d̃(Ξ)` when `vol(Ξ) > 0` and `0` otherwise.

Two quantities are therefore needed per subpolytope: its **mean width**, and a verdict on
whether its **volume is zero**. They have wildly different costs, which is the whole
reason the pipeline is split in two.

## 2. Three states, not two

| state | question | how it is decided | measured cost |
|---|---|---|---|
| **empty** | is there *any* point in it? | one feasibility LP | ~1–35 s |
| **zero volume** | non-empty, but lower-dimensional? | Chebyshev radius `ϱ = 0` ⟺ `vol = 0` | ~1600–2650 s |
| **full-dimensional** | `vol > 0` | — | — |

"Empty" is a special case of "zero volume", and by far the common one: on the CNN only
1.75 (at b=4) to 2.74 (at b=16) of the ten classes are non-empty, and on the MLP 1.01. An
empty subpolytope contributes nothing either way, so detecting it early avoids computing
a mean width that would be zero regardless.

The cost ratio between the two tests is about **1:2000**. That is why the Chebyshev radius
is not computed everywhere.

## 3. Stage 1 — emptiness, then mean widths

Per tile (one augmented data point at one bit-width):

1. **Build the polytopes** from the shortcut weights (< 1 s).
2. **Pre-screen**: one feasibility LP per class — "does any point satisfy the constraints
   of `P3^k`?". Classes proven infeasible are dropped; their mean width is 0 by definition.
3. **Mean widths**: `N = 100` random directions; for each, two LPs give the width of `P2`
   and two more the width of each non-empty `P3^k`.

Three points of implementation that are not incidental:

**The same directions are used for `P2` and for every `P3^k`.** So if `P3^k = P2` as sets,
their widths coincide along *every* direction, not merely in expectation. This is what
makes the equality of section 4 testable to solver precision, and it answers the question
of why 100 directions sufficed for so many matching decimals: it is not the accuracy of
the estimator — the two bodies are the same set.

**The feasibility LPs are solved by interior point, not simplex.** A feasibility LP has a
zero objective, which leaves a simplex nothing to guide it while it tries to certify
infeasibility. Measured on the CNN at b=4, on three classes: simplex 195.8 / 194.1 /
599.9 s, interior point 37.0 / 37.3 / 33.1 s, same verdicts throughout. In production the
pre-screen went from 797 s to 8.5 s. Where interior point cannot decide either, the class
is **kept** and the direction LPs settle it — nothing is ever recorded as empty without
proof.

**`P1` is not computed.** It appears nowhere in (19), which involves only `P2` and its
subpolytopes, yet it cost two LPs per direction — about a quarter of each task. It is
computed on a separate subsample when the `V2/V1` ratio is wanted.

## 4. The criterion — where the lemma does the work for free

By the lemma (equal mean width ⟹ same set): if some `k*` satisfies `d̃(P3^{k*}) = d̃(P2)`,
then `P3^{k*} = P2`, and since the `P3^k` partition `P2` with disjoint interiors, **every
other subpolytope has empty interior, hence zero volume**. Equation (18) is settled for
that tile without a single Chebyshev LP.

Testing the *correct* class alone is enough: the indicator `1[V3^c = V2]` is 0 whether the
equality is carried by another class (the whole tile is misclassified) or by none.

The tolerance is not a free parameter. Because the directions are paired, an equality that
holds at all holds to solver precision — median gaps `1e-10` (CNN) and `1e-13` (MLP). On
the real data the verdict is constant for any tolerance in `[1e-8, 1e-4]`, four orders of
magnitude; below `1e-9` genuine equalities start being rejected. The `--tol_scan` option
prints this table so the claim can be shown rather than asserted.

**Frequency:** the criterion applies in 79–94% of CNN cases and in at least 98.9% of MLP
cases.

**Its limitation, stated plainly:** when it does not apply — `P2` split between several
subpolytopes of positive volume — the correctly classified part of `P2` receives no credit
at all. The resulting `γ` is therefore a **lower bound**, and its bias equals the frequency
of that case, which varies with `b`. That frequency is reported alongside `γ`, because
without it the curve is not interpretable. Equation (19) itself does not have this
limitation: it gives such a tile its proper partial credit.

## 5. Stage 2 — Chebyshev radii, only where needed

Run on the tiles the criterion could not settle, plus a sample of tiles it did settle, as
a cross-check: there the lemma predicts `ϱ = 0` for the other classes, and confirming it
validates the two criteria against each other.

Kept as a separate array on purpose. A Chebyshev LP costs 17–19 times a mean-width LP and
its cost is unpredictable; in the unscaled formulation it does not converge at all at b=4.
Running it inside stage 1 would put the primary computation at the mercy of the
unpredictable one.

Two implementation details that matter for correctness:

**The constraints are scaled by `‖a_j‖`**, so the `ϱ` column becomes all-ones. This is the
same LP — dividing an inequality by a positive number does not change its solution set —
but it is what makes b=4 solvable: unscaled, the LP fails after 2029 s; scaled, it
converges in 1628 s. At b=16 both agree to `6e-6`.

**A capped or failed LP is reported as `failed`, never as `empty`.** The distinction is not
cosmetic: `empty` would zero the polytope in (19), whereas `failed` says the value is
unusable and must be reported as such. Non-convergence is itself a result — it bears on
whether (18) is computable at scale.

## 6. What comes out

Per bit-width:

- **`γ` from (19)** with the zero-volume exclusion, once stage 2 supplies the radii;
- **`γ` from the lemma criterion**, a lower bound, free from stage 1;
- **the frequency of the cases where the criterion does not apply**, without which the
  second number cannot be read;
- **the tolerance scan**, showing the verdict does not depend on where the cut is made.

Reporting both, with the case frequency, is deliberate: they answer the same question
under different assumptions, and their disagreement is informative rather than
embarrassing.

## 7. Cost, measured

| | |
|---|---|
| mean-width LP (CNN, b=16) | 138 s |
| Chebyshev LP, scaled | 1628 s (b=4) — 2370 s (b=16) |
| full mean width, one polytope (200 LPs) | ~7.8 core-hours |
| one stage-1 task, 40 cores | ~15 min |
| stage 1, 1423 tiles | ~14,300 core-hours |

Direction LPs are solved by the **simplex**, deliberately. Interior point is faster there
too, but the two solvers disagree on the aggregated widths by `5.7e-08` — larger than the
`1e-08` tolerance the criterion decides on — and on the tile tested they classify it
differently. A faster solver that silently moves tiles between cases is not a gain.
