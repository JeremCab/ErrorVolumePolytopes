# Augmentation v2 — Strategies for Finding x' with V3 << V2

## Why the current MCMC augmentation is insufficient

The current `find_augmented_point` (v1) accepts x' when the **q-model activation pattern**
at x' differs from the one at x₀.  This is a necessary but not sufficient condition for
V3(x') << V2(x').

The gap V2 − V3 is determined solely by how tightly the **q-model classification boundary**
cuts into V2.  That boundary is tight only when x' is **near the q-model's decision boundary
for class c**, i.e. when

    q_c(x') ≈  max_{k ≠ c}  q_k(x')

A different activation pattern just means the q-model operates in a different linear region
at x'; within that region the q-model can still classify c with very high confidence →
V3(x') ≈ V2(x').

---

## Strategy A — LP margin minimisation (no MCMC)

**Goal.** Find x' inside Polytope #1 that *minimises* the q-model's classification margin
for class c:

    min   q_c(x') − max_{k ≠ c} q_k(x')
    s.t.  x' ∈ Polytope #1  (A_base x' + b_base ≤ 0)

Since the q-model is piecewise-linear, the objective is linear within any fixed linear
region.  A two-step approach:

1. Fix the q-model activation pattern at x₀ (i.e. stay in the same linear region).  Solve
   the LP.  The solution x'* has the smallest possible margin within that region.
2. Optionally, explore other activation-pattern regions by initialising from x'* and
   repeating.

**If the LP solution has margin < 0**, the q-model misclassifies x' → V3(x') = 0
(empty Polytope #3), which is the extreme case.
**If margin ≥ 0 but small**, the q-model classification constraint is tight →
V3(x') << V2(x').

**Pros.** Directly targets the quantity that drives V2 − V3.  Uses the LP infrastructure
already in place (same constraint format Ax + b ≤ 0).
**Cons.** One extra LP per sample.  Requires picking a target competitor class k.

---

## Strategy B — Chord-endpoint greedy search (no extra LPs)

**Goal.** Replace uniform sampling `t ~ Uniform(t_min, t_max)` with a targeted selection
that maximises the chance of landing near the q-model's classification boundary.

**Algorithm** (replaces the inner loop of `find_augmented_point`):

```
best_x'    = None
best_score = +∞        # score = q-model margin for class c (lower = better)

for _ in range(max_tries):
    d = random unit direction
    t_min, t_max = chord_interval(x0, A, b, d)
    if not valid: continue

    # Evaluate q-model at both chord endpoints (+ small inward step for numerical safety)
    eps_t = 1e-4 * (t_max - t_min)
    for t in [t_min + eps_t, t_max - eps_t]:
        x_cand = x0 + t * d
        logits  = qmodel(x_cand)          # raw logits, shape (n_classes,)
        margin  = logits[c] - max logits excluding c
        if margin < best_score:
            best_score = margin
            best_x'    = x_cand

return best_x'   # point with the smallest q-model margin across all directions
```

**If best_score < 0**: q-model misclassifies best_x' → V3 = 0.
**If best_score ≥ 0 but small**: q-model barely predicts c → V3 << V2.

The returned x' is still inside Polytope #1 by construction (it is on the boundary of
Polytope #1, where either the model activation or model classification constraint is
active).

**Pros.** No new LP calls.  Reuses the existing chord infrastructure.  The 200 random
directions already drawn provide a good coverage of the boundary.
**Cons.** x' is forced to be on the Polytope #1 boundary — may produce slightly
degenerate Polytope #2 (but it is still a valid interior point of Polytope #2 unless
the q-model activation boundary also happens to pass through x').

---

## Recommendation

Start with **Strategy B** (chord-endpoint greedy): it requires minimal code changes, no
extra LP calls, and directly searches for q-model uncertainty.  If the resulting GACC
curves are still flat, escalate to **Strategy A** (LP margin minimisation) which is the
theoretically optimal approach.
