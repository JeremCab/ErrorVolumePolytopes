# Hit-and-Run MCMC: Chord Intersection with Polytope #1

## Context

Used to find augmented data points x' inside Polytope #1 (model activation + model
classification constraints) where the q-model has a **different activation pattern**
than at the original x₀. One augmented point per original x_i is targeted.

## Convention

The codebase uses `Ax + b ≤ 0` throughout (A and b arrays as stored in
`build_polytopes.py`). `_prep_lp` converts to scipy's `A_ub @ x ≤ b_ub` by
setting `b_ub = -b + eps` before passing to `linprog`.

## Chord Computation (no LP solver needed)

Given:
- **x₀**: current point, inside Polytope #1 (i.e. Ax₀ + b ≤ 0)
- **d**: random unit direction in ℝ⁷⁸⁴ (or ℝ^input_dim)
- **Polytope #1**: `{x : Ax + b ≤ 0}`

Parameterize the line: `x(t) = x₀ + t·d`.

Substituting into the constraint:

```
A(x₀ + t·d) + b ≤ 0
⟺  t·(Ad) ≤ −(Ax₀ + b)
```

Define:
```python
c = A @ d          # shape (m,)
s = -(A @ x0 + b)  # shape (m,), all >= 0 since x0 is inside
```

Then row by row:

| case   | constraint on t        |
|--------|------------------------|
| cᵢ > 0 | t ≤ sᵢ/cᵢ  (upper)     |
| cᵢ < 0 | t ≥ sᵢ/cᵢ  (lower)     |
| cᵢ = 0 | always satisfied       |

```python
t_max = np.min(s[c > 0] / c[c > 0])   # farthest point in +d direction
t_min = np.max(s[c < 0] / c[c < 0])   # farthest point in -d direction
```

Sample `t ~ Uniform(t_min, t_max)`, then `x' = x₀ + t·d`.

Note: `t_min < 0 < t_max` is guaranteed when x₀ is strictly interior.

## Acceptance Step

After sampling x', evaluate the q-model's activation pattern at x'. Accept x' if
the pattern differs from the one at x₀. Otherwise, resample a new direction and
repeat.

## Cost

- Chord endpoints: 2 `np.dot` + 2 scalar min/max — **no LP call**
- Volume at x': 200 × 2 LP calls (same as for original x_i), to compute V₂(x') and V₃(x')
- Target: 1 accepted x' per original x_i (2K augmented points for a 2K-sample experiment)
