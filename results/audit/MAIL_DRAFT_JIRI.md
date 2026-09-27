# Draft mail to Jiri — short version

Dear Jiri,

We have finished the audit and the augmented campaign. The short version: **we found
no bug in the linearisation**, but we did find why the CNN curve misbehaves, and it
is a problem with the estimator rather than with our code.

## 1. Where the CNN/MLP difference comes from

γ by formula (19) on mean widths, non-augmented sets, 200 directions per polytope
(999 CNN / 989 MLP points):

| b  | CNN γ | MLP γ | share of (19)'s denominator carried by **wrong** classes, CNN | MLP |
|----|-------|-------|------|------|
| 4  | 0.578 | 0.970 | 42.2 % | 3.0 % |
| 6  | 0.516 | 0.990 | 48.4 % | 1.0 % |
| 8  | 0.438 | 0.992 | 56.3 % | 0.8 % |
| 10 | 0.424 | 0.992 | 57.7 % | 0.8 % |
| 12 | 0.431 | 0.992 | 56.9 % | 0.8 % |
| 16 | 0.421 | 0.992 | 57.9 % | 0.8 % |

For the CNN, the *incorrect* classes carry **42 → 58 %** of the denominator, and
their share *grows* with b. For the MLP they carry under 3 %. That share is exactly
1 − γ, so this is the mechanism of the "spurious improvement" at small b: not that
the CNN gets better when coarsely quantised, but that at small b the region P2 is
small enough to sit inside a single class, while at large b it grows until it meets
several class regions — and each sliver keeps a near-maximal mean width.

## 2. Why mean width cannot arbitrate this — one line

For convex bodies of comparable shape, mean width scales as V^(1/n). At n = 784:

    w'/w = 0.999  =>  V'/V = 0.999^784 = 0.46

**A relative difference of 0.1 % in mean width is a factor 2.2 in volume.** Mean
width bounds volume from above but not from below: a flat sliver keeps a nearly
maximal width. So the wrong classes contribute to the denominator in proportion to
a quantity that is blind to how thin they are.

## 3. Every variant we tried is governed by an arbitrary tolerance

Using the equal-mean-width lemma, γ_lemma = Σ_x V2·1[V3^c = V2] / Σ_x V2. The only
free parameter is the tolerance at which "=" is declared. CNN:

| tolerance | b=4 | b=6 | b=8 | b=10 | b=12 | b=16 | increasing in b? |
|-----------|-----|-----|-----|------|------|------|------------------|
| 1e-2 | 0.963 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | **yes** |
| 1e-3 | 0.929 | 0.993 | 0.999 | 1.000 | 1.000 | 1.000 | **yes** |
| 1e-4 | 0.835 | 0.973 | 0.983 | 0.996 | 0.997 | 0.936 | no |
| 1e-6 | 0.771 | 0.932 | 0.952 | 0.970 | 0.970 | 0.906 | no |
| 1e-8 | 0.763 | 0.923 | 0.928 | 0.930 | 0.930 | 0.870 | no |

**The trend flips at the tolerance.** And by §2, a tolerance of 1e-3 — where the
trend looks right — admits pairs whose volumes differ by more than a factor two, so
the "P3^c fills P2" verdict is not a volume statement at that setting. At 1e-8,
where the test is meaningful, the curve is non-monotone and drops at b = 16.

The same holds for the other variants (splitting handled as *ultra-strict* /
*strict* / *partial*): at tolerance 1e-3 they all give the expected increasing
curve — but they then agree with ordinary ACC to within 5·10⁻⁴ (CNN: γ = 0.9424,
0.9947, 0.9989, 0.9994, 1.000, 1.000 against ACC = 0.9419, 0.9950, 0.9990, 0.9990,
1.000, 1.000), so they carry no information beyond accuracy.

## 4. Two things about the CNN geometry that you should know

**(a) x₀ is not in the interior of its own polytope, for the CNN only.** At b = 16,
sample 0: of 32 289 effective constraints, 812 are tight at x₀ and 386 are violated
by up to 5·10⁻⁷. The MLP has 0 and 0, with a margin of −3.7·10⁻³. All of it comes
from the two max-pool blocks; every ReLU block is clean.

The cause is **exact ties**: 402 constraints at pool 1 are equalities between two
*active* neurons of non-zero value (likely two 3×3 windows lying inside the same
constant background region, giving bit-identical outputs). These are genuine faces —
the two tied positions are different affine functions that coincide at x₀ — so the
polytope is correct and they must not be removed. **But it means Ξ_x is not unique:
which linearity cell is retained depends on max_pool2d's tie-break.** For a
max-pool architecture, γ depends on an arbitrary convention. The MLP escapes this.

We checked whether this degeneracy is what makes the CNN polytopes thin. It is not:
removing the 812 tight rows changes the Chebyshev radius from 1.7488·10⁻⁵ to
1.7483·10⁻⁵ (ratio 0.9997). The thinness has another cause — the CNN has 11 856
ReLU neurons against the MLP's 1 920, hence 32 289 half-spaces against 3 849 cutting
the same box. For reference ϱ(P2) is 1.7·10⁻⁵ for the CNN and 3.2·10⁻¹ for the MLP.

**(b) The paper and the code model max-pool differently.** The paper replaces it by
a ReLU tournament, max(x,y) = R(x−y) + y, which also fixes the order of the losers
and so yields a *smaller* polytope. The code imposes "i* is the max" directly,
giving the maximal linearity region. These should be reconciled.

## 5. Coverage of P₁ at b = 4

Your MAIL_21 suggestion (more MCMC iterations until P₁ is covered): raising the
representative cap from 50 to 500 on 5 originals returned **500 for each** — the cap
saturates at every value we try, so the tile count at b = 4 is never approached,
consistent with the 2^|H̃| bound on p. 10. At b ≥ 10 the question is moot: 0 new
representatives are found, because P2 already fills P1 (V2/V1 = 0.974 at b=10,
0.9998 at b=16).

## What we propose

Retrain the CNN **without max-pool** (stride-2 convolutions). It removes the tie
degeneracy by construction, and it cuts the neuron count from 11 856 to 3 036 at
*identical* parameter count (97 066), so the comparison stays fair. On a probe with
random weights, P2 goes from 41 361 to 6 081 rows, tight rows from 812 to 0, and
ϱ(P2) from 1.7·10⁻⁵ to 1.0·10⁻¹.

This will not repair §2 — that is architecture-independent — but it removes the
confound: at present we cannot tell whether the CNN's inverted γ is caused by the
estimator or by the architecture.

Best,
Jérémie
