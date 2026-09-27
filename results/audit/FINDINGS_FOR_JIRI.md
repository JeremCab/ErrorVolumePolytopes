# Findings — augmented GACC campaign and CNN audit

Jérémie Cabessa, 11 September 2026. Everything below is measured; the one
paragraph that is a hypothesis is labelled as such.

---

## 1. The problem, stated precisely

γ computed by formula (19) on the mean widths, over the full non-augmented sets
(`results/data_for_jiri/results_df_{cnn,mlp}.csv`, 200 directions per polytope):

| b  | CNN γ | CNN ACC | MLP γ | MLP ACC |
|----|-------|---------|-------|---------|
| 4  | 0.5777 | 0.9419 | 0.9696 | 0.9869 |
| 6  | 0.5158 | 0.9950 | 0.9899 | 0.9990 |
| 8  | 0.4375 | 0.9990 | 0.9917 | 0.9990 |
| 10 | 0.4235 | 0.9990 | 0.9917 | 1.0000 |
| 12 | 0.4314 | 1.0000 | 0.9916 | 1.0000 |
| 16 | 0.4214 | 1.0000 | 0.9916 | 1.0000 |

**The MLP behaves as the theory predicts — γ increases with b. The CNN does the
opposite: γ *decreases* as the approximation gets better.** The augmented campaign
(1423 tiles) reproduces the same inverted shape (0.6136 at b=4 → 0.4297 at b=16 —
**TO RE-CHECK against the Jean Zay output before sending; the table above is
recomputed locally and verified, these two figures are not**), so augmentation does
not repair it.

## 2. Why no correction could have worked — mean width cannot see volume in n = 784

For convex bodies of comparable shape, mean width scales as V^(1/n). At n = 784:

    w1/w2 = 0.99   =>   V1/V2 = 0.99^784 ≈ 1/2600

**One percent of mean width is a factor 2600 in volume.** Measured confirmation:
each original point's tile versus its own augmented representatives gives
V2(aug)/V2(orig) = 0.99 / 1.00 / 1.00 (p10/p50/p90) at b=10 and 0.99 / 1.01 / 1.01
at b=16 — the tiles are indistinguishable in width while their volumes may differ
by three orders of magnitude.

Mean width bounds volume from above (a narrow body has small volume) but **not
from below**: a flat body keeps a near-maximal width. That asymmetry is the whole
problem. It explains, in one line, why every correction we tried failed:

- **(18), the zero-volume exclusion**, removes almost nothing: only *exactly* zero
  volume is visible to a width estimate.
- **incorrect P3^k carry 42–58 % of the denominator** — width ≈ 1.5, volume negligible.
- **the equal-width lemma's verdict is set by an arbitrary tolerance.**
  CORRECTED 2026-09-11 — the earlier claim "no variant restores the trend" was too
  strong. At tolerance >= 1e-3 the lemma variants DO give the expected increasing
  curve for the CNN; at <= 1e-4 they do not, and drop at b=16. See the tolerance
  sweep in MAIL_DRAFT_JIRI.md section 3. And where the trend looks right, the
  variants agree with ordinary ACC to 5e-4, so they add nothing to accuracy.
- **weighting** (tile size vs uniform; by d̃, ϱ, d̃·ϱ, ϱⁿ) changes nothing.

This is not a defect of our implementation nor of definition (19). It is a limit of
the surrogate, and it is provable in one line.

## 3. Coverage of P1 at b = 4 appears unattainable

Your MAIL_21 plan (more MCMC iterations until P1 is covered): re-running 5 originals
with the representative cap raised from 50 to 500 returned **500 for every one of
the 5** — the cap saturates at every value we try, so the tile count at b = 4 is not
approached. Consistent with the paper's own 2^|H̃| bound (p. 10). At b ≥ 10 the
question is moot: 0 new representatives are found because P2 already fills P1
(V2/V1 = 0.974 at b=10, 0.9998 at b=16).

Consequence: at small b, any augmented γ is a Monte-Carlo estimate over a sample of
P1, not an exhaustive volume-weighted sum.

## 4. The CNN's polytope is not well defined

x0 does not lie strictly inside its own polytope P2 — **for the CNN only**:

| | constraints | violated by x0 | margin of x0 |
|---|---|---|---|
| MLP, samples 0/1/2 | 3 849 | **0** | **−3.7e−03 / −3.9e−03 / −1.3e−03** |
| CNN, samples 0/1/2 | 41 361 | **258 / 186 / 577** | **+3.6e−07 / +2.4e−07 / +8.1e−07** |

At b = 16, sample 0: of 32 289 effective rows, **812 are tight** (|Ax+b| < 1e−6).
Splitting by originating layer, every ReLU block is clean (0 tight rows, median
margin 0.34–8.15); **all of it comes from the two max-pool blocks.**

The cause is **exact ties**: counting `other == max` exactly, pool 1 has 3 069 ties
of 5 880 constraints and pool 2 has 1 873 of 2 940. Most are windows where every
post-ReLU value is 0 — those rows have zero normal *and* zero bias, so they are
vacuous. But **402 ties at pool 1 (and 4 at pool 2) are between two *active*
neurons of non-zero value**. Likely mechanism (plausible, not verified):
Fashion-MNIST has large constant-background regions, and two 3×3 convolution
windows lying entirely inside one produce bit-identical outputs.

**These are genuine faces, not numerical artefacts.** The two tied positions are
different affine functions of the input — they depend on different pixels, coincide
at x0, and separate as soon as one moves. So the polytope we build is theoretically
correct, and the constraints must not be removed.

**But it means Ξ_x is not unique.** If x0 lies on the common boundary of several
linearity cells, which cell is retained depends on `max_pool2d`'s tie-break
(PyTorch takes the first index). Another convention gives another polytope, other
widths, another γ. **For a max-pool architecture on inputs with constant regions,
γ depends on an arbitrary tie-breaking convention.** The MLP escapes this entirely.

### 4b. A related discrepancy between the paper and the code

The paper replaces max-pool by a ReLU sub-network (max(x,y) = R(x−y) + y), i.e. a
tournament, which additionally fixes the order of the *losers* and therefore yields
a **smaller** polytope. The code imposes "i* is the max" directly (3 inequalities
per window), giving the maximal linearity region. The two differ and should be
reconciled.

## 5. The degeneracy is NOT what makes the CNN polytopes thin

We tested it directly. Removing the 812 tight rows and recomputing the Chebyshev
radius (CNN, sample 0, b = 16):

    ϱ(P2) full    = 1.748811e−05
    ϱ(P2) pruned  = 1.748321e−05      ratio 0.9997

**No effect.** (The tiny decrease is solver tolerance; removing constraints can only
increase the radius.) So although the boundary degeneracy is real, it does not
explain the thinness. For comparison, the MLP on the same sample and bit-width gives
**ϱ(P2) = 3.236e−01** — a factor 18 500.

*Hypothesis, not established:* the thinness comes simply from constraint count —
the CNN has 11 856 ReLU neurons against the MLP's 1 920, hence 32 289 effective
half-spaces against 3 849 cutting the same input box. This is directly testable
(see below) and the factor 18 500 is larger than a naive count argument would
suggest, so it may well be incomplete.

## 6. A solver finding that affects our ϱ results

The Chebyshev LPs were run with `scipy.linprog(method="highs")` (dual simplex). On
CNN sample 0 at b = 16 that **fails to converge within a 1800 s cap**, for both the
full and the pruned system. `method="highs-ipm"` (interior point) solves the same LP
in **472 s**. The same ordering gave a ×94 speed-up on the mean-width pre-screen.

So an unknown fraction of the ϱ entries recorded as `failed` are not hard cases but
solver choices. The code now accepts `--lp_method {highs, highs-ipm, auto}`; the
default is deliberately left unchanged, because switching turns `failed` entries
into real radii and therefore alters which polytopes enter (19).

---

## What we think this means

The negative result is robust: neither augmentation nor (18) restores the trend for
the CNN, and §2 explains why no correction of that family can. The positive content
is §4 — a property of the *definition* rather than of our code — and §6.

The natural next experiment is a CNN **without max-pool** (stride-2 convolutions),
which removes the tie degeneracy entirely and cuts the neuron count from 11 856 to
3 036. It tests §5's hypothesis and removes the confound: at present we cannot tell
whether the CNN's inverted γ is the surrogate's fault or the architecture's.
