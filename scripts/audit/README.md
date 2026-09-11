# Audit scripts

Written 2026-09-11 to answer one question: **the MLP shows the expected trend and the
CNN does not, so what is CNN-specific?** The suspect was the CNN→MLP linearisation,
since every CNN polytope is built from it.

Run from the repository root. On a Jean Zay login node: `module load pytorch-gpu/py3/2.8.0`.

| script | question | what it found |
|---|---|---|
| `shortcut_invariant.py` | do the shortcut weights reproduce the network's logits? | **yes, both networks** — 2.1e-07 (MLP) and 9.9e-07 (CNN) relative at x0. The linearisation is NOT the bug. |
| `x0_membership.py` | is x0 inside its own polytope? | **MLP yes** (margin −3.7e−03, 0 violations); **CNN no** — x0 violates 258/186/577 constraints by ~1e−07. This is what the `eps = 1e-6` slack in `_prep_lp` has been hiding. |
| `constraints_by_layer.py` | where does the degeneracy come from? | **entirely from max-pool.** All six ReLU blocks are healthy (0 tight, margins 0.34–8.15). The two max-pool blocks hold 4 536 zero-norm rows of 8 820 and 4 942 tight constraints. |
| `float_precision.py` | is the polytope well defined in float32? | **at b=16 yes** (faces move by at most 1.08e−05, below the radii attributed to noise, so Jiri's rounding hypothesis does not hold); **at b=4 no** — 10 neurons of the quantised network flip between float32 and float64. |

The mechanism: max-pool follows ReLU, so in a window where several inputs are saturated
their post-ReLU values are exactly 0 and "other ≤ max" becomes 0 ≤ 0. Those rows are
all-zero (ReLU zeroes `s_bias` too) and get pruned, so they are harmless; the ~400 per
pool that are tight with a NON-zero normal are the ones putting x0 on the boundary, and
they come from windows whose maximum is an active neuron sitting just above threshold.

**Not established:** that this degeneracy *causes* the thin polytopes (ϱ/V2 ~ 1e−4), the
Chebyshev convergence failures, or the ϱ(P3^k)/ϱ(P2) ≈ 1 anomaly. That is a hypothesis.
The test that would settle it: recompute ϱ(P2) with the tight max-pool rows removed and
see whether the inradius grows.

Note also a discrepancy with the paper: it replaces max-pool by a ReLU tournament
(`max(x,y) = R(x−y) + y`), whereas the code imposes "i* is the max" directly. The
tournament additionally fixes the order of the losers, so it yields a *smaller* polytope.
