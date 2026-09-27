Draft mail to Jiri — the augmented campaign, GACC trend. 12 Sep 2026, v2.
Table 1 merges the former tables 1+2 (tolerance as columns); Table 2 is the
radius-based exclusion on b = 6, 10, 16.
Deliberately leaves out the max-pool / x0-on-the-boundary findings
(see MAIL_DRAFT_JIRI.md) — that is a separate message.

---

Hi Jiri,

I hope you are doing well.

The augmented campaign is finished: 150 original points, at most 5
representatives each, b in {4, 6, 10, 16}, i.e. 1423 tiles in total. The
short version is that I cannot restore the expected trend for the CNN, and
that this is not caused by b=4.


1. Generalized accuracy, as a function of the tolerance in (18)

All 1423 tiles. "no excl." is (19) with d_0 = d, i.e. no zero-volume
exclusion at all. The other columns apply (18) wherever our lemma proves the
volume is zero, the equality d(P3^k) = d(P2) being declared when the
RELATIVE gap |d(P3^c) - d(P2)| / d(P2) falls below "tol". Each column is
therefore a candidate curve for Fig. 2 (right); read downwards.

                                   tol on the relative gap of mean widths
   b   tiles   no excl.      1e-3    1e-4    1e-6    1e-8   1e-10
   4     900     0.6136     0.925   0.865   0.822   0.800   0.748
   6     206     0.5450     1.000   1.000   0.868   0.814   0.630
  10     158     0.4328     1.000   1.000   0.916   0.773   0.459
  16     159     0.4297     1.000   0.971   0.950   0.773   0.447
                --------   ------  ------  ------  ------  ------
  trend in b     decreas.    incr.   incr.   incr.    flat  decreas.

Two things worry me here.

The "no excl." column is the only quantity in the table that contains no
free parameter — the only subpolytopes removed are those an LP proves
infeasible — and it decreases with b, exactly as in Fig. 2 (right). So the
augmentation alone does not repair the figure.

And the sign of the trend is set by the tolerance. I had expected the rows
to be stable between 1e-8 and 1e-4, and they are not. The equalities
themselves are sharp: the median gap is 1e-10 to 2e-9, and below 1e-9 we
start rejecting genuine ones. So 1e-8 is the tightest value I can justify,
and it is precisely where the trend disappears. At 1e-6 the curve is
monotone increasing, which is what we want, but I have no argument for
preferring 1e-6 to 1e-8 other than the result it gives.

Removing b=4 does not help: 0.5450 -> 0.4328 -> 0.4297 in the first column,
and the same picture in the others. The problem is not confined to the small
bitwidths. (For reference, the fraction of tiles where the lemma does not
apply, at tol = 1e-8, is 16.1 / 12.1 / 6.3 / 12.6 % for b = 4 / 6 / 10 / 16.)

One remark on the lemma, since it matters for reading this table. It settles
(18) whenever SOME class fills P2, not only when the correct one does — if
P3^k' = P2 for k' != c, then P3^c has zero volume and the tile scores 0. Such
tiles exist only at b=4: 22 out of 755, and none at all at b = 6, 10, 16. So
whole-tile misclassification is a b=4 phenomenon, and the three larger
bitwidths are unaffected by this subtlety.


2. The same, with (18) applied from the actual Chebyshev radii

Here no lemma is used: every rho is computed, and a subpolytope is dropped
when its radius falls below an ABSOLUTE threshold. b=4 is omitted because
only 274 of its 589 LPs converged within 3600 s; at b = 6, 10, 16 the
convergence is 100%.

                                                    % of incorrect   median
   b   tiles   no excl.   rho<=1e-6  <=5e-5  <=1e-4   zeroed @1e-4   rho(P3^k)/rho(P2)
   6      27     0.3440      0.2817  0.2760  0.2760          14.0%      1.00
  10      14     0.2258      0.2258  0.2190  0.2190           1.7%      1.00
  16      25     0.3458      0.3550  0.8498  1.0000         100.0%      0.027

The exclusion removes 14%, 1.7% and 100% of the incorrect subpolytopes at
the same threshold — it is not monotone in b, so it cannot produce a smooth
curve. The reason is that an absolute threshold meets a distribution of
radii that slides with b: at b=10 the incorrect radii are around 2.8e-4,
above the cut, and nothing is removed; at b=16 they are around 1.4e-5, below
it, and everything is removed, which sends gamma to 1. In other words the
gate is not detecting zero volume, it is detecting the scale of the tile at
that bitwidth.

Two caveats on this table. First, these tiles are the ones where the lemma
does not apply, so they are far from a random sample — they are selected on
"no class fills P2", which is directly correlated with the quantity being
averaged. The cleanest way to see it is the "no excl." column, which is the
very same formula as in Table 1: it reads 0.3440 / 0.2258 / 0.3458 here
against 0.5450 / 0.4328 / 0.4297 there, i.e. 0.20, 0.21 and 0.08 lower. That
gap is the selection bias expressed in the units of gamma itself, and it is
not constant in b. So the two tables should be read column against column
within each one, and never across. Second, and this is something I cannot
explain: the last
column. Since the P3^k partition P2, at most one of them can contain P2's
largest inscribed ball, so the ratio rho(P3^k)/rho(P2) should be well below
1 for all the others. At b=6 and b=10 its median over the INCORRECT classes
is 1.00. Only b=16 behaves as the partition requires. Either the radii are
wrong at those bitwidths, or I am missing something geometric. Do you have
an idea?


3. On your suggestion of eps = 5e-5, or 1e-4, in (18)

I tried to measure the numerical noise rather than assume it, and found two
independent floors. The LP itself: the radii come back with a minimum of
-2.4e-8, and a negative radius is meaningless, so the solver cannot resolve
rho below roughly 2e-8. The constraints: rebuilding the same polytopes in
float64 and measuring how far each face moves, the largest displacement at
b=16 is 1.08e-5.

The radii in question have median 5.4e-5 and 1.8e-4, i.e. 5 to 17 times
above the larger of the two floors, so at b=16 I do not think they can be
attributed to accumulated float32 error. At b=4 the situation is different,
and worse: 10 neurons out of 11856 change activation between float32 and
float64, so P2 itself is not uniquely defined there.


4. One more negative result

The cross-check failed. On the tiles where the lemma applies, it predicts
that every incorrect subpolytope has zero volume; only 1 out of 18 has
rho <= 1e-6. Either the lemma cannot be transferred to widths estimated over
100 directions — equality of the widths in 200 sampled directions does not
imply equality of the sets — or these radii are not reliable. I lean towards
the first, but the anomaly in Table 2 makes me less sure.


Before going further I would like your opinion on one point. The mean width
is blind to how flat a body is: at n = 784, a difference of 1% in width
corresponds to a factor of 2600 in volume, and a flat sliver keeps a nearly
maximal width. Given this, do you think a zero / non-zero gate on the volume
can repair (19) in principle? Or should we rather look for a different size
measure?

All the best,
Jérémie
