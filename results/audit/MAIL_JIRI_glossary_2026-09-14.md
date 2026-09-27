Short reply to Jiri — glossary + answer to his ps. 14 Sep 2026.

---

Hi Jiri,

Sorry about that -- I wrote with my own working vocabulary and made the text
harder to read than the experiment itself. Here is the dictionary, in the
notation of the paper.

- tile = one data point y of the augmented dataset T*, together with its
  correct polytope Xi_y (which I called P2) and its sub-polytopes Xi~_y^k
  (which I called P3^k). I use the word because the Xi_y tile the extended
  polytope of the original point. "1423 tiles" means 1423 pairs (augmented
  point, b).

- Computation of polytopes = building the constraint matrices (9)-(13) from
  the shortcut weights, for Xi_y and for each Xi~_y^k.

- emptiness test (1 LP) = for each class k, one feasibility LP with a zero
  objective, asking whether Xi~_y^k is empty. When HiGHS proves infeasibility,
  the sub-polytope is empty, d(Xi~_y^k) = 0, and I do not need the 2 x 100
  linear programs of its mean width. On the CNN only 1.75 (at b=4) to 2.74 (at
  b=16) of the ten classes are non-empty on average, so this is what makes the
  campaign affordable.

- filter = apply (18), i.e. set d_0 = 0 for the sub-polytopes declared of zero
  volume. Nothing is removed from the dataset; only terms of the denominator
  of (19) are set to zero.

- "the fraction of tiles where the Lemma does not apply" = the fraction of
  points y for which NO class k satisfies d(Xi~_y^k) = d(Xi_y). There the
  Lemma says nothing, and (18) needs the Chebyshev radius instead.

So the two steps are:

(STEP 1) for each y and each b: build (9)-(13); one feasibility LP per class;
then the mean width of Xi_y and of every non-empty Xi~_y^k, at 2 x 100 LPs
each.

(STEP 2) for the points where the Lemma does not apply: the Chebyshev radius
of Xi_y and of every non-empty Xi~_y^k.


About your ps -- I think we agree.

The Lemma itself does not need additivity: it only tests an equality between
two mean widths and concludes Xi~^k = Xi_y as sets. But you are right that the
equality is tested on estimates over 100 directions, and equality in 100
directions cannot certify that two sets are equal, so the criterion can indeed
discard sub-polytopes of positive volume. And you are right that a
mean-width criterion feeding a mean-width GACC is suspect: that is exactly
what Table 1 shows, since the sign of the trend follows the tolerance.

One small correction though: the "no excl." column is the one where the Lemma
is NOT used at all -- every non-empty sub-polytope is counted. The Lemma is
what the tol columns apply. So if we keep only rigorously proven zero volumes,
we are in the tol column at the tightest value the solver supports, which is
1e-8 (below 1e-9 the solver's own precision starts rejecting genuine
equalities). The two columns I would defend are therefore "no excl."
(0.6136 -> 0.4297) and tol = 1e-8 (0.800 -> 0.773), and neither gives the
trend we want.

Yes, let us talk -- would <day, time> suit you on WhatsApp?

All the best,
Jérémie
