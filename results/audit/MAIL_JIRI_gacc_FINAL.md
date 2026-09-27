Mail to Jiri — augmented campaign, GACC trend. 13 Sep 2026, final.
Jérémie's own draft, with the corrections and additions of 2026-09-13.

---

Hi Jiri,

I hope you are doing well.

I ran the following experiment :

(DATASET) 150 original points + data augmentation using MCMC hit & run and
keeping at most 5 most distant representatives for each b in {4, 6, 10, 16}
(this is too few for b=4, but the augmentation keeps finding a very large
number of new augmented points in this case) => 1423 tiles, including the 150
original points.

(STEP 1) Computation of polytopes, filter by emptyness test (1 LP),
computation of volumes of polytopes and filter by the Lemma criterion: if
V_3^k = V_2, then P_3^k = P_2 and thus V_3^k' = 0 for all k' != k.

(STEP 2) Computation of Chebyshev radii, on the polytopes that survived step 1
in the tiles where the Lemma did not apply -- in the other tiles the Lemma has
already assigned every volume.

The short version is that I cannot restore the expected trend for the CNN --
except at a few tolerances, none of which I can justify --, and that this is
not caused by the b=4 problematic case.


1. Generalized accuracy (GACC) using our Lemma (no Chebyshev).

--------------------------------------------------------------
                        tol on the relative gap of mean widths
 b   #tiles   no excl.  1e-3    1e-4    1e-6    1e-8   1e-10
--------------------------------------------------------------
 4     900    0.6136   0.925   0.865   0.822   0.800   0.748
 6     206    0.5450   1.000   1.000   0.868   0.814   0.630
10     158    0.4328   1.000   1.000   0.916   0.773   0.459
16     159    0.4297   1.000   0.971   0.950   0.773   0.447
--------------------------------------------------------------
trend in b    decr.    incr.  neither  incr.  neither  decr.
--------------------------------------------------------------

- The "no excl." column means all P_3^k surviving step 1 are kept in the
  computation of the GACC.
- The tol columns filter the P_3^k by relative gap, i.e.: if
  |d(P3^k) - d(P2)| / d(P2) <= tol, then P3^k = P_2 (by the Lemma) and thus
  V3^k' = 0 for all k' != k; then computes the GACC.

Hence:

(i) The "no excl." column is the only quantity in the table that contains no
free parameter. Here, the augmentation does not repair the trend.

(ii) The sign of the trend is set by the tolerance at which the Lemma is
applied. A tolerance of 1e-6 seems to give nice results! But we have no
argument for preferring one tolerance over another.

(iii) Removing b=4 does not help. The problem is not confined to the small bit
widths.

(iv) For reference, the fraction of tiles where the Lemma does not apply, at
tol = 1e-8, is 16.1% / 12.1% / 6.3% / 12.6% for b = 4 / 6 / 10 / 16.

(v) Note that the exclusion does remove a large part of the denominator: at
b=16, gamma goes from 0.4297 ("no excl.") to 0.773 at tol=1e-8. So your
correction (18) does a lot of work -- it simply does not change the trend.


2. Generalized accuracy (GACC) using the Chebyshev radius (rho).

Here, we focus on column 1e-8 of Table 1 and compute the rho of the surviving
polytopes P_2 and P_3^k. Then, we filter the P_3^k's as follows: if
rho(P_3^k) <= threshold, then V_3^k = 0.

Note that at b = 6, 10 and 16 every tile receives a verdict: 181 + 25,
148 + 10 and 139 + 20 tiles are settled by the Lemma and by rho respectively.
So Table 2 is (19) computed in full -- no tile left out, and no lower bound
involved.

I chose to focus on column 1e-8 of Table 1 as it represents the solver's
precision, but a posteriori, I should have also done it for column 1e-6 which
provided us the expected trend. The problem is that we don't have any
criterion to choose one or another threshold at this time. Also, I left b=4
out of Table 2: only 274 of the 589 Chebyshev LPs converged within the 3600 s
limit there, so the radii are simply not computable at that bit width for now
(other solvers to be investigated).

--------------------------------------------------------------
                                      threshold on rho(P3^k)
 b   #tiles   rho<=0    1e-6    1e-5    5e-5    1e-4    5e-4
--------------------------------------------------------------
 6     206   0.8142  0.8137  0.8137  0.8220  0.8220  0.8652
10     158   0.7735  0.7735  0.7735  0.7741  0.7741  1.0000
16     159   0.7729  0.7860  0.8223  0.9801  1.0000  1.0000
--------------------------------------------------------------
trend in b   decr.  neither neither neither neither  incr.
--------------------------------------------------------------

- The "rho<=0" column means that if rho(P_3^k) <= 0, then V_3^k = 0. A radius
  cannot truly be negative, so this case is purely numerical -- but it is also
  exactly condition (18) read literally.
- The threshold columns mean: if rho(P_3^k) <= threshold, then V_3^k = 0.

Recall that the tile's radii vary with b, so choosing an absolute threshold is
complicated.


3. Comparison of Table 1 and Table 2

In Table 1, the polytopes that survive the Lemma filter are considered to have
non-zero volume, and are thus kept in (19). In Table 2, these surviving
polytopes are tested for zero volume by Chebyshev, and kept in (19) or not
depending on the verdict. So in Table 2 every polytope receives a zero or a
non-zero volume, one way or the other.

Comparing the two is the cleanest test of (18) I can make for now:

--------------------------------------------------------------------
                          Table 1                 Table 2
  b   #tiles (case B)   lemma tol=1e-8    lemma tol=1e-8 + rho<=0
--------------------------------------------------------------------
  6      206  (25)          0.8136               0.8142
 10      158  (10)          0.7735               0.7735
 16      159  (20)          0.7729               0.7729
--------------------------------------------------------------------

Both columns are computed over all the tiles; the number in brackets is how
many of them needed a Chebyshev radius, the others being settled by the Lemma.
Computing the radius of every sub-polytope of those tiles changes gamma by
less than 0.001, and by nothing at all to four decimals at b = 10 and b = 16.
So condition (18), applied as it is defined -- rho = 0 means zero volume --
removes essentially nothing.


4. Thresholds in Table 2

You suggested implementing the zero condition of (18) with eps = 5e-5, or even
1e-4. In Table 2, the trend only turns increasing at rho <= 5e-4. I tried to
measure the numerical noise of rho rather than assume it, and found two
floors.

(i) The smallest obtained radii are negative and equal -2.4e-8. As a negative
radius is meaningless, this means that the solver cannot compute rho at a
precision below about 2e-8.

(ii) By rebuilding the same polytopes in float64, the largest displacement of
a face at b=16 is 1.08e-5 (quite large).

A cut at rho <= 5e-4 is four orders above case (i) and 1.5 above case (ii), so
I cannot keep it as a good threshold.

(iii) I also looked for a gap to cut in, by histogramming log10(rho) for the
sub-polytope of the true class c -- the one forming the numerator of (19) --
and for those of the other classes, which only enter the denominator. The hope
behind (18) is that the latter are flat slivers while the former is a
full-dimensional body. At b=10 the two distributions have exactly the same
shape, so no threshold on rho can separate them there. At b=6 they do differ,
but in the wrong direction: 11 of the 27 correct sub-polytopes have rho <= 0,
against only 5 of the 50 incorrect ones -- applied literally, (18) would
delete the numerator of (19) more often than its denominator. Only at b=16 do
the two separate the way we hoped, by about two orders of magnitude.


5. Anomalies

The ratios rho(P3^k)/rho(P2) show it most clearly, since they are computed
within each tile. At b = 6 and b = 10 their median over the incorrect classes
is 1.00, whereas the P3^k partition P2: at most one of them can contain P2's
largest inscribed ball, and all the others should be strictly smaller. Only
b=16 behaves as the partition requires (median 0.027).

Separately, rho(P2) itself comes back <= 0 on 16 of the 206 tiles at b=6,
against 1 of 158 at b=10 and 1 of 159 at b=16 -- while incorrect
sub-polytopes in the same population sit around 1e-3.5, i.e. thicker than the
polytope that contains them.

Either the radii are unreliable at those bit widths, or there are precision
issues that I cannot control well, or I am missing some geometric argument. Do
you have an idea?


My tentative conclusion is that the difficulty is not where we put the
threshold but what the mean width can see: at n = 784, a difference of 1% in
width is a factor 2600 in volume, and a flat sliver keeps a nearly maximal
width. Do you think a zero / non-zero gate on the volume can repair (19) in
principle? Or should we rather look for a different size measure?

All the best,

Jérémie
