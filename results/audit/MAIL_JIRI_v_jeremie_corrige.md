Version de Jérémie, corrigée. 24 sep 2026.
Corrections : justification logique du 2(i) ; Table 1 recalculee pour les DEUX
reseaux depuis le centre de Chebyshev ; renvois de la Table 3 ; terminologie T* ;
etiquette alpha_T ; coquilles.

---

Hi Jiri,

I hope you are doing well.

I worked a bit on the experiments, consulted AI, etc. Before running further
expensive simulations, here are some observations.


1. Measuring vol(P3^c) / vol(P1) directly

Note that the ratio (19) can be approximated by the following simple Monte Carlo
process (called gamma_2):

for each data point x_0 of class c:
  - sample M points uniformly inside P1 by Hit-and-Run;
  - for each b, evaluate N~^b at all points;
  - for each b, count the points correctly and incorrectly classified by N~^b.

Then, for each b, the number of correctly classified points divided by the total
number of points is gamma_2(b).

This is an estimate of vol({ y in P1 : N~^b(y) = c }) / vol(P1), obtained without
mean width, Chebyshev radius, zero-volume condition or tolerance. The error is
binomial, ~1/sqrt(M), and does not grow with the dimension.


2. Results: our GACC (19) is correct, but almost equal to the classical ACC

I ran a small experiment for both MLP and CNN on 74 original data points of T,
all correctly classified by the original MLP and CNN. For each point I drew 800
uniform samples inside P1 -- these are Monte Carlo samples used for the
integration, not the augmented dataset T* of the paper.

------------------------------------------------------------
CNN             b=4     b=6     b=8     b=10    b=12    b=16
------------------------------------------------------------
gamma_2         0.917   1.000   0.996   1.000   1.000   1.000
alpha_T(N~^b)   0.919   1.000   1.000   1.000   1.000   1.000
------------------------------------------------------------
MLP             b=4     b=6     b=8     b=10    b=12    b=16
------------------------------------------------------------
gamma_2         0.9934  1.0000  1.0000  1.0000  1.0000  1.0000
alpha_T(N~^b)   1.0000  1.0000  1.0000  1.0000  1.0000  1.0000
------------------------------------------------------------

(Standard errors: 0.030 for the CNN at b=4, 0.004 for the MLP at b=4, at most
0.003 elsewhere -- so the CNN's 0.996 at b=8 is compatible with 1.000.)

(i) Note that gamma_2 increases with b, as the theory predicts. Since gamma_2
estimates the very quantity that (19) is designed to approximate -- the mean
width being introduced in the paper as a surrogate for the volume -- this shows
that our definition aims at the right thing: the volume ratio does behave as the
theory says. It does not by itself validate the mean-width estimate of it, which
is a separate question.

(ii) Note also that gamma_2 is essentially equal to the classical ACC alpha_T.


3. One remark before the next section

I took a box of half-width r around the Chebyshev centre of P1, sampled points
uniformly inside this box, and checked the percentage falling inside P1. As r
increases the box starts to exceed P1 and the percentage drops.

Table 1
---------------------------------------------
r      0.001   0.003   0.01   0.03   0.1
---------------------------------------------
       %-age of samples of box(r) inside P1
---------------------------------------------
MLP    100.0   100.0   100.0  100.0  85.3
CNN     52.6     0.0     0.0    0.0   0.0
---------------------------------------------

So the MLP's P1 entirely contains a box of half-width 0.03, whereas the CNN's
does not even contain one of half-width 0.003 -- a factor of at least thirty in
radius. This is consistent with the Chebyshev radii, about 1.9e-3 for the CNN
against 3.2e-1 for the MLP.

The centre of the box matters here, and by a lot. Centring it at x instead of at
the Chebyshev centre gives 96.3 / 89.1 / 65.4 / 26.5 / 1.2 for the MLP and
9.4 / 0.3 / 0 / 0 / 0 for the CNN -- because x lies on the boundary of its own
P1 (for the CNN I find several hundred constraints exactly tight at x, on every
point I checked, all coming from the max-pooling layers). Centred at x one would
be measuring the position of x rather than the size of P1.


4. Some results

Here I sample uniformly in a box of half-width r around x -- instead of inside
P1 -- and compute gamma_2 as in section 2. This gamma_2 is thus an estimate of

    vol({ y in box(x,r) : N(y) = c and N~^b(y) = c })
    -----------------------------------------------------
           vol({ y in box(x,r) : N(y) = c })

As r -> 0 the box collapses onto x and the average is exactly alpha_T. Note that
x is comfortably interior to { y : N(y) = c }, so centring the box at x raises
no difficulty here, unlike in section 3 where the object measured was P1 itself.


Table 2a
-----------------------------------------------------------------
MLP, 74 original points correctly classified by N, each with 800
uniform samples per radius.
-----------------------------------------------------------------
    r    |     b=4      b=6      b=8     b=10     b=12     b=16
-----------------------------------------------------------------
alpha_T  |  1.0000   1.0000   1.0000   1.0000   1.0000   1.0000
   0.01  |  1.0000   1.0000   1.0000   1.0000   1.0000   1.0000
   0.03  |  1.0000   0.9999   1.0000   1.0000   1.0000   1.0000
   0.10  |  0.9988   0.9979   0.9999   1.0000   1.0000   1.0000
   0.30  |  0.9923   0.9981   1.0000   0.9999   1.0000   1.0000
   1.00  |  0.9815   0.9945   0.9993   0.9996   0.9999   1.0000
-----------------------------------------------------------------


Table 2b
-----------------------------------------------------------------
CNN, the same protocol on 74 original points.
-----------------------------------------------------------------
    r    |     b=4      b=6      b=8     b=10     b=12     b=16
-----------------------------------------------------------------
alpha_T  |  0.9189   1.0000   1.0000   1.0000   1.0000   1.0000
   0.01  |  0.9174   1.0000   1.0000   1.0000   1.0000   1.0000
   0.03  |  0.9163   1.0000   0.9954   1.0000   1.0000   1.0000
   0.10  |  0.9222   0.9949   0.9922   1.0000   1.0000   1.0000
   0.30  |  0.9231   0.9852   0.9969   0.9997   0.9997   1.0000
   1.00  |  0.6401   0.9380   0.9893   0.9952   0.9992   0.9999
-----------------------------------------------------------------

(i) Good news: for a fixed neighbourhood half-width r, the coarser the
quantization the more N~^b degrades. This is exactly the statement of the paper.

(ii) Zooming in at b=4

Now, where does P1 sit among these neighbourhoods? Not by inclusion -- P1 is a
long thin sliver, not a box, and it reaches far in directions no small box
visits. But we can place it by what it REVEALS. Comparing the gap between
gamma_2 and alpha_T at b=4:

Table 3
------------------------------------------------------------
                  gap gamma_2 - alpha_T      equivalent box
------------------------------------------------------------
MLP
------------------------------------------------------------
P1                  -0.0066 (section 2)          r ~ 0.25
box r = 1           -0.0185 (Table 2a)
------------------------------------------------------------
CNN
------------------------------------------------------------
P1                  -0.0019 (section 2)          r ~ 0.02
box r = 1           -0.2789 (Table 2b)
------------------------------------------------------------

(i) For the MLP, going from P1 to a larger neighbourhood (box of r = 1) would
increase the difference between ACC and GACC by a factor of 3.

(ii) For the CNN, the same change would increase it by a factor of 150. So for
the CNN, P1 is too small to reveal what our GACC adds over the ACC.


I'll read your last long email and answer tomorrow evening.

Concerning the visit, yes, November is fine.

All the best,

Jérémie
