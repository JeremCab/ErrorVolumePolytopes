Draft mail to Jiri — the neighbourhood diagnosis. 20 Sep 2026.
Written entirely in the paper's own notation (Xi_x, Xi-bar_x, alpha_T, gamma_T,
(18), (19)) — MAIL_25 showed that new vocabulary is what made the last one
unreadable. No "tiles", no "filters".

---

Hi Jiri,

I hope you are doing well.

Since my last email I found a way to measure the volume RATIO directly, without
the mean width and without any linear program. It changes the picture, and I
think it gives us a clear diagnosis. Good news first: our definition is fine.


1. Measuring vol(Xi~_x^c) / vol(Xi-bar_x) directly

The quantity (19) approximates is a ratio of volumes of nested convex bodies.
Computing an absolute volume is #P-hard, but a RATIO is not: it is enough to
sample uniformly and count.

So, for a data point x of class c:

  - sample M points uniformly inside Xi-bar_x by Hit-and-Run;
  - evaluate N~^b at each of them;
  - the fraction predicting c estimates vol(correct part) / vol(Xi-bar_x).

No mean width, no Chebyshev radius, no zero-volume condition, no tolerance. The
error is binomial, ~1/sqrt(M), and does NOT grow with the dimension. One walk
serves every b, since Xi-bar_x does not depend on N~.

Two practical notes. The walk cannot start at x itself: normalised images have a
few hundred pixels at exactly +-1, so x lies on hundreds of faces of the input
box and every chord through it has length zero. I start at the Chebyshev centre
of Xi-bar_x instead, which costs one LP. And I verified the sampler on bodies
whose answer is known in closed form, and by rerunning with 4x the thinning: the
averages move by less than 0.001.


2. The result: our gamma is right, and it equals alpha_T

74 random data points per network, 800 samples each, all six bitwidths.
Recall that T contains only points correctly classified by both full-precision
networks, so alpha_T(N) = 1 by construction and every error below is due to
quantisation alone. On these 74 points the quantised MLP happens to make no
pointwise error at all, at any b, which is why its alpha_T row is 1.0000
throughout.

                                   b=4      b=6      b=8     b=10     b=12     b=16
  CNN   vol ratio in Xi-bar_x    0.917    1.000    0.996    1.000    1.000    1.000
        alpha_T (same points)    0.919    1.000    1.000    1.000    1.000    1.000

  MLP   vol ratio in Xi-bar_x   0.9934   1.0000   1.0000   1.0000   1.0000   1.0000
        alpha_T (same points)   1.0000   1.0000   1.0000   1.0000   1.0000   1.0000

Standard errors are 0.030 for the CNN at b=4, 0.004 for the MLP at b=4, and at
most 0.003 everywhere else -- so the 0.996 of the CNN at b=8 is compatible with
1.000 and should not be read as an anomaly.

Two things to read here.

(i) The quantity (19) aims at INCREASES with b, as the theory predicts. Our
definition is sound; it was the mean-width estimate without (18) that was not.
So your correction (18) is not cosmetic, it is essential.

(ii) But the measured value is essentially alpha_T. For the MLP it is strictly
below (0.9934 against 1.0000 -- all 74 points are correctly classified, yet
three of their neighbourhoods are not, which is exactly our claim). For the CNN
it is not below at all.


3. Why Xi-bar_x reveals so much less for the CNN than for the MLP

Xi-bar_x has a mean width of about 1.5, so it is not a small set. But it is very
ELONGATED and extremely THIN, and the two networks differ enormously in how thin.
Drawing uniformly in a box of half-width r, the share of samples falling inside
Xi-bar_x -- same activation pattern under N, and N still correct -- is:

    r (per pixel)     0.001   0.003    0.01    0.03     0.1
    MLP               0.956   0.867   0.616   0.247   0.023
    CNN               0.526   0.000   0.000       0       0

The CNN's Xi-bar_x does not even contain a box of half-width 0.003, while the
MLP's still holds a quarter of a box of half-width 0.03 -- a factor of ten in
radius. The Chebyshev radii say the same: about 1.9e-3 for the CNN against
3.2e-1 for the MLP.

One methodological point, because it changes the CNN figures a lot. The box is
centred at the Chebyshev centre of Xi-bar_x, not at x. For the CNN, x lies on
the boundary of its own polytope -- several hundred constraints are exactly
tight at x, on every one of the 72 points I checked, all of them coming from the
max-pooling layers -- so a box centred at x leaves Xi-bar_x immediately whatever
the size of the polytope, and one would be measuring the position of x rather
than the polytope. Centring at x rather than at the centre changes the CNN
figure at r = 0.001 from 0.53 to 0.09. For the MLP the question does not arise:
x is comfortably interior (its largest constraint value is -3.7e-3).


4. Measuring the same quantity on larger neighbourhoods

Same measurement as in section 2, with one change: instead of sampling inside
Xi-bar_x, I sample uniformly in a box of half-width r around x, KEEP the samples
N itself still classifies as c, and count the share N~^b also calls c. Each entry
below is

    vol({ y in box(x,r) : N(y) = c and N~^b(y) = c })
    -----------------------------------------------------
           vol({ y in box(x,r) : N(y) = c })

i.e. the quantity of section 2 over a different neighbourhood. At r -> 0 the box
collapses onto x and the average is exactly alpha_T. Here x is comfortably
interior to { y : N(y) = c }, so centring the box at x raises no difficulty --
unlike in section 3, where the object measured was the linearity region itself.

MLP, 74 points; alpha_T = 1.0000 at every b, so each entry reads directly
against 1:

    r    |      b=4      b=6      b=8     b=10     b=12     b=16
  (ACC)  |   1.0000   1.0000   1.0000   1.0000   1.0000   1.0000
   0.01  |   1.0000   1.0000   1.0000   1.0000   1.0000   1.0000
   0.03  |   1.0000   0.9999   1.0000   1.0000   1.0000   1.0000
   0.10  |   0.9988   0.9979   0.9999   1.0000   1.0000   1.0000
   0.30  |   0.9923   0.9981   1.0000   0.9999   1.0000   1.0000
   1.00  |   0.9815   0.9945   0.9993   0.9996   0.9999   1.0000

CNN, the same 74 points; alpha_T = 0.9189 at b=4 and 1.0000 elsewhere:

    r    |      b=4      b=6      b=8     b=10     b=12     b=16
  (ACC)  |   0.9189   1.0000   1.0000   1.0000   1.0000   1.0000
   0.01  |   0.9174   1.0000   1.0000   1.0000   1.0000   1.0000
   0.03  |   0.9163   1.0000   0.9954   1.0000   1.0000   1.0000
   0.10  |   0.9222   0.9949   0.9922   1.0000   1.0000   1.0000
   0.30  |   0.9231   0.9852   0.9969   0.9997   0.9997   1.0000
   1.00  |   0.6401   0.9380   0.9893   0.9952   0.9992   0.9999

Read ACROSS a row, at a fixed neighbourhood. At r = 1 the MLP gives 0.9815,
0.9945, 0.9993, 0.9996, 0.9999, 1.0000 and the CNN 0.6401, 0.9380, 0.9893,
0.9952, 0.9992, 0.9999. Monotone in b, on both networks, on all six bitwidths:
at a fixed neighbourhood, the coarser the quantisation the more N~ degrades.
That is exactly the statement of the paper, and it involves N~ alone -- N merely
delimits the domain, identically for the six columns.

Now, where does Xi-bar_x sit among these neighbourhoods? Not by inclusion --
Xi-bar_x is a long thin sliver, not a box, and it reaches far in directions no
small box visits. But we can place it by what it REVEALS. Comparing the gap to
alpha_T at b=4:

                      gap to alpha_T     equivalent box
    MLP   Xi-bar_x          -0.0066           r ~ 0.25
          box r = 1          -0.0185
    CNN   Xi-bar_x          -0.0019           r ~ 0.02
          box r = 1          -0.2789

This is the point I would draw your attention to. For the MLP, Xi-bar_x already
does most of the work: enlarging the neighbourhood all the way to r = 1 gains a
factor of three. For the CNN it gains a factor of a hundred and fifty. In other
words the MLP's neighbourhood is adequate and the CNN's is not, by two orders of
magnitude -- which is consistent with section 3, where the CNN's Xi-bar_x turned
out to be some ten times thinner.

One last thing, which explains why the CNN looked inert at small r. Splitting
its 74 points at b=4 into the 68 where N~ classifies x correctly and the 6 where
it does not:

    r        x correct    x wrong    average
    0.010       0.9983     0.0000     0.9174
    0.030       0.9947     0.0283     0.9163
    0.100       0.9889     0.1659     0.9222
    0.300       0.9654     0.4513     0.9231
    0.500       0.9191     0.5510     0.8892
    1.000       0.6528     0.4953     0.6401

On correctly classified points the neighbourhood degrades monotonically -- our
claim, cleanly, and by a factor of two hundred between r = 0.01 and r = 1. On
misclassified points it IMPROVES, because N~ recovers as one moves away from x.
The two cancel almost exactly out to r = 0.3, which is why the CNN average sat
on alpha_T. The MLP has alpha_T = 1 exactly, so it feels only the first half,
which is why its gap was clean there.


5. What I would propose

We define Xi_x by requiring N AND N~ to be linear, plus N correct, and we
already relax the linearity of N~ to obtain Xi-bar_x. I suggest relaxing it for
N as well, and taking as the neighbourhood

    { y : N(y) = c }  intersected with a box of half-width r around x.

This is a UNION of adjacent linearity regions, so we stay inside the same
framework -- but it is about a hundred times larger in radius, and it is exactly
where the effect lives. It needs no mean width, no Chebyshev radius and no
tolerance, and its error does not grow with the dimension.

It would also give us a family of curves gamma(r) ordered in b, rather than a
single number: a richer figure than Fig. 2, and one that shows the scale at
which approximation starts to hurt.

Before I go further: does relaxing the linearity of N look acceptable to you, or
do you see a reason to keep the neighbourhood inside a single linearity region?

All the best,
Jérémie
