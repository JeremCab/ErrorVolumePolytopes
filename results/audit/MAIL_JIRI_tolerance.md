Message a Jiri — la calibration de la tolerance + les questions en attente.
27 sep 2026. Notation du papier. Les questions de fond (role des polytopes,
nouveau voisinage) sont renvoyees a la visite, comme il le suggere lui-meme.

---

Hi Jiri,

Several things at once, the first of which concerns Fig. 2 (right) and should
probably reach you before the final version.


1. The tolerance in (18) can now be calibrated instead of chosen

The four points of Fig. 2 (right) come from the row tol = 1e-06 of the tolerance
scan I sent you, "tol" being the relative gap at which we declare
d(Xi~_x^k) = d(Xi_x) and apply (18) through the lemma. That scan gives an
increasing curve at 1e-06, a flat one at 1e-08 and a decreasing one at 1e-10, so
the choice decides the shape -- and until now nothing justified one value over
another, which is precisely the objection you raised against gamma'.

The Monte Carlo of my previous email settles it. The tiles Xi_y of T_x partition
Xi-bar_x, so the aggregate of their volume ratios IS the volume ratio over
Xi-bar_x. The sampled value is therefore not a proxy for gamma_{T_P}: it is the
quantity gamma_{T_P} estimates. Comparing the two says which tolerance is right.

Sampling Xi-bar_x for the 150 original points of the campaign, 800 samples
each:

                        b=4       b=6      b=10      b=16    mean error
  measured volume    0.9491    0.9986    0.9999    1.0000
  (standard error)   0.0164    0.0009    0.0001    0.0000
  gamma_TP 1e-03     0.9253    1.0000    1.0000    1.0000        0.0063
  gamma_TP 1e-04     0.8651    1.0000    1.0000    0.9712        0.0286
  gamma_TP 1e-05     0.8398    0.9201    0.9947    0.9548        0.0596
  gamma_TP 1e-06     0.8220    0.8677    0.9159    0.9500        0.0980
  gamma_TP 1e-08     0.7995    0.8136    0.7735    0.7729        0.1970
  gamma_TP 1e-10     0.7476    0.6298    0.4591    0.4470        0.4160

The clearest case is b = 16, because neither of the two usual objections applies
there: T_P has 159 tiles for 150 original points, i.e. essentially the data
points themselves, and Xi_x fills Xi-bar_x (V2/V1 = 0.998), so neither the
partial coverage of Xi-bar_x nor the mean-width weighting is in play.

  measured volume    1.0000   (standard error 0.0000)
  gamma_TP at 1e-03  1.0000
  gamma_TP at 1e-06  0.9500   <- the point plotted in Fig. 2 (right)

At b = 16 the deficit of 0.05 is therefore due to the tolerance alone.

The consequence for the figure is not comfortable: at the calibrated tolerance
the curve reads 0.9253 / 1.0000 / 1.0000 / 1.0000, almost flat, so the visible
gap between gamma_{T_P} and alpha_T comes from the choice of 1e-06.


1b. But our claim does hold inside Xi-bar_x -- it is simply small

Extending the same measurement to b = 2 and 3, on all 150 points:

                   b=2      b=3      b=4      b=6     b=10     b=16
  volume        0.1000   0.5522   0.9491   0.9986   0.9999   1.0000
  alpha_T       0.1000   0.5667   0.9533   1.0000   1.0000   1.0000
  difference   +0.0000  -0.0145  -0.0042  -0.0014  -0.0001  +0.0000

The difference is negative wherever there is anything to measure, and it is
monotone in b: -0.0145, -0.0042, -0.0014, -0.0001, 0.0000 from b = 3 to b = 16.
That is exactly our claim -- generalized accuracy is stricter than pointwise
accuracy, and the more so the coarser the quantization -- verified INSIDE
Xi-bar_x, which I had thought impossible after my previous email. The effect is
real; it is just an order of magnitude smaller than Fig. 2 suggests, at most
1.45 points rather than 15.

The curve also locates the cliff: 0.9491 at b = 4, 0.5522 at b = 3, 0.1000 at
b = 2. That is where quantization actually destroys the network.

One correction to my previous email, since I had it the wrong way round. I wrote
that the neighbourhood came out BETTER classified than the data points (0.9678
against 0.9533). That compared 135 sampled points with alpha_T over 150, and the
15 missing ones happened to be the hardest cases -- they had been computed
earlier and were skipped by the job array. On the same 150 points the difference
is -0.0042, i.e. in the direction we claim.

One caveat remains: the coverage of Xi-bar_x by the tiles is partial at b = 4
(five representatives kept out of the five hundred found, and chosen by maximal
distance, hence atypical), so part of the b = 4 discrepancy between gamma_TP and
the volume may be coverage rather than tolerance. It does not affect b = 16.

I realise this is awkward with the submission going out, but I think the
methodological gain is real: the free parameter you objected to in gamma' now
has an answer, and it comes from a measurement rather than from an argument.


2. Your question about the 1,000 data points

They were drawn from the 60,000 TRAINING points, not from the 10,000 validation
ones and not from a mixture of the two. The 56,334 figure is the intersection of
the two correctly classified subsets: 59,297 for N_2 (98.83% of 60,000) and
56,680 for N_1 (94.47%), which match the reported accuracies exactly.

Since the reviewer asked, it may be worth anticipating the follow-up: T comes
from data the networks have seen, while the paper is about generalization. The
answer is, I think, that generalized accuracy measures the behaviour of the
APPROXIMATED network on neighbourhoods the original network was never trained
on -- but we should probably say so explicitly rather than leave it implicit.


3. b = 2 and 3, as you asked

I added both to the radius experiment, on 74 data points per network (500
samples per radius, boxes of half-width r, keeping only what N still classifies
correctly). The r -> 0 row is alpha_T by construction.

CNN
    r    |    b=2      b=3      b=4      b=6      b=8     b=10     b=12     b=16
 alpha_T | 0.1081   0.6757   0.9189   1.0000   1.0000   1.0000   1.0000   1.0000
   0.01  | 0.1081   0.6740   0.9174   1.0000   1.0000   1.0000   1.0000   1.0000
   0.03  | 0.1081   0.6628   0.9163   1.0000   0.9954   1.0000   1.0000   1.0000
   0.10  | 0.1081   0.6383   0.9222   0.9949   0.9922   1.0000   1.0000   1.0000
   0.30  | 0.1096   0.5396   0.9231   0.9852   0.9969   0.9997   0.9997   1.0000
   0.50  | 0.1081   0.4594   0.8892   0.9725   0.9952   0.9991   0.9997   1.0000
   1.00  | 0.1081   0.3207   0.6401   0.9380   0.9893   0.9952   0.9992   0.9999

MLP
    r    |    b=2      b=3      b=4      b=6      b=8     b=10     b=12     b=16
 alpha_T | 0.1216   0.7162   1.0000   1.0000   1.0000   1.0000   1.0000   1.0000
   0.01  | 0.1216   0.7162   1.0000   1.0000   1.0000   1.0000   1.0000   1.0000
   0.03  | 0.1216   0.7163   1.0000   0.9999   1.0000   1.0000   1.0000   1.0000
   0.10  | 0.1216   0.7151   0.9988   0.9979   0.9999   1.0000   1.0000   1.0000
   0.30  | 0.1216   0.7080   0.9923   0.9981   1.0000   0.9999   1.0000   1.0000
   0.50  | 0.1216   0.7051   0.9905   0.9974   0.9998   0.9999   1.0000   1.0000
   1.00  | 0.1216   0.6932   0.9815   0.9945   0.9993   0.9996   0.9999   1.0000

Gap to alpha_T at r = 1:

            b=2       b=3       b=4       b=6       b=8      b=10      b=12      b=16
  CNN    0.0000   -0.3550   -0.2788   -0.0620   -0.0107   -0.0048   -0.0008   -0.0001
  MLP    0.0000   -0.0230   -0.0185   -0.0055   -0.0007   -0.0004   -0.0001    0.0000

So b = 3 is the most revealing regime we have: for the CNN the gap reaches
-0.355, larger than at b = 4, and the ordering in b continues cleanly downwards.

b = 2, on the other hand, falls outside the usable range, and in an instructive
way. The value does not move at all with r -- 0.1081 at every radius. The
decomposition shows why: of the 74 points, N~^2 classifies 8 correctly, and for
those 8 the neighbourhood is correct at 100% at EVERY radius, while for the
other 66 it is wrong at 100% everywhere. The network no longer depends on its
input at all: it has collapsed the classes. It is not a degraded network but a
destroyed one, so generalized accuracy has nothing to measure there.

(These 74 points are not the 150 original points of the campaign, hence the
small differences in alpha_T with section 1.)


4. The curse of dimensionality -- Vera was right, and it depends where

Two different samplings are involved and they do not behave the same way.

Sampling uniformly in a BOX is i.i.d., so the 1/sqrt(M) binomial error is exact
and genuinely independent of the dimension. That is the case of the experiments
in section 4 of my previous email.

Sampling inside Xi-bar_x is NOT: Hit-and-Run is a Markov chain, consecutive
samples are correlated, and the dimension enters through the mixing time. So
1/sqrt(M) is optimistic there, exactly as Vera said. I tested it by rerunning
with four times the thinning and a different seed: individual per-point values
moved by up to 0.133 against a nominal binomial error of 0.035, so the effective
sample size is well below M. The AGGREGATE moved by less than 0.001, because the
errors cancel across points. So the averages I quote are sound and the per-point
values in the mid-range are not, and I no longer quote them.


5. The comparison in 4(ii) that was unclear

My fault -- the wording hid the point. Xi-bar_x is not placed among the boxes by
INCLUSION: it is a long thin sliver, not a box, and it reaches far in directions
no small box visits. It is placed by what it REVEALS, i.e. by the size of the
gap to alpha_T that it produces. On that scale Xi-bar_x behaves like a box of
half-width about 0.25 for the MLP and about 0.02 for the CNN.


6. The role of the polytopes

This is the right question, and I agree with your reading: once we sample and
evaluate the network, the shortcut weights are no longer needed, and the
Chebyshev centre only matters because Xi-bar_x is defined by a linearity we may
no longer require. What the polytope still buys us is the DEFINITION of the
neighbourhood -- without it we fall back on an arbitrary epsilon-ball, i.e. the
certified-robustness literature we distinguish ourselves from -- and the
size weights of (19), which sampling cannot provide since absolute volumes are
#P-hard.

As you say, this is better discussed in person. I would rather bring it to
Versailles than settle it by email.


7. The visit

<Nov 5-15 / Nov 26-Dec 6 — a completer>

All the best,
Jérémie
