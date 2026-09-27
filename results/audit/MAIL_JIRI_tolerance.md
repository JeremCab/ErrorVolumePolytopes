Court message a Jiri — la tolerance de (18) est desormais calibrable. 27 sep 2026.
Notation du papier. Volontairement bref : il soumet en ce moment.

---

Hi Jiri,

One more result, and I think you should have it before the final version, since
it concerns the four points of Fig. 2 (right).

Those four values come from the row tol = 1e-06 of the tolerance scan I sent
you, where "tol" is the relative gap at which we declare d(Xi~_x^k) = d(Xi_x)
and apply (18) through the lemma. The scan gives an increasing curve at 1e-06, a
flat one at 1e-08 and a decreasing one at 1e-10, so that choice decides the
shape. Until now we had no way to justify one value over another -- which is the
objection you yourself raised against gamma'.

We can now fix it by measurement rather than by convention.

The tiles Xi_y of T_x partition Xi-bar_x, so the aggregate of their volume
ratios IS the volume ratio over Xi-bar_x. That is exactly what the Monte Carlo
of my previous email measures, with no mean width and no threshold. So the
sampled value is not a proxy for gamma_{T_P}: it is the quantity gamma_{T_P}
estimates. Comparing the two says which tolerance is right.

I sampled P1 for 135 of the 150 original points of the campaign, 800 samples
each:

                        b=4       b=6      b=10      b=16    mean error
  measured volume    0.9678    0.9993    0.9999    1.0000
  (standard error)   0.0133    0.0006    0.0001    0.0000
  gamma_TP 1e-03     0.9253    1.0000    1.0000    1.0000        0.0108
  gamma_TP 1e-04     0.8651    1.0000    1.0000    0.9712        0.0331
  gamma_TP 1e-06     0.8220    0.8677    0.9159    0.9500        0.1029
  gamma_TP 1e-08     0.7995    0.8136    0.7735    0.7729        0.2019

The clearest case is b = 16, because neither of the two usual objections applies
there: T_P has 159 tiles for 150 original points, i.e. essentially the data
points themselves, and Xi_x fills Xi-bar_x (V2/V1 = 0.998), so neither the
partial coverage of Xi-bar_x nor the mean-width weighting is in play.

  measured volume   1.0000   (standard error 0.0000)
  gamma_TP at 1e-03 1.0000
  gamma_TP at 1e-06 0.9500   <- the point plotted in Fig. 2 (right)

At b = 16 the deficit of 0.05 is therefore entirely due to the tolerance.

The consequence is not comfortable: at the calibrated tolerance the curve reads
0.9253 / 1.0000 / 1.0000 / 1.0000, i.e. almost flat, so the visible gap between
gamma_{T_P} and alpha_T in the figure comes from the choice of 1e-06. And on
these 150 points the measured volume at b = 4 is 0.9678 against alpha_T =
0.9533, so the neighbourhood is in fact slightly BETTER classified than the data
points -- the neighbourhoods of the misclassified points being largely correct.

Two honest caveats. The coverage of Xi-bar_x by the tiles is partial at b = 4
(five representatives kept out of the five hundred found, and chosen by maximal
distance, hence atypical), so part of the b = 4 discrepancy may be coverage
rather than tolerance. And 15 of the 150 points are still missing, from LPs that
did not converge. Neither affects b = 16.

I realise this is awkward with the submission going out. But I think the
methodological gain is real: the free parameter you objected to in gamma' now
has an answer, and it comes from a measurement rather than from an argument. We
can say in the paper that the tolerance is calibrated against a direct volume
estimate, which is a much stronger position than choosing it.

Happy to go through this in Versailles.

All the best,
Jérémie
