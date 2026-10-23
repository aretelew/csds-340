"""Exp 3b: follow-up on spline GAM (stronger L2, fewer knots) and poly2 at very strong L2."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

from ntcommon import FeatX, run

LOG = dict(tf="log", c=1e4)
for nk, deg in [(4, 3), (3, 2)]:
    for C in [0.003, 0.01]:
        run(make_pipeline(FeatX(**LOG),
                          SplineTransformer(n_knots=nk, degree=deg, knots="uniform", extrapolation="linear"),
                          StandardScaler(), LogisticRegression(C=C, max_iter=5000)),
            f"LR spline k{nk}d{deg} C{C} | log1e4", "LR-spline-GAM")
run(make_pipeline(FeatX(**LOG), StandardScaler(), PolynomialFeatures(2, include_bias=False),
                  StandardScaler(), LogisticRegression(C=0.0003, max_iter=5000)),
    "LR poly2 L2 C0.0003 | log1e4", "LR-poly2")
