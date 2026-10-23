"""Exp 3: logistic regression family — penalty x C grid, degree-2 polynomial, spline GAM."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

from ntcommon import FeatX, run

LOG = dict(tf="log", c=1e4)
CS = [0.01, 0.03, 0.1, 0.3, 1.0]

for l1r, pen in [(0.0, "L2"), (1.0, "L1"), (0.5, "EN.5")]:
    for C in CS:
        solver = "lbfgs" if l1r == 0 else "saga"
        run(make_pipeline(FeatX(**LOG), StandardScaler(),
                          LogisticRegression(C=C, l1_ratio=l1r, solver=solver, max_iter=5000)),
            f"LR {pen} C{C} | log1e4", "LR")
for C in CS:
    run(make_pipeline(FeatX(tf="quant"), StandardScaler(), LogisticRegression(C=C, max_iter=2000)),
        f"LR L2 C{C} | quantile", "LR")

for C in [0.001, 0.003, 0.01, 0.03]:
    run(make_pipeline(FeatX(**LOG), StandardScaler(), PolynomialFeatures(2, include_bias=False),
                      StandardScaler(), LogisticRegression(C=C, max_iter=5000)),
        f"LR poly2 L2 C{C} | log1e4", "LR-poly2")

for nk in [4, 6]:
    for C in [0.03, 0.1, 0.3, 1.0]:
        run(make_pipeline(FeatX(**LOG),
                          SplineTransformer(n_knots=nk, degree=3, knots="uniform", extrapolation="linear"),
                          StandardScaler(), LogisticRegression(C=C, max_iter=5000)),
            f"LR spline k{nk} C{C} | log1e4", "LR-spline-GAM")
