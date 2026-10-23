"""Exp 2: follow-up transform screen (larger log constants, combos, iterative imputation)."""
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
import numpy as np

from ntcommon import FeatX, run


def _lognan(Z, c=1000.0):
    Z = np.array(Z, dtype=float, copy=True)
    Z[Z == -1] = np.nan
    return np.log1p(c * Z)


T = {
    "log3000": dict(tf="log", c=3000),
    "log1e4": dict(tf="log", c=1e4),
    "log1e5": dict(tf="log", c=1e5),
    "log1000+pres": dict(tf="log", c=1000, presence=True),
    "log1000+agg": dict(tf="log", c=1000, agg=True),
    "sqrt+pres": dict(tf="sqrt", presence=True),
    "quantile+agg": dict(tf="quant", agg=True),
    "log1000 mean(raw)": dict(tf="log", c=1000, space="raw"),
}
for tname, kw in T.items():
    run(make_pipeline(FeatX(**kw), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        f"LR C1 | {tname}", "transform-screen-LR")
    run(make_pipeline(FeatX(**kw), StandardScaler(), SVC(C=1.0, gamma="scale")),
        f"SVC C1 | {tname}", "transform-screen-SVC")

it = lambda: make_pipeline(FunctionTransformer(_lognan), IterativeImputer(max_iter=10, random_state=0),
                           StandardScaler())
run(make_pipeline(it(), LogisticRegression(C=1.0, max_iter=2000)), "LR C1 | log1000 iterative(tf)", "transform-screen-LR")
run(make_pipeline(it(), SVC(C=1.0)), "SVC C1 | log1000 iterative(tf)", "transform-screen-SVC")
