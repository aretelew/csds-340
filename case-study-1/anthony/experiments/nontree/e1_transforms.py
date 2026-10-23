"""Exp 1: feature-transform / imputation screen with two fixed probe models
(L2 logistic regression C=1, RBF SVC C=1 gamma='scale'), standardized after FeatX."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ntcommon import FeatX, run

T = {
    "id mean": dict(tf="id"),
    "log10": dict(tf="log", c=10),
    "log100": dict(tf="log", c=100),
    "log1000": dict(tf="log", c=1000),
    "sqrt": dict(tf="sqrt"),
    "pres": dict(tf="pres"),
    "log100+pres": dict(tf="log", c=100, presence=True),
    "quantile": dict(tf="quant"),
    "log100+agg": dict(tf="log", c=100, agg=True),
    "log100+pres+agg": dict(tf="log", c=100, presence=True, agg=True),
    "log100 median(tf)": dict(tf="log", c=100, impute="median"),
    "log100 zero": dict(tf="log", c=100, impute="zero"),
    "log100 mean(raw)": dict(tf="log", c=100, space="raw"),
    "log100 median(raw)": dict(tf="log", c=100, impute="median", space="raw"),
    "log100+ind": dict(tf="log", c=100, ind=True),
}

for tname, kw in T.items():
    run(make_pipeline(FeatX(**kw), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        f"LR C1 | {tname}", "transform-screen-LR")
for tname, kw in T.items():
    run(make_pipeline(FeatX(**kw), StandardScaler(), SVC(C=1.0, gamma="scale")),
        f"SVC C1 | {tname}", "transform-screen-SVC")
