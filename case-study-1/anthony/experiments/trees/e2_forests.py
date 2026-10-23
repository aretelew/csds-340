"""Exp 2: RandomForest / ExtraTrees. Round A: missing handling at fixed
(1000 trees, sqrt, msl=1). Round B (e2b) tunes max_features / msl / n_estimators."""
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.ensemble import RandomForestClassifier as RF

from common import imp, make_pipeline, run, to_nan

MISS = {
    "NaN": to_nan,
    "mean": lambda: imp("mean"),
    "median": lambda: imp("median"),
    "zero": lambda: imp("constant", 0.0),
}


def forest(cls, miss, n=1000, mf="sqrt", msl=1):
    return make_pipeline(MISS[miss](), cls(n_estimators=n, max_features=mf,
                                           min_samples_leaf=msl, random_state=0,
                                           n_jobs=1))


configs = []
for cname, cls in [("RF", RF), ("ET", ET)]:
    for miss in MISS:
        configs.append((f"{cname} 1000 sqrt msl1 {miss}", forest(cls, miss)))

if __name__ == "__main__":
    run(configs, "forest")
