"""Exp 4: cheap engineered row features (RowStats) on top of the best tree models.
raw  = n_missing, n_nonzero, sum, max (over observed entries)
norm = n_missing, frac_nonzero_of_observed, mean_of_observed, max
ind  = per-feature missing indicators (SimpleImputer add_indicator) for imputed ET."""
import sys

from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.ensemble import RandomForestClassifier as RF

from common import RowStats, imp, make_pipeline, run, to_nan

HGB_BEST = dict(learning_rate=0.03, max_iter=800, max_leaf_nodes=4, min_samples_leaf=10,
                l2_regularization=1.0, max_features=0.5, random_state=0)


def hgb(stats=None):
    steps = ([RowStats(stats)] if stats else []) + [to_nan(), HGB(**HGB_BEST)]
    return make_pipeline(*steps)


def et(stats=None, miss="NaN", mf="sqrt", msl=1, ind=False):
    pre = to_nan() if miss == "NaN" else imp("mean", add_indicator=ind)
    steps = ([RowStats(stats)] if stats else []) + [
        pre, ET(n_estimators=1000, max_features=mf, min_samples_leaf=msl,
                random_state=0, n_jobs=1)]
    return make_pipeline(*steps)


def rf(stats=None, mf="sqrt", msl=1):
    steps = ([RowStats(stats)] if stats else []) + [
        to_nan(), RF(n_estimators=1000, max_features=mf, min_samples_leaf=msl,
                     random_state=0, n_jobs=1)]
    return make_pipeline(*steps)


ET_KW = dict(mf="sqrt", msl=1)   # updated after e2b
RF_KW = dict(mf="sqrt", msl=1)

configs = [
    ("HGB L4 mf.5 +raw stats", hgb("raw")),
    ("HGB L4 mf.5 +norm stats", hgb("norm")),
    ("ET NaN +raw stats", et("raw", **ET_KW)),
    ("ET NaN +norm stats", et("norm", **ET_KW)),
    ("ET mean +indicators", et(None, miss="mean", ind=True, **ET_KW)),
    ("ET mean +norm stats", et("norm", miss="mean", **ET_KW)),
    ("RF NaN +norm stats", rf("norm", **RF_KW)),
]

if __name__ == "__main__":
    run(configs, "features")
