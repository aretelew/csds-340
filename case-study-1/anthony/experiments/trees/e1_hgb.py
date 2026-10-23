"""Exp 1: sklearn HistGradientBoostingClassifier."""
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

from common import imp, make_pipeline, run, to_nan


def hgb(**kw):
    base = dict(random_state=0)
    base.update(kw)
    return make_pipeline(to_nan(), HGB(**base))


slow = dict(learning_rate=0.03, max_iter=500, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=1.0)

configs = [
    ("HGB default (lr.1 it100 L31 msl20)", hgb()),
    ("HGB lr.05 it300 L15 msl20 l2=1", hgb(learning_rate=0.05, max_iter=300,
        max_leaf_nodes=15, l2_regularization=1.0)),
    ("HGB lr.03 it500 L15 msl20 l2=1", hgb(**slow)),
    ("HGB lr.03 it500 L7 msl10 l2=1", hgb(**{**slow, "max_leaf_nodes": 7,
                                              "min_samples_leaf": 10})),
    ("HGB lr.03 it500 L15 msl40 l2=1", hgb(**{**slow, "min_samples_leaf": 40})),
    ("HGB lr.03 it500 L15 msl20 l2=1 mf.5", hgb(**slow, max_features=0.5)),
    ("HGB lr.03 it2000 L15 earlystop", hgb(**{**slow, "max_iter": 2000},
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=50,
        scoring="roc_auc")),
    ("HGB lr.03 it500 L15 median-impute", make_pipeline(
        imp("median"), HGB(random_state=0, **slow))),
]

if __name__ == "__main__":
    run(configs, "HGB")
