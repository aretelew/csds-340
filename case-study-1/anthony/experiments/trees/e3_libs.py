"""Exp 3: XGBoost / LightGBM / CatBoost, conservative settings, native NaN handling.
10 configs total (3 XGB, 3 LGBM, 4 CatBoost). Single-threaded per model."""
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from common import make_pipeline, run, to_nan


def xgb(**kw):
    p = dict(n_estimators=600, learning_rate=0.03, max_depth=3, subsample=0.8,
             colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0,
             tree_method="hist", random_state=0, n_jobs=1, verbosity=0)
    p.update(kw)
    return make_pipeline(to_nan(), XGBClassifier(**p))


def lgb(**kw):
    p = dict(n_estimators=600, learning_rate=0.03, num_leaves=7, min_child_samples=10,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
             random_state=0, n_jobs=1, verbose=-1)
    p.update(kw)
    return make_pipeline(to_nan(), LGBMClassifier(**p))


def cat(**kw):
    p = dict(iterations=1000, learning_rate=0.03, depth=4, l2_leaf_reg=3,
             random_seed=0, thread_count=1, verbose=0, allow_writing_files=False)
    p.update(kw)
    return make_pipeline(to_nan(), CatBoostClassifier(**p))


configs = [
    ("XGB d3 lr.03 n600 ss.8 cs.8", xgb()),
    ("XGB d2 lr.03 n800 ss.8 cs.8", xgb(max_depth=2, n_estimators=800)),
    ("XGB d4 lr.02 n1000 ss.7 cs.5 mcw2", xgb(max_depth=4, learning_rate=0.02,
        n_estimators=1000, subsample=0.7, colsample_bytree=0.5, min_child_weight=2)),
    ("LGBM L7 lr.03 n600 mcs10 ss.8 cs.5", lgb()),
    ("LGBM L4 lr.03 n800 mcs10 ss.8 cs.5", lgb(num_leaves=4, n_estimators=800)),
    ("LGBM L15 lr.03 n600 mcs20 ss.8 cs.8", lgb(num_leaves=15, min_child_samples=20,
        colsample_bytree=0.8)),
    ("CAT d4 lr.03 n1000", cat()),
    ("CAT d6 lr.03 n1000", cat(depth=6)),
    ("CAT d4 lr.03 n1000 rsm.5", cat(rsm=0.5)),
    ("CAT d4 lr.03 n1000 nan=Max", cat(nan_mode="Max")),
]

if __name__ == "__main__":
    run(configs, "boostlib")
