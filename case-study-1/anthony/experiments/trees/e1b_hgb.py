"""Exp 1b: second (last) HGB round, around the two winners of e1 (shallow trees,
feature subsampling). 6 configs."""
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

from common import imp, make_pipeline, run, to_nan


def hgb(**kw):
    return make_pipeline(to_nan(), HGB(random_state=0, **kw))


b = dict(learning_rate=0.03, max_iter=500, max_leaf_nodes=7, min_samples_leaf=10,
         l2_regularization=1.0, max_features=0.5)

configs = [
    ("HGB lr.03 it500 L7 msl10 l2=1 mf.5", hgb(**b)),
    ("HGB lr.03 it500 L7 msl20 l2=1 mf.5", hgb(**{**b, "min_samples_leaf": 20})),
    ("HGB lr.03 it800 L4 msl10 l2=1 mf.5", hgb(**{**b, "max_leaf_nodes": 4,
                                                   "max_iter": 800})),
    ("HGB lr.015 it1000 L7 msl10 l2=1 mf.5", hgb(**{**b, "learning_rate": 0.015,
                                                     "max_iter": 1000})),
    ("HGB L7 mf.5 mean-impute", make_pipeline(imp("mean"), HGB(random_state=0, **b))),
    ("HGB L7 mf.5 zero-impute", make_pipeline(imp("constant", 0.0),
                                              HGB(random_state=0, **b))),
]

if __name__ == "__main__":
    run(configs, "HGB")
