"""Candidate models for the final comparison: the best configs from the trees/
and nontree/ screens (5-repeat harness_v1), rebuilt here so they can be rerun
with the 10-repeat harness and combined into deployable ensembles.

Every model takes the raw features (missing = -1). `score()` returns a
continuous score (decision_function when available, so boosted-tree
probabilities never saturate into ties).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import SVC

from experiments.harness import to_nan


def score(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


class Feat(BaseEstimator, TransformerMixin):
    """-1 -> NaN, elementwise transform, then fill missing with the training
    mean of the transformed feature. Same as nontree FeatX(tf, c) defaults.
    tf: 'log' = log1p(c*x), 'sqrt' = sqrt(x)."""

    def __init__(self, tf="log", c=1e4):
        self.tf = tf
        self.c = c

    def _f(self, X):
        Z = np.array(X, dtype=float, copy=True)
        Z[Z == -1] = np.nan
        return np.log1p(self.c * Z) if self.tf == "log" else np.sqrt(Z)

    def fit(self, X, y=None):
        self.fill_ = np.nanmean(self._f(X), axis=0)
        return self

    def transform(self, X):
        F = self._f(X)
        return np.where(np.isnan(F), self.fill_, F)


class SeedBag(BaseEstimator, ClassifierMixin):
    """Mean score of n copies of `est` that differ only in `param` (a seed)."""

    def __init__(self, est, n=5, param="random_state"):
        self.est = est
        self.n = n
        self.param = param

    def fit(self, X, y):
        self.ms_ = [clone(self.est).set_params(**{self.param: s}).fit(X, y)
                    for s in range(self.n)]
        self.classes_ = self.ms_[0].classes_
        return self

    def decision_function(self, X):
        return np.mean([score(m, X) for m in self.ms_], axis=0)


def hgb():
    return make_pipeline(to_nan(), HistGradientBoostingClassifier(
        learning_rate=0.03, max_iter=800, max_leaf_nodes=4, min_samples_leaf=10,
        l2_regularization=1.0, max_features=0.5, early_stopping=False,
        random_state=0))


def et_nan():
    return make_pipeline(to_nan(), ExtraTreesClassifier(
        n_estimators=1000, max_features="sqrt", min_samples_leaf=1,
        random_state=0, n_jobs=1))


def et_mean3():
    return make_pipeline(SimpleImputer(missing_values=-1, strategy="mean"),
                         ExtraTreesClassifier(n_estimators=1000, max_features=3,
                                              min_samples_leaf=1, random_state=0,
                                              n_jobs=1))


def lgbm():
    from lightgbm import LGBMClassifier
    return make_pipeline(to_nan(), LGBMClassifier(
        n_estimators=800, learning_rate=0.03, num_leaves=4, min_child_samples=10,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
        random_state=0, n_jobs=1, verbose=-1))


def mlp():
    return make_pipeline(Feat("log"), StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(16,), alpha=30.0, solver="lbfgs", max_iter=1000,
        random_state=0))


def spline_lr():
    return make_pipeline(
        Feat("log"),
        SplineTransformer(n_knots=3, degree=2, knots="uniform", extrapolation="linear"),
        StandardScaler(), LogisticRegression(C=0.01, max_iter=5000))


def lr():
    return make_pipeline(Feat("log"), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=5000))


def svc():
    return make_pipeline(Feat("sqrt"), StandardScaler(), SVC(C=10.0, gamma=0.01))


# In-class methods (CSDS 340 list), best config per family from
# inclass_screen.py; all linear/distance models on log1p(1e4x) + standardize.
def lin(est):
    return make_pipeline(Feat("log"), StandardScaler(), est)


def perceptron():
    from sklearn.linear_model import SGDClassifier
    return lin(SGDClassifier(loss="perceptron", penalty="l2", alpha=0.01, average=True,
                             max_iter=2000, random_state=0))


def adaline():
    """Adaline's least-squares objective, solved in closed form (ridge)."""
    from sklearn.linear_model import RidgeClassifier
    return lin(RidgeClassifier(alpha=1000.0))


def linsvc():
    from sklearn.svm import LinearSVC
    return lin(LinearSVC(C=0.01, max_iter=20000))


def svc_log():
    return lin(SVC(C=30.0, gamma=0.001))


def knn():
    from sklearn.neighbors import KNeighborsClassifier
    return lin(KNeighborsClassifier(n_neighbors=151, weights="distance"))


def tree():
    from sklearn.tree import DecisionTreeClassifier
    return make_pipeline(to_nan(), DecisionTreeClassifier(max_depth=8, min_samples_leaf=20,
                                                          random_state=0))


def tabpfn():
    """Open TabPFN v2 weights, CPU. Needs torch + tabpfn, which are not in
    pyproject (see final/README.md)."""
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    return make_pipeline(to_nan(), TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, device="cpu", random_state=0,
        ignore_pretraining_limits=True))


MODELS = {
    "hgb": hgb,
    "hgb_bag5": lambda: SeedBag(hgb(), 5, "histgradientboostingclassifier__random_state"),
    "et_nan": et_nan,
    "et_mean3": et_mean3,
    "lgbm": lgbm,
    "mlp": mlp,
    "mlp_bag5": lambda: SeedBag(mlp(), 5, "mlpclassifier__random_state"),
    "spline_lr": spline_lr,
    "lr": lr,
    "svc": svc,
    "tabpfn": tabpfn,
    "perceptron": perceptron,
    "adaline": adaline,
    "linsvc": linsvc,
    "svc_log": svc_log,
    "knn": knn,
    "tree": tree,
}
