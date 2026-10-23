"""Draft predictTest: equal-weight percentile ensemble of MEMBERS.

Self-contained (numpy + scikit-learn >= 1.6 only) so it can be copied into
classifySpam.py. Each member is fit on the full training set; its test scores
are mapped to percentiles of its own 5-fold out-of-fold scores on the training
set and the percentiles are averaged. Nothing depends on the rest of the test
batch, so a test email's score is the same whether it is scored alone or with
1500 others. Identical to experiments.final.{models,nested,ensemble}; check
with `uv run python -m experiments.final.check_predict`.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer, StandardScaler

MEMBERS = ["hgb", "et_nan", "mlp", "spline_lr"]


def _neg1_to_nan(Z):
    Z = np.array(Z, dtype=float, copy=True)
    Z[Z == -1] = np.nan
    return Z


class LogMean(BaseEstimator, TransformerMixin):
    """log1p(1e4*x), missing (-1) filled with the training mean of the log."""

    def fit(self, X, y=None):
        self.fill_ = np.nanmean(np.log1p(1e4 * _neg1_to_nan(X)), axis=0)
        return self

    def transform(self, X):
        F = np.log1p(1e4 * _neg1_to_nan(X))
        return np.where(np.isnan(F), self.fill_, F)


def _model(name):
    if name == "hgb":
        return make_pipeline(FunctionTransformer(_neg1_to_nan), HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=800, max_leaf_nodes=4, min_samples_leaf=10,
            l2_regularization=1.0, max_features=0.5, early_stopping=False,
            random_state=0))
    if name == "et_nan":
        return make_pipeline(FunctionTransformer(_neg1_to_nan), ExtraTreesClassifier(
            n_estimators=1000, max_features="sqrt", min_samples_leaf=1,
            random_state=0, n_jobs=-1))
    if name == "mlp":
        return make_pipeline(LogMean(), StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(16,), alpha=30.0, solver="lbfgs", max_iter=1000,
            random_state=0))
    if name == "spline_lr":
        return make_pipeline(
            LogMean(),
            SplineTransformer(n_knots=3, degree=2, knots="uniform", extrapolation="linear"),
            StandardScaler(), LogisticRegression(C=0.01, max_iter=5000))
    raise ValueError(name)


def _score(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def _pct_map(ref, s):
    """Strictly increasing map of scores s to percentiles of reference scores
    ref: interpolated mid-rank percentiles, with rational tails past both ends
    so test scores above the training max do not tie at 1."""
    s = np.asarray(s, dtype=float)
    r, counts = np.unique(ref, return_counts=True)
    q = (np.cumsum(counts) - counts / 2) / counts.sum()
    out = np.interp(s, r, q)
    k = max(1, len(r) // 10)
    hi = (q[-1] - q[-1 - k]) / (r[-1] - r[-1 - k])
    lo = (q[k] - q[0]) / (r[k] - r[0])
    above, below = s > r[-1], s < r[0]
    out[above] = 1 - (1 - q[-1]) / (1 + (s[above] - r[-1]) * hi / (1 - q[-1]))
    out[below] = q[0] / (1 + (r[0] - s[below]) * lo / q[0])
    return out


def predictTest(trainFeatures, trainLabels, testFeatures):
    X, y = np.asarray(trainFeatures, dtype=float), np.asarray(trainLabels)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pcts = []
    for name in MEMBERS:
        ref = np.empty(len(y))
        for tr, te in cv.split(X, y):
            ref[te] = _score(_model(name).fit(X[tr], y[tr]), X[te])
        s = _score(_model(name).fit(X, y), testFeatures)
        pcts.append(_pct_map(ref, s))
    return np.mean(pcts, axis=0)
