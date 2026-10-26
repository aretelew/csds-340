"""Shared data loading, metrics and transforms for Ethan's experiments.

Evaluation protocol (kept identical to anthony/experiments/harness.py so the
numbers are directly comparable): 10 repeats of stratified 10-fold CV with
seeds 0..9; out-of-fold scores are pooled over all 1500 emails per repeat and
the metrics are computed once per repeat. TPR at 1% FPR uses the grader's exact
formula from evaluateClassifier.py.
"""
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_curve

DATA = Path(__file__).resolve().parent.parent / "spamTrain1.csv"
N_REPEATS = 10
N_SPLITS = 10


def load(path=DATA):
    d = np.loadtxt(path, delimiter=",")
    return d[:, :-1], d[:, -1]


def tprAtFPR(labels, outputs, desiredFPR=0.01):
    """Copied from evaluateClassifier.py (the grader)."""
    fpr, tpr, thres = roc_curve(labels, outputs)
    maxFprIndex = np.where(fpr <= desiredFPR)[0][-1]
    fprBelow = fpr[maxFprIndex]
    fprAbove = fpr[maxFprIndex + 1]
    tprBelow = tpr[maxFprIndex]
    tprAbove = tpr[maxFprIndex + 1]
    return ((tprAbove - tprBelow) / (fprAbove - fprBelow)
            * (desiredFPR - fprBelow) + tprBelow)


def neg1_to_nan(X):
    Z = np.array(X, dtype=float, copy=True)
    Z[Z == -1] = np.nan
    return Z


def pct_map(ref, s):
    """Strictly increasing map of scores s to percentiles of the reference
    scores ref: interpolated mid-rank percentiles, with rational tails past
    both ends so scores beyond the reference range do not tie at 0 or 1.
    Copied from anthony/classifySpam.py so ensembles combine the same way."""
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


class LogFill(BaseEstimator, TransformerMixin):
    """log1p(c*x) per feature; missing (-1) filled with the training mean of
    the transformed feature. Word proportions are heavily right-skewed with
    many zeros, so the log spreads out the small nonzero values."""

    def __init__(self, c=1e4):
        self.c = c

    def _f(self, X):
        return np.log1p(self.c * neg1_to_nan(X))

    def fit(self, X, y=None):
        self.fill_ = np.nanmean(self._f(X), axis=0)
        return self

    def transform(self, X):
        F = self._f(X)
        return np.where(np.isnan(F), self.fill_, F)


def continuous_score(model, X):
    """A continuous spam score from one fitted model: decision_function when
    the final step has one (SVM, perceptron), else P(spam). Hard votes would
    leave ties that wreck AUC and TPR at 1% FPR."""
    final = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(final, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]
