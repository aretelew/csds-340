# -*- coding: utf-8 -*-
"""
CSDS 340 Case Study 1 spam classifier.

Ensemble of logistic regression, an RBF SVM and a random forest. Each model's
test scores are converted to percentiles of its 5-fold CV scores on the
training set, and the percentiles are averaged.

Follows the same structure as Professor Xu's classifySpam.py file
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

MEMBERS = ["lr", "svc_log", "rf"]


def _neg1_to_nan(Z):
    Z = np.array(Z, dtype=float, copy=True)
    Z[Z == -1] = np.nan
    return Z


class LogMean(BaseEstimator, TransformerMixin):
    # log1p(1e4*x), missing values filled with the column mean

    def fit(self, X, y=None):
        self.fill_ = np.nanmean(np.log1p(1e4 * _neg1_to_nan(X)), axis=0)
        return self

    def transform(self, X):
        F = np.log1p(1e4 * _neg1_to_nan(X))
        return np.where(np.isnan(F), self.fill_, F)


def _model(name):
    if name == "lr":
        return make_pipeline(LogMean(), StandardScaler(),
                             LogisticRegression(C=0.1, max_iter=5000))
    if name == "svc_log":
        return make_pipeline(LogMean(), StandardScaler(), SVC(C=30.0, gamma=0.001))
    if name == "rf":
        return make_pipeline(FunctionTransformer(_neg1_to_nan), RandomForestClassifier(
            n_estimators=1000, max_features=1, min_samples_leaf=1,
            random_state=0, n_jobs=-1))
    raise ValueError(name)


def _score(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def _pct_map(ref, s):
    # map scores s to percentiles of ref, extrapolating past the ends
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
    X = np.asarray(trainFeatures, dtype=float)
    y = np.asarray(trainLabels)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pcts = []
    for name in MEMBERS:
        ref = np.empty(len(y))
        for tr, te in cv.split(X, y):
            ref[te] = _score(_model(name).fit(X[tr], y[tr]), X[te])
        s = _score(_model(name).fit(X, y), testFeatures)
        pcts.append(_pct_map(ref, s))
    return np.mean(pcts, axis=0)


# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    from pathlib import Path
    from sklearn.metrics import roc_auc_score, roc_curve

    def tprAtFPR(labels, outputs, desiredFPR=0.01):
        fpr, tpr, thres = roc_curve(labels, outputs)
        maxFprIndex = np.where(fpr <= desiredFPR)[0][-1]
        return ((tpr[maxFprIndex + 1] - tpr[maxFprIndex])
                / (fpr[maxFprIndex + 1] - fpr[maxFprIndex])
                * (desiredFPR - fpr[maxFprIndex]) + tpr[maxFprIndex])

    path = Path(__file__).resolve().parent.parent / "spamTrain1.csv"
    data = np.loadtxt(path if path.exists() else "spamTrain1.csv", delimiter=",")
    features, labels = data[:, :-1], data[:, -1]

    # 10-fold CV
    outputs = np.empty(len(labels))
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for tr, te in cv.split(features, labels):
        outputs[te] = predictTest(features[tr], labels[tr], features[te])
    print(f"10-fold CV AUC: {roc_auc_score(labels, outputs):.4f}")
    print(f"10-fold CV TPR at FPR = 0.01: {tprAtFPR(labels, outputs):.4f}")