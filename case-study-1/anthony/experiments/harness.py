"""Shared evaluation harness for the spam case study.

Protocol: repeated stratified 10-fold CV. Within each repeat, out-of-fold
scores are pooled over all training emails and the metrics are computed once
on the pooled scores (the grader's test set is also 1500 emails); we report
mean and std across repeats. Models receive the raw features (missing = -1),
exactly like predictTest, so every pipeline must handle -1 itself (use
`to_nan()` as a first step if you want NaN).

TPR at 1% FPR uses the grader's exact formula from evaluateClassifier.py.

Model comparison rule: model A beats model B if it has the higher mean over
10 repeats AND wins in a majority of the paired repeats (same splits).

Usage (from case-study-1/anthony/):
    uv run python -m experiments.some_script
    from experiments.harness import load, evaluate, compare, to_nan, fmt
"""
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer

DATA = Path(__file__).resolve().parents[2] / "spamTrain1.csv"
N_REPEATS = 10


def load(path=DATA):
    d = np.loadtxt(path, delimiter=",")
    return d[:, :-1], d[:, -1]


def _neg1_to_nan(Z):
    Z = np.array(Z, dtype=float, copy=True)
    Z[Z == -1] = np.nan
    return Z


def to_nan():
    return FunctionTransformer(_neg1_to_nan)


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


def scores_of(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def _fit_score(model, X, y, tr, te):
    m = clone(model)
    m.fit(X[tr], y[tr])
    return te, scores_of(m, X[te])


def oof_scores(model, X, y, seed, n_splits=10, n_jobs=-1):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = Parallel(n_jobs=n_jobs)(
        delayed(_fit_score)(model, X, y, tr, te) for tr, te in cv.split(X, y))
    s = np.empty(len(y))
    for te, st in out:
        s[te] = st
    return s


def evaluate(model, X, y, name="", n_repeats=N_REPEATS, n_splits=10,
             n_jobs=-1, return_oof=False):
    """Pooled-OOF metrics per repeat. Seeds 0..n_repeats-1 are fixed, so every
    caller sees identical splits and results can be compared pairwise."""
    aucs, tprs, paucs, oofs = [], [], [], []
    for r in range(n_repeats):
        s = oof_scores(model, X, y, seed=r, n_splits=n_splits, n_jobs=n_jobs)
        oofs.append(s)
        aucs.append(roc_auc_score(y, s))
        tprs.append(tprAtFPR(y, s))
        paucs.append(roc_auc_score(y, s, max_fpr=0.05))
    res = dict(name=name,
               auc=np.mean(aucs), auc_sd=np.std(aucs), aucs=np.array(aucs),
               tpr1=np.mean(tprs), tpr1_sd=np.std(tprs), tprs=np.array(tprs),
               pauc05=np.mean(paucs))
    if return_oof:
        res["oof"] = oofs
    return res


def compare(a, b, metric="tprs"):
    """Paired comparison of two evaluate() results on the same splits.
    Returns (mean difference a-b, repeats won by a, a beats b by our rule)."""
    da, db = a[metric], b[metric]
    diff = da.mean() - db.mean()
    wins = int((da > db).sum())
    return diff, wins, bool(diff > 0 and wins > len(da) / 2)


def fmt(r):
    return (f"{r['name']:40s} AUC {r['auc']:.4f}±{r['auc_sd']:.4f}  "
            f"TPR@1% {r['tpr1']:.3f}±{r['tpr1_sd']:.3f}  "
            f"pAUC.05 {r['pauc05']:.4f}")
