"""Shared evaluation harness for the CSDS 340 spam case study.

Protocol: repeated stratified 10-fold CV. Within each repeat, out-of-fold scores
are pooled over all 1500 emails and the metrics are computed once on the pooled
scores; we report mean and std across repeats. Models receive the raw features
(missing = -1), exactly like the grader's predictTest, so every pipeline must
handle -1 itself (use `to_nan()` as a first step if you want NaN).

Usage:
    from harness import load, evaluate, to_nan
    X, y = load()
    print(evaluate(model, X, y, name="my model"))
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer

DATA = "/Users/aretelew/Developer/csds-340/case-study-1/spamTrain1.csv"


def load(path=DATA):
    d = np.loadtxt(path, delimiter=",")
    return d[:, :-1], d[:, -1]


def _neg1_to_nan(Z):
    Z = np.array(Z, dtype=float, copy=True)
    Z[Z == -1] = np.nan
    return Z


def to_nan():
    return FunctionTransformer(_neg1_to_nan)


def scores_of(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def tpr_at_fpr(y, s, fpr_target=0.01):
    """Returns (interpolated TPR, step TPR = max TPR with FPR <= target)."""
    fpr, tpr, _ = roc_curve(y, s)
    return np.interp(fpr_target, fpr, tpr), tpr[fpr <= fpr_target].max()


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


def evaluate(model, X, y, name="", n_repeats=5, n_splits=10, n_jobs=-1,
             return_oof=False):
    """Pooled-OOF metrics averaged over repeats. Seeds 0..n_repeats-1 are fixed,
    so every caller sees identical splits."""
    aucs, pauc, ti, ts, oofs = [], [], [], [], []
    for r in range(n_repeats):
        s = oof_scores(model, X, y, seed=r, n_splits=n_splits, n_jobs=n_jobs)
        oofs.append(s)
        aucs.append(roc_auc_score(y, s))
        pauc.append(roc_auc_score(y, s, max_fpr=0.05))
        a, b = tpr_at_fpr(y, s)
        ti.append(a)
        ts.append(b)
    res = dict(name=name,
               auc=np.mean(aucs), auc_sd=np.std(aucs),
               tpr1=np.mean(ti), tpr1_sd=np.std(ti),
               tpr1_step=np.mean(ts),
               pauc05=np.mean(pauc))
    if return_oof:
        res["oof"] = oofs
    return res


def fmt(r):
    return (f"{r['name']:40s} AUC {r['auc']:.4f}±{r['auc_sd']:.4f}  "
            f"TPR@1% {r['tpr1']:.3f}±{r['tpr1_sd']:.3f} (step {r['tpr1_step']:.3f})  "
            f"pAUC.05 {r['pauc05']:.4f}")
