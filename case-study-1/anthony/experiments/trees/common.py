"""Shared helpers for the tree-model experiments (wraps ../harness.py)."""
import json
import os
import time
import warnings

import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from harness import evaluate, fmt, load, to_nan, tpr_at_fpr

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results.jsonl")
OOF_DIR = os.path.join(HERE, "oof")
os.makedirs(OOF_DIR, exist_ok=True)
N_JOBS = 10  # one worker per fold; models themselves are single-threaded


def imp(strategy="mean", fill_value=None, add_indicator=False):
    kw = dict(missing_values=-1, strategy=strategy, add_indicator=add_indicator)
    if strategy == "constant":
        kw["fill_value"] = fill_value
    return SimpleImputer(**kw)


class RowStats(BaseEstimator, TransformerMixin):
    """Appends per-row summary features. Input has -1 (or NaN) for missing.
    Stateless, so nothing is learned from the data.
    mode='raw':  n_missing, n_nonzero, sum, max (over observed entries)
    mode='norm': n_missing, frac_nonzero_observed, mean_observed, max
    Output keeps missing entries as they came in (-1 or NaN)."""

    def __init__(self, mode="raw"):
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        miss = (X == -1) | np.isnan(X)
        obs = np.where(miss, 0.0, X)
        n_miss = miss.sum(1)
        n_obs = np.maximum((~miss).sum(1), 1)
        nnz = ((obs > 0) & ~miss).sum(1)
        s = obs.sum(1)
        mx = obs.max(1)
        if self.mode == "raw":
            extra = [n_miss, nnz, s, mx]
        else:
            extra = [n_miss, nnz / n_obs, s / n_obs, mx]
        return np.column_stack([X] + extra)


class Avg(BaseEstimator, ClassifierMixin):
    """Fits each estimator; predict_proba[:,1] = mean of per-model normalized ranks
    (how='rank') or mean of probabilities (how='prob')."""

    def __init__(self, estimators, how="rank"):
        self.estimators = estimators
        self.how = how

    def fit(self, X, y):
        self.fitted_ = [clone(e).fit(X, y) for e in self.estimators]
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        S = [m.predict_proba(X)[:, 1] for m in self.fitted_]
        p = np.mean(S, axis=0)
        if self.how == "rank":
            # rank/len values repeat across CV folds, creating artificial ties in
            # pooled OOF scores; a tiny mean-prob term breaks them without
            # reordering anything within a batch.
            p = np.mean([rankdata(s) / len(s) for s in S], axis=0) + 1e-6 * p
        return np.column_stack([1 - p, p])


def fit_time(model, X, y, reps=1):
    """Wall time of one fit on all 1500 rows, single-threaded."""
    with threadpool_limits(limits=1):
        t = time.perf_counter()
        for _ in range(reps):
            clone(model).fit(X, y)
    return (time.perf_counter() - t) / reps


def run(configs, group, X=None, y=None, n_jobs=N_JOBS):
    """configs: list of (name, model). Appends rows to results.jsonl, saves OOFs."""
    if X is None:
        X, y = load()
    out = []
    for name, model in configs:
        t0 = time.perf_counter()
        r = evaluate(model, X, y, name=name, n_jobs=n_jobs, return_oof=True)
        wall = time.perf_counter() - t0
        np.save(os.path.join(OOF_DIR, _slug(name) + ".npy"), np.array(r.pop("oof")))
        r["fit_s"] = fit_time(model, X, y)
        r["eval_wall_s"] = wall
        r["group"] = group
        print(fmt(r) + f"  fit {r['fit_s']:.2f}s  (eval {wall:.0f}s)", flush=True)
        with open(RES, "a") as f:
            f.write(json.dumps(r) + "\n")
        out.append(r)
    return out


def _slug(name):
    return "".join(c if c.isalnum() else "_" for c in name)


def load_oof(name):
    return np.load(os.path.join(OOF_DIR, _slug(name) + ".npy"))


def paired(name_a, name_b, y):
    """Per-repeat paired differences a-b (same splits): AUC and interp TPR@1%."""
    A, B = load_oof(name_a), load_oof(name_b)
    da = [roc_auc_score(y, a) - roc_auc_score(y, b) for a, b in zip(A, B)]
    dt = [tpr_at_fpr(y, a)[0] - tpr_at_fpr(y, b)[0] for a, b in zip(A, B)]
    return np.mean(da), np.std(da), np.mean(dt), np.std(dt)
