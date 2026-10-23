"""Deployable equal-weight percentile ensembles, scored from nested.py caches.

A deployed ensemble cannot rank across the test batch, so each member's test
score is mapped to a percentile of that member's out-of-fold scores on the
training data (pct_map) and the percentiles are averaged. Here the reference
for outer fold f is the inner 5-fold OOF on f's training part, which is what
predictTest would compute on the full training set.
"""
from functools import lru_cache

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.final.nested import CACHE
from experiments.harness import N_REPEATS, load, tprAtFPR


def pct_map(ref, s):
    """Map scores `s` to percentiles of the reference scores `ref`.

    Mid-rank percentiles at the unique reference values, linear interpolation
    between them, and rational tails past both ends, 1 - (1-q)/(1+x), that
    start with the slope of the outer 10% of the reference and approach 0/1
    only polynomially (an exponential tail rounds to exactly 1.0 a short way
    past the max). The map is strictly increasing, so it keeps each member's
    own ranking and creates no ties above the training max (clipping would)."""
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


@lru_cache(maxsize=None)
def cache(name):
    d = np.load(CACHE / f"{name}.npz")
    return {k: d[k] for k in d.files}


@lru_cache(maxsize=None)
def labels():
    return load()[1]


@lru_cache(maxsize=None)
def pct_scores(name):
    """(N_REPEATS, n) outer test scores mapped through each fold's reference."""
    c = cache(name)
    out = np.empty_like(c["test"])
    for r in range(N_REPEATS):
        for f in range(10):
            te = c["fold"][r] == f
            ref = c["ref"][r, f]
            out[r, te] = pct_map(ref[~np.isnan(ref)], c["test"][r, te])
    return out


def metrics(S, y, name):
    """S: (N_REPEATS, n) pooled scores -> same fields as harness.evaluate."""
    aucs = np.array([roc_auc_score(y, s) for s in S])
    tprs = np.array([tprAtFPR(y, s) for s in S])
    return dict(name=name, auc=aucs.mean(), auc_sd=aucs.std(), aucs=aucs,
                tpr1=tprs.mean(), tpr1_sd=tprs.std(), tprs=tprs,
                pauc05=np.mean([roc_auc_score(y, s, max_fpr=0.05) for s in S]))


def raw(name):
    """Single model, raw pooled scores (what harness.evaluate reports)."""
    return metrics(cache(name)["test"], labels(), name)


def combo(names):
    """Equal-weight percentile average (a single name = the mapped model)."""
    S = np.mean([pct_scores(n) for n in names], axis=0)
    return metrics(S, labels(), "+".join(names))
