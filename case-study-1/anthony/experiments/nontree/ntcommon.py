"""Shared feature transforms + result logging for the non-tree experiments.

Everything here is a proper sklearn transformer: statistics (impute values,
quantiles) are learned in fit() on training rows only.
"""
import json
import os
import time
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.preprocessing import QuantileTransformer

from harness import evaluate, fmt, load

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.jsonl")


def _tf(Z, tf, c):
    """Elementwise transform; NaN stays NaN."""
    if tf == "id":
        return Z
    if tf == "log":
        return np.log1p(c * Z)
    if tf == "sqrt":
        return np.sqrt(Z)
    if tf == "pres":
        out = (Z > 0).astype(float)
        out[np.isnan(Z)] = np.nan
        return out
    raise ValueError(tf)


class FeatX(BaseEstimator, TransformerMixin):
    """Raw (-1 = missing) -> transformed, imputed feature matrix.

    tf:       'id' | 'log' (log1p(c*x)) | 'sqrt' | 'pres' (x>0) | 'quant' (QuantileTransformer normal)
    impute:   'mean' | 'median' | 'zero'
    space:    'tf'  -> impute after transforming (statistics in transformed space)
              'raw' -> impute raw proportions, then transform
    presence: also append x>0 indicators (imputed with their training mean, or 0 if impute='zero')
    agg:      append row aggregates over observed entries: n_nonzero, n_missing,
              frac_nonzero, log1p(100*sum), log1p(100*mean)
    ind:      append missing indicators
    """

    def __init__(self, tf="log", c=100.0, impute="mean", space="tf",
                 presence=False, agg=False, ind=False):
        self.tf = tf
        self.c = c
        self.impute = impute
        self.space = space
        self.presence = presence
        self.agg = agg
        self.ind = ind

    @staticmethod
    def _nan(X):
        Z = np.array(X, dtype=float, copy=True)
        Z[Z == -1] = np.nan
        return Z

    def _stat(self, Z):
        if self.impute == "zero":
            return np.zeros(Z.shape[1])
        f = np.nanmean if self.impute == "mean" else np.nanmedian
        return f(Z, axis=0)

    def _main(self, Z):
        if self.tf == "quant":
            return self.qt_.transform(Z)
        return _tf(Z, self.tf, self.c)

    def fit(self, X, y=None):
        Z = self._nan(X)
        if self.tf == "quant":
            self.qt_ = QuantileTransformer(output_distribution="normal",
                                           n_quantiles=min(1000, len(Z))).fit(Z)
        if self.space == "raw":
            self.fill_ = self._stat(Z)
        else:
            self.fill_ = self._stat(self._main(Z))
        P = _tf(Z, "pres", 1)
        self.pfill_ = np.zeros(Z.shape[1]) if self.impute == "zero" else np.nanmean(P, 0)
        return self

    def transform(self, X):
        Z = self._nan(X)
        M = np.isnan(Z)
        if self.space == "raw":
            Zr = np.where(M, self.fill_, Z)
            F = self._main(Zr)
        else:
            F = self._main(Z)
            F = np.where(M, self.fill_, F)
        parts = [F]
        if self.presence:
            P = _tf(Z, "pres", 1)
            parts.append(np.where(M, self.pfill_, P))
        if self.ind:
            parts.append(M.astype(float))
        if self.agg:
            obs = (~M).sum(1)
            nz = np.nansum(Z > 0, 1)
            s = np.nansum(Z, 1)
            parts.append(np.column_stack([
                nz, M.sum(1), nz / np.maximum(obs, 1),
                np.log1p(100 * s), np.log1p(100 * s / np.maximum(obs, 1))]))
        return np.hstack(parts)


class ZInfNB(BaseEstimator, ClassifierMixin):
    """Missing-aware zero-inflated naive Bayes ("hurdle" NB).

    Per feature and class: P(x>0 | y) Bernoulli, and log(x) | x>0, y ~ Gaussian.
    Missing features are marginalised out (simply skipped in the likelihood),
    which is the exact NB treatment of MCAR missingness.
    Score = log posterior odds (continuous, so no probability saturation ties).
    """

    def __init__(self, alpha=1.0, var_smooth=0.1, gaussian=True):
        self.alpha = alpha
        self.var_smooth = var_smooth
        self.gaussian = gaussian

    def fit(self, X, y):
        Z = FeatX._nan(X)
        y = np.asarray(y)
        self.classes_ = np.array([0.0, 1.0])
        self.prior_ = np.array([np.mean(y == 0), np.mean(y == 1)])
        L = np.log(np.where(Z > 0, Z, np.nan))
        pooled_var = np.nanvar(L, 0)
        self.p_, self.mu_, self.var_ = [], [], []
        for k in (0, 1):
            Zk, Lk = Z[y == k], L[y == k]
            obs = (~np.isnan(Zk)).sum(0)
            pos = np.nansum(Zk > 0, 0)
            self.p_.append((pos + self.alpha) / (obs + 2 * self.alpha))
            # features with <2 positives in this class fall back to the pooled Gaussian
            few = pos < 2
            mu = np.where(few, np.nanmean(L, 0), np.nanmean(np.where(few, 0, Lk), 0))
            var = np.where(few, pooled_var, np.nanvar(np.where(few, 0, Lk), 0))
            self.mu_.append(mu)
            self.var_.append(var + self.var_smooth * pooled_var + 1e-3)
        return self

    def _ll(self, Z, k):
        p, mu, var = self.p_[k], self.mu_[k], self.var_[k]
        pos = Z > 0
        zero = Z == 0
        ll = np.where(zero, np.log1p(-p), 0.0) + np.where(pos, np.log(p), 0.0)
        if self.gaussian:
            L = np.log(np.where(pos, Z, 1.0))
            g = -0.5 * (np.log(2 * np.pi * var) + (L - mu) ** 2 / var)
            ll = ll + np.where(pos, g, 0.0)
        return ll.sum(1) + np.log(self.prior_[k])

    def decision_function(self, X):
        Z = FeatX._nan(X)
        return self._ll(Z, 1) - self._ll(Z, 0)


class ScoreAvg(BaseEstimator, ClassifierMixin):
    """Average of member decision scores, each z-scored with its *training-set*
    score mean/sd (so the combination is fit on training data only)."""

    def __init__(self, members, weights=None):
        self.members = members
        self.weights = weights

    @staticmethod
    def _s(m, X):
        if hasattr(m, "decision_function"):
            return m.decision_function(X)
        p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    def fit(self, X, y):
        self.fitted_ = [clone(m).fit(X, y) for m in self.members]
        self.stats_ = []
        for m in self.fitted_:
            s = self._s(m, X)
            self.stats_.append((s.mean(), s.std() + 1e-12))
        self.classes_ = self.fitted_[0].classes_
        return self

    def decision_function(self, X):
        w = self.weights or [1.0] * len(self.fitted_)
        return sum(wi * (self._s(m, X) - mu) / sd
                   for wi, m, (mu, sd) in zip(w, self.fitted_, self.stats_)) / sum(w)


X, y = load()


def run(model, name, family, oof=False):
    t = time.time()
    r = evaluate(model, X, y, name=name, return_oof=oof)
    r["family"] = family
    r["secs"] = round(time.time() - t, 1)
    print(fmt(r), f"[{r['secs']}s]", flush=True)
    rec = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
           for k, v in r.items() if k != "oof"}
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if oof:
        os.makedirs(os.path.join(HERE, "oof"), exist_ok=True)
        np.save(os.path.join(HERE, "oof", name.replace(" ", "_").replace("/", "-") + ".npy"),
                np.array(r["oof"]))
    return r


class JLL(BaseEstimator, ClassifierMixin):
    """Wraps an sklearn NB so the score is the joint-log-likelihood difference
    (log posterior odds) instead of predict_proba, which saturates to 0/1 and ties."""

    def __init__(self, nb):
        self.nb = nb

    def fit(self, X, y):
        self.nb_ = clone(self.nb).fit(X, y)
        self.classes_ = self.nb_.classes_
        return self

    def decision_function(self, X):
        j = self.nb_.predict_joint_log_proba(X)
        return j[:, 1] - j[:, 0]
