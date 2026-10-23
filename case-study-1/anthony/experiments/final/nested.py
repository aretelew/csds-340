"""Cache each candidate's cross-validated scores so ensembles can be scored
later without refitting.

For every repeat r (seeds 0..9, the same outer splits as harness.evaluate)
and outer fold f, we store
  test[r]         the outer-fold test scores, pooled over folds (1500,)
  ref[r, f]       inner 5-fold OOF scores on fold f's training part, NaN on
                  its test part (1500,). This is the reference distribution a
                  deployed ensemble maps each member's test scores through
                  (ensemble.pct_map), exactly as predictTest would do.

Usage (from case-study-1/anthony/):
    uv run python -m experiments.final.nested hgb et_nan mlp ...
    ... --no-inner     outer scores only (single-model metrics, 6x faster)
    ... --jobs 1       run folds sequentially (TabPFN: torch uses all cores)
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from experiments.final.models import MODELS, score
from experiments.harness import N_REPEATS, load

CACHE = Path(__file__).resolve().parent / "cache"
INNER_SPLITS = 5


def inner_oof(name, X, y, n_splits=INNER_SPLITS, seed=0):
    """OOF scores of model `name` on (X, y) from a stratified k-fold."""
    s = np.empty(len(y))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y):
        s[te] = score(MODELS[name]().fit(X[tr], y[tr]), X[te])
    return s


def _task(name, X, y, tr, te, inner):
    warnings.filterwarnings("ignore")
    s_te = score(MODELS[name]().fit(X[tr], y[tr]), X[te])
    ref = inner_oof(name, X[tr], y[tr]) if inner else None
    return s_te, ref


def run(name, inner=True, n_jobs=-1):
    X, y = load()
    n = len(y)
    tasks = []
    for r in range(N_REPEATS):
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=r)
        tasks += [(r, f, tr, te) for f, (tr, te) in enumerate(cv.split(X, y))]
    t = time.time()
    out = Parallel(n_jobs=n_jobs)(
        delayed(_task)(name, X, y, tr, te, inner) for _, _, tr, te in tasks)
    test = np.empty((N_REPEATS, n))
    fold = np.empty((N_REPEATS, n), dtype=int)
    ref = np.full((N_REPEATS, 10, n), np.nan) if inner else np.empty(0)
    for (r, f, tr, te), (s_te, s_ref) in zip(tasks, out):
        test[r, te] = s_te
        fold[r, te] = f
        if inner:
            ref[r, f, tr] = s_ref
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}{'' if inner else '_outer'}.npz"
    np.savez_compressed(path, test=test, fold=fold, ref=ref, y=y)
    print(f"{name}: cached to {path.name} [{time.time() - t:.0f}s]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", choices=sorted(MODELS))
    ap.add_argument("--no-inner", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    for name in a.names:
        run(name, inner=not a.no_inner, n_jobs=a.jobs)
