"""Ensemble screen: do the bagged learners from bagging_screen.py improve on
Anthony's rf + lr + svc_log ensemble when combined with it or with each other?

Fixed before looking at any ensemble result (Anthony's split-half check showed
that picking the best of hundreds of combos does not generalize):

  Members. From each family in bagging_screen.py, the bagged config with the
  highest mean AUC (the less noisy of the two metrics). Anthony's three
  members are refit here with his settings on the same splits.

  Combos. COMBOS below, nothing else.

  Combining. Each member's scores are mapped to percentiles of a reference
  set of that member's scores and the percentiles are averaged with equal
  weights (Anthony's pct_map, copied into common.py). The deployed version
  uses inner-CV scores on the training set as the reference. Here, as a
  screening shortcut Anthony validated (within ~0.001 AUC / ~0.005 TPR of
  the exact nested result), a fold's reference is that repeat's out-of-fold
  scores on the other nine folds, which are also training-only for that fold.

  Decision. A combo beats the baseline if it has the higher mean over the 10
  repeats AND wins a majority of the paired repeats.

    uv run python ensemble_screen.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from bagging_screen import N_JOBS, OUT as SCREEN_RESULTS, set_low_priority
from common import (N_REPEATS, N_SPLITS, LogFill, continuous_score, load, neg1_to_nan,
                    pct_map, tprAtFPR)

HERE = Path(__file__).resolve().parent
SCREEN_CACHE = HERE / "cache" / "bagging_screen"
CACHE = HERE / "cache" / "ensemble_screen"
OUT = HERE / "results" / "ensemble_screen.txt"

FAMILIES = {"lr": ("lr_",), "mlp": ("mlp_",), "svc": ("svc_",), "knn": ("knn_",),
            "tree": ("tree_",), "perceptron": ("perceptron",)}

# Anthony's members (anthony/classifySpam.py), refit on the same splits
ANTHONY = {
    "A_rf": lambda: make_pipeline(FunctionTransformer(neg1_to_nan), RandomForestClassifier(
        n_estimators=1000, max_features=1, min_samples_leaf=1, random_state=0, n_jobs=1)),
    "A_lr": lambda: make_pipeline(LogFill(), StandardScaler(),
                                  LogisticRegression(C=0.1, max_iter=5000)),
    "A_svc": lambda: make_pipeline(LogFill(), StandardScaler(), SVC(C=30.0, gamma=0.001)),
}

ANTHONY3 = ("A_rf", "A_lr", "A_svc")
BASELINE = "1 anthony: rf+lr+svc"
COMBOS = {
    BASELINE: ANTHONY3,
    "2 anthony + bagged mlp": ANTHONY3 + ("E_mlp",),
    "3 bagged mlp + rf + bagged lr": ("E_mlp", "A_rf", "E_lr"),
    "4 rf + bagged lr + bagged svc": ("A_rf", "E_lr", "E_svc"),
    "5 bagged mlp+svc+tree+lr": ("E_mlp", "E_svc", "E_tree", "E_lr"),
    "6 best bagged of each family": ("E_lr", "E_mlp", "E_svc", "E_knn", "E_tree", "E_perceptron"),
}


def splits(y):
    """Fold index of every email, per repeat (same splits as the harness)."""
    fold = np.empty((N_REPEATS, len(y)), dtype=int)
    for r in range(N_REPEATS):
        cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=r)
        for k, (_, te) in enumerate(cv.split(np.zeros(len(y)), y)):
            fold[r, te] = k
    return fold


def pick_members():
    """Best-AUC bagged config per family from the bagging screen."""
    rows = [json.loads(line) for line in SCREEN_RESULTS.read_text().splitlines() if line]
    picks = {}
    for fam, prefixes in FAMILIES.items():
        cands = [r for r in rows if r["scheme"] != "none" and r["base"].startswith(prefixes)]
        best = max(cands, key=lambda r: r["by_k"][str(r["n_estimators"])]["auc"])
        picks[f"E_{fam}"] = best["name"]
    return picks


def _fit_score(name, X, y, tr, te):
    warnings.filterwarnings("ignore")
    set_low_priority()
    return te, continuous_score(ANTHONY[name]().fit(X[tr], y[tr]), X[te])


def anthony_oof(name, X, y):
    path = CACHE / f"{name}.npy"
    if path.exists():
        return np.load(path)
    tasks = [(r, te, tr) for r in range(N_REPEATS)
             for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=r).split(X, y)]
    res = Parallel(n_jobs=N_JOBS)(delayed(_fit_score)(name, X, y, tr, te)
                                  for _, te, tr in tasks)
    oof = np.empty((N_REPEATS, len(y)))
    for (r, _, _), (te, s) in zip(tasks, res):
        oof[r, te] = s
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, oof)
    return oof


def mapped(oof, fold):
    """Each fold's scores as percentiles of the other folds' scores."""
    out = np.empty_like(oof)
    for r in range(N_REPEATS):
        for k in range(N_SPLITS):
            te = fold[r] == k
            out[r, te] = pct_map(oof[r, ~te], oof[r, te])
    return out


def metrics(scores, y):
    aucs = np.array([roc_auc_score(y, s) for s in scores])
    tprs = np.array([tprAtFPR(y, s) for s in scores])
    return aucs, tprs


def main():
    set_low_priority()
    X, y = load()
    fold = splits(y)
    picks = pick_members()
    oof = {name: anthony_oof(name, X, y) for name in ANTHONY}
    for key, cfg in picks.items():
        oof[key] = np.load(SCREEN_CACHE / (cfg.replace(" | ", "__").replace(".", "p") + ".npy"))
    pct = {k: mapped(v, fold) for k, v in oof.items()}

    lines = ["members (E_* = best-AUC bagged config per family):"]
    lines += [f"  {k:14s} {v}" for k, v in picks.items()]
    lines += ["", "single members, percentile-mapped (mean ± sd over 10 repeats):"]
    for k in pct:
        a, t = metrics(pct[k], y)
        lines.append(f"  {k:14s} AUC {a.mean():.4f}±{a.std():.4f}  TPR@1% {t.mean():.3f}±{t.std():.3f}")

    res = {name: metrics(np.mean([pct[m] for m in members], axis=0), y)
           for name, members in COMBOS.items()}
    base_a, base_t = res[BASELINE]
    lines += ["", "combos (equal-weight percentile average), paired vs the baseline:"]
    for name, (a, t) in res.items():
        line = (f"  {name:32s} AUC {a.mean():.4f}±{a.std():.4f}  "
                f"TPR@1% {t.mean():.3f}±{t.std():.3f}")
        if name != BASELINE:
            da, dt = a - base_a, t - base_t
            verdict = "BEATS" if dt.mean() > 0 and (dt > 0).sum() > N_REPEATS / 2 else "-"
            line += (f"   dAUC {da.mean():+.4f} ({(da > 0).sum()}/10)  "
                     f"dTPR {dt.mean():+.3f} ({(dt > 0).sum()}/10)  {verdict}")
        lines.append(line)
    lines += ["", "per-repeat TPR@1%:"]
    for name, (_, t) in res.items():
        lines.append(f"  {name:32s} " + " ".join(f"{v:.3f}" for v in t))

    text = "\n".join(lines)
    print(text)
    OUT.write_text(text + "\n")


if __name__ == "__main__":
    main()
