"""Bagging screen: which base learner gains the most from bagging?

Each base learner is evaluated unbagged ("none") and inside a
BaggingClassifier with four resampling schemes:
    rows      bootstrap the emails (classic bagging), all features
    half      half the emails without replacement (subbagging), all features
    subspace  all emails, a random half of the features (random subspace)
    both      bootstrap the emails AND a random half of the features

Base-learner settings are deliberately low-bias / high-variance (deep trees,
small k, large C, weak MLP penalty): bagging only cancels variance, so it
pays off for learners that overfit individually.

The bag's own predict_proba falls back to hard votes for SVMs and the
perceptron, so scores are averaged here from each member's continuous score
(see common.continuous_score). One fit yields the scores at every checkpoint
in CHECKPOINTS (first k members), which shows whether more members help.

Results go to results/bagging_screen.jsonl (one line per config, resumable:
finished configs are skipped) and pooled OOF scores to cache/ (gitignored).
The process runs at below-normal priority on half the cores.

    uv run python bagging_screen.py              # run everything not yet done
    uv run python bagging_screen.py --only knn   # configs whose name contains knn
    uv run python bagging_screen.py --report     # print the summary
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common import N_REPEATS, N_SPLITS, LogFill, continuous_score, load, tprAtFPR

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "bagging_screen.jsonl"
CACHE = HERE / "cache" / "bagging_screen"
N_JOBS = max(1, (os.cpu_count() or 2) // 2)
CHECKPOINTS = (1, 10, 25, 50, 100, 200, 300, 500, 1000)


def lin(est):
    return make_pipeline(LogFill(), StandardScaler(), est)


# name -> (factory, number of bag members). Trees see the raw -1: every real
# value is >= 0, so a single split isolates the missing entries.
BASES = {
    "tree_msl1": (lambda: DecisionTreeClassifier(min_samples_leaf=1), 1000),
    "tree_msl5": (lambda: DecisionTreeClassifier(min_samples_leaf=5), 1000),
    "perceptron": (lambda: lin(Perceptron(max_iter=1000, tol=None)), 1000),
    "lr_C1": (lambda: lin(LogisticRegression(C=1.0, max_iter=5000)), 500),
    "knn_k15": (lambda: lin(KNeighborsClassifier(n_neighbors=15, weights="distance")), 300),
    "knn_k51": (lambda: lin(KNeighborsClassifier(n_neighbors=51, weights="distance")), 300),
    "svc_C3_g.03": (lambda: lin(SVC(C=3.0, gamma=0.03)), 300),
    "svc_C100_g.01": (lambda: lin(SVC(C=100.0, gamma=0.01)), 300),
    # 50 members, not more: each MLP takes ~1 s, so 200 would mean ~8 h here
    "mlp_h32_a1": (lambda: lin(MLPClassifier(hidden_layer_sizes=(32,), alpha=1.0,
                                             solver="lbfgs", max_iter=2000)), 50),
    "mlp_h32_a10": (lambda: lin(MLPClassifier(hidden_layer_sizes=(32,), alpha=10.0,
                                              solver="lbfgs", max_iter=2000)), 50),
}

SCHEMES = {
    "rows": dict(max_samples=1.0, bootstrap=True, max_features=1.0),
    "half": dict(max_samples=0.5, bootstrap=False, max_features=1.0),
    "subspace": dict(max_samples=1.0, bootstrap=False, max_features=0.5),
    "both": dict(max_samples=1.0, bootstrap=True, max_features=0.5),
}


def configs():
    out = []
    for base, (factory, n) in BASES.items():
        out.append((f"{base} | none", base, None, 1))
        for scheme in SCHEMES:
            out.append((f"{base} | {scheme}", base, scheme, n))
    return out


def set_low_priority():
    """Below-normal priority so the screen yields to anything interactive.
    joblib's workers do not inherit it on Windows, so each task calls this
    too."""
    if sys.platform == "win32":
        import ctypes
        BELOW_NORMAL_PRIORITY_CLASS = 0x4000
        k32 = ctypes.windll.kernel32
        # without these the 64-bit process handle is truncated to a C int
        # and the call silently fails
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
    else:
        os.nice(10)


def fit_score(base, scheme, n, X, y, tr, te):
    """Fit one bag on tr; return te and the mean member score on te at every
    checkpoint <= n."""
    set_low_priority()
    warnings.filterwarnings("ignore")
    factory = BASES[base][0]
    if scheme is None:
        m = factory()
        m.set_params(**{p: 0 for p in m.get_params() if p.endswith("random_state")})
        return te, {1: continuous_score(m.fit(X[tr], y[tr]), X[te])}
    bag = BaggingClassifier(factory(), n_estimators=n, random_state=0, n_jobs=1,
                            **SCHEMES[scheme]).fit(X[tr], y[tr])
    total = np.zeros(len(te))
    out = {}
    for i, (est, feats) in enumerate(zip(bag.estimators_, bag.estimators_features_), 1):
        total += continuous_score(est, X[te][:, feats])
        if i in CHECKPOINTS:
            out[i] = total / i
    return te, out


def run(name, base, scheme, n, X, y):
    tasks = [(r, tr, te) for r in range(N_REPEATS)
             for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=r).split(X, y)]
    t0 = time.time()
    res = Parallel(n_jobs=N_JOBS)(delayed(fit_score)(base, scheme, n, X, y, tr, te)
                                  for r, tr, te in tasks)
    ks = sorted(res[0][1])
    oof = {k: np.empty((N_REPEATS, len(y))) for k in ks}
    for (r, _, _), (te, scores) in zip(tasks, res):
        for k in ks:
            oof[k][r, te] = scores[k]
    by_k = {}
    for k in ks:
        aucs = [roc_auc_score(y, s) for s in oof[k]]
        tprs = [tprAtFPR(y, s) for s in oof[k]]
        by_k[str(k)] = dict(auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs)),
                            tpr1=float(np.mean(tprs)), tpr1_sd=float(np.std(tprs)),
                            aucs=[float(a) for a in aucs], tprs=[float(t) for t in tprs])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / (name.replace(" | ", "__").replace(".", "p") + ".npy"), oof[ks[-1]])
    row = dict(name=name, base=base, scheme=scheme or "none", n_estimators=n,
               seconds=round(time.time() - t0, 1), by_k=by_k)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    final = by_k[str(ks[-1])]
    print(f"{name:28s} AUC {final['auc']:.4f}±{final['auc_sd']:.4f}  "
          f"TPR@1% {final['tpr1']:.3f}±{final['tpr1_sd']:.3f}  ({row['seconds']:.0f}s)",
          flush=True)


def read_rows():
    if not OUT.exists():
        return {}
    rows = [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()]
    return {r["name"]: r for r in rows}


# Anthony's numbers on the same splits (anthony/experiments/final/README.md)
REFERENCE = [("anthony presubmission rf+lr+svc_log", 0.9246, 0.575),
             ("anthony rf (1000 trees, max_features=1)", 0.9167, 0.541),
             ("anthony mlp (16,), alpha=30", 0.9234, 0.563)]


def report():
    rows = read_rows()
    if not rows:
        print("no results yet")
        return

    def final(r):
        return r["by_k"][str(max(map(int, r["by_k"])))]

    print("== final checkpoint, sorted by TPR@1%FPR (mean ± sd over 10 repeats) ==")
    for r in sorted(rows.values(), key=lambda r: -final(r)["tpr1"]):
        f = final(r)
        print(f"{r['name']:28s} n={r['n_estimators']:<5d} AUC {f['auc']:.4f}±{f['auc_sd']:.4f}  "
              f"TPR@1% {f['tpr1']:.3f}±{f['tpr1_sd']:.3f}")
    print("\nreference:")
    for name, auc, tpr in REFERENCE:
        print(f"  {name:40s} AUC {auc:.4f}  TPR@1% {tpr:.3f}")

    print("\n== bagged vs the same learner unbagged (paired over 10 repeats) ==")
    for r in rows.values():
        ref = rows.get(f"{r['base']} | none")
        if r["scheme"] == "none" or ref is None:
            continue
        f, g = final(r), final(ref)
        dauc = np.subtract(f["aucs"], g["aucs"])
        dtpr = np.subtract(f["tprs"], g["tprs"])
        print(f"{r['name']:28s} dAUC {dauc.mean():+.4f} ({(dauc > 0).sum()}/10)  "
              f"dTPR {dtpr.mean():+.3f} ({(dtpr > 0).sum()}/10)")

    print("\n== convergence: AUC / TPR@1% by number of members ==")
    for r in rows.values():
        if r["scheme"] == "none":
            continue
        cells = [f"{k}: {v['auc']:.4f}/{v['tpr1']:.3f}"
                 for k, v in sorted(r["by_k"].items(), key=lambda kv: int(kv[0]))]
        print(f"{r['name']:28s} " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    if args.report:
        report()
        return
    set_low_priority()
    X, y = load()
    done = read_rows()
    todo = [c for c in configs() if args.only in c[0] and c[0] not in done]
    print(f"{len(todo)} configs to run on {N_JOBS} workers", flush=True)
    for c in todo:
        run(*c, X, y)


if __name__ == "__main__":
    main()
