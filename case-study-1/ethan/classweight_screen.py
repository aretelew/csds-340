"""Class-weight screen: does up-weighting non-spam (class 0) in training help
TPR at 1% FPR, which is decided by the few most spam-like real emails?

Fixed in advance:
  Weights. class_weight={0: w, 1: 1} with w in WEIGHTS; w=1 is the unweighted
  model, read from the bagging/ensemble screen caches.
  Members. Every member of the ensembles below that accepts class_weight:
  Anthony's rf, lr, svc and the bagged lr, svc, tree picked by
  ensemble_screen.py. The bagged MLP has no class_weight and stays as is.
  Bags reuse random_state=0, so weighted and unweighted bags draw the same
  samples and features and the comparison is paired.
  Combos. The same weight for every weighted member of a combo.
  Combining and decision rule: as in ensemble_screen.py.

    uv run python classweight_screen.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold

from bagging_screen import BASES, N_JOBS, SCHEMES, set_low_priority
from common import N_REPEATS, N_SPLITS, continuous_score, load
from ensemble_screen import ANTHONY, CACHE as ENS_CACHE, SCREEN_CACHE, mapped, metrics, splits

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "classweight_screen"
OUT = HERE / "results" / "classweight_screen.txt"
WEIGHTS = (1, 2, 4)

# key -> (factory, class_weight parameter, bagging scheme or None, members,
#         cached unweighted OOF file)
MEMBERS = {
    "A_rf": (ANTHONY["A_rf"], "randomforestclassifier__class_weight", None, 1,
             ENS_CACHE / "A_rf.npy"),
    "A_lr": (ANTHONY["A_lr"], "logisticregression__class_weight", None, 1,
             ENS_CACHE / "A_lr.npy"),
    "A_svc": (ANTHONY["A_svc"], "svc__class_weight", None, 1, ENS_CACHE / "A_svc.npy"),
    "E_lr": (BASES["lr_C1"][0], "logisticregression__class_weight", "subspace", 500,
             SCREEN_CACHE / "lr_C1__subspace.npy"),
    "E_svc": (BASES["svc_C100_g.01"][0], "svc__class_weight", "both", 300,
              SCREEN_CACHE / "svc_C100_gp01__both.npy"),
    "E_tree": (BASES["tree_msl1"][0], "class_weight", "subspace", 1000,
               SCREEN_CACHE / "tree_msl1__subspace.npy"),
}
UNWEIGHTED = {"E_mlp": SCREEN_CACHE / "mlp_h32_a10__subspace.npy"}

COMBOS = {
    "anthony: rf+lr+svc": ("A_rf", "A_lr", "A_svc"),
    "rf + bagged lr + bagged svc": ("A_rf", "E_lr", "E_svc"),
    "bagged mlp+svc+tree+lr": ("E_mlp", "E_svc", "E_tree", "E_lr"),
}
BASELINE = ("anthony: rf+lr+svc", 1)


def _fit_score(key, w, X, y, tr, te):
    warnings.filterwarnings("ignore")
    set_low_priority()
    make, param, scheme, n, _ = MEMBERS[key]
    model = make().set_params(**{param: {0: w, 1: 1}})
    if scheme is None:
        return te, continuous_score(model.fit(X[tr], y[tr]), X[te])
    bag = BaggingClassifier(model, n_estimators=n, random_state=0, n_jobs=1,
                            **SCHEMES[scheme]).fit(X[tr], y[tr])
    return te, np.mean([continuous_score(est, X[te][:, feats])
                        for est, feats in zip(bag.estimators_, bag.estimators_features_)],
                       axis=0)


def oof(key, w, X, y):
    if w == 1:
        return np.load(MEMBERS[key][4])
    path = CACHE / f"{key}_w{w}.npy"
    if path.exists():
        return np.load(path)
    t0 = time.time()
    tasks = [(r, tr, te) for r in range(N_REPEATS)
             for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=r).split(X, y)]
    res = Parallel(n_jobs=N_JOBS)(delayed(_fit_score)(key, w, X, y, tr, te)
                                  for _, tr, te in tasks)
    out = np.empty((N_REPEATS, len(y)))
    for (r, _, _), (te, s) in zip(tasks, res):
        out[r, te] = s
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    print(f"  fitted {key} w={w} ({time.time() - t0:.0f}s)", flush=True)
    return out


def main():
    set_low_priority()
    X, y = load()
    fold = splits(y)
    pct = {(k, w): mapped(oof(k, w, X, y), fold) for k in MEMBERS for w in WEIGHTS}
    for k, path in UNWEIGHTED.items():
        p = mapped(np.load(path), fold)
        pct.update({(k, w): p for w in WEIGHTS})

    lines = ["single members by non-spam weight w (percentile-mapped, mean over 10 repeats):"]
    for k in MEMBERS:
        cells = []
        for w in WEIGHTS:
            a, t = metrics(pct[(k, w)], y)
            cells.append(f"w={w}: {a.mean():.4f}/{t.mean():.3f}")
        lines.append(f"  {k:8s} " + "   ".join(cells))

    res = {(name, w): metrics(np.mean([pct[(m, w)] for m in members], axis=0), y)
           for name, members in COMBOS.items() for w in WEIGHTS}
    base_a, base_t = res[BASELINE]
    lines += ["", "combos, paired vs anthony w=1 (wins out of 10):"]
    for (name, w), (a, t) in res.items():
        line = (f"  {name:28s} w={w}  AUC {a.mean():.4f}±{a.std():.4f}  "
                f"TPR@1% {t.mean():.3f}±{t.std():.3f}")
        if (name, w) != BASELINE:
            da, dt = a - base_a, t - base_t
            verdict = "BEATS" if dt.mean() > 0 and (dt > 0).sum() > N_REPEATS / 2 else "-"
            line += (f"   dAUC {da.mean():+.4f} ({(da > 0).sum()}/10)  "
                     f"dTPR {dt.mean():+.3f} ({(dt > 0).sum()}/10)  {verdict}")
        lines.append(line)

    text = "\n".join(lines)
    print(text)
    OUT.write_text(text + "\n")


if __name__ == "__main__":
    main()
