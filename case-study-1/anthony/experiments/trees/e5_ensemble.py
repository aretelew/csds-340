"""Exp 5: rank-average ensembles.
Step 1 (screen): average the per-repeat pooled-OOF ranks saved by earlier runs
(identical splits, seeds 0..4) for 6 pre-chosen combos.
Step 2 (confirm): re-evaluate the chosen combos with a real wrapper estimator (Avg),
which fits every member inside each fold and averages within-fold ranks, as
predictTest would."""
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.metrics import roc_auc_score

from common import Avg, load, load_oof, run
from e2_forests import forest
from e3_libs import lgb, xgb
from e4_features import hgb
from harness import tpr_at_fpr

X, y = load()

M = {
    "HGB": ("HGB lr.03 it800 L4 msl10 l2=1 mf.5", lambda: hgb()),
    "XGB": ("XGB d3 lr.03 n600 ss.8 cs.8", lambda: xgb()),
    "LGBM": ("LGBM L4 lr.03 n800 mcs10 ss.8 cs.5", lambda: lgb(num_leaves=4, n_estimators=800)),
    "ETnan": ("ET 1000 sqrt msl1 NaN", lambda: forest(ET, "NaN")),
    "ETmean3": ("ET 1000 mf3 msl1 mean", lambda: forest(ET, "mean", mf=3)),
}
COMBOS = [("HGB", "ETnan"), ("HGB", "ETmean3"), ("HGB", "XGB", "ETnan"),
          ("HGB", "LGBM", "ETnan"), ("HGB", "XGB", "LGBM"), ("HGB", "ETnan", "ETmean3")]


def screen():
    print("OOF-rank screen (pooled OOF ranks averaged, mean over 5 repeats):")
    for combo in COMBOS:
        oofs = [load_oof(M[k][0]) for k in combo]
        aucs, tprs = [], []
        for r in range(5):
            s = np.mean([rankdata(o[r]) for o in oofs], axis=0)
            aucs.append(roc_auc_score(y, s))
            tprs.append(tpr_at_fpr(y, s)[0])
        print(f"  {'+'.join(combo):22s} AUC {np.mean(aucs):.4f}±{np.std(aucs):.4f}  "
              f"TPR@1% {np.mean(tprs):.3f}±{np.std(tprs):.3f}")


def confirm(combos, how="rank"):
    cfg = []
    for combo in combos:
        cfg.append((f"RankAvg {'+'.join(combo)}" if how == "rank" else
                    f"ProbAvg {'+'.join(combo)}",
                    Avg([M[k][1]() for k in combo], how=how)))
    run(cfg, "ensemble", X, y)


if __name__ == "__main__":
    if sys.argv[1] == "screen":
        screen()
    else:
        combos = [tuple(c.split("+")) for c in sys.argv[2:]]
        confirm(combos, how=sys.argv[1])
