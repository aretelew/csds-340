"""Exp 6: how big is the data-sampling noise (what a held-out set would see),
vs the CV-split noise reported by the harness? Paired bootstrap over emails of the
pooled OOF scores (500 resamples; per resample, metric diff averaged over the 5
repeats). Reports the bootstrap SE and 95% interval of (A - B)."""
import numpy as np
from sklearn.metrics import roc_auc_score

from common import load, load_oof
from harness import tpr_at_fpr

_, y = load()
REF = "HGB lr.03 it800 L4 msl10 l2=1 mf.5"
PAIRS = [
    (REF, "HGB default (lr.1 it100 L31 msl20)"),
    ("LGBM L4 lr.03 n800 mcs10 ss.8 cs.5", REF),
    ("XGB d3 lr.03 n600 ss.8 cs.8", REF),
    ("ET 1000 sqrt msl1 NaN", REF),
    ("RankAvg HGB+ETmean3", REF),
    ("RankAvg HGB+ETnan", REF),
    ("RankAvg HGB+LGBM+ETnan", REF),
]
rng = np.random.default_rng(0)
B = 500
idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]


def metrics(yb, s):
    return roc_auc_score(yb, s), tpr_at_fpr(yb, s)[0]


print("single-model bootstrap SE (REF):", end=" ")
A = load_oof(REF)
m = np.array([[metrics(y[i], A[r][i]) for r in range(5)] for i in idx]).mean(1)
print(f"AUC SE {m[:, 0].std():.4f}, TPR@1% SE {m[:, 1].std():.3f}")

for a, b in PAIRS:
    A, Bo = load_oof(a), load_oof(b)
    d = []
    for i in idx:
        da = np.mean([metrics(y[i], A[r][i])[0] - metrics(y[i], Bo[r][i])[0]
                      for r in range(5)])
        dt = np.mean([metrics(y[i], A[r][i])[1] - metrics(y[i], Bo[r][i])[1]
                      for r in range(5)])
        d.append((da, dt))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    print(f"{a:38s} - {b[:24]:24s} dAUC {d[:, 0].mean():+.4f} (SE {d[:, 0].std():.4f}, "
          f"95% [{lo[0]:+.4f},{hi[0]:+.4f}])  dTPR1 {d[:, 1].mean():+.3f} "
          f"(SE {d[:, 1].std():.3f}, [{lo[1]:+.3f},{hi[1]:+.3f}])")
