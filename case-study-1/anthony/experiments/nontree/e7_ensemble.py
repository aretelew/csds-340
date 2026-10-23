"""Exp 7: OOF scores for the finalists, rank-average ensembles (pooled OOF, same splits),
a proper fit-on-train-only score-average estimator, and paired bootstrap noise estimates."""
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import SVC

from e6b_gpc_mlp import SeedBag
from harness import to_nan, tpr_at_fpr
from ntcommon import RESULTS, FeatX, ScoreAvg, ZInfNB, X, run, y

LOG = lambda: FeatX(tf="log", c=1e4)
M = {
    "A_LR": make_pipeline(LOG(), StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
    "B_GAM": make_pipeline(LOG(), SplineTransformer(n_knots=3, degree=2, knots="uniform", extrapolation="linear"),
                           StandardScaler(), LogisticRegression(C=0.01, max_iter=5000)),
    "C_MLPbag": make_pipeline(LOG(), StandardScaler(), SeedBag(
        MLPClassifier(hidden_layer_sizes=(16,), alpha=30.0, solver="lbfgs", max_iter=1000, random_state=0), n=5)),
    "D_SVC": make_pipeline(FeatX(tf="sqrt"), StandardScaler(), SVC(C=10.0, gamma=0.003)),
    "E_ZINB": ZInfNB(var_smooth=0.1),
    "HGB_ref": make_pipeline(to_nan(), HistGradientBoostingClassifier(random_state=0)),
}
# don't re-log the finalists into the results table (they're already there); just get OOF
import ntcommon
ntcommon.RESULTS = RESULTS.replace("results.jsonl", "results_finalists.jsonl")
oof = {k: np.array(run(m, k, "finalist", oof=True)["oof"]) for k, m in M.items()}


def metrics(S):
    a = [roc_auc_score(y, s) for s in S]
    t = [tpr_at_fpr(y, s) for s in S]
    p = [roc_auc_score(y, s, max_fpr=0.05) for s in S]
    return np.mean(a), np.std(a), np.mean([u[0] for u in t]), np.mean([u[1] for u in t]), np.mean(p)


def rankavg(keys):
    return np.array([np.mean([rankdata(oof[k][r]) for k in keys], 0) for r in range(5)])


ntcommon.RESULTS = RESULTS
import json
combos = [("B_GAM", "C_MLPbag"), ("B_GAM", "C_MLPbag", "D_SVC"), ("B_GAM", "C_MLPbag", "E_ZINB"),
          ("A_LR", "C_MLPbag", "E_ZINB"), ("B_GAM", "C_MLPbag", "D_SVC", "E_ZINB")]
print("\nrank-average ensembles (pooled OOF):")
for ks in combos + [("B_GAM", "C_MLPbag", "HGB_ref")]:
    a, sd, ti, ts, p = metrics(rankavg(ks))
    name = "RankAvg " + "+".join(ks)
    print(f"{name:45s} AUC {a:.4f}±{sd:.4f}  TPR@1% {ti:.3f} (step {ts:.3f})  pAUC.05 {p:.4f}")
    with open(RESULTS, "a") as f:
        f.write(json.dumps(dict(name=name, family="ensemble" if "HGB_ref" not in ks else "ensemble(+tree, ref only)",
                                auc=a, auc_sd=sd, tpr1=ti, tpr1_step=ts, pauc05=p)) + "\n")

# proper estimator: z-scored decision averaging, stats from training fold only
run(ScoreAvg([M["B_GAM"], M["C_MLPbag"], M["E_ZINB"]]), "ScoreAvg(z) GAM+MLPbag+ZINB (estimator)", "ensemble")
run(ScoreAvg([M["B_GAM"], M["C_MLPbag"]]), "ScoreAvg(z) GAM+MLPbag (estimator)", "ensemble")

# paired bootstrap on repeat-0 pooled OOF: sampling noise of each metric and of differences
rng = np.random.default_rng(0)
B = 2000
idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]
cand = dict(oof)
cand["Ens_GAM+MLP+ZINB"] = rankavg(("B_GAM", "C_MLPbag", "E_ZINB"))


def boot(k):
    s = cand[k][0]
    return np.array([[roc_auc_score(y[i], s[i]), tpr_at_fpr(y[i], s[i])[0]] for i in idx])


bs = {k: boot(k) for k in cand}
print("\nbootstrap (repeat-0 OOF, n=1500): metric sd")
for k, v in bs.items():
    print(f"  {k:20s} AUC sd {v[:,0].std():.4f}  TPR@1% sd {v[:,1].std():.3f}")
print("\npaired differences (row - col): mean [95% CI]")
pairs = [("B_GAM", "A_LR"), ("C_MLPbag", "A_LR"), ("E_ZINB", "A_LR"), ("D_SVC", "A_LR"),
         ("A_LR", "HGB_ref"), ("B_GAM", "HGB_ref"), ("C_MLPbag", "HGB_ref"), ("Ens_GAM+MLP+ZINB", "HGB_ref"),
         ("Ens_GAM+MLP+ZINB", "B_GAM"), ("Ens_GAM+MLP+ZINB", "C_MLPbag")]
for a, b in pairs:
    d = bs[a] - bs[b]
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    print(f"  {a:18s} - {b:10s} AUC {d[:,0].mean():+.4f} [{lo[0]:+.4f},{hi[0]:+.4f}]   "
          f"TPR@1% {d[:,1].mean():+.3f} [{lo[1]:+.3f},{hi[1]:+.3f}]")
