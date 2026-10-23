# Final comparison: 10-repeat harness + deployable ensembles

The best configs from `trees/` and `nontree/` (screened with the 5-repeat
`harness_v1`), rerun on `../harness.py` (10 repeats of stratified 10-fold, the
same splits for every model) and combined into ensembles that `predictTest`
can actually deploy.

## How it works

- `models.py`: the candidates, all fed raw features (missing = -1).
- `nested.py`: for each repeat and outer fold it caches the outer test scores
  and the inner 5-fold OOF scores on that fold's training part
  (`cache/<name>.npz`, gitignored, about 1 MB each).
- `ensemble.py`: a deployable ensemble cannot rank across the test batch, so
  each member's test score is mapped to a percentile of that member's OOF
  scores on the training set (`pct_map`), and the percentiles are averaged
  with equal weights. `pct_map` is strictly increasing with rational tails
  past the training min/max, so nothing ties at the top. An exponential tail
  was tried first and rounded to exactly 1.0 only ~0.2 HGB-logit past the
  training max.
- `report.py` produces `report.txt`: every single model, every equal-weight
  combo of 2 to 4 members (312 candidates), the selection rule, paired
  compares, and a split-half check of the selection itself.
- `predict_test.py`: a self-contained `predictTest` (numpy + sklearn only).
  `check_predict.py` confirms it reproduces the cached ensemble scores
  (max abs diff 1e-16 on the folds checked) and that an email scored alone
  gets the same score as in a batch.

Run from `case-study-1/anthony/`:

```
uv run python -m experiments.final.nested hgb et_nan mlp spline_lr ...   # ~40 min for all 10
uv run python -m experiments.final.report > experiments/final/report.txt
```

## Results (10 repeats; mean ± sd over repeats)

Single models, raw pooled OOF (what `harness.evaluate` reports):

| model | AUC | TPR@1%FPR |
|---|---|---|
| spline_lr (log1p(1e4x), 3 knots deg 2, C=0.01) | 0.9249 ± 0.0008 | 0.495 ± 0.010 |
| mlp_bag5 | 0.9236 ± 0.0009 | 0.562 ± 0.023 |
| mlp ((16,), lbfgs, alpha=30, log1p(1e4x)) | 0.9234 ± 0.0010 | 0.563 ± 0.023 |
| lr (L2, C=0.1, log1p(1e4x)) | 0.9223 ± 0.0008 | 0.529 ± 0.028 |
| hgb (lr .03, 800 it, 4 leaves, mf .5) | 0.9214 ± 0.0013 | 0.498 ± 0.033 |
| hgb_bag5 | 0.9213 ± 0.0013 | 0.500 ± 0.028 |
| lgbm (4 leaves, 800) | 0.9189 ± 0.0015 | 0.517 ± 0.022 |
| svc (rbf, C=10, gamma=.01, sqrt) | 0.9186 ± 0.0007 | 0.534 ± 0.019 |
| et_mean3 | 0.9161 ± 0.0015 | 0.514 ± 0.030 |
| et_nan | 0.9076 ± 0.0015 | 0.534 ± 0.013 |

Ensembles (equal-weight percentile average): the best AUC is 0.9274
(`et_mean3+hgb+mlp_bag5+spline_lr`, but TPR only 0.506), so the bar is
AUC >= 0.9244. 194 of the 312 candidates clear it. The top of the TPR ranking:

| ensemble | AUC | TPR@1%FPR |
|---|---|---|
| lgbm+lr+mlp_bag5+spline_lr (rule winner) | 0.9260 ± 0.0007 | 0.571 ± 0.033 |
| et_nan+lgbm+mlp_bag5+spline_lr | 0.9246 ± 0.0008 | 0.570 ± 0.014 |
| et_nan+hgb+mlp+spline_lr (best sklearn-only) | 0.9252 ± 0.0008 | 0.568 ± 0.012 |
| hgb_bag5+lr+mlp+spline_lr | 0.9264 ± 0.0007 | 0.567 ± 0.026 |
| mlp alone, mapped (fails the AUC bar by 0.0006) | 0.9238 ± 0.0009 | 0.567 ± 0.022 |
| hgb alone, mapped | 0.9218 ± 0.0011 | 0.510 ± 0.036 |

Paired over the 10 repeats:
- The rule winner beats `et_nan+hgb+mlp+spline_lr` by +0.003 TPR (7/10)
  and +0.0008 AUC (10/10).
- It beats hgb alone by +0.062 TPR (9/10).
- Against the MLP alone it wins on AUC (+0.002, 10/10) but not on TPR
  (+0.005, 4/10).

## What the numbers say

- **The MLP carries TPR@1%.** Every top ensemble contains it, and none beats
  it on TPR in most repeats. The ensembles add about +0.002 AUC on top.
- **The top ~15 ensembles are a statistical tie** (TPR 0.565 to 0.571). The
  repeats share one dataset, and the paired bootstrap noise on a 1500-email
  test set is about ±0.07 TPR. The rule winner is mostly the luckiest of 312
  near-equal candidates.
- **Split-half check.** Picking the winner with half the labels and scoring
  it on the other half (40 halves) gave held-out TPR 0.534. Fixed choices did
  as well or better: mlp 0.555, the average of all 8 distinct models 0.546,
  hgb 0.516. Searching combos buys nothing beyond "include the MLP plus a few
  diverse models".
- Percentile mapping slightly *helps* pooled single-model metrics (it puts
  each fold on one scale). Seed bagging changes nothing measurable.

## TabPFN v2 (local test only)

Open v2 weights (`Prior-Labs/TabPFN-v2-clf`, not gated, no login), tabpfn
9.2.0 + torch 2.14 CPU. These live in a separate venv, not in pyproject:

```
uv venv tabpfn-venv && VIRTUAL_ENV=tabpfn-venv uv pip install torch tabpfn "scikit-learn>=1.9,<1.10" \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
tabpfn-venv/bin/python -m experiments.final.nested tabpfn --no-inner --jobs 1   # 46 min on 4 cores
```

Above 1000 training rows it refuses to run on CPU unless
`ignore_pretraining_limits=True`. One fit+predict on 1350 rows takes ~28 s on
4 cores. Results, outer CV only (no inner refs; computing them would take
~4.5 h):

| model | AUC | TPR@1%FPR |
|---|---|---|
| tabpfn, raw pooled | 0.9263 ± 0.0012 | 0.576 ± 0.014 |

- **Best single model.** It beats the MLP (TPR +0.013, wins 7/10; AUC wins
  10/10) and HGB (wins 10/10 on both).
- **It does not beat the best ensemble.** For ensembles we used a shortcut:
  each fold's percentile reference is the other 9 folds' outer scores, with
  no inner fits. On known ensembles this lands within about 0.001 AUC and
  0.005 TPR of the exact nested result. TabPFN alone, mapped this way, scores
  AUC 0.9260 / TPR 0.570, against 0.9260 / 0.571 for
  `lgbm+lr+mlp_bag5+spline_lr`. TabPFN wins 3/10 on TPR and 4/10 on AUC.
- **Adding sklearn members doesn't help.** The best TabPFN ensembles reach
  TPR ~0.575 at AUC 0.924 to 0.926.

Every strong approach converges near AUC 0.926 and TPR 0.57. Deploying
TabPFN would need torch and tabpfn on the grader's machine, an internet
download of the weights, and about 1 min of CPU time. Since code that errors
scores 0, that risk buys nothing measurable.
