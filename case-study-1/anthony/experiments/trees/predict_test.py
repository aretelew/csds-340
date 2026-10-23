"""Draft predictTest for the best sklearn-only tree configs (no xgboost/lightgbm
dependency). VARIANT 'hgb_etmean' = best CV AUC; 'hgb_etnan' = best CV TPR@1%.
Requires scikit-learn >= 1.4 (HGB max_features); 'hgb_etnan' needs >= 1.6
(NaN support in ExtraTrees)."""
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

VARIANT = "hgb_etmean"


def _neg1_to_nan(Z):
    Z = np.array(Z, dtype=float, copy=True)
    Z[Z == -1] = np.nan
    return Z


def _models(variant=VARIANT):
    hgb = make_pipeline(
        FunctionTransformer(_neg1_to_nan),
        HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=800, max_leaf_nodes=4, min_samples_leaf=10,
            l2_regularization=1.0, max_features=0.5, early_stopping=False,
            random_state=0))
    if variant == "hgb_etmean":
        et = make_pipeline(
            SimpleImputer(missing_values=-1, strategy="mean"),
            ExtraTreesClassifier(n_estimators=1000, max_features=3, min_samples_leaf=1,
                                 random_state=0, n_jobs=-1))
    else:
        et = make_pipeline(
            FunctionTransformer(_neg1_to_nan),
            ExtraTreesClassifier(n_estimators=1000, max_features="sqrt",
                                 min_samples_leaf=1, random_state=0, n_jobs=-1))
    return [hgb, et]


def predictTest(trainFeatures, trainLabels, testFeatures):
    scores = []
    for m in _models():
        m.fit(trainFeatures, trainLabels)
        s = m.predict_proba(testFeatures)[:, 1]
        scores.append(rankdata(s) / len(s))
    return np.mean(scores, axis=0)


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    from harness import load, tpr_at_fpr
    X, y = load()
    rng = np.random.default_rng(1)
    p = rng.permutation(len(y))
    tr, te = p[:1000], p[1000:]
    s = predictTest(X[tr], y[tr], X[te])
    print(f"sanity split (1000/500): AUC {roc_auc_score(y[te], s):.4f}  "
          f"TPR@1% {tpr_at_fpr(y[te], s)[0]:.3f}")
