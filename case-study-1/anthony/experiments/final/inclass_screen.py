"""Screen the in-class model families that the earlier screens did not cover
(or covered only loosely): perceptron, adaline / linear regression, linear
SVM, kernel SVM, kNN, a single decision tree, random forest, bagging and
AdaBoost. 10-repeat harness, outer CV only; the best config per family goes
into models.MODELS for nested caching and ensembling.

    uv run python -m experiments.final.inclass_screen
"""
import json
import time
import warnings
from pathlib import Path

from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Perceptron, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from experiments.final.models import Feat
from experiments.harness import evaluate, fmt, load, to_nan

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "inclass_screen.jsonl"


def lin(est):
    return make_pipeline(Feat("log"), StandardScaler(), est)


def tree(**kw):
    return DecisionTreeClassifier(random_state=0, **kw)


CONFIGS = [
    ("perceptron", lin(Perceptron(random_state=0))),
    *[(f"perceptron avg a{a}", lin(SGDClassifier(loss="perceptron", penalty="l2", alpha=a,
                                                  average=True, max_iter=2000, random_state=0)))
      for a in (1e-4, 1e-3, 1e-2, 1e-1)],
    *[(f"adaline (ridge) a{a}", lin(RidgeClassifier(alpha=a))) for a in (1e-6, 10, 100, 1000)],
    *[(f"linsvc C{C}", lin(LinearSVC(C=C, max_iter=20000))) for C in (0.001, 0.01, 0.1, 1)],
    *[(f"svc rbf C{C} g{g} log", lin(SVC(C=C, gamma=g)))
      for C, g in ((3, 0.003), (10, 0.003), (30, 0.001), (3, 0.01))],
    *[(f"knn k{k} {w}", lin(KNeighborsClassifier(n_neighbors=k, weights=w)))
      for k in (51, 101, 151, 201) for w in ("distance",)],
    *[(f"tree d{d} msl{m} NaN", make_pipeline(to_nan(), tree(max_depth=d, min_samples_leaf=m)))
      for d in (4, 6, 8) for m in (5, 20)],
    *[(f"rf 1000 {mf} msl{m} NaN", make_pipeline(to_nan(), RandomForestClassifier(
        n_estimators=1000, max_features=mf, min_samples_leaf=m, random_state=0, n_jobs=1)))
      for mf in ("sqrt", 2) for m in (1, 3, 5)],
    *[(f"bagged trees 500 msl{m} NaN", make_pipeline(to_nan(), BaggingClassifier(
        tree(min_samples_leaf=m), n_estimators=500, random_state=0, n_jobs=1)))
      for m in (1, 5)],
    *[(f"adaboost d{d} n{n} lr{lr} raw-1", AdaBoostClassifier(
        tree(max_depth=d), n_estimators=n, learning_rate=lr, random_state=0))
      for d, n, lr in ((1, 500, 0.5), (2, 500, 0.1), (2, 500, 0.3), (3, 300, 0.1), (3, 500, 0.05))],
    ("adaboost d2 n500 lr0.1 mean", make_pipeline(
        SimpleImputer(missing_values=-1, strategy="mean"),
        AdaBoostClassifier(tree(max_depth=2), n_estimators=500, learning_rate=0.1, random_state=0))),
]

if __name__ == "__main__":
    X, y = load()
    for name, model in CONFIGS:
        t = time.time()
        r = evaluate(model, X, y, name=name)
        print(fmt(r), f"[{time.time() - t:.0f}s]", flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v)
                                for k, v in r.items()}) + "\n")
