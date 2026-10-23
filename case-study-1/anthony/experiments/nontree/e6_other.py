"""Exp 6: kNN, LDA/QDA, Gaussian process classifier, MLP (all on log1e4 + standardize)."""
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ntcommon import FeatX, run

F = lambda **kw: FeatX(**({"tf": "log", "c": 1e4} | kw))

for k in [15, 31, 61, 101]:
    for w in ["uniform", "distance"]:
        run(make_pipeline(F(), StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights=w)),
            f"kNN k{k} {w} | log1e4", "kNN")
run(make_pipeline(FeatX(tf="sqrt"), StandardScaler(), KNeighborsClassifier(n_neighbors=61, weights="distance")),
    "kNN k61 distance | sqrt", "kNN")
run(make_pipeline(FeatX(tf="pres", impute="zero"), KNeighborsClassifier(n_neighbors=61, weights="distance", metric="jaccard")),
    "kNN k61 distance jaccard | pres zero-imp", "kNN")

run(make_pipeline(F(), StandardScaler(), LDA()), "LDA | log1e4", "LDA/QDA")
run(make_pipeline(F(), StandardScaler(), LDA(solver="lsqr", shrinkage="auto")), "LDA shrink=auto | log1e4", "LDA/QDA")
run(make_pipeline(FeatX(tf="quant"), StandardScaler(), LDA()), "LDA | quantile", "LDA/QDA")
for rp in [0.1, 0.3, 0.6, 0.9]:
    run(make_pipeline(F(), StandardScaler(), QDA(reg_param=rp)), f"QDA reg{rp} | log1e4", "LDA/QDA")

for h in [(16,), (64,)]:
    for a in [1.0, 10.0, 30.0]:
        run(make_pipeline(F(), StandardScaler(),
                          MLPClassifier(hidden_layer_sizes=h, alpha=a, solver="lbfgs", max_iter=1000, random_state=0)),
            f"MLP{h} lbfgs alpha{a} | log1e4", "MLP")
for a in [1e-3, 1e-1]:
    run(make_pipeline(F(), StandardScaler(),
                      MLPClassifier(hidden_layer_sizes=(64,), alpha=a, early_stopping=True, max_iter=1000,
                                    random_state=0)),
        f"MLP(64,) adam earlystop alpha{a} | log1e4", "MLP")

t = time.time()
run(make_pipeline(F(), StandardScaler(),
                  GaussianProcessClassifier(ConstantKernel(1.0) * RBF(length_scale=10.0), random_state=0)),
    "GPC C*RBF (ML-II) | log1e4", "GPC")
