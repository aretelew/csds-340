"""Exp 5: RBF SVC (C x gamma grid on two transforms) and Nystroem + LR."""
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ntcommon import FeatX, run

TF = {"sqrt": dict(tf="sqrt"), "log1e4": dict(tf="log", c=1e4)}
for tn, kw in TF.items():
    for g in [0.003, 0.01, 0.03]:
        for C in [0.3, 1.0, 3.0, 10.0]:
            run(make_pipeline(FeatX(**kw), StandardScaler(), SVC(C=C, gamma=g)),
                f"SVC rbf C{C} g{g} | {tn}", "SVM")
for g in [0.003, 0.01]:
    for C in [0.1, 1.0]:
        run(make_pipeline(FeatX(tf="log", c=1e4), StandardScaler(),
                          Nystroem(gamma=g, n_components=500, random_state=0), StandardScaler(),
                          LogisticRegression(C=C, max_iter=5000)),
            f"Nystroem500 g{g} + LR C{C} | log1e4", "SVM")
