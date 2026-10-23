"""Exp 6b: GPC (score = Laplace latent mean; sklearn 1.9.1 predict_proba returns NaN here because
the latent variance comes out negative)
and MLP follow-up at stronger alpha + seed bagging."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ntcommon import FeatX, run


class GPCScore(BaseEstimator, ClassifierMixin):
    def __init__(self, gpc):
        self.gpc = gpc

    def fit(self, X, y):
        self.g_ = clone(self.gpc).fit(X, y)
        self.classes_ = self.g_.classes_
        return self

    def decision_function(self, X):
        mu, var = self.g_.base_estimator_.latent_mean_and_variance(X)
        return mu  # latent variance comes out negative (numerically unstable) -> rank by latent mean


class SeedBag(BaseEstimator, ClassifierMixin):
    def __init__(self, est, n=5):
        self.est = est
        self.n = n

    def fit(self, X, y):
        self.ms_ = [clone(self.est).set_params(random_state=s).fit(X, y) for s in range(self.n)]
        self.classes_ = self.ms_[0].classes_
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.ms_], 0)


F = lambda: FeatX(tf="log", c=1e4)
mlp = lambda h, a: MLPClassifier(hidden_layer_sizes=h, alpha=a, solver="lbfgs", max_iter=1000, random_state=0)

import sys
GPC_ONLY = "--gpc-only" in sys.argv
for h, a in ([] if GPC_ONLY else [((16,), 60.0), ((16,), 100.0), ((64,), 100.0)]):
    run(make_pipeline(F(), StandardScaler(), mlp(h, a)), f"MLP{h} lbfgs alpha{a} | log1e4", "MLP")
GPC_ONLY or run(make_pipeline(F(), StandardScaler(), SeedBag(mlp((16,), 30.0), n=5)),
    "MLP(16,) alpha30 x5-seed bag | log1e4", "MLP")
run(make_pipeline(F(), StandardScaler(),
                  GPCScore(GaussianProcessClassifier(ConstantKernel(1.0) * RBF(length_scale=10.0), random_state=0))),
    "GPC C*RBF (ML-II) | log1e4", "GPC")
