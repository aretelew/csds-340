"""Exp 4: naive Bayes variants. All scored by log posterior odds (no saturation ties)."""
import numpy as np
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from ntcommon import JLL, FeatX, ZInfNB, run

x1000 = lambda: FunctionTransformer(lambda Z: 1000 * Z)

# reference: GaussianNB via predict_proba (saturating) vs log-odds, on raw mean-imputed
run(make_pipeline(FeatX(tf="id"), GaussianNB()), "GNB proba | id mean", "NB")
run(make_pipeline(FeatX(tf="id"), JLL(GaussianNB())), "GNB logodds | id mean", "NB")
for c in [100, 1e4]:
    run(make_pipeline(FeatX(tf="log", c=c), JLL(GaussianNB())), f"GNB logodds | log{c:g}", "NB")
run(make_pipeline(FeatX(tf="quant"), JLL(GaussianNB())), "GNB logodds | quantile", "NB")

for imp in ["zero", "mean"]:
    run(make_pipeline(FeatX(tf="pres", impute=imp), JLL(BernoulliNB(alpha=1.0, binarize=None))),
        f"BernoulliNB a1 | pres {imp}-imputed", "NB")
for imp in ["zero", "mean"]:
    for a in [0.1, 1.0]:
        run(make_pipeline(FeatX(tf="id", impute=imp, space="raw"), x1000(), JLL(MultinomialNB(alpha=a))),
            f"MultinomialNB a{a} | 1000x {imp}-imp", "NB")
        run(make_pipeline(FeatX(tf="id", impute=imp, space="raw"), x1000(), JLL(ComplementNB(alpha=a))),
            f"ComplementNB a{a} | 1000x {imp}-imp", "NB")

run(ZInfNB(gaussian=False), "Bernoulli NB missing-marginalised", "NB")
for vs in [0.0, 0.1, 0.5]:
    run(ZInfNB(var_smooth=vs), f"ZeroInfl-LogNormal NB vs{vs} (marginalised)", "NB")
