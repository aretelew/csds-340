"""Check that the submission (anthony/classifySpam.py) reproduces the cached
nested-CV ensemble scores of its MEMBERS exactly, on a few outer folds of
repeat 0.

    uv run python -m experiments.final.check_predict
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold

import classifySpam as P
from experiments.final import ensemble as E
from experiments.harness import load

X, y = load()
expected = np.mean([E.pct_scores(n)[0] for n in P.MEMBERS], axis=0)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
for f, (tr, te) in enumerate(cv.split(X, y)):
    if f == 3:
        break
    got = P.predictTest(X[tr], y[tr], X[te])
    one = P.predictTest(X[tr], y[tr], X[te[:1]])
    print(f"fold {f}: max |predictTest - cached| = {np.abs(got - expected[te]).max():.2e}; "
          f"single-email batch matches: {np.isclose(one[0], got[0])}")
