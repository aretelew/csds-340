"""Exp 2c: n_estimators (500 / 2000) and one smaller max_features (3) check,
since sqrt(30)~5 beat 0.3 and 0.5 everywhere. 6 configs."""
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.ensemble import RandomForestClassifier as RF

from common import run
from e2_forests import forest

configs = [
    ("ET 500 sqrt msl1 NaN", forest(ET, "NaN", n=500)),
    ("ET 2000 sqrt msl1 NaN", forest(ET, "NaN", n=2000)),
    ("ET 2000 sqrt msl1 mean", forest(ET, "mean", n=2000)),
    ("ET 1000 mf3 msl1 NaN", forest(ET, "NaN", mf=3)),
    ("ET 1000 mf3 msl1 mean", forest(ET, "mean", mf=3)),
    ("RF 2000 sqrt msl3 NaN", forest(RF, "NaN", n=2000, msl=3)),
]

if __name__ == "__main__":
    run(configs, "forest")
