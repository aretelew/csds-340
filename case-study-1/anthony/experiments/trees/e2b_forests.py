"""Exp 2b: max_features / min_samples_leaf for the 3 best round-A forests
(RF NaN, ET NaN, ET mean). 12 configs; n_estimators checked in e2c."""
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.ensemble import RandomForestClassifier as RF

from common import run
from e2_forests import forest

configs = []
for cname, cls, miss in [("RF", RF, "NaN"), ("ET", ET, "NaN"), ("ET", ET, "mean")]:
    for mf, msl in [(0.3, 1), (0.5, 1), ("sqrt", 3), ("sqrt", 5)]:
        configs.append((f"{cname} 1000 {mf} msl{msl} {miss}",
                        forest(cls, miss, mf=mf, msl=msl)))

if __name__ == "__main__":
    run(configs, "forest")
