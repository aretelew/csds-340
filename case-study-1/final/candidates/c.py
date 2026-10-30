# -*- coding: utf-8 -*-
"""
Spam classifier for CSDS 340 Case Study 1: an equal-weight ensemble of
logistic regression and a random forest.

The models score on different scales, so each model's test scores are
converted to z-scores using the mean and standard deviation of its 5-fold
cross-validation scores on the training set. The final score is the average
of the two z-scores.

Requires scikit-learn 1.4 or newer (random forest with missing values).
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

MODEL_NAMES = ["logistic_regression", "random_forest"]


def replaceMissingWithNan(features):
    features = np.array(features, dtype=float, copy=True)
    features[features == -1] = np.nan
    return features


def logTransform(features):
    features = replaceMissingWithNan(features)
    return np.log1p(10000 * features)


class LogTransformAndFillMissing(BaseEstimator, TransformerMixin):
    """Log-transform the features and fill missing values with the training
    mean of each (log-transformed) feature."""

    def fit(self, X, y=None):
        logFeatures = logTransform(X)
        self.columnMeans_ = np.nanmean(logFeatures, axis=0)
        return self

    def transform(self, X):
        logFeatures = logTransform(X)
        isMissing = np.isnan(logFeatures)
        filledFeatures = np.where(isMissing, self.columnMeans_, logFeatures)
        return filledFeatures


def makeModel(name):
    if name == "logistic_regression":
        return make_pipeline(
            LogTransformAndFillMissing(),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=5000),
        )

    if name == "svm":
        return make_pipeline(
            LogTransformAndFillMissing(),
            StandardScaler(),
            SVC(C=50.0, gamma=0.002),
        )

    if name == "random_forest":
        # One random feature per split (max_features=1) beat sqrt and 2 in CV
        return make_pipeline(
            FunctionTransformer(replaceMissingWithNan),
            RandomForestClassifier(
                n_estimators=3000,
                max_features=1,
                min_samples_leaf=1,
                random_state=0,
                n_jobs=-1,
            ),
        )

    raise ValueError(f"Unknown model name: {name}")


def getSpamScores(name, model, features):
    if name == "random_forest":
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def getCrossValidationScores(name, trainFeatures, trainLabels):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cvScores = np.empty(len(trainLabels))

    for fitIndices, heldOutIndices in folds.split(trainFeatures, trainLabels):
        model = makeModel(name)
        model.fit(trainFeatures[fitIndices], trainLabels[fitIndices])
        heldOutScores = getSpamScores(name, model, trainFeatures[heldOutIndices])
        cvScores[heldOutIndices] = heldOutScores

    return cvScores


def predictTest(trainFeatures, trainLabels, testFeatures):
    trainFeatures = np.asarray(trainFeatures, dtype=float)
    trainLabels = np.asarray(trainLabels)

    zScoresForEachModel = []
    for name in MODEL_NAMES:
        cvScores = getCrossValidationScores(name, trainFeatures, trainLabels)
        cvMean = cvScores.mean()
        cvStd = cvScores.std()

        model = makeModel(name)
        model.fit(trainFeatures, trainLabels)
        testScores = getSpamScores(name, model, testFeatures)

        zScores = (testScores - cvMean) / cvStd
        zScoresForEachModel.append(zScores)

    finalScores = np.mean(zScoresForEachModel, axis=0)
    return finalScores


# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    from pathlib import Path
    from sklearn.metrics import roc_auc_score, roc_curve

    def tprAtFPR(labels, outputs, desiredFPR=0.01):
        """Same calculation as evaluateClassifier.py."""
        fpr, tpr, thres = roc_curve(labels, outputs)
        maxFprIndex = np.where(fpr <= desiredFPR)[0][-1]
        fprBelow = fpr[maxFprIndex]
        fprAbove = fpr[maxFprIndex + 1]
        tprBelow = tpr[maxFprIndex]
        tprAbove = tpr[maxFprIndex + 1]
        tprAt = ((tprAbove - tprBelow) / (fprAbove - fprBelow)
                 * (desiredFPR - fprBelow) + tprBelow)
        return tprAt

    dataPath = Path(__file__).resolve().parent.parent / "spamTrain1.csv"
    if not dataPath.exists():
        dataPath = Path("spamTrain1.csv")
    data = np.loadtxt(dataPath, delimiter=",")
    features = data[:, :-1]
    labels = data[:, -1]

    outputs = np.empty(len(labels))
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for trainIndices, testIndices in folds.split(features, labels):
        outputs[testIndices] = predictTest(features[trainIndices],
                                           labels[trainIndices],
                                           features[testIndices])

    print(f"10-fold CV AUC: {roc_auc_score(labels, outputs):.4f}")
    print(f"10-fold CV TPR at FPR = 0.01: {tprAtFPR(labels, outputs):.4f}")
