# -*- coding: utf-8 -*-
"""
Spam classifier for CSDS 340 Case Study 1: an equal-weight average of three
logistic regressions that differ only in how the word frequencies are
transformed: log1p(10000x), log1p(100x) and sqrt(x).

The models score on different scales, so each model's test scores are
converted to z-scores using the mean and standard deviation of its 5-fold
cross-validation scores on the training set. The final score is the average
of the three z-scores.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ["log10000", "log100", "sqrt"]


def transform(name, features):
    features = np.array(features, dtype=float, copy=True)
    features[features == -1] = np.nan
    if name == "log10000":
        return np.log1p(10000 * features)
    if name == "log100":
        return np.log1p(100 * features)
    if name == "sqrt":
        return np.sqrt(features)
    raise ValueError(f"Unknown model name: {name}")


class TransformAndFillMissing(BaseEstimator, TransformerMixin):
    """Transform the features and fill missing values with the training mean
    of each transformed feature."""

    def __init__(self, name="log10000"):
        self.name = name

    def fit(self, X, y=None):
        self.columnMeans_ = np.nanmean(transform(self.name, X), axis=0)
        return self

    def transform(self, X):
        transformed = transform(self.name, X)
        return np.where(np.isnan(transformed), self.columnMeans_, transformed)


def makeModel(name):
    return make_pipeline(
        TransformAndFillMissing(name),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=5000),
    )


def getCrossValidationScores(name, trainFeatures, trainLabels):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cvScores = np.empty(len(trainLabels))

    for fitIndices, heldOutIndices in folds.split(trainFeatures, trainLabels):
        model = makeModel(name)
        model.fit(trainFeatures[fitIndices], trainLabels[fitIndices])
        cvScores[heldOutIndices] = model.decision_function(
            trainFeatures[heldOutIndices])

    return cvScores


def predictTest(trainFeatures, trainLabels, testFeatures):
    trainFeatures = np.asarray(trainFeatures, dtype=float)
    trainLabels = np.asarray(trainLabels)

    zScoresForEachModel = []
    for name in MODEL_NAMES:
        cvScores = getCrossValidationScores(name, trainFeatures, trainLabels)
        model = makeModel(name)
        model.fit(trainFeatures, trainLabels)
        testScores = model.decision_function(testFeatures)
        zScoresForEachModel.append((testScores - cvScores.mean()) / cvScores.std())

    return np.mean(zScoresForEachModel, axis=0)
