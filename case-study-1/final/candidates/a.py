# -*- coding: utf-8 -*-
"""
Spam classifier for CSDS 340 Case Study 1: logistic regression on
log-transformed word frequencies.

Missing values (-1) are filled with the training mean of the log-transformed
feature, then the features are standardized.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def logTransform(features):
    features = np.array(features, dtype=float, copy=True)
    features[features == -1] = np.nan
    return np.log1p(10000 * features)


class LogTransformAndFillMissing(BaseEstimator, TransformerMixin):
    """Log-transform the features and fill missing values with the training
    mean of each (log-transformed) feature."""

    def fit(self, X, y=None):
        self.columnMeans_ = np.nanmean(logTransform(X), axis=0)
        return self

    def transform(self, X):
        logFeatures = logTransform(X)
        return np.where(np.isnan(logFeatures), self.columnMeans_, logFeatures)


def predictTest(trainFeatures, trainLabels, testFeatures):
    model = make_pipeline(
        LogTransformAndFillMissing(),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=5000),
    )
    model.fit(trainFeatures, trainLabels)
    return model.decision_function(testFeatures)
