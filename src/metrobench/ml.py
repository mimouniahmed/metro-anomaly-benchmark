"""Classical unsupervised anomaly detectors on the window features.

Each detector is fitted on the complete training rows only (no reported failure in
that period) and returns one score per minute, higher = more anomalous, NaN where a
feature is missing. Hyperparameters are library defaults or textbook values fixed in
advance; none is tuned on the test period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

SEED = 0


def drift_adjusted(X: pd.DataFrame, train: pd.DataFrame, window: str = "7D", min_hours: int = 24,
                   floor: float = 0.1) -> pd.DataFrame:
    """Express each feature relative to its own recent past (robust z-score against the previous ``window``).

    The reference median and MAD are computed on hourly medians for speed, shifted by one
    hour so that the current hour never belongs to its own reference. The scale is floored
    at ``floor`` times the feature's training standard deviation: during a quiet week the
    rolling MAD can be (numerically) zero, which would turn tiny changes into huge scores.
    """
    hourly = X.resample("1h").median()
    median = hourly.shift(1).rolling(window, min_periods=min_hours).median()
    mad = (hourly.shift(1) - median).abs().rolling(window, min_periods=min_hours).median() * 1.4826
    scale = mad.clip(lower=floor * train.std(), axis=1)
    median = median.reindex(X.index, method="ffill")
    scale = scale.reindex(X.index, method="ffill")
    return (X - median) / scale


def _prepare(X: pd.DataFrame, train: pd.DataFrame, max_train: int | None, seed: int):
    """Robust-scale with training statistics, return (scaled complete rows, scaled training sample)."""
    train = train.dropna()
    scaler = RobustScaler().fit(train)
    complete = X.dropna()
    Z = pd.DataFrame(scaler.transform(complete), index=complete.index, columns=X.columns)
    Z_train = pd.DataFrame(scaler.transform(train), index=train.index, columns=X.columns)
    if max_train and len(Z_train) > max_train:
        Z_train = Z_train.sample(max_train, random_state=seed).sort_index()
    return Z, Z_train


def _as_series(values, index, full_index) -> pd.Series:
    return pd.Series(values, index=index).reindex(full_index)


def isolation_forest(X: pd.DataFrame, train: pd.DataFrame, n_estimators: int = 200, seed: int = SEED) -> pd.Series:
    """Isolation Forest: anomalies are isolated by fewer random splits (shorter average path)."""
    Z, Z_train = _prepare(X, train, None, seed)
    model = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1).fit(Z_train)
    return _as_series(-model.score_samples(Z), Z.index, X.index)


def one_class_svm(X: pd.DataFrame, train: pd.DataFrame, nu: float = 0.05, max_train: int = 10_000,
                  seed: int = SEED) -> pd.Series:
    """One-Class SVM (RBF kernel): a boundary enclosing all but a fraction ``nu`` of the training data.

    Training cost grows quadratically with the number of rows, hence a random subsample.
    """
    Z, Z_train = _prepare(X, train, max_train, seed)
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu).fit(Z_train)
    return _as_series(-model.decision_function(Z), Z.index, X.index)


def local_outlier_factor(X: pd.DataFrame, train: pd.DataFrame, n_neighbors: int = 50, max_train: int = 20_000,
                         seed: int = SEED) -> pd.Series:
    """Local Outlier Factor (novelty mode): density around a point compared with that of its neighbours."""
    Z, Z_train = _prepare(X, train, max_train, seed)
    model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, n_jobs=-1).fit(Z_train)
    return _as_series(-model.score_samples(Z), Z.index, X.index)


def pca_reconstruction(X: pd.DataFrame, train: pd.DataFrame, variance: float = 0.95,
                       seed: int = SEED) -> tuple[pd.Series, pd.DataFrame]:
    """PCA reconstruction error and its breakdown per feature.

    The principal components keeping ``variance`` of the training variance describe the
    normal correlations between features; a point that breaks them is poorly
    reconstructed. Returns the total squared error and each feature's share of it, which
    tells which signals deviate.
    """
    Z, Z_train = _prepare(X, train, None, seed)
    pca = PCA(n_components=variance, random_state=seed).fit(Z_train)
    residual = Z - pca.inverse_transform(pca.transform(Z))
    squared = residual**2
    contributions = squared.reindex(X.index)
    return squared.sum(axis=1).reindex(X.index), contributions
