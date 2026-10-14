"""Statistical baseline detectors. Each returns one anomaly score per minute (higher = more anomalous).

Reference statistics (means, deviations, covariances) are always estimated on the
training period only, passed in as ``train``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAD_TO_STD = 1.4826  # scales the median absolute deviation to a standard deviation for Gaussian data


def robust_zscores(X: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Standardise each column with the training median and MAD, which outliers barely move.

    For a nearly constant discrete feature the MAD is zero; its standard deviation is used instead.
    """
    median = train.median()
    mad = (train - median).abs().median() * MAD_TO_STD
    scale = mad.where(mad > 0, train.std())
    return (X - median) / scale


def max_abs_zscore(X: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    """Largest absolute robust z-score across features: flags the most extreme feature."""
    z = robust_zscores(X, train)
    return z.abs().max(axis=1, skipna=False)


def hotelling_t2(X: pd.DataFrame, train: pd.DataFrame) -> pd.Series:
    """Hotelling T² (squared Mahalanobis distance) to the training mean.

    Unlike per-feature z-scores, it accounts for correlations: a combination of values
    that never occurs together in training scores high even if each value is common.
    """
    clean = train.dropna()
    mean, std = clean.mean(), clean.std()
    Z = (X - mean) / std
    inv = np.linalg.pinv(((clean - mean) / std).cov().to_numpy())
    z = Z.to_numpy()
    t2 = np.einsum("ij,jk,ik->i", z, inv, z)
    return pd.Series(t2, index=X.index)


def ewma_chart(x: pd.Series, train: pd.Series, lam: float = 0.05) -> pd.Series:
    """Standardised EWMA control chart statistic (upper side).

    The exponentially weighted mean ``z_t = lam * x_t + (1 - lam) * z_{t-1}`` is centred
    on the training mean and divided by its asymptotic standard deviation
    ``sigma * sqrt(lam / (2 - lam))``. Minutes without data are skipped and get no score.
    """
    mu, sigma = train.mean(), train.std()
    z = x.ewm(alpha=lam, adjust=False, ignore_na=True).mean()
    return ((z - mu) / (sigma * np.sqrt(lam / (2 - lam)))).where(x.notna())


def cusum(x: pd.Series, train: pd.Series, k: float = 0.5, reset_after: int = 5) -> pd.Series:
    """One-sided upper CUSUM on the standardised series.

    ``S_t = max(0, S_{t-1} + z_t - k)`` accumulates evidence of a sustained upward shift
    larger than ``k`` standard deviations. The sum is reset after ``reset_after``
    consecutive minutes without data, since evidence should not carry across a gap.
    """
    mu, sigma = train.mean(), train.std()
    z = ((x - mu) / sigma).to_numpy()
    out = np.full(len(z), np.nan)
    s, missing = 0.0, 0
    for i, value in enumerate(z):
        if np.isnan(value):
            missing += 1
            if missing >= reset_after:
                s = 0.0
            continue
        missing = 0
        s = max(0.0, s + value - k)
        out[i] = s
    return pd.Series(out, index=x.index)


def rolling_baseline_zscore(x: pd.Series, window: str = "7D", min_periods: int = 24 * 60) -> pd.Series:
    """Robust z-score against the previous ``window`` of data instead of a fixed training period.

    The reference (median, MAD) follows slow drifts such as the seasonal rise of the oil
    temperature. It is shifted by one minute so that the current value is never part of
    its own reference.
    """
    past = x.shift(1).rolling(window, min_periods=min_periods)
    median = past.median()
    mad = (x.shift(1) - median).abs().rolling(window, min_periods=min_periods).median() * MAD_TO_STD
    return (x - median) / mad.replace(0, np.nan)
