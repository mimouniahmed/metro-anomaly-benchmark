"""Detectors that model time explicitly: forecast residuals and Matrix Profile discords.

Both return one score per minute on the grid index (higher = more anomalous), use only
past data at each instant, and fit or compare against the training period only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import stumpy
from sklearn.ensemble import HistGradientBoostingRegressor

SEED = 0


def lagged_design(X: pd.DataFrame, horizon: str = "1h") -> pd.DataFrame:
    """Inputs for a forecast ``horizon`` ahead: every feature as it was ``horizon`` earlier, plus the hour of day."""
    lagged = X.shift(freq=horizon).reindex(X.index)
    lagged.columns = [f"{c}_lag" for c in X.columns]
    hour = X.index.hour + X.index.minute / 60
    lagged["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    lagged["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return lagged


def forecast_residual(X: pd.DataFrame, target: str, train_end: pd.Timestamp, horizon: str = "1h",
                      seed: int = SEED) -> tuple[pd.Series, pd.Series]:
    """Positive part of ``actual - forecast`` for one feature, forecast ``horizon`` ahead.

    A gradient-boosted regressor learns, on the training period, how the target evolves
    from the state observed ``horizon`` earlier. A leak makes the compressor work more than
    the recent state predicts, giving a positive residual. Returns ``(score, forecast)``.
    """
    design = lagged_design(X, horizon)
    y = X[target]
    usable = design.notna().all(axis=1) & y.notna()
    fit_rows = usable & (X.index < train_end)
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)
    model.fit(design[fit_rows], y[fit_rows])
    forecast = pd.Series(np.nan, index=X.index)
    forecast[usable] = model.predict(design[usable])
    return (y - forecast).clip(lower=0), forecast


def _subsequence_scores(distances: np.ndarray, index: pd.DatetimeIndex, m: int) -> pd.Series:
    """Align each subsequence distance on the subsequence's last time stamp (so the score is causal)."""
    d = np.asarray(distances, dtype=float)
    d[~np.isfinite(d)] = np.nan
    return pd.Series(d, index=index[m - 1:])


def matrix_profile(x: pd.Series, train_end: pd.Timestamp, m: int = 24, step: str = "5min",
                   normalize: bool = False, exclude: pd.DataFrame | None = None) -> pd.Series:
    """Distance of each length-``m`` subsequence to its nearest neighbour in the training period.

    The series is first averaged on a ``step`` grid (24 x 5 min = 2 hours by default).
    Training subsequences are compared with the rest of the training period (self-join,
    trivial neighbours excluded); later ones with the whole training period (AB-join).
    With ``normalize=False`` distances are Euclidean on raw values, so a change of level
    counts; ``normalize=True`` compares shapes only (z-normalised subsequences).
    Subsequences containing missing values get no score. The result is put back on the
    minute index of ``x``, each value held for the following ``step``.

    ``exclude`` (start/end intervals) removes stretches from the training *reference*, for
    example leak-like episodes found by the expert rule: test subsequences can no longer
    find their nearest neighbour there, and the stretches get no training score.
    """
    coarse = x.resample(step).mean()
    train = coarse[coarse.index < train_end]
    later = coarse[coarse.index >= train_end - (m - 1) * pd.Timedelta(step)]
    reference = train.copy()
    for start, end in ([] if exclude is None else exclude[["start", "end"]].itertuples(index=False)):
        reference.loc[start:end] = np.nan
    # Self-join on the (cleaned) reference: trivial neighbours are excluded, and excluded
    # stretches contain NaN, so they get no training score and do not weigh on calibration.
    self_join = np.asarray(stumpy.stump(reference.to_numpy(dtype=float), m, normalize=normalize)[:, 0], dtype=float)
    ab_join = np.asarray(stumpy.stump(later.to_numpy(dtype=float), m, T_B=reference.to_numpy(dtype=float),
                                      ignore_trivial=False, normalize=normalize)[:, 0], dtype=float)
    scores = pd.concat([
        _subsequence_scores(self_join, train.index, m),
        _subsequence_scores(ab_join, later.index, m).loc[train_end:],
    ])
    scores = scores[~scores.index.duplicated()]
    minute = scores.reindex(x.index, method="ffill", limit=int(pd.Timedelta(step) / pd.Timedelta("1min")) - 1)
    return minute.where(x.notna())
