"""Regular one-minute grid and rolling-window features describing the compressor cycle."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import find_runs, frozen_mask, load

WINDOWS = ("30min", "2h")


def minute_grid(df: pd.DataFrame, frozen_min: str = "30min") -> pd.DataFrame:
    """Average the raw readings on a regular one-minute grid.

    Adds ``load_starts`` (number of transitions to the loaded state within the minute)
    and ``valid`` (the minute has readings and is not inside a frozen stretch longer
    than ``frozen_min``). Invalid minutes are kept as rows with NaN values so that the
    grid stays regular.
    """
    starts = df["DV_eletric"].diff().eq(1) & df.index.to_series().diff().le(pd.Timedelta("60s"))
    frozen = pd.Series(False, index=df.index)
    for start, end in find_runs(frozen_mask(df), min_duration=frozen_min)[["start", "end"]].itertuples(index=False):
        frozen.loc[start:end] = True

    keep = df[~frozen]
    grid = keep.resample("1min").mean()
    grid["load_starts"] = starts[~frozen].resample("1min").sum()
    grid["valid"] = keep["DV_eletric"].resample("1min").count().gt(0)
    grid = grid.reindex(pd.date_range(df.index.min().floor("1min"), df.index.max().floor("1min"), freq="1min"))
    grid["valid"] = grid["valid"].fillna(False).astype(bool)
    grid.loc[~grid["valid"], grid.columns != "valid"] = float("nan")
    return grid


def window_features(grid: pd.DataFrame, windows=WINDOWS, min_coverage: float = 0.8) -> pd.DataFrame:
    """Backward-looking rolling features on the minute grid.

    For each window, a feature is computed only if at least ``min_coverage`` of the
    window's minutes are valid; otherwise the row is NaN. Features never use the future.
    """
    out = {}
    for w in windows:
        roll = grid.rolling(w)
        n_minutes = pd.Timedelta(w) / pd.Timedelta("1min")
        covered = grid["valid"].astype(float).rolling(w).sum() >= min_coverage * n_minutes
        feats = {
            f"load_share_{w}": roll["DV_eletric"].mean(),
            f"load_starts_{w}": roll["load_starts"].sum() * 60 / n_minutes,  # per hour
            f"current_mean_{w}": roll["Motor_current"].mean(),
            f"current_std_{w}": roll["Motor_current"].std(),
            f"oil_mean_{w}": roll["Oil_temperature"].mean(),
            f"oil_max_{w}": roll["Oil_temperature"].max(),
            f"tp2_mean_{w}": roll["TP2"].mean(),
            f"reservoirs_mean_{w}": roll["Reservoirs"].mean(),
            f"reservoirs_min_{w}": roll["Reservoirs"].min(),
        }
        for name, values in feats.items():
            out[name] = values.where(covered & grid["valid"])
    return pd.DataFrame(out, index=grid.index)


def load_features(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(grid, features)``, computed from :func:`metrobench.data.load` and cached as Parquet."""
    paths = Path(data_dir) / "grid.parquet", Path(data_dir) / "features.parquet"
    if all(p.exists() for p in paths):
        return pd.read_parquet(paths[0]), pd.read_parquet(paths[1])
    grid = minute_grid(load(data_dir))
    features = window_features(grid)
    grid.to_parquet(paths[0])
    features.to_parquet(paths[1])
    return grid, features
