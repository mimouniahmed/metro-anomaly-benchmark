import numpy as np
import pandas as pd

from metrobench.data import ANALOG, DIGITAL
from metrobench.features import minute_grid, window_features


def synthetic_readings(minutes=180, gap=(60, 90)):
    """10-second readings with a compressor loaded 1 minute out of 10 and a recording gap."""
    index = pd.date_range("2020-01-01", periods=minutes * 6, freq="10s")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(1, 0.01, size=(len(index), len(ANALOG))), index=index, columns=ANALOG)
    for c in DIGITAL:
        df[c] = 0
    df["DV_eletric"] = ((np.arange(len(index)) // 6) % 10 == 0).astype(int)
    keep = (index < index[0] + pd.Timedelta(minutes=gap[0])) | (index >= index[0] + pd.Timedelta(minutes=gap[1]))
    return df[keep]


def test_minute_grid_is_regular_and_flags_gaps():
    grid = minute_grid(synthetic_readings())
    assert (grid.index.to_series().diff().dropna() == pd.Timedelta("1min")).all()
    assert len(grid) == 180
    assert grid["valid"].sum() == 150
    assert grid.loc[~grid["valid"], "Motor_current"].isna().all()


def test_minute_grid_counts_load_starts():
    grid = minute_grid(synthetic_readings(gap=(200, 200)))  # no gap
    assert grid["load_starts"].sum() == 17  # 18 loaded minutes, the first one starts the series


def test_window_features_need_coverage_and_ignore_the_future():
    df = synthetic_readings()
    grid = minute_grid(df)
    feats = window_features(grid, windows=("30min",))
    # load share over a full 30-min window: 3 loaded minutes out of 30
    assert np.isclose(feats["load_share_30min"].iloc[59], 0.1)
    # just after the gap, the window is mostly empty -> NaN
    assert feats.iloc[95].isna().all()
    # changing future readings does not change past features
    future = df.copy()
    future.loc[future.index >= "2020-01-01 02:30", "Motor_current"] = 100
    feats_future = window_features(minute_grid(future), windows=("30min",))
    pd.testing.assert_frame_equal(feats.iloc[:150], feats_future.iloc[:150])
