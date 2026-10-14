import numpy as np
import pandas as pd

from metrobench import timeseries as ts

INDEX = pd.date_range("2020-01-01", periods=6 * 24 * 60, freq="1min")  # six days
SPLIT = pd.Timestamp("2020-01-04")


def cycling_signal(seed=0):
    """A regular 20-minute on/off cycle with noise."""
    rng = np.random.default_rng(seed)
    on = (np.arange(len(INDEX)) % 20) < 5
    return pd.Series(on * 6.0 + rng.normal(0, 0.1, len(INDEX)), index=INDEX)


def test_lagged_design_only_uses_the_past():
    X = pd.DataFrame({"a": np.arange(len(INDEX), dtype=float)}, index=INDEX)
    design = ts.lagged_design(X, horizon="1h")
    assert design.loc[INDEX[120], "a_lag"] == 60  # value one hour earlier
    assert design["a_lag"].iloc[:60].isna().all()


def test_matrix_profile_flags_a_new_pattern_and_is_causal():
    x = cycling_signal()
    x.loc["2020-01-05 12:00":"2020-01-05 14:00"] = 6.0  # the cycle stops: stays "loaded"
    score = ts.matrix_profile(x, SPLIT, m=12, step="5min")
    normal = score.loc["2020-01-04 06:00":"2020-01-05 10:00"]
    assert score.loc["2020-01-05 13:30":"2020-01-05 14:00"].min() > 3 * normal.max()
    # the score at time t only depends on data up to t: nothing is flagged before the change
    assert score.loc[:"2020-01-05 11:55"].max() < score.loc["2020-01-05 13:30"]


def test_matrix_profile_reference_cleaning():
    x = cycling_signal()
    x.loc["2020-01-02 12:00":"2020-01-02 14:00"] = 6.0  # the same pattern already occurs in training
    x.loc["2020-01-05 12:00":"2020-01-05 14:00"] = 6.0
    exclude = pd.DataFrame({"start": [pd.Timestamp("2020-01-02 11:30")], "end": [pd.Timestamp("2020-01-02 14:30")]})
    full = ts.matrix_profile(x, SPLIT, m=12, step="5min")
    cleaned = ts.matrix_profile(x, SPLIT, m=12, step="5min", exclude=exclude)
    t = pd.Timestamp("2020-01-05 13:55")
    assert full.loc[t] < 1  # its nearest neighbour is the training occurrence
    assert cleaned.loc[t] > 5 * full.loc[t]
    assert cleaned.loc["2020-01-02 13:00":"2020-01-02 14:00"].isna().all()  # excluded stretch: no training score
