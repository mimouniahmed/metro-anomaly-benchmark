import numpy as np
import pandas as pd

from metrobench import baselines as bl

INDEX = pd.date_range("2020-01-01", periods=2000, freq="1min")


def test_robust_zscores_fall_back_to_std_for_constant_features():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [2.0, 2.0, 2.0, 2.0, 3.0]})
    z = bl.robust_zscores(train, train)
    assert np.isfinite(z.to_numpy()).all()
    assert z.loc[2, "a"] == 0


def test_hotelling_flags_unusual_combinations():
    rng = np.random.default_rng(0)
    a = rng.normal(size=1000)
    train = pd.DataFrame({"a": a, "b": a + rng.normal(scale=0.1, size=1000)})  # strongly correlated
    test = pd.DataFrame({"a": [1.0, 1.0], "b": [1.0, -1.0]})  # same values, second breaks the correlation
    t2 = bl.hotelling_t2(test, train)
    assert t2.iloc[1] > 50 * t2.iloc[0]


def test_ewma_chart_is_centred_and_standardised_on_training_data():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(10, 2, size=len(INDEX)), index=INDEX)
    stat = bl.ewma_chart(x, x, lam=0.1)
    assert abs(stat.iloc[200:].mean()) < 0.2
    assert 0.8 < stat.iloc[200:].std() < 1.2


def test_cusum_accumulates_a_shift_and_resets_after_a_gap():
    x = pd.Series(0.0, index=INDEX)
    x.iloc[:1000] = np.tile([-1.0, 1.0], 500)  # training: mean 0, std 1
    x.iloc[1000:1100] = 2.0  # sustained shift of 2 sigma
    x.iloc[1100:1110] = np.nan  # gap
    s = bl.cusum(x, x.iloc[:1000], k=0.5)
    assert s.iloc[999] <= 1.5
    assert np.isclose(s.iloc[1099], s.iloc[999] + 100 * 1.5, atol=1)  # grows by z - k = 1.5 per minute
    assert np.isnan(s.iloc[1105])
    assert s.iloc[1110] == 0  # x = 0 after the gap: the sum restarted from zero


def test_rolling_baseline_ignores_the_current_value():
    x = pd.Series(np.tile([0.0, 1.0], 1000), index=INDEX)
    x.iloc[-1] = 1000.0
    z = bl.rolling_baseline_zscore(x, window="1D", min_periods=100)
    assert z.iloc[-1] > 100  # the spike is judged against the past only
