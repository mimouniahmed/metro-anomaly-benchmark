import numpy as np
import pandas as pd

from metrobench import deep

INDEX = pd.date_range("2020-01-01", periods=4 * 24 * 60, freq="1min")  # four days
SPLIT = pd.Timestamp("2020-01-03")


def test_dense_autoencoder_scores_broken_correlations_higher():
    rng = np.random.default_rng(0)
    a = rng.normal(size=len(INDEX))
    X = pd.DataFrame({"a": a, "b": a + rng.normal(scale=0.1, size=len(INDEX)), "c": rng.normal(size=len(INDEX))},
                     index=INDEX)
    X.iloc[-100:, 1] = -X.iloc[-100:, 0]  # b stops following a
    X.iloc[10:20] = np.nan
    score, errors, history = deep.dense_autoencoder(X, X.iloc[:3000], seed=0)
    assert score.iloc[10:20].isna().all()
    assert score.iloc[-100:].median() > 5 * score.iloc[:3000].median()
    assert errors.columns.tolist() == ["a", "b", "c"]
    assert history[-1][1] < history[0][1]  # validation loss decreased


def test_make_sequences_skips_incomplete_windows_and_ends_on_last_step():
    grid = pd.DataFrame({s: np.arange(len(INDEX), dtype=float) for s in ["x", "y"]}, index=INDEX)
    grid.iloc[100:110] = np.nan
    windows, ends, coarse = deep.make_sequences(grid, ["x", "y"], length=6, step="5min")
    assert windows.shape[1:] == (6, 2)
    assert not np.isnan(windows).any()
    assert ends[0] == coarse.index[5]
    # the 5-min bins 20-21 contain missing minutes: no window may include them
    bad = coarse.index[20]
    assert not ((ends >= bad) & (ends - pd.Timedelta("25min") <= bad)).any()


def test_lstm_autoencoder_flags_a_stopped_cycle():
    on = ((np.arange(len(INDEX)) % 20) < 5).astype(float)
    grid = pd.DataFrame({"Motor_current": on * 6, "DV_eletric": on}, index=INDEX)
    grid["valid"] = True
    grid.loc["2020-01-04 06:00":"2020-01-04 09:00", ["Motor_current", "DV_eletric"]] = [6.0, 1.0]
    score, _, fit = deep.lstm_autoencoder(grid, SPLIT, signals=["Motor_current", "DV_eletric"], length=12, seed=0)
    normal = score.loc["2020-01-03 06:00":"2020-01-04 05:00"]
    assert score.loc["2020-01-04 08:00":"2020-01-04 09:00"].min() > 3 * normal.quantile(0.99)
    actual, rebuilt = fit.reconstruct(grid, pd.Timestamp("2020-01-03 12:00"))
    assert actual.shape == rebuilt.shape == (12, 2)
    assert (actual - rebuilt).abs().to_numpy().mean() < 1.5  # normal cycles are reconstructed in original units
