import pandas as pd

from metrobench.data import FAILURES, failure_mask, find_gaps, find_runs, frozen_mask


def test_find_gaps_reports_only_long_interruptions():
    index = pd.DatetimeIndex(["2020-01-01 00:00:00", "2020-01-01 00:00:10", "2020-01-01 00:05:10",
                              "2020-01-01 00:05:20", "2020-01-01 00:05:50"])
    gaps = find_gaps(index, threshold="60s")
    assert len(gaps) == 1
    assert gaps.loc[0, "start"] == pd.Timestamp("2020-01-01 00:00:10")
    assert gaps.loc[0, "end"] == pd.Timestamp("2020-01-01 00:05:10")
    assert gaps.loc[0, "duration"] == pd.Timedelta("5min")


def test_find_gaps_on_regular_series_is_empty():
    index = pd.date_range("2020-01-01", periods=100, freq="10s")
    assert find_gaps(index).empty


def test_failure_mask_includes_window_bounds():
    failures = pd.DataFrame({"start": [pd.Timestamp("2020-01-01 01:00")],
                             "end": [pd.Timestamp("2020-01-01 02:00")]})
    index = pd.date_range("2020-01-01 00:00", "2020-01-01 03:00", freq="30min")
    mask = failure_mask(index, failures)
    assert mask.tolist() == [False, False, True, True, True, False, False]


def test_find_runs_splits_at_gaps_and_filters_short_runs():
    index = pd.DatetimeIndex(["2020-01-01 00:00:00", "2020-01-01 00:00:10", "2020-01-01 00:00:20",
                              "2020-01-01 01:00:00", "2020-01-01 01:00:10", "2020-01-01 01:00:20",
                              "2020-01-01 01:00:30"])
    mask = pd.Series([True, True, True, True, True, False, True], index=index)
    runs = find_runs(mask, max_step="60s")
    assert runs["start"].tolist() == [index[0], index[3], index[6]]
    assert runs["end"].tolist() == [index[2], index[4], index[6]]
    assert len(find_runs(mask, min_duration="15s")) == 1


def test_frozen_mask_flags_repeated_rows():
    df = pd.DataFrame({"a": [1.0, 1.0, 2.0, 2.0], "b": [5.0, 5.0, 5.0, 6.0]})
    assert frozen_mask(df, ["a", "b"]).tolist() == [False, True, False, False]


def test_reported_failures_are_ordered_and_disjoint():
    assert (FAILURES["start"] < FAILURES["end"]).all()
    assert (FAILURES["start"].iloc[1:].to_numpy() > FAILURES["end"].iloc[:-1].to_numpy()).all()
