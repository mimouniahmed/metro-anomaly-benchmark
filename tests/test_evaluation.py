import numpy as np
import pandas as pd

from metrobench import evaluation as ev

FAILURE = pd.DataFrame({"start": [pd.Timestamp("2020-01-10 12:00")], "end": [pd.Timestamp("2020-01-10 18:00")]},
                       index=pd.Index([1], name="failure"))
INDEX = pd.date_range("2020-01-01", "2020-01-20", freq="1min")


def scores_with(*bursts):
    """Zero scores with value 1 on the given (start, end) bursts."""
    s = pd.Series(0.0, index=INDEX)
    for start, end in bursts:
        s.loc[start:end] = 1.0
    return s


def test_alarm_events_merges_close_alarms():
    s = scores_with(("2020-01-02 00:00", "2020-01-02 00:10"), ("2020-01-02 00:20", "2020-01-02 00:30"),
                    ("2020-01-03 00:00", "2020-01-03 00:05"))
    events = ev.alarm_events(s >= 1, merge="30min")
    assert len(events) == 2
    assert events.iloc[0]["end"] == pd.Timestamp("2020-01-02 00:30")


def test_early_persistent_alarm_counts_with_its_full_lead():
    # starts 14 h before the failure (outside the 12 h window) but lasts into it
    s = scores_with(("2020-01-09 22:00", "2020-01-10 13:00"))
    r = ev.evaluate(s, 1.0, FAILURE)
    assert r["detected"] == 1
    assert r["lead_hours"][1] == 14
    assert r["false_alarms"] == 0


def test_alarm_too_early_is_a_false_alarm_and_late_one_is_grace():
    s = scores_with(("2020-01-09 20:00", "2020-01-09 21:00"), ("2020-01-11 06:00", "2020-01-11 07:00"))
    r = ev.evaluate(s, 1.0, FAILURE)
    assert r["detected"] == 0
    assert r["false_alarms"] == 1
    assert r["events"]["label"].tolist() == ["false alarm", "grace"]


def test_false_alarm_rate_uses_valid_weeks_only():
    s = scores_with(("2020-01-03 00:00", "2020-01-03 01:00"))
    s.loc["2020-01-12":] = np.nan  # only the first ~11 days are scored
    r = ev.evaluate(s, 1.0, FAILURE)
    weeks = s.notna().sum() / (7 * 24 * 60)
    assert np.isclose(r["false_alarms_per_week"], 1 / weeks)


def test_calibrate_respects_the_alarm_budget():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=len(INDEX)), index=INDEX)
    threshold = ev.calibrate(s, max_alarms_per_week=1)
    weeks = len(s) / (7 * 24 * 60)
    assert len(ev.alarm_events(s >= threshold)) / weeks <= 1


def test_calibrate_handles_saturated_scores():
    s = scores_with(("2020-01-02 00:00", "2020-01-02 03:00"))  # one training event at the maximum
    threshold = ev.calibrate(s, max_alarms_per_week=1)
    assert threshold == 1.0


def test_chance_detections_grow_with_alarm_rate():
    assert ev.chance_detections(0, FAILURE) == 0
    assert 0 < ev.chance_detections(1, FAILURE) < ev.chance_detections(5, FAILURE) < 1


def test_overlaps():
    events = pd.DataFrame({"start": pd.to_datetime(["2020-01-02 00:00", "2020-01-05 00:00"]),
                           "end": pd.to_datetime(["2020-01-02 01:00", "2020-01-05 01:00"])})
    intervals = pd.DataFrame({"start": pd.to_datetime(["2020-01-02 00:30"]), "end": pd.to_datetime(["2020-01-02 05:00"])})
    assert ev.overlaps(events, intervals).tolist() == [True, False]
