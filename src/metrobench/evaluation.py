"""Event-based evaluation of anomaly scores against the reported failures.

A detector outputs one anomaly score per minute. Minutes whose score reaches a
threshold (``score >= threshold``) are alarm minutes, merged into alarm events. Each event is then classified against the failure
windows:

- **detection**: the event overlaps ``[failure start - lead, failure end]``; the
  earliest such event gives the lead time (positive = before the reported start);
- **grace**: the event starts within ``grace`` after a failure ends (the same leak is
  often still visible before the repair), neither rewarded nor penalised;
- **false alarm**: any other event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FAILURES, find_runs

SPLIT = pd.Timestamp("2020-04-11")  # train: before, test: from this date


def split(frame, at: pd.Timestamp = SPLIT):
    """Chronological split into a training (normal) period and a test period."""
    return frame[frame.index < at], frame[frame.index >= at]


def alarm_events(alarms: pd.Series, merge: str = "30min") -> pd.DataFrame:
    """Group alarm minutes into events; alarms less than ``merge`` apart form one event."""
    on = alarms[alarms.fillna(False).astype(bool)]
    if on.empty:
        return pd.DataFrame({"start": pd.to_datetime([]), "end": pd.to_datetime([])})
    runs = find_runs(pd.Series(True, index=on.index), max_step=merge)
    return runs[["start", "end"]]


def classify_events(events: pd.DataFrame, failures: pd.DataFrame = FAILURES,
                    lead: str = "12h", grace: str = "24h") -> pd.Series:
    """Label each alarm event 'detection', 'grace' or 'false alarm' (as a Series aligned on ``events``)."""
    labels = pd.Series("false alarm", index=events.index, dtype=object)
    for start, end in failures[["start", "end"]].itertuples(index=False):
        hit = (events["end"] >= start - pd.Timedelta(lead)) & (events["start"] <= end)
        after = (events["start"] > end) & (events["start"] <= end + pd.Timedelta(grace))
        labels[after & (labels == "false alarm")] = "grace"
        labels[hit] = "detection"
    return labels


def evaluate(scores: pd.Series, threshold: float, failures: pd.DataFrame = FAILURES,
             lead: str = "12h", grace: str = "24h", merge: str = "30min") -> dict:
    """Event-based metrics of ``scores >= threshold`` over the period covered by ``scores``.

    Only failures whose window starts inside the scored period are counted. False
    alarms are normalised by the number of weeks of valid (non-NaN) scores.
    """
    failures = failures[(failures["start"] >= scores.index.min()) & (failures["start"] <= scores.index.max())]
    events = alarm_events(scores >= threshold, merge=merge)
    labels = classify_events(events, failures, lead=lead, grace=grace)
    weeks = scores.notna().sum() / (7 * 24 * 60)

    leads = {}
    for i, (start, end) in failures[["start", "end"]].iterrows():
        hit = events[(events["end"] >= start - pd.Timedelta(lead)) & (events["start"] <= end)]
        if len(hit):
            leads[i] = (start - hit["start"].min()) / pd.Timedelta("1h")
    return {
        "threshold": threshold,
        "failures": len(failures),
        "detected": len(leads),
        "lead_hours": leads,
        "median_lead_h": float(np.median(list(leads.values()))) if leads else np.nan,
        "false_alarms": int((labels == "false alarm").sum()),
        "false_alarms_per_week": float((labels == "false alarm").sum() / weeks) if weeks else np.nan,
        "alarm_share_%": float(100 * (scores >= threshold).sum() / scores.notna().sum()),
        "events": events.assign(label=labels),
    }


def calibrate(train_scores: pd.Series, max_alarms_per_week: float = 1.0, max_alarm_share: float = 0.05,
              merge: str = "30min") -> float:
    """Lowest threshold that keeps the training period within the alarm budget.

    The training period contains no reported failure, so every training alarm counts
    against the budget: at most ``max_alarms_per_week`` alarm events per week and at most
    ``max_alarm_share`` of the time in alarm (without the second limit, a threshold so low
    that the alarm never stops would count as a single event). Candidate thresholds are
    score quantiles, scanned from the highest down until the budget is exceeded.
    """
    valid = train_scores.dropna()
    weeks = len(valid) / (7 * 24 * 60)
    candidates = np.unique(valid.quantile(np.concatenate([np.linspace(0.9, 0.999, 100), np.linspace(0.999, 1, 101)])))
    best = float(np.nextafter(valid.max(), np.inf))  # no training alarm at all
    for threshold in candidates[::-1]:
        alarms = train_scores >= threshold
        if (len(alarm_events(alarms, merge=merge)) / weeks > max_alarms_per_week
                or alarms.sum() / len(valid) > max_alarm_share):
            break
        best = float(threshold)
    return best


def overlaps(events: pd.DataFrame, intervals: pd.DataFrame) -> pd.Series:
    """True for each event that overlaps at least one of the ``intervals`` (start/end columns)."""
    return pd.Series(
        [bool(((intervals["start"] <= end) & (intervals["end"] >= start)).any())
         for start, end in events[["start", "end"]].itertuples(index=False)],
        index=events.index, dtype=bool,
    )


def chance_detections(alarms_per_week: float, failures: pd.DataFrame = FAILURES, lead: str = "12h") -> float:
    """Expected number of failures "detected" by alarms fired at random (Poisson) at the given rate.

    A failure is hit by chance if at least one random alarm falls in its detection
    window, which lasts ``lead`` plus the failure duration.
    """
    window_weeks = (failures["end"] - failures["start"] + pd.Timedelta(lead)) / pd.Timedelta("7D")
    return float((1 - np.exp(-alarms_per_week * window_weeks)).sum())


def operating_curve(scores: pd.Series, failures: pd.DataFrame = FAILURES, n: int = 60, **kwargs) -> pd.DataFrame:
    """Detected failures, median lead time and false alarms per week across thresholds."""
    valid = scores.dropna()
    thresholds = np.unique(valid.quantile(np.linspace(0.5, 1, n)))
    rows = []
    for t in thresholds:
        r = evaluate(scores, float(t), failures, **kwargs)
        rows.append({k: r[k] for k in ("threshold", "detected", "median_lead_h", "false_alarms_per_week", "alarm_share_%")})
    return pd.DataFrame(rows)


def best_at_budget(curve: pd.DataFrame, max_alarms_per_week: float = 1.0) -> pd.Series:
    """Best operating point (most failures detected, then longest lead) within a false-alarm budget."""
    ok = curve[curve["false_alarms_per_week"] <= max_alarms_per_week]
    if ok.empty:
        return pd.Series(dtype=float)
    return ok.sort_values(["detected", "median_lead_h"], ascending=False).iloc[0]
