"""Download, load and annotate the MetroPT-3 air compressor dataset."""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
RAW_CSV = "MetroPT3(AirCompressor).csv"
CACHE = "metropt3.parquet"

ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
DIGITAL = ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]

UNITS = {
    "TP2": "bar", "TP3": "bar", "H1": "bar", "DV_pressure": "bar", "Reservoirs": "bar",
    "Oil_temperature": "°C", "Motor_current": "A",
}

DESCRIPTIONS = {
    "TP2": "pressure on the compressor",
    "TP3": "pressure at the pneumatic panel",
    "H1": "pressure drop when the cyclonic separator filter discharges",
    "DV_pressure": "pressure drop when the air-dryer towers discharge (0 = compressor under load)",
    "Reservoirs": "downstream reservoir pressure (should track TP3)",
    "Oil_temperature": "compressor oil temperature",
    "Motor_current": "current of one motor phase (~0 off, ~4 offloaded, ~7 loaded, ~9 start-up)",
    "COMP": "air intake valve signal (active = no intake: compressor off or offloaded)",
    "DV_eletric": "outlet valve signal (active = compressor under load)",
    "Towers": "which air-dryer tower is drying (0 = tower one, 1 = tower two)",
    "MPG": "starts the compressor under load when pressure falls below 8.2 bar",
    "LPS": "low-pressure signal, active below 7 bar",
    "Pressure_switch": "detects discharge in the air-drying towers",
    "Oil_level": "oil level signal",
    "Caudal_impulses": "pulses counting the air flow from the unit to the reservoirs",
}

# Failure reports published with the dataset (all air leaks, severity "high stress").
# The source table numbers the second failure "#1" and dates its maintenance "30 Apr",
# which precedes the failure itself; it is renumbered here and the maintenance dates
# are not used.
FAILURES = pd.DataFrame(
    {
        "start": pd.to_datetime(["2020-04-18 00:00", "2020-05-29 23:30", "2020-06-05 10:00", "2020-07-15 14:30"]),
        "end": pd.to_datetime(["2020-04-18 23:59", "2020-05-30 06:00", "2020-06-07 14:30", "2020-07-15 19:00"]),
    },
    index=pd.Index([1, 2, 3, 4], name="failure"),
)


def download(data_dir: str | Path = "data") -> Path:
    """Download and extract the raw CSV into ``data_dir/raw`` unless already present."""
    raw_dir = Path(data_dir) / "raw"
    csv = raw_dir / RAW_CSV
    if not csv.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        archive.extractall(raw_dir)
    return csv


def load(data_dir: str | Path = "data") -> pd.DataFrame:
    """Return the sensor readings indexed by timestamp.

    The first call parses the CSV (downloading it if needed) and writes a Parquet
    cache; later calls read the cache. Analog signals are float32, digital ones int8.
    """
    cache = Path(data_dir) / CACHE
    if cache.exists():
        return pd.read_parquet(cache)
    df = pd.read_csv(download(data_dir), index_col=0, parse_dates=["timestamp"])
    df = df.set_index("timestamp")[ANALOG + DIGITAL]
    df = df.astype({c: "float32" for c in ANALOG} | {c: "int8" for c in DIGITAL})
    df.to_parquet(cache)
    return df


def find_gaps(index: pd.DatetimeIndex, threshold: str = "60s") -> pd.DataFrame:
    """List the interruptions longer than ``threshold`` between consecutive readings."""
    step = index.to_series().diff()
    is_gap = step > pd.Timedelta(threshold)
    gaps = pd.DataFrame({"start": index[:-1][is_gap.to_numpy()[1:]], "end": index[is_gap.to_numpy()]})
    gaps["duration"] = gaps["end"] - gaps["start"]
    return gaps


def find_runs(mask: pd.Series, max_step: str = "60s", min_duration: str = "0s") -> pd.DataFrame:
    """List the consecutive stretches where a boolean time-indexed series is True.

    A stretch is broken by a False value or by a recording gap longer than ``max_step``,
    so that an episode never spans missing data. Stretches shorter than ``min_duration``
    are dropped.
    """
    times = mask.index.to_series()
    new_run = (mask != mask.shift()) | (times.diff() > pd.Timedelta(max_step))
    run_id = new_run.cumsum()[mask]
    runs = times[mask].groupby(run_id).agg(["min", "max"]).set_axis(["start", "end"], axis=1)
    runs["duration"] = runs["end"] - runs["start"]
    return runs[runs["duration"] >= pd.Timedelta(min_duration)].reset_index(drop=True)


def frozen_mask(df: pd.DataFrame, columns: list[str] = ANALOG) -> pd.Series:
    """True where every analog reading repeats the previous one exactly (stalled logger)."""
    return df[columns].diff().abs().sum(axis=1, min_count=1).eq(0)


def failure_mask(index: pd.DatetimeIndex, failures: pd.DataFrame = FAILURES) -> pd.Series:
    """Boolean series, True for timestamps that fall inside a reported failure window."""
    mask = pd.Series(False, index=index)
    for start, end in failures[["start", "end"]].itertuples(index=False):
        mask |= (index >= start) & (index <= end)
    return mask
