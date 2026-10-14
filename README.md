# MetroPT Anomaly Benchmark — Detecting air leaks on a metro train compressor

[![tests](https://github.com/mimouniahmed/metro-anomaly-benchmark/actions/workflows/tests.yml/badge.svg)](https://github.com/mimouniahmed/metro-anomaly-benchmark/actions/workflows/tests.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**English** · [Français](README.fr.md)

Comparison of anomaly detection methods on **real industrial sensor data**: 7 months of readings from the air production unit of a Porto metro train, with four documented air-leak failures. The methods range from statistical control rules to classical machine learning and deep learning. All of them run on the same data with the same evaluation protocol.

**Question**: which method detects the air leaks earliest, with the fewest false alarms, and is the added complexity worth it?

> **Status**: in progress. Notebooks 01–05 done (data, protocol, statistical baselines, classical ML, time series methods); deep learning and the final comparison are next.

## Findings so far

| | |
|---|---|
| **Failure signature** | During a leak the compressor stops cycling and stays **under load 99.5% of the time** (vs. 15% normally), with oil **~13 °C hotter**. It is a *collective* anomaly: each reading is within the normal range, so detection needs time windows, not single readings |
| **Existing alarm** | The on-board low-pressure signal fires on only 2.4% of failure readings |
| **Data quality** | Readings every ~10 s (not 1 Hz as documented); 18% of the period missing; 7 days of frozen sensor values that must be excluded |
| **Ground truth** | 8 sustained-load episodes ≥ 1 h are **not** in the failure reports, so labels are incomplete. Evaluation is therefore event-based, with unreported episodes reported separately |
| **Early warning** | Before failure 4 the compressor was already overloaded for 14 hours, so detection ahead of the reported start is plausible |
| **Windows, not readings** | Under the evaluation protocol, the raw motor current detects 0/4 failures; its 2-hour mean detects 4/4 with 0.9 false alarms per week, failure 4 flagged 13.7 h early |
| **Drift** | A winter-calibrated threshold on oil temperature gives 1.4 false alarms per week in summer, where a well-placed one would give none |
| **Best statistical baselines** | Oil temperature z-score against a 7-day rolling reference: 4/4 failures, 0.28 false alarms per week. CUSUM on motor current: 4/4, 0.64 per week, failure 4 flagged 12.9 h early. Multivariate distances (max \|z\|, Hotelling T²) fail, swamped by one heavy-tailed feature |
| **Classical ML** | LOF on the window features matches the best charts (4/4 on all seeds, 0.55 false alarms per week, 12.6 h early on failure 4) without beating them; One-Class SVM detects 2/4. Results vary up to threefold across random seeds, so every stochastic result is a median over five. PCA adds what scores lack: which signals explain each alarm |
| **Contaminated reference** | The Matrix Profile finds the nearest training neighbour of every failure inside an unreported leak-like episode of March 29, and detects 0/4. Removing the episodes the expert rule finds in training data takes it to 4/4 with 0.71 false alarms per week. The same cleaning drops LOF from 4/4 to 2/4: its earlier success partly relied on that unreported leak. Forecast residuals detect the onset of leaks, not the leaks, with 2–6 false alarms per week |

![Failure signature](docs/figures/01_failures.png)

## Approach

| Notebook | Content | Status |
|---|---|---|
| [01 — Data](notebooks/01_data_exploration.ipynb) | Sensors, sampling, gaps, frozen readings, operating cycle, failure signature, label completeness | ✅ |
| [02 — Evaluation protocol](notebooks/02_evaluation_protocol.ipynb) | One-minute grid, 18 window features, chronological split, event-based scoring, threshold calibration, chance level | ✅ |
| [03 — Statistical baselines](notebooks/03_statistical_baselines.ipynb) | Expert rule, robust z-score, Hotelling T², EWMA and CUSUM charts, rolling-baseline z-score | ✅ |
| [04 — Classical ML](notebooks/04_classical_ml.ipynb) | Isolation Forest, One-Class SVM, Local Outlier Factor, PCA reconstruction; raw vs. drift-adjusted inputs; seed variability; alarm explanation | ✅ |
| [05 — Time series methods](notebooks/05_time_series_methods.ipynb) | Forecast residuals (gradient boosting), Matrix Profile (raw and z-normalised, full and cleaned reference), cleaning cross-check on LOF | ✅ |
| 06 — Deep learning | Dense and LSTM autoencoders | planned |
| 07 — Results | Comparison table, cost of complexity, recommendation | planned |

## Data

**MetroPT-3**, [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset) (CC BY 4.0). Veloso, B., Ribeiro, R.P., Pereira, P.M., Gama, J. *The MetroPT dataset for predictive maintenance.* Scientific Data 9, 764 (2022).

1.5 M readings from February to August 2020 of 15 signals:

| Type | Signals |
|---|---|
| Analog (7) | `TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs` (pressures, bar), `Oil_temperature` (°C), `Motor_current` (A) |
| Digital (8) | `COMP`, `DV_eletric`, `Towers`, `MPG`, `LPS`, `Pressure_switch`, `Oil_level`, `Caudal_impulses` |

Labels: four air-leak windows from the operator's failure reports.

## Installation and reproduction

```bash
git clone https://github.com/mimouniahmed/metro-anomaly-benchmark.git
cd metro-anomaly-benchmark
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

pytest
```

The dataset (218 MB) is downloaded automatically on first use by `metrobench.data.load()` and cached as Parquet (22 MB) in `data/`.

## Structure

```
metro-anomaly-benchmark/
├── src/metrobench/
│   ├── data.py         # download, Parquet cache, gaps, frozen readings, runs, failure labels
│   ├── features.py     # one-minute grid and backward-looking window features
│   ├── evaluation.py   # time split, alarm events, event-based metrics, threshold calibration
│   ├── baselines.py    # robust z-score, Hotelling T², EWMA, CUSUM, rolling-baseline z-score
│   ├── ml.py           # Isolation Forest, One-Class SVM, LOF, PCA reconstruction, drift-adjusted inputs
│   ├── timeseries.py   # forecast residuals, Matrix Profile with optional reference cleaning
│   ├── deep.py         # dense and LSTM autoencoders (PyTorch)
│   └── plotstyle.py    # shared figure style
├── notebooks/          # analysis, 01 → 07
├── results/            # result tables per notebook (CSV)
├── tests/              # unit tests
├── docs/figures/
└── .github/workflows/  # CI
```

## Stack

Python 3.12 · pandas · NumPy · PyArrow · scikit-learn · stumpy · PyTorch · Matplotlib · pytest · GitHub Actions

## Author

**Ahmed Mimouni** — [github.com/mimouniahmed](https://github.com/mimouniahmed)

## License

Released under the [MIT License](LICENSE). The MetroPT-3 dataset is distributed by UCI under CC BY 4.0 and is not redistributed in this repository.
