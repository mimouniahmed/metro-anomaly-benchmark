# MetroPT Anomaly Benchmark — Detecting air leaks on a metro train compressor

[![tests](https://github.com/mimouniahmed/metro-anomaly-benchmark/actions/workflows/tests.yml/badge.svg)](https://github.com/mimouniahmed/metro-anomaly-benchmark/actions/workflows/tests.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**English** · [Français](README.fr.md)

Comparison of anomaly detection methods on **real industrial sensor data**: 7 months of readings from the air production unit of a Porto metro train, with four documented air-leak failures. The methods range from statistical control rules to classical machine learning and deep learning. All of them run on the same data with the same evaluation protocol.

**Question**: which method detects the air leaks earliest, with the fewest false alarms, and is the added complexity worth it?

> **Status**: in progress. Data analysis done (notebook 01); detection methods and evaluation are next.

## Findings so far

| | |
|---|---|
| **Failure signature** | During a leak the compressor stops cycling and stays **under load 99.5% of the time** (vs. 15% normally), with oil **~13 °C hotter**. It is a *collective* anomaly: each reading is within the normal range, so detection needs time windows, not single readings |
| **Existing alarm** | The on-board low-pressure signal fires on only 2.4% of failure readings |
| **Data quality** | Readings every ~10 s (not 1 Hz as documented); 18% of the period missing; 7 days of frozen sensor values that must be excluded |
| **Ground truth** | 8 sustained-load episodes ≥ 1 h are **not** in the failure reports, so labels are incomplete. Evaluation is therefore event-based, with unreported episodes reported separately |
| **Early warning** | Before failure 4 the compressor was already overloaded for 14 hours, so detection ahead of the reported start is plausible |

![Failure signature](docs/figures/01_failures.png)

## Approach

| Notebook | Content | Status |
|---|---|---|
| [01 — Data](notebooks/01_data_exploration.ipynb) | Sensors, sampling, gaps, frozen readings, operating cycle, failure signature, label completeness | ✅ |
| 02 — Evaluation protocol | Windowed features, train/test split in time, event-based metrics (detection, lead time, false alarms per week) | planned |
| 03 — Statistical baselines | Sustained-load rule, z-score, control charts (EWMA, CUSUM) | planned |
| 04 — Classical ML | Isolation Forest, One-Class SVM, Local Outlier Factor, PCA reconstruction | planned |
| 05 — Time series methods | Forecast residuals, Matrix Profile | planned |
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
│   └── plotstyle.py    # shared figure style
├── notebooks/          # analysis, 01 → 07
├── tests/              # unit tests
├── docs/figures/
└── .github/workflows/  # CI
```

## Stack

Python 3.12 · pandas · NumPy · PyArrow · scikit-learn · Matplotlib · pytest · GitHub Actions

## Author

**Ahmed Mimouni** — [github.com/mimouniahmed](https://github.com/mimouniahmed)

## License

Released under the [MIT License](LICENSE). The MetroPT-3 dataset is distributed by UCI under CC BY 4.0 and is not redistributed in this repository.
