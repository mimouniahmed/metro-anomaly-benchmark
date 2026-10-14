# MetroPT Anomaly Benchmark — Detecting air leaks on a metro train compressor

[![tests](https://github.com/mimouniahmed/metro-anomaly-benchmark/actions/workflows/tests.yml/badge.svg)](https://github.com/mimouniahmed/metro-anomaly-benchmark/actions/workflows/tests.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**English** · [Français](README.fr.md)

Benchmark of **34 anomaly detection configurations** from five families (expert rule, statistical process control, classical machine learning, time series methods, deep learning) on **real industrial sensor data**: 7 months of readings from the air production unit of a Porto metro train, with four reported air leaks. Every method runs under the same protocol: chronological split, threshold calibrated on training data only, event-based scoring.

**Question**: which method detects the leaks earliest, with the fewest false alarms, and is the added complexity of machine learning worth it?

## Key results

| | |
|---|---|
| **Recommendation** | CUSUM chart on motor current for early warning, confirmed by an oil-temperature z-score against the previous 7 days: **4/4 failures, 0.71 false alarms per week, failure 4 flagged 12.9 h before its reported start** |
| **Best detectors** | 18 of 34 configurations detect all four failures. The best of each family reach 0.3–0.8 false alarms per week; the two statistical detectors run in under half a second on one signal each |
| **Complexity** | Machine learning and deep learning match the statistical detectors without beating them, and cost seeds, training time and a curated training set |
| **Main lesson** | Unreported leaks in the training data matter more than the algorithm: they blind the Matrix Profile (0/4) and the autoencoders (1–2/4) until removed, and gave LOF a success it loses once removed |
| **Uncertainty** | With four failures, 4/4 is compatible with a true detection rate as low as 40%, and the false-alarm intervals of the five best detectors overlap: only coarse rankings are established |

![Detectors that find all four failures](docs/figures/07_summary.png)

➡️ **[Executive report](docs/report.md)** (question, protocol, results, recommendation, limitations)

## Technical highlights

- **Data understanding before modelling**
  - sampling every ~10 s, not 1 Hz as documented;
  - 18% of the period missing, 7 days of frozen sensor values;
  - the failure signature (the compressor stops cycling and stays loaded) and its physics;
  - 8 leak-like episodes missing from the failure reports.
- **A protocol that resists leakage and luck**
  - backward-looking window features, with a unit test that changes future data;
  - chronological split, thresholds calibrated on training data under an alarm budget;
  - event-based metrics with lead times, and a chance level computed for random alarms.
- **Statistical process control**: robust z-scores, Hotelling T², EWMA and CUSUM charts, a drift-adjusted reference that cuts false alarms fivefold.
- **Classical ML**
  - Isolation Forest, One-Class SVM, LOF, PCA, each run with five seeds;
  - alarms explained by a per-feature breakdown of the PCA reconstruction error.
- **Time series and deep learning**
  - forecast residuals (gradient boosting) and Matrix Profile (stumpy, with reference cleaning);
  - dense and LSTM autoencoders (PyTorch, early stopping).
- **Diagnosis, not just scores**
  - why multivariate distances fail (one heavy-tailed feature);
  - why the nearest neighbour of a leak is a leak;
  - why an autoencoder reconstructs the faults it has seen;
  - why LOF succeeded for the wrong reason.
- **Honest conclusions**: Clopper-Pearson and Poisson intervals, a selection-effect simulation for 34 configurations, sensitivity analyses kept separate from model selection.
- **Engineering**: Python package, 32 unit tests, GitHub Actions CI, result tables saved per notebook.

## Approach

| Notebook | Content | Main result |
|---|---|---|
| [01 — Data](notebooks/01_data_exploration.ipynb) | Sensors, sampling, gaps, frozen readings, operating cycle, failure signature, label completeness | A leak is a collective anomaly; the failure reports miss 8 similar episodes |
| [02 — Evaluation protocol](notebooks/02_evaluation_protocol.ipynb) | One-minute grid, 18 window features, chronological split, event-based scoring, threshold calibration, chance level | Raw readings detect 0/4, 2-hour windows 4/4 |
| [03 — Statistical baselines](notebooks/03_statistical_baselines.ipynb) | Expert rule, robust z-score, Hotelling T², EWMA, CUSUM, rolling-baseline z-score | Oil z-score vs. last 7 days: 4/4, 0.28 false alarms/week; CUSUM: 12.9 h early warning |
| [04 — Classical ML](notebooks/04_classical_ml.ipynb) | Isolation Forest, One-Class SVM, LOF, PCA; raw vs. drift-adjusted inputs; seed variability; alarm explanation | LOF matches the best charts without beating them |
| [05 — Time series methods](notebooks/05_time_series_methods.ipynb) | Forecast residuals, Matrix Profile (raw/z-normalised, full/cleaned reference), cleaning cross-check on LOF | A contaminated reference blinds the Matrix Profile; cleaning fixes it but hurts LOF |
| [06 — Deep learning](notebooks/06_deep_learning.ipynb) | Dense and LSTM autoencoders, full vs. cleaned training data, reconstruction analysis | 1–2/4 on full data, 4/4 once cleaned, no early warning |
| [07 — Results](notebooks/07_results.ipynb) | All 34 configurations, confidence intervals, selection effect, alarm-budget robustness, cost of complexity, recommendation | CUSUM + drift-adjusted oil score |

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
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu   # CPU build is enough
pip install -r requirements.txt
pip install -e .

pytest   # 32 tests
```

The dataset (218 MB) is downloaded automatically on first use by `metrobench.data.load()` and cached as Parquet (22 MB) in `data/`. Run the notebooks in order, 01 → 07; each saves its results table in `results/`, which notebook 07 combines.

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
├── docs/
│   ├── report.md       # executive report
│   └── figures/
└── .github/workflows/  # CI
```

## Stack

Python 3.12 · pandas · NumPy · PyArrow · SciPy · scikit-learn · stumpy · PyTorch · Matplotlib · pytest · GitHub Actions

## Author

**Ahmed Mimouni** — [github.com/mimouniahmed](https://github.com/mimouniahmed)

## License

Released under the [MIT License](LICENSE). The MetroPT-3 dataset is distributed by UCI under CC BY 4.0 and is not redistributed in this repository.
