# Detecting air leaks on a metro train compressor — Executive report

## Question

Which anomaly detection method should monitor the air production unit of a Porto metro train, so that air leaks are flagged as early as possible with few false alarms? And is the added complexity of machine learning and deep learning worth it?

## Data

MetroPT-3 (UCI, CC BY 4.0): 1.5 million readings of 15 compressor signals, February–August 2020, recorded every ~10 seconds. The operator reported four air leaks.

Three properties of the data shaped the whole study:

- **A leak is a sustained state, not a spike.** During a leak the compressor stops cycling and stays under load (99.5% of the time vs. 15% normally); each individual reading stays within its normal range.
- **The failure reports are incomplete.** Eight more episodes with exactly the same signature, four of them in the training period, are not in the reports.
- **The data drift and have gaps.** The compressor works three times more from March than in February, the oil runs ~5 °C hotter in summer, 18% of the period is missing and 7 days of readings are frozen.

## Protocol

- **Inputs**: a one-minute grid, with gaps and frozen readings kept as missing; 18 backward-looking window features (30 min, 2 h).
- **Split**: chronological. Training from February 1 to April 10 (no reported failure), test from April 11 to September 1 (four failures).
- **Threshold**: calibrated on training data only, at most one alarm per week.
- **Scoring**: per failure event. A failure counts as detected if an alarm falls in [start − 12 h, end], and the lead time is measured. False alarms are counted per week.
- **Reporting**: every configuration tried (34) is reported, stochastic methods are run with 3–5 seeds, and the chance level is computed.

## Results

| Family | Best detector | Failures detected | False alarms / week | Warning on failure 4 |
|---|---|---|---|---|
| Expert rule | loaded ≥ 90% of the last 30 min | 4/4 | 1.25 | none |
| Statistical | oil temperature z-score against the previous 7 days | 4/4 | **0.28** | 0.3 h |
| Statistical | CUSUM chart on motor current | 4/4 | 0.64 | **12.9 h** |
| Classical ML | Local Outlier Factor, 18 features | 4/4 (5 seeds) | 0.55 (0.35–1.17) | 12.6 h |
| Time series | Matrix Profile on load state, cleaned reference | 4/4 | 0.71 | 11.1 h (z-normalised) |
| Deep learning | dense autoencoder, cleaned training data | 4/4 (3 seeds) | 0.83 | none |

![Summary](figures/07_summary.png)

## What the benchmark shows

1. **Simple, physically meaningful detectors are as good as anything else.** The two statistical detectors use one signal each, run in under half a second, are deterministic and explain their alarms.
2. **Curating the training data matters more than choosing the algorithm.** The unreported leaks in the training period blind the nearest-neighbour and reconstruction methods: the Matrix Profile detects 0/4 and the autoencoders 1–2/4 until those episodes are removed with the expert rule. The same contamination gave LOF a success it loses once they are removed.
3. **Four failures allow only a coarse ranking.** Detecting 4/4 is compatible with a true detection rate as low as 40%. The false-alarm intervals of the five best detectors overlap, and the best score is about what the luckiest of 34 equivalent detectors would show.

## Recommendation

Monitor the compressor with the **CUSUM chart on motor current** for early warning, and confirm with the **drift-adjusted oil temperature score**.

- **Combined result**: 4/4 failures, 0.71 false alarms per week (about three a month), failure 4 flagged 12.9 hours before its reported start.
- **Caveat**: the combination was chosen after seeing the test results, so these figures should be confirmed on new failures.
- **Why these two**: both components need no curated training set, are robust to the alarm budget, and point to one physical signal that a technician can check directly.

Machine learning and deep learning are not worth their cost on this machine. They are worth revisiting with more trains, more failures and other fault types.

## Limitations

- **Small sample**: four failures of one type, on one train.
- **Incomplete labels**: false-alarm rates are upper bounds.
- **One split**: a single chronological split was used.
- **Choices informed by the test period**: some decisions (detection window, combination) were informed by earlier analyses of the same test period, although every variant tried is reported.
