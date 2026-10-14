import warnings

import numpy as np
import pandas as pd

from metrobench import ml

INDEX = pd.date_range("2020-01-01", periods=3000, freq="1min")


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=len(INDEX))
    X = pd.DataFrame({"a": a, "b": a + rng.normal(scale=0.2, size=len(INDEX)), "c": rng.normal(size=len(INDEX))},
                     index=INDEX)
    X.iloc[2500:2510] = [6.0, -6.0, 0.0]  # breaks the a-b correlation
    X.iloc[100:105] = np.nan
    return X, X.iloc[:2000]


def test_detectors_score_the_injected_anomaly_highest_and_keep_nans():
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    X, train = make_data()
    # The injected point mostly breaks the a-b correlation. Isolation Forest, with axis-parallel
    # splits, ranks it lower than the methods that model distances or correlations.
    expected_percentile = {"if": 0.95, "svm": 0.999, "lof": 0.999, "pca": 0.999}
    scores = {"if": ml.isolation_forest(X, train), "svm": ml.one_class_svm(X, train, max_train=1000),
              "lof": ml.local_outlier_factor(X, train, n_neighbors=20), "pca": ml.pca_reconstruction(X, train)[0]}
    for name, score in scores.items():
        assert score.index.equals(X.index)
        assert score.iloc[100:105].isna().all()
        assert score.iloc[2500:2510].min() > score.iloc[:2000].quantile(expected_percentile[name]), name


def test_pca_contributions_add_up_to_the_score():
    X, train = make_data()
    score, contributions = ml.pca_reconstruction(X, train, variance=0.9)
    np.testing.assert_allclose(contributions.sum(axis=1, min_count=1), score)


def test_drift_adjusted_follows_a_slow_trend_and_floors_the_scale():
    hours = len(INDEX) / 60
    trend = pd.Series(np.linspace(0, 10, len(INDEX)), index=INDEX)
    X = pd.DataFrame({"trend": trend, "flat": 1.0}, index=INDEX)
    adjusted = ml.drift_adjusted(X, X, window="12h", min_hours=6)
    assert hours > 12
    late = adjusted.iloc[-60:]
    assert late["trend"].abs().max() < 1e3  # bounded, although the rolling dispersion is tiny
    assert not np.isinf(late["flat"]).any()  # a constant feature gives NaN (no information), never infinity
