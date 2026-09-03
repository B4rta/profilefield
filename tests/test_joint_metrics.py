from __future__ import annotations

import numpy as np

from profilefield.evaluation.joint import derived_profile_rows, profile_variogram_score
from profilefield.evaluation.spatial import posterior_cross_variogram_error


def test_profile_variogram_score_is_zero_for_perfect_ensemble() -> None:
    truth = np.array([[1.0, 3.0, 6.0], [2.0, 2.5, 5.0]])
    samples = np.stack([truth, truth], axis=0)
    assert profile_variogram_score(truth, samples) == 0.0


def test_derived_thresholds_are_computed_from_training_profiles() -> None:
    percentiles = np.array([0.0, 25.0, 50.0, 75.0, 98.0, 100.0])
    train = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            [0.0, 3.0, 6.0, 9.0, 12.0, 15.0],
        ]
    )
    test = train[:2]
    samples = np.stack([test, test], axis=0)
    rows, thresholds = derived_profile_rows("perfect", train, test, samples, percentiles)
    assert thresholds["rh98_q50"] == 8.0
    assert thresholds["upper_canopy_depth_q50"] == 4.0
    assert any(row["metric"] == "event_brier" for row in rows)
    assert (
        next(
            row["value"]
            for row in rows
            if row["scope"] == "rh98" and row["metric"] == "derived_rmse"
        )
        == 0.0
    )


def test_posterior_spatial_variogram_score_is_zero_for_perfect_joint_draws() -> None:
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    truth = np.array([[0.0, 1.0], [1.0, 2.0], [1.5, 4.0], [3.0, 5.0]])
    samples = np.stack([truth, truth], axis=0)
    errors = posterior_cross_variogram_error(
        coordinates, truth, samples, n_bins=2, max_pairs=100, max_draws=2
    )
    assert np.isclose(errors["posterior_variogram_mismatch"], 0.0)
    assert np.isclose(errors["posterior_cross_variogram_mismatch"], 0.0)
