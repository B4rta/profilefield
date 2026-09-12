import numpy as np
import pytest

from profilefield.evaluation.ensemble import empirical_crps, ensemble_calibration_rows


def test_crps_matches_explicit_pairwise_distribution_and_fair_estimator() -> None:
    rng = np.random.default_rng(31)
    draws, truth = rng.normal(size=(7, 3, 2)), rng.normal(size=(3, 2))
    first = np.abs(draws - truth).mean(axis=0)
    pair_sum = np.abs(draws[:, None] - draws[None, :]).sum(axis=(0, 1))
    np.testing.assert_allclose(empirical_crps(truth, draws), first - pair_sum / (2 * 49))
    np.testing.assert_allclose(empirical_crps(truth, draws, fair=True), first - pair_sum / (2 * 42))


def test_single_draw_crps_is_absolute_error() -> None:
    np.testing.assert_allclose(empirical_crps(np.zeros((1, 2)), np.array([[[2., -3.]]])), [[2., 3.]])
    with pytest.raises(ValueError, match="at least two"):
        empirical_crps(np.zeros((1, 2)), np.zeros((1, 1, 2)), fair=True)


def test_quantile_coverage_exposes_difference_between_marginal_and_whole_profile() -> None:
    draws = np.array([[[-1., -1.], [-1., -1.]], [[1., 1.], [1., 1.]]])
    rows = ensemble_calibration_rows("m", np.array([[0., 2.], [0., 0.]]), draws, (0.9,))
    assert next(r for r in rows if r["output"] == "all")["empirical_coverage"] == 0.75
    joint = next(r for r in rows if r["output"] == "whole_profile")
    assert joint["empirical_coverage"] == 0.5
    assert joint["coverage_error"] is None
