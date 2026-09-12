from __future__ import annotations

import numpy as np
import pytest

from profilefield.profiles.functional_residual import (
    FunctionalResidualBasis,
    project_monotone_profiles,
    select_residual_mean_weight,
)


def test_functional_basis_is_train_only_and_decodes_shapes() -> None:
    context = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]])
    truth = context + np.array([[0.0, 0.2, 0.4, 0.6], [0.0, -0.1, 0.2, 0.4], [0.0, 0.1, 0.3, 0.8]])
    identifiers = ["train-a", "train-b", "train-c"]
    basis = FunctionalResidualBasis(3).fit(truth, context, identifiers)
    basis.assert_fit_on(set(identifiers))
    scores = basis.encode(truth, context)
    draws = np.stack([scores, scores], axis=0)
    decoded = basis.decode(draws, context[None, :, :])
    assert scores.shape == (3, 3)
    assert decoded.shape == (2, 3, 4)
    assert np.all(np.diff(decoded, axis=-1) >= 0.0)
    assert basis.score_variance_ratio_ is not None
    assert np.isclose(basis.score_variance_ratio_.sum(), 1.0)
    assert basis.reconstruction_variance_explained_ is not None
    assert basis.reconstruction_variance_explained_ > 0.9
    nugget = basis.sample_reconstruction_nugget(4, 5, np.random.default_rng(1))
    assert nugget.shape == (4, 5, 4)
    with pytest.raises(AssertionError, match="training set only"):
        basis.assert_fit_on({"train-a", "test-leak"})


def test_monotone_projection_preserves_valid_profiles() -> None:
    valid = np.array([[0.0, 1.0, 1.0, 3.0]])
    invalid = np.array([[-1.0, 2.0, 1.0, 4.0]])
    assert np.array_equal(project_monotone_profiles(valid), valid)
    assert np.array_equal(project_monotone_profiles(invalid), [[0.0, 1.5, 1.5, 4.0]])


def test_isotonic_projection_is_nearest_bounded_profile() -> None:
    # Clipping before isotonic regression would incorrectly give [0.5, 0.5].
    values = np.array([[[1.0, -2.0], [3.0, 1.0]], [[-3.0, -1.0], [0.0, 4.0]]])
    projected = project_monotone_profiles(values)
    assert np.array_equal(projected, [[[0.0, 0.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 4.0]]])
    assert np.array_equal(project_monotone_profiles(projected), projected)
    assert not np.shares_memory(values, projected)
    # The variational inequality characterizes Euclidean projection onto a
    # closed convex set: <values - projection, feasible - projection> <= 0.
    rng = np.random.default_rng(124)
    feasible = np.sort(rng.uniform(0.0, 8.0, size=(100, *values.shape)), axis=-1)
    products = np.sum((values - projected) * (feasible - projected), axis=-1)
    assert np.all(products <= 1e-12)


def test_signed_isotonic_projection_preserves_negative_relative_heights() -> None:
    valid = np.array([-3.0, -1.0, 2.0])
    assert np.array_equal(project_monotone_profiles(valid, lower_bound=None), valid)
    assert np.array_equal(
        project_monotone_profiles(np.array([-1.0, -3.0, 2.0]), lower_bound=None),
        [-2.0, -2.0, 2.0],
    )
    assert np.array_equal(project_monotone_profiles(valid, lower_bound=-2.0), [-2.0, -1.0, 2.0])


@pytest.mark.parametrize("values", [np.array(1.0), np.array([]), np.array([0.0, np.nan])])
def test_isotonic_projection_rejects_invalid_profiles(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        project_monotone_profiles(values)


def test_isotonic_projection_rejects_nonfinite_bound() -> None:
    with pytest.raises(ValueError, match="lower_bound"):
        project_monotone_profiles(np.array([0.0, 1.0]), lower_bound=np.inf)


def test_residual_mean_weight_is_selected_only_from_supplied_validation_values() -> None:
    context = np.array([[0.0, 1.0], [0.0, 1.0]])
    correction = np.array([[0.0, 2.0], [0.0, 2.0]])
    truth = np.array([[0.0, 2.0], [0.0, 2.0]])
    assert select_residual_mean_weight(context, correction, truth) == 0.5


def test_residual_mean_weight_respects_signed_profile_constraint() -> None:
    context = np.array([[-3.0, -2.0]])
    correction = np.array([[2.0, 1.0]])
    truth = context + correction
    assert select_residual_mean_weight(context, correction, truth, lower_bound=None) == 1.0
