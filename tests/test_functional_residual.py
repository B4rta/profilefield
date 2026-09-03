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
    assert np.array_equal(project_monotone_profiles(invalid), [[0.0, 2.0, 2.0, 4.0]])


def test_residual_mean_weight_is_selected_only_from_supplied_validation_values() -> None:
    context = np.array([[0.0, 1.0], [0.0, 1.0]])
    correction = np.array([[0.0, 2.0], [0.0, 2.0]])
    truth = np.array([[0.0, 2.0], [0.0, 2.0]])
    assert select_residual_mean_weight(context, correction, truth) == 0.5
