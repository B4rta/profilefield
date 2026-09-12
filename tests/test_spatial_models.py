from __future__ import annotations

import gpytorch
import numpy as np
import pytest
import torch

from profilefield.spatial.independent_gp import IndependentSVGP
from profilefield.spatial.lmc_svgp import LMCSVGP


@pytest.fixture(scope="module")
def tiny_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(7)
    coordinates = rng.random((14, 2))
    common = np.sin(coordinates[:, 0] * 4.0) + np.cos(coordinates[:, 1] * 2.0)
    targets = np.column_stack([common, 0.8 * common + 0.1 * coordinates[:, 0]])
    return coordinates, targets, [f"id-{index}" for index in range(len(coordinates))]


@pytest.mark.parametrize("kind", ["independent", "lmc"])
def test_posterior_dimensions_sampling_and_positive_variance(
    tiny_data: tuple[np.ndarray, np.ndarray, list[str]], kind: str
) -> None:
    coordinates, targets, ids = tiny_data
    model: IndependentSVGP | LMCSVGP
    if kind == "independent":
        model = IndependentSVGP(2, 6, training_steps=3, seed=41)
    else:
        model = LMCSVGP(2, 2, 6, training_steps=3, seed=41)
    prediction = model.fit(coordinates, targets, ids).predict(coordinates[:4], 5)
    assert prediction.mean.shape == (4, 2)
    assert prediction.variance.shape == (4, 2)
    assert prediction.samples.shape == (5, 4, 2)
    assert prediction.full_covariance is not None
    assert prediction.full_covariance.shape == (8, 8)
    assert np.all(prediction.variance > 0)


def test_fixed_seed_lmc_is_reproducible(
    tiny_data: tuple[np.ndarray, np.ndarray, list[str]],
) -> None:
    coordinates, targets, ids = tiny_data
    left = LMCSVGP(2, 2, 5, training_steps=2, seed=123).fit(coordinates, targets, ids)
    right = LMCSVGP(2, 2, 5, training_steps=2, seed=123).fit(coordinates, targets, ids)
    left_prediction = left.predict(coordinates[:3], 3)
    right_prediction = right.predict(coordinates[:3], 3)
    assert np.allclose(left_prediction.mean, right_prediction.mean)
    assert np.allclose(left_prediction.variance, right_prediction.variance)


@pytest.mark.parametrize("kind", ["independent", "lmc"])
def test_joint_draws_reproduce_location_and_output_covariance(
    tiny_data: tuple[np.ndarray, np.ndarray, list[str]], kind: str
) -> None:
    coordinates, targets, ids = tiny_data
    model: IndependentSVGP | LMCSVGP
    if kind == "independent":
        model = IndependentSVGP(2, 5, training_steps=2, seed=91)
    else:
        model = LMCSVGP(2, 2, 5, training_steps=2, seed=91)
    model.fit(coordinates, targets, ids)
    query = np.array([[0.4, 0.4], [0.405, 0.405]])
    torch.manual_seed(431)
    reference = model.predict(query, posterior_samples=4, full_covariance=True)
    torch.manual_seed(881)
    sampled = model.predict(query, posterior_samples=4096, full_covariance=False)
    assert sampled.full_covariance is None
    assert reference.full_covariance is not None
    expected = reference.full_covariance
    empirical = np.cov(sampled.samples.reshape(4096, -1), rowvar=False)
    scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
    assert np.max(np.abs(empirical - expected) / scale) < 0.09
    assert np.allclose(np.diag(expected).reshape(2, 2), sampled.variance)
    # Suppressing the returned covariance must not turn joint spatial draws into
    # independent-location draws. Observation noise still makes duplicates differ.
    assert abs(expected[0, 2]) / scale[0, 2] > 0.05
    if kind == "independent":
        assert expected[0, 1] == 0.0
        assert expected[0, 3] == 0.0
    else:
        assert abs(expected[0, 1]) / scale[0, 1] > 0.05


@pytest.mark.parametrize("kind", ["independent", "lmc"])
def test_sampler_preserves_variance_even_when_global_root_budget_is_tiny(
    tiny_data: tuple[np.ndarray, np.ndarray, list[str]], kind: str
) -> None:
    coordinates, targets, ids = tiny_data
    model = (IndependentSVGP(2, 5, training_steps=2, seed=61) if kind == "independent"
             else LMCSVGP(2, 2, 5, training_steps=2, seed=61))
    model.fit(coordinates, targets, ids)
    query = np.random.default_rng(2).uniform(size=(48, 2))
    with gpytorch.settings.max_cholesky_size(0), gpytorch.settings.max_root_decomposition_size(3):
        torch.manual_seed(111)
        prediction = model.predict(query, posterior_samples=4096, full_covariance=False)
    variance_ratio = prediction.samples.var(axis=0) / prediction.variance
    np.testing.assert_allclose(variance_ratio, 1., rtol=0.09)
