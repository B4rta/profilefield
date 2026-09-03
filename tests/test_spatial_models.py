from __future__ import annotations

import numpy as np
import pytest

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
