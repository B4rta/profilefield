from __future__ import annotations

import numpy as np

from profilefield.data.synthetic import generate_synthetic, matern_covariance

CONFIG = {
    "n_side": 6,
    "output_dim": 3,
    "profile_points": 12,
    "matern_nu": 1.5,
    "lengthscale": [0.2, 0.3],
    "noise_std": 0.05,
    "jitter": 0.0,
    "coregionalization": [[1.0, 0.7, -0.2], [0.7, 1.1, 0.4], [-0.2, 0.4, 0.8]],
}


def test_synthetic_truth_has_known_non_diagonal_covariance() -> None:
    dataset = generate_synthetic(CONFIG, 11)
    assert dataset.coregionalization.shape == (3, 3)
    assert (
        np.max(np.abs(dataset.coregionalization - np.diag(np.diag(dataset.coregionalization))))
        > 0.5
    )
    assert np.linalg.eigvalsh(dataset.coregionalization).min() > 0
    assert np.all(np.diff(dataset.profiles_true, axis=1) >= -1e-12)


def test_synthetic_generation_is_reproducible() -> None:
    left = generate_synthetic(CONFIG, 91)
    right = generate_synthetic(CONFIG, 91)
    assert np.array_equal(left.coordinates, right.coordinates)
    assert np.array_equal(left.latent_observed, right.latent_observed)
    assert left.manifest == right.manifest


def test_matern_covariance_is_positive_definite() -> None:
    coordinates = np.asarray([[0.0, 0.0], [0.3, 0.1], [0.7, 0.8], [1.0, 1.0]])
    for nu in (0.5, 1.5, 2.5):
        covariance = matern_covariance(coordinates, [0.4, 0.7], nu)
        assert np.linalg.eigvalsh(covariance).min() > 0


def test_independence_is_supported_as_an_explicit_negative_control() -> None:
    config = {
        **CONFIG,
        "coregionalization": np.diag([1.0, 1.1, 0.8]).tolist(),
        "require_cross_output_dependence": False,
    }
    dataset = generate_synthetic(config, 17)
    assert (
        np.count_nonzero(dataset.coregionalization - np.diag(np.diag(dataset.coregionalization)))
        == 0
    )
    assert dataset.manifest["require_cross_output_dependence"] is False
