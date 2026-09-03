"""Synthetic correlated spatial latent fields with a deterministic profile decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from scipy.special import expit

from profilefield.reproducibility import sha256_json

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticDataset:
    coordinates: FloatArray
    latent_true: FloatArray
    latent_observed: FloatArray
    profiles_true: FloatArray
    sample_ids: tuple[str, ...]
    coregionalization: FloatArray
    decoder_weights: FloatArray
    decoder_bias: FloatArray
    manifest: dict[str, Any]

    def decode(self, latent: FloatArray) -> FloatArray:
        logits = np.asarray(latent, dtype=np.float64) @ self.decoder_weights + self.decoder_bias
        increments = np.logaddexp(0.0, logits)
        cumulative = np.cumsum(increments, axis=-1)
        return cumulative / np.maximum(cumulative[..., -1:], 1e-12)


def matern_covariance(
    coordinates: FloatArray,
    lengthscale: tuple[float, float] | list[float],
    nu: float,
) -> FloatArray:
    scaled = np.asarray(coordinates, dtype=np.float64) / np.asarray(lengthscale, dtype=np.float64)
    distance = cdist(scaled, scaled)
    if nu == 0.5:
        return np.exp(-distance)
    if nu == 1.5:
        root = np.sqrt(3.0) * distance
        return (1.0 + root) * np.exp(-root)
    if nu == 2.5:
        root = np.sqrt(5.0) * distance
        return (1.0 + root + 5.0 * distance**2 / 3.0) * np.exp(-root)
    raise ValueError("Only Matérn nu=0.5, nu=1.5 and nu=2.5 are supported")


def generate_synthetic(config: dict[str, Any], seed: int) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    n_side = int(config["n_side"])
    output_dim = int(config["output_dim"])
    profile_points = int(config["profile_points"])
    axis = np.linspace(0.0, 1.0, n_side)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    coordinates = np.column_stack([xx.ravel(), yy.ravel()])
    jitter = float(config.get("jitter", 0.0))
    coordinates = np.clip(coordinates + rng.normal(0.0, jitter, coordinates.shape), 0.0, 1.0)

    coregionalization = np.asarray(config["coregionalization"], dtype=np.float64)
    if coregionalization.shape != (output_dim, output_dim):
        raise ValueError("Coregionalization matrix does not match output_dim")
    if not np.allclose(coregionalization, coregionalization.T):
        raise ValueError("Coregionalization matrix must be symmetric")
    if np.min(np.linalg.eigvalsh(coregionalization)) <= 0:
        raise ValueError("Coregionalization matrix must be positive definite")
    off_diagonal = coregionalization - np.diag(np.diag(coregionalization))
    require_dependence = bool(config.get("require_cross_output_dependence", True))
    if require_dependence and np.max(np.abs(off_diagonal)) < 0.1:
        raise ValueError("Synthetic truth must contain meaningful cross-output dependence")

    spatial_covariance = matern_covariance(
        coordinates,
        config["lengthscale"],
        float(config["matern_nu"]),
    )
    spatial_cholesky = np.linalg.cholesky(spatial_covariance + np.eye(len(coordinates)) * 1e-6)
    output_cholesky = np.linalg.cholesky(coregionalization)
    latent_true = (
        spatial_cholesky @ rng.standard_normal((len(coordinates), output_dim)) @ output_cholesky.T
    )
    noise_std = float(config["noise_std"])
    latent_observed = latent_true + rng.normal(0.0, noise_std, latent_true.shape)

    decoder_weights = rng.normal(0.0, 0.45, size=(output_dim, profile_points))
    trend = np.linspace(-1.25, 0.75, profile_points)
    decoder_bias = trend + 0.2 * expit(np.linspace(-3.0, 3.0, profile_points))
    logits = latent_true @ decoder_weights + decoder_bias
    increments = np.logaddexp(0.0, logits)
    profiles_true = np.cumsum(increments, axis=1)
    profiles_true /= np.maximum(profiles_true[:, -1:], 1e-12)

    parameters = {
        "generator": "separable_matern_coregionalized_v1",
        "seed": seed,
        "n_samples": len(coordinates),
        "n_side": n_side,
        "output_dim": output_dim,
        "profile_points": profile_points,
        "matern_nu": float(config["matern_nu"]),
        "lengthscale": list(config["lengthscale"]),
        "noise_std": noise_std,
        "coordinate_jitter": jitter,
        "coregionalization": coregionalization.tolist(),
        "require_cross_output_dependence": require_dependence,
    }
    manifest = {**parameters, "parameter_hash": sha256_json(parameters)}
    return SyntheticDataset(
        coordinates=coordinates,
        latent_true=latent_true,
        latent_observed=latent_observed,
        profiles_true=profiles_true,
        sample_ids=tuple(f"synthetic-{index:05d}" for index in range(len(coordinates))),
        coregionalization=coregionalization,
        decoder_weights=decoder_weights,
        decoder_bias=decoder_bias,
        manifest=manifest,
    )
