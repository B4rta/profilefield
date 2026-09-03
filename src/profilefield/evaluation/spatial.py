"""Empirical variograms, cross-variograms, and local-gradient diagnostics."""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

FloatArray = NDArray[np.float64]


def empirical_cross_variograms(
    coordinates: FloatArray,
    values: FloatArray,
    n_bins: int,
) -> list[dict[str, Any]]:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    row, column = np.triu_indices(len(coordinates), k=1)
    distances = squareform(pdist(coordinates))[row, column]
    maximum = float(distances.max()) if len(distances) else 1.0
    edges = np.linspace(0.0, maximum + np.finfo(float).eps, n_bins + 1)
    bins = np.clip(np.digitize(distances, edges) - 1, 0, n_bins - 1)
    rows: list[dict[str, Any]] = []
    differences = values[row] - values[column]
    for left, right in combinations_with_replacement(range(values.shape[1]), 2):
        semivariance = 0.5 * differences[:, left] * differences[:, right]
        for bin_index in range(n_bins):
            selected = bins == bin_index
            rows.append(
                {
                    "left_output": left,
                    "right_output": right,
                    "bin": bin_index,
                    "distance": float(distances[selected].mean()) if selected.any() else np.nan,
                    "semivariance": float(semivariance[selected].mean())
                    if selected.any()
                    else np.nan,
                    "pair_count": int(selected.sum()),
                }
            )
    return rows


def cross_variogram_error(
    truth_rows: list[dict[str, Any]],
    estimate_rows: list[dict[str, Any]],
) -> dict[str, float]:
    truth = np.asarray([row["semivariance"] for row in truth_rows], dtype=np.float64)
    estimate = np.asarray([row["semivariance"] for row in estimate_rows], dtype=np.float64)
    left = np.asarray([row["left_output"] for row in truth_rows])
    right = np.asarray([row["right_output"] for row in truth_rows])
    valid = np.isfinite(truth) & np.isfinite(estimate)
    auto = valid & (left == right)
    cross = valid & (left != right)

    def normalized_rmse(mask: NDArray[np.bool_]) -> float:
        if not mask.any():
            return float("nan")
        scale = max(float(np.sqrt(np.mean(truth[mask] ** 2))), 1e-12)
        return float(np.sqrt(np.mean((estimate[mask] - truth[mask]) ** 2)) / scale)

    return {
        "variogram_mismatch": normalized_rmse(auto),
        "cross_variogram_mismatch": normalized_rmse(cross),
    }


def posterior_cross_variogram_error(
    coordinates: FloatArray,
    truth: FloatArray,
    samples: FloatArray,
    n_bins: int,
    *,
    seed: int = 0,
    max_pairs: int = 50_000,
    max_draws: int = 16,
) -> dict[str, float]:
    """Dependence-sensitive spatial score from joint conditional simulations.

    A deterministic subset of location pairs bounds memory and runtime on the
    full GEDI test set. Truth and posterior use the identical pairs and bins.
    """

    coordinate_array = np.asarray(coordinates, dtype=np.float64)
    observed = np.asarray(truth, dtype=np.float64)
    draws = np.asarray(samples, dtype=np.float64)
    if observed.ndim != 2 or draws.ndim != 3 or draws.shape[1:] != observed.shape:
        raise ValueError("Samples must have shape [draw, sample, output]")
    row, column = np.triu_indices(len(coordinate_array), k=1)
    if len(row) == 0:
        return {
            "posterior_variogram_mismatch": float("nan"),
            "posterior_cross_variogram_mismatch": float("nan"),
        }
    rng = np.random.default_rng(seed)
    if len(row) > max_pairs:
        selected = np.sort(rng.choice(len(row), size=max_pairs, replace=False))
        row, column = row[selected], column[selected]
    distances = np.linalg.norm(coordinate_array[row] - coordinate_array[column], axis=1)
    edges = np.linspace(0.0, float(distances.max()) + np.finfo(float).eps, n_bins + 1)
    bins = np.clip(np.digitize(distances, edges) - 1, 0, n_bins - 1)
    draw_indices = np.linspace(0, len(draws) - 1, min(max_draws, len(draws))).round().astype(int)
    truth_difference = observed[row] - observed[column]
    posterior_difference = draws[draw_indices][:, row, :] - draws[draw_indices][:, column, :]
    truth_values: list[float] = []
    posterior_values: list[float] = []
    auto_flags: list[bool] = []
    for left, right in combinations_with_replacement(range(observed.shape[1]), 2):
        truth_semivariance = 0.5 * truth_difference[:, left] * truth_difference[:, right]
        posterior_semivariance = 0.5 * np.mean(
            posterior_difference[:, :, left] * posterior_difference[:, :, right], axis=0
        )
        for bin_index in range(n_bins):
            in_bin = bins == bin_index
            if not in_bin.any():
                continue
            truth_values.append(float(np.mean(truth_semivariance[in_bin])))
            posterior_values.append(float(np.mean(posterior_semivariance[in_bin])))
            auto_flags.append(left == right)
    truth_array = np.asarray(truth_values)
    posterior_array = np.asarray(posterior_values)
    auto = np.asarray(auto_flags, dtype=bool)

    def normalized_rmse(mask: NDArray[np.bool_]) -> float:
        if not mask.any():
            return float("nan")
        scale = max(float(np.sqrt(np.mean(truth_array[mask] ** 2))), 1e-12)
        return float(np.sqrt(np.mean((posterior_array[mask] - truth_array[mask]) ** 2)) / scale)

    return {
        "posterior_variogram_mismatch": normalized_rmse(auto),
        "posterior_cross_variogram_mismatch": normalized_rmse(~auto),
    }


def local_gradient_distribution(coordinates: FloatArray, values: FloatArray) -> FloatArray:
    distance = squareform(pdist(coordinates))
    np.fill_diagonal(distance, np.inf)
    nearest = np.argmin(distance, axis=1)
    separation = np.maximum(distance[np.arange(len(coordinates)), nearest], 1e-12)
    return np.linalg.norm(values - values[nearest], axis=1) / separation
