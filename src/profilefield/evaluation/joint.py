"""Dependence-sensitive and derived-profile posterior diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from profilefield.evaluation.metrics import covariance_error

FloatArray = NDArray[np.float64]


def profile_variogram_score(
    truth: FloatArray,
    samples: FloatArray,
    output_indices: NDArray[np.int64] | None = None,
    power: float = 0.5,
) -> float:
    """Proper variogram score for within-profile dependence.

    The score compares observed and posterior expected pairwise increments across
    selected RH ordinates. Unlike marginal CRPS, it changes when the dependence
    between profile heights changes while all marginals remain fixed.
    """

    observed, draws = _validate_samples(truth, samples)
    indices = _indices(observed.shape[1], output_indices)
    observed = observed[:, indices]
    draws = draws[:, :, indices]
    left, right = np.triu_indices(len(indices), k=1)
    observed_increment = np.abs(observed[:, left] - observed[:, right]) ** power
    posterior_increment = np.mean(np.abs(draws[:, :, left] - draws[:, :, right]) ** power, axis=0)
    return float(np.mean((observed_increment - posterior_increment) ** 2))


def posterior_profile_covariance_rows(
    model_name: str,
    truth: FloatArray,
    samples: FloatArray,
    output_indices: NDArray[np.int64] | None = None,
) -> list[dict[str, Any]]:
    """Compare observed and posterior population covariance across profiles."""

    observed, draws = _validate_samples(truth, samples)
    indices = _indices(observed.shape[1], output_indices)
    true_covariance = np.atleast_2d(np.cov(observed[:, indices], rowvar=False))
    estimated_covariance = np.mean(
        [np.atleast_2d(np.cov(draw[:, indices], rowvar=False)) for draw in draws], axis=0
    )
    errors = covariance_error(estimated_covariance, true_covariance)
    return [
        {
            "model": model_name,
            "scope": "selected_rh_joint",
            "metric": f"posterior_{metric}",
            "value": value,
        }
        for metric, value in errors.items()
    ]


def derived_profile_rows(
    model_name: str,
    train_truth: FloatArray,
    test_truth: FloatArray,
    samples: FloatArray,
    rh_percentiles: FloatArray,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Score posterior distributions of physically interpretable functionals.

    Event thresholds are computed exclusively from training observations. The
    returned threshold dictionary is persisted as a diagnostic provenance item.
    """

    observed, draws = _validate_samples(test_truth, samples)
    train = np.asarray(train_truth, dtype=np.float64)
    percentiles = np.asarray(rh_percentiles, dtype=np.float64)
    if train.ndim != 2 or train.shape[1] != observed.shape[1]:
        raise ValueError("train_truth must share the test profile dimension")
    functions = _derived_functions(percentiles)
    rows: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    for name, function in functions.items():
        train_value = function(train)
        truth_value = function(observed)
        sample_value = function(draws)
        mean_value = np.mean(sample_value, axis=0)
        lower, upper = np.quantile(sample_value, [0.05, 0.95], axis=0)
        rows.extend(
            {
                "model": model_name,
                "scope": name,
                "metric": metric,
                "value": value,
            }
            for metric, value in (
                ("derived_rmse", float(np.sqrt(np.mean((mean_value - truth_value) ** 2)))),
                ("derived_mae", float(np.mean(np.abs(mean_value - truth_value)))),
                ("derived_crps", float(np.mean(_ensemble_crps(truth_value, sample_value)))),
                (
                    "derived_coverage_90",
                    float(np.mean((truth_value >= lower) & (truth_value <= upper))),
                ),
                ("derived_interval_width_90", float(np.mean(upper - lower))),
            )
        )
        for quantile in (0.50, 0.75):
            threshold = float(np.quantile(train_value, quantile))
            label = f"{name}_q{int(100 * quantile):02d}"
            thresholds[label] = threshold
            probability = np.mean(sample_value > threshold, axis=0)
            event = truth_value > threshold
            probability_safe = np.clip(probability, 1e-8, 1.0 - 1e-8)
            rows.extend(
                [
                    {
                        "model": model_name,
                        "scope": label,
                        "metric": "event_brier",
                        "value": float(np.mean((probability - event) ** 2)),
                    },
                    {
                        "model": model_name,
                        "scope": label,
                        "metric": "event_log_score",
                        "value": float(
                            -np.mean(
                                event * np.log(probability_safe)
                                + (~event) * np.log(1.0 - probability_safe)
                            )
                        ),
                    },
                ]
            )
    return rows, thresholds


def _derived_functions(percentiles: FloatArray) -> dict[str, Callable[[FloatArray], FloatArray]]:
    index = {value: int(np.argmin(np.abs(percentiles - value))) for value in (25, 50, 75, 98)}
    return {
        "rh50": lambda value: value[..., index[50]],
        "rh98": lambda value: value[..., index[98]],
        "upper_canopy_depth": lambda value: value[..., index[98]] - value[..., index[50]],
        "interquartile_depth": lambda value: value[..., index[75]] - value[..., index[25]],
        "profile_auc": lambda value: (
            np.trapezoid(value, percentiles, axis=-1)
            / max(float(percentiles[-1] - percentiles[0]), 1.0)
        ),
    }


def _ensemble_crps(truth: FloatArray, samples: FloatArray) -> FloatArray:
    """Exact empirical-ensemble CRPS in O(m log m) per observation."""

    draws = np.asarray(samples, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[1] != len(truth):
        raise ValueError("Derived samples must have shape [draw, sample]")
    ordered = np.sort(draws, axis=0)
    m = len(ordered)
    ranks = 2.0 * np.arange(1, m + 1, dtype=np.float64) - m - 1.0
    pair_term = np.sum(ranks[:, None] * ordered, axis=0) / (m * m)
    return np.mean(np.abs(draws - truth[None, :]), axis=0) - pair_term


def _validate_samples(truth: FloatArray, samples: FloatArray) -> tuple[FloatArray, FloatArray]:
    observed = np.asarray(truth, dtype=np.float64)
    draws = np.asarray(samples, dtype=np.float64)
    if observed.ndim != 2 or draws.ndim != 3 or draws.shape[1:] != observed.shape:
        raise ValueError("Samples must have shape [draw, sample, output]")
    if len(draws) < 2:
        raise ValueError("At least two posterior samples are required")
    return observed, draws


def _indices(output_count: int, supplied: NDArray[np.int64] | None) -> NDArray[np.int64]:
    if supplied is None:
        return np.arange(output_count, dtype=np.int64)
    indices = np.asarray(supplied, dtype=np.int64)
    if (
        indices.ndim != 1
        or len(indices) < 2
        or np.any(indices < 0)
        or np.any(indices >= output_count)
    ):
        raise ValueError("output_indices must contain at least two valid indices")
    return indices
