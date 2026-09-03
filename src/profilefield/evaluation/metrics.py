"""Probabilistic and latent-distribution metrics."""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist
from scipy.special import ndtr

FloatArray = NDArray[np.float64]


def gaussian_nll(truth: FloatArray, mean: FloatArray, variance: FloatArray) -> float:
    safe_variance = np.maximum(np.asarray(variance, dtype=np.float64), 1e-10)
    residual = np.asarray(truth, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    return float(np.mean(0.5 * (np.log(2.0 * np.pi * safe_variance) + residual**2 / safe_variance)))


def gaussian_crps(truth: FloatArray, mean: FloatArray, variance: FloatArray) -> FloatArray:
    standard_deviation = np.sqrt(np.maximum(variance, 1e-10))
    z = (truth - mean) / standard_deviation
    density = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    return standard_deviation * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * density - 1.0 / np.sqrt(np.pi))


def energy_score(truth: FloatArray, samples: FloatArray) -> float:
    """Mean multivariate energy score across spatial locations."""
    observed = np.asarray(truth, dtype=np.float64)
    draws = np.asarray(samples, dtype=np.float64)
    if draws.ndim != 3 or draws.shape[1:] != observed.shape:
        raise ValueError("Samples must have shape [draw, sample, output]")
    scores = []
    for index in range(len(observed)):
        local = draws[:, index, :]
        first = np.linalg.norm(local - observed[index], axis=1).mean()
        # pdist avoids materializing [draw, draw, output]; convert its off-diagonal
        # average to the average over the full matrix, whose diagonal is zero.
        pairwise = float(pdist(local).mean()) * (len(local) - 1) / len(local)
        scores.append(first - 0.5 * pairwise)
    return float(np.mean(scores))


def correlation_from_covariance(covariance: FloatArray) -> FloatArray:
    diagonal = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    return covariance / np.outer(diagonal, diagonal)


def covariance_error(estimate: FloatArray, truth: FloatArray) -> dict[str, float]:
    difference = np.asarray(estimate) - np.asarray(truth)
    denominator = max(float(np.linalg.norm(truth, ord="fro")), 1e-12)
    estimate_correlation = correlation_from_covariance(estimate)
    truth_correlation = correlation_from_covariance(truth)
    return {
        "covariance_frobenius": float(np.linalg.norm(difference, ord="fro")),
        "covariance_relative_frobenius": float(np.linalg.norm(difference, ord="fro") / denominator),
        "correlation_frobenius": float(
            np.linalg.norm(estimate_correlation - truth_correlation, ord="fro")
        ),
    }


def effective_rank(covariance: FloatArray) -> float:
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = eigenvalues.sum()
    if total <= 1e-12:
        return 0.0
    probabilities = eigenvalues / total
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-15)))
    return float(np.exp(entropy))


def calibration_rows(
    model_name: str,
    truth: FloatArray,
    mean: FloatArray,
    variance: FloatArray,
    levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95),
) -> list[dict[str, Any]]:
    from scipy.stats import norm

    rows: list[dict[str, Any]] = []
    standard_deviation = np.sqrt(np.maximum(variance, 1e-10))
    for level in levels:
        critical = float(norm.ppf((1.0 + level) / 2.0))
        lower = mean - critical * standard_deviation
        upper = mean + critical * standard_deviation
        covered = (truth >= lower) & (truth <= upper)
        widths = upper - lower
        for output in range(truth.shape[1]):
            empirical = float(covered[:, output].mean())
            rows.append(
                {
                    "model": model_name,
                    "output": output,
                    "nominal_coverage": level,
                    "empirical_coverage": empirical,
                    "coverage_error": abs(empirical - level),
                    "mean_interval_width": float(widths[:, output].mean()),
                }
            )
        empirical_all = float(covered.mean())
        rows.append(
            {
                "model": model_name,
                "output": "all",
                "nominal_coverage": level,
                "empirical_coverage": empirical_all,
                "coverage_error": abs(empirical_all - level),
                "mean_interval_width": float(widths.mean()),
            }
        )
    return rows


def latent_metric_rows(
    model_name: str,
    truth: FloatArray,
    mean: FloatArray,
    variance: FloatArray,
    samples: FloatArray,
    output_covariance: FloatArray,
    true_covariance: FloatArray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    crps = gaussian_crps(truth, mean, variance)
    for output in range(truth.shape[1]):
        residual = mean[:, output] - truth[:, output]
        rows.extend(
            [
                {
                    "model": model_name,
                    "scope": f"latent_{output}",
                    "metric": "rmse",
                    "value": float(np.sqrt(np.mean(residual**2))),
                },
                {
                    "model": model_name,
                    "scope": f"latent_{output}",
                    "metric": "crps",
                    "value": float(crps[:, output].mean()),
                },
                {
                    "model": model_name,
                    "scope": f"latent_{output}",
                    "metric": "nll",
                    "value": gaussian_nll(truth[:, output], mean[:, output], variance[:, output]),
                },
            ]
        )
    errors = covariance_error(output_covariance, true_covariance)
    summary = {
        "rmse": float(np.sqrt(np.mean((mean - truth) ** 2))),
        "crps": float(crps.mean()),
        "nll": gaussian_nll(truth, mean, variance),
        "energy_score": energy_score(truth, samples),
        "estimated_effective_rank": effective_rank(output_covariance),
        "true_effective_rank": effective_rank(true_covariance),
        **errors,
    }
    rows.extend(
        {"model": model_name, "scope": "all", "metric": metric, "value": value}
        for metric, value in summary.items()
    )
    for left, right in combinations_with_replacement(range(truth.shape[1]), 2):
        rows.append(
            {
                "model": model_name,
                "scope": f"covariance_{left}_{right}",
                "metric": "estimated",
                "value": float(output_covariance[left, right]),
            }
        )
        rows.append(
            {
                "model": model_name,
                "scope": f"covariance_{left}_{right}",
                "metric": "truth",
                "value": float(true_covariance[left, right]),
            }
        )
    return rows
