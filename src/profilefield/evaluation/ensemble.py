"""Scores and coverage for the actual empirical predictive distribution."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _validate(truth: FloatArray, samples: FloatArray) -> tuple[FloatArray, FloatArray]:
    observed, draws = np.asarray(truth, dtype=np.float64), np.asarray(samples, dtype=np.float64)
    if observed.ndim != 2 or draws.ndim != 3 or draws.shape[1:] != observed.shape:
        raise ValueError("Expected truth [case, output] and draws [draw, case, output]")
    if 0 in draws.shape or not np.isfinite(observed).all() or not np.isfinite(draws).all():
        raise ValueError("Truth and draws must be nonempty and finite")
    return observed, draws


def empirical_crps(truth: FloatArray, samples: FloatArray, *, fair: bool = False) -> FloatArray:
    """CRPS of the empirical CDF, or the unbiased iid-ensemble estimator.

    The default scores the distribution actually delivered to a user. ``fair``
    estimates the underlying distribution's score when draws are iid; it needs
    at least two draws. Sorting avoids allocating a quadratic pairwise tensor.
    """
    observed, draws = _validate(truth, samples)
    count = len(draws)
    if fair and count < 2:
        raise ValueError("Fair CRPS requires at least two draws")
    coefficients = (2 * np.arange(count) - count + 1)[:, None, None]
    denominator = count * (count - 1) if fair else count * count
    correction = np.sum(coefficients * np.sort(draws, axis=0), axis=0) / denominator
    return np.mean(np.abs(draws - observed[None, :, :]), axis=0) - correction


def ensemble_calibration_rows(
    model: str,
    truth: FloatArray,
    samples: FloatArray,
    levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
) -> list[dict[str, Any]]:
    """Marginal quantile-band coverage and descriptive whole-profile coverage.

    Whole-profile coverage of marginal bands is not a simultaneous confidence
    guarantee. Its nominal marginal level is retained explicitly for comparison.
    """
    observed, draws = _validate(truth, samples)
    rows: list[dict[str, Any]] = []
    for level in levels:
        if not 0 < level < 1:
            raise ValueError("Coverage levels must lie strictly between zero and one")
        lower, upper = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2], axis=0)
        covered = (observed >= lower) & (observed <= upper)
        for output in [*range(observed.shape[1]), "all"]:
            local = covered if output == "all" else covered[:, int(output)]
            width = upper - lower if output == "all" else upper[:, int(output)] - lower[:, int(output)]
            rows.append({"model": model, "output": output, "interval_kind": "empirical_quantile",
                         "nominal_coverage": level, "empirical_coverage": float(local.mean()),
                         "coverage_error": abs(float(local.mean()) - level),
                         "mean_interval_width": float(width.mean())})
        rows.append({"model": model, "output": "whole_profile", "interval_kind": "marginal_bands_joint_event",
                     "nominal_coverage": level, "empirical_coverage": float(covered.all(axis=1).mean()),
                     "coverage_error": None, "mean_interval_width": float((upper - lower).mean())})
    return rows
