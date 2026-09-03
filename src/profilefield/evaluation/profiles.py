"""Profile fidelity, shape, and physical consistency metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def profile_metric_rows(
    model_name: str,
    truth: FloatArray,
    prediction: FloatArray,
    percentiles: FloatArray | None = None,
    ordered: bool = True,
) -> list[dict[str, Any]]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape:
        raise ValueError("True and predicted profiles must align")
    p = (
        np.asarray(percentiles, dtype=np.float64)
        if percentiles is not None
        else np.linspace(0.0, 100.0, truth.shape[1])
    )
    residual = prediction - truth
    rows: list[dict[str, Any]] = []
    for index, percentile in enumerate(p):
        rows.append(
            {
                "model": model_name,
                "scope": f"RH{percentile:g}",
                "metric": "mae",
                "value": float(np.mean(np.abs(residual[:, index]))),
            }
        )
        rows.append(
            {
                "model": model_name,
                "scope": f"RH{percentile:g}",
                "metric": "rmse",
                "value": float(np.sqrt(np.mean(residual[:, index] ** 2))),
            }
        )
    differences = np.diff(prediction, axis=1)
    violations = int(np.count_nonzero(differences < -1e-7)) if ordered else 0
    wasserstein = np.mean(
        np.trapezoid(np.abs(np.sort(prediction, axis=1) - np.sort(truth, axis=1)), x=p, axis=1)
        / max(float(p[-1] - p[0]), 1.0)
    )
    summary = {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "integrated_profile_rmse": float(
            np.sqrt(np.mean(np.trapezoid(residual**2, x=p, axis=1) / max(p[-1] - p[0], 1.0)))
        ),
        "wasserstein_1d": float(wasserstein),
    }
    if ordered:
        summary["monotonicity_violation_count"] = float(violations)
        summary["monotonicity_violation_percentage"] = float(
            100.0 * violations / max(differences.size, 1)
        )
    rows.extend(
        {"model": model_name, "scope": "all", "metric": metric, "value": value}
        for metric, value in summary.items()
    )
    return rows


def per_block_profile_rows(
    model_name: str,
    truth: FloatArray,
    prediction: FloatArray,
    block_ids: NDArray[np.int64],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block_id in np.unique(block_ids):
        selected = block_ids == block_id
        residual = prediction[selected] - truth[selected]
        rows.extend(
            [
                {
                    "model": model_name,
                    "block_id": int(block_id),
                    "metric": "profile_mae",
                    "value": float(np.mean(np.abs(residual))),
                    "n_samples": int(selected.sum()),
                },
                {
                    "model": model_name,
                    "block_id": int(block_id),
                    "metric": "profile_rmse",
                    "value": float(np.sqrt(np.mean(residual**2))),
                    "n_samples": int(selected.sum()),
                },
            ]
        )
    return rows
