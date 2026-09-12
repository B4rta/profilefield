"""Training-only, block-held-out context predictions for residual learning."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.model_selection import GroupKFold


def spatial_crossfit_predictions(
    estimator: Any,
    features: NDArray[np.float64],
    targets: NDArray[np.float64],
    block_ids: NDArray[np.int64],
    sample_ids: list[str],
    folds: int = 5,
) -> tuple[NDArray[np.float64], list[dict[str, Any]]]:
    """Predict each training case with a clone fitted without its entire block.

    Callers must supply the outer training partition only. The fitted production
    context estimator is untouched; these predictions estimate residual targets,
    not a new mean model for validation, calibration, or testing.
    """
    x, y, blocks = np.asarray(features), np.asarray(targets), np.asarray(block_ids)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or blocks.shape != (len(y),):
        raise ValueError("Expected aligned 2D features/targets and one block ID per case")
    if len(sample_ids) != len(y) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Sample IDs must be unique and aligned with the training cases")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Cross-fitting inputs must be finite")
    if folds < 2 or len(np.unique(blocks)) < folds:
        raise ValueError("Cross-fitting requires at least folds distinct training blocks")
    predictions = np.empty_like(y, dtype=np.float64)
    assigned = np.zeros(len(y), dtype=bool)
    manifest: list[dict[str, Any]] = []
    for fold, (fit, held_out) in enumerate(GroupKFold(folds).split(x, y, blocks)):
        model = clone(estimator).fit(x[fit], y[fit])
        predictions[held_out] = np.asarray(model.predict(x[held_out])).reshape(y[held_out].shape)
        assigned[held_out] = True
        manifest.append({
            "fold": fold,
            "fit_sample_ids": [sample_ids[i] for i in fit],
            "held_out_sample_ids": [sample_ids[i] for i in held_out],
            "fit_blocks": np.unique(blocks[fit]).tolist(),
            "held_out_blocks": np.unique(blocks[held_out]).tolist(),
        })
    if not assigned.all() or not np.isfinite(predictions).all():
        raise RuntimeError("Cross-fitting failed to generate all held-out predictions")
    return predictions, manifest
