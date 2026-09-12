"""Geographic block bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def block_bootstrap_interval(
    values: FloatArray,
    block_ids: NDArray[np.int64],
    statistic: Callable[[FloatArray], float] = lambda x: float(np.mean(x)),
    replicates: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Percentile interval resampling whole geographic blocks.

    The statistic is evaluated on pooled observations, preserving the declared
    observation-weighted estimand when block sizes differ. Coverage requires
    exchangeable, approximately independent blocks; this function cannot infer
    that assumption from numeric block identifiers. For equal-block effects,
    first reduce the paired observations to one effect per block.
    """
    values = np.asarray(values, dtype=np.float64)
    identifiers = np.asarray(block_ids)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("values must be a nonempty, finite one-dimensional array")
    if identifiers.ndim != 1 or len(identifiers) != len(values):
        raise ValueError("block_ids must be one-dimensional and align with values")
    if identifiers.dtype.kind not in "iu":
        raise ValueError("block_ids must contain integer geographic identifiers")
    if (
        isinstance(replicates, (bool, np.bool_))
        or not isinstance(replicates, (int, np.integer))
        or replicates < 1
    ):
        raise ValueError("replicates must be a positive integer")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    estimate = float(statistic(values))
    if not np.isfinite(estimate):
        raise ValueError("statistic must return a finite scalar")
    blocks = np.unique(identifiers)
    if len(blocks) < 2:
        return estimate, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        chunks = [values[identifiers == block] for block in selected_blocks]
        bootstrap[replicate] = statistic(np.concatenate(chunks))
    if not np.all(np.isfinite(bootstrap)):
        raise ValueError("statistic returned nonfinite bootstrap replicates")
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap, [alpha, 1.0 - alpha])
    return estimate, float(lower), float(upper)
