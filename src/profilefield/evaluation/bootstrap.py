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
    values = np.asarray(values, dtype=np.float64)
    blocks = np.unique(block_ids)
    if len(blocks) < 2:
        estimate = statistic(values)
        return estimate, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        chunks = [values[block_ids == block] for block in selected_blocks]
        bootstrap[replicate] = statistic(np.concatenate(chunks))
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap, [alpha, 1.0 - alpha])
    return statistic(values), float(lower), float(upper)
