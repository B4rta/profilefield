"""Shared spatial kernels and inducing-point initialization."""

from __future__ import annotations

import numpy as np
import torch
from gpytorch.kernels import MaternKernel, ScaleKernel
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def matern_kernel(
    input_dim: int,
    nu: float,
    batch_shape: torch.Size | None = None,
) -> ScaleKernel:
    if nu not in (1.5, 2.5):
        raise ValueError("ProfileField supports Matérn-3/2 and Matérn-5/2")
    resolved_batch_shape = torch.Size() if batch_shape is None else batch_shape
    base = MaternKernel(nu=nu, ard_num_dims=input_dim, batch_shape=resolved_batch_shape)
    return ScaleKernel(base, batch_shape=resolved_batch_shape)


def select_inducing_points(coordinates: FloatArray, count: int) -> FloatArray:
    """Deterministic farthest-point selection shared by both GP ablations."""
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Coordinates must be a matrix")
    count = min(max(1, int(count)), len(values))
    centroid = values.mean(axis=0)
    first = int(np.argmax(np.sum((values - centroid) ** 2, axis=1)))
    selected = [first]
    minimum_distance = np.sum((values - values[first]) ** 2, axis=1)
    for _ in range(1, count):
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        candidate_distance = np.sum((values - values[next_index]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, candidate_distance)
    return values[np.asarray(selected, dtype=np.int64)]
