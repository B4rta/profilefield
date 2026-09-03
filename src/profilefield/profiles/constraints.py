"""Architectural physical constraints for ordered profiles."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def monotonic_from_logits(
    logits: Tensor,
    base: Tensor | float = 0.0,
    normalize_to: Tensor | float | None = None,
) -> Tensor:
    """Convert logits to a non-decreasing profile using positive increments."""
    increments = F.softplus(logits)
    profile = torch.cumsum(increments, dim=-1) + torch.as_tensor(
        base, dtype=logits.dtype, device=logits.device
    )
    if normalize_to is not None:
        target = torch.as_tensor(normalize_to, dtype=logits.dtype, device=logits.device)
        profile = profile / profile[..., -1:].clamp_min(torch.finfo(logits.dtype).eps)
        profile = profile * target
    return profile


def monotonic_violation_count(profiles: Tensor, tolerance: float = 1e-7) -> int:
    differences = torch.diff(profiles, dim=-1)
    return int(torch.count_nonzero(differences < -tolerance).item())
