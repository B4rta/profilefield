from __future__ import annotations

import torch

from profilefield.profiles.constraints import monotonic_from_logits, monotonic_violation_count
from profilefield.profiles.decoders import MonotonicRHDecoder


def test_monotonic_construction_has_zero_violations() -> None:
    generator = torch.Generator().manual_seed(23)
    logits = torch.randn(128, 101, generator=generator) * 20.0
    profiles = monotonic_from_logits(logits)
    assert monotonic_violation_count(profiles) == 0
    assert torch.all(torch.diff(profiles, dim=-1) >= 0)


def test_monotonic_decoder_dimensions_and_gradients() -> None:
    decoder = MonotonicRHDecoder(4, 12, 16)
    latent = torch.randn(7, 4, requires_grad=True)
    profile = decoder(latent)
    assert profile.shape == (7, 12)
    assert monotonic_violation_count(profile) == 0
    profile.sum().backward()
    assert latent.grad is not None
