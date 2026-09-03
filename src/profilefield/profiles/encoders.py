"""Shared profile encoders."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ProfileEncoder(nn.Module):
    """Capacity-controlled MLP encoder for ordered or generic profiles."""

    def __init__(self, profile_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(profile_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_variance = nn.Linear(hidden_dim, latent_dim)

    def forward(self, profiles: Tensor) -> tuple[Tensor, Tensor]:
        features = self.backbone(profiles)
        return self.mean(features), self.log_variance(features).clamp(-12.0, 8.0)

    @staticmethod
    def sample(mean: Tensor, log_variance: Tensor) -> Tensor:
        return mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
