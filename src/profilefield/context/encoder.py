"""Deterministic context mean f_context(c(s))."""

from __future__ import annotations

from torch import Tensor, nn


class ContextEncoder(nn.Module):
    def __init__(self, context_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, context: Tensor) -> Tensor:
        return self.network(context)
