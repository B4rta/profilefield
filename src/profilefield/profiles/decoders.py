"""Generic and physically constrained profile decoders."""

from __future__ import annotations

from torch import Tensor, nn

from profilefield.profiles.constraints import monotonic_from_logits


class ProfileDecoder(nn.Module):
    def __init__(self, latent_dim: int, profile_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, profile_dim),
        )

    def forward(self, latent: Tensor) -> Tensor:
        return self.network(latent)


class MonotonicRHDecoder(ProfileDecoder):
    """GEDI RH decoder with a hard non-decreasing constraint."""

    def __init__(
        self,
        latent_dim: int,
        profile_dim: int,
        hidden_dim: int,
        normalize_to: float | None = None,
    ) -> None:
        super().__init__(latent_dim, profile_dim, hidden_dim)
        self.normalize_to = normalize_to

    def forward(self, latent: Tensor) -> Tensor:
        return monotonic_from_logits(self.network(latent), normalize_to=self.normalize_to)
