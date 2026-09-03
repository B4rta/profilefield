"""Shared-capacity deterministic and conditional variational profile models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from profilefield.context.encoder import ContextEncoder
from profilefield.profiles.decoders import MonotonicRHDecoder
from profilefield.profiles.encoders import ProfileEncoder


class DeterministicContextProfile(nn.Module):
    def __init__(
        self, context_dim: int, latent_dim: int, profile_dim: int, hidden_dim: int
    ) -> None:
        super().__init__()
        self.context_encoder = ContextEncoder(context_dim, latent_dim, hidden_dim)
        self.decoder = MonotonicRHDecoder(latent_dim, profile_dim, hidden_dim)

    def forward(self, context: Tensor) -> Tensor:
        return self.decoder(self.context_encoder(context))


class ContextualProfileVAE(nn.Module):
    """q(z|profile), context-conditioned prior mean, and shared RH decoder."""

    def __init__(
        self, context_dim: int, latent_dim: int, profile_dim: int, hidden_dim: int
    ) -> None:
        super().__init__()
        self.profile_encoder = ProfileEncoder(profile_dim, latent_dim, hidden_dim)
        self.context_encoder = ContextEncoder(context_dim, latent_dim, hidden_dim)
        self.decoder = MonotonicRHDecoder(latent_dim, profile_dim, hidden_dim)

    def forward(self, context: Tensor, profiles: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mean, log_variance = self.profile_encoder(profiles)
        context_mean = self.context_encoder(context)
        latent = self.profile_encoder.sample(mean, log_variance)
        return self.decoder(latent), mean, log_variance, context_mean

    def loss(self, context: Tensor, profiles: Tensor, kl_weight: float) -> Tensor:
        reconstruction, mean, log_variance, context_mean = self(context, profiles)
        reconstruction_loss = F.mse_loss(reconstruction, profiles)
        kl = 0.5 * torch.mean(
            (mean - context_mean) ** 2 + torch.exp(log_variance) - log_variance - 1.0
        )
        return reconstruction_loss + kl_weight * kl

    def encode_residual(self, context: Tensor, profiles: Tensor) -> Tensor:
        mean, _ = self.profile_encoder(profiles)
        return mean - self.context_encoder(context)

    def predict_context_only(self, context: Tensor) -> Tensor:
        return self.decoder(self.context_encoder(context))


@dataclass(frozen=True)
class NeuralTrainingResult:
    history: list[float]
    residual_variance: Tensor


def train_deterministic_profile(
    model: DeterministicContextProfile,
    context: Tensor,
    profiles: Tensor,
    steps: int,
    learning_rate: float,
) -> NeuralTrainingResult:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(context)
        loss = F.mse_loss(prediction, profiles)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        residual_variance = torch.var(profiles - model(context), dim=0, unbiased=True).clamp_min(
            1e-6
        )
    return NeuralTrainingResult(history, residual_variance)


def train_contextual_vae(
    model: ContextualProfileVAE,
    context: Tensor,
    profiles: Tensor,
    steps: int,
    learning_rate: float,
    kl_weight: float,
) -> NeuralTrainingResult:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(context, profiles, kl_weight)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        prediction = model.predict_context_only(context)
        residual_variance = torch.var(profiles - prediction, dim=0, unbiased=True).clamp_min(1e-6)
    return NeuralTrainingResult(history, residual_variance)
