# -*- coding: utf-8 -*-
"""Positive Random Features (PRF) sigmoid approximation based on FAVOR+."""

import math
from typing import Tuple, Union

import torch
import torch.nn as nn


def orthogonal_random_features(d: int, m: int, device: torch.device = None) -> torch.Tensor:
    """
    Generate orthogonal random features matrix via QR decomposition.

    Returns: (m, d)
    """
    num_full_blocks = m // d
    remainder = m % d
    blocks = []

    for _ in range(num_full_blocks):
        random_matrix = torch.randn(d, d, device=device)
        q, _ = torch.linalg.qr(random_matrix)
        blocks.append(q * math.sqrt(d))

    if remainder > 0:
        random_matrix = torch.randn(d, d, device=device)
        q, _ = torch.linalg.qr(random_matrix)
        blocks.append(q[:remainder] * math.sqrt(d))

    return torch.cat(blocks, dim=0)  # (m, d)


def positive_random_features(
    x: torch.Tensor,
    omega: torch.Tensor,
    return_norm: bool = False,
    eps: float = 1e-6,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Compute FAVOR+ positive random features: φ(x) = exp(x @ Ω^T - ||x||²/2 - max) / sqrt(m).

    Args:
        x: (..., d)
        omega: (m, d)
        return_norm: if True, also return ||x||² and max for correction

    Returns:
        features: (..., m)
        If return_norm: (features, norm_sq, max)
    """
    projection = x @ omega.T  # (..., m)
    x_norm_sq = (x**2).sum(dim=-1, keepdim=True)  # (..., 1)

    exponent = projection - x_norm_sq / 2
    exponent_max = exponent.max(dim=-1, keepdim=True)[0]
    exponent_stabilized = exponent - exponent_max

    features = torch.exp(exponent_stabilized) + eps
    features = features / math.sqrt(omega.shape[0])

    if return_norm:
        return features, x_norm_sq, exponent_max
    return features


def prf_sigmoid_batched(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    omega: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Approximate sigmoid(h @ w.T / T) using FAVOR+ PRF.

    For sigmoid(z/T), scale inputs by 1/sqrt(T) before computing features:
        h_scaled @ w_scaled.T = h @ w.T / T

    Args:
        hidden_states: (n, d)
        weights: (k, d)
        omega: (m, d)
        temperature: scaling factor

    Returns: (n, k)
    """
    if temperature != 1.0:
        temp_scale = 1.0 / math.sqrt(temperature)
        hidden_states = hidden_states * temp_scale
        weights = weights * temp_scale

    phi_h, h_norm_sq, h_max = positive_random_features(hidden_states, omega, return_norm=True)
    phi_w, w_norm_sq, w_max = positive_random_features(weights, omega, return_norm=True)

    rf_kernel = phi_h @ phi_w.T  # (n, k)

    # Correction: (||h||² + ||w||²) / 2 + max_h + max_w
    log_correction = (h_norm_sq + w_norm_sq.T) / 2 + h_max + w_max.T
    log_correction = torch.clamp(log_correction, min=-80, max=80)

    log_rf_kernel = torch.log(rf_kernel + 1e-10)
    log_exp_approx = torch.clamp(log_rf_kernel + log_correction, min=-80, max=80)
    exp_approx = torch.exp(log_exp_approx)

    return exp_approx / (1.0 + exp_approx)


def prf_sigmoid_single(
    hidden_state: torch.Tensor,
    weight: torch.Tensor,
    omega: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Single-pair PRF sigmoid for O(log V) inference."""
    h = hidden_state.unsqueeze(0)  # (1, d)
    w = weight.unsqueeze(0)  # (1, d)
    return prf_sigmoid_batched(h, w, omega, temperature).squeeze()


def standard_sigmoid(logit: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Standard sigmoid with temperature scaling."""
    return torch.sigmoid(logit / temperature)


class PRFSigmoid(nn.Module):
    """PRF-based sigmoid approximation module for HSM binary decisions."""

    def __init__(self, hidden_size: int, num_random_features: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_random_features = num_random_features
        self.register_buffer(
            "random_matrix",
            orthogonal_random_features(hidden_size, num_random_features),
        )

    def forward_batched(
        self,
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Batched PRF sigmoid: (n, d) x (k, d) -> (n, k)."""
        return prf_sigmoid_batched(hidden_states, weights, self.random_matrix, temperature)

    def forward_single(
        self,
        hidden_state: torch.Tensor,
        weight: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Single-pair PRF sigmoid for inference."""
        return prf_sigmoid_single(hidden_state, weight, self.random_matrix, temperature)

    def redraw_random_features(self, device: torch.device = None):
        """Redraw random features to reduce approximation bias."""
        device = device or self.random_matrix.device
        new_matrix = orthogonal_random_features(
            self.hidden_size, self.num_random_features, device=device
        )
        self.register_buffer("random_matrix", new_matrix)
