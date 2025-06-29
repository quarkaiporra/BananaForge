"""Gumbel Softmax implementation for differentiable discrete sampling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GumbelSoftmax(nn.Module):
    """Gumbel Softmax layer for differentiable discrete sampling.

    Allows gradient-based optimization over discrete choices by using
    the Gumbel-Softmax trick to create differentiable approximations
    of categorical distributions.
    """

    def __init__(self, temperature: float = 1.0, hard: bool = False):
        """Initialize Gumbel Softmax layer.

        Args:
            temperature: Controls sharpness of softmax. Lower = more discrete.
            hard: If True, returns one-hot vectors with straight-through gradients.
        """
        super().__init__()
        self.temperature = temperature
        self.hard = hard

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply Gumbel Softmax to logits.

        Args:
            logits: Input logits tensor of shape (..., num_classes)

        Returns:
            Differentiable samples from categorical distribution
        """
        return gumbel_softmax(logits, tau=self.temperature, hard=self.hard, dim=-1)

    def set_temperature(self, temperature: float) -> None:
        """Update temperature parameter."""
        self.temperature = temperature


def gumbel_softmax(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
) -> torch.Tensor:
    """Sample from Gumbel-Softmax distribution.

    Args:
        logits: Unnormalized log probabilities
        tau: Temperature parameter
        hard: If True, returns one-hot with straight-through gradients
        eps: Small constant for numerical stability
        dim: Dimension to apply softmax

    Returns:
        Samples from Gumbel-Softmax distribution
    """
    # Sample Gumbel noise
    gumbels = sample_gumbel(logits.shape, eps=eps, device=logits.device)

    # Add noise to logits and apply temperature scaling
    y = (logits + gumbels) / tau

    # Apply softmax
    y_soft = F.softmax(y, dim=dim)

    if hard:
        # Straight-through estimator
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        # Use straight-through gradients
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


def sample_gumbel(
    shape: Tuple[int, ...], eps: float = 1e-10, device: torch.device = None
) -> torch.Tensor:
    """Sample from Gumbel(0, 1) distribution.

    Args:
        shape: Shape of samples to generate
        eps: Small constant for numerical stability
        device: Device to generate samples on

    Returns:
        Gumbel noise samples
    """
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


class TemperatureScheduler:
    """Scheduler for Gumbel Softmax temperature annealing."""

    def __init__(
        self,
        initial_temp: float = 1.0,
        final_temp: float = 0.1,
        decay_type: str = "linear",
    ):
        """Initialize temperature scheduler.

        Args:
            initial_temp: Starting temperature
            final_temp: Final temperature
            decay_type: Type of decay ("linear", "exponential", "cosine")
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.decay_type = decay_type

    def get_temperature(self, step: int, total_steps: int) -> float:
        """Get temperature for current step.

        Args:
            step: Current optimization step
            total_steps: Total number of steps

        Returns:
            Temperature value for current step
        """
        progress = min(step / total_steps, 1.0)

        if self.decay_type == "linear":
            return self.initial_temp + progress * (self.final_temp - self.initial_temp)
        elif self.decay_type == "exponential":
            return self.initial_temp * (self.final_temp / self.initial_temp) ** progress
        elif self.decay_type == "cosine":
            return self.final_temp + 0.5 * (self.initial_temp - self.final_temp) * (
                1 + torch.cos(torch.tensor(progress * torch.pi))
            )
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")
