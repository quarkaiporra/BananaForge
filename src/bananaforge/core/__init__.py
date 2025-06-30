"""Core optimization engine for BananaForge."""

from .gumbel import GumbelSoftmax
from .loss import ColorLoss, PerceptualLoss, SmoothnessLoss
from .optimizer import LayerOptimizer
from .enhanced_optimizer import (
    EnhancedLayerOptimizer,
    EnhancedOptimizationConfig,
    DiscreteValidator,
    LearningRateScheduler,
    EnhancedEarlyStopping,
    MixedPrecisionManager,
)

__all__ = [
    "LayerOptimizer",
    "EnhancedLayerOptimizer",
    "EnhancedOptimizationConfig",
    "DiscreteValidator",
    "LearningRateScheduler",
    "EnhancedEarlyStopping",
    "MixedPrecisionManager",
    "PerceptualLoss",
    "ColorLoss",
    "SmoothnessLoss",
    "GumbelSoftmax",
]
