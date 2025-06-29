"""BananaForge: AI-powered multi-layer 3D printing optimization."""

__version__ = "0.1.0"
__author__ = "BananaForge Team"
__email__ = "info@bananaforge.com"

from .core.optimizer import LayerOptimizer
from .image.processor import ImageProcessor
from .materials.manager import MaterialManager
from .output.exporter import ModelExporter

__all__ = [
    "LayerOptimizer",
    "ImageProcessor",
    "MaterialManager",
    "ModelExporter",
]
