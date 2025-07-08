# Transparency-Based Color Mixing Example

This example demonstrates BananaForge's advanced transparency features that create more colors with fewer materials through strategic layer transparency.

## Overview

Transparency mixing allows you to:
- **Create 30+ colors from 6 base materials**
- **Reduce material swaps by 30-50%**
- **Achieve smooth gradients and color transitions**
- **Optimize base layer selection for maximum contrast**

## Three-Layer Opacity Model

BananaForge uses a physics-based transparency model:

- **33% opacity**: Light transparency for subtle color mixing
- **67% opacity**: Medium transparency for gradient effects  
- **100% opacity**: Full color for vibrant base layers

## Basic Transparency Usage

### Command Line

```bash
# Enable transparency with default settings
bananaforge convert photo.jpg --enable-transparency

# Full transparency optimization
bananaforge convert photo.jpg \
  --enable-transparency \
  --opacity-levels "0.33,0.67,1.0" \
  --optimize-base-layers \
  --enable-gradients \
  --max-materials 6 \
  --output ./transparent_model/
```

### Python API

```python
from bananaforge.materials.transparency_mixer import TransparencyColorMixer
from bananaforge.materials.transparency_optimizer import TransparencyOptimizer

# Initialize transparency mixer
mixer = TransparencyColorMixer()

# Configure opacity levels
opacity_levels = [0.33, 0.67, 1.0]
mixer.set_opacity_levels(opacity_levels)

# Calculate achievable color palette
base_materials = ["#000000", "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"]
achievable_colors = mixer.calculate_color_palette(base_materials)

print(f"Generated {len(achievable_colors)} colors from {len(base_materials)} materials")
```

## Advanced Transparency Features

### Base Layer Optimization

```python
from bananaforge.materials.base_layer_optimizer import BaseLayerOptimizer

# Initialize base layer optimizer
base_optimizer = BaseLayerOptimizer()

# Analyze image for optimal base colors
base_analysis = base_optimizer.analyze_image("portrait.jpg")
print(f"Recommended base colors: {base_analysis.recommended_colors}")
print(f"Contrast potential: {base_analysis.contrast_score:.2f}")

# Optimize base layer selection
optimized_base = base_optimizer.optimize_base_selection(
    image="portrait.jpg",
    available_materials=materials,
    max_base_colors=2,
    prioritize_dark_base=True  # Dark base for maximum contrast
)
```

### Gradient Processing

```python
from bananaforge.materials.gradient_processor import GradientProcessor

# Initialize gradient processor
gradient_processor = GradientProcessor()

# Detect gradient regions in image
gradient_regions = gradient_processor.detect_gradients(
    image="sunset.jpg",
    min_gradient_length=50,  # Minimum pixels for gradient
    smoothness_threshold=0.8  # Gradient smoothness requirement
)

print(f"Found {len(gradient_regions)} gradient regions")

# Optimize gradients with transparency
for region in gradient_regions:
    optimized_gradient = gradient_processor.optimize_gradient(
        region=region,
        opacity_levels=[0.33, 0.67, 1.0],
        materials=materials
    )
    print(f"Gradient optimized: {optimized_gradient.material_sequence}")
```

### Transparency Integration

```python
from bananaforge.materials.transparency_integration import TransparencyIntegration

# Initialize transparency integration
integration = TransparencyIntegration()

# Full transparency workflow
result = integration.optimize_with_transparency(
    image="complex_image.jpg",
    materials=materials,
    transparency_config={
        'opacity_levels': [0.33, 0.67, 1.0],
        'optimize_base_layers': True,
        'enable_gradients': True,
        'transparency_threshold': 0.3,  # Minimum transparency savings
        'max_transparency_layers': 15   # Limit transparent layers
    }
)

# Analyze transparency results
print(f"Original materials needed: {result.original_material_count}")
print(f"Transparency materials used: {result.transparency_material_count}")
print(f"Material swap reduction: {result.swap_reduction_percentage:.1f}%")
print(f"Cost savings: ${result.cost_savings:.2f}")
```

## Practical Examples

### Portrait with Skin Tones

```python
# Portrait optimization focuses on smooth skin tone transitions
result = transparency_optimizer.optimize(
    image="portrait.jpg",
    materials=materials,
    enable_transparency=True,
    transparency_config={
        'opacity_levels': [0.25, 0.5, 0.75, 1.0],  # More opacity levels for skin
        'optimize_base_layers': True,
        'base_layer_preference': 'dark',  # Dark base for contrast
        'enable_gradients': True,
        'gradient_smoothness': 0.9,  # High smoothness for skin
        'max_materials': 8
    }
)
```

### Landscape with Sky Gradients

```python
# Landscape optimization emphasizes sky gradients
result = transparency_optimizer.optimize(
    image="landscape.jpg",
    materials=materials,
    enable_transparency=True,
    transparency_config={
        'opacity_levels': [0.2, 0.4, 0.6, 0.8, 1.0],  # Fine gradient control
        'optimize_base_layers': True,
        'enable_gradients': True,
        'gradient_regions': ['sky', 'water'],  # Focus on these regions
        'max_materials': 6
    }
)
```

### Logo with Sharp Edges

```python
# Logo optimization preserves sharp edges while using transparency for backgrounds
result = transparency_optimizer.optimize(
    image="logo.png",
    materials=materials,
    enable_transparency=True,
    transparency_config={
        'opacity_levels': [0.0, 0.5, 1.0],  # Include full transparency
        'preserve_edges': True,
        'edge_threshold': 0.8,  # Sharp edge preservation
        'optimize_base_layers': False,  # Preserve original colors
        'max_materials': 4
    }
)
```

## Understanding Transparency Results

### Transparency Analysis Report

```python
# Generate detailed transparency analysis
analysis = result.transparency_analysis

print("=== Transparency Analysis ===")
print(f"Base materials: {analysis.base_material_count}")
print(f"Transparency layers: {analysis.transparency_layer_count}")
print(f"Total achievable colors: {analysis.total_color_count}")
print(f"Original vs optimized materials: {analysis.original_materials} → {analysis.optimized_materials}")
print(f"Swap reduction: {analysis.swap_reduction_percentage:.1f}%")
print(f"Cost savings: ${analysis.cost_savings:.2f}")

# Color mixing breakdown
for combo in analysis.color_combinations:
    print(f"  {combo.base_color} + {combo.transparency_level:.0%} → {combo.result_color}")
```

### Layer-by-Layer Transparency

```python
# Analyze transparency usage by layer
for layer_idx, layer_info in enumerate(result.layer_transparency):
    print(f"Layer {layer_idx}:")
    print(f"  Material: {layer_info.material_name}")
    print(f"  Opacity: {layer_info.opacity:.0%}")
    print(f"  Base color: {layer_info.base_color}")
    print(f"  Effective color: {layer_info.effective_color}")
```

## Export Options for Transparency

### STL with Alpha Channel

```python
# Export STL with alpha channel information
exporter.export_stl(
    result, 
    "transparent_model.stl",
    include_alpha=True,
    alpha_threshold=0.1  # Minimum alpha to include
)
```

### 3MF with Transparency

```python
# Export 3MF with full transparency support
exporter.export_3mf(
    result,
    "transparent_model.3mf",
    include_transparency=True,
    transparency_precision=0.01  # Transparency precision
)
```

### Instructions with Transparency Notes

```python
# Export enhanced instructions with transparency information
exporter.export_instructions(
    result,
    "transparency_instructions.txt",
    include_transparency_notes=True,
    format="detailed"
)
```

## Performance Optimization

### Memory-Efficient Transparency

```python
# For large images, use memory-efficient transparency
config = TransparencyConfig(
    opacity_levels=[0.33, 0.67, 1.0],
    batch_size=1,  # Process one layer at a time
    use_mixed_precision=True,  # Requires CUDA
    cache_transparency_calculations=False  # Save memory
)

result = transparency_optimizer.optimize(
    image="large_image.jpg",
    materials=materials,
    config=config
)
```

### GPU Acceleration

```python
# Use GPU for faster transparency calculations
import torch

if torch.cuda.is_available():
    device = "cuda"
    mixed_precision = True
else:
    device = "cpu"
    mixed_precision = False

transparency_optimizer = TransparencyOptimizer(
    device=device,
    mixed_precision=mixed_precision
)
```

## Tips for Best Results

### Material Selection
1. **Start with dark base colors** (black, dark brown) for maximum contrast
2. **Choose primary colors** that mix well (RGB, CMY color wheels)
3. **Include white or light colors** for highlights and opacity effects
4. **Consider material compatibility** - some materials don't layer well

### Image Preparation
1. **High contrast images** work best with transparency
2. **Smooth gradients** benefit most from transparency optimization
3. **Consider the intended viewing angle** - transparency effects vary by perspective
4. **Test with smaller images first** to validate settings

### Optimization Settings
1. **More opacity levels** = smoother transitions but more complexity
2. **Enable gradient processing** for images with sky, water, or smooth transitions
3. **Optimize base layers** for maximum material efficiency
4. **Balance quality vs. print complexity** based on your printer capabilities

## Troubleshooting

### Transparency Not Reducing Materials
```bash
# Lower transparency threshold
bananaforge convert photo.jpg --enable-transparency --transparency-threshold 0.1

# Increase material count to allow more base colors
bananaforge convert photo.jpg --enable-transparency --max-materials 8
```

### Poor Color Mixing Results
```bash
# Optimize base layers
bananaforge convert photo.jpg --enable-transparency --optimize-base-layers

# Use more opacity levels
bananaforge convert photo.jpg --enable-transparency --opacity-levels "0.2,0.4,0.6,0.8,1.0"
```

### Long Processing Times
```bash
# Use GPU acceleration
bananaforge convert photo.jpg --enable-transparency --device cuda

# Reduce resolution
bananaforge convert photo.jpg --enable-transparency --resolution 200
```

## Next Steps

- Learn about [3MF Export](3mf-export.md) for professional transparency support
- Explore [Advanced Optimization](advanced-optimization.md) techniques
- Check out [Material Management](material-management.md) for custom transparency materials
- Try [Batch Processing](batch-processing.md) with transparency settings