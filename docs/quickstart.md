# Quick Start Guide

Get up and running with BananaForge in 5 minutes! This guide covers basic conversion and the new transparency mixing features.

## Prerequisites

- BananaForge installed ([Installation Guide](installation.md))
- An image file (JPG, PNG, etc.)
- Basic familiarity with command line
- Optional: CUDA-capable GPU for faster processing

## Your First Conversion

### Step 1: Prepare Your Image

Choose an image with good contrast and clear colors. For best results:

- **Resolution**: 256x256 to 1024x1024 pixels
- **Format**: JPG or PNG
- **Content**: Clear subjects, good color separation
- **Avoid**: Very dark/light images, low contrast

### Step 2: Basic Conversion

```bash
# Simple conversion with default settings
bananaforge convert my_image.jpg
```

**New in v1.0**: This now uses enhanced optimization with:
- LAB color space for better color matching
- Two-stage K-means clustering for height maps
- Discrete validation tracking
- Learning rate scheduling

This creates an `output/` folder with:
- `bananaforge_model.stl` - 3D model file (now with alpha channel support)
- `bananaforge_model_instructions.txt` - Printing instructions
- `bananaforge_model_cost_report.txt` - Cost analysis
- `bananaforge_model_transparency_analysis.txt` - Transparency features report

### Step 3: View Your Results

Check the output folder:

```bash
ls output/
# Shows: 
# bananaforge_model.stl
# bananaforge_model_instructions.txt  
# bananaforge_model_cost_report.txt
# bananaforge_model_summary.txt
```

Load the STL file in your favorite 3D slicer to preview the model!

## Advanced Features

### Transparency Mixing (New!)

Reduce material swaps by 30% using transparency-based color mixing:

```bash
# Enable transparency features
bananaforge convert my_image.jpg --enable-transparency

# With custom opacity levels
bananaforge convert my_image.jpg \
  --enable-transparency \
  --opacity-levels 0.25,0.5,0.75,1.0

# Optimize base layers for maximum contrast
bananaforge convert my_image.jpg \
  --enable-transparency \
  --optimize-base-layers
```

### Using Specific Materials

```bash
# Export default material database first
bananaforge export-materials --output materials.csv

# Convert with specific materials
bananaforge convert my_image.jpg --materials materials.csv

# With transparency mixing
bananaforge convert my_image.jpg \
  --materials materials.csv \
  --enable-transparency
```

### GPU Acceleration and Performance

```bash
# Use CUDA for faster processing
bananaforge convert my_image.jpg --device cuda

# Apple Silicon MPS support
bananaforge convert my_image.jpg --device mps

# Mixed precision for memory efficiency
bananaforge convert my_image.jpg --device cuda --mixed-precision

# Quality vs Speed options
# Fast conversion (lower quality)
bananaforge convert my_image.jpg --iterations 500 --resolution 128

# High quality (slower)  
bananaforge convert my_image.jpg --iterations 2000 --resolution 512

# Balanced (default)
bananaforge convert my_image.jpg --iterations 1000 --resolution 256
```

### Controlling Materials and Size

```bash
bananaforge convert my_image.jpg \
  --max-materials 6 \
  --physical-size 150 \
  --layer-height 0.15 \
  --project-name "my_awesome_model"
```

## Understanding the Output

### STL File
Your 3D model ready for slicing. Import into:
- PrusaSlicer
- Bambu Studio  
- Cura
- Any other slicer

### Swap Instructions
Tells you when to change materials during printing:

```
MATERIAL SWAP INSTRUCTIONS
==========================

SWAP #1
Layer: 15
Height: 3.00mm
Action: Change from Basic PLA Black to Basic PLA Red
Estimated time: 3m 45s
```

### Cost Report
Material usage and cost breakdown with transparency savings:

```
MATERIAL USAGE:
Material: Basic PLA Black (Base Layer)
  Weight: 8.20g
  Cost: $0.25
  Layers: 15
  Transparency: 100% (Opaque)

Material: Basic PLA Red (Overlay)
  Weight: 4.30g  
  Cost: $0.13
  Layers: 10
  Transparency: 67% (Partial)

TRANSPARENCY SAVINGS:
  Material Swaps Reduced: 35% (12 → 8 swaps)
  Material Savings: $0.87 (was $2.25)
  Time Savings: 8 minutes

TOTAL MATERIAL COST: $0.38
TOTAL WEIGHT: 12.50g
```

## Analyzing Before Converting

Preview color matching before full conversion:

```bash
# Analyze what materials would be selected
bananaforge analyze-colors my_image.jpg --max-materials 8

# Output shows:
# Suggested materials (8):
#   1. Basic PLA White - #FFFFFF (RGB: 1.00, 1.00, 1.00)
#   2. Basic PLA Black - #000000 (RGB: 0.00, 0.00, 0.00)
#   ...
```

## Common Workflows

### Workflow 1: Quick Test Print with Transparency

```bash
# Fast, small test print with transparency features
bananaforge convert image.jpg \
  --iterations 200 \
  --resolution 64 \
  --max-materials 4 \
  --physical-size 50 \
  --enable-transparency \
  --project-name "test_print"
```

### Workflow 2: Production Quality with Full Transparency

```bash
# High quality for final prints with all transparency features
bananaforge convert image.jpg \
  --iterations 2000 \
  --resolution 512 \
  --max-materials 8 \
  --physical-size 200 \
  --layer-height 0.15 \
  --enable-transparency \
  --optimize-base-layers \
  --enable-gradients \
  --device cuda \
  --mixed-precision \
  --export-format stl instructions hueforge cost_report transparency_analysis
```

### Workflow 3: Transparency Optimization Focus

```bash
# Maximize material savings with transparency
bananaforge convert image.jpg \
  --materials materials.csv \
  --enable-transparency \
  --transparency-threshold 0.35 \
  --optimize-base-layers \
  --enable-gradients \
  --export-format stl instructions transparency_analysis cost_report

# For specific printer with transparency
bananaforge convert image.jpg \
  --materials bambu_materials.csv \
  --enable-transparency \
  --export-format stl instructions hueforge transparency_analysis
```

## Configuration Profiles

Use predefined profiles for common scenarios:

```bash
# Create config file
bananaforge init-config --output my_config.json

# Edit my_config.json to set your preferences, then:
bananaforge convert image.jpg --config my_config.json
```

Example profiles in config:
- **prototype**: Fast, low-res for testing
- **balanced**: Good quality/speed balance (default)
- **quality**: High-res, slow, best results

## GPU Acceleration

Speed up processing with GPU:

```bash
# Use CUDA (NVIDIA)
bananaforge convert image.jpg --device cuda

# Use MPS (Apple Silicon)
bananaforge convert image.jpg --device mps

# Use CPU (fallback)
bananaforge convert image.jpg --device cpu
```

## Performance Tips

### For Fast Iteration
```bash
# Low resolution for quick testing
bananaforge convert image.jpg --resolution 128

# Use CPU for small images
bananaforge convert image.jpg --device cpu
```

### For Best Quality
```bash
# High resolution for final prints
bananaforge convert image.jpg --iterations 2000

# Use GPU acceleration
bananaforge convert image.jpg --resolution 512
```

### For Large Images
```bash
# Process high-res images efficiently
bananaforge convert image.jpg --iterations 2000

# Use GPU with high resolution
bananaforge convert image.jpg --resolution 512
```

## Next Steps

1. 🎨 **[Materials Guide](materials.md)** - Learn about material management
2. ⚙️ **[Configuration](configuration.md)** - Customize your workflow  
3. 📖 **[CLI Reference](cli-reference.md)** - Complete command reference
4. 🐍 **[Python API](api-reference.md)** - Use BananaForge in your code

## Tips for Success

### Image Selection
- **High contrast** images work best
- **Simple compositions** are easier to print
- **Good lighting** improves color accuracy
- **Avoid gradients** that require many materials

### Print Settings
- Start with **fewer materials** (4-6) for easier printing
- Use **thicker layers** (0.2-0.3mm) for faster prints
- Test with **smaller sizes** first
- Consider **support material** for overhangs

### Material Management
- **Organize materials** by color families
- **Test material combinations** before big prints
- **Keep spare filament** for longer prints
- **Document successful** material combinations

Happy printing! 🎉