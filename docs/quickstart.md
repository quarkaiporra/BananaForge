# Quick Start Guide

Get up and running with BananaForge in 5 minutes! This guide will walk you through your first image-to-3D conversion.

## Prerequisites

- BananaForge installed ([Installation Guide](installation.md))
- An image file (JPG, PNG, etc.)
- Basic familiarity with command line

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

This creates an `output/` folder with:
- `bananaforge_model.stl` - 3D model file
- `bananaforge_model_instructions.txt` - Printing instructions
- `bananaforge_model_cost_report.txt` - Cost analysis

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

## Customizing Your Conversion

### Using Specific Materials

```bash
# Export default material database first
bananaforge export-materials --output bambu_materials.csv

# Convert with specific materials
bananaforge convert my_image.jpg --materials bambu_materials.csv
```

### Adjusting Quality vs Speed

```bash
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
Material usage and cost breakdown:

```
MATERIAL USAGE:
Material: Basic PLA Black
  Weight: 12.50g
  Cost: $0.37
  Layers: 25

TOTAL MATERIAL COST: $1.25
TOTAL WEIGHT: 45.20g
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

### Workflow 1: Quick Test Print

```bash
# Fast, small test print
bananaforge convert image.jpg \
  --iterations 200 \
  --resolution 64 \
  --max-materials 4 \
  --physical-size 50 \
  --project-name "test_print"
```

### Workflow 2: Production Quality

```bash
# High quality for final prints
bananaforge convert image.jpg \
  --iterations 2000 \
  --resolution 512 \
  --max-materials 8 \
  --physical-size 200 \
  --layer-height 0.15 \
  --export-format stl instructions hueforge cost_report
```

### Workflow 3: Specific Printer Setup

```bash
# For Bambu Lab printer
bananaforge convert image.jpg \
  --materials bambu_materials.csv \
  --export-format stl instructions bambu cost_report

# For Prusa printer  
bananaforge convert image.jpg \
  --export-format stl instructions prusa cost_report
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