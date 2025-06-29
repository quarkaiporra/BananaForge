# CLI Reference

Complete reference for BananaForge command-line interface.

## Global Options

These options work with all commands:

```bash
bananaforge [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

### Global Options
- `--verbose, -v` - Enable verbose logging
- `--quiet, -q` - Suppress non-error output  
- `--config PATH` - Path to configuration file
- `--help` - Show help message

## Commands Overview

| Command | Description |
|---------|-------------|
| [`convert`](#convert) | Convert image to 3D model |
| [`analyze-colors`](#analyze-colors) | Analyze image colors and suggest materials |
| [`export-materials`](#export-materials) | Export material database to file |
| [`validate-stl`](#validate-stl) | Validate STL file for 3D printing |
| [`init-config`](#init-config) | Create configuration file |
| [`version`](#version) | Show version information |

---

## convert

Convert an image to a multi-layer 3D model.

```bash
bananaforge convert INPUT_IMAGE [OPTIONS]
```

### Arguments
- `INPUT_IMAGE` - Path to input image file (JPG, PNG, etc.)

### Core Options

#### Output Control
- `--output, -o PATH` - Output directory (default: `./output`)
- `--project-name TEXT` - Name for generated files (default: `bananaforge_model`)

#### Material Settings
- `--materials PATH` - Material database file (CSV or JSON)
- `--max-materials INT` - Maximum number of materials (default: 8)

#### Model Parameters
- `--max-layers INT` - Maximum number of layers (default: 50)
- `--layer-height FLOAT` - Layer height in mm (default: 0.2)
- `--physical-size FLOAT` - Size of longest dimension in mm (default: 100.0)
- `--resolution INT` - Processing resolution in pixels (default: 256)

#### Optimization Settings
- `--iterations INT` - Number of optimization iterations (default: 1000)
- `--learning-rate FLOAT` - Learning rate for optimization (default: 0.01)
- `--device [cpu|cuda|mps]` - Device for computation (default: cpu)

#### Export Options
- `--export-format [stl|instructions|hueforge|prusa|bambu|cost_report]` - Export formats (can specify multiple)
- `--preview` - Generate preview visualization

### Examples

#### Basic Conversion
```bash
bananaforge convert photo.jpg
```

#### High Quality Conversion
```bash
bananaforge convert photo.jpg \
  --iterations 2000 \
  --resolution 512 \
  --max-materials 8 \
  --physical-size 150
```

#### Specific Materials and Export
```bash
bananaforge convert photo.jpg \
  --materials bambu_pla.csv \
  --export-format stl instructions bambu \
  --project-name "my_lithophane"
```

#### GPU Accelerated
```bash
bananaforge convert photo.jpg \
  --device cuda \
  --iterations 1500 \
  --resolution 512
```

---

## analyze-colors

Analyze image colors and suggest materials without full conversion.

```bash
bananaforge analyze-colors INPUT_IMAGE [OPTIONS]
```

### Arguments
- `INPUT_IMAGE` - Path to input image file

### Options
- `--materials PATH` - Material database file
- `--max-materials INT` - Maximum materials to suggest (default: 8)
- `--method [perceptual|euclidean|lab]` - Color matching method (default: perceptual)
- `--output, -o PATH` - Save analysis results to JSON file

### Examples

#### Basic Color Analysis
```bash
bananaforge analyze-colors photo.jpg --max-materials 6
```

#### Advanced Analysis with Custom Materials
```bash
bananaforge analyze-colors photo.jpg \
  --materials my_filaments.csv \
  --method lab \
  --output color_analysis.json
```

### Output
```
Color analysis for photo.jpg
Method: perceptual
Suggested materials (6):
  1. Basic PLA White - #FFFFFF (RGB: 1.00, 1.00, 1.00)
  2. Basic PLA Black - #000000 (RGB: 0.00, 0.00, 0.00)
  3. Basic PLA Red - #FF0000 (RGB: 1.00, 0.00, 0.00)
  ...
```

---

## export-materials

Export material database to CSV or JSON file.

```bash
bananaforge export-materials [OPTIONS]
```

### Options
- `--format [csv|json]` - Output format (default: csv)
- `--output, -o PATH` - Output file path (required)
- `--brand TEXT` - Filter by brand (can specify multiple)
- `--max-materials INT` - Maximum number of materials
- `--color-diversity` - Optimize for color diversity (default: true)

### Examples

#### Export Default Materials
```bash
bananaforge export-materials --output materials.csv
```

#### Export Specific Brand
```bash
bananaforge export-materials \
  --output bambu_only.csv \
  --brand "Bambu Lab" \
  --max-materials 15
```

#### Export as JSON
```bash
bananaforge export-materials \
  --format json \
  --output materials.json \
  --color-diversity
```

---

## validate-stl

Validate STL file for 3D printing quality.

```bash
bananaforge validate-stl STL_FILE
```

### Arguments
- `STL_FILE` - Path to STL file to validate

### Examples

```bash
bananaforge validate-stl model.stl
```

### Output
```
STL Validation Report
====================

File: model.stl
Size: 2.4 MB
Triangles: 12,847
Vertices: 6,425

Quality Checks:
✅ Watertight mesh
✅ No degenerate triangles
✅ Proper orientation
✅ Reasonable dimensions

Printability: EXCELLENT
```

---

## init-config

Create a configuration file with default settings.

```bash
bananaforge init-config [OPTIONS]
```

### Options
- `--output, -o PATH` - Output file path (default: `config.json`)
- `--template [default|quality|fast]` - Configuration template

### Examples

#### Create Default Config
```bash
bananaforge init-config
```

#### Create Quality-Focused Config
```bash
bananaforge init-config \
  --template quality \
  --output quality_config.json
```

#### Create Fast Config
```bash
bananaforge init-config \
  --template fast \
  --output fast_config.json
```

### Generated Config Example
```json
{
  "optimization": {
    "max_layers": 50,
    "layer_height": 0.2,
    "max_materials": 8,
    "iterations": 1000,
    "learning_rate": 0.01,
    "resolution": 256
  },
  "output": {
    "physical_size": 100.0,
    "export_formats": ["stl", "instructions", "cost_report"],
    "project_name": "bananaforge_model"
  },
  "materials": {
    "database_path": null,
    "color_matching": "perceptual"
  },
  "system": {
    "device": "auto",
    "verbose": false,
    "quiet": false
  }
}
```

---

## version

Show BananaForge version information.

```bash
bananaforge version
```

### Examples

```bash
bananaforge version
```

### Output
```
BananaForge 0.1.0
```

---

## Configuration Files

BananaForge supports configuration files in JSON format. This creates a JSON file you can edit to customize BananaForge settings.

### Using Configuration Files

```bash
# Create config file
bananaforge init-config --output my_config.json

# Use config file
bananaforge convert photo.jpg --config my_config.json
```

### Configuration Options

#### Optimization Settings
- `max_layers`: Maximum number of layers (default: 50)
- `layer_height`: Layer height in mm (default: 0.2)
- `max_materials`: Maximum materials to use (default: 8)
- `iterations`: Optimization iterations (default: 1000)
- `learning_rate`: Learning rate (default: 0.01)
- `resolution`: Processing resolution (default: 256)

#### Output Settings
- `physical_size`: Physical size in mm (default: 100.0)
- `export_formats`: List of export formats
- `project_name`: Default project name

#### Material Settings
- `database_path`: Path to material database
- `color_matching`: Color matching method

#### System Settings
- `device`: Computation device (auto/cpu/cuda/mps)
- `verbose`: Enable verbose logging
- `quiet`: Suppress output

### Environment Variables

- `BANANAFORGE_CONFIG`: Path to configuration file
- `BANANAFORGE_LOG_LEVEL`: Logging level
- `BANANAFORGE_DEVICE`: Default device

---

## Examples

### Complete Workflow

```bash
# 1. Analyze image colors
bananaforge analyze-colors photo.jpg --max-materials 6

# 2. Export materials
bananaforge export-materials --format csv --output custom_materials.csv

# 3. Convert with custom settings
bananaforge convert photo.jpg \
  --materials custom_materials.csv \
  --max-materials 6 \
  --iterations 1500 \
  --physical-size 120 \
  --layer-height 0.2 \
  --project-name "photo_lithophane" \
  --output ./my_prints/photo_lithophane/
```

### Batch Processing

```bash
# Process multiple images
for img in *.jpg; do
  bananaforge convert "$img" \
    --output "./output/${img%.*}/" \
    --project-name "${img%.*}"
done
```

### Quality Profiles

```bash
# Fast prototype
bananaforge convert image.jpg \
  --iterations 200 \
  --resolution 128 \
  --max-materials 4

# Balanced quality
bananaforge convert image.jpg \
  --iterations 1000 \
  --resolution 256 \
  --max-materials 8

# High quality
bananaforge convert image.jpg \
  --iterations 2000 \
  --resolution 512 \
  --max-materials 12
```

---

## Troubleshooting

### Common Issues

#### Command Not Found
```bash
# Check installation
pip show bananaforge

# Try alternative
python -m bananaforge --help
```

#### Permission Errors
```bash
# Use user installation
pip install --user bananaforge

# Or use virtual environment
python -m venv venv
source venv/bin/activate
pip install bananaforge
```

#### Memory Issues
```bash
# Reduce resolution
bananaforge convert image.jpg --resolution 128

# Use CPU
bananaforge convert image.jpg --device cpu
```

#### GPU Issues
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU
bananaforge convert image.jpg --device cpu
```

### Getting Help

```bash
# General help
bananaforge --help

# Command-specific help
bananaforge convert --help
bananaforge analyze-colors --help
```

### Exit Codes

- `0`: Success
- `1`: General error
- `2`: Invalid arguments
- `3`: File not found
- `4`: Configuration error

---

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

---

## Support

For more help:

- **Documentation**: [docs/](https://github.com/bananaforge/bananaforge/docs)
- **Issues**: [GitHub Issues](https://github.com/bananaforge/bananaforge/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bananaforge/bananaforge/discussions)

Show BananaForge version information.

```bash
bananaforge version
```

BananaForge 0.1.0