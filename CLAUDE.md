# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ForgeBambu is an AI-powered multi-layer 3D printing optimization tool that converts 2D images into optimized multi-layer 3D models for color printing. It uses PyTorch-based differentiable optimization with Gumbel softmax sampling for layer-by-layer material assignments and height map generation.

## Development Commands

### Installation and Setup
```bash
# Development installation
pip install -e .[dev]

# Install main dependencies only
pip install -e .
```

### Testing
```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test
pytest tests/test_basic_functionality.py -v

# Run with coverage
pytest tests/ --cov=forgebambu --cov-report=html
```

### Code Quality
```bash
# Format code
black src/

# Sort imports
isort src/

# Type checking
mypy src/

# Linting
flake8 src/
```

### CLI Usage
```bash
# Basic conversion
forgebambu convert input.jpg --materials materials.csv

# Full workflow with custom options
forgebambu convert input.jpg \
  --materials filaments.csv \
  --max-layers 50 \
  --layer-height 0.2 \
  --max-colors 8 \
  --iterations 1000 \
  --output ./output/

# Export default materials
forgebambu export-materials --output materials.csv

# Analyze image colors
forgebambu analyze-colors input.jpg --materials materials.csv

# Validate STL files
forgebambu validate-stl output.stl
```

## Architecture Overview

### Core Components

1. **Core Engine** (`src/forgebambu/core/`)
   - `optimizer.py`: Main `LayerOptimizer` class using differentiable optimization
   - `gumbel.py`: Gumbel softmax implementation for discrete sampling
   - `loss.py`: Combined loss functions for optimization

2. **Image Processing** (`src/forgebambu/image/`)
   - `processor.py`: `ImageProcessor` for loading and preprocessing images
   - `heightmap.py`: Height map generation and analysis
   - `filters.py`: Image filtering and enhancement

3. **Material System** (`src/forgebambu/materials/`)
   - `manager.py`: High-level `MaterialManager` interface
   - `database.py`: `MaterialDatabase` for storing material properties
   - `matcher.py`: `ColorMatcher` for optimal material selection
   - `optimizer.py`: Material-specific optimization algorithms

4. **Output Generation** (`src/forgebambu/output/`)
   - `exporter.py`: `ModelExporter` for complete model export
   - `stl_generator.py`: STL file generation
   - `instructions.py`: Swap instruction and cost calculation
   - `mesh.py`: Mesh processing and validation

5. **Utilities** (`src/forgebambu/utils/`)
   - `config.py`: Configuration management
   - `logging.py`: Logging setup
   - `visualization.py`: Debugging and preview generation

### Key Workflows

1. **Image to 3D Conversion**:
   - Load image → Process → Match materials → Optimize layers → Export STL + instructions

2. **Optimization Process**:
   - Uses PyTorch with Gumbel softmax for joint height map and material assignment optimization
   - Gradient-based optimization with temperature scheduling
   - Custom loss functions balancing color accuracy, smoothness, and printability

3. **Material Matching**:
   - Color space optimization (LAB, RGB, perceptual)
   - Support for custom material databases (CSV/JSON)
   - Default material sets (Bambu Lab PLA, HueForge)

## Key Dependencies

- **PyTorch**: Core optimization engine
- **OpenCV & Pillow**: Image processing
- **Trimesh**: 3D mesh generation and STL export
- **Click**: CLI interface
- **Pydantic**: Configuration validation

## Testing Strategy

Tests are located in `tests/test_basic_functionality.py` and cover:
- Component initialization and basic functionality
- Image processing pipeline
- Material matching algorithms
- Optimization workflow
- STL generation and export
- CLI command availability

## Configuration

The tool supports JSON configuration files created with:
```bash
forgebambu init-config --output config.json
```

Default material databases can be exported for customization:
```bash
forgebambu export-materials --format csv --output custom_materials.csv
```