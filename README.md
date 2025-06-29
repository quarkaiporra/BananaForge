# BananaForge

AI-powered multi-layer 3D printing optimization tool.

## Overview

BananaForge converts 2D images into optimized multi-layer 3D models for color printing. Using advanced machine learning techniques including differentiable optimization and Gumbel softmax sampling, it automatically generates layer-by-layer material assignments and height maps.

## Key Features

- **AI Optimization**: PyTorch-based gradient descent with custom loss functions
- **Multi-Material Support**: Automatic filament selection and layer assignment
- **Export Formats**: STL files, swap instructions, and project files
- **Height Map Optimization**: Intelligent layer height calculation
- **Color Matching**: Advanced color space optimization for material selection

## Quick Start

```bash
pip install bananaforge
bananaforge convert image.jpg --materials materials.csv --output ./output/
```

## Installation

### From PyPI (recommended)
```bash
pip install bananaforge
```

### Development Installation
```bash
git clone https://github.com/bananaforge/bananaforge.git
cd bananaforge
pip install -e .[dev]
```

## Usage

### Basic Conversion
```bash
bananaforge convert input.jpg --materials filaments.csv
```

### Advanced Options
```bash
bananaforge convert input.jpg \
  --materials filaments.csv \
  --max-layers 50 \
  --layer-height 0.2 \
  --max-colors 8 \
  --iterations 1000 \
  --output ./my_output/
```

## Architecture

- **Core Engine**: Differentiable optimization with Gumbel softmax
- **Image Processing**: Multi-scale analysis and preprocessing 
- **Material System**: Color space optimization and material matching
- **Output Generation**: STL export and printing instruction generation

## License

To be added

## Contributing

We welcome contributions! Please see our contributing guidelines for details.