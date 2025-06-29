# BananaForge Documentation

Welcome to BananaForge - the professional AI-powered multi-layer 3D printing optimization tool.

## 🚀 Quick Start

```bash
# Install BananaForge
pip install bananaforge

# Convert your first image
bananaforge convert photo.jpg --materials bambu_pla.csv --output ./my_model/

# Analyze colors before converting
bananaforge analyze-colors photo.jpg --max-materials 8
```

## 📚 Documentation Sections

### Getting Started
1. **[Install BananaForge](installation.md)** - System requirements and installation
2. **[Quick Start Guide](quickstart.md)** - Your first conversion in 5 minutes
3. **[Configuration](configuration.md)** - Customize settings and workflows
4. **[Materials Guide](materials.md)** - Manage materials and color matching

### Reference
5. **[CLI Reference](cli-reference.md)** - Complete command-line interface
6. **[API Reference](api-reference.md)** - Python programming interface
7. **[Examples](examples/README.md)** - Real-world usage examples

### Advanced
8. **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## 🎯 What is BananaForge?

BananaForge is a tool that converts 2D images into optimized multi-layer 3D models for color printing. Unlike other solutions, BananaForge:

- **AI-Powered**: Uses machine learning for optimal material selection
- **Multi-Material**: Supports unlimited filament types and colors
- **Universal**: Works with any 3D printer brand
- **Optimized**: Minimizes material waste and print time
- **Professional**: Production-ready with enterprise features

## 🔧 Key Features

- **Intelligent Color Matching**: Advanced color space optimization
- **Layer Height Optimization**: Automatic height map generation
- **Material Cost Analysis**: Detailed cost and weight calculations
- **Multiple Export Formats**: STL, instructions, cost reports
- **GPU Acceleration**: CUDA and MPS support for faster processing
- **Batch Processing**: Convert multiple images efficiently

## 📖 Examples

```python
from bananaforge import LayerOptimizer, ImageProcessor, MaterialDatabase

# Load and process image
processor = ImageProcessor()
image = processor.load("my_photo.jpg")

# Setup materials
materials = MaterialDatabase.load_default()

# Optimize layers
optimizer = LayerOptimizer(materials)
result = optimizer.optimize(image, max_layers=50)

# Export results
result.export_stl("output.stl")
result.export_instructions("instructions.txt")
```

## 🤝 Community & Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/bananaforge/bananaforge/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/bananaforge/bananaforge/discussions)
- 📧 **Contact**: support@bananaforge.com

## 📄 License

MIT License - see [LICENSE](../LICENSE) for details.