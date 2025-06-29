# BananaForge Examples

Real-world examples and tutorials for using BananaForge effectively.

## Example Categories

### Basic Usage
- [Simple Conversion](basic-conversion.md) - Your first image conversion
- [Color Analysis](color-analysis.md) - Analyzing colors before conversion
- [Material Management](material-management.md) - Working with material databases

### Advanced Workflows
- [Batch Processing](batch-processing.md) - Processing multiple images
- [Custom Materials](custom-materials.md) - Creating custom material sets
- [Quality Optimization](quality-optimization.md) - Getting the best results

### Integration Examples
- [Python Scripts](python-integration.md) - Using the Python API
- [Web Application](web-integration.md) - Building a web interface
- [Automation](automation-scripts.md) - Automated processing workflows

### Printer-Specific
- [Bambu Lab](bambu-lab-workflow.md) - Bambu Lab printer workflow
- [Prusa](prusa-workflow.md) - Prusa printer workflow
- [Multi-printer](multi-printer-setup.md) - Managing multiple printers

### Use Case Examples
- [Portrait Photography](portrait-printing.md) - Converting portraits
- [Landscape Images](landscape-printing.md) - Nature and landscape conversion
- [Logos and Graphics](logo-printing.md) - Converting graphic designs
- [Art Reproduction](art-reproduction.md) - Reproducing artwork

## Quick Reference

### Common Commands

```bash
# Basic conversion
bananaforge convert image.jpg

# High quality conversion
bananaforge convert image.jpg --iterations 2000 --resolution 512

# Analyze colors first
bananaforge analyze-colors image.jpg --max-materials 8

# Export materials database
bananaforge export-materials --output my_materials.csv

# Validate STL file
bananaforge validate-stl model.stl
```

### Python API Snippets

```python
# Basic optimization
from bananaforge import LayerOptimizer, ImageProcessor, MaterialDatabase
processor = ImageProcessor()
image = processor.load_image("photo.jpg")
# ... optimization code ...

# Material matching
from bananaforge.materials.matcher import ColorMatcher
matcher = ColorMatcher(material_db)
materials, colors, _ = matcher.optimize_material_selection(image, 6)

# Export results
from bananaforge.output.exporter import ModelExporter
exporter = ModelExporter()
files = exporter.export_complete_model(height_map, assignments, material_db, materials, "./output")
```

## Getting Started

1. **Begin with [Simple Conversion](basic-conversion.md)** - Learn the basics
2. **Try [Color Analysis](color-analysis.md)** - Understand material matching
3. **Explore [Material Management](material-management.md)** - Organize your materials
4. **Move to [Advanced Workflows](batch-processing.md)** - Scale your usage

## Example Files

All examples include:
- **Complete code** - Ready to run examples
- **Sample inputs** - Test images and materials
- **Expected outputs** - What to expect
- **Troubleshooting** - Common issues and solutions

## Contributing Examples

Have a great BananaForge workflow? Share it!

1. Create a new markdown file
2. Include complete code examples
3. Add sample inputs/outputs
4. Document any special requirements
5. Submit a pull request

## Support

- 📖 **Documentation**: [Main Docs](../README.md)
- 🐛 **Issues**: [GitHub Issues](https://github.com/bananaforge/bananaforge/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/bananaforge/bananaforge/discussions)