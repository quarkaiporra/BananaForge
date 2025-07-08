# Basic Conversion Example

This example demonstrates the most basic usage of BananaForge to convert a 2D image into a multi-layer 3D model.

## Prerequisites

- BananaForge installed (`pip install -e .`)
- A sample image file (JPG, PNG, or similar)
- Basic understanding of 3D printing concepts

## Simple Conversion

### Command Line Usage

```bash
# Most basic conversion
bananaforge convert photo.jpg

# With custom output directory
bananaforge convert photo.jpg --output ./my_models/

# With specific material count
bananaforge convert photo.jpg --max-materials 6 --output ./output/
```

### Python API Usage

```python
from bananaforge import LayerOptimizer, ImageProcessor
from bananaforge.materials.database import DefaultMaterials

# Load image
processor = ImageProcessor()
image = processor.load("photo.jpg")

# Setup materials
materials = DefaultMaterials.create_bambu_basic_pla()

# Create optimizer
optimizer = LayerOptimizer()

# Optimize
result = optimizer.optimize(image, materials=materials)

# Export STL
result.export_stl("my_model.stl")
```

## Step-by-Step Breakdown

### 1. Image Loading and Preprocessing

```python
from bananaforge.image.processor import ImageProcessor

processor = ImageProcessor()

# Load image
image = processor.load("photo.jpg")
print(f"Image loaded: {image.shape}")

# The processor automatically:
# - Converts to RGB if needed
# - Normalizes pixel values
# - Handles different image formats
```

### 2. Material Database Setup

```python
from bananaforge.materials.database import DefaultMaterials

# Use default Bambu Lab PLA materials
materials = DefaultMaterials.create_bambu_basic_pla()
print(f"Loaded {len(materials)} materials")

# Or load custom materials
# materials = MaterialDatabase.load_from_csv("my_materials.csv")
```

### 3. Basic Optimization

```python
from bananaforge.core.optimizer import LayerOptimizer, OptimizationConfig

# Configure optimization
config = OptimizationConfig(
    max_layers=30,      # Limit number of layers
    max_materials=6,    # Limit number of different materials
    iterations=500,     # Optimization iterations
    layer_height=0.2    # Layer height in mm
)

# Create optimizer
optimizer = LayerOptimizer(config=config)

# Run optimization
result = optimizer.optimize(image, materials=materials)
```

### 4. Export Results

```python
from bananaforge.output.exporter import ModelExporter

exporter = ModelExporter()

# Export STL file
exporter.export_stl(result, "model.stl")

# Export material swap instructions
exporter.export_instructions(result, "instructions.txt")

# Export cost analysis
exporter.export_cost_report(result, "cost_report.txt")
```

## Understanding the Output

### STL File
- Contains the 3D geometry of your model
- Each layer represents a different height
- Ready for slicing with your 3D printer software

### Instructions File
Contains material swap instructions like:
```
Layer 0-5: Bambu PLA White
Layer 6-12: Bambu PLA Red  
Layer 13-18: Bambu PLA Blue
Layer 19-25: Bambu PLA White
```

### Cost Report
Provides analysis including:
- Total material usage
- Cost breakdown by material
- Estimated print time
- Material swap count

## Example Output

Running the basic conversion on a 300x300 pixel image typically produces:

```
Processing image: photo.jpg
Image resolution: 300x300 pixels
Materials available: 8 colors
Optimization iterations: 500
Target layers: 30

Optimization Results:
- Final loss: 0.0234
- Layers used: 28
- Materials used: 5
- Total height: 5.6mm
- Material swaps: 12

Export complete:
- model.stl (2.3MB)
- instructions.txt
- cost_report.txt
```

## Tips for Better Results

### Image Preparation
- Use high-contrast images for better layer definition
- Consider cropping to focus on the main subject
- Ensure good lighting and color saturation

### Material Selection
- Start with a basic set of 4-6 colors
- Choose colors that match your image palette
- Consider material compatibility with your printer

### Optimization Settings
- More iterations = better quality but longer processing time
- More layers = finer detail but more material swaps
- Balance quality vs. complexity for your needs

## Common Issues and Solutions

### Low Quality Output
```bash
# Increase optimization iterations
bananaforge convert photo.jpg --iterations 1000

# Increase layer count
bananaforge convert photo.jpg --max-layers 50
```

### Too Many Material Swaps
```bash
# Reduce material count
bananaforge convert photo.jpg --max-materials 4

# Reduce layer count
bananaforge convert photo.jpg --max-layers 20
```

### Large File Sizes
```bash
# Reduce processing resolution
bananaforge convert photo.jpg --resolution 200

# Reduce layer count
bananaforge convert photo.jpg --max-layers 25
```

## Next Steps

- Try [Advanced Transparency Features](transparency-mixing.md)
- Learn about [Material Management](material-management.md)  
- Explore [Batch Processing](batch-processing.md)
- Check out [3MF Export](3mf-export.md) for professional workflows