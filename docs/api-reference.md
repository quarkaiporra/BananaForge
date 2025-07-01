# API Reference

Use BananaForge programmatically in your Python applications.

## Overview

BananaForge provides a comprehensive Python API for integrating multi-layer 3D printing optimization into your applications.

## Quick Start

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

## Core Components

### LayerOptimizer

The main optimization engine that converts images to 3D models.

```python
from bananaforge.core.optimizer import LayerOptimizer, OptimizationConfig

# Create optimizer with default settings
optimizer = LayerOptimizer()

# Or with custom configuration
config = OptimizationConfig(
    max_layers=60,
    layer_height=0.15,
    max_materials=8,
    iterations=1500
)
optimizer = LayerOptimizer(config=config)

# Optimize image
result = optimizer.optimize(image_path, config=config)
```

#### Methods

- `optimize(image, config=None)`: Main optimization method
- `set_materials(materials)`: Set material database
- `get_progress()`: Get optimization progress
- `save_checkpoint(path)`: Save optimization state
- `load_checkpoint(path)`: Load optimization state

### ImageProcessor

Handles image loading, preprocessing, and analysis.

```python
from bananaforge.image.processor import ImageProcessor

processor = ImageProcessor()

# Load image
image = processor.load("photo.jpg")

# Preprocess image
processed = processor.preprocess(
    image,
    resolution=256,
    normalize=True
)

# Analyze colors
colors = processor.analyze_colors(image, max_colors=8)
```

#### Methods

- `load(path)`: Load image from file
- `preprocess(image, resolution=256, normalize=True)`: Preprocess image
- `analyze_colors(image, max_colors=8)`: Analyze dominant colors
- `resize(image, size)`: Resize image
- `normalize(image)`: Normalize pixel values

### MaterialDatabase

Manages material information and color matching.

```python
from bananaforge.materials.database import MaterialDatabase, Material

# Load default materials
materials = MaterialDatabase.load_default()

# Load custom materials
materials = MaterialDatabase.load_from_csv("my_materials.csv")

# Create custom material
custom_material = Material(
    name="My PLA Red",
    color="#FF0000",
    density=1.24,
    cost_per_kg=25.0
)

# Add to database
materials.add_material(custom_material)
```

#### Methods

- `load_default()`: Load built-in material database
- `load_from_csv(path)`: Load from CSV file
- `load_from_json(path)`: Load from JSON file
- `add_material(material)`: Add material to database
- `remove_material(name)`: Remove material by name
- `get_material(name)`: Get material by name
- `find_best_match(color, method="perceptual")`: Find best color match

### ColorMatcher

Advanced color matching algorithms.

```python
from bananaforge.materials.matcher import ColorMatcher

matcher = ColorMatcher()

# Match single color
best_match = matcher.match_color(
    target_color="#FF0000",
    materials=materials,
    method="perceptual"
)

# Match multiple colors
matches = matcher.match_colors(
    colors=["#FF0000", "#00FF00", "#0000FF"],
    materials=materials,
    max_materials=6
)
```

#### Methods

- `match_color(color, materials, method="perceptual")`: Match single color
- `match_colors(colors, materials, max_materials=8)`: Match multiple colors
- `calculate_distance(color1, color2, method="perceptual")`: Calculate color distance

## Output Generation

### ModelExporter

Handles export of optimized models to various formats.

```python
from bananaforge.output.exporter import ModelExporter

exporter = ModelExporter()

# Export STL file
exporter.export_stl(result, "model.stl")

# Export instructions
exporter.export_instructions(result, "instructions.txt")

# Export cost report
exporter.export_cost_report(result, "cost_report.txt")

# Export all formats
exporter.export_all(result, output_dir="./output/")
```

#### Methods

- `export_stl(result, path)`: Export STL file
- `export_instructions(result, path)`: Export swap instructions
- `export_cost_report(result, path)`: Export cost analysis
- `export_all(result, output_dir)`: Export all formats

### STLGenerator

Generates STL mesh files from optimization results.

```python
from bananaforge.output.stl_generator import STLGenerator

generator = STLGenerator()

# Generate STL from height map
mesh = generator.generate_mesh(
    height_map=result.height_map,
    layer_height=0.2,
    physical_size=100.0
)

# Save STL file
generator.save_stl(mesh, "output.stl")
```

## Configuration

### ConfigManager

Manages application configuration.

```python
from bananaforge.utils.config import ConfigManager, Config

# Load configuration
config_manager = ConfigManager("config.json")
config = config_manager.get_config()

# Create custom configuration
custom_config = Config(
    max_layers=60,
    layer_height=0.15,
    max_materials=8,
    iterations=1500,
    resolution=384
)

# Save configuration
config_manager.save_config(custom_config, "my_config.json")
```

## Advanced Usage

### Custom Optimization Loop

```python
from bananaforge.core.optimizer import LayerOptimizer
from bananaforge.utils.logging import get_logger

logger = get_logger(__name__)

optimizer = LayerOptimizer()

# Custom optimization with progress tracking
for iteration in range(1000):
    optimizer.step()
    
    if iteration % 100 == 0:
        loss = optimizer.get_loss()
        logger.info(f"Iteration {iteration}, Loss: {loss:.4f}")
    
    # Early stopping
    if optimizer.should_stop():
        break

result = optimizer.get_result()
```

### Batch Processing

```python
import os
from pathlib import Path
from bananaforge import LayerOptimizer, ImageProcessor

optimizer = LayerOptimizer()
processor = ImageProcessor()

# Process multiple images
input_dir = Path("input_images/")
output_dir = Path("output_models/")

for image_path in input_dir.glob("*.jpg"):
    # Process image
    image = processor.load(str(image_path))
    
    # Optimize
    result = optimizer.optimize(image)
    
    # Export
    output_path = output_dir / f"{image_path.stem}_model.stl"
    result.export_stl(str(output_path))
```

### Custom Material Matching

```python
from bananaforge.materials.matcher import ColorMatcher
from bananaforge.materials.database import MaterialDatabase

matcher = ColorMatcher()
materials = MaterialDatabase.load_default()

# Custom color matching function
def custom_color_match(target_color, available_materials):
    # Your custom logic here
    matches = matcher.match_color(
        target_color,
        available_materials,
        method="lab"
    )
    
    # Apply custom filtering
    filtered_matches = [
        m for m in matches 
        if m.cost_per_kg < 30.0  # Only affordable materials
    ]
    
    return filtered_matches
```

## Error Handling

```python
from bananaforge.utils.logging import get_logger
from bananaforge.core.exceptions import OptimizationError

logger = get_logger(__name__)

try:
    optimizer = LayerOptimizer()
    result = optimizer.optimize("image.jpg")
except OptimizationError as e:
    logger.error(f"Optimization failed: {e}")
    # Handle error appropriately
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    # Handle missing file
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Handle other errors
```

## Performance Optimization

### GPU Acceleration

```python
import torch
from bananaforge.core.optimizer import LayerOptimizer

# Check available devices
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Create optimizer with specific device
optimizer = LayerOptimizer(device=device)
```

### Memory Management

```python
from bananaforge.core.optimizer import LayerOptimizer

# Use lower resolution for memory-constrained systems
config = OptimizationConfig(
    resolution=128,  # Lower resolution
    max_layers=30,   # Fewer layers
    batch_size=1     # Smaller batch size
)

optimizer = LayerOptimizer(config=config)
```

## Integration Examples

### Flask Web Application

```python
from flask import Flask, request, jsonify
from bananaforge import LayerOptimizer, ImageProcessor
import tempfile
import os

app = Flask(__name__)

@app.route('/convert', methods=['POST'])
def convert_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        file.save(tmp.name)
        image_path = tmp.name
    
    try:
        # Process image
        processor = ImageProcessor()
        image = processor.load(image_path)
        
        # Optimize
        optimizer = LayerOptimizer()
        result = optimizer.optimize(image)
        
        # Export
        output_path = f"output_{os.path.basename(image_path)}.stl"
        result.export_stl(output_path)
        
        return jsonify({
            'success': True,
            'output_file': output_path
        })
    
    finally:
        # Clean up
        os.unlink(image_path)
```

### Jupyter Notebook

```python
# In a Jupyter notebook
import matplotlib.pyplot as plt
from bananaforge import LayerOptimizer, ImageProcessor
from bananaforge.utils.visualization import plot_optimization_results

# Load and process image
processor = ImageProcessor()
image = processor.load("sample.jpg")

# Display original image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")

# Optimize
optimizer = LayerOptimizer()
result = optimizer.optimize(image)

# Display results
plt.subplot(1, 2, 2)
plot_optimization_results(result)
plt.title("Optimization Results")
plt.show()
```

## Troubleshooting

### Common Issues

#### Import Errors
```python
# Make sure BananaForge is installed
pip install bananaforge

# Check installation
import bananaforge
print(bananaforge.__version__)
```

#### Memory Issues
```python
# Use lower resolution
config = OptimizationConfig(resolution=128)

# Use CPU instead of GPU
optimizer = LayerOptimizer(device="cpu")
```

#### GPU Issues
```python
import torch

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")

# Check MPS availability (Apple Silicon)
print(f"MPS available: {torch.backends.mps.is_available()}")
```

## Support

For API support:

- **Documentation**: [docs/](https://github.com/eddieoz/BananaForge/docs)
- **Issues**: [GitHub Issues](https://github.com/eddieoz/BananaForge/issues)
- **Examples**: [examples/](https://github.com/eddieoz/BananaForge/examples)