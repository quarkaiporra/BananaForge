# Material Management Example

This example demonstrates how to manage materials in BananaForge, including creating custom material databases, color matching, and optimizing material selection for your specific printer and filaments.

## Overview

Material management in BananaForge includes:
- **Loading and customizing material databases**
- **Advanced color matching algorithms**
- **Material cost optimization**
- **Printer-specific material profiles**
- **Transparency material handling**

## Built-in Material Databases

### Default Materials

```python
from bananaforge.materials.database import DefaultMaterials

# Bambu Lab PLA Basic colors
bambu_pla = DefaultMaterials.create_bambu_basic_pla()
print(f"Bambu PLA materials: {len(bambu_pla)} colors")

# HueForge compatible materials
hueforge_materials = DefaultMaterials.create_hueforge_set()
print(f"HueForge materials: {len(hueforge_materials)} colors")

# Generic PLA set
generic_pla = DefaultMaterials.create_generic_pla()
print(f"Generic PLA materials: {len(generic_pla)} colors")
```

### Export Default Materials

```bash
# Export Bambu materials to CSV
bananaforge export-materials --format csv --preset bambu --output bambu_materials.csv

# Export HueForge materials to JSON
bananaforge export-materials --format json --preset hueforge --output hueforge_materials.json

# Export all available materials
bananaforge export-materials --format csv --output all_materials.csv
```

## Creating Custom Material Databases

### CSV Format

Create a CSV file with material specifications:

```csv
id,name,brand,color_hex,transparency,td,density,temperature,cost,notes
pla_black,PLA Black,Bambu Lab,#000000,0.0,4.0,1.24,220,28.99,Base layer material
pla_white,PLA White,Bambu Lab,#FFFFFF,0.0,4.0,1.24,220,28.99,Highlight material
pla_red,PLA Red,Bambu Lab,#FF0000,0.0,4.0,1.24,220,28.99,Primary color
pla_blue,PLA Blue,Bambu Lab,#0000FF,0.0,4.0,1.24,220,28.99,Primary color
pla_yellow,PLA Yellow,Bambu Lab,#FFFF00,0.0,4.0,1.24,220,28.99,Primary color
pla_clear,PLA Clear,Bambu Lab,#FFFFFF,0.8,4.0,1.24,220,35.99,Transparency layers
```

### JSON Format

```json
{
  "materials": [
    {
      "id": "pla_black",
      "name": "PLA Black", 
      "brand": "Bambu Lab",
      "color_hex": "#000000",
      "transparency": 0.0,
      "td": 4.0,
      "density": 1.24,
      "temperature": 220,
      "cost": 28.99,
      "notes": "Base layer material"
    },
    {
      "id": "pla_clear",
      "name": "PLA Clear",
      "brand": "Bambu Lab", 
      "color_hex": "#FFFFFF",
      "transparency": 0.8,
      "td": 4.0,
      "density": 1.24,
      "temperature": 220,
      "cost": 35.99,
      "notes": "Transparency layers"
    }
  ]
}
```

### Loading Custom Materials

```python
from bananaforge.materials.database import MaterialDatabase

# Load from CSV
materials = MaterialDatabase.load_from_csv("my_materials.csv")

# Load from JSON
materials = MaterialDatabase.load_from_json("my_materials.json")

# Combine multiple databases
bambu_materials = MaterialDatabase.load_from_csv("bambu_materials.csv")
custom_materials = MaterialDatabase.load_from_csv("custom_materials.csv")
combined = bambu_materials.merge(custom_materials)
```

## Advanced Color Matching

### Color Matching Methods

```python
from bananaforge.materials.matcher import ColorMatcher

matcher = ColorMatcher()

# Different color matching algorithms
target_color = "#FF6B35"  # Orange color

# Perceptual matching (LAB color space)
perceptual_match = matcher.match_color(
    target_color,
    materials,
    method="perceptual"
)

# Euclidean matching (RGB color space)
euclidean_match = matcher.match_color(
    target_color,
    materials,
    method="euclidean"
)

# Delta-E matching (most accurate)
delta_e_match = matcher.match_color(
    target_color,
    materials,
    method="delta_e"
)

print(f"Perceptual match: {perceptual_match.name} (distance: {perceptual_match.distance:.3f})")
print(f"Euclidean match: {euclidean_match.name} (distance: {euclidean_match.distance:.3f})")
print(f"Delta-E match: {delta_e_match.name} (distance: {delta_e_match.distance:.3f})")
```

### Batch Color Matching

```python
# Match multiple colors at once
image_colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"]

matched_materials = matcher.match_colors(
    colors=image_colors,
    materials=materials,
    method="perceptual",
    max_materials=6,  # Limit total materials
    allow_duplicates=True
)

for color, material in matched_materials.items():
    print(f"{color} → {material.name} ({material.color_hex})")
```

### Color Palette Optimization

```python
# Optimize material selection for entire image
from bananaforge.materials.optimizer import MaterialOptimizer

optimizer = MaterialOptimizer()

# Analyze image colors
image_analysis = optimizer.analyze_image_colors(
    image="photo.jpg",
    method="kmeans",
    max_colors=20
)

# Optimize material selection
optimized_materials = optimizer.optimize_material_selection(
    image_colors=image_analysis.dominant_colors,
    available_materials=materials,
    max_materials=6,
    optimization_goal="color_accuracy"  # or "cost", "material_swaps"
)

print(f"Optimized materials: {[m.name for m in optimized_materials]}")
```

## Material Properties and Specifications

### Material Properties

```python
from bananaforge.materials.database import Material

# Create custom material with full properties
custom_material = Material(
    id="custom_pla_orange",
    name="Custom PLA Orange",
    brand="Generic",
    color_hex="#FF6B35",
    transparency=0.0,
    td=4.2,  # Thermal deflection temperature
    density=1.24,  # g/cm³
    temperature=215,  # Printing temperature °C
    cost=22.50,  # Cost per kg
    bed_temperature=60,  # Bed temperature °C
    flow_rate=1.0,  # Flow rate multiplier
    retraction_distance=0.8,  # Retraction distance mm
    print_speed=60,  # Print speed mm/s
    notes="Custom orange filament for vibrant prints"
)

# Add to database
materials.add_material(custom_material)
```

### Material Validation

```python
# Validate material database
validation_results = materials.validate()

if validation_results.is_valid:
    print("Material database is valid")
else:
    print("Validation errors:")
    for error in validation_results.errors:
        print(f"  - {error}")
```

## Transparency Materials

### Transparency Material Setup

```python
# Create transparency materials
clear_materials = [
    Material(
        id="pla_clear_33",
        name="PLA Clear 33%",
        color_hex="#FFFFFF",
        transparency=0.33,
        cost=35.99
    ),
    Material(
        id="pla_clear_67", 
        name="PLA Clear 67%",
        color_hex="#FFFFFF",
        transparency=0.67,
        cost=35.99
    )
]

# Add to material database
for material in clear_materials:
    materials.add_material(material)
```

### Transparency Optimization

```python
from bananaforge.materials.transparency_optimizer import TransparencyOptimizer

# Configure transparency materials
transparency_optimizer = TransparencyOptimizer()

# Optimize with transparency materials
result = transparency_optimizer.optimize(
    image="photo.jpg",
    materials=materials,
    transparency_config={
        'opacity_levels': [0.33, 0.67, 1.0],
        'transparency_materials': ['pla_clear_33', 'pla_clear_67'],
        'base_materials': ['pla_black', 'pla_white'],
        'optimize_base_layers': True
    }
)
```

## Printer-Specific Material Profiles

### Bambu Lab Profile

```python
# Create Bambu Lab specific materials
bambu_profile = MaterialDatabase()

bambu_materials = [
    Material(id="bambu_pla_basic_black", name="Bambu PLA Basic Black", 
             color_hex="#000000", temperature=220, bed_temperature=45),
    Material(id="bambu_pla_basic_white", name="Bambu PLA Basic White",
             color_hex="#FFFFFF", temperature=220, bed_temperature=45),
    Material(id="bambu_pla_basic_red", name="Bambu PLA Basic Red",
             color_hex="#FF0000", temperature=220, bed_temperature=45),
    # ... more materials
]

for material in bambu_materials:
    bambu_profile.add_material(material)

# Save profile
bambu_profile.save_to_csv("bambu_lab_profile.csv")
```

### Prusa Profile

```python
# Create Prusa specific materials
prusa_profile = MaterialDatabase()

prusa_materials = [
    Material(id="prusament_pla_black", name="Prusament PLA Black",
             color_hex="#000000", temperature=215, bed_temperature=60),
    Material(id="prusament_pla_white", name="Prusament PLA White", 
             color_hex="#FFFFFF", temperature=215, bed_temperature=60),
    # ... more materials
]

for material in prusa_materials:
    prusa_profile.add_material(material)
```

## Cost Optimization

### Material Cost Analysis

```python
from bananaforge.materials.cost_analyzer import CostAnalyzer

analyzer = CostAnalyzer()

# Analyze material costs for image
cost_analysis = analyzer.analyze_image_costs(
    image="photo.jpg",
    materials=materials,
    optimization_settings={
        'max_materials': 6,
        'layer_height': 0.2,
        'physical_size': 100  # mm
    }
)

print(f"Estimated material cost: ${cost_analysis.total_cost:.2f}")
print(f"Material breakdown:")
for material, cost in cost_analysis.material_costs.items():
    print(f"  {material.name}: ${cost:.2f}")
```

### Cost-Optimized Material Selection

```python
# Optimize for minimum cost
cost_optimized = optimizer.optimize_material_selection(
    image_colors=image_analysis.dominant_colors,
    available_materials=materials,
    max_materials=6,
    optimization_goal="cost",
    cost_weight=0.7,  # 70% cost, 30% color accuracy
    color_accuracy_weight=0.3
)

print(f"Cost-optimized selection: {[m.name for m in cost_optimized]}")
```

## CLI Material Management

### Material Database Operations

```bash
# List available materials
bananaforge analyze-colors --materials materials.csv --list-materials

# Find best matches for specific colors
bananaforge analyze-colors image.jpg --materials materials.csv --method perceptual

# Export material usage analysis
bananaforge convert image.jpg --materials materials.csv --export-format cost_report
```

### Material Validation

```bash
# Validate material database
bananaforge validate-materials materials.csv

# Check material compatibility
bananaforge validate-materials materials.csv --check-compatibility
```

## Advanced Material Features

### Material Mixing Simulation

```python
from bananaforge.materials.mixer import MaterialMixer

mixer = MaterialMixer()

# Simulate material mixing
material_a = materials.get_material("pla_red")
material_b = materials.get_material("pla_yellow")

mixed_color = mixer.mix_colors(
    material_a.color_hex,
    material_b.color_hex,
    ratio=0.5  # 50/50 mix
)

print(f"Mixed color: {mixed_color}")
```

### Material Recommendation Engine

```python
from bananaforge.materials.recommender import MaterialRecommender

recommender = MaterialRecommender()

# Get recommendations for image
recommendations = recommender.recommend_materials(
    image="portrait.jpg",
    printer_type="bambu_x1c",
    budget_limit=200,  # $200 budget
    skill_level="beginner"
)

print("Recommended materials:")
for rec in recommendations:
    print(f"  {rec.material.name} - Score: {rec.score:.2f}")
    print(f"    Reasoning: {rec.reasoning}")
```

## Best Practices

### Material Database Management

1. **Start with tested materials** - Use materials you've successfully printed with
2. **Maintain material properties** - Keep temperature, flow rate, and other settings accurate
3. **Regular updates** - Update costs and availability regularly
4. **Version control** - Keep different versions of material databases for different projects
5. **Documentation** - Add notes about material performance and quirks

### Color Matching Strategies

1. **Use perceptual matching** for most images - more accurate than RGB
2. **Consider printer limitations** - some printers handle certain colors better
3. **Test material combinations** - verify materials work well together
4. **Account for layer transparency** - materials may look different when layered
5. **Calibrate your printer** - ensure printed colors match expected results

### Cost Optimization

1. **Balance cost vs. quality** - sometimes more expensive materials give better results
2. **Consider material waste** - factor in purge tower and failed prints
3. **Buy in bulk** - larger spools often have better per-kg pricing
4. **Track actual usage** - monitor real consumption vs. estimates
5. **Plan ahead** - order materials before starting large projects

## Troubleshooting

### Poor Color Matches

```python
# Try different color matching methods
for method in ["perceptual", "euclidean", "delta_e"]:
    match = matcher.match_color(target_color, materials, method=method)
    print(f"{method}: {match.name} (distance: {match.distance:.3f})")
```

### Material Database Errors

```python
# Debug material loading issues
try:
    materials = MaterialDatabase.load_from_csv("materials.csv")
except Exception as e:
    print(f"Error loading materials: {e}")
    # Check file format, encoding, required columns
```

### Cost Calculation Issues

```python
# Verify material cost data
for material in materials:
    if material.cost is None or material.cost <= 0:
        print(f"Warning: {material.name} has invalid cost: {material.cost}")
```

## Next Steps

- Try [Transparency Mixing](transparency-mixing.md) with custom materials
- Learn about [3MF Export](3mf-export.md) with material properties
- Explore [Batch Processing](batch-processing.md) with different material sets
- Check out [Advanced Optimization](advanced-optimization.md) techniques