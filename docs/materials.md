# Material Management

Learn how to manage materials, create custom databases, and optimize color matching in BananaForge with advanced transparency mixing.

## Overview

Materials are the foundation of multi-color 3D printing. BananaForge provides powerful tools to:

- **Import** material databases from CSV/JSON files
- **Match** image colors to available materials using LAB color space
- **Analyze** color harmony and transparency compatibility
- **Export** material sets optimized for transparency mixing
- **Customize** material properties with transparency support
- **🌈 NEW: Transparency Mixing** - Create more colors with fewer materials (30%+ swap reduction)

## Material Database Structure

### Material Properties

Each material in BananaForge has these properties:

| Property | Type | Description | Example |
|----------|------|-------------|---------|
| `id` | String | Unique identifier | `"bambu_pla_red"` |
| `name` | String | Display name | `"Basic PLA Red"` |
| `brand` | String | Manufacturer | `"Bambu Lab"` |
| `color_rgb` | Tuple | RGB values (0-1) | `(1.0, 0.0, 0.0)` |
| `color_hex` | String | Hex color code | `"#FF0000"` |
| `transparency` | Float | Transparency (0=opaque, 1=clear) | `0.0` |
| `transmission_distance` | Float | TD value for thickness | `4.0` |
| `density` | Float | Material density (g/cm³) | `1.24` |
| `temperature` | Integer | Print temperature (°C) | `220` |
| `cost_per_kg` | Float | Cost per kilogram | `29.99` |
| `available` | Boolean | Whether material is in stock | `true` |
| `tags` | List | Custom tags | `["basic", "pla", "opaque"]` |

## Built-in Material Sets

BananaForge includes several pre-configured material databases:

### Bambu Lab Basic PLA
```bash
# Export default Bambu Lab set
bananaforge export-materials --output bambu_pla.csv
```

Includes 14 common colors:
- White, Black, Red, Blue, Green, Yellow
- Orange, Purple, Pink, Brown, Gray
- Light Blue, Light Green, Transparent

### HueForge Compatible
```bash
# Export HueForge-style materials
bananaforge export-materials --format json --output hueforge_materials.json
```

Generic materials compatible with HueForge workflows.

### Rainbow Test Set
```bash
# Create rainbow colors for testing
bananaforge export-materials --brand "Test" --max-materials 12 --output rainbow.csv
```

## Creating Material Databases

### CSV Format

Create a CSV file with required columns:

```csv
id,name,brand,color_hex,transparency,td,density,temperature,cost,available
bambu_white,Basic PLA White,Bambu Lab,#FFFFFF,0.0,4.0,1.24,220,29.99,true
bambu_black,Basic PLA Black,Bambu Lab,#000000,0.0,4.0,1.24,220,29.99,true
bambu_red,Basic PLA Red,Bambu Lab,#FF0000,0.0,4.0,1.24,220,29.99,true
prusa_orange,Prusament PLA Orange,Prusa,#FF8800,0.0,4.0,1.25,215,34.99,true
```

**Required Columns:**
- `id` - Unique identifier
- `name` - Material name
- `brand` - Manufacturer
- `color_hex` - Color in hex format

**Optional Columns:**
- `transparency` (default: 0.0)
- `td` (default: 4.0)
- `density` (default: 1.25)
- `temperature` (default: 210)
- `cost` (default: 25.0)
- `available` (default: true)

### JSON Format

For more complex data, use JSON:

```json
{
  "materials": [
    {
      "id": "bambu_white",
      "name": "Basic PLA White",
      "brand": "Bambu Lab",
      "color_rgb": [1.0, 1.0, 1.0],
      "color_hex": "#FFFFFF",
      "transparency": 0.0,
      "transmission_distance": 4.0,
      "density": 1.24,
      "temperature": 220,
      "cost_per_kg": 29.99,
      "available": true,
      "tags": ["basic", "pla", "white", "opaque"]
    },
    {
      "id": "bambu_translucent",
      "name": "Basic PLA Natural",
      "brand": "Bambu Lab", 
      "color_rgb": [0.95, 0.95, 0.9],
      "color_hex": "#F2F2E6",
      "transparency": 0.3,
      "transmission_distance": 8.0,
      "density": 1.24,
      "temperature": 220,
      "cost_per_kg": 29.99,
      "available": true,
      "tags": ["basic", "pla", "natural", "translucent"]
    }
  ]
}
```

## Managing Your Material Inventory

### From HueForge

If you use HueForge, export your filament library:

1. **In HueForge**: Go to Filaments → Export → Select your materials → Export CSV
2. **Convert format** (if needed):

```bash
# Analyze HueForge CSV format
head -5 hueforge_export.csv

# Import and reformat for BananaForge
python -c "
import pandas as pd
df = pd.read_csv('hueforge_export.csv')
# Rename columns to match BananaForge format
df_forge = df.rename(columns={
    'Filament Name': 'name',
    'Brand': 'brand', 
    'Color': 'color_hex'
})
df_forge['id'] = df_forge['brand'].str.lower() + '_' + df_forge['name'].str.lower().str.replace(' ', '_')
df_forge.to_csv('my_materials.csv', index=False)
"
```

### From Printer Manufacturer

**Bambu Lab Materials:**
```bash
# Export default Bambu set and customize
bananaforge export-materials --brand "Bambu Lab" --output bambu_base.csv

# Edit bambu_base.csv to mark which materials you actually have
# Set available=false for materials you don't own
```

**Prusa Materials:**
```csv
id,name,brand,color_hex,temperature,cost
prusa_galaxy_black,Prusament PLA Galaxy Black,Prusa,#1A1A1A,215,34.99
prusa_galaxy_silver,Prusament PLA Galaxy Silver,Prusa,#C0C0C0,215,34.99
prusa_orange,Prusament PLA Orange,Prusa,#FF6600,215,34.99
```

### Creating Custom Materials

**Mixing Your Own Colors:**
```csv
id,name,brand,color_hex,notes
custom_forest,Forest Green Mix,Custom,#228B22,"50% Bambu Green + 50% Bambu Black"
custom_sunset,Sunset Orange,Custom,#FF4500,"Custom mixed color"
```

**Specialty Materials:**
```csv
id,name,brand,color_hex,transparency,temperature,notes
wood_pla,Wood PLA,Hatchbox,#8B4513,0.0,190,"Wood-filled PLA"
glow_green,Glow in Dark Green,SUNLU,#32CD32,0.0,200,"Phosphorescent material"
```

## Color Matching Methods

BananaForge offers advanced color matching algorithms with transparency support:

### LAB Color Space Matching (Default & Recommended)

Uses perceptually uniform LAB color space for optimal accuracy:

```bash
bananaforge analyze-colors image.jpg --method lab
```

**🌈 Enhanced in v1.0:**
- **Transparency-aware matching** - considers achievable colors through mixing
- **Base layer optimization** - prioritizes dark colors for better contrast
- **Gradient detection** - identifies areas suitable for transparency effects

**Best for:**
- All image types (now default method)
- Portrait and skin tones
- Natural images
- Transparency mixing optimization

**Characteristics:**
- Perceptually uniform color calculations
- Better skin tone and gradient matching
- Optimized for transparency features

### Euclidean Matching

Simple RGB distance calculation:

```bash
bananaforge analyze-colors image.jpg --method euclidean
```

**Best for:**
- Graphics and logos
- High-contrast images
- When you need predictable results

**Characteristics:**
- Fast computation
- Simple RGB distance
- May not match human perception

### LAB Matching

Perceptual color space with optimal assignment:

```bash
bananaforge analyze-colors image.jpg --method lab
```

**Best for:**
- Scientific accuracy
- Color-critical applications
- When you need the best mathematical match

**Characteristics:**
- Most computationally intensive
- Optimal color assignment
- Best for color-critical work

## Material Selection Strategies

### Color Diversity Optimization with Transparency

Optimize material selection for maximum color coverage through transparency mixing:

```bash
# Select 8 materials optimized for transparency mixing
bananaforge export-materials \
  --max-materials 8 \
  --transparency-optimized \
  --output transparency_set.csv

# Traditional color diversity (still available)
bananaforge export-materials \
  --max-materials 8 \
  --color-diversity \
  --output diverse_set.csv
```

**🌈 New Transparency Optimization:**
- Selects base colors that maximize achievable color palette
- Prioritizes dark base colors for better contrast
- Considers three-layer opacity model (33%, 67%, 100%)
- Can achieve 3x more colors than traditional selection

### Brand-Specific Sets

Create sets for specific printer brands:

```bash
# Bambu Lab only
bananaforge export-materials \
  --brand "Bambu Lab" \
  --max-materials 10 \
  --output bambu_only.csv

# Multiple brands
bananaforge export-materials \
  --brand "Bambu Lab" \
  --brand "Prusa" \
  --output multi_brand.csv
```

### Use Case Specific Sets

**Portrait Photography:**
```csv
id,name,brand,color_hex,notes
flesh_light,Light Skin Tone,Custom,#FDBCB4,"Mixed for portraits"
flesh_medium,Medium Skin Tone,Custom,#E7A888,"Mixed for portraits"  
flesh_dark,Dark Skin Tone,Custom,#A67C5A,"Mixed for portraits"
hair_brown,Brown Hair,Custom,#8B4513,"Natural hair color"
hair_black,Black Hair,Custom,#1C1C1C,"Deep black for hair"
```

**Landscape Photography:**
```csv
id,name,brand,color_hex,notes
sky_blue,Sky Blue,Custom,#87CEEB,"Clear sky color"
grass_green,Grass Green,Custom,#7CFC00,"Vibrant grass"
earth_brown,Earth Brown,Custom,#8B4513,"Soil and bark"
stone_gray,Stone Gray,Custom,#696969,"Rock and concrete"
```

## Advanced Material Features

### Transparency and Transmission

BananaForge now uses transparency for **advanced color mixing** beyond just translucent materials:

```csv
id,name,brand,color_hex,transparency,transmission_distance,mixing_capability
clear_pla,Clear PLA,Generic,#FFFFFF,0.9,12.0,excellent
frosted_white,Frosted White,Generic,#F8F8FF,0.6,8.0,good
tinted_blue,Tinted Blue,Generic,#E6F3FF,0.7,10.0,good
opaque_black,Opaque Black PLA,Generic,#000000,0.0,0.0,base_layer
opaque_red,Opaque Red PLA,Generic,#FF0000,0.0,0.0,overlay
```

**Transparency Values:**
- `0.0` = Completely opaque (ideal for base layers)
- `0.33` = Light transparency (three-layer model)
- `0.67` = Medium transparency (three-layer model)
- `1.0` = Completely clear

**Mixing Capability:**
- `base_layer` = Optimal for dark base colors (maximizes contrast)
- `overlay` = Good for transparency overlay effects
- `excellent` = Perfect for transparency mixing
- `good` = Suitable for transparency effects

### Cost Analysis

Accurate cost tracking:

```csv
id,name,brand,color_hex,cost_per_kg,density
premium_white,Premium PLA White,Brand X,#FFFFFF,45.99,1.24
budget_white,Budget PLA White,Brand Y,#FFFFFF,19.99,1.28
carbon_black,Carbon Fiber PLA,Premium,#1C1C1C,89.99,1.30
```

BananaForge calculates:
- Material volume needed
- Weight based on density  
- Total cost per print
- Cost per color

### Material Tags and Metadata

Organize materials with tags:

```json
{
  "id": "bambu_silk_gold",
  "name": "Silk PLA Gold",
  "brand": "Bambu Lab",
  "color_hex": "#FFD700",
  "tags": ["silk", "metallic", "premium", "decorative"],
  "properties": {
    "finish": "silk",
    "print_speed": "medium",
    "difficulty": "easy"
  }
}
```

## Color Analysis Tools

### Pre-Conversion Analysis with Transparency

Analyze image colors and transparency potential before converting:

```bash
# Basic analysis with transparency features
bananaforge analyze-colors portrait.jpg \
  --max-materials 6 \
  --enable-transparency

# Detailed transparency analysis
bananaforge analyze-colors portrait.jpg \
  --materials my_materials.csv \
  --enable-transparency \
  --transparency-threshold 0.35 \
  --output transparency_analysis.json

# Traditional analysis (still available)
bananaforge analyze-colors portrait.jpg \
  --materials my_materials.csv \
  --method lab \
  --output analysis.json
```

### Transparency and Color Analysis

The analysis now includes transparency mixing potential:

```json
{
  "transparency_analysis": {
    "achievable_colors": 24,
    "base_materials": 6,
    "color_expansion_factor": 4.0,
    "estimated_swap_reduction": 0.35,
    "gradient_regions": 3,
    "transparency_suitability": "excellent"
  },
  "harmony_metrics": {
    "hue_variance": 0.234,
    "saturation_variance": 0.156,
    "complementary_score": 0.8,
    "analogous_score": 0.3,
    "triadic_score": 0.1
  }
}
```

**Harmony Scores (0-1):**
- **Complementary**: Colors opposite on color wheel
- **Analogous**: Colors next to each other
- **Triadic**: Three evenly spaced colors

### Material Matching Quality

See how well materials match image colors:

```bash
bananaforge analyze-colors image.jpg --materials materials.csv
```

```
Color analysis for image.jpg
Method: lab (transparency-aware)
Suggested materials (6):
  1. Basic PLA Black - #000000 (Base Layer) [Contrast: 98%]
  2. Basic PLA White - #FFFFFF (RGB: 1.00, 1.00, 1.00) [Match: 95%]
  3. Basic PLA Red - #DC143C (RGB: 0.86, 0.08, 0.24) [Match: 87%] 
  4. Basic PLA Blue - #4169E1 (RGB: 0.25, 0.41, 0.88) [Match: 82%]

Transparency Analysis:
  Achievable colors through mixing: 18 (3x base materials)
  Estimated material swap reduction: 32%
  Gradient regions suitable for transparency: 2
  Base layer optimization: Excellent (dark base available)
```

## Workflow Examples

### Professional Studio Workflow

```bash
# 1. Maintain master material database
cp studio_master_materials.csv current_inventory.csv

# 2. Update availability based on current stock
# Edit current_inventory.csv in spreadsheet

# 3. Analyze client image
bananaforge analyze-colors client_image.jpg \
  --materials current_inventory.csv \
  --max-materials 8 \
  --output client_analysis.json

# 4. Create project-specific material set
# Based on analysis, create subset for this project
# project_materials.csv

# 5. Convert with optimized materials
bananaforge convert client_image.jpg \
  --materials project_materials.csv \
  --max-materials 6 \
  --project-name "client_project_v1"
```

### Home User Workflow

```bash
# 1. Start with printer's default materials
bananaforge export-materials \
  --brand "Bambu Lab" \
  --output my_filaments.csv

# 2. Edit CSV to mark what you actually have
# Set available=false for colors you don't own

# 3. Quick color check before buying new filaments
bananaforge analyze-colors vacation_photo.jpg \
  --materials my_filaments.csv \
  --max-materials 6

# 4. If analysis shows poor matches, consider buying additional colors
# Add new materials to my_filaments.csv

# 5. Convert when ready
bananaforge convert vacation_photo.jpg \
  --materials my_filaments.csv \
  --max-materials 6
```

### Batch Processing Workflow

```bash
#!/bin/bash
# Process multiple images with consistent materials

MATERIALS="production_materials.csv"
OUTPUT_BASE="batch_output"

for image in photos/*.jpg; do
    basename=$(basename "$image" .jpg)
    
    # Analyze first
    bananaforge analyze-colors "$image" \
      --materials "$MATERIALS" \
      --output "${OUTPUT_BASE}/${basename}_analysis.json"
    
    # Convert if analysis looks good
    # (could add automated quality checks here)
    bananaforge convert "$image" \
      --materials "$MATERIALS" \
      --max-materials 6 \
      --project-name "$basename" \
      --output "${OUTPUT_BASE}/${basename}/"
done
```

## Troubleshooting Material Issues

### Poor Color Matches

**Problem:** Materials don't match image colors well

**Solutions:**
```bash
# 1. Try different matching method
bananaforge analyze-colors image.jpg --method lab

# 2. Increase material count
bananaforge analyze-colors image.jpg --max-materials 10

# 3. Expand material database
# Add more diverse colors to your CSV

# 4. Check color diversity
bananaforge export-materials --color-diversity --max-materials 12
```

### Limited Material Selection

**Problem:** Not enough material variety

**Solutions:**
1. **Add more brands** to your database
2. **Include specialty materials** (silk, metallic, etc.)
3. **Create custom mixed colors**
4. **Use color-diverse selection**

### Material Availability Issues

**Problem:** Suggested materials not in stock

**Solutions:**
```bash
# 1. Update availability in CSV
# Set available=false for out-of-stock materials

# 2. Create subset of available materials
# Remove unavailable materials from CSV

# 3. Find substitutes
bananaforge analyze-colors image.jpg \
  --materials available_only.csv \
  --max-materials 8
```

### Cost Optimization

**Problem:** Print costs too high

**Solutions:**
```bash
# 1. Reduce material count
bananaforge convert image.jpg --max-materials 4

# 2. Use budget materials
# Edit CSV to include lower-cost alternatives

# 3. Optimize model size
bananaforge convert image.jpg --physical-size 80

# 4. Increase layer height
bananaforge convert image.jpg --layer-height 0.3
```

## Best Practices

### Database Management

1. **Version control** your material databases
2. **Regular updates** - mark materials as unavailable when out of stock  
3. **Backup successful** material combinations
4. **Document custom** mixes and settings
5. **Test new materials** with simple conversions first

### Color Matching

1. **Start with perceptual** matching for most images
2. **Use LAB matching** for color-critical work
3. **Analyze before converting** to preview material selection
4. **Consider image type** when choosing matching method
5. **Build diverse material** sets for better coverage

### Organization

1. **Separate databases** by use case (portraits, landscapes, etc.)
2. **Brand-specific files** for different printers
3. **Tag materials** consistently for easy filtering
4. **Document costs** and suppliers for reordering
5. **Share successful** configurations with team/community

---

## Related Documentation

- [CLI Reference](cli-reference.md) - Material-related commands
- [Configuration](configuration.md) - Material configuration options
- [Output Formats](output-formats.md) - Material information in exports
- [Quick Start](quickstart.md) - Basic material workflows