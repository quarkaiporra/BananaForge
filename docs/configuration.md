# Configuration Guide

Learn how to configure BananaForge for your specific needs and workflows.

## Configuration Methods

BananaForge supports multiple ways to configure settings:

1. **Configuration Files** - JSON/YAML files with persistent settings
2. **Command Line Arguments** - Override settings per command  
3. **Environment Variables** - System-level defaults
4. **Configuration Profiles** - Predefined setting combinations

## Configuration Files

### Creating a Configuration File

```bash
# Create default configuration
bananaforge init-config --output my_config.json

# Edit the file with your preferred settings
# Then use it:
bananaforge convert image.jpg --config my_config.json
```

### Configuration File Structure

```json
{
  "optimization": {
    "iterations": 1000,
    "learning_rate": 0.01,
    "initial_temperature": 1.0,
    "final_temperature": 0.1,
    "temperature_decay": "linear",
    "early_stopping_patience": 100,
    "device": "cpu"
  },
  "model": {
    "layer_height": 0.2,
    "base_height": 0.4,
    "max_layers": 50,
    "physical_size": 100.0,
    "resolution": 256
  },
  "materials": {
    "max_materials": 8,
    "color_matching_method": "perceptual",
    "default_database": "bambu_pla"
  },
  "export": {
    "default_formats": ["stl", "instructions", "cost_report"],
    "project_name": "bananaforge_model",
    "generate_preview": false
  },
  "loss_weights": {
    "perceptual": 1.0,
    "color": 1.0,
    "smoothness": 0.1,
    "consistency": 0.5
  },
  "output": {
    "directory": "./output",
    "compress_files": false,
    "keep_intermediate": false
  },
  "advanced": {
    "mesh_optimization": true,
    "support_generation": false,
    "hollowing": false,
    "infill_percentage": 15.0
  }
}
```

### YAML Configuration

You can also use YAML format:

```yaml
# config.yaml
optimization:
  iterations: 1500
  learning_rate: 0.015
  device: cuda

model:
  layer_height: 0.15
  max_layers: 60
  physical_size: 120.0
  resolution: 384

materials:
  max_materials: 6
  color_matching_method: perceptual

export:
  default_formats:
    - stl
    - instructions  
    - bambu
    - cost_report
  generate_preview: true
```

## Configuration Sections

### Optimization Settings

Controls the AI optimization process:

```json
{
  "optimization": {
    "iterations": 1000,          // Number of optimization steps
    "learning_rate": 0.01,       // How fast the AI learns (0.005-0.02)
    "initial_temperature": 1.0,   // Starting temperature for Gumbel softmax
    "final_temperature": 0.1,     // Final temperature (lower = more discrete)
    "temperature_decay": "linear", // How temperature changes ("linear", "exponential", "cosine")
    "early_stopping_patience": 100, // Stop if no improvement for N steps
    "device": "cpu"              // "cpu", "cuda", or "mps"
  }
}
```

**Tuning Tips:**
- **Higher iterations** = better quality, slower processing
- **Lower learning rate** = more stable, slower convergence  
- **CUDA/MPS device** = much faster on compatible hardware

### Model Parameters

Physical properties of the 3D model:

```json
{
  "model": {
    "layer_height": 0.2,         // Layer thickness in mm
    "base_height": 0.4,          // Base plate thickness in mm
    "max_layers": 50,            // Maximum number of layers
    "physical_size": 100.0,      // Size of longest dimension in mm
    "resolution": 256            // Processing resolution in pixels
  }
}
```

**Guidelines:**
- **Layer height**: 0.1-0.3mm (thinner = more detail, more swaps)
- **Physical size**: 50-300mm typical range
- **Resolution**: 128-512 pixels (higher = more detail, slower)

### Material Settings

Controls material selection and matching:

```json
{
  "materials": {
    "max_materials": 8,                    // Maximum materials to use
    "color_matching_method": "perceptual", // "perceptual", "euclidean", "lab"
    "default_database": "bambu_pla"        // Default material set
  }
}
```

**Methods:**
- **perceptual**: Best visual results (recommended)
- **euclidean**: Simple RGB distance matching
- **lab**: Perceptual color space matching

### Export Options

What files to generate:

```json
{
  "export": {
    "default_formats": [
      "stl",           // 3D model file
      "instructions",  // Material swap instructions  
      "cost_report"    // Cost and usage analysis
    ],
    "project_name": "bananaforge_model",
    "generate_preview": false
  }
}
```

**Available Formats:**
- `stl` - Standard 3D model file
- `instructions` - Material change instructions (TXT/CSV)
- `hueforge` - HueForge project file (.hfp)
- `prusa` - PrusaSlicer project file
- `bambu` - Bambu Studio project file  
- `cost_report` - Material usage and cost analysis
- `preview` - 3D preview with colors

### Loss Function Weights

Fine-tune the optimization objectives:

```json
{
  "loss_weights": {
    "perceptual": 1.0,    // Visual similarity (VGG features)
    "color": 1.0,         // Color accuracy (LAB space)
    "smoothness": 0.1,    // Height map smoothness
    "consistency": 0.5    // Material region consistency
  }
}
```

**Tuning Guidelines:**
- **Increase perceptual** for better visual match
- **Increase color** for accurate colors
- **Increase smoothness** for easier printing
- **Increase consistency** for cleaner material regions

## Predefined Profiles

BananaForge includes several built-in profiles:

### Available Profiles

#### Prototype Profile
Fast settings for quick testing:
```json
{
  "optimization": {
    "iterations": 200,
    "learning_rate": 0.05,
    "early_stopping_patience": 20
  },
  "model": {
    "resolution": 64,
    "max_layers": 15
  },
  "materials": {
    "max_materials": 4
  }
}
```

#### Balanced Profile (Default)
Good balance of quality and speed:
```json
{
  "optimization": {
    "iterations": 1000,
    "learning_rate": 0.01
  },
  "model": {
    "resolution": 256,
    "max_layers": 50
  }
}
```

#### Quality Profile
Best results, slower processing:
```json
{
  "optimization": {
    "iterations": 2000,
    "learning_rate": 0.005,
    "early_stopping_patience": 200
  },
  "model": {
    "resolution": 512,
    "max_layers": 75
  },
  "loss_weights": {
    "perceptual": 1.5,
    "color": 1.2,
    "smoothness": 0.2,
    "consistency": 0.8
  }
}
```

#### Fast Profile
Quick results with lower quality:
```json
{
  "optimization": {
    "iterations": 500,
    "learning_rate": 0.02,
    "early_stopping_patience": 50
  },
  "model": {
    "resolution": 128,
    "max_layers": 25
  }
}
```

### Using Profiles

Create a config file with your chosen profile:

```json
{
  "_profile": "quality",
  "model": {
    "physical_size": 150.0  // Override profile settings as needed
  }
}
```

## Environment Variables

Set system-wide defaults with environment variables:

### Available Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `BANANAFORGE_DEVICE` | Default computation device | `cuda` |
| `BANANAFORGE_ITERATIONS` | Default iterations | `1500` |
| `BANANAFORGE_LEARNING_RATE` | Default learning rate | `0.015` |
| `BANANAFORGE_MAX_MATERIALS` | Default max materials | `6` |
| `BANANAFORGE_PHYSICAL_SIZE` | Default physical size | `120` |
| `BANANAFORGE_LAYER_HEIGHT` | Default layer height | `0.15` |
| `BANANAFORGE_OUTPUT_DIR` | Default output directory | `./my_outputs` |

### Setting Environment Variables

**Linux/macOS:**
```bash
export BANANAFORGE_DEVICE=cuda
export BANANAFORGE_ITERATIONS=1500
export BANANAFORGE_MAX_MATERIALS=6
```

**Windows:**
```cmd
set BANANAFORGE_DEVICE=cuda
set BANANAFORGE_ITERATIONS=1500
```

**Permanent Settings:**
Add to your shell profile (`.bashrc`, `.zshrc`, etc.):
```bash
# BananaForge defaults
export BANANAFORGE_DEVICE=cuda
export BANANAFORGE_ITERATIONS=1500
export BANANAFORGE_OUTPUT_DIR=~/3d_models/bananaforge
```

## Configuration Priority

Settings are applied in this order (later overrides earlier):

1. **Built-in defaults**
2. **Environment variables**  
3. **Configuration file**
4. **Command line arguments**

Example:
```bash
# Environment variable
export BANANAFORGE_ITERATIONS=1000

# Config file sets iterations to 1500
# Command line overrides to 2000
bananaforge convert image.jpg --config my_config.json --iterations 2000
# Final value: 2000 iterations
```

## Use Case Configurations

### Home User - Bambu Lab Printer

```json
{
  "optimization": {
    "iterations": 1200,
    "device": "cpu"
  },
  "model": {
    "layer_height": 0.2,
    "physical_size": 100.0,
    "resolution": 256
  },
  "materials": {
    "max_materials": 6,
    "default_database": "bambu_pla"
  },
  "export": {
    "default_formats": ["stl", "instructions", "bambu", "cost_report"]
  }
}
```

### Professional Studio - High Quality

```json
{
  "optimization": {
    "iterations": 2500,
    "learning_rate": 0.008,
    "device": "cuda"
  },
  "model": {
    "layer_height": 0.15,
    "physical_size": 200.0,
    "resolution": 512
  },
  "materials": {
    "max_materials": 10
  },
  "export": {
    "default_formats": ["stl", "instructions", "hueforge", "prusa", "bambu", "cost_report", "preview"],
    "generate_preview": true
  },
  "loss_weights": {
    "perceptual": 1.2,
    "color": 1.1,
    "smoothness": 0.15,
    "consistency": 0.6
  }
}
```

### Rapid Prototyping - Speed Focus

```json
{
  "optimization": {
    "iterations": 400,
    "learning_rate": 0.025,
    "device": "cuda",
    "early_stopping_patience": 30
  },
  "model": {
    "layer_height": 0.3,
    "physical_size": 80.0,
    "resolution": 128
  },
  "materials": {
    "max_materials": 4
  },
  "export": {
    "default_formats": ["stl", "instructions"]
  }
}
```

### Educational/Learning

```json
{
  "optimization": {
    "iterations": 800,
    "device": "cpu"
  },
  "model": {
    "layer_height": 0.25,
    "physical_size": 60.0,
    "resolution": 128
  },
  "materials": {
    "max_materials": 5
  },
  "export": {
    "default_formats": ["stl", "instructions", "cost_report"],
    "generate_preview": true
  },
  "output": {
    "keep_intermediate": true
  }
}
```

## Advanced Configuration

### Custom Loss Weights for Specific Use Cases

**Portrait/Face Images:**
```json
{
  "loss_weights": {
    "perceptual": 1.5,  // High visual accuracy
    "color": 0.8,       // Slightly lower color priority
    "smoothness": 0.2,  // More smoothness for skin
    "consistency": 0.7  // Good region consistency
  }
}
```

**Landscape Images:**
```json
{
  "loss_weights": {
    "perceptual": 1.0,
    "color": 1.3,       // Higher color accuracy for nature
    "smoothness": 0.05, // Less smoothness for detail
    "consistency": 0.4  // Allow more color variation
  }
}
```

**Logos/Graphics:**
```json
{
  "loss_weights": {
    "perceptual": 0.8,
    "color": 1.5,       // Exact color matching
    "smoothness": 0.3,  // Some smoothness for clean edges
    "consistency": 1.0  // High consistency for clean regions
  }
}
```

### Device-Specific Optimization

**High-End NVIDIA GPU:**
```json
{
  "optimization": {
    "iterations": 3000,
    "device": "cuda"
  },
  "model": {
    "resolution": 768
  }
}
```

**Apple Silicon Mac:**
```json
{
  "optimization": {
    "iterations": 2000,
    "device": "mps"
  },
  "model": {
    "resolution": 512
  }
}
```

**CPU-Only System:**
```json
{
  "optimization": {
    "iterations": 800,
    "device": "cpu",
    "learning_rate": 0.02
  },
  "model": {
    "resolution": 256
  }
}
```

## Validation and Debugging

### Configuration Validation

BananaForge automatically validates your configuration:

```bash
# This will show validation errors if any
bananaforge convert image.jpg --config invalid_config.json
```

Common validation errors:
- Negative values for iterations, learning rate, etc.
- Invalid device names
- Impossible physical constraints

### Debug Configuration

Enable verbose logging to see which settings are being used:

```bash
bananaforge --verbose convert image.jpg --config my_config.json
```

This shows:
- Which config file was loaded
- Final configuration values after all overrides
- Environment variables that were applied

## Best Practices

### Configuration Management

1. **Start with profiles** - Use built-in profiles as starting points
2. **Version control configs** - Save successful configurations
3. **Document changes** - Note why you changed specific settings
4. **Test incrementally** - Change one setting at a time

### Performance Tuning

1. **Start small** - Test with low iterations and resolution first
2. **Profile your hardware** - Find optimal settings for your system
3. **Monitor resources** - Watch CPU/GPU/memory usage
4. **Scale gradually** - Increase quality settings step by step

### Organization

```
my_configs/
├── prototype.json      # Fast testing
├── production.json     # Final quality
├── portraits.json      # Portrait-specific settings
├── landscapes.json     # Landscape-specific settings
└── batch_processing.json # Automated workflows
```

Use descriptive config files for different scenarios:

```bash
bananaforge convert portrait.jpg --config configs/portraits.json
bananaforge convert landscape.jpg --config configs/landscapes.json
```

---

## Related Documentation

- [CLI Reference](cli-reference.md) - Command line options
- [Material Management](materials.md) - Material database setup
- [Output Formats](output-formats.md) - Export configuration
- [API Reference](api-reference.md) - Python configuration API