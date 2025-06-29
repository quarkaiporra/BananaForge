# Troubleshooting Guide

Common issues and solutions for ForgeBambu.

## Installation Issues

### "Command not found: forgebambu"

**Problem**: Terminal doesn't recognize `forgebambu` command

**Solutions**:
```bash
# Check if installed correctly
pip show forgebambu

# Check if pip install directory is in PATH
python -m pip show forgebambu

# Try running directly
python -m forgebambu --help

# Reinstall if needed
pip uninstall forgebambu
pip install forgebambu
```

**For virtual environments**:
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Then install
pip install forgebambu
```

### "No module named 'forgebambu'"

**Problem**: Python can't find the ForgeBambu module

**Solutions**:
```bash
# Check Python version
python --version  # Should be 3.9+

# Check if installed in correct environment
pip list | grep forgebambu

# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall with verbose output
pip install -v forgebambu
```

### CUDA/GPU Installation Issues

**Problem**: GPU not being used or CUDA errors

**Solutions**:
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check CUDA version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For Apple Silicon (M1/M2)
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Dependencies Issues

**Problem**: Missing or conflicting dependencies

**Solutions**:
```bash
# Update pip first
pip install --upgrade pip

# Install with all dependencies
pip install forgebambu --upgrade

# For development dependencies
pip install forgebambu[dev]

# Clean install
pip uninstall forgebambu
pip cache purge
pip install forgebambu
```

---

## Runtime Errors

### Out of Memory Errors

**Problem**: "CUDA out of memory" or "RuntimeError: out of memory"

**Solutions**:

1. **Reduce processing resolution**:
```bash
forgebambu convert image.jpg --resolution 128
```

2. **Use CPU instead of GPU**:
```bash
forgebambu convert image.jpg --device cpu
```

3. **Reduce model complexity**:
```bash
forgebambu convert image.jpg \
  --max-materials 4 \
  --max-layers 25 \
  --iterations 500
```

4. **Clear GPU memory**:
```python
import torch
torch.cuda.empty_cache()
```

5. **Monitor memory usage**:
```bash
# NVIDIA GPUs
watch nvidia-smi

# System memory
top    # Linux/Mac
# or
htop   # If installed
```

### "File not found" Errors

**Problem**: Can't find input image or material files

**Solutions**:
```bash
# Check file exists
ls -la image.jpg

# Use absolute path
forgebambu convert /full/path/to/image.jpg

# Check file permissions
chmod 644 image.jpg

# For material files
forgebambu convert image.jpg --materials /full/path/to/materials.csv
```

### "Invalid image format" Errors

**Problem**: Image format not supported

**Solutions**:
```bash
# Convert image format first
convert input.webp input.jpg  # Using ImageMagick
# or
python -c "
from PIL import Image
img = Image.open('input.webp')
img.convert('RGB').save('input.jpg')
"

# Supported formats: JPG, PNG, BMP, TIFF
forgebambu convert input.jpg
```

---

## Performance Issues

### Slow Processing

**Problem**: Optimization takes too long

**Solutions**:

1. **Use GPU acceleration**:
```bash
forgebambu convert image.jpg --device cuda  # or mps
```

2. **Reduce quality settings**:
```bash
forgebambu convert image.jpg \
  --iterations 500 \
  --resolution 128 \
  --max-materials 4
```

3. **Use fast profile**:
```bash
forgebambu init-config --output fast_config.json
# Edit config to use "fast" profile
forgebambu convert image.jpg --config fast_config.json
```

4. **Enable early stopping**:
```bash
forgebambu convert image.jpg \
  --iterations 2000 \
  # Early stopping is enabled by default (patience=100)
```

### High Memory Usage

**Problem**: System running out of RAM

**Solutions**:

1. **Reduce image resolution**:
```bash
forgebambu convert image.jpg --resolution 128
```

2. **Process smaller batches**:
```bash
# Instead of processing many images at once
for img in *.jpg; do
    forgebambu convert "$img"
done
```

3. **Monitor memory usage**:
```python
import psutil
import torch

# System memory
memory = psutil.virtual_memory()
print(f"Memory usage: {memory.percent}%")

# GPU memory
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
```

### Poor Convergence

**Problem**: Optimization not improving

**Solutions**:

1. **Increase iterations**:
```bash
forgebambu convert image.jpg --iterations 2000
```

2. **Adjust learning rate**:
```bash
forgebambu convert image.jpg --learning-rate 0.005  # Lower for stability
```

3. **Check material selection**:
```bash
# Analyze colors first
forgebambu analyze-colors image.jpg --max-materials 8
# Ensure good material matches
```

4. **Use different initialization**:
```python
# In Python API
from forgebambu.core.optimizer import HeightMapInitializer
initializer = HeightMapInitializer()
init_height = initializer.depth_based_init(image, max_layers)
```

---

## Quality Issues

### Poor Color Matching

**Problem**: Output colors don't match original image

**Solutions**:

1. **Try different matching methods**:
```bash
forgebambu analyze-colors image.jpg --method lab
forgebambu analyze-colors image.jpg --method perceptual
```

2. **Increase material count**:
```bash
forgebambu convert image.jpg --max-materials 10
```

3. **Check material database**:
```bash
# Ensure diverse color coverage
forgebambu export-materials --color-diversity --max-materials 12
```

4. **Enhance image preprocessing**:
```python
from forgebambu.image.processor import ImageProcessor
processor = ImageProcessor()
enhanced = processor.enhance_contrast(image, factor=1.2)
```

### Rough/Jagged Output

**Problem**: Height map or STL has rough surfaces

**Solutions**:

1. **Increase smoothness weight**:
```python
# In config file
{
  "loss_weights": {
    "smoothness": 0.3  # Increase from default 0.1
  }
}
```

2. **Use mesh smoothing**:
```python
from forgebambu.output.stl_generator import STLGenerator
generator = STLGenerator()
mesh = generator.generate_stl(
    height_map, "output.stl", 
    smooth_mesh=True
)
```

3. **Reduce layer height**:
```bash
forgebambu convert image.jpg --layer-height 0.15  # Instead of 0.2
```

### Too Many Material Swaps

**Problem**: Generated model requires too many material changes

**Solutions**:

1. **Reduce material count**:
```bash
forgebambu convert image.jpg --max-materials 4
```

2. **Increase consistency weight**:
```python
{
  "loss_weights": {
    "consistency": 1.0  # Increase from default 0.5
  }
}
```

3. **Optimize swap sequence**:
```python
from forgebambu.output.instructions import SwapInstructionGenerator
generator = SwapInstructionGenerator()
optimized_instructions = generator.optimize_swap_sequence(
    instructions, material_db, minimize_swaps=True
)
```

---

## File Format Issues

### STL File Problems

**Problem**: Generated STL has issues

**Solutions**:

1. **Validate STL file**:
```bash
forgebambu validate-stl model.stl
```

2. **Repair mesh issues**:
```python
from forgebambu.output.mesh import MeshProcessor
processor = MeshProcessor()
repaired_mesh = processor.repair_mesh(mesh)
```

3. **Check mesh quality**:
```python
quality = processor.analyze_mesh_quality(mesh)
print(f"Issues: {quality['issues']}")
```

### Export Format Issues

**Problem**: Can't open project files in slicer

**Solutions**:

1. **Check slicer compatibility**:
```bash
# Generate slicer-specific files
forgebambu convert image.jpg --export-format bambu  # For Bambu Studio
forgebambu convert image.jpg --export-format prusa  # For PrusaSlicer
```

2. **Verify file format**:
```bash
# Check file was created
ls -la output/
file output/project.hfp  # Check file type
```

---

## Material Database Issues

### Material File Loading Errors

**Problem**: Can't load CSV or JSON material files

**Solutions**:

1. **Check file format**:
```bash
head -5 materials.csv  # Check CSV structure
python -c "import json; print(json.load(open('materials.json')))"  # Validate JSON
```

2. **Fix CSV formatting**:
```csv
# Ensure required columns exist
id,name,brand,color_hex
mat1,Red PLA,Brand,#FF0000
```

3. **Validate material data**:
```python
from forgebambu.materials.database import Material
try:
    material = Material(
        id="test", name="Test", brand="Test",
        color_rgb=(1.0, 0.0, 0.0), color_hex="#FF0000"
    )
    print("Material valid")
except Exception as e:
    print(f"Material invalid: {e}")
```

### Color Matching Failures

**Problem**: No suitable materials found

**Solutions**:

1. **Expand material database**:
```bash
# Add more diverse colors
forgebambu export-materials --color-diversity --max-materials 20
```

2. **Check color range**:
```python
from forgebambu.materials.matcher import ColorMatcher
matcher = ColorMatcher(material_db)
# Try wider color matching tolerance
materials = db.get_materials_by_color_range(
    target_color, max_distance=0.5  # Increase from 0.3
)
```

---

## Configuration Issues

### Config File Problems

**Problem**: Configuration file not working

**Solutions**:

1. **Validate JSON syntax**:
```bash
python -c "import json; json.load(open('config.json'))"
```

2. **Use default config as template**:
```bash
forgebambu init-config --output template.json
# Copy and modify template.json
```

3. **Check config priority**:
```bash
# Command line overrides config file
forgebambu convert image.jpg --config my_config.json --iterations 2000
```

### Environment Variable Issues

**Problem**: Environment variables not working

**Solutions**:

1. **Check variable names**:
```bash
# Correct variable names
export FORGEBAMBU_DEVICE=cuda
export FORGEBAMBU_ITERATIONS=1500

# Check if set
echo $FORGEBAMBU_DEVICE
```

2. **Verify variable loading**:
```bash
forgebambu --verbose convert image.jpg  # Shows which settings are used
```

---

## Platform-Specific Issues

### Windows Issues

**Problem**: Windows-specific errors

**Solutions**:

1. **Path separators**:
```cmd
REM Use forward slashes or double backslashes
forgebambu convert image.jpg --output "./output"
forgebambu convert image.jpg --output ".\\output"
```

2. **Long path names**:
```cmd
REM Enable long paths in Windows or use shorter paths
forgebambu convert image.jpg --project-name "short_name"
```

3. **Antivirus interference**:
```cmd
REM Add Python and pip to antivirus exceptions
REM Or run in Windows Defender excluded folder
```

### macOS Issues

**Problem**: macOS-specific errors

**Solutions**:

1. **Permission issues**:
```bash
# Fix pip permissions
pip install --user forgebambu

# Or use homebrew Python
brew install python
pip3 install forgebambu
```

2. **Apple Silicon GPU**:
```bash
# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"

# Use MPS device
forgebambu convert image.jpg --device mps
```

### Linux Issues

**Problem**: Linux-specific errors

**Solutions**:

1. **System dependencies**:
```bash
# Ubuntu/Debian
sudo apt install python3-dev libgl1-mesa-glx libglib2.0-0

# CentOS/RHEL
sudo yum install python3-devel mesa-libGL

# Arch Linux
sudo pacman -S python python-pip mesa
```

2. **Display issues** (for headless servers):
```bash
# Install virtual display
sudo apt install xvfb

# Run with virtual display
xvfb-run -a forgebambu convert image.jpg
```

---

## Getting Help

### Debug Information

When reporting issues, include:

```bash
# System information
forgebambu version
python --version
pip show forgebambu

# Verbose output
forgebambu --verbose convert image.jpg

# System details
python -c "
import platform
import torch
print(f'Platform: {platform.platform()}')
print(f'Python: {platform.python_version()}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.backends.mps.is_available():
    print('MPS: Available')
"
```

### Common Error Messages

| Error Message | Likely Cause | Solution |
|---------------|--------------|----------|
| "CUDA out of memory" | GPU memory exhausted | Use `--device cpu` or reduce `--resolution` |
| "No module named" | Installation issue | Reinstall with `pip install forgebambu` |
| "File not found" | Wrong file path | Use absolute path or check file exists |
| "Invalid image format" | Unsupported format | Convert to JPG or PNG |
| "Configuration error" | Invalid config values | Use `forgebambu init-config` for template |
| "No suitable materials" | Material database issue | Check material CSV/JSON format |

### Resources

- 📖 **Documentation**: [Full Documentation](README.md)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/forgebambu/forgebambu/issues)
- 💬 **Community**: [GitHub Discussions](https://github.com/forgebambu/forgebambu/discussions)
- 📧 **Support**: support@forgebambu.com

### Performance Monitoring

Monitor ForgeBambu performance:

```python
import time
import psutil
import torch

def monitor_performance():
    # Memory usage
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.percent}%")
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU: {gpu_memory:.1f}GB")
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU: {cpu_percent}%")

# Use during optimization
monitor_performance()
```

---

Still having issues? Don't hesitate to reach out to the community or file a detailed bug report!