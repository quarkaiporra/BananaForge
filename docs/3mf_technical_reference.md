# 3MF Technical Reference

## Implementation Overview

BananaForge's 3MF export system is implemented across several key modules:

### Core Classes

#### `ThreeMFExporter`
**Location**: `src/bananaforge/output/threemf_exporter.py`

Main export coordinator that orchestrates the 3MF generation process.

```python
class ThreeMFExporter:
    def __init__(self, device: str = "cpu", material_db: Optional[MaterialDatabase] = None)
    def export(self, optimization_results: Dict[str, Any], 
               output_path: Union[str, Path],
               config: Optional[ThreeMFExportConfig] = None) -> Dict[str, Any]
```

**Key Methods**:
- `create_3mf_container()`: Generates ZIP archive with all 3MF components
- `_extract_geometry_data()`: Converts heightmaps to mesh geometry
- `_extract_material_data()`: Processes material assignments and properties
- `_heightmap_to_mesh()`: Converts 2D heightmaps to 3D triangle meshes

#### `ModelXMLGenerator`
Generates the core 3D model XML with geometry and material assignments.

```python
def generate(self, vertices: List[Tuple[float, float, float]], 
             triangles: List[Tuple[int, int, int]],
             materials: Optional[Dict[str, Any]] = None,
             layer_materials: Optional[Dict[int, LayerMaterial]] = None) -> str
```

#### `ThreeMFNamespaceManager`
Manages XML namespaces for 3MF specification compliance.

**Supported Namespaces**:
- `3mf`: Core 3MF specification
- `material`: Material properties extension
- `slice`: Layer/slice information extension
- `opc_rel`: Package relationships
- `opc_ct`: Content types

### Data Structures

#### `LayerMaterial`
```python
@dataclass
class LayerMaterial:
    layer_index: int
    material_id: str
    transparency: float = 1.0
    layer_height: float = 0.2
```

#### `ThreeMFExportConfig`
```python
@dataclass
class ThreeMFExportConfig:
    bambu_compatible: bool = False
    include_metadata: bool = True
    include_thumbnail: bool = False
    compress_xml: bool = True
    validate_output: bool = True
```

### File Structure Generation

#### Content Types ([Content_Types].xml)
```xml
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
  <Override PartName="/3D/3dmodel.model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
```

#### Package Relationships (_rels/.rels)
```xml
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
                Target="3D/3dmodel.model" Id="rel-1"/>
</Relationships>
```

#### 3D Model (3D/3dmodel.model)
```xml
<model unit="millimeter" xml:lang="en-US" 
       xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
       xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">
  <resources>
    <basematerials>
      <base name="Material Name" displaycolor="#RRGGBB"/>
    </basematerials>
    <object id="1" type="model">
      <mesh>
        <vertices>
          <vertex x="0.0" y="0.0" z="0.0"/>
        </vertices>
        <triangles>
          <triangle v1="0" v2="1" v3="2" pid="materialId"/>
        </triangles>
      </mesh>
    </object>
  </resources>
  <build>
    <item objectid="1"/>
  </build>
</model>
```

## Material Integration

### Material Database Interface

The 3MF exporter integrates with BananaForge's material system:

```python
# Material properties used in 3MF export
material.color_rgb       # RGB values (0-1) -> hex color
material.name           # Display name in 3MF
material.transparency   # Alpha/transparency value
material.temperature    # Printing temperature (metadata)
material.density        # Material density (metadata)
```

### Per-Layer Assignment

Materials are assigned per-layer, not per-face:

```python
layer_materials = {
    0: LayerMaterial(layer_index=0, material_id="base_material", transparency=1.0),
    1: LayerMaterial(layer_index=1, material_id="accent_material", transparency=0.8),
    # ... additional layers
}
```

## Geometry Processing

### Heightmap to Mesh Conversion

The system converts 2D heightmaps to 3D triangle meshes:

```python
def _heightmap_to_mesh(self, heightmap: np.ndarray) -> Tuple[List[Tuple[float, float, float]], 
                                                           List[Tuple[int, int, int]]]:
    # Handle different heightmap dimensions (4D, 3D, 2D)
    # Generate vertices from heightmap grid
    # Create triangles for quad tessellation
    # Return vertices and triangle indices
```

**Supported Heightmap Formats**:
- 4D tensors: `(batch, channels, height, width)` - squeeze to 2D
- 3D tensors: `(channels, height, width)` - use first channel or mean
- 2D arrays: `(height, width)` - use directly

### Mesh Optimization

- **Vertex generation**: Grid-based vertex placement from heightmap
- **Triangle tessellation**: Two triangles per heightmap quad
- **Index optimization**: Sequential vertex indexing for efficiency
- **Normal calculation**: Implicit from triangle winding order

## Validation System

### File Structure Validation

```python
def _validate_3mf_file(self, threemf_data: bytes) -> Dict[str, Any]:
    # Check ZIP archive integrity
    # Verify required files present
    # Validate XML structure
    # Return validation report
```

**Validation Checks**:
- ZIP archive can be opened
- Required files present: `[Content_Types].xml`, `_rels/.rels`, `3D/3dmodel.model`
- XML files are well-formed
- No critical structural errors

### Error Handling

The system provides comprehensive error handling:

```python
try:
    result = exporter.export(optimization_results, output_path, config)
    if result['success']:
        # Export successful
    else:
        # Handle export error: result['error']
except Exception as e:
    # Handle unexpected errors
```

## Performance Considerations

### Memory Management
- Geometry generated on-demand
- ZIP compression reduces file size
- Streaming XML generation for large models

### Optimization Strategies
- Mesh decimation for large heightmaps
- Material deduplication
- Efficient vertex indexing

### Scalability Limits
- Tested up to 512x512 heightmaps
- Supports 100+ layers
- File sizes typically <50MB

## Extension Points

### Custom Material Properties
```python
# Add custom material metadata
material_data['properties'].update({
    'custom_property': value,
    'application_specific': data
})
```

### Bambu Studio Extensions
```python
if config.bambu_compatible:
    # Add Bambu-specific metadata
    # Optimize material assignments
    # Include slicer hints
```

### Future Extensions
- Slice extension support for layer-by-layer instructions
- Production extension for manufacturing workflows
- Custom namespace support for application-specific data

## Testing Strategy

### Unit Tests
- Individual component testing (XML generators, validators)
- Material integration tests
- Geometry conversion tests

### Integration Tests
- End-to-end export pipeline
- File structure validation
- Slicer compatibility verification

### Performance Tests
- Large model handling
- Memory usage profiling
- Export time benchmarking

## API Reference

### CLI Integration
```bash
bananaforge convert input.jpg --export-format 3mf [options]
```

### Python API
```python
from bananaforge.output.threemf_exporter import ThreeMFExporter, ThreeMFExportConfig

exporter = ThreeMFExporter(device="cuda", material_db=material_database)
result = exporter.export(optimization_results, "output.3mf", config)
```

### Configuration Options
- `bambu_compatible`: Enable Bambu Studio optimizations
- `include_metadata`: Add detailed export metadata
- `validate_output`: Perform post-export validation
- `compress_xml`: Enable XML compression in ZIP

---

This technical reference covers the core implementation details of BananaForge's 3MF export system. For usage examples and user guides, see [3MF Export Guide](3mf_export_guide.md).