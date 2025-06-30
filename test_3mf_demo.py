#!/usr/bin/env python3
"""Demo script to test 3MF export functionality end-to-end."""

import torch
import numpy as np
from pathlib import Path

from src.bananaforge.materials.database import DefaultMaterials
from src.bananaforge.output.threemf_exporter import ThreeMFExporter, ThreeMFExportConfig


def create_test_optimization_results():
    """Create mock optimization results for testing."""
    # Create a simple heightmap (10x10 grid with varying heights)
    heightmap = torch.zeros(10, 10)
    for i in range(10):
        for j in range(10):
            heightmap[i, j] = (i + j) * 0.1  # Simple gradient
    
    # Create per-layer material assignments
    layer_materials = {
        0: {'material_id': 'bambu_pla_basic_red', 'transparency': 1.0, 'layer_height': 0.2},
        1: {'material_id': 'bambu_pla_basic_red', 'transparency': 1.0, 'layer_height': 0.2},
        2: {'material_id': 'bambu_pla_basic_white', 'transparency': 1.0, 'layer_height': 0.2},
        3: {'material_id': 'bambu_pla_basic_white', 'transparency': 0.67, 'layer_height': 0.2},  # Transparency layer
        4: {'material_id': 'bambu_pla_basic_blue', 'transparency': 1.0, 'layer_height': 0.2},
        5: {'material_id': 'bambu_pla_basic_blue', 'transparency': 1.0, 'layer_height': 0.2},
    }
    
    return {
        'heightmap': heightmap,
        'layer_materials': layer_materials,
        'optimization_metadata': {
            'iterations': 1000,
            'final_loss': 0.0234,
            'convergence_time': 123.45
        }
    }


def main():
    """Test 3MF export functionality."""
    print("🧪 Testing BananaForge 3MF Export")
    print("=" * 50)
    
    # Create material database
    material_db = DefaultMaterials.create_bambu_basic_pla()
    print(f"✅ Material database created with {len(material_db.materials)} materials")
    
    # Create 3MF exporter
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exporter = ThreeMFExporter(device=device, material_db=material_db)
    print(f"✅ 3MF exporter initialized (device: {device})")
    
    # Create test optimization results
    optimization_results = create_test_optimization_results()
    print(f"✅ Test optimization results created")
    print(f"   - Heightmap shape: {optimization_results['heightmap'].shape}")
    print(f"   - Layer materials: {len(optimization_results['layer_materials'])} layers")
    
    # Configure export
    config = ThreeMFExportConfig(
        bambu_compatible=True,
        include_metadata=True,
        validate_output=True
    )
    
    # Export to 3MF
    output_path = Path("test_output.3mf")
    print(f"\n📦 Exporting to 3MF: {output_path}")
    
    try:
        result = exporter.export(optimization_results, output_path, config)
        
        if result['success']:
            print("✅ 3MF export successful!")
            print(f"   - Output file: {result['output_file']}")
            print(f"   - File size: {result['file_size']:,} bytes")
            print(f"   - Materials count: {result['materials_count']}")
            
            if result.get('validation'):
                validation = result['validation']
                if validation['valid']:
                    print("✅ 3MF validation passed")
                else:
                    print("⚠️  3MF validation warnings:")
                    for warning in validation['warnings']:
                        print(f"      - {warning}")
        else:
            print(f"❌ 3MF export failed: {result['error']}")
            
    except Exception as e:
        print(f"❌ Export error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 3MF export test completed")


if __name__ == "__main__":
    main()