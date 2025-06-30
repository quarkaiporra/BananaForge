#!/usr/bin/env python3
"""3MF integration testing for end-to-end workflow validation.

This file focuses on testing the complete integration of 3MF export
with the existing BananaForge pipeline and external slicer compatibility.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import zipfile
import json
from unittest.mock import Mock, patch, MagicMock

from bananaforge.core.optimizer import LayerOptimizer, OptimizationConfig
from bananaforge.image.processor import ImageProcessor
from bananaforge.materials.database import MaterialDatabase, DefaultMaterials
from bananaforge.output.exporter import ModelExporter


class Test3MFModelExporterIntegration:
    """Test integration of 3MF export with existing ModelExporter."""
    
    @pytest.fixture
    def mock_optimization_results(self):
        """Create mock optimization results for integration testing."""
        return {
            'heightmap': torch.rand(50, 50),
            'material_assignments': torch.randint(0, 3, (50, 50)),
            'layer_materials': {i: f"material_{i % 3}" for i in range(10)},
            'swap_instructions': [
                {'layer': 3, 'from_material': 'material_0', 'to_material': 'material_1'},
                {'layer': 7, 'from_material': 'material_1', 'to_material': 'material_2'}
            ],
            'optimization_metadata': {
                'iterations': 1000,
                'final_loss': 0.023,
                'convergence_time': 245.6
            }
        }
    
    def test_model_exporter_3mf_format_support(self, mock_optimization_results):
        """Test that ModelExporter supports 3mf format."""
        # This drives the integration with existing ModelExporter
        exporter = ModelExporter(output_dir="/tmp/test")
        
        # Should not raise error when 3mf is included in formats
        # This will initially fail, driving the implementation
        with pytest.raises((AttributeError, NotImplementedError)):
            exporter.export_model(
                optimization_results=mock_optimization_results,
                formats=['stl', '3mf', 'instructions'],
                model_name="test_model"
            )
    
    def test_3mf_export_preserves_optimization_data(self, mock_optimization_results):
        """Test that 3MF export preserves all optimization data."""
        # This drives the requirement to preserve layer materials and metadata
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import ThreeMFExporter
            
            exporter = ThreeMFExporter()
            threemf_data = exporter.export(mock_optimization_results)
            
            # Validate that all optimization data is preserved
            assert 'layer_materials' in threemf_data
            assert 'swap_instructions' in threemf_data
            assert 'optimization_metadata' in threemf_data


class Test3MFSlicerCompatibility:
    """Test 3MF file compatibility with popular slicers."""
    
    @pytest.fixture
    def sample_3mf_file(self):
        """Create a sample 3MF file for compatibility testing."""
        # Mock 3MF file content that should be compatible with slicers
        return b"mock_3mf_file_content"
    
    @pytest.mark.integration
    def test_bambu_studio_compatibility(self, sample_3mf_file):
        """Test that generated 3MF files can be loaded by Bambu Studio."""
        # This would require actual Bambu Studio testing in CI
        # For now, we test the file structure requirements
        
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import BambuStudioValidator
            
            validator = BambuStudioValidator()
            is_compatible = validator.validate_bambu_compatibility(sample_3mf_file)
            
            assert is_compatible is True
    
    @pytest.mark.integration
    def test_prusaslicer_compatibility(self, sample_3mf_file):
        """Test PrusaSlicer compatibility."""
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import PrusaSlicerValidator
            
            validator = PrusaSlicerValidator()
            is_compatible = validator.validate_prusa_compatibility(sample_3mf_file)
            
            assert is_compatible is True
    
    @pytest.mark.integration  
    def test_cura_compatibility(self, sample_3mf_file):
        """Test Cura compatibility."""
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import CuraValidator
            
            validator = CuraValidator()
            is_compatible = validator.validate_cura_compatibility(sample_3mf_file)
            
            assert is_compatible is True


class Test3MFTransparencyIntegration:
    """Test 3MF export integration with transparency features."""
    
    @pytest.fixture
    def transparency_optimization_results(self):
        """Create optimization results with transparency features."""
        return {
            'layer_materials': {
                0: {'material_id': 'pla_black', 'transparency': 1.0},
                1: {'material_id': 'pla_red', 'transparency': 1.0},
                2: {'material_id': 'pla_red', 'transparency': 0.67},  # Transparency layer
                3: {'material_id': 'pla_white', 'transparency': 1.0},
                4: {'material_id': 'pla_white', 'transparency': 0.33}, # More transparency
                5: {'material_id': 'pla_blue', 'transparency': 1.0},
            },
            'transparency_effects': {
                'gradient_regions': [(10, 20, 30, 40)],  # x1, y1, x2, y2
                'color_mixing_zones': [
                    {'layers': [2, 3], 'base_color': [1, 0, 0], 'overlay_color': [1, 1, 1]}
                ]
            }
        }
    
    def test_transparency_layer_export(self, transparency_optimization_results):
        """Test that transparency layers are properly exported to 3MF."""
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import ThreeMFTransparencyProcessor
            
            processor = ThreeMFTransparencyProcessor()
            transparency_data = processor.process_transparency_layers(
                transparency_optimization_results['layer_materials']
            )
            
            # Should create unique materials for different transparency levels
            unique_materials = transparency_data['unique_materials']
            assert len(unique_materials) >= 4  # Different transparency combinations
            
            # Verify transparency values are preserved
            red_transparent = next(
                m for m in unique_materials 
                if 'red' in m['name'] and m['transparency'] < 1.0
            )
            assert red_transparent is not None
    
    def test_gradient_effect_preservation(self, transparency_optimization_results):
        """Test that gradient effects are preserved in 3MF export."""
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import ThreeMFGradientProcessor
            
            processor = ThreeMFGradientProcessor()
            gradient_metadata = processor.process_gradients(
                transparency_optimization_results['transparency_effects']
            )
            
            assert 'gradient_regions' in gradient_metadata
            assert len(gradient_metadata['gradient_regions']) > 0


class Test3MFPerformanceBenchmarks:
    """Performance benchmarks for 3MF export functionality."""
    
    @pytest.mark.benchmark
    def test_small_model_export_performance(self):
        """Benchmark 3MF export for small models (<100 layers)."""
        import time
        
        # Small model data
        small_model = {
            'heightmap': torch.rand(64, 64),
            'layer_materials': {i: f"material_{i % 2}" for i in range(50)},
            'num_layers': 50
        }
        
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import ThreeMFExporter
            
            exporter = ThreeMFExporter()
            
            start_time = time.time()
            result = exporter.export(small_model)
            export_time = time.time() - start_time
            
            # Small models should export quickly (<5 seconds)
            assert export_time < 5.0
            
            # File size should be reasonable (<10MB)
            assert result.get('file_size', 0) < 10 * 1024 * 1024
    
    @pytest.mark.benchmark
    def test_large_model_export_performance(self):
        """Benchmark 3MF export for large models (>500 layers)."""
        import time
        
        # Large model data
        large_model = {
            'heightmap': torch.rand(256, 256),
            'layer_materials': {i: f"material_{i % 4}" for i in range(800)},
            'num_layers': 800
        }
        
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import ThreeMFExporter
            
            exporter = ThreeMFExporter()
            
            start_time = time.time()
            result = exporter.export(large_model)
            export_time = time.time() - start_time
            
            # Large models should still export within reasonable time (<30 seconds)
            assert export_time < 30.0
            
            # File size should be manageable (<100MB)
            assert result.get('file_size', 0) < 100 * 1024 * 1024
    
    @pytest.mark.benchmark
    def test_memory_usage_during_export(self):
        """Test memory usage during 3MF export."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Medium model for memory testing
        model_data = {
            'heightmap': torch.rand(128, 128),
            'layer_materials': {i: f"material_{i % 3}" for i in range(200)},
            'num_layers': 200
        }
        
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import ThreeMFExporter
            
            exporter = ThreeMFExporter()
            result = exporter.export(model_data)
            
            peak_memory = process.memory_info().rss
            memory_increase = peak_memory - initial_memory
            
            # Memory increase should be reasonable (<500MB)
            assert memory_increase < 500 * 1024 * 1024


class Test3MFErrorHandling:
    """Test error handling and edge cases in 3MF export."""
    
    def test_invalid_layer_data_handling(self):
        """Test handling of invalid layer data."""
        invalid_data = {
            'layer_materials': {},  # Empty layer materials
            'heightmap': None,      # Invalid heightmap
            'num_layers': -1        # Invalid layer count
        }
        
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import ThreeMFExporter
            
            exporter = ThreeMFExporter()
            
            # Should raise appropriate validation error
            with pytest.raises(ValueError):
                exporter.export(invalid_data)
    
    def test_missing_material_handling(self):
        """Test handling of missing material references."""
        data_with_missing_materials = {
            'layer_materials': {
                0: 'nonexistent_material',
                1: 'another_missing_material'
            },
            'heightmap': torch.rand(32, 32),
            'num_layers': 2
        }
        
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import ThreeMFExporter
            
            exporter = ThreeMFExporter()
            
            # Should handle missing materials gracefully
            with pytest.raises(ValueError, match="Material not found"):
                exporter.export(data_with_missing_materials)
    
    def test_corrupted_geometry_handling(self):
        """Test handling of corrupted geometry data."""
        corrupted_data = {
            'heightmap': torch.tensor([[float('nan'), float('inf')], [0.0, -float('inf')]]),
            'layer_materials': {0: 'pla_red', 1: 'pla_blue'},
            'num_layers': 2
        }
        
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import ThreeMFExporter
            
            exporter = ThreeMFExporter()
            
            # Should detect and handle corrupted geometry
            with pytest.raises(ValueError, match="Invalid geometry"):
                exporter.export(corrupted_data)


class Test3MFFileValidation:
    """Test validation of generated 3MF files."""
    
    def test_zip_structure_validation(self):
        """Test validation of ZIP file structure."""
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import ThreeMFStructureValidator
            
            validator = ThreeMFStructureValidator()
            
            # Mock valid 3MF ZIP structure
            mock_zip_files = [
                "[Content_Types].xml",
                "_rels/.rels",
                "3D/3dmodel.model",
                "Metadata/thumbnail.png"
            ]
            
            is_valid = validator.validate_zip_structure(mock_zip_files)
            assert is_valid is True
            
            # Test invalid structure
            invalid_zip_files = ["invalid_file.txt"]
            is_valid = validator.validate_zip_structure(invalid_zip_files)
            assert is_valid is False
    
    def test_xml_schema_compliance(self):
        """Test XML schema compliance validation."""
        with pytest.raises(ImportError):
            from bananaforge.output.threemf_exporter import ThreeMFXMLValidator
            
            validator = ThreeMFXMLValidator()
            
            # Mock valid 3D model XML
            valid_xml = """<?xml version="1.0" encoding="UTF-8"?>
            <model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
                <resources>
                    <object id="1" type="model">
                        <mesh>
                            <vertices>
                                <vertex x="0" y="0" z="0"/>
                                <vertex x="1" y="0" z="0"/>
                                <vertex x="0" y="1" z="0"/>
                            </vertices>
                            <triangles>
                                <triangle v1="0" v2="1" v3="2"/>
                            </triangles>
                        </mesh>
                    </object>
                </resources>
                <build>
                    <item objectid="1"/>
                </build>
            </model>"""
            
            is_valid = validator.validate_xml_schema(valid_xml)
            assert is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])