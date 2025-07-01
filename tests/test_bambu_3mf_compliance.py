"""
Test suite for Bambu Studio 3MF compliance.
Using BDD approach to ensure generated 3MF files work in Bambu Studio.
"""

import pytest
import zipfile
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
import tempfile

class TestBambu3MFCompliance:
    """BDD tests for Bambu Studio 3MF compliance."""
    
    @pytest.fixture
    def reference_3mf(self) -> Path:
        """Path to working reference 3MF file."""
        return Path(__file__).parent.parent / "examples" / "totoro-bambu.3mf"
    
    @pytest.fixture
    def generated_3mf(self) -> Path:
        """Path to generated 3MF file for testing."""
        return Path(__file__).parent.parent / "examples" / "outputs" / "totoro" / "bananaforge_model.3mf"
    
    def test_file_structure_validation(self, generated_3mf: Path):
        """
        Scenario 1: File Structure Validation
        Given a Bambu Studio compatible 3MF file
        When I extract the ZIP contents
        Then it should contain exactly 16 files
        And all required directories should exist
        """
        assert generated_3mf.exists(), f"Generated 3MF file not found: {generated_3mf}"
        
        with zipfile.ZipFile(generated_3mf, 'r') as zf:
            file_list = zf.namelist()
            
        # Should have exactly 16 files like reference
        assert len(file_list) == 16, f"Expected 16 files, got {len(file_list)}: {file_list}"
        
        # Required files
        required_files = [
            "[Content_Types].xml",
            "_rels/.rels",
            "3D/3dmodel.model",
            "3D/_rels/3dmodel.model.rels",
            "3D/Objects/object_1.model",
            "Metadata/custom_gcode_per_layer.xml",
            "Metadata/project_settings.config",
            "Metadata/model_settings.config",
            "Metadata/cut_information.xml",
            "Metadata/slice_info.config",
            "Metadata/plate_1.json",
            "Metadata/plate_1.png",
            "Metadata/plate_1_small.png",
            "Metadata/plate_no_light_1.png",
            "Metadata/top_1.png",
            "Metadata/pick_1.png"
        ]
        
        for required_file in required_files:
            assert required_file in file_list, f"Missing required file: {required_file}"
    
    def test_content_types_validation(self, generated_3mf: Path):
        """
        Scenario 2: Content Types Validation
        Given the [Content_Types].xml file
        When I parse the XML content
        Then it should declare required extensions
        """
        with zipfile.ZipFile(generated_3mf, 'r') as zf:
            content_types_xml = zf.read("[Content_Types].xml").decode('utf-8')
        
        root = ET.fromstring(content_types_xml)
        
        # Check for required content types
        extensions = []
        for default in root.findall('.//{http://schemas.openxmlformats.org/package/2006/content-types}Default'):
            extensions.append(default.get('Extension'))
        
        required_extensions = ['rels', 'model', 'png', 'gcode']
        for ext in required_extensions:
            assert ext in extensions, f"Missing content type for extension: {ext}"
    
    def test_main_model_structure(self, generated_3mf: Path):
        """
        Scenario 3: Main Model Structure
        Given the 3D/3dmodel.model file
        When I parse the XML structure
        Then it should have proper namespaces and metadata
        """
        with zipfile.ZipFile(generated_3mf, 'r') as zf:
            model_xml = zf.read("3D/3dmodel.model").decode('utf-8')
        
        root = ET.fromstring(model_xml)
        
        # Check namespaces
        assert 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02' in model_xml
        assert 'http://schemas.bambulab.com/package/2021' in model_xml
        assert 'http://schemas.microsoft.com/3dmanufacturing/production/2015/06' in model_xml
        
        # Check required metadata
        metadata_names = []
        for metadata in root.findall('.//metadata'):
            metadata_names.append(metadata.get('name'))
        
        required_metadata = [
            'Application', 'BambuStudio:3mfVersion', 'Copyright', 'CreationDate',
            'Description', 'Designer', 'DesignerCover', 'DesignerUserId',
            'License', 'ModificationDate', 'Origin', 'Title'
        ]
        
        for meta in required_metadata:
            assert meta in metadata_names, f"Missing metadata: {meta}"
        
        # Check build transform
        build_item = root.find('.//item')
        assert build_item is not None, "Build item not found"
        transform = build_item.get('transform')
        assert transform == "1 0 0 0 1 0 0 0 1 90 90 0.799999952", f"Wrong build transform: {transform}"
    
    def test_object_geometry_validation(self, generated_3mf: Path):
        """
        Scenario 4: Object Geometry Validation
        Given the 3D/Objects/object_1.model file
        When I parse the mesh data
        Then vertices should have coordinates in printable range
        """
        with zipfile.ZipFile(generated_3mf, 'r') as zf:
            object_xml = zf.read("3D/Objects/object_1.model").decode('utf-8')
        
        root = ET.fromstring(object_xml)
        
        # Check that NO material definitions exist in object model
        materials = root.findall('.//{http://schemas.microsoft.com/3dmanufacturing/material/2015/02}basematerials')
        assert len(materials) == 0, "Object model should not contain material definitions for Bambu compatibility"
        
        # Check vertex coordinates are in proper range
        ns = {'default': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}
        vertices = root.findall('.//default:vertex', ns)
        assert len(vertices) > 0, "No vertices found in object model"
        
        # Sample first few vertices to check coordinate ranges
        for i, vertex in enumerate(vertices[:10]):
            x = float(vertex.get('x'))
            y = float(vertex.get('y'))
            z = float(vertex.get('z'))
            
            # Based on reference: X should be negative (around -90 to -80)
            # This is the critical test that will fail with current implementation
            assert x < 0, f"Vertex {i}: X coordinate should be negative, got {x}"
            assert -100 < x < 0, f"Vertex {i}: X coordinate out of range: {x}"
            assert 0 < y < 100, f"Vertex {i}: Y coordinate out of range: {y}"
            assert 0 < z < 10, f"Vertex {i}: Z coordinate out of range: {z}"
        
        # Check triangles have NO pid attributes
        triangles = root.findall('.//default:triangle', ns)
        assert len(triangles) > 0, "No triangles found in object model"
        
        for i, triangle in enumerate(triangles[:10]):
            pid = triangle.get('pid')
            assert pid is None, f"Triangle {i} should not have pid attribute for Bambu compatibility"
    
    def test_project_settings_validation(self, generated_3mf: Path):
        """
        Scenario 5: Project Settings Validation
        Given the Metadata/project_settings.config file
        When I parse the JSON content
        Then it should contain comprehensive Bambu Studio settings
        """
        with zipfile.ZipFile(generated_3mf, 'r') as zf:
            project_settings_data = zf.read("Metadata/project_settings.config")
        
        # File size check - should be substantial (reference is 48KB)
        assert len(project_settings_data) > 40000, f"Project settings too small: {len(project_settings_data)} bytes (expected >40KB)"
        
        settings = json.loads(project_settings_data.decode('utf-8'))
        
        # Check for critical settings that Bambu Studio expects
        required_settings = [
            'filament_colour', 'filament_type', 'filament_settings_id',
            'printer_model', 'version', 'layer_height', 'nozzle_diameter'
        ]
        
        for setting in required_settings:
            assert setting in settings, f"Missing critical setting: {setting}"
        
        # Check that it's comprehensive (should have 100+ settings like reference)
        assert len(settings) > 100, f"Too few settings: {len(settings)} (expected >100)"
    
    def test_coordinate_system_compatibility(self, generated_3mf: Path, reference_3mf: Path):
        """
        Scenario 6: Coordinate System Compatibility
        Given generated and reference 3MF files
        When I compare coordinate ranges
        Then they should be in similar ranges for Bambu compatibility
        """
        # Extract coordinates from both files
        generated_coords = self._extract_coordinates(generated_3mf)
        reference_coords = self._extract_coordinates(reference_3mf)
        
        # Compare coordinate ranges (this will help identify the coordinate issue)
        gen_x_range = (min(c[0] for c in generated_coords), max(c[0] for c in generated_coords))
        ref_x_range = (min(c[0] for c in reference_coords), max(c[0] for c in reference_coords))
        
        # This test documents the issue: coordinates should be in similar ranges
        # Reference X: approximately -90 to -80
        # Our current X: approximately 0 to 100 (wrong!)
        
        assert ref_x_range[0] < 0, "Reference should have negative X coordinates"
        assert gen_x_range[0] < 0, f"Generated X coordinates should be negative like reference. Got range: {gen_x_range}, Reference: {ref_x_range}"
    
    def _extract_coordinates(self, mf_path: Path) -> List[Tuple[float, float, float]]:
        """Extract vertex coordinates from 3MF file."""
        with zipfile.ZipFile(mf_path, 'r') as zf:
            object_xml = zf.read("3D/Objects/object_1.model").decode('utf-8')
        
        root = ET.fromstring(object_xml)
        vertices = root.findall('.//vertex')
        
        coordinates = []
        for vertex in vertices[:100]:  # Sample first 100 vertices
            x = float(vertex.get('x'))
            y = float(vertex.get('y'))
            z = float(vertex.get('z'))
            coordinates.append((x, y, z))
        
        return coordinates

# Utility function to run tests manually if needed
def run_bambu_compliance_tests():
    """Run the Bambu 3MF compliance tests manually."""
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

if __name__ == "__main__":
    run_bambu_compliance_tests()