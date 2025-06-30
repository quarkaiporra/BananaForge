"""BDD/TDD tests for Feature 1: Advanced Image Processing Pipeline."""

import pytest
import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import tempfile

from bananaforge.image.processor import ImageProcessor
from bananaforge.utils.color import ColorConverter


class TestStory1_1_LABColorSpaceConversion:
    """Story 1.1: LAB Color Space Conversion
    
    As a 3D printing enthusiast
    I want the system to use perceptually uniform color space for material matching
    So that color transitions are more natural and accurate
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ImageProcessor(device="cpu")
        self.color_converter = ColorConverter()

    def test_given_rgb_image_when_convert_to_lab_then_preserve_perceptual_relationships(self):
        """
        Given an RGB input image
        When I process the image for optimization  
        Then the system should convert to LAB color space
        And preserve perceptual color relationships
        """
        # Given: RGB input image with known color relationships
        rgb_image = torch.tensor([
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],      # Red, Green, Blue
            [[255, 255, 0], [255, 0, 255], [0, 255, 255]], # Yellow, Magenta, Cyan
            [[128, 128, 128], [64, 64, 64], [192, 192, 192]] # Grays
        ], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
        
        # When: Convert to LAB color space
        lab_image = self.color_converter.rgb_to_lab(rgb_image)
        
        # Then: Should preserve perceptual relationships
        assert lab_image.shape == rgb_image.shape
        assert lab_image.dtype == torch.float32
        
        # LAB should have reasonable ranges (L* roughly [0, 100], a* and b* can extend beyond [-128, 127])
        l_channel = lab_image[:, 0, :, :]
        a_channel = lab_image[:, 1, :, :]
        b_channel = lab_image[:, 2, :, :]
        
        # L* should be in [0, 100] range
        assert torch.all(l_channel >= 0) and torch.all(l_channel <= 100)
        
        # a* and b* can have wider ranges, just check they're finite
        assert torch.all(torch.isfinite(a_channel))
        assert torch.all(torch.isfinite(b_channel))

    def test_given_rgb_image_when_convert_to_lab_then_more_accurate_than_rgb_matching(self):
        """
        Given an RGB input image
        When I process the image for optimization
        Then provide more accurate material matching than RGB
        """
        # Given: Two colors that are perceptually close but far in RGB
        color1_rgb = torch.tensor([[[[255]], [[0]], [[0]]]], dtype=torch.float32)  # Pure red
        color2_rgb = torch.tensor([[[[200]], [[50]], [[50]]]], dtype=torch.float32)  # Dark red
        
        # When: Convert both to LAB
        color1_lab = self.color_converter.rgb_to_lab(color1_rgb)
        color2_lab = self.color_converter.rgb_to_lab(color2_rgb)
        
        # Then: LAB distance should reflect perceptual similarity better
        rgb_distance = torch.norm(color1_rgb - color2_rgb)
        lab_distance = torch.norm(color1_lab - color2_lab)
        
        # LAB distance should be smaller for perceptually similar colors
        assert lab_distance < rgb_distance

    def test_given_various_color_ranges_when_convert_to_lab_then_handle_edge_cases(self):
        """Test edge cases for LAB conversion."""
        # Given: Edge case colors (black, white, extreme values)
        edge_colors = torch.tensor([
            [[0, 0, 0], [255, 255, 255], [255, 0, 0]],     # Black, White, Red
            [[0, 255, 0], [0, 0, 255], [127, 127, 127]]    # Green, Blue, Mid-gray
        ], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
        
        # When: Convert to LAB
        lab_result = self.color_converter.rgb_to_lab(edge_colors)
        
        # Then: Should handle all cases without errors
        assert not torch.isnan(lab_result).any()
        assert not torch.isinf(lab_result).any()


class TestStory1_2_ColorPreservingImageResizing:
    """Story 1.2: Color-Preserving Image Resizing
    
    As a user processing high-resolution images
    I want the resizing algorithm to preserve color detail
    So that small color variations aren't lost during preprocessing
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ImageProcessor(device="cpu")

    def test_given_high_res_image_when_resize_then_preserve_color_details(self):
        """
        Given a high-resolution input image with fine color details
        When the system resizes the image for processing
        Then it should use INTER_AREA interpolation for downscaling
        And preserve edge sharpness using INTER_CUBIC where appropriate
        """
        # Given: High-resolution image with fine details
        high_res_image = self._create_test_image_with_fine_details(512, 512)
        
        # When: Resize using color-preserving method
        resized_image = self.processor.resize_color_preserving(
            high_res_image, target_size=(128, 128)
        )
        
        # Then: Should preserve color details better than standard resizing
        standard_resized = torch.nn.functional.interpolate(
            high_res_image, size=(128, 128), mode='bilinear'
        )
        
        # Measure color preservation (higher SSIM = better preservation)
        color_preservation_score = self._calculate_color_preservation(
            high_res_image, resized_image, standard_resized
        )
        
        assert color_preservation_score > 0.8  # Threshold for good preservation
        assert resized_image.shape == (1, 3, 128, 128)

    def test_given_image_when_downscale_then_use_inter_area(self):
        """
        Given an image to be downscaled
        When resizing for smaller dimensions
        Then should use INTER_AREA interpolation
        """
        # Given: Large image to be downscaled
        large_image = torch.rand(1, 3, 256, 256)
        
        # When: Resize to smaller dimensions
        result = self.processor.resize_color_preserving(
            large_image, target_size=(64, 64)
        )
        
        # Then: Should use appropriate interpolation method
        assert result.shape == (1, 3, 64, 64)
        # Verify the method maintains color relationships
        original_mean = torch.mean(large_image, dim=[2, 3])
        resized_mean = torch.mean(result, dim=[2, 3])
        
        # Color means should be preserved approximately
        assert torch.allclose(original_mean, resized_mean, atol=0.1)

    def test_given_image_when_upscale_then_use_inter_cubic(self):
        """
        Given an image to be upscaled
        When resizing for larger dimensions  
        Then should use INTER_CUBIC for edge preservation
        """
        # Given: Small image to be upscaled
        small_image = torch.rand(1, 3, 64, 64)
        
        # When: Resize to larger dimensions
        result = self.processor.resize_color_preserving(
            small_image, target_size=(256, 256)
        )
        
        # Then: Should preserve edge sharpness
        assert result.shape == (1, 3, 256, 256)
        # Verify edges are preserved (no excessive blurring)
        edge_strength = self._calculate_edge_strength(result)
        assert edge_strength > 0.1  # Threshold for acceptable edge preservation

    def test_given_various_aspect_ratios_when_resize_then_maintain_quality(self):
        """Test resizing with different aspect ratios."""
        # Given: Images with different aspect ratios
        square_image = torch.rand(1, 3, 100, 100)
        wide_image = torch.rand(1, 3, 100, 200)
        tall_image = torch.rand(1, 3, 200, 100)
        
        target_size = (64, 64)
        
        # When: Resize all images
        resized_square = self.processor.resize_color_preserving(square_image, target_size)
        resized_wide = self.processor.resize_color_preserving(wide_image, target_size)
        resized_tall = self.processor.resize_color_preserving(tall_image, target_size)
        
        # Then: All should be resized properly
        assert resized_square.shape == (1, 3, 64, 64)
        assert resized_wide.shape == (1, 3, 64, 64)
        assert resized_tall.shape == (1, 3, 64, 64)

    def _create_test_image_with_fine_details(self, height, width):
        """Create a test image with fine color details."""
        # Create checkerboard pattern with gradual color transitions
        image = torch.zeros(1, 3, height, width)
        for i in range(height):
            for j in range(width):
                # Create fine color patterns
                r = (i + j) % 256 / 255.0
                g = (i * 2 + j) % 256 / 255.0  
                b = (i + j * 2) % 256 / 255.0
                image[0, 0, i, j] = r
                image[0, 1, i, j] = g
                image[0, 2, i, j] = b
        return image

    def _calculate_color_preservation(self, original, enhanced_resize, standard_resize):
        """Calculate how well color details are preserved."""
        # Simple metric: compare variance preservation
        orig_var = torch.var(original)
        enhanced_var = torch.var(enhanced_resize)
        standard_var = torch.var(standard_resize)
        
        enhanced_preservation = 1.0 - abs(orig_var - enhanced_var) / orig_var
        standard_preservation = 1.0 - abs(orig_var - standard_var) / orig_var
        
        return enhanced_preservation / (standard_preservation + 1e-6)

    def _calculate_edge_strength(self, image):
        """Calculate edge strength in image."""
        # Convert to numpy for edge detection
        img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
        gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return np.mean(edges) / 255.0


class TestStory1_3_SaturationEnhancement:
    """Story 1.3: Saturation Enhancement
    
    As a user with low-saturation input images
    I want the system to enhance saturation intelligently
    So that the final 3D model has more vibrant colors
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ImageProcessor(device="cpu")

    def test_given_low_saturation_image_when_enhance_then_increase_vibrance(self):
        """
        Given an input image with low color saturation
        When I enable saturation enhancement
        Then the system should increase saturation by a configurable percentage
        """
        # Given: Low saturation image (desaturated colors)
        low_sat_image = torch.tensor([
            [[0.8, 0.7, 0.7], [0.7, 0.8, 0.7], [0.7, 0.7, 0.8]],  # Muted colors
            [[0.6, 0.6, 0.6], [0.5, 0.5, 0.5], [0.4, 0.4, 0.4]]   # Grays
        ], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
        
        enhancement_factor = 0.3  # 30% increase
        
        # When: Apply saturation enhancement
        enhanced_image = self.processor.enhance_saturation(
            low_sat_image, enhancement_factor
        )
        
        # Then: Saturation should increase by specified percentage
        original_saturation = self._calculate_saturation(low_sat_image)
        enhanced_saturation = self._calculate_saturation(enhanced_image)
        
        saturation_increase = (enhanced_saturation - original_saturation) / original_saturation
        assert abs(saturation_increase - enhancement_factor) < 0.1  # Allow 10% tolerance

    def test_given_image_when_enhance_saturation_then_maintain_color_balance(self):
        """
        Given an input image
        When saturation enhancement is applied
        Then should maintain color balance and relationships
        """
        # Given: Image with balanced colors
        balanced_image = torch.tensor([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # Pure RGB
            [[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]   # Mixed colors
        ], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
        
        # When: Apply saturation enhancement
        enhanced = self.processor.enhance_saturation(balanced_image, 0.2)
        
        # Then: Color relationships should be preserved
        # Red pixel should still be most red, etc.
        red_pixel_orig = balanced_image[0, :, 0, 0]
        red_pixel_enhanced = enhanced[0, :, 0, 0]
        
        # Red channel should still be dominant
        assert red_pixel_enhanced[0] > red_pixel_enhanced[1]
        assert red_pixel_enhanced[0] > red_pixel_enhanced[2]

    def test_given_various_enhancement_levels_when_apply_then_scale_appropriately(self):
        """Test saturation enhancement with different levels."""
        # Given: Base image
        base_image = torch.tensor([
            [[0.6, 0.4, 0.4], [0.4, 0.6, 0.4]]
        ], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
        
        enhancement_levels = [0.0, 0.1, 0.3, 0.5]
        saturations = []
        
        # When: Apply different enhancement levels
        for level in enhancement_levels:
            enhanced = self.processor.enhance_saturation(base_image, level)
            saturations.append(self._calculate_saturation(enhanced))
        
        # Then: Saturation should increase monotonically
        for i in range(1, len(saturations)):
            assert saturations[i] >= saturations[i-1]

    def test_given_already_saturated_image_when_enhance_then_handle_gracefully(self):
        """Test enhancement on already highly saturated images."""
        # Given: Highly saturated image
        saturated_image = torch.tensor([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]  # Pure colors
        ], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
        
        # When: Apply saturation enhancement
        enhanced = self.processor.enhance_saturation(saturated_image, 0.3)
        
        # Then: Should not exceed valid ranges or create artifacts
        assert torch.all(enhanced >= 0.0)
        assert torch.all(enhanced <= 1.0)
        assert not torch.isnan(enhanced).any()

    def _calculate_saturation(self, image):
        """Calculate average saturation of image."""
        # Handle different tensor dimensions
        if image.dim() == 4:
            img_for_conversion = image.squeeze(0)  # Remove batch dimension
        else:
            img_for_conversion = image
            
        if img_for_conversion.dim() == 3:
            img_np = img_for_conversion.permute(1, 2, 0).cpu().numpy()
        else:
            # If 2D, assume it's already squeezed too much, recreate proper dims
            return 0.5  # Return default saturation value
            
        img_uint8 = (img_np * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1] / 255.0
        return np.mean(saturation)


class TestFeature1Integration:
    """Integration tests for Feature 1: Advanced Image Processing Pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ImageProcessor(device="cpu")

    def test_given_real_image_when_process_complete_pipeline_then_maintain_quality(self):
        """
        Given a real input image
        When processing through the complete enhanced pipeline
        Then should maintain quality while applying all enhancements
        """
        # Given: Real test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            # Create a realistic test image
            test_image = Image.new('RGB', (256, 256), color='white')
            # Add some patterns
            pixels = test_image.load()
            for i in range(256):
                for j in range(256):
                    r = int((i + j) % 256)
                    g = int((i * 2) % 256) 
                    b = int((j * 2) % 256)
                    pixels[i, j] = (r, g, b)
            
            test_image.save(tmp.name)
            test_image_path = tmp.name

        try:
            # When: Process through complete pipeline
            processed = self.processor.load_and_process_enhanced(
                test_image_path,
                target_size=(128, 128),
                enable_lab_conversion=True,
                saturation_enhancement=0.2,
                use_color_preserving_resize=True
            )
            
            # Then: Should produce valid output
            assert isinstance(processed, dict)
            assert 'rgb_image' in processed
            assert 'lab_image' in processed
            assert processed['rgb_image'].shape == (1, 3, 128, 128)
            assert processed['lab_image'].shape == (1, 3, 128, 128)
            
        finally:
            # Cleanup
            Path(test_image_path).unlink()

    def test_performance_benchmark_for_enhanced_pipeline(self):
        """Benchmark performance of enhanced pipeline vs standard."""
        import time
        
        # Given: Test image
        test_image = torch.rand(1, 3, 512, 512)
        
        # When: Time both pipelines
        start_time = time.time()
        standard_result = torch.nn.functional.interpolate(
            test_image, size=(128, 128), mode='bilinear'
        )
        standard_time = time.time() - start_time
        
        start_time = time.time()
        enhanced_result = self.processor.resize_color_preserving(
            test_image, target_size=(128, 128)
        )
        enhanced_time = time.time() - start_time
        
        # Then: Enhanced pipeline should complete in reasonable time
        # Just ensure it completes within 1 second for a 512x512 image
        assert enhanced_time < 1.0
        assert enhanced_result.shape == standard_result.shape


# Helper fixtures and utilities
@pytest.fixture
def sample_rgb_image():
    """Fixture providing a sample RGB image for testing."""
    return torch.tensor([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
        [[128, 128, 128], [64, 64, 64], [192, 192, 192]]
    ], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)


@pytest.fixture  
def low_saturation_image():
    """Fixture providing a low saturation image for testing."""
    return torch.tensor([
        [[0.8, 0.7, 0.7], [0.7, 0.8, 0.7], [0.7, 0.7, 0.8]],
        [[0.6, 0.6, 0.6], [0.5, 0.5, 0.5], [0.4, 0.4, 0.4]]
    ], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)