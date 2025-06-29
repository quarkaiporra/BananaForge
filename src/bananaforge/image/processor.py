"""Image processing utilities for BananaForge."""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Union
import cv2


class ImageProcessor:
    """Main image processing class for preparing images for optimization."""

    def __init__(self, device: str = "cpu"):
        """Initialize image processor.

        Args:
            device: Device for tensor operations
        """
        self.device = torch.device(device)

        # Standard transforms
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

        # Normalization for neural networks
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def load_image(
        self,
        image_path: str,
        target_size: Optional[Tuple[int, int]] = None,
        maintain_aspect: bool = True,
    ) -> torch.Tensor:
        """Load and preprocess image from file.

        Args:
            image_path: Path to image file
            target_size: Optional target size (height, width)
            maintain_aspect: Whether to maintain aspect ratio

        Returns:
            Preprocessed image tensor (1, 3, H, W)
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Resize if needed
        if target_size is not None:
            if maintain_aspect:
                image = self._resize_with_aspect(image, target_size)
            else:
                image = image.resize((target_size[1], target_size[0]), Image.LANCZOS)

        # Convert to tensor
        tensor = self.to_tensor(image).unsqueeze(0).to(self.device)

        return tensor

    def _resize_with_aspect(
        self, image: Image.Image, target_size: Tuple[int, int]
    ) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        target_h, target_w = target_size
        orig_w, orig_h = image.size

        # Calculate scaling factor
        scale = min(target_w / orig_w, target_h / orig_h)

        # Calculate new size
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Resize image
        image = image.resize((new_w, new_h), Image.LANCZOS)

        # Pad to target size
        if new_w != target_w or new_h != target_h:
            # Create padded image
            padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))

            # Calculate padding offset
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2

            # Paste resized image
            padded.paste(image, (x_offset, y_offset))
            image = padded

        return image

    def preprocess_for_optimization(
        self, image: torch.Tensor, target_resolution: int = 256
    ) -> torch.Tensor:
        """Preprocess image for optimization.

        Args:
            image: Input image tensor (1, 3, H, W)
            target_resolution: Target resolution for optimization

        Returns:
            Preprocessed image tensor
        """
        # Resize to target resolution
        if image.shape[-1] != target_resolution or image.shape[-2] != target_resolution:
            image = F.interpolate(
                image,
                size=(target_resolution, target_resolution),
                mode="bilinear",
                align_corners=False,
            )

        # Ensure values are in [0, 1]
        image = torch.clamp(image, 0, 1)

        return image

    def enhance_contrast(
        self, image: torch.Tensor, factor: float = 1.2
    ) -> torch.Tensor:
        """Enhance image contrast.

        Args:
            image: Input image tensor (1, 3, H, W)
            factor: Contrast enhancement factor

        Returns:
            Contrast-enhanced image
        """
        # Convert to grayscale for mean calculation
        gray = torch.mean(image, dim=1, keepdim=True)
        mean_val = torch.mean(gray)

        # Apply contrast enhancement
        enhanced = (image - mean_val) * factor + mean_val

        return torch.clamp(enhanced, 0, 1)

    def adjust_gamma(self, image: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        """Apply gamma correction.

        Args:
            image: Input image tensor (1, 3, H, W)
            gamma: Gamma value (< 1 brightens, > 1 darkens)

        Returns:
            Gamma-corrected image
        """
        return torch.pow(image, gamma)

    def apply_bilateral_filter(
        self,
        image: torch.Tensor,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75,
    ) -> torch.Tensor:
        """Apply bilateral filter for edge-preserving smoothing.

        Args:
            image: Input image tensor (1, 3, H, W)
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in color space
            sigma_space: Filter sigma in coordinate space

        Returns:
            Filtered image tensor
        """
        # Convert to numpy for OpenCV
        np_image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        np_image = (np_image * 255).astype(np.uint8)

        # Apply bilateral filter
        filtered = cv2.bilateralFilter(np_image, d, sigma_color, sigma_space)

        # Convert back to tensor
        filtered_tensor = torch.from_numpy(filtered).float() / 255.0
        filtered_tensor = filtered_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        return filtered_tensor

    def extract_edges(
        self, image: torch.Tensor, threshold1: float = 50, threshold2: float = 150
    ) -> torch.Tensor:
        """Extract edges using Canny edge detection.

        Args:
            image: Input image tensor (1, 3, H, W)
            threshold1: First threshold for edge linking
            threshold2: Second threshold for edge linking

        Returns:
            Edge map tensor (1, 1, H, W)
        """
        # Convert to grayscale
        gray = torch.mean(image, dim=1, keepdim=True)

        # Convert to numpy for OpenCV
        np_gray = gray.squeeze().cpu().numpy()
        np_gray = (np_gray * 255).astype(np.uint8)

        # Apply Canny edge detection
        edges = cv2.Canny(np_gray, threshold1, threshold2)

        # Convert back to tensor
        edge_tensor = torch.from_numpy(edges).float() / 255.0
        edge_tensor = edge_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        return edge_tensor

    def create_distance_transform(self, edge_map: torch.Tensor) -> torch.Tensor:
        """Create distance transform from edge map.

        Args:
            edge_map: Binary edge map (1, 1, H, W)

        Returns:
            Distance transform (1, 1, H, W)
        """
        # Convert to numpy
        edges_np = edge_map.squeeze().cpu().numpy()
        edges_np = (edges_np > 0.5).astype(np.uint8)

        # Compute distance transform
        dist_transform = cv2.distanceTransform(
            1 - edges_np, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )

        # Normalize
        if dist_transform.max() > 0:
            dist_transform = dist_transform / dist_transform.max()

        # Convert back to tensor
        dist_tensor = torch.from_numpy(dist_transform).float()
        dist_tensor = dist_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        return dist_tensor

    def segment_colors(
        self, image: torch.Tensor, num_colors: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Segment image into dominant colors using K-means.

        Args:
            image: Input image tensor (1, 3, H, W)
            num_colors: Number of color clusters

        Returns:
            Tuple of (segmented_image, color_centers)
        """
        # Reshape for clustering
        h, w = image.shape[-2:]
        pixels = image.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
        pixels_np = pixels.cpu().numpy()

        # K-means clustering
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels_np)
        centers = kmeans.cluster_centers_

        # Create segmented image
        segmented_pixels = centers[labels]
        segmented_image = torch.from_numpy(segmented_pixels).float()
        segmented_image = segmented_image.reshape(h, w, 3).permute(2, 0, 1).unsqueeze(0)
        segmented_image = segmented_image.to(self.device)

        # Color centers
        color_centers = torch.from_numpy(centers).float().to(self.device)

        return segmented_image, color_centers

    def create_depth_cues(self, image: torch.Tensor) -> torch.Tensor:
        """Create depth cues from image characteristics.

        Args:
            image: Input image tensor (1, 3, H, W)

        Returns:
            Depth cue map (1, 1, H, W)
        """
        # Convert to grayscale
        gray = torch.mean(image, dim=1, keepdim=True)

        # Brightness-based depth (darker = further)
        brightness_depth = 1.0 - gray

        # Blur-based depth (more blurred = further)
        blur_kernel = torch.ones(1, 1, 5, 5, device=self.device) / 25.0
        blurred = F.conv2d(gray, blur_kernel, padding=2)
        blur_diff = torch.abs(gray - blurred)
        blur_depth = 1.0 - blur_diff / (blur_diff.max() + 1e-6)

        # Edge-based depth (fewer edges = further)
        edges = self.extract_edges(image)
        edge_depth = 1.0 - edges

        # Combine depth cues
        depth_cues = 0.4 * brightness_depth + 0.3 * blur_depth + 0.3 * edge_depth

        return depth_cues

    def save_tensor_as_image(self, tensor: torch.Tensor, filepath: str) -> None:
        """Save tensor as image file.

        Args:
            tensor: Image tensor (1, 3, H, W) or (3, H, W)
            filepath: Output file path
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # Clamp and convert to PIL
        tensor = torch.clamp(tensor, 0, 1)
        pil_image = self.to_pil(tensor.cpu())
        pil_image.save(filepath)

    def create_thumbnail(
        self, image: torch.Tensor, size: Tuple[int, int] = (128, 128)
    ) -> torch.Tensor:
        """Create thumbnail of image.

        Args:
            image: Input image tensor (1, 3, H, W)
            size: Thumbnail size (height, width)

        Returns:
            Thumbnail tensor
        """
        return F.interpolate(image, size=size, mode="bilinear", align_corners=False)


class BatchImageProcessor:
    """Batch processing utilities for multiple images."""

    def __init__(self, device: str = "cpu"):
        """Initialize batch processor."""
        self.processor = ImageProcessor(device)
        self.device = device

    def process_batch(
        self, image_paths: list, target_size: Tuple[int, int], batch_size: int = 4
    ) -> torch.Tensor:
        """Process multiple images in batches.

        Args:
            image_paths: List of image file paths
            target_size: Target size for all images
            batch_size: Batch size for processing

        Returns:
            Batched image tensor (N, 3, H, W)
        """
        processed_images = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []

            for path in batch_paths:
                img = self.processor.load_image(path, target_size)
                batch_images.append(img.squeeze(0))

            if batch_images:
                batch_tensor = torch.stack(batch_images, dim=0)
                processed_images.append(batch_tensor)

        return (
            torch.cat(processed_images, dim=0) if processed_images else torch.empty(0)
        )

    def apply_augmentations(
        self, images: torch.Tensor, augment_prob: float = 0.5
    ) -> torch.Tensor:
        """Apply random augmentations to batch of images.

        Args:
            images: Batch of images (N, 3, H, W)
            augment_prob: Probability of applying each augmentation

        Returns:
            Augmented images
        """
        augmented = images.clone()

        for i in range(images.shape[0]):
            img = images[i : i + 1]

            # Random contrast adjustment
            if torch.rand(1) < augment_prob:
                factor = 0.8 + torch.rand(1) * 0.4  # 0.8 to 1.2
                img = self.processor.enhance_contrast(img, factor.item())

            # Random gamma correction
            if torch.rand(1) < augment_prob:
                gamma = 0.8 + torch.rand(1) * 0.4  # 0.8 to 1.2
                img = self.processor.adjust_gamma(img, gamma.item())

            augmented[i] = img.squeeze(0)

        return augmented
