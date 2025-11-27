"""
Data Augmentation for Multi-channel Body Shape Estimation

Implements augmentation transforms for multi-channel input:
- Normal maps (channels 0-2)
- Depth map (channel 3)
- Mask (channel 4)
- Joint heatmaps (channels 5-20)

Important: Geometric transforms (flip, rotation, scale) must be applied
consistently across all channels, while photometric transforms (brightness,
contrast) should only be applied to normal maps.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np


class MultiChannelAugmentation:
    """
    Augmentation for multi-channel input [21, H, W].

    Applies consistent geometric transforms and channel-specific
    photometric transforms.
    """

    def __init__(
        self,
        horizontal_flip_prob=0.5,
        rotation_degrees=10,
        scale_range=(0.9, 1.1),
        brightness_range=(0.8, 1.2),
        contrast_range=(0.8, 1.2),
        apply_photometric=True
    ):
        """
        Initialize augmentation parameters.

        Args:
            horizontal_flip_prob: Probability of horizontal flip (default: 0.5)
            rotation_degrees: Max rotation angle in degrees (default: 10)
            scale_range: (min, max) scale factors (default: 0.9 to 1.1)
            brightness_range: (min, max) brightness multipliers (default: 0.8 to 1.2)
            contrast_range: (min, max) contrast multipliers (default: 0.8 to 1.2)
            apply_photometric: Whether to apply brightness/contrast augmentation
        """
        self.horizontal_flip_prob = horizontal_flip_prob
        self.rotation_degrees = rotation_degrees
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.apply_photometric = apply_photometric

    def __call__(self, multichannel_input):
        """
        Apply augmentation to multi-channel input.

        Args:
            multichannel_input: Tensor [21, H, W]

        Returns:
            Augmented tensor [21, H, W]
        """
        # Add batch dimension for processing
        x = multichannel_input.unsqueeze(0)  # [1, 21, H, W]

        # 1. Random horizontal flip
        if torch.rand(1).item() < self.horizontal_flip_prob:
            x = self._horizontal_flip(x)

        # 2. Random rotation
        if self.rotation_degrees > 0:
            angle = (torch.rand(1).item() * 2 - 1) * self.rotation_degrees
            x = self._rotate(x, angle)

        # 3. Random scale
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            scale = torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
            x = self._scale(x, scale)

        # 4. Photometric augmentation (only on normal maps)
        if self.apply_photometric:
            x = self._photometric_augmentation(x)

        # Remove batch dimension
        x = x.squeeze(0)  # [21, H, W]

        return x

    def _horizontal_flip(self, x):
        """
        Flip horizontally. Need to negate X-component of normal maps.

        Args:
            x: Tensor [B, 21, H, W]

        Returns:
            Flipped tensor [B, 21, H, W]
        """
        # Flip all channels
        x_flipped = torch.flip(x, dims=[3])  # Flip width dimension

        # For normal maps, negate X-component (channel 0)
        x_flipped[:, 0:1, :, :] = -x_flipped[:, 0:1, :, :]

        return x_flipped

    def _rotate(self, x, angle):
        """
        Rotate by given angle.

        Args:
            x: Tensor [B, 21, H, W]
            angle: Rotation angle in degrees

        Returns:
            Rotated tensor [B, 21, H, W]
        """
        # Convert angle to radians
        angle_rad = angle * np.pi / 180.0

        # Create rotation matrix
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Affine transformation matrix for rotation around center
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0)

        # Generate grid and apply rotation
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_rotated = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        # For normal maps (channels 0-2), we need to rotate the normal vectors as well
        # Apply 2D rotation to X,Y components of normal map
        normal_xy = x_rotated[:, :2, :, :]  # [B, 2, H, W]

        # Reshape for rotation
        B, _, H, W = normal_xy.shape
        normal_xy_flat = normal_xy.permute(0, 2, 3, 1).reshape(-1, 2)  # [B*H*W, 2]

        # Rotation matrix for normal vectors (2D)
        rot_matrix = torch.tensor([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ], dtype=x.dtype, device=x.device)

        # Apply rotation
        normal_xy_rotated = torch.matmul(normal_xy_flat, rot_matrix.T)
        normal_xy_rotated = normal_xy_rotated.reshape(B, H, W, 2).permute(0, 3, 1, 2)

        # Update normal map with rotated vectors
        x_rotated[:, :2, :, :] = normal_xy_rotated

        return x_rotated

    def _scale(self, x, scale):
        """
        Apply random scaling (zoom in/out).

        Args:
            x: Tensor [B, 21, H, W]
            scale: Scale factor (> 1.0 = zoom in, < 1.0 = zoom out)

        Returns:
            Scaled tensor [B, 21, H, W]
        """
        # Create scaling matrix
        theta = torch.tensor([
            [scale, 0, 0],
            [0, scale, 0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0)

        # Generate grid and apply scaling
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_scaled = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        return x_scaled

    def _photometric_augmentation(self, x):
        """
        Apply brightness and contrast augmentation to normal maps only.

        Args:
            x: Tensor [B, 21, H, W]

        Returns:
            Augmented tensor [B, 21, H, W]
        """
        # Random brightness
        brightness = torch.rand(1).item() * (self.brightness_range[1] - self.brightness_range[0]) + self.brightness_range[0]

        # Random contrast
        contrast = torch.rand(1).item() * (self.contrast_range[1] - self.contrast_range[0]) + self.contrast_range[0]

        # Apply only to normal map channels (0-2)
        normal_maps = x[:, :3, :, :]

        # Brightness: multiply by factor
        normal_maps = normal_maps * brightness

        # Contrast: adjust around mean
        mean = normal_maps.mean(dim=[2, 3], keepdim=True)
        normal_maps = (normal_maps - mean) * contrast + mean

        # Clamp to [0, 1]
        normal_maps = torch.clamp(normal_maps, 0, 1)

        # Update normal map channels
        x[:, :3, :, :] = normal_maps

        return x


class NoAugmentation:
    """Dummy augmentation that does nothing (for validation/testing)."""

    def __call__(self, x):
        return x


def test_augmentation():
    """Test augmentation transforms."""
    print("="*60)
    print("Testing Multi-Channel Augmentation")
    print("="*60)

    # Create dummy multi-channel input
    batch_size = 2
    channels = 21
    height, width = 256, 256

    # Simulate realistic input
    x = torch.zeros(batch_size, channels, height, width)

    # Normal map (channels 0-2): random RGB-like values
    x[:, 0:3, :, :] = torch.rand(batch_size, 3, height, width)

    # Depth map (channel 3): gradient from top to bottom
    depth_gradient = torch.linspace(0, 1, height).view(1, 1, height, 1).expand(batch_size, 1, height, width)
    x[:, 3:4, :, :] = depth_gradient

    # Mask (channel 4): circular mask
    y_grid, x_grid = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing='ij')
    circle_mask = ((x_grid ** 2 + y_grid ** 2) < 0.5).float()
    x[:, 4:5, :, :] = circle_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, height, width)

    # Joint heatmaps (channels 5-20): random Gaussian blobs
    x[:, 5:21, :, :] = torch.rand(batch_size, 16, height, width) * 0.5

    print(f"\nInput shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")

    # Create augmentation
    augmentation = MultiChannelAugmentation(
        horizontal_flip_prob=0.5,
        rotation_degrees=10,
        scale_range=(0.9, 1.1),
        brightness_range=(0.8, 1.2),
        contrast_range=(0.8, 1.2),
        apply_photometric=True
    )

    # Test augmentation on single sample
    print("\n" + "-"*60)
    print("Testing Single Sample Augmentation")
    print("-"*60)

    x_single = x[0]  # [21, H, W]
    x_aug = augmentation(x_single)

    print(f"\nOriginal shape: {x_single.shape}")
    print(f"Augmented shape: {x_aug.shape}")
    print(f"Augmented range: [{x_aug.min():.3f}, {x_aug.max():.3f}]")

    # Test batch processing
    print("\n" + "-"*60)
    print("Testing Batch Augmentation")
    print("-"*60)

    x_aug_batch = torch.stack([augmentation(x[i]) for i in range(batch_size)])

    print(f"\nOriginal batch shape: {x.shape}")
    print(f"Augmented batch shape: {x_aug_batch.shape}")

    # Test with no augmentation
    print("\n" + "-"*60)
    print("Testing No Augmentation (validation mode)")
    print("-"*60)

    no_aug = NoAugmentation()
    x_no_aug = no_aug(x_single)

    print(f"\nOriginal shape: {x_single.shape}")
    print(f"No-aug shape: {x_no_aug.shape}")
    print(f"Are they equal? {torch.equal(x_single, x_no_aug)}")

    print("\n" + "="*60)
    print("✓ Augmentation test passed!")
    print("="*60)


if __name__ == "__main__":
    test_augmentation()
