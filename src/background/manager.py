"""
Background Manager Component
Handles HDRI background extraction and management
"""
import numpy as np
from PIL import Image
import os


class BackgroundManager:
    """
    Background manager for extracting and managing backgrounds

    Features:
    - Extract specific directions from HDRI panoramas
    - Manage studio backgrounds
    - Support for multiple HDRI files
    """

    def __init__(self, hdri_dir='data/hdri_backgrounds'):
        """
        Initialize background manager

        Args:
            hdri_dir: Directory containing HDRI files
        """
        self.hdri_dir = hdri_dir
        self.current_background = None

        # Known studio backgrounds (work best with Sapiens)
        self.studio_files = [
            'studio_small_03_1k.jpg',  # Index 5
            'studio_small_08_1k.jpg'   # Index 6
        ]

    def extract_from_hdri(
        self,
        hdri_path,
        direction='back',
        output_size=(1024, 1024)
    ):
        """
        Extract background from HDRI panorama

        Args:
            hdri_path: Path to HDRI file
            direction: Direction to extract ('front', 'back', 'left', 'right')
            output_size: Output size (width, height)

        Returns:
            Extracted background [H, W, 3]
        """
        # Load HDRI panorama
        hdri = Image.open(hdri_path)
        hdri_array = np.array(hdri)

        # HDRI is 360° equirectangular projection
        pano_width = hdri_array.shape[1]
        pano_height = hdri_array.shape[0]

        # Direction mapping (0° = front, 90° = right, 180° = back, 270° = left)
        direction_map = {
            'front': 0.0,     # 0°
            'right': 0.25,    # 90°
            'back': 0.5,      # 180°
            'left': 0.75      # 270°
        }

        if direction not in direction_map:
            raise ValueError(f"Unknown direction: {direction}")

        # Calculate center position in panorama
        center_x = int(pano_width * direction_map[direction])

        # Extract 90° FOV region
        fov_width = pano_width // 4  # 90° = 1/4 of 360°

        # Calculate crop boundaries
        left = (center_x - fov_width // 2) % pano_width
        right = (center_x + fov_width // 2) % pano_width

        # Handle wrap-around at panorama edges
        if left < right:
            crop = hdri_array[:, left:right]
        else:
            # Stitch wrapped parts
            part1 = hdri_array[:, left:]
            part2 = hdri_array[:, :right]
            crop = np.concatenate([part1, part2], axis=1)

        # Resize to output size
        crop_image = Image.fromarray(crop)
        crop_image = crop_image.resize(output_size, Image.LANCZOS)

        extracted = np.array(crop_image)
        self.current_background = extracted

        return extracted

    def load_studio_background(
        self,
        index=0,
        direction='back',
        output_size=(1024, 1024)
    ):
        """
        Load studio background (optimized for Sapiens)

        Args:
            index: Studio index (0 or 1)
            direction: Direction to extract
            output_size: Output size

        Returns:
            Studio background [H, W, 3]
        """
        if index < 0 or index >= len(self.studio_files):
            raise ValueError(f"Studio index {index} out of range [0, {len(self.studio_files)-1}]")

        hdri_file = self.studio_files[index]
        hdri_path = os.path.join(self.hdri_dir, hdri_file)

        if not os.path.exists(hdri_path):
            raise FileNotFoundError(f"Studio HDRI not found: {hdri_path}")

        print(f"Loading studio background: {hdri_file}")
        print(f"Direction: {direction}")

        background = self.extract_from_hdri(hdri_path, direction, output_size)

        print(f"✓ Extracted studio background: {background.shape}")

        return background

    def create_solid_background(
        self,
        width,
        height,
        color=(240, 240, 240)
    ):
        """
        Create solid color background

        Args:
            width: Width in pixels
            height: Height in pixels
            color: RGB color tuple

        Returns:
            Solid background [H, W, 3]
        """
        background = np.full((height, width, 3), color, dtype=np.uint8)
        self.current_background = background
        return background

    def get_background(self):
        """Get current background"""
        return self.current_background

    def save_background(self, output_path):
        """Save current background to file"""
        if self.current_background is None:
            raise ValueError("No background loaded")

        Image.fromarray(self.current_background).save(output_path)
        print(f"✓ Saved background: {output_path}")


if __name__ == "__main__":
    # Test
    print("\nTesting BackgroundManager...")

    manager = BackgroundManager()

    print("\n1. Loading studio background...")
    studio_bg = manager.load_studio_background(index=0, direction='back')
    print(f"Studio background: {studio_bg.shape}")

    print("\n2. Creating solid background...")
    solid_bg = manager.create_solid_background(512, 512, color=(220, 230, 255))
    print(f"Solid background: {solid_bg.shape}")

    print("\n✓ BackgroundManager test complete")
