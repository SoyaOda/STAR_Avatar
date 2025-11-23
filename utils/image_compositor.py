"""
Image Compositor - Combine rendered person with background

Composites RGBA person renders with fixed background images.
This allows complete control over background without HDRI coupling.

Usage:
    compositor = ImageCompositor(background_path='bg.jpg')
    result = compositor.composite(person_rgba)
"""

import numpy as np
from PIL import Image
from typing import Optional, Union
import os


class ImageCompositor:
    """
    Composite rendered person (with alpha) onto fixed background.

    This cleanly separates rendering from background selection.
    """

    def __init__(
        self,
        background: Optional[Union[str, np.ndarray]] = None,
        default_background_color: tuple = (240, 240, 240)
    ):
        """
        Initialize compositor.

        Args:
            background: Background image path or array (None = solid color)
            default_background_color: RGB color if no background provided
        """
        self.background = None
        self.default_background_color = default_background_color

        if background is not None:
            self.set_background(background)

    def set_background(self, background: Union[str, np.ndarray]):
        """
        Set background image.

        Args:
            background: Image path or numpy array [H, W, 3]
        """
        if isinstance(background, str):
            # Load from file
            bg_image = Image.open(background)
            self.background = np.array(bg_image.convert('RGB'))
        else:
            # Use provided array
            self.background = background

        print(f"Background set: {self.background.shape}")

    def composite(
        self,
        person_rgba: np.ndarray,
        background: Optional[np.ndarray] = None,
        resize_background: bool = True
    ) -> np.ndarray:
        """
        Composite person onto background.

        Args:
            person_rgba: Person image with alpha [H, W, 4]
            background: Optional background (None = use self.background)
            resize_background: Resize background to match person size

        Returns:
            Composited RGB image [H, W, 3]
        """
        # Use provided background or default
        if background is None:
            background = self.background

        # Get person dimensions
        height, width = person_rgba.shape[:2]

        # Create background if not provided
        if background is None:
            background = np.full(
                (height, width, 3),
                self.default_background_color,
                dtype=np.uint8
            )
        elif resize_background:
            # Resize background to match person size
            bg_image = Image.fromarray(background)
            bg_image = bg_image.resize((width, height), Image.LANCZOS)
            background = np.array(bg_image)

        # Ensure background is correct size
        if background.shape[:2] != (height, width):
            raise ValueError(
                f"Background size {background.shape[:2]} doesn't match "
                f"person size {(height, width)}"
            )

        # Extract alpha channel
        person_rgb = person_rgba[:, :, :3]
        alpha = person_rgba[:, :, 3:4] / 255.0  # Normalize to [0, 1]

        # Alpha blending
        composited = (
            person_rgb * alpha +
            background * (1 - alpha)
        ).astype(np.uint8)

        return composited

    def composite_batch(
        self,
        person_rgba_list: list,
        background: Optional[np.ndarray] = None
    ) -> list:
        """
        Composite multiple person images onto same background.

        Args:
            person_rgba_list: List of person RGBA images
            background: Background image (None = use self.background)

        Returns:
            List of composited RGB images
        """
        results = []
        for person_rgba in person_rgba_list:
            result = self.composite(person_rgba, background)
            results.append(result)
        return results


def create_solid_background(
    width: int,
    height: int,
    color: tuple = (240, 240, 240)
) -> np.ndarray:
    """
    Create solid color background.

    Args:
        width: Image width
        height: Image height
        color: RGB color tuple

    Returns:
        Background image [H, W, 3]
    """
    return np.full((height, width, 3), color, dtype=np.uint8)


def extract_studio_background(
    hdri_path: str,
    direction: str = 'back',
    output_size: tuple = (1024, 1024)
) -> np.ndarray:
    """
    Extract studio background from HDRI panorama.

    Args:
        hdri_path: Path to HDRI image
        direction: Direction to extract ('front', 'back', 'left', 'right')
        output_size: Output image size (width, height)

    Returns:
        Extracted background [H, W, 3]
    """
    # Load HDRI panorama
    hdri = Image.open(hdri_path)
    hdri_array = np.array(hdri)

    # HDRI is 360° panorama (equirectangular projection)
    # Extract specific direction
    pano_width = hdri_array.shape[1]
    pano_height = hdri_array.shape[0]

    # Direction to horizontal position mapping
    direction_map = {
        'front': 0.0,     # 0°
        'right': 0.25,    # 90°
        'back': 0.5,      # 180°
        'left': 0.75      # 270°
    }

    if direction not in direction_map:
        raise ValueError(f"Unknown direction: {direction}")

    # Calculate center position
    center_x = int(pano_width * direction_map[direction])

    # Extract FOV (field of view) region
    # Approximate 90° horizontal FOV
    fov_width = pano_width // 4  # 90° = 1/4 of 360°

    # Calculate crop region
    left = (center_x - fov_width // 2) % pano_width
    right = (center_x + fov_width // 2) % pano_width

    # Handle wrap-around
    if left < right:
        crop = hdri_array[:, left:right]
    else:
        # Wrap around (e.g., left=3500, right=100 in 4000-wide image)
        part1 = hdri_array[:, left:]
        part2 = hdri_array[:, :right]
        crop = np.concatenate([part1, part2], axis=1)

    # Resize to output size
    crop_image = Image.fromarray(crop)
    crop_image = crop_image.resize(output_size, Image.LANCZOS)

    return np.array(crop_image)


def test_compositor():
    """Test image compositor."""
    print("\n" + "="*70)
    print("Image Compositor Test")
    print("="*70)

    # Create test person image with alpha
    print("\n1. Creating test person image (RGBA)...")
    size = 512
    person_rgba = np.zeros((size, size, 4), dtype=np.uint8)

    # Draw a simple circle (person placeholder)
    center = size // 2
    radius = size // 3
    y, x = np.ogrid[:size, :size]
    mask = (x - center)**2 + (y - center)**2 <= radius**2

    person_rgba[mask, :3] = [200, 150, 120]  # Skin color
    person_rgba[mask, 3] = 255  # Full opacity

    # Create gradient alpha for smooth edges
    distance = np.sqrt((x - center)**2 + (y - center)**2)
    edge_mask = (distance > radius - 20) & (distance <= radius)
    alpha_gradient = 255 * (1 - (distance[edge_mask] - (radius - 20)) / 20)
    person_rgba[edge_mask, 3] = alpha_gradient.astype(np.uint8)

    print(f"  Created test person: {person_rgba.shape}")

    # Test 1: Solid background
    print("\n2. Test 1: Solid background...")
    compositor = ImageCompositor(default_background_color=(220, 220, 255))
    result1 = compositor.composite(person_rgba)

    output_dir = "outputs/compositor_test"
    os.makedirs(output_dir, exist_ok=True)

    Image.fromarray(result1).save(f"{output_dir}/solid_background.png")
    print(f"  ✓ Saved: {output_dir}/solid_background.png")

    # Test 2: Extract studio background from HDRI
    print("\n3. Test 2: Studio background from HDRI...")
    studio_hdri = "data/hdri_backgrounds/studio_small_03_1k.jpg"

    if os.path.exists(studio_hdri):
        studio_bg = extract_studio_background(
            studio_hdri,
            direction='back',  # Studio is at back (180°)
            output_size=(size, size)
        )

        # Save background
        Image.fromarray(studio_bg).save(f"{output_dir}/studio_background.png")
        print(f"  ✓ Extracted studio background: {studio_bg.shape}")

        # Composite
        result2 = compositor.composite(person_rgba, background=studio_bg)
        Image.fromarray(result2).save(f"{output_dir}/studio_composite.png")
        print(f"  ✓ Saved: {output_dir}/studio_composite.png")
    else:
        print(f"  ⚠️  Studio HDRI not found: {studio_hdri}")

    print("\n" + "="*70)
    print("✓ Compositor Test Complete!")
    print("="*70)
    print(f"\nOutput directory: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    test_compositor()
