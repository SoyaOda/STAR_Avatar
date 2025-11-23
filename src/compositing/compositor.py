"""
Compositor Component
Handles alpha compositing of rendered person with background
"""
import numpy as np
from PIL import Image
from typing import Optional, Union


class Compositor:
    """
    Image compositor for combining RGBA person with background

    Features:
    - Alpha blending
    - Automatic background resizing
    - Batch processing support
    """

    def __init__(self, background=None):
        """
        Initialize compositor

        Args:
            background: Background image (path, array, or None)
        """
        self.background = None

        if background is not None:
            self.set_background(background)

    def set_background(self, background: Union[str, np.ndarray]):
        """
        Set background image

        Args:
            background: Image path or numpy array [H, W, 3]
        """
        if isinstance(background, str):
            # Load from file
            bg_image = Image.open(background)
            self.background = np.array(bg_image.convert('RGB'))
        else:
            # Use provided array
            self.background = background.copy()

        print(f"✓ Background set: {self.background.shape}")

    def composite(
        self,
        person_rgba: np.ndarray,
        background: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Composite person onto background using alpha blending

        Args:
            person_rgba: Person RGBA image [H, W, 4]
            background: Optional background [H, W, 3] (None = use self.background)

        Returns:
            Composited RGB image [H, W, 3]
        """
        # Use provided background or stored background
        if background is None:
            background = self.background

        if background is None:
            raise ValueError("No background provided and no background set")

        # Get dimensions
        height, width = person_rgba.shape[:2]

        # Resize background if needed
        if background.shape[:2] != (height, width):
            bg_image = Image.fromarray(background)
            bg_image = bg_image.resize((width, height), Image.LANCZOS)
            background = np.array(bg_image)

        # Extract RGB and alpha
        person_rgb = person_rgba[:, :, :3]
        alpha = person_rgba[:, :, 3:4] / 255.0  # Normalize to [0, 1]

        # Alpha blending: result = foreground * alpha + background * (1 - alpha)
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
        Composite multiple person images onto same background

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

    def get_background(self):
        """Get current background"""
        return self.background


if __name__ == "__main__":
    # Test
    print("\nTesting Compositor...")

    # Create test person RGBA (circle with alpha)
    size = 512
    person_rgba = np.zeros((size, size, 4), dtype=np.uint8)

    center = size // 2
    radius = size // 3
    y, x = np.ogrid[:size, :size]
    mask = (x - center)**2 + (y - center)**2 <= radius**2

    person_rgba[mask, :3] = [200, 150, 120]  # Skin color
    person_rgba[mask, 3] = 255  # Full opacity

    print(f"Test person: {person_rgba.shape}")

    # Test with solid background
    print("\n1. Solid background...")
    solid_bg = np.full((size, size, 3), (220, 230, 255), dtype=np.uint8)

    compositor = Compositor(background=solid_bg)
    result = compositor.composite(person_rgba)

    print(f"Result: {result.shape}, dtype: {result.dtype}")

    # Test batch
    print("\n2. Batch compositing...")
    batch_results = compositor.composite_batch([person_rgba, person_rgba])
    print(f"Batch results: {len(batch_results)} images")

    print("\n✓ Compositor test complete")
