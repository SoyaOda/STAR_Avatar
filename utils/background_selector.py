"""
Background Selector - Separate background management from rendering

This module manages background selection independently from the renderer.
Keeps rendering logic clean and allows easy background switching.

Usage:
    selector = BackgroundSelector(category='studio')
    bg_image = selector.get_background(index=5)

    # Or get random background
    bg_image = selector.get_random_background()
"""

import numpy as np
from PIL import Image
import os
from typing import Optional, List
from utils.hdri_background_manager import HDRIBackgroundManager


class BackgroundSelector:
    """
    Manages background selection separately from rendering.

    Supports:
    - Fixed background selection
    - Random background selection
    - Category-based filtering
    - Background caching
    """

    def __init__(
        self,
        hdri_manager: Optional[HDRIBackgroundManager] = None,
        category: str = 'all',
        fixed_index: Optional[int] = None
    ):
        """
        Initialize background selector.

        Args:
            hdri_manager: HDRIBackgroundManager instance (optional)
            category: Background category ('studio', 'indoor', 'outdoor', 'all')
            fixed_index: Fix to specific background index (None = random)
        """
        self.hdri_manager = hdri_manager
        self.category = category
        self.fixed_index = fixed_index
        self.cache = {}

        # Initialize HDRI manager if not provided
        if self.hdri_manager is None:
            self.hdri_manager = HDRIBackgroundManager(
                cache_dir='data/hdri_backgrounds',
                image_size=1024
            )

        # Build allowed indices based on category
        if category == 'all':
            self.allowed_indices = list(range(self.get_background_count()))
        else:
            self.allowed_indices = self.filter_by_category(category)

        print(f"BackgroundSelector initialized:")
        print(f"  - Total backgrounds: {self.get_background_count()}")
        print(f"  - Category filter: {category}")
        print(f"  - Allowed backgrounds: {len(self.allowed_indices)}")
        if self.allowed_indices:
            print(f"  - Allowed indices: {self.allowed_indices}")
        print(f"  - Fixed index: {fixed_index if fixed_index is not None else 'Random from allowed'}")

    def get_background_count(self) -> int:
        """Get total number of available backgrounds."""
        return self.hdri_manager.get_background_count()

    def list_backgrounds(self) -> List[str]:
        """List all available background files."""
        if hasattr(self.hdri_manager, 'background_files'):
            return self.hdri_manager.background_files
        else:
            # Fallback: list files in directory
            bg_dir = self.hdri_manager.cache_dir
            files = [
                f for f in os.listdir(bg_dir)
                if f.endswith(('.jpg', '.png', '.hdr', '.exr'))
            ]
            return sorted(files)

    def get_background(self, index: Optional[int] = None) -> np.ndarray:
        """
        Get background image.

        Args:
            index: Background index (None = use fixed_index or random from allowed)

        Returns:
            Background image as numpy array [H, W, 3]
        """
        # Determine which index to use
        if index is None:
            if self.fixed_index is not None:
                index = self.fixed_index
            else:
                # Random selection from allowed indices only
                if not self.allowed_indices:
                    raise ValueError(f"No backgrounds available for category '{self.category}'")
                index = np.random.choice(self.allowed_indices)

        # Check cache
        if index in self.cache:
            return self.cache[index]

        # Load background
        bg_files = self.list_backgrounds()
        if index >= len(bg_files):
            raise ValueError(f"Background index {index} out of range (0-{len(bg_files)-1})")

        bg_file = bg_files[index]
        bg_path = os.path.join(self.hdri_manager.cache_dir, bg_file)

        # Load image
        bg_image = Image.open(bg_path)
        bg_array = np.array(bg_image)

        # Ensure RGB format
        if bg_array.ndim == 2:  # Grayscale
            bg_array = np.stack([bg_array] * 3, axis=-1)
        elif bg_array.shape[-1] == 4:  # RGBA
            bg_array = bg_array[:, :, :3]

        # Cache it
        self.cache[index] = bg_array

        return bg_array

    def get_random_background(self) -> np.ndarray:
        """Get random background image."""
        return self.get_background(index=None)

    def get_background_info(self, index: int) -> dict:
        """
        Get information about specific background.

        Args:
            index: Background index

        Returns:
            Dictionary with background info
        """
        bg_files = self.list_backgrounds()
        if index >= len(bg_files):
            raise ValueError(f"Background index {index} out of range")

        bg_file = bg_files[index]
        bg_path = os.path.join(self.hdri_manager.cache_dir, bg_file)

        # Determine category from filename
        filename_lower = bg_file.lower()
        if 'studio' in filename_lower:
            category = 'studio'
        elif any(word in filename_lower for word in ['indoor', 'warehouse', 'building']):
            category = 'indoor'
        elif any(word in filename_lower for word in ['outdoor', 'street', 'alley', 'bridge']):
            category = 'outdoor'
        else:
            category = 'unknown'

        return {
            'index': index,
            'filename': bg_file,
            'path': bg_path,
            'category': category
        }

    def filter_by_category(self, category: str) -> List[int]:
        """
        Get indices of backgrounds matching category.

        Args:
            category: 'studio', 'indoor', 'outdoor', or 'all'

        Returns:
            List of background indices
        """
        if category == 'all':
            return list(range(self.get_background_count()))

        matching_indices = []
        for i in range(self.get_background_count()):
            info = self.get_background_info(i)
            if info['category'] == category:
                matching_indices.append(i)

        return matching_indices

    def get_studio_backgrounds(self) -> List[int]:
        """Get indices of studio backgrounds only."""
        return self.filter_by_category('studio')

    def print_available_backgrounds(self):
        """Print all available backgrounds with categories."""
        print("\nAvailable Backgrounds:")
        print("=" * 70)

        for i in range(self.get_background_count()):
            info = self.get_background_info(i)
            print(f"  [{i:2d}] {info['filename']:40s} ({info['category']})")

        print("=" * 70)

        # Summary by category
        categories = {}
        for i in range(self.get_background_count()):
            info = self.get_background_info(i)
            cat = info['category']
            categories[cat] = categories.get(cat, 0) + 1

        print("\nSummary:")
        for cat, count in sorted(categories.items()):
            print(f"  - {cat}: {count}")


def test_background_selector():
    """Test background selector."""
    print("\n" + "="*70)
    print("Background Selector Test")
    print("="*70)

    # Initialize selector
    selector = BackgroundSelector()

    # Print available backgrounds
    selector.print_available_backgrounds()

    # Get studio backgrounds
    studio_indices = selector.get_studio_backgrounds()
    print(f"\nStudio backgrounds: {studio_indices}")

    # Test loading a background
    if studio_indices:
        print(f"\nLoading studio background #{studio_indices[0]}...")
        bg = selector.get_background(studio_indices[0])
        print(f"  Shape: {bg.shape}")
        print(f"  Dtype: {bg.dtype}")
        print(f"  Range: [{bg.min()}, {bg.max()}]")

        # Save for inspection
        output_path = "outputs/background_test.png"
        os.makedirs("outputs", exist_ok=True)
        Image.fromarray(bg).save(output_path)
        print(f"  ✓ Saved: {output_path}")

    print("\n" + "="*70)
    print("✓ Background Selector Test Complete!")
    print("="*70)


if __name__ == "__main__":
    test_background_selector()
