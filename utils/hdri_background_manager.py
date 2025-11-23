"""
HDRI Background Manager Component

Manages HDRI background images for photorealistic rendering.
Downloads, caches, and provides random selection of HDRI backgrounds
following Sapiens best practices.
"""

import os
import random
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple
import urllib.request
import json


class HDRIBackgroundManager:
    """
    Manages HDRI background images for compositing with rendered humans.

    Features:
    - Downloads HDRI backgrounds from Poly Haven
    - Caches backgrounds locally
    - Provides random selection for diversity
    - Converts HDRI to tone-mapped backgrounds
    - Follows Sapiens best practices (environments where people are observed)
    """

    # Curated list of suitable HDRI environments from Poly Haven
    # Criteria: Places where people are commonly observed, no people visible
    RECOMMENDED_HDRIS = {
        'indoor': [
            'studio_small_03',
            'studio_small_08',
            'empty_warehouse_01',
            'modern_buildings_2',
            'adams_place_bridge',
        ],
        'outdoor': [
            'urban_alley_01',
            'kloppenheim_02',
            'venice_sunset',
            'city_street',
            'wide_street_01',
        ],
        'neutral': [
            'kiara_1_dawn',
            'sunflowers',
            'qwantani',
            'canary_wharf',
            'misty_pines',
        ]
    }

    def __init__(self, cache_dir: str = 'data/hdri_backgrounds', image_size: int = 1024):
        """
        Initialize HDRI background manager.

        Args:
            cache_dir: Directory to cache downloaded backgrounds
            image_size: Target image size for backgrounds
        """
        self.cache_dir = cache_dir
        self.image_size = image_size
        os.makedirs(cache_dir, exist_ok=True)

        self.backgrounds: List[str] = []
        self._scan_cached_backgrounds()

    def _scan_cached_backgrounds(self):
        """Scan cache directory for existing background images."""
        if os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    self.backgrounds.append(os.path.join(self.cache_dir, file))

        print(f"Found {len(self.backgrounds)} cached HDRI backgrounds")

    def download_polyhaven_hdri(self, hdri_name: str, resolution: str = '1k') -> Optional[str]:
        """
        Download HDRI from Poly Haven.

        Args:
            hdri_name: Name of HDRI on Poly Haven (e.g., 'studio_small_03')
            resolution: Resolution to download ('1k', '2k', '4k')

        Returns:
            Path to downloaded/converted background image, or None if failed
        """
        output_path = os.path.join(self.cache_dir, f"{hdri_name}_{resolution}.jpg")

        # Skip if already cached
        if os.path.exists(output_path):
            print(f"Using cached: {hdri_name}")
            return output_path

        try:
            # Poly Haven API URL for getting download links
            api_url = f"https://api.polyhaven.com/files/{hdri_name}"

            print(f"Downloading {hdri_name} from Poly Haven...")

            # Get file info from API
            with urllib.request.urlopen(api_url) as response:
                file_info = json.loads(response.read().decode())

            # Get tonemapped JPG (better for backgrounds than raw HDR)
            if 'tonemapped' in file_info and 'url' in file_info['tonemapped']:
                download_url = file_info['tonemapped']['url']

                # Download the file
                urllib.request.urlretrieve(download_url, output_path)
                print(f"✓ Downloaded: {hdri_name} -> {output_path}")
                return output_path
            # Fallback to HDRI if tonemapped not available
            elif 'hdri' in file_info and resolution in file_info['hdri']:
                # Download HDR file (we'll convert it later if needed)
                hdr_url = file_info['hdri'][resolution]['url']
                hdr_path = output_path.replace('.jpg', '.hdr')

                print(f"  Tonemapped not available, downloading HDR instead...")
                urllib.request.urlretrieve(hdr_url, hdr_path)

                # Try to convert HDR to JPG using PIL/imageio
                try:
                    import imageio
                    import numpy as np
                    from PIL import Image

                    # Read HDR
                    hdr_image = imageio.imread(hdr_path, format='HDR-FI')

                    # Simple tone mapping (Reinhard)
                    rgb = hdr_image[:, :, :3]
                    rgb_normalized = rgb / (1.0 + rgb)
                    rgb_uint8 = (np.clip(rgb_normalized, 0, 1) * 255).astype(np.uint8)

                    # Save as JPG
                    Image.fromarray(rgb_uint8).save(output_path)
                    os.remove(hdr_path)  # Clean up HDR file

                    print(f"✓ Converted HDR to JPG: {hdri_name} -> {output_path}")
                    return output_path
                except ImportError:
                    print(f"⚠️  imageio not available, keeping HDR file")
                    return hdr_path
                except Exception as e:
                    print(f"✗ HDR conversion failed: {e}")
                    return None
            else:
                print(f"✗ No suitable format available for {hdri_name}")
                return None

        except Exception as e:
            print(f"✗ Failed to download {hdri_name}: {e}")
            return None

    def download_recommended_set(self, category: str = 'all', max_count: int = 15) -> int:
        """
        Download a recommended set of HDRI backgrounds.

        Args:
            category: 'indoor', 'outdoor', 'neutral', or 'all'
            max_count: Maximum number to download

        Returns:
            Number of successfully downloaded backgrounds
        """
        print("\n" + "="*70)
        print("Downloading Recommended HDRI Backgrounds from Poly Haven")
        print("="*70)

        hdri_list = []
        if category == 'all':
            for cat_list in self.RECOMMENDED_HDRIS.values():
                hdri_list.extend(cat_list)
        elif category in self.RECOMMENDED_HDRIS:
            hdri_list = self.RECOMMENDED_HDRIS[category]
        else:
            print(f"Unknown category: {category}")
            return 0

        # Limit to max_count
        hdri_list = hdri_list[:max_count]

        success_count = 0
        for hdri_name in hdri_list:
            result = self.download_polyhaven_hdri(hdri_name, resolution='1k')
            if result:
                success_count += 1
                if result not in self.backgrounds:
                    self.backgrounds.append(result)

        print(f"\n✓ Successfully downloaded/cached {success_count}/{len(hdri_list)} backgrounds")
        return success_count

    def load_background(self, path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Load and resize a background image.

        Args:
            path: Path to background image
            target_size: Target size (width, height), or None to use self.image_size

        Returns:
            Background image as numpy array [H, W, 3] in RGB, values [0, 255]
        """
        if target_size is None:
            target_size = (self.image_size, self.image_size)

        img = Image.open(path).convert('RGB')

        # Resize to target size
        # Use LANCZOS for high-quality downsampling
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)

        return np.array(img_resized, dtype=np.uint8)

    def get_random_background(self, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Get a random background image.

        Args:
            target_size: Target size (width, height)

        Returns:
            Random background image as numpy array [H, W, 3]
        """
        if not self.backgrounds:
            raise RuntimeError(
                "No backgrounds available. "
                "Run download_recommended_set() first or add images to cache_dir."
            )

        bg_path = random.choice(self.backgrounds)
        return self.load_background(bg_path, target_size)

    def composite_rgba_with_background(
        self,
        rgba_foreground: np.ndarray,
        background: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Composite RGBA foreground with background using alpha blending.

        Args:
            rgba_foreground: RGBA image [H, W, 4] with values [0, 255]
            background: Background image [H, W, 3] (optional, random if None)

        Returns:
            Composited RGB image [H, W, 3] with values [0, 255]
        """
        # Get background if not provided
        if background is None:
            h, w = rgba_foreground.shape[:2]
            background = self.get_random_background(target_size=(w, h))

        # Ensure background matches foreground size
        h, w = rgba_foreground.shape[:2]
        if background.shape[:2] != (h, w):
            background_pil = Image.fromarray(background)
            background_pil = background_pil.resize((w, h), Image.Resampling.LANCZOS)
            background = np.array(background_pil)

        # Extract alpha channel and normalize to [0, 1]
        alpha = rgba_foreground[:, :, 3:4].astype(np.float32) / 255.0

        # Extract RGB channels
        foreground_rgb = rgba_foreground[:, :, :3].astype(np.float32)
        background_rgb = background.astype(np.float32)

        # Alpha blending: result = foreground * alpha + background * (1 - alpha)
        composited = foreground_rgb * alpha + background_rgb * (1.0 - alpha)

        # Convert back to uint8
        composited = np.clip(composited, 0, 255).astype(np.uint8)

        return composited

    def add_custom_background(self, image_path: str):
        """
        Add a custom background image to the collection.

        Args:
            image_path: Path to custom background image
        """
        if os.path.exists(image_path):
            self.backgrounds.append(image_path)
            print(f"Added custom background: {image_path}")
        else:
            print(f"Background not found: {image_path}")

    def get_background_count(self) -> int:
        """Get number of available backgrounds."""
        return len(self.backgrounds)

    def list_backgrounds(self):
        """Print list of available backgrounds."""
        print(f"\nAvailable backgrounds ({len(self.backgrounds)}):")
        for i, bg in enumerate(self.backgrounds, 1):
            print(f"  {i}. {os.path.basename(bg)}")


if __name__ == "__main__":
    """Test HDRI background manager."""

    print("Testing HDRI Background Manager\n")

    # Initialize manager
    manager = HDRIBackgroundManager(cache_dir='data/hdri_backgrounds', image_size=1024)

    # Download recommended set
    print("\n1. Downloading recommended HDRI backgrounds...")
    count = manager.download_recommended_set(category='all', max_count=15)

    # List available backgrounds
    print("\n2. Available backgrounds:")
    manager.list_backgrounds()

    # Test loading a random background
    if manager.get_background_count() > 0:
        print("\n3. Testing random background loading...")
        bg = manager.get_random_background(target_size=(512, 512))
        print(f"   Loaded background shape: {bg.shape}, dtype: {bg.dtype}")
        print(f"   Value range: [{bg.min()}, {bg.max()}]")

        # Save test image
        test_output = 'outputs/test_hdri_background.png'
        os.makedirs('outputs', exist_ok=True)
        Image.fromarray(bg).save(test_output)
        print(f"   Saved test background: {test_output}")

        # Test compositing with dummy RGBA
        print("\n4. Testing alpha compositing...")
        dummy_rgba = np.zeros((512, 512, 4), dtype=np.uint8)
        # Create a circular foreground object
        center = 256
        radius = 150
        y, x = np.ogrid[:512, :512]
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        dummy_rgba[mask, 0] = 200  # Red circle
        dummy_rgba[mask, 3] = 255  # Full opacity

        composited = manager.composite_rgba_with_background(dummy_rgba, bg)

        composite_output = 'outputs/test_hdri_composite.png'
        Image.fromarray(composited).save(composite_output)
        print(f"   Saved test composite: {composite_output}")

    print("\n✓ HDRI Background Manager test complete!")
