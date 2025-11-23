"""
Simple Clothing Generator for STAR Body Models

Adds basic sports clothing (sports bra + shorts) to 3D body meshes
via vertex displacement. This is a fast, lightweight approach suitable
for synthetic data generation.

Features:
- Sports bra (upper body coverage)
- Athletic shorts (lower body coverage)
- Configurable expansion factors
- Gender-specific adjustments
- No external dependencies (pure NumPy)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class SimpleClothingGenerator:
    """
    Simple clothing generator using vertex displacement.

    This approach expands vertices in specific body regions to create
    the appearance of clothing. While not physically accurate, it provides
    a fast and effective way to add basic clothing for synthetic data.

    Usage:
        clothing_gen = SimpleClothingGenerator()
        clothed_vertices = clothing_gen.add_clothing(vertices, ['shorts', 'sports_bra'])
    """

    # Clothing region definitions (in STAR coordinate space)
    CLOTHING_REGIONS = {
        'shorts': {
            'name': 'Athletic Shorts',
            'y_min': -0.85,      # From upper thigh
            'y_max': -0.05,      # To waist
            'expansion': 1.035,  # 3.5% radial expansion (visible clothing layer)
            'description': 'Athletic shorts covering lower body'
        },
        'sports_bra': {
            'name': 'Sports Bra',
            'y_min': 0.35,       # From below chest
            'y_max': 0.75,       # To shoulders
            'expansion': 1.030,  # 3.0% radial expansion (visible clothing layer)
            'description': 'Sports bra covering upper torso'
        },
        'tank_top': {
            'name': 'Tank Top',
            'y_min': 0.20,       # From waist
            'y_max': 0.80,       # To shoulders
            'expansion': 1.040,  # 4.0% radial expansion (looser fit)
            'description': 'Tank top covering torso'
        },
        'leggings': {
            'name': 'Leggings',
            'y_min': -1.0,       # From feet
            'y_max': -0.05,      # To waist
            'expansion': 1.025,  # 2.5% radial expansion (tight fit)
            'description': 'Full-length leggings'
        }
    }

    def __init__(self, gender: str = 'neutral'):
        """
        Initialize clothing generator.

        Args:
            gender: Body gender ('neutral', 'male', 'female')
                    Used for gender-specific adjustments
        """
        self.gender = gender

    def get_clothing_masks(
        self,
        vertices: np.ndarray,
        clothing_types: List[str] = ['shorts', 'sports_bra']
    ) -> Dict[str, np.ndarray]:
        """
        Get boolean masks for clothing regions.

        Args:
            vertices: STAR vertices [N, 3]
            clothing_types: List of clothing types

        Returns:
            Dictionary mapping clothing types to boolean masks [N]
        """
        masks = {}

        for clothing_type in clothing_types:
            if clothing_type not in self.CLOTHING_REGIONS:
                continue

            region = self.CLOTHING_REGIONS[clothing_type]

            # Find vertices in this region (based on Y coordinate)
            mask = (vertices[:, 1] >= region['y_min']) & \
                   (vertices[:, 1] <= region['y_max'])

            masks[clothing_type] = mask

        return masks

    def add_clothing(
        self,
        vertices: np.ndarray,
        clothing_types: List[str] = ['shorts', 'sports_bra'],
        custom_expansion: Optional[Dict[str, float]] = None,
        return_masks: bool = False
    ) -> np.ndarray:
        """
        Add simple clothing to STAR vertices.

        Args:
            vertices: STAR vertices [6890, 3] numpy array (x, y, z coordinates)
            clothing_types: List of clothing types to add
                           Options: 'shorts', 'sports_bra', 'tank_top', 'leggings'
            custom_expansion: Optional dict to override default expansion factors
                            e.g., {'shorts': 1.02, 'sports_bra': 1.015}
            return_masks: If True, return (clothed_vertices, masks) tuple

        Returns:
            clothed_vertices: Modified vertices [6890, 3]
            OR (clothed_vertices, masks) if return_masks=True
        """
        # Validate input
        if vertices.shape[1] != 3:
            raise ValueError(f"Expected vertices with shape [N, 3], got {vertices.shape}")

        # Work on a copy to avoid modifying original
        vertices_clothed = vertices.copy()
        clothing_masks = {} if return_masks else None

        # Apply each clothing type
        for clothing_type in clothing_types:
            if clothing_type not in self.CLOTHING_REGIONS:
                print(f"⚠️  Unknown clothing type: {clothing_type}")
                print(f"   Available types: {list(self.CLOTHING_REGIONS.keys())}")
                continue

            region = self.CLOTHING_REGIONS[clothing_type].copy()

            # Apply custom expansion if provided
            if custom_expansion and clothing_type in custom_expansion:
                region['expansion'] = custom_expansion[clothing_type]

            # Apply gender-specific adjustments
            if self.gender == 'female' and clothing_type == 'sports_bra':
                region['expansion'] *= 1.003  # Slightly more expansion for female

            # Find vertices in this region (based on Y coordinate)
            mask = (vertices[:, 1] >= region['y_min']) & \
                   (vertices[:, 1] <= region['y_max'])

            if mask.sum() == 0:
                print(f"⚠️  No vertices found in region for {clothing_type}")
                continue

            # Calculate regional center (weighted by Y to maintain natural shape)
            regional_vertices = vertices[mask]
            weights = (regional_vertices[:, 1] - region['y_min']) / \
                     (region['y_max'] - region['y_min'])
            weights = np.clip(weights, 0, 1)[:, np.newaxis]

            # Expand radially from weighted center
            center = np.average(regional_vertices, axis=0, weights=weights.flatten())

            # Apply radial expansion
            offset = regional_vertices - center
            vertices_clothed[mask] = center + offset * region['expansion']

            # Store mask if requested
            if return_masks:
                clothing_masks[clothing_type] = mask

        if return_masks:
            return vertices_clothed, clothing_masks
        else:
            return vertices_clothed

    def add_clothing_with_smooth_transition(
        self,
        vertices: np.ndarray,
        clothing_types: List[str] = ['shorts', 'sports_bra'],
        transition_width: float = 0.05
    ) -> np.ndarray:
        """
        Add clothing with smooth transitions at boundaries.

        This method applies a smooth blending function at the edges of
        clothing regions to avoid sharp discontinuities.

        Args:
            vertices: STAR vertices [6890, 3]
            clothing_types: List of clothing types to add
            transition_width: Width of smooth transition zone (in Y coordinate units)

        Returns:
            clothed_vertices: Modified vertices with smooth transitions
        """
        vertices_clothed = vertices.copy()

        for clothing_type in clothing_types:
            if clothing_type not in self.CLOTHING_REGIONS:
                continue

            region = self.CLOTHING_REGIONS[clothing_type]

            # Expanded region boundaries for smooth transition
            y_min_inner = region['y_min'] + transition_width
            y_max_inner = region['y_max'] - transition_width

            # Calculate blend weights for smooth transition
            y_coords = vertices[:, 1]

            # Bottom transition
            bottom_blend = np.clip(
                (y_coords - region['y_min']) / transition_width,
                0, 1
            )

            # Top transition
            top_blend = np.clip(
                (region['y_max'] - y_coords) / transition_width,
                0, 1
            )

            # Combined blend weight (0 outside, 1 in center)
            blend_weight = np.minimum(bottom_blend, top_blend)

            # Only process vertices within extended region
            mask = (y_coords >= region['y_min']) & (y_coords <= region['y_max'])

            if mask.sum() == 0:
                continue

            # Calculate center and expansion
            regional_vertices = vertices[mask]
            center = regional_vertices.mean(axis=0)
            offset = regional_vertices - center

            # Apply blended expansion
            expansion_factor = 1.0 + (region['expansion'] - 1.0) * blend_weight[mask, np.newaxis]
            vertices_clothed[mask] = center + offset * expansion_factor

        return vertices_clothed

    def get_clothing_info(self) -> Dict[str, Dict]:
        """
        Get information about available clothing types.

        Returns:
            Dictionary with clothing type information
        """
        return self.CLOTHING_REGIONS.copy()

    def visualize_regions(self, vertices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Visualize clothing regions by returning vertex masks.

        Useful for debugging and visualization.

        Args:
            vertices: STAR vertices [6890, 3]

        Returns:
            Dictionary mapping clothing types to boolean masks
        """
        masks = {}

        for clothing_type, region in self.CLOTHING_REGIONS.items():
            mask = (vertices[:, 1] >= region['y_min']) & \
                   (vertices[:, 1] <= region['y_max'])
            masks[clothing_type] = mask

        return masks


def create_random_clothing_combination() -> List[str]:
    """
    Create a random clothing combination for diversity.

    Returns:
        List of clothing types (randomly selected combinations)
    """
    import random

    combinations = [
        ['shorts', 'sports_bra'],    # Athletic outfit
        ['leggings', 'sports_bra'],  # Yoga outfit
        ['shorts', 'tank_top'],      # Casual athletic
        ['leggings', 'tank_top'],    # Full coverage athletic
    ]

    return random.choice(combinations)


if __name__ == "__main__":
    """
    Test simple clothing generator.
    """
    print("\n" + "="*70)
    print("Simple Clothing Generator Test")
    print("="*70)

    # Create dummy STAR vertices (simplified for testing)
    print("\n1. Creating test vertices...")
    np.random.seed(42)
    test_vertices = np.random.randn(6890, 3) * 0.3
    test_vertices[:, 1] = np.linspace(-1.0, 1.0, 6890)  # Y from -1 to 1

    print(f"   Test vertices shape: {test_vertices.shape}")
    print(f"   Y range: [{test_vertices[:, 1].min():.2f}, {test_vertices[:, 1].max():.2f}]")

    # Initialize generator
    print("\n2. Initializing clothing generator...")
    clothing_gen = SimpleClothingGenerator(gender='neutral')

    # Show available clothing types
    print("\n3. Available clothing types:")
    for clothing_type, info in clothing_gen.get_clothing_info().items():
        print(f"   - {clothing_type}: {info['description']}")
        print(f"     Y range: [{info['y_min']:.2f}, {info['y_max']:.2f}]")
        print(f"     Expansion: {info['expansion']:.3f} ({(info['expansion']-1)*100:.1f}%)")

    # Test basic clothing addition
    print("\n4. Adding clothing (sports_bra + shorts)...")
    clothed_vertices = clothing_gen.add_clothing(
        test_vertices,
        clothing_types=['shorts', 'sports_bra']
    )

    # Calculate changes
    vertex_displacement = np.linalg.norm(clothed_vertices - test_vertices, axis=1)

    print(f"\n5. Results:")
    print(f"   Modified vertices: {(vertex_displacement > 0).sum()} / {len(test_vertices)}")
    print(f"   Max displacement: {vertex_displacement.max():.4f}")
    print(f"   Mean displacement: {vertex_displacement[vertex_displacement > 0].mean():.4f}")

    # Test smooth transition
    print("\n6. Testing smooth transition...")
    clothed_smooth = clothing_gen.add_clothing_with_smooth_transition(
        test_vertices,
        clothing_types=['shorts', 'sports_bra'],
        transition_width=0.08
    )

    smooth_displacement = np.linalg.norm(clothed_smooth - test_vertices, axis=1)
    print(f"   Smooth max displacement: {smooth_displacement.max():.4f}")

    # Visualize regions
    print("\n7. Visualizing regions...")
    masks = clothing_gen.visualize_regions(test_vertices)
    for clothing_type, mask in masks.items():
        print(f"   {clothing_type}: {mask.sum()} vertices")

    # Test random combinations
    print("\n8. Testing random clothing combinations...")
    for i in range(3):
        combo = create_random_clothing_combination()
        print(f"   Combination {i+1}: {combo}")

    print("\n" + "="*70)
    print("✓ Simple Clothing Generator test complete!")
    print("="*70)
