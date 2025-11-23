"""
Advanced Clothing Generator for STAR Body Models

Implements more sophisticated clothing generation based on:
- Normal-based offset (shell generation)
- Proper region masking
- Various pants styles (leggings, shorts, cargo pants, etc.)
- Edge handling for hems

Based on research from SMPLicit, TailorNet, and CLOTH3D approaches.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class AdvancedClothingGenerator:
    """
    Advanced clothing generator using shell generation and normal-based offset.

    This approach:
    1. Masks specific body regions (legs, torso, etc.)
    2. Extracts those regions as separate clothing mesh
    3. Offsets vertices along surface normals (4mm recommended)
    4. Handles edge cases (waistband, hems, cuffs)

    Usage:
        clothing_gen = AdvancedClothingGenerator()
        clothing_verts, clothing_faces = clothing_gen.generate_pants(
            vertices, faces, style='leggings'
        )
    """

    # Pants style definitions
    PANTS_STYLES = {
        'leggings': {
            'name': 'Tight Leggings',
            'y_min': -1.0,      # From feet
            'y_max': -0.05,     # To waist
            'offset': 0.004,    # 4mm offset (research recommendation)
            'tightness': 'tight',
            'description': 'Full-length tight leggings'
        },
        'shorts': {
            'name': 'Athletic Shorts',
            'y_min': -0.60,     # Mid-thigh
            'y_max': -0.05,     # To waist
            'offset': 0.005,    # Slightly looser
            'tightness': 'loose',
            'description': 'Athletic shorts'
        },
        'capri': {
            'name': 'Capri Pants',
            'y_min': -0.80,     # Below knee
            'y_max': -0.05,     # To waist
            'offset': 0.004,
            'tightness': 'tight',
            'description': 'Capri-length pants (below knee)'
        },
        'bermuda': {
            'name': 'Bermuda Shorts',
            'y_min': -0.45,     # Just above knee
            'y_max': -0.05,     # To waist
            'offset': 0.006,    # Looser fit
            'tightness': 'loose',
            'description': 'Bermuda shorts (knee-length)'
        },
        'joggers': {
            'name': 'Jogger Pants',
            'y_min': -1.0,      # Full length
            'y_max': -0.05,     # To waist
            'offset': 0.008,    # Much looser
            'tightness': 'loose',
            'description': 'Loose jogger-style pants'
        },
        'cargo_shorts': {
            'name': 'Cargo Shorts',
            'y_min': -0.55,     # Below mid-thigh
            'y_max': -0.05,     # To waist
            'offset': 0.007,    # Loose fit
            'tightness': 'loose',
            'description': 'Cargo shorts with room'
        }
    }

    def __init__(self):
        """Initialize advanced clothing generator."""
        pass

    def compute_vertex_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        Compute per-vertex normals from face normals.

        Args:
            vertices: Vertex positions [N, 3]
            faces: Face indices [F, 3]

        Returns:
            Vertex normals [N, 3] (normalized)
        """
        # Initialize vertex normals
        vertex_normals = np.zeros_like(vertices)

        # Compute face normals
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        # Cross product for face normals
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normals = np.cross(edge1, edge2)

        # Normalize face normals
        face_norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = face_normals / (face_norms + 1e-8)

        # Accumulate face normals to vertices
        for i, face in enumerate(faces):
            for vertex_idx in face:
                vertex_normals[vertex_idx] += face_normals[i]

        # Normalize vertex normals
        vertex_norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vertex_normals = vertex_normals / (vertex_norms + 1e-8)

        return vertex_normals

    def create_region_mask(
        self,
        vertices: np.ndarray,
        y_min: float,
        y_max: float,
        x_range: Optional[Tuple[float, float]] = None,
        z_range: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Create boolean mask for vertices in specified region.

        Args:
            vertices: Vertex positions [N, 3]
            y_min: Minimum Y coordinate
            y_max: Maximum Y coordinate
            x_range: Optional (x_min, x_max) range
            z_range: Optional (z_min, z_max) range

        Returns:
            Boolean mask [N]
        """
        mask = (vertices[:, 1] >= y_min) & (vertices[:, 1] <= y_max)

        if x_range is not None:
            mask &= (vertices[:, 0] >= x_range[0]) & (vertices[:, 0] <= x_range[1])

        if z_range is not None:
            mask &= (vertices[:, 2] >= z_range[0]) & (vertices[:, 2] <= z_range[1])

        return mask

    def offset_vertices_normal(
        self,
        vertices: np.ndarray,
        normals: np.ndarray,
        offset: float,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Offset vertices along surface normals.

        Args:
            vertices: Vertex positions [N, 3]
            normals: Vertex normals [N, 3]
            offset: Offset distance (meters, e.g., 0.004 for 4mm)
            mask: Optional boolean mask [N] (if None, offset all)

        Returns:
            Offset vertices [N, 3]
        """
        offset_verts = vertices.copy()

        if mask is None:
            mask = np.ones(len(vertices), dtype=bool)

        # Offset along normals
        offset_verts[mask] = vertices[mask] + normals[mask] * offset

        return offset_verts

    def smooth_hem_transition(
        self,
        vertices: np.ndarray,
        mask: np.ndarray,
        y_hem: float,
        transition_height: float = 0.05
    ) -> np.ndarray:
        """
        Smooth transition at hem (bottom edge of pants).

        Args:
            vertices: Vertex positions [N, 3]
            mask: Clothing region mask [N]
            y_hem: Y coordinate of hem
            transition_height: Height of smooth transition zone

        Returns:
            Updated mask with smooth transition [N]
        """
        # Find vertices near hem
        y_coords = vertices[:, 1]

        # Smooth transition zone
        transition_mask = (y_coords >= y_hem) & (y_coords <= y_hem + transition_height)

        # Blend weight (0 at hem, 1 at top of transition)
        blend_weights = np.zeros(len(vertices))
        blend_weights[transition_mask] = (
            (y_coords[transition_mask] - y_hem) / transition_height
        )

        return blend_weights

    def generate_pants(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        style: str = 'leggings',
        custom_offset: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate pants clothing mesh.

        Args:
            vertices: STAR body vertices [N, 3]
            faces: STAR body faces [F, 3]
            style: Pants style ('leggings', 'shorts', etc.)
            custom_offset: Optional custom offset distance

        Returns:
            (clothing_vertices, clothing_faces)
        """
        if style not in self.PANTS_STYLES:
            raise ValueError(f"Unknown style: {style}. Available: {list(self.PANTS_STYLES.keys())}")

        style_config = self.PANTS_STYLES[style]
        offset = custom_offset if custom_offset is not None else style_config['offset']

        print(f"\n  Generating: {style_config['name']}")
        print(f"    Y range: [{style_config['y_min']:.2f}, {style_config['y_max']:.2f}]")
        print(f"    Offset: {offset*1000:.1f}mm")
        print(f"    Tightness: {style_config['tightness']}")

        # 1. Compute vertex normals
        normals = self.compute_vertex_normals(vertices, faces)

        # 2. Create region mask for pants area
        pants_mask = self.create_region_mask(
            vertices,
            y_min=style_config['y_min'],
            y_max=style_config['y_max']
        )

        print(f"    Masked vertices: {pants_mask.sum()} / {len(vertices)}")

        # 3. Offset vertices along normals
        clothed_verts = self.offset_vertices_normal(
            vertices,
            normals,
            offset,
            mask=pants_mask
        )

        # 4. Extract clothing region as separate mesh
        # Find faces where at least 2 vertices are in clothing region
        faces_vertex_count = np.sum(pants_mask[faces], axis=1)
        clothing_faces_mask = faces_vertex_count >= 2
        clothing_faces = faces[clothing_faces_mask]

        # Get unique vertices used in clothing faces
        clothing_vertex_indices = np.unique(clothing_faces.flatten())
        clothing_vertices = clothed_verts[clothing_vertex_indices]

        # Remap face indices
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(clothing_vertex_indices)}
        clothing_faces_remapped = np.array([
            [vertex_map[face[0]], vertex_map[face[1]], vertex_map[face[2]]]
            for face in clothing_faces
        ])

        print(f"    Clothing mesh: {len(clothing_vertices)} verts, {len(clothing_faces_remapped)} faces")

        return clothing_vertices, clothing_faces_remapped

    def generate_combined_outfit(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        pants_style: str = 'leggings',
        top_style: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate combined outfit (pants + optional top).

        Args:
            vertices: STAR body vertices [N, 3]
            faces: STAR body faces [F, 3]
            pants_style: Pants style
            top_style: Optional top style (if None, pants only)

        Returns:
            (clothing_vertices, clothing_faces)
        """
        # For now, just generate pants
        # TODO: Add top generation
        return self.generate_pants(vertices, faces, style=pants_style)

    def get_available_styles(self) -> Dict[str, Dict]:
        """Get all available pants styles."""
        return self.PANTS_STYLES.copy()


def write_obj(filepath: str, vertices: np.ndarray, faces: np.ndarray):
    """Write mesh to OBJ file."""
    with open(filepath, 'w') as f:
        f.write("# OBJ file generated by Advanced Clothing Generator\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n\n")

        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")

        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


if __name__ == "__main__":
    """Test advanced clothing generator."""
    print("\n" + "="*70)
    print("Advanced Clothing Generator Test")
    print("="*70)

    # Load STAR model
    import os
    star_path = "data/star_models/neutral/model.npz"

    if not os.path.exists(star_path):
        print(f"❌ STAR model not found: {star_path}")
        exit(1)

    star_data = np.load(star_path, allow_pickle=True)
    v_template = star_data['v_template']
    faces = star_data['f']

    print(f"\nLoaded STAR model: {len(v_template)} vertices, {len(faces)} faces")

    # Initialize generator
    clothing_gen = AdvancedClothingGenerator()

    # Show available styles
    print("\n" + "="*70)
    print("Available Pants Styles:")
    print("="*70)
    for style, config in clothing_gen.get_available_styles().items():
        print(f"\n  {style}:")
        print(f"    Name: {config['name']}")
        print(f"    Description: {config['description']}")
        print(f"    Offset: {config['offset']*1000:.1f}mm")
        print(f"    Tightness: {config['tightness']}")

    # Test each style
    print("\n" + "="*70)
    print("Testing All Styles:")
    print("="*70)

    output_dir = "outputs/advanced_clothing_test"
    os.makedirs(output_dir, exist_ok=True)

    for style in ['leggings', 'shorts', 'capri', 'bermuda', 'joggers', 'cargo_shorts']:
        print(f"\n{'='*70}")
        print(f"Style: {style}")
        print(f"{'='*70}")

        clothing_verts, clothing_faces = clothing_gen.generate_pants(
            v_template.copy(),
            faces,
            style=style
        )

        # Save OBJ
        obj_path = os.path.join(output_dir, f"{style}.obj")
        write_obj(obj_path, clothing_verts, clothing_faces)
        print(f"    ✓ Saved: {obj_path}")

    # Also save body for comparison
    body_path = os.path.join(output_dir, "body.obj")
    write_obj(body_path, v_template, faces)
    print(f"\n✓ Saved body: {body_path}")

    print("\n" + "="*70)
    print("✓ Advanced Clothing Generator Test Complete!")
    print("="*70)
    print(f"\nOutput directory: {os.path.abspath(output_dir)}")
