"""
Body Smoothing Utilities for STAR Model

Provides utilities to smooth specific body regions, particularly
the genital area, for creating neutral/anatomically simplified models.

Techniques:
1. Laplacian smoothing on selected vertices
2. Vertex displacement reduction
3. Region-based mesh smoothing
"""

import numpy as np
from typing import Optional, Tuple
from scipy import sparse
from scipy.sparse.linalg import spsolve


class BodySmoother:
    """
    Smooth specific regions of STAR body mesh.

    Usage:
        smoother = BodySmoother()
        smoothed_verts = smoother.smooth_genital_region(vertices, faces)
    """

    def __init__(self):
        """Initialize body smoother."""
        pass

    def identify_genital_region(
        self,
        vertices: np.ndarray,
        method: str = 'geometric'
    ) -> np.ndarray:
        """
        Identify vertices in genital region.

        Args:
            vertices: Vertex positions [N, 3]
            method: 'geometric' (coordinate-based) or 'anatomical' (preset indices)

        Returns:
            Boolean mask [N] for genital region vertices
        """
        if method == 'geometric':
            # Identify based on anatomical position
            # Genital region approximately:
            # - Y: -0.3 to -0.6 (below waist, above thighs)
            # - X: -0.08 to 0.08 (centered)
            # - Z: Front protrusion (positive Z)

            y_mask = (vertices[:, 1] >= -0.65) & (vertices[:, 1] <= -0.25)
            x_mask = (vertices[:, 0] >= -0.10) & (vertices[:, 0] <= 0.10)
            z_mask = vertices[:, 2] > 0.0  # Front side only

            mask = y_mask & x_mask & z_mask

            return mask
        else:
            # TODO: Use preset vertex indices if available
            raise NotImplementedError("Anatomical method not implemented")

    def compute_laplacian_matrix(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> sparse.csr_matrix:
        """
        Compute Laplacian matrix for mesh smoothing.

        Args:
            vertices: Vertex positions [N, 3]
            faces: Face indices [F, 3]

        Returns:
            Laplacian matrix (sparse)
        """
        num_verts = len(vertices)

        # Build adjacency information
        edges = set()
        for face in faces:
            edges.add(tuple(sorted([face[0], face[1]])))
            edges.add(tuple(sorted([face[1], face[2]])))
            edges.add(tuple(sorted([face[2], face[0]])))

        # Count neighbors for each vertex
        neighbor_count = np.zeros(num_verts)
        row_indices = []
        col_indices = []
        data = []

        for v1, v2 in edges:
            neighbor_count[v1] += 1
            neighbor_count[v2] += 1

            # Add off-diagonal entries
            row_indices.extend([v1, v2])
            col_indices.extend([v2, v1])
            data.extend([-1, -1])

        # Add diagonal entries
        for i in range(num_verts):
            row_indices.append(i)
            col_indices.append(i)
            data.append(neighbor_count[i])

        # Create sparse matrix
        laplacian = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(num_verts, num_verts)
        )

        return laplacian

    def smooth_region_laplacian(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        mask: np.ndarray,
        iterations: int = 5,
        lambda_smooth: float = 0.5
    ) -> np.ndarray:
        """
        Apply Laplacian smoothing to masked region.

        Args:
            vertices: Vertex positions [N, 3]
            faces: Face indices [F, 3]
            mask: Boolean mask for region to smooth [N]
            iterations: Number of smoothing iterations
            lambda_smooth: Smoothing factor (0-1, higher = more smoothing)

        Returns:
            Smoothed vertices [N, 3]
        """
        smoothed = vertices.copy()

        # Simple iterative Laplacian smoothing
        for _ in range(iterations):
            # Build neighbor averaging
            new_positions = smoothed.copy()

            for face in faces:
                v0, v1, v2 = face

                # Only smooth if vertex is in mask
                if mask[v0]:
                    new_positions[v0] += (smoothed[v1] + smoothed[v2]) * lambda_smooth / 2
                if mask[v1]:
                    new_positions[v1] += (smoothed[v0] + smoothed[v2]) * lambda_smooth / 2
                if mask[v2]:
                    new_positions[v2] += (smoothed[v0] + smoothed[v1]) * lambda_smooth / 2

            # Normalize by number of neighbors
            smoothed[mask] = new_positions[mask]

        return smoothed

    def flatten_region(
        self,
        vertices: np.ndarray,
        mask: np.ndarray,
        axis: int = 2,  # Z-axis (front-back)
        flatten_amount: float = 0.7
    ) -> np.ndarray:
        """
        Flatten region along specified axis.

        Args:
            vertices: Vertex positions [N, 3]
            mask: Boolean mask for region to flatten [N]
            axis: Axis to flatten (0=X, 1=Y, 2=Z)
            flatten_amount: Amount to flatten (0=none, 1=complete)

        Returns:
            Flattened vertices [N, 3]
        """
        flattened = vertices.copy()

        # Get median value along axis for reference
        median_val = np.median(vertices[mask, axis])

        # Move vertices toward median
        flattened[mask, axis] = (
            vertices[mask, axis] * (1 - flatten_amount) +
            median_val * flatten_amount
        )

        return flattened

    def smooth_genital_region(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        method: str = 'combined',
        smoothing_iterations: int = 5,
        flatten_amount: float = 0.6
    ) -> np.ndarray:
        """
        Smooth genital region using combined techniques.

        Args:
            vertices: Vertex positions [N, 3]
            faces: Face indices [F, 3]
            method: 'laplacian', 'flatten', or 'combined'
            smoothing_iterations: Number of Laplacian smoothing iterations
            flatten_amount: Amount of flattening to apply

        Returns:
            Smoothed vertices [N, 3]
        """
        # Identify genital region
        mask = self.identify_genital_region(vertices)

        print(f"Smoothing genital region: {mask.sum()} vertices")

        if method == 'laplacian':
            # Laplacian smoothing only
            smoothed = self.smooth_region_laplacian(
                vertices, faces, mask,
                iterations=smoothing_iterations
            )

        elif method == 'flatten':
            # Flattening only
            smoothed = self.flatten_region(
                vertices, mask,
                axis=2,  # Z-axis (front-back)
                flatten_amount=flatten_amount
            )

        elif method == 'combined':
            # First flatten, then smooth for natural transition
            smoothed = self.flatten_region(
                vertices, mask,
                axis=2,
                flatten_amount=flatten_amount
            )
            smoothed = self.smooth_region_laplacian(
                smoothed, faces, mask,
                iterations=smoothing_iterations,
                lambda_smooth=0.3
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        return smoothed


def test_body_smoother():
    """Test body smoother with STAR model."""
    import os

    print("\n" + "="*70)
    print("Body Smoother Test")
    print("="*70)

    # Load STAR model
    star_path = "data/star_models/neutral/model.npz"

    if not os.path.exists(star_path):
        print(f"❌ STAR model not found: {star_path}")
        return

    star_data = np.load(star_path, allow_pickle=True)
    v_template = star_data['v_template']
    faces = star_data['f']

    print(f"\nLoaded STAR model: {len(v_template)} vertices, {len(faces)} faces")

    # Initialize smoother
    smoother = BodySmoother()

    # Test genital region identification
    mask = smoother.identify_genital_region(v_template)
    print(f"\nIdentified genital region: {mask.sum()} vertices")
    print(f"  Y range: [{v_template[mask, 1].min():.3f}, {v_template[mask, 1].max():.3f}]")
    print(f"  X range: [{v_template[mask, 0].min():.3f}, {v_template[mask, 0].max():.3f}]")
    print(f"  Z range: [{v_template[mask, 2].min():.3f}, {v_template[mask, 2].max():.3f}]")

    # Test smoothing methods
    methods = ['flatten', 'laplacian', 'combined']

    for method in methods:
        print(f"\nTesting method: {method}")
        smoothed = smoother.smooth_genital_region(
            v_template.copy(),
            faces,
            method=method
        )

        # Calculate displacement
        displacement = np.linalg.norm(smoothed - v_template, axis=1)
        print(f"  Max displacement: {displacement.max():.4f}")
        print(f"  Mean displacement (region): {displacement[mask].mean():.4f}")

        # Save for visual inspection
        output_dir = "outputs/body_smoothing_test"
        os.makedirs(output_dir, exist_ok=True)

        obj_path = os.path.join(output_dir, f"smoothed_{method}.obj")

        with open(obj_path, 'w') as f:
            for v in smoothed:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        print(f"  ✓ Saved: {obj_path}")

    print("\n" + "="*70)
    print("✓ Body Smoother Test Complete!")
    print("="*70)
    print(f"\nOutput directory: {os.path.abspath('outputs/body_smoothing_test')}")


if __name__ == "__main__":
    test_body_smoother()
