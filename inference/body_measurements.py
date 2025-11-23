"""
Body Measurement Calculation from STAR Mesh

Calculates key body measurements from 3D mesh:
- Height
- Shoulder width
- Chest circumference
- Waist circumference
- Hip circumference
- Inseam (leg length)
- Arm length
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy.spatial import ConvexHull


class BodyMeasurements:
    """Calculate body measurements from STAR mesh."""

    # Key vertex indices for measurements (approximations)
    # These are based on typical body landmarks in STAR model
    VERTEX_LANDMARKS = {
        'head_top': 411,  # Top of head
        'chin': 2878,  # Bottom of head
        'left_shoulder': 1228,
        'right_shoulder': 4883,
        'left_hip': 1330,
        'right_hip': 4929,
        'left_ankle': 3336,
        'right_ankle': 6662,
        'left_wrist': 2205,
        'right_wrist': 5572,
        'left_elbow': 1652,
        'right_elbow': 5022,
    }

    # Vertex regions for circumference measurements
    CHEST_REGION = list(range(1200, 1300)) + list(range(4800, 4900))  # Approximate chest region
    WAIST_REGION = list(range(1100, 1200)) + list(range(4700, 4800))  # Approximate waist region
    HIP_REGION = list(range(1300, 1400)) + list(range(4900, 5000))  # Approximate hip region

    def __init__(self):
        """Initialize measurement calculator."""
        pass

    def calculate_all(self, vertices, joints=None):
        """
        Calculate all body measurements.

        Args:
            vertices: Mesh vertices [N, 3] or [B, N, 3] (in meters)
            joints: Optional joint positions [24, 3] or [B, 24, 3] (in meters)

        Returns:
            Dictionary of measurements (in centimeters)
        """
        # Remove batch dimension if present
        if vertices.ndim == 3:
            vertices = vertices[0]
        if joints is not None and joints.ndim == 3:
            joints = joints[0]

        vertices = vertices.cpu().numpy() if torch.is_tensor(vertices) else vertices
        if joints is not None:
            joints = joints.cpu().numpy() if torch.is_tensor(joints) else joints

        measurements = {}

        # Height
        measurements['height'] = self.calculate_height(vertices)

        # Shoulder width
        measurements['shoulder_width'] = self.calculate_shoulder_width(vertices)

        # Circumferences
        measurements['chest_circumference'] = self.calculate_chest_circumference(vertices)
        measurements['waist_circumference'] = self.calculate_waist_circumference(vertices)
        measurements['hip_circumference'] = self.calculate_hip_circumference(vertices)

        # Limb lengths
        measurements['inseam'] = self.calculate_inseam(vertices)
        measurements['arm_length'] = self.calculate_arm_length(vertices)

        return measurements

    def calculate_height(self, vertices):
        """
        Calculate body height.

        Args:
            vertices: Mesh vertices [N, 3] in meters

        Returns:
            Height in centimeters
        """
        # Height is the Y-axis (vertical) range
        min_y = vertices[:, 1].min()
        max_y = vertices[:, 1].max()
        height_m = max_y - min_y
        return height_m * 100.0  # Convert to cm

    def calculate_shoulder_width(self, vertices):
        """
        Calculate shoulder width.

        Args:
            vertices: Mesh vertices [N, 3] in meters

        Returns:
            Shoulder width in centimeters
        """
        left_shoulder = vertices[self.VERTEX_LANDMARKS['left_shoulder']]
        right_shoulder = vertices[self.VERTEX_LANDMARKS['right_shoulder']]

        width_m = np.linalg.norm(left_shoulder - right_shoulder)
        return width_m * 100.0  # Convert to cm

    def calculate_circumference(self, vertices, vertex_indices, slice_axis=1, slice_pos=None):
        """
        Calculate circumference at a specific body cross-section.

        Args:
            vertices: Mesh vertices [N, 3] in meters
            vertex_indices: Indices of vertices in the region of interest
            slice_axis: Axis perpendicular to the slice plane (0=X, 1=Y, 2=Z)
            slice_pos: Position along slice_axis (default: mean position of region)

        Returns:
            Circumference in centimeters
        """
        # Get vertices in the region of interest
        region_vertices = vertices[vertex_indices]

        if slice_pos is None:
            slice_pos = region_vertices[:, slice_axis].mean()

        # Filter vertices near the slice plane
        tolerance = 0.02  # 2cm tolerance
        mask = np.abs(region_vertices[:, slice_axis] - slice_pos) < tolerance
        slice_vertices = region_vertices[mask]

        if len(slice_vertices) < 3:
            return 0.0

        # Get 2D coordinates (project onto slice plane)
        axes = [0, 1, 2]
        axes.remove(slice_axis)
        points_2d = slice_vertices[:, axes]

        # Compute convex hull to get perimeter points
        try:
            hull = ConvexHull(points_2d)
            hull_points = points_2d[hull.vertices]

            # Calculate perimeter
            perimeter = 0.0
            for i in range(len(hull_points)):
                p1 = hull_points[i]
                p2 = hull_points[(i + 1) % len(hull_points)]
                perimeter += np.linalg.norm(p2 - p1)

            return perimeter * 100.0  # Convert to cm

        except Exception:
            # Fallback: approximate as circle
            # Get radius as std dev of distances from center
            center = points_2d.mean(axis=0)
            radii = np.linalg.norm(points_2d - center, axis=1)
            avg_radius = radii.mean()
            circumference = 2 * np.pi * avg_radius
            return circumference * 100.0  # Convert to cm

    def calculate_chest_circumference(self, vertices):
        """Calculate chest circumference."""
        return self.calculate_circumference(vertices, self.CHEST_REGION, slice_axis=1)

    def calculate_waist_circumference(self, vertices):
        """Calculate waist circumference."""
        return self.calculate_circumference(vertices, self.WAIST_REGION, slice_axis=1)

    def calculate_hip_circumference(self, vertices):
        """Calculate hip circumference."""
        return self.calculate_circumference(vertices, self.HIP_REGION, slice_axis=1)

    def calculate_inseam(self, vertices):
        """
        Calculate inseam (inside leg length).

        Args:
            vertices: Mesh vertices [N, 3] in meters

        Returns:
            Inseam in centimeters
        """
        # Approximate as distance from hip to ankle
        left_hip = vertices[self.VERTEX_LANDMARKS['left_hip']]
        left_ankle = vertices[self.VERTEX_LANDMARKS['left_ankle']]

        inseam_m = np.linalg.norm(left_hip - left_ankle)
        return inseam_m * 100.0  # Convert to cm

    def calculate_arm_length(self, vertices):
        """
        Calculate arm length (shoulder to wrist).

        Args:
            vertices: Mesh vertices [N, 3] in meters

        Returns:
            Arm length in centimeters
        """
        left_shoulder = vertices[self.VERTEX_LANDMARKS['left_shoulder']]
        left_elbow = vertices[self.VERTEX_LANDMARKS['left_elbow']]
        left_wrist = vertices[self.VERTEX_LANDMARKS['left_wrist']]

        upper_arm = np.linalg.norm(left_shoulder - left_elbow)
        forearm = np.linalg.norm(left_elbow - left_wrist)
        arm_length_m = upper_arm + forearm

        return arm_length_m * 100.0  # Convert to cm


def test_measurements():
    """Test body measurements on a generated mesh."""
    from models.star_layer import STARLayer

    print("="*60)
    print("Testing Body Measurements")
    print("="*60)

    # Initialize STAR model
    print("\nInitializing STAR model...")
    star = STARLayer(gender='neutral', num_betas=10)

    # Generate a mesh with random shape
    beta = torch.randn(1, 10) * 0.5
    vertices, joints = star(beta, pose=None, trans=None)

    print(f"Generated mesh with β: {beta[0].numpy()}")
    print(f"Vertices shape: {vertices.shape}")

    # Calculate measurements
    print("\nCalculating body measurements...")
    calculator = BodyMeasurements()
    measurements = calculator.calculate_all(vertices, joints)

    # Print results
    print("\n" + "-"*60)
    print("Body Measurements (in cm)")
    print("-"*60)
    for name, value in measurements.items():
        print(f"  {name.replace('_', ' ').title()}: {value:.2f} cm")

    print("\n" + "="*60)
    print("✓ Measurement test completed!")
    print("="*60)


if __name__ == "__main__":
    test_measurements()
