"""
STAR Model Layer - PyTorch implementation
Based on: https://github.com/ahmedosman/STAR

STAR: Sparse Trained Articulated Human Body Regressor
A drop-in replacement for SMPL with improved deformations
"""
import torch
import torch.nn as nn
import numpy as np
import os
import pickle

class STARLayer(nn.Module):
    """
    STAR body model layer

    Args:
        gender: 'neutral', 'male', or 'female'
        num_betas: Number of shape parameters to use (default: 10, max: 300)
        model_path: Path to STAR .npz model file
    """
    def __init__(self, gender='neutral', num_betas=10, model_path=None):
        super(STARLayer, self).__init__()

        self.gender = gender
        self.num_betas = num_betas

        # Auto-detect model path if not provided
        if model_path is None:
            default_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'data', 'star_models',
                gender, 'model.npz'
            )
            if os.path.exists(default_path):
                model_path = default_path

        # Try to load official STAR model
        if model_path and os.path.exists(model_path):
            self._load_official_model(model_path)
        else:
            print("⚠️  Official STAR model not found. Using minimal template.")
            self._create_minimal_model()

    def _load_official_model(self, model_path):
        """Load official STAR model from .npz file"""
        print(f"Loading STAR model from {model_path}...")

        model_data = np.load(model_path, allow_pickle=True)

        # Template mesh (mean shape in T-pose)
        # Shape: (6890, 3) for STAR
        self.register_buffer('v_template',
                           torch.tensor(model_data['v_template'], dtype=torch.float32))

        # Shape blend shapes: (6890, 3, num_betas)
        # Each beta parameter deforms the mesh
        shapedirs = model_data['shapedirs'][:, :, :self.num_betas]
        self.register_buffer('shapedirs',
                           torch.tensor(shapedirs, dtype=torch.float32))

        # Joint regressor: (24, 6890)
        # Computes 3D joint positions from vertices
        J_regressor = model_data['J_regressor']
        # Handle both sparse and dense matrices
        if hasattr(J_regressor, 'toarray'):
            J_regressor = J_regressor.toarray()
        self.register_buffer('J_regressor',
                           torch.tensor(J_regressor, dtype=torch.float32))

        # Skinning weights: (6890, 24)
        # LBS weights for each vertex to each joint
        self.register_buffer('weights',
                           torch.tensor(model_data['weights'], dtype=torch.float32))

        # Pose blend shapes: (6890, 3, 207)
        # Corrective shapes for pose deformations
        if 'posedirs' in model_data:
            self.register_buffer('posedirs',
                               torch.tensor(model_data['posedirs'], dtype=torch.float32))

        # Faces: triangle mesh connectivity
        self.register_buffer('faces',
                           torch.tensor(model_data['f'].astype(np.int64)))

        self.num_joints = self.J_regressor.shape[0]
        self.num_vertices = self.v_template.shape[0]

        print(f"✓ Loaded STAR {self.gender} model:")
        print(f"  - Vertices: {self.num_vertices}")
        print(f"  - Joints: {self.num_joints}")
        print(f"  - Shape parameters: {self.num_betas}")

    def _create_minimal_model(self):
        """Create a minimal simplified human mesh for testing"""
        print("Creating minimal human mesh template...")

        # Simplified human mesh (cylinder-based approximation)
        # In production, replace with actual STAR model

        # Create a simple humanoid mesh
        # This is a VERY simplified version - just for testing structure
        vertices, faces = self._create_simple_humanoid()

        self.num_vertices = len(vertices)
        self.num_joints = 24  # STAR/SMPL standard

        # Template vertices
        self.register_buffer('v_template',
                           torch.tensor(vertices, dtype=torch.float32))

        # Random shape directions for demo
        shapedirs = np.random.randn(self.num_vertices, 3, self.num_betas) * 0.01
        self.register_buffer('shapedirs',
                           torch.tensor(shapedirs, dtype=torch.float32))

        # Simple joint regressor (random for demo)
        J_regressor = np.zeros((self.num_joints, self.num_vertices))
        # Place joints at specific vertices
        joint_indices = np.linspace(0, self.num_vertices-1, self.num_joints, dtype=int)
        for i, idx in enumerate(joint_indices):
            J_regressor[i, idx] = 1.0
        self.register_buffer('J_regressor',
                           torch.tensor(J_regressor, dtype=torch.float32))

        # LBS weights (simplified)
        weights = np.zeros((self.num_vertices, self.num_joints))
        for i in range(self.num_vertices):
            closest_joint = joint_indices[np.argmin(np.abs(joint_indices - i))]
            joint_idx = np.where(joint_indices == closest_joint)[0][0]
            weights[i, joint_idx] = 1.0
        self.register_buffer('weights',
                           torch.tensor(weights, dtype=torch.float32))

        # Faces
        self.register_buffer('faces',
                           torch.tensor(faces, dtype=torch.int64))

        print(f"✓ Created minimal template:")
        print(f"  - Vertices: {self.num_vertices}")
        print(f"  - Faces: {len(faces)}")
        print(f"  - Joints: {self.num_joints}")
        print(f"  ⚠️  This is a simplified model. Download official STAR for accurate results.")

    def _create_simple_humanoid(self):
        """Create a very simple humanoid mesh (cylinder-based)"""
        # This is a placeholder - creates a simple cylindrical mesh
        # In production, use actual STAR model

        n_rings = 20
        n_segments = 16
        height = 1.7  # meters
        radius = 0.15

        vertices = []

        for i in range(n_rings):
            y = (i / (n_rings - 1)) * height - height/2  # Center at origin
            r = radius * (1.0 - 0.3 * abs(2*i/(n_rings-1) - 1))  # Taper at ends

            for j in range(n_segments):
                angle = 2 * np.pi * j / n_segments
                x = r * np.cos(angle)
                z = r * np.sin(angle)
                vertices.append([x, y, z])

        vertices = np.array(vertices, dtype=np.float32)

        # Create faces
        faces = []
        for i in range(n_rings - 1):
            for j in range(n_segments):
                # Two triangles per quad
                v1 = i * n_segments + j
                v2 = i * n_segments + (j + 1) % n_segments
                v3 = (i + 1) * n_segments + j
                v4 = (i + 1) * n_segments + (j + 1) % n_segments

                faces.append([v1, v2, v3])
                faces.append([v2, v4, v3])

        faces = np.array(faces, dtype=np.int32)

        return vertices, faces

    def forward(self, betas, pose=None, trans=None):
        """
        Forward pass: generate mesh from parameters

        Args:
            betas: Shape parameters [batch_size, num_betas]
            pose: Pose parameters [batch_size, 72] (optional, defaults to T-pose)
            trans: Translation [batch_size, 3] (optional)

        Returns:
            vertices: [batch_size, num_vertices, 3]
            joints: [batch_size, num_joints, 3]
        """
        batch_size = betas.shape[0]

        # Apply shape blend shapes
        # v_shaped = v_template + sum(beta_i * shapedir_i)
        v_shaped = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        # Add shape deformations
        for i in range(min(betas.shape[1], self.num_betas)):
            # shapedirs: [num_vertices, 3, num_betas]
            # betas[:, i]: [batch_size]
            v_shaped = v_shaped + betas[:, i].view(batch_size, 1, 1) * self.shapedirs[:, :, i]

        # Compute joints
        # J: [batch_size, num_joints, 3]
        joints = torch.einsum('bvk,jv->bjk', v_shaped, self.J_regressor)

        # For now, skip pose deformations (T-pose only)
        # In full implementation, apply LBS with pose parameters
        vertices = v_shaped

        # Apply translation if provided
        if trans is not None:
            vertices = vertices + trans.view(batch_size, 1, 3)
            joints = joints + trans.view(batch_size, 1, 3)

        return vertices, joints

    def get_faces(self):
        """Return face indices"""
        return self.faces.cpu().numpy()


def test_star_layer():
    """Quick test of STAR layer"""
    print("\n" + "="*60)
    print("Testing STAR Layer")
    print("="*60 + "\n")

    # Create STAR layer
    star = STARLayer(gender='neutral', num_betas=10)

    # Random beta parameters
    batch_size = 1
    betas = torch.randn(batch_size, 10) * 0.5  # Small random variations

    print(f"\nInput beta parameters: {betas.shape}")
    print(f"Beta values: {betas[0].numpy()}")

    # Generate mesh
    vertices, joints = star(betas)

    print(f"\nOutput:")
    print(f"  - Vertices: {vertices.shape}")
    print(f"  - Joints: {joints.shape}")
    print(f"  - Vertex range: [{vertices.min().item():.3f}, {vertices.max().item():.3f}]")

    # Get faces
    faces = star.get_faces()
    print(f"  - Faces: {faces.shape}")

    return star, vertices, joints


if __name__ == "__main__":
    test_star_layer()
