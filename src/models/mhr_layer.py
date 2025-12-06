"""
MHR Model Layer - TorchScript-based implementation
Based on: https://github.com/facebookresearch/MHR

MHR: Momentum Human Rig
An anatomically-inspired parametric whole-body digital human model
"""
import torch
import torch.nn as nn
import numpy as np
import os


class MHRLayer(nn.Module):
    """
    MHR body model layer using TorchScript

    Args:
        model_path: Path to mhr_model.pt TorchScript file
        mesh_path: Path to mesh data (.npz with faces)
        device: torch device ('cpu' or 'cuda')
    """

    # Parameter dimensions
    NUM_IDENTITY = 45      # Shape/identity parameters
    NUM_POSE = 204         # Pose parameters
    NUM_EXPRESSION = 72    # Facial expression parameters

    def __init__(self, model_path=None, mesh_path=None, device='cpu'):
        super(MHRLayer, self).__init__()

        self.device = device

        # Auto-detect paths if not provided
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'data', 'mhr_models',
                'assets', 'mhr_model.pt'
            )

        if mesh_path is None:
            mesh_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'data', 'mhr_models',
                'mhr_mesh_lod1_fixed.npz'
            )

        # Load TorchScript model
        if os.path.exists(model_path):
            self._load_model(model_path)
        else:
            raise FileNotFoundError(
                f"MHR model not found at {model_path}. "
                "Download from https://github.com/facebookresearch/MHR/releases"
            )

        # Load mesh topology (faces)
        if os.path.exists(mesh_path):
            self._load_mesh(mesh_path)
        else:
            raise FileNotFoundError(
                f"MHR mesh data not found at {mesh_path}. "
                "Run mesh extraction script first."
            )

    def _load_model(self, model_path):
        """Load TorchScript MHR model"""
        print(f"Loading MHR model from {model_path}...")

        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        # Get model info
        self.num_identity = self.model.get_num_identity_blendshapes()
        self.num_expression = self.model.get_num_face_expression_blendshapes()

        # Run test to get vertex count
        with torch.no_grad():
            test_identity = torch.zeros(1, self.num_identity)
            test_pose = torch.zeros(1, self.NUM_POSE)
            test_expr = torch.zeros(1, self.num_expression)
            vertices, _ = self.model(test_identity, test_pose, test_expr)
            self.num_vertices = vertices.shape[1]

        print(f"  Loaded MHR model:")
        print(f"  - Vertices: {self.num_vertices}")
        print(f"  - Identity params: {self.num_identity}")
        print(f"  - Pose params: {self.NUM_POSE}")
        print(f"  - Expression params: {self.num_expression}")

    def _load_mesh(self, mesh_path):
        """Load mesh topology data"""
        print(f"Loading mesh topology from {mesh_path}...")

        mesh_data = np.load(mesh_path)
        self.register_buffer('faces',
                           torch.tensor(mesh_data['faces'], dtype=torch.int64))

        self.num_faces = len(self.faces)
        print(f"  Faces: {self.num_faces}")

    def forward(self, identity=None, pose=None, expression=None,
                apply_correctives=True):
        """
        Forward pass: generate mesh from parameters

        Args:
            identity: Identity/shape parameters [batch_size, 45] or None for average
            pose: Pose parameters [batch_size, 204] or None for T-pose
            expression: Expression parameters [batch_size, 72] or None for neutral
            apply_correctives: Apply pose correctives (default: True)

        Returns:
            vertices: [batch_size, num_vertices, 3]
            skeleton_state: [batch_size, num_joints, 8]
        """
        # Determine batch size
        if identity is not None:
            batch_size = identity.shape[0]
        elif pose is not None:
            batch_size = pose.shape[0]
        elif expression is not None:
            batch_size = expression.shape[0]
        else:
            batch_size = 1

        # Default parameters
        if identity is None:
            identity = torch.zeros(batch_size, self.num_identity, device=self.device)
        if pose is None:
            pose = torch.zeros(batch_size, self.NUM_POSE, device=self.device)
        if expression is None:
            expression = torch.zeros(batch_size, self.num_expression, device=self.device)

        # Ensure correct device
        identity = identity.to(self.device)
        pose = pose.to(self.device)
        expression = expression.to(self.device)

        # Run MHR model
        with torch.no_grad():
            vertices, skeleton_state = self.model(
                identity, pose, expression, apply_correctives
            )

        return vertices, skeleton_state

    def get_faces(self):
        """Return face indices as numpy array"""
        return self.faces.cpu().numpy()

    def get_template_vertices(self):
        """Get average body vertices (identity=0, pose=0)"""
        vertices, _ = self.forward()
        return vertices[0].cpu().numpy()


def test_mhr_layer():
    """Quick test of MHR layer"""
    print("\n" + "="*60)
    print("Testing MHR Layer")
    print("="*60 + "\n")

    # Create MHR layer
    mhr = MHRLayer(device='cpu')

    # Random identity parameters
    batch_size = 1
    identity = torch.randn(batch_size, 45) * 0.5

    print(f"\nInput identity parameters: {identity.shape}")
    print(f"Identity values range: [{identity.min():.3f}, {identity.max():.3f}]")

    # Generate mesh
    vertices, skeleton = mhr(identity=identity)

    print(f"\nOutput:")
    print(f"  - Vertices: {vertices.shape}")
    print(f"  - Skeleton: {skeleton.shape}")
    print(f"  - Vertex range: [{vertices.min().item():.3f}, {vertices.max().item():.3f}]")

    # Get faces
    faces = mhr.get_faces()
    print(f"  - Faces: {faces.shape}")

    return mhr, vertices, skeleton


if __name__ == "__main__":
    test_mhr_layer()
