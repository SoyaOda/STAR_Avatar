"""
MHR Model Generator Component
Handles 3D body generation with identity, pose, and expression parameters
"""
import torch
import numpy as np
from src.models.mhr_layer import MHRLayer


class MHRGenerator:
    """
    MHR-based 3D body generator

    Handles:
    - Loading MHR model
    - Generating bodies with random identity parameters
    - Converting to numpy for rendering

    Note: MHR outputs are in centimeters, so we convert to meters for consistency
    """

    # MHR outputs in centimeters, convert to meters
    SCALE_FACTOR = 0.01

    def __init__(self, num_identity=45, device='cpu'):
        """
        Initialize MHR generator

        Args:
            num_identity: Number of identity parameters to use (max: 45)
            device: torch device ('cpu' or 'cuda')
        """
        self.num_identity = min(num_identity, 45)
        self.device = device

        print("="*70)
        print("Initializing MHR Model")
        print("="*70)

        self.mhr_model = MHRLayer(device=device)

        print("  MHR model loaded")

    def generate_body(self, identity=None, pose=None, expression=None,
                      identity_std=0.5):
        """
        Generate 3D body with identity parameters

        Args:
            identity: Identity parameters [num_identity] or None for random
            pose: Pose parameters [204] or None for T-pose
            expression: Expression parameters [72] or None for neutral
            identity_std: Standard deviation for random identity params

        Returns:
            dict with:
                - vertices: numpy array [num_vertices, 3] in meters
                - faces: numpy array [num_faces, 3]
                - identity: identity parameters used
                - pose: pose parameters used
                - expression: expression parameters used
        """
        # Generate random identity if not provided
        if identity is None:
            identity = torch.randn(1, self.num_identity) * identity_std
        else:
            if isinstance(identity, np.ndarray):
                identity = torch.from_numpy(identity).float()
            if identity.dim() == 1:
                identity = identity.unsqueeze(0)

        # Handle pose
        if pose is not None:
            if isinstance(pose, np.ndarray):
                pose = torch.from_numpy(pose).float()
            if pose.dim() == 1:
                pose = pose.unsqueeze(0)

        # Handle expression
        if expression is not None:
            if isinstance(expression, np.ndarray):
                expression = torch.from_numpy(expression).float()
            if expression.dim() == 1:
                expression = expression.unsqueeze(0)

        # Generate mesh
        vertices_torch, skeleton_torch = self.mhr_model(
            identity=identity,
            pose=pose,
            expression=expression
        )

        # Convert to numpy and scale to meters
        vertices = vertices_torch[0].cpu().numpy() * self.SCALE_FACTOR

        # Center the mesh (MHR has feet at y=0, center vertically)
        vertices_centered = vertices.copy()
        y_min = vertices[:, 1].min()
        y_max = vertices[:, 1].max()
        height = y_max - y_min
        vertices_centered[:, 1] = vertices[:, 1] - y_min - height / 2

        faces = self.mhr_model.get_faces()

        # Prepare output parameters
        identity_np = identity[0].cpu().numpy() if identity.dim() > 1 else identity.cpu().numpy()
        pose_np = pose[0].cpu().numpy() if pose is not None else None
        expression_np = expression[0].cpu().numpy() if expression is not None else None

        return {
            'vertices': vertices_centered,
            'faces': faces,
            'identity': identity_np,
            'pose': pose_np,
            'expression': expression_np,
            'skeleton': skeleton_torch[0].cpu().numpy() if skeleton_torch is not None else None,
            'height_m': height  # Height in meters
        }

    def get_model_info(self):
        """Get information about loaded model"""
        return {
            'model_type': 'MHR',
            'num_vertices': self.mhr_model.num_vertices,
            'num_faces': self.mhr_model.num_faces,
            'num_identity': self.num_identity,
            'num_pose': MHRLayer.NUM_POSE,
            'num_expression': MHRLayer.NUM_EXPRESSION
        }


if __name__ == "__main__":
    # Test
    print("\nTesting MHRGenerator...")

    generator = MHRGenerator(num_identity=45)

    print("\nGenerating random body...")
    body = generator.generate_body(identity_std=0.8)

    print(f"\nGenerated body:")
    print(f"  Vertices: {body['vertices'].shape}")
    print(f"  Faces: {body['faces'].shape}")
    print(f"  Identity range: [{body['identity'].min():.3f}, {body['identity'].max():.3f}]")
    print(f"  Height: {body['height_m']:.2f}m")
    print(f"  Vertex range Y: [{body['vertices'][:, 1].min():.3f}, {body['vertices'][:, 1].max():.3f}]")

    print("\n  MHRGenerator test complete")
