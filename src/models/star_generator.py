"""
STAR Model Generator Component
Handles 3D body generation with shape parameters
"""
import torch
import numpy as np
from src.models.star_layer import STARLayer


class STARGenerator:
    """
    STAR-based 3D body generator

    Handles:
    - Loading STAR model
    - Generating bodies with random shape parameters
    - Converting to numpy for rendering
    """

    def __init__(self, gender='neutral', num_betas=10):
        """
        Initialize STAR generator

        Args:
            gender: 'neutral', 'male', or 'female'
            num_betas: Number of shape parameters (default: 10)
        """
        self.gender = gender
        self.num_betas = num_betas

        print("="*70)
        print("Initializing STAR Model")
        print("="*70)

        self.star_model = STARLayer(gender=gender, num_betas=num_betas)

        print("✓ STAR model loaded")

    def generate_body(self, betas=None, beta_std=0.5):
        """
        Generate 3D body with shape parameters

        Args:
            betas: Shape parameters [num_betas] or None for random
            beta_std: Standard deviation for random betas

        Returns:
            dict with:
                - vertices: numpy array [num_vertices, 3]
                - faces: numpy array [num_faces, 3]
                - betas: shape parameters used
        """
        # Generate random betas if not provided
        if betas is None:
            betas = torch.randn(1, self.num_betas) * beta_std
        else:
            if isinstance(betas, np.ndarray):
                betas = torch.from_numpy(betas).float()
            if betas.dim() == 1:
                betas = betas.unsqueeze(0)

        # Generate mesh
        vertices_torch, joints_torch = self.star_model(betas)

        # Convert to numpy
        vertices = vertices_torch[0].cpu().numpy()
        faces = self.star_model.get_faces()
        betas_np = betas[0].cpu().numpy()

        return {
            'vertices': vertices,
            'faces': faces,
            'betas': betas_np,
            'joints': joints_torch[0].cpu().numpy()
        }

    def get_model_info(self):
        """Get information about loaded model"""
        return {
            'gender': self.gender,
            'num_vertices': self.star_model.num_vertices,
            'num_joints': self.star_model.num_joints,
            'num_betas': self.num_betas
        }


if __name__ == "__main__":
    # Test
    print("\nTesting STARGenerator...")

    generator = STARGenerator(gender='neutral', num_betas=10)

    print("\nGenerating random body...")
    body = generator.generate_body(beta_std=0.5)

    print(f"\nGenerated body:")
    print(f"  Vertices: {body['vertices'].shape}")
    print(f"  Faces: {body['faces'].shape}")
    print(f"  Beta range: [{body['betas'].min():.3f}, {body['betas'].max():.3f}]")

    print("\n✓ STARGenerator test complete")
