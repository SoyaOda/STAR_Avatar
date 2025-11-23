"""
Inference Script for Shape Estimation

Predicts body shape (β) and global translation (T) from input images,
then generates the 3D body mesh using the STAR model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
from pathlib import Path

from models.shape_estimator import ShapeEstimator
from models.star_layer import STARLayer
from data.synthetic_dataset import SyntheticDataset


class ShapePredictor:
    """Shape prediction and 3D mesh generation."""

    def __init__(self, checkpoint_path, device='cpu'):
        """
        Initialize predictor.

        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Get model arguments from checkpoint
        args = checkpoint.get('args', {})
        num_betas = args.get('num_betas', 10)
        num_joints = args.get('num_joints', 16)
        attr_dim = args.get('attr_dim', 3)

        # Initialize model
        print("Initializing ShapeEstimator...")
        self.model = ShapeEstimator(
            num_betas=num_betas,
            num_joints=num_joints,
            attr_dim=attr_dim,
            use_pretrained=False  # Not needed for inference
        ).to(self.device)

        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded successfully (trained for {checkpoint['epoch']} epochs)")

        # Initialize STAR model for mesh generation
        print("Initializing STAR model...")
        self.star_model = STARLayer(
            gender='neutral',
            num_betas=num_betas
        ).to(self.device)

        print("Predictor ready!")

    @torch.no_grad()
    def predict(self, front_input, back_input, attr_input=None):
        """
        Predict shape parameters and translation.

        Args:
            front_input: Front view tensor [B, 21, H, W] or [21, H, W]
            back_input: Back view tensor [B, 21, H, W] or [21, H, W]
            attr_input: Optional user attributes [B, 3] or [3]

        Returns:
            dict with keys:
                - 'beta': Predicted shape parameters [B, num_betas]
                - 'T': Predicted global translation [B, 3]
                - 'vertices': 3D mesh vertices [B, 6890, 3]
                - 'joints': 3D joint positions [B, 24, 3]
        """
        # Ensure batch dimension
        if front_input.ndim == 3:
            front_input = front_input.unsqueeze(0)
        if back_input.ndim == 3:
            back_input = back_input.unsqueeze(0)
        if attr_input is not None and attr_input.ndim == 1:
            attr_input = attr_input.unsqueeze(0)

        # Move to device
        front_input = front_input.to(self.device)
        back_input = back_input.to(self.device)
        if attr_input is not None:
            attr_input = attr_input.to(self.device)

        # Predict beta and T
        beta_pred, T_pred = self.model(front_input, back_input, attr_input)

        # Generate 3D mesh
        vertices, joints = self.star_model(beta_pred, pose=None, trans=None)

        return {
            'beta': beta_pred.cpu(),
            'T': T_pred.cpu(),
            'vertices': vertices.cpu(),
            'joints': joints.cpu()
        }

    def predict_from_sample(self, sample):
        """
        Predict from a dataset sample.

        Args:
            sample: Dictionary from SyntheticDataset

        Returns:
            Prediction dictionary with additional ground truth
        """
        front_input = sample['front_input']
        back_input = sample['back_input']
        attr_input = sample.get('attr_input', None)

        # Get prediction
        pred = self.predict(front_input, back_input, attr_input)

        # Add ground truth if available
        if 'beta_gt' in sample:
            pred['beta_gt'] = sample['beta_gt']
        if 'T_gt' in sample:
            pred['T_gt'] = sample['T_gt']

        return pred

    def save_mesh(self, vertices, faces, output_path):
        """
        Save mesh to OBJ file.

        Args:
            vertices: Vertex positions [N, 3] or [B, N, 3]
            faces: Face indices [F, 3]
            output_path: Output OBJ file path
        """
        # Remove batch dimension if present
        if vertices.ndim == 3:
            vertices = vertices[0]

        vertices = vertices.cpu().numpy()
        faces = faces.cpu().numpy() if torch.is_tensor(faces) else faces

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        print(f"Mesh saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Inference for Shape Estimation')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='outputs/synthetic_data',
                        help='Path to test data directory')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Index of sample to test (default: 0)')
    parser.add_argument('--output_dir', type=str, default='outputs/predictions',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to run inference on')
    parser.add_argument('--save_mesh', action='store_true',
                        help='Save predicted mesh to OBJ file')

    args = parser.parse_args()

    # Initialize predictor
    predictor = ShapePredictor(args.checkpoint, device=args.device)

    # Load test data
    print(f"\nLoading test data from {args.data_dir}...")
    dataset = SyntheticDataset(
        data_dir=args.data_dir,
        transform=None,
        use_attributes=True
    )

    print(f"Total samples: {len(dataset)}")

    if args.sample_idx >= len(dataset):
        print(f"Error: sample_idx {args.sample_idx} >= dataset size {len(dataset)}")
        return

    # Get sample
    sample = dataset[args.sample_idx]
    print(f"\nProcessing sample {args.sample_idx}...")

    # Predict
    result = predictor.predict_from_sample(sample)

    # Print results
    print("\n" + "="*60)
    print("Prediction Results")
    print("="*60)

    print(f"\nPredicted β (shape parameters):")
    print(f"  {result['beta'][0].numpy()}")

    print(f"\nPredicted T (global translation):")
    print(f"  Tx: {result['T'][0, 0].item():.4f}")
    print(f"  Ty: {result['T'][0, 1].item():.4f}")
    print(f"  Tz: {result['T'][0, 2].item():.4f}")

    if 'beta_gt' in result:
        beta_error = torch.abs(result['beta'][0] - result['beta_gt']).mean()
        print(f"\nGround Truth β:")
        print(f"  {result['beta_gt'].numpy()}")
        print(f"  Mean absolute error: {beta_error.item():.4f}")

    if 'T_gt' in result:
        T_error = torch.abs(result['T'][0] - result['T_gt']).mean()
        print(f"\nGround Truth T:")
        print(f"  Tx: {result['T_gt'][0].item():.4f}")
        print(f"  Ty: {result['T_gt'][1].item():.4f}")
        print(f"  Tz: {result['T_gt'][2].item():.4f}")
        print(f"  Mean absolute error: {T_error.item():.4f}")

    print(f"\nGenerated mesh:")
    print(f"  Vertices: {result['vertices'].shape}")
    print(f"  Joints: {result['joints'].shape}")

    # Save mesh if requested
    if args.save_mesh:
        output_path = Path(args.output_dir) / f"predicted_sample_{args.sample_idx}.obj"
        faces = predictor.star_model.get_faces()
        predictor.save_mesh(result['vertices'], faces, output_path)

        # Also save ground truth mesh for comparison
        if 'beta_gt' in result:
            vertices_gt, _ = predictor.star_model(
                result['beta_gt'].unsqueeze(0).to(predictor.device),
                pose=None,
                trans=None
            )
            gt_path = Path(args.output_dir) / f"ground_truth_sample_{args.sample_idx}.obj"
            predictor.save_mesh(vertices_gt, faces, gt_path)

    print("\n" + "="*60)
    print("Inference completed!")
    print("="*60)


if __name__ == "__main__":
    main()
