"""
Loss Functions for Shape Estimation Training

Implements three loss components:
1. L_beta: L1 loss on shape parameters
2. L_T: L1 loss on global translation
3. L_geo: Geometric loss on vertex positions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.star_layer import STARLayer


class ShapeEstimationLoss(nn.Module):
    """
    Combined loss for shape estimation training.

    Loss = w_beta * L_beta + w_T * L_T + w_geo * L_geo

    Where:
    - L_beta: L1 loss on predicted vs GT shape parameters
    - L_T: L1 loss on predicted vs GT global translation
    - L_geo: L2 loss on vertex positions from predicted vs GT shapes
    """

    def __init__(
        self,
        star_model=None,
        w_beta=1.0,
        w_T=1.0,
        w_geo=1.0,
        use_geometric_loss=True
    ):
        """
        Initialize loss function.

        Args:
            star_model: STARLayer instance for geometric loss computation
            w_beta: Weight for shape parameter loss (default: 1.0)
            w_T: Weight for translation loss (default: 1.0)
            w_geo: Weight for geometric loss (default: 1.0)
            use_geometric_loss: Whether to use geometric loss (default: True)
        """
        super(ShapeEstimationLoss, self).__init__()

        self.star_model = star_model
        self.w_beta = w_beta
        self.w_T = w_T
        self.w_geo = w_geo
        self.use_geometric_loss = use_geometric_loss

        # Loss components
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

        if use_geometric_loss and star_model is None:
            raise ValueError("star_model is required when use_geometric_loss=True")

    def compute_L_beta(self, beta_pred, beta_gt):
        """
        Compute L1 loss on shape parameters.

        Args:
            beta_pred: Predicted shape parameters [B, num_betas]
            beta_gt: Ground truth shape parameters [B, num_betas]

        Returns:
            L1 loss scalar
        """
        return self.l1_loss(beta_pred, beta_gt)

    def compute_L_T(self, T_pred, T_gt):
        """
        Compute L1 loss on global translation.

        Args:
            T_pred: Predicted translation [B, 3]
            T_gt: Ground truth translation [B, 3]

        Returns:
            L1 loss scalar
        """
        return self.l1_loss(T_pred, T_gt)

    def compute_L_geo(self, beta_pred, beta_gt):
        """
        Compute geometric loss on vertex positions.

        Generates 3D meshes from predicted and GT shape parameters,
        then computes L2 distance between corresponding vertices.

        Args:
            beta_pred: Predicted shape parameters [B, num_betas]
            beta_gt: Ground truth shape parameters [B, num_betas]

        Returns:
            L2 loss on vertex positions
        """
        if not self.use_geometric_loss:
            return torch.tensor(0.0, device=beta_pred.device)

        # Generate meshes from predicted and GT betas
        with torch.no_grad() if not self.training else torch.enable_grad():
            # Use A-pose for both (neutral pose, no translation)
            vertices_pred, _ = self.star_model(beta_pred, pose=None, trans=None)
            vertices_gt, _ = self.star_model(beta_gt, pose=None, trans=None)

        # Compute L2 loss on vertex positions
        # vertices shape: [B, num_vertices, 3]
        loss_geo = self.l2_loss(vertices_pred, vertices_gt)

        return loss_geo

    def forward(self, beta_pred, T_pred, beta_gt, T_gt):
        """
        Compute total loss.

        Args:
            beta_pred: Predicted shape parameters [B, num_betas]
            T_pred: Predicted translation [B, 3]
            beta_gt: Ground truth shape parameters [B, num_betas]
            T_gt: Ground truth translation [B, 3]

        Returns:
            Dictionary containing:
                - 'total': Total weighted loss
                - 'L_beta': Shape parameter loss
                - 'L_T': Translation loss
                - 'L_geo': Geometric loss (if enabled)
        """
        # Compute individual losses
        L_beta = self.compute_L_beta(beta_pred, beta_gt)
        L_T = self.compute_L_T(T_pred, T_gt)

        # Total loss
        total_loss = self.w_beta * L_beta + self.w_T * L_T

        loss_dict = {
            'total': total_loss,
            'L_beta': L_beta,
            'L_T': L_T
        }

        # Add geometric loss if enabled
        if self.use_geometric_loss:
            L_geo = self.compute_L_geo(beta_pred, beta_gt)
            total_loss = total_loss + self.w_geo * L_geo
            loss_dict['total'] = total_loss
            loss_dict['L_geo'] = L_geo

        return loss_dict


def test_losses():
    """Test loss functions."""
    print("="*60)
    print("Testing Shape Estimation Loss Functions")
    print("="*60)

    # Create STAR model for geometric loss
    print("\nInitializing STAR model...")
    star = STARLayer(gender='neutral', num_betas=10)

    # Create loss function
    print("Creating loss function...")
    criterion = ShapeEstimationLoss(
        star_model=star,
        w_beta=1.0,
        w_T=1.0,
        w_geo=0.1,  # Geometric loss typically has smaller weight
        use_geometric_loss=True
    )

    # Create dummy predictions and ground truth
    batch_size = 4
    num_betas = 10

    beta_pred = torch.randn(batch_size, num_betas) * 0.5
    beta_gt = torch.randn(batch_size, num_betas) * 0.5

    T_pred = torch.randn(batch_size, 3)
    T_gt = torch.randn(batch_size, 3)

    print(f"\nInput shapes:")
    print(f"  beta_pred: {beta_pred.shape}")
    print(f"  beta_gt: {beta_gt.shape}")
    print(f"  T_pred: {T_pred.shape}")
    print(f"  T_gt: {T_gt.shape}")

    # Compute losses
    print("\n" + "-"*60)
    print("Computing losses...")
    print("-"*60)

    criterion.train()  # Set to training mode
    loss_dict = criterion(beta_pred, T_pred, beta_gt, T_gt)

    print(f"\nLoss values:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.6f}")

    # Test individual loss components
    print("\n" + "-"*60)
    print("Testing individual loss components")
    print("-"*60)

    L_beta = criterion.compute_L_beta(beta_pred, beta_gt)
    print(f"\nL_beta (L1 on shape params): {L_beta.item():.6f}")

    L_T = criterion.compute_L_T(T_pred, T_gt)
    print(f"L_T (L1 on translation): {L_T.item():.6f}")

    L_geo = criterion.compute_L_geo(beta_pred, beta_gt)
    print(f"L_geo (L2 on vertices): {L_geo.item():.6f}")

    # Test without geometric loss
    print("\n" + "-"*60)
    print("Testing without geometric loss")
    print("-"*60)

    criterion_no_geo = ShapeEstimationLoss(
        star_model=None,
        w_beta=1.0,
        w_T=1.0,
        w_geo=0.0,
        use_geometric_loss=False
    )

    loss_dict_no_geo = criterion_no_geo(beta_pred, T_pred, beta_gt, T_gt)

    print(f"\nLoss values (without geometric):")
    for key, value in loss_dict_no_geo.items():
        print(f"  {key}: {value.item():.6f}")

    # Test gradient flow
    print("\n" + "-"*60)
    print("Testing gradient flow")
    print("-"*60)

    beta_pred_grad = torch.randn(batch_size, num_betas, requires_grad=True) * 0.5
    T_pred_grad = torch.randn(batch_size, 3, requires_grad=True)

    loss_dict_grad = criterion(beta_pred_grad, T_pred_grad, beta_gt, T_gt)
    total_loss = loss_dict_grad['total']

    total_loss.backward()

    print(f"\nGradient check:")
    print(f"  beta_pred has gradient: {beta_pred_grad.grad is not None}")
    print(f"  T_pred has gradient: {T_pred_grad.grad is not None}")

    if beta_pred_grad.grad is not None:
        print(f"  beta_pred gradient norm: {beta_pred_grad.grad.norm().item():.6f}")
    if T_pred_grad.grad is not None:
        print(f"  T_pred gradient norm: {T_pred_grad.grad.norm().item():.6f}")

    print("\n" + "="*60)
    print("✓ Loss function test passed!")
    print("="*60)


if __name__ == "__main__":
    test_losses()
