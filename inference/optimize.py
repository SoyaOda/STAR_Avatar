"""
Optimization Module for Refining Shape Predictions

Uses LBFGS optimization to refine predicted shape parameters
by minimizing the difference between predicted and observed features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.optim import LBFGS


class ShapeOptimizer:
    """Optimize shape parameters to fit observations."""

    def __init__(self, star_model, device='cpu'):
        """
        Initialize optimizer.

        Args:
            star_model: STARLayer instance
            device: Device to run optimization on
        """
        self.star_model = star_model
        self.device = torch.device(device)

    def optimize_to_target(
        self,
        beta_init,
        target_vertices=None,
        target_joints=None,
        max_iter=20,
        lr=1.0,
        verbose=True
    ):
        """
        Optimize shape parameters to match target vertices/joints.

        Args:
            beta_init: Initial shape parameters [B, num_betas] or [num_betas]
            target_vertices: Optional target vertices [B, N, 3] or [N, 3]
            target_joints: Optional target joints [B, J, 3] or [J, 3]
            max_iter: Maximum LBFGS iterations (default: 20)
            lr: Learning rate (default: 1.0)
            verbose: Print optimization progress

        Returns:
            Optimized beta parameters
        """
        # Ensure batch dimension and move to device
        if beta_init.ndim == 1:
            beta_init = beta_init.unsqueeze(0)
        beta_init = beta_init.to(self.device)

        if target_vertices is not None:
            if target_vertices.ndim == 2:
                target_vertices = target_vertices.unsqueeze(0)
            target_vertices = target_vertices.to(self.device)

        if target_joints is not None:
            if target_joints.ndim == 2:
                target_joints = target_joints.unsqueeze(0)
            target_joints = target_joints.to(self.device)

        # Initialize optimizable beta
        beta_opt = beta_init.clone().detach().requires_grad_(True)

        # LBFGS optimizer
        optimizer = LBFGS(
            [beta_opt],
            lr=lr,
            max_iter=max_iter,
            line_search_fn='strong_wolfe'
        )

        # Track loss history
        loss_history = []

        def closure():
            """Closure function for LBFGS."""
            optimizer.zero_grad()

            # Generate mesh with current beta
            vertices, joints = self.star_model(beta_opt, pose=None, trans=None)

            # Compute loss
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            if target_vertices is not None:
                vertex_loss = F.mse_loss(vertices, target_vertices)
                loss = loss + vertex_loss

            if target_joints is not None:
                joint_loss = F.mse_loss(joints, target_joints)
                loss = loss + 10.0 * joint_loss  # Higher weight for joints

            # Regularization: keep beta close to initial estimate
            reg_loss = 0.01 * F.mse_loss(beta_opt, beta_init)
            loss = loss + reg_loss

            loss.backward()

            loss_history.append(loss.item())

            if verbose and len(loss_history) % 5 == 0:
                print(f"  Iteration {len(loss_history)}: loss = {loss.item():.6f}")

            return loss

        # Run optimization
        if verbose:
            print("\nRunning LBFGS optimization...")
            print(f"  Max iterations: {max_iter}")

        optimizer.step(closure)

        if verbose:
            print(f"  Final loss: {loss_history[-1]:.6f}")
            print(f"  Total iterations: {len(loss_history)}")

        return beta_opt.detach()


def test_optimization():
    """Test shape optimization."""
    from models.star_layer import STARLayer

    print("="*60)
    print("Testing Shape Optimization")
    print("="*60)

    # Initialize STAR model
    print("\nInitializing STAR model...")
    star = STARLayer(gender='neutral', num_betas=10)

    # Create ground truth shape
    print("\nCreating target shape...")
    beta_gt = torch.randn(1, 10) * 0.5
    vertices_gt, joints_gt = star(beta_gt, pose=None, trans=None)

    print(f"Ground truth β: {beta_gt[0].numpy()}")

    # Create initial guess (with noise)
    beta_init = beta_gt + torch.randn_like(beta_gt) * 0.3

    print(f"Initial β: {beta_init[0].numpy()}")
    print(f"Initial error: {torch.abs(beta_init - beta_gt).mean().item():.4f}")

    # Optimize
    print("\n" + "-"*60)
    optimizer = ShapeOptimizer(star, device='cpu')

    beta_optimized = optimizer.optimize_to_target(
        beta_init=beta_init,
        target_vertices=vertices_gt,
        target_joints=joints_gt,
        max_iter=20,
        verbose=True
    )

    # Evaluate
    print("\n" + "-"*60)
    print("Optimization Results")
    print("-"*60)

    print(f"\nOptimized β: {beta_optimized[0].numpy()}")
    print(f"\nErrors:")
    print(f"  Initial β MAE: {torch.abs(beta_init - beta_gt).mean().item():.4f}")
    print(f"  Optimized β MAE: {torch.abs(beta_optimized - beta_gt).mean().item():.4f}")

    # Compute vertex error
    vertices_init, _ = star(beta_init, pose=None, trans=None)
    vertices_opt, _ = star(beta_optimized, pose=None, trans=None)

    vertex_error_init = torch.norm(vertices_init - vertices_gt, dim=2).mean()
    vertex_error_opt = torch.norm(vertices_opt - vertices_gt, dim=2).mean()

    print(f"\n  Initial vertex error: {vertex_error_init.item()*100:.2f} cm")
    print(f"  Optimized vertex error: {vertex_error_opt.item()*100:.2f} cm")

    improvement = (vertex_error_init - vertex_error_opt) / vertex_error_init * 100
    print(f"\n  Improvement: {improvement.item():.1f}%")

    print("\n" + "="*60)
    print("✓ Optimization test completed!")
    print("="*60)


if __name__ == "__main__":
    test_optimization()
