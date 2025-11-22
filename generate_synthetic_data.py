"""
Generate synthetic training data from STAR model
Based on spec1.md合成データ生成 section
"""
import torch
import numpy as np
import os
from models.star_layer import STARLayer
from visualizations.pytorch_renderer import STARRenderer
import cv2


def generate_synthetic_sample(
    star_model: STARLayer,
    renderer: STARRenderer,
    camera_distance: float = 3.0,
    normalize_height: float = 1.7,
    save_dir: str = None
) -> dict:
    """
    Generate one synthetic data sample

    Args:
        star_model: STARLayer instance
        renderer: STARRenderer instance
        camera_distance: Camera distance in meters (default: 3.0, spec: 2.5-3.5)
        normalize_height: Normalize depth maps to this height (spec: 1.7m)
        save_dir: Directory to save outputs (optional)

    Returns:
        Dictionary with synthetic data tensors
    """
    # 1. Sample shape parameters β from standard normal distribution
    # (spec1.md line 155)
    num_betas = star_model.num_betas
    betas = torch.randn(1, num_betas) * 0.5  # Slightly reduced variance

    # 2. Sample pose parameters θ (small variations around A-pose)
    # (spec1.md line 156: ±10° shoulders, ±5° elbows, etc.)
    # For now, use default A-pose
    pose = None  # Default A-pose in STAR
    trans = None

    # 3. Generate 3D mesh from STAR model
    # (spec1.md line 157)
    vertices, joints = star_model(betas, pose, trans)

    # Get faces
    faces = star_model.get_faces()

    # Convert to appropriate types
    vertices = vertices[0]  # Remove batch dim [6890, 3]
    joints = joints[0]  # [24, 3]
    faces_tensor = torch.from_numpy(faces).long()

    # 4. Setup camera with random distance variation
    # (spec1.md line 158: D=2.5~3.5m, Δx/Δy=±2%)
    delta_x = np.random.uniform(-0.02, 0.02) * camera_distance
    delta_y = np.random.uniform(-0.02, 0.02) * camera_distance

    # 5. Render front and back views
    # (spec1.md line 159-171)
    print(f"Rendering synthetic data...")
    print(f"  - Beta shape: {betas.shape}, range: [{betas.min():.2f}, {betas.max():.2f}]")
    print(f"  - Camera distance: {camera_distance:.2f}m")

    # Front view
    front_outputs = renderer.render_all(
        vertices=vertices,
        faces=faces_tensor,
        joints_3d=joints,
        camera_distance=camera_distance,
        view='front',
        normalize_height=normalize_height
    )

    # Back view
    back_outputs = renderer.render_all(
        vertices=vertices,
        faces=faces_tensor,
        joints_3d=joints,
        camera_distance=camera_distance,
        view='back',
        normalize_height=normalize_height
    )

    # Compile outputs
    synthetic_data = {
        'beta_gt': betas.numpy(),  # Ground truth shape parameters
        'front': front_outputs,
        'back': back_outputs,
        'camera_distance': camera_distance,
        'normalize_height': normalize_height
    }

    # Save outputs if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Save front view outputs
        cv2.imwrite(
            os.path.join(save_dir, 'front_normal.png'),
            front_outputs['normal']
        )
        cv2.imwrite(
            os.path.join(save_dir, 'front_depth.png'),
            (front_outputs['depth'] * 100).astype(np.uint8)  # Scale for visibility
        )
        cv2.imwrite(
            os.path.join(save_dir, 'front_mask.png'),
            front_outputs['mask'] * 255
        )

        # Save back view outputs
        cv2.imwrite(
            os.path.join(save_dir, 'back_normal.png'),
            back_outputs['normal']
        )
        cv2.imwrite(
            os.path.join(save_dir, 'back_depth.png'),
            (back_outputs['depth'] * 100).astype(np.uint8)
        )
        cv2.imwrite(
            os.path.join(save_dir, 'back_mask.png'),
            back_outputs['mask'] * 255
        )

        # Save joint heatmaps (sum across joints for visualization)
        front_heatmap_vis = front_outputs['joint_heatmaps'].sum(axis=2)
        front_heatmap_vis = (front_heatmap_vis / front_heatmap_vis.max() * 255).astype(np.uint8)
        cv2.imwrite(
            os.path.join(save_dir, 'front_joints_heatmap.png'),
            front_heatmap_vis
        )

        back_heatmap_vis = back_outputs['joint_heatmaps'].sum(axis=2)
        back_heatmap_vis = (back_heatmap_vis / back_heatmap_vis.max() * 255).astype(np.uint8)
        cv2.imwrite(
            os.path.join(save_dir, 'back_joints_heatmap.png'),
            back_heatmap_vis
        )

        # Save beta parameters
        np.save(os.path.join(save_dir, 'beta_gt.npy'), betas.numpy())

        print(f"\n✓ Synthetic data saved to: {save_dir}")

    return synthetic_data


def main():
    """Generate sample synthetic data"""
    print("\n" + "="*70)
    print("Synthetic Data Generation (spec1.md)")
    print("="*70)

    # Initialize STAR model
    print("\nInitializing STAR model...")
    star = STARLayer(gender='neutral', num_betas=10)

    # Initialize renderer
    print("Initializing PyTorch renderer...")
    renderer = STARRenderer(
        image_size=512,
        focal_length=50.0,  # 50mm (spec requirement)
        device='cpu'
    )

    # Generate synthetic samples
    num_samples = 3
    print(f"\nGenerating {num_samples} synthetic samples...\n")

    for i in range(num_samples):
        # Random camera distance (2.5-3.5m as per spec)
        camera_dist = np.random.uniform(2.5, 3.5)

        save_dir = f"outputs/synthetic_data/sample_{i+1}"
        synthetic_data = generate_synthetic_sample(
            star_model=star,
            renderer=renderer,
            camera_distance=camera_dist,
            normalize_height=1.7,  # Spec requirement
            save_dir=save_dir
        )

        print(f"\n{'='*70}")
        print(f"Sample {i+1}/{num_samples} completed")
        print(f"{'='*70}\n")

    print("\n✓ All synthetic samples generated!")
    print(f"  Location: outputs/synthetic_data/")
    print(f"\nOutputs per sample:")
    print(f"  - front_normal.png: Camera-space normal map (front)")
    print(f"  - front_depth.png: Depth map (front)")
    print(f"  - front_mask.png: Person segmentation mask (front)")
    print(f"  - front_joints_heatmap.png: Joint position heatmaps (front)")
    print(f"  - back_normal.png: Camera-space normal map (back)")
    print(f"  - back_depth.png: Depth map (back)")
    print(f"  - back_mask.png: Person segmentation mask (back)")
    print(f"  - back_joints_heatmap.png: Joint position heatmaps (back)")
    print(f"  - beta_gt.npy: Ground truth shape parameters")


if __name__ == "__main__":
    main()
