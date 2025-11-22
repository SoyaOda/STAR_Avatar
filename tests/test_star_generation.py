"""
Test STAR mesh generation

This script tests the basic functionality of generating 3D human meshes
from STAR body model parameters (beta).
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from models.star_layer import STARLayer
from visualizations.mesh_viewer import (
    visualize_mesh_open3d,
    visualize_mesh_matplotlib,
    visualize_joints,
    save_mesh_obj
)


def test_basic_generation():
    """Test 1: Basic mesh generation with default parameters"""
    print("\n" + "="*70)
    print("Test 1: Basic Mesh Generation (Beta = 0)")
    print("="*70)

    # Create STAR layer
    star = STARLayer(gender='neutral', num_betas=10)

    # Zero beta = average body shape
    betas = torch.zeros(1, 10)

    print(f"\nGenerating mesh with beta = {betas[0].numpy()}")

    # Generate mesh
    vertices, joints = star(betas)

    print(f"\nResults:")
    print(f"  - Generated {vertices.shape[1]} vertices")
    print(f"  - Generated {joints.shape[1]} joints")

    # Get faces
    faces = star.get_faces()

    # Visualize
    print("\nVisualizing mesh...")
    visualize_mesh_matplotlib(
        vertices, faces,
        title="STAR Mesh - Average Body (Beta=0)"
    )

    return star, vertices, joints, faces


def test_shape_variations():
    """Test 2: Generate meshes with different beta parameters"""
    print("\n" + "="*70)
    print("Test 2: Shape Variations")
    print("="*70)

    star = STARLayer(gender='neutral', num_betas=10)

    # Test different beta configurations
    test_cases = [
        ("Average", torch.zeros(1, 10)),
        ("PC1 +2σ", torch.tensor([[2.0] + [0.0]*9])),
        ("PC1 -2σ", torch.tensor([[-2.0] + [0.0]*9])),
        ("PC2 +2σ", torch.tensor([[0.0, 2.0] + [0.0]*8])),
        ("Random", torch.randn(1, 10) * 0.5),
    ]

    faces = star.get_faces()
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    for name, betas in test_cases:
        print(f"\n{name}:")
        print(f"  Beta: {betas[0].numpy()}")

        vertices, joints = star(betas)

        # Calculate body measurements
        height = vertices[0, :, 1].max() - vertices[0, :, 1].min()
        width = vertices[0, :, 0].max() - vertices[0, :, 0].min()
        depth = vertices[0, :, 2].max() - vertices[0, :, 2].min()

        print(f"  Height: {height:.3f} m")
        print(f"  Width:  {width:.3f} m")
        print(f"  Depth:  {depth:.3f} m")

        # Save mesh
        obj_path = os.path.join(output_dir, f"mesh_{name.replace(' ', '_').lower()}.obj")
        save_mesh_obj(vertices, faces, obj_path)

    print(f"\n✓ Saved {len(test_cases)} meshes to: {output_dir}")


def test_batch_generation():
    """Test 3: Batch generation (multiple meshes at once)"""
    print("\n" + "="*70)
    print("Test 3: Batch Generation")
    print("="*70)

    star = STARLayer(gender='neutral', num_betas=10)

    # Generate batch of random bodies
    batch_size = 5
    betas = torch.randn(batch_size, 10) * 0.5

    print(f"\nGenerating {batch_size} meshes in batch...")
    vertices, joints = star(betas)

    print(f"\nResults:")
    print(f"  - Vertices: {vertices.shape}")
    print(f"  - Joints: {joints.shape}")

    for i in range(batch_size):
        height = vertices[i, :, 1].max() - vertices[i, :, 1].min()
        print(f"  Mesh {i}: Height = {height:.3f} m")

    return vertices, joints


def test_with_translation():
    """Test 4: Mesh generation with translation"""
    print("\n" + "="*70)
    print("Test 4: Mesh with Translation")
    print("="*70)

    star = STARLayer(gender='neutral', num_betas=10)

    betas = torch.zeros(1, 10)
    trans = torch.tensor([[0.5, 0.0, 0.0]])  # Move 0.5m to the right

    print(f"\nGenerating mesh with translation: {trans[0].numpy()}")

    vertices_no_trans, _ = star(betas)
    vertices_trans, _ = star(betas, trans=trans)

    print(f"\nCentroid without translation: {vertices_no_trans.mean(dim=1)[0].numpy()}")
    print(f"Centroid with translation:    {vertices_trans.mean(dim=1)[0].numpy()}")

    diff = vertices_trans - vertices_no_trans
    print(f"Translation applied: {diff.mean(dim=1)[0].numpy()}")


def test_joint_visualization():
    """Test 5: Visualize joint positions"""
    print("\n" + "="*70)
    print("Test 5: Joint Visualization")
    print("="*70)

    star = STARLayer(gender='neutral', num_betas=10)

    betas = torch.zeros(1, 10)
    vertices, joints = star(betas)

    print(f"\nJoint positions:")
    for i, joint in enumerate(joints[0]):
        print(f"  Joint {i:2d}: [{joint[0]:6.3f}, {joint[1]:6.3f}, {joint[2]:6.3f}]")

    # Visualize joints
    visualize_joints(joints, title="STAR Joint Positions (T-pose)")


def interactive_3d_viewer():
    """Interactive 3D viewer using Open3D"""
    print("\n" + "="*70)
    print("Interactive 3D Viewer (Open3D)")
    print("="*70)

    star = STARLayer(gender='neutral', num_betas=10)

    # Generate a random body
    betas = torch.randn(1, 10) * 0.5
    print(f"\nBeta parameters: {betas[0].numpy()}")

    vertices, joints = star(betas)
    faces = star.get_faces()

    # Calculate measurements
    height = vertices[0, :, 1].max() - vertices[0, :, 1].min()
    print(f"Body height: {height:.3f} m ({height*100:.1f} cm)")

    # Open interactive viewer
    try:
        visualize_mesh_open3d(
            vertices, faces,
            window_name="STAR Body - Interactive Viewer"
        )
    except Exception as e:
        print(f"⚠️  Could not open interactive viewer: {e}")
        print("Showing static matplotlib view instead...")
        visualize_mesh_matplotlib(
            vertices, faces,
            title="STAR Body (Interactive view unavailable)"
        )


def main():
    """Run all tests"""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  STAR Body Model - Mesh Generation Tests".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)

    print("\nThis test suite will:")
    print("  1. Generate basic mesh with default parameters")
    print("  2. Test shape variations with different beta values")
    print("  3. Test batch generation")
    print("  4. Test mesh translation")
    print("  5. Visualize joint positions")
    print("  6. Open interactive 3D viewer")

    input("\nPress Enter to continue...")

    # Run tests
    try:
        # Test 1: Basic generation
        star, vertices, joints, faces = test_basic_generation()

        # Test 2: Shape variations
        test_shape_variations()

        # Test 3: Batch generation
        test_batch_generation()

        # Test 4: Translation
        test_with_translation()

        # Test 5: Joint visualization
        test_joint_visualization()

        # Test 6: Interactive viewer
        print("\n" + "="*70)
        print("Would you like to open the interactive 3D viewer?")
        response = input("(y/n): ").strip().lower()
        if response == 'y':
            interactive_3d_viewer()

        # Summary
        print("\n" + "#"*70)
        print("#" + " "*68 + "#")
        print("#" + "  All Tests Completed Successfully! ✓".center(68) + "#")
        print("#" + " "*68 + "#")
        print("#"*70)

        print("\n📁 Output files saved in: ./outputs/")
        print("\nNext steps:")
        print("  1. Download official STAR models from https://star.is.tue.mpg.de/")
        print("  2. Replace minimal model with official .npz files")
        print("  3. Implement shape estimation network (Phase 1.2)")

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
