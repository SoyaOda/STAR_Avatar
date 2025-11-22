"""
Test 2D rendering of STAR meshes
Generate front and back view images from 3D body models
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from models.star_layer import STARLayer
from visualizations.renderer import MeshRenderer


def test_basic_rendering():
    """Test 1: Basic front/back rendering of average body"""
    print("\n" + "="*70)
    print("Test 1: Basic Rendering - Average Body")
    print("="*70)

    # Load STAR model
    star = STARLayer(gender='neutral', num_betas=10)

    # Average body (beta = 0)
    betas = torch.zeros(1, 10)
    vertices, joints = star(betas)
    faces = star.get_faces()

    # Create renderer
    renderer = MeshRenderer(
        image_size=512,
        camera_distance=3.0,
        focal_length=50.0
    )

    # Render and display
    renderer.render_multi_view_figure(
        vertices, faces,
        title="Average Body (Beta = 0)",
        save_path="outputs/renders/average_body_views.png"
    )


def test_shape_variations():
    """Test 2: Render different body shapes"""
    print("\n" + "="*70)
    print("Test 2: Shape Variations Rendering")
    print("="*70)

    star = STARLayer(gender='neutral', num_betas=10)
    faces = star.get_faces()

    # Create output directory
    os.makedirs('outputs/renders', exist_ok=True)

    # Test cases
    test_cases = [
        ("average", torch.zeros(1, 10), "Average Body"),
        ("tall_thin", torch.tensor([[2.0, -1.0] + [0.0]*8]), "Tall & Thin"),
        ("short_heavy", torch.tensor([[-2.0, 1.5] + [0.0]*8]), "Short & Heavy"),
        ("pc1_plus", torch.tensor([[1.5] + [0.0]*9]), "PC1 +1.5σ"),
        ("pc2_plus", torch.tensor([[0.0, 1.5] + [0.0]*8]), "PC2 +1.5σ"),
    ]

    renderer = MeshRenderer(image_size=512, camera_distance=3.0)

    for name, betas, description in test_cases:
        print(f"\nProcessing: {description}")

        # Generate mesh
        vertices, joints = star(betas)

        # Render front and back
        front_img, back_img = renderer.render_front_back(
            vertices, faces,
            save_prefix=f"outputs/renders/{name}"
        )

        # Calculate measurements
        height = (vertices[0, :, 1].max() - vertices[0, :, 1].min()).item()
        print(f"  Height: {height:.3f}m ({height*100:.1f}cm)")

    print(f"\n✓ Rendered {len(test_cases)} body shapes")


def test_comparison_grid():
    """Test 3: Create comparison grid of all shapes"""
    print("\n" + "="*70)
    print("Test 3: Comparison Grid")
    print("="*70)

    star = STARLayer(gender='neutral', num_betas=10)
    faces = star.get_faces()

    test_cases = [
        ("Average", torch.zeros(1, 10)),
        ("Tall & Thin", torch.tensor([[2.0, -1.0] + [0.0]*8])),
        ("Short & Heavy", torch.tensor([[-2.0, 1.5] + [0.0]*8])),
        ("Random 1", torch.randn(1, 10) * 0.5),
        ("Random 2", torch.randn(1, 10) * 0.5),
        ("Random 3", torch.randn(1, 10) * 0.5),
    ]

    renderer = MeshRenderer(image_size=256, camera_distance=3.0)

    # Create grid
    n_cases = len(test_cases)
    fig, axes = plt.subplots(n_cases, 2, figsize=(8, 4*n_cases))

    for i, (name, betas) in enumerate(test_cases):
        print(f"Rendering: {name}")

        vertices, joints = star(betas)
        height = (vertices[0, :, 1].max() - vertices[0, :, 1].min()).item()

        # Render front and back
        front_img = renderer.render_view(vertices, faces, view='front')
        back_img = renderer.render_view(vertices, faces, view='back')

        # Plot
        axes[i, 0].imshow(front_img)
        axes[i, 0].set_title(f"{name} - Front ({height*100:.1f}cm)", fontweight='bold')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(back_img)
        axes[i, 1].set_title(f"{name} - Back", fontweight='bold')
        axes[i, 1].axis('off')

    plt.tight_layout()

    save_path = "outputs/renders/comparison_grid.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison grid: {save_path}")

    plt.show()


def test_camera_distances():
    """Test 4: Different camera distances"""
    print("\n" + "="*70)
    print("Test 4: Camera Distance Variations")
    print("="*70)

    star = STARLayer(gender='neutral', num_betas=10)
    betas = torch.zeros(1, 10)
    vertices, joints = star(betas)
    faces = star.get_faces()

    distances = [2.0, 2.5, 3.0, 3.5, 4.0]

    fig, axes = plt.subplots(1, len(distances), figsize=(15, 3))

    for i, distance in enumerate(distances):
        renderer = MeshRenderer(image_size=256, camera_distance=distance)
        front_img = renderer.render_view(vertices, faces, view='front')

        axes[i].imshow(front_img)
        axes[i].set_title(f"{distance:.1f}m", fontweight='bold')
        axes[i].axis('off')

    fig.suptitle("Camera Distance Variations (Front View)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = "outputs/renders/camera_distances.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")

    plt.show()


def test_gender_comparison():
    """Test 5: Compare male, female, neutral models"""
    print("\n" + "="*70)
    print("Test 5: Gender Model Comparison")
    print("="*70)

    genders = ['neutral', 'male', 'female']
    betas = torch.zeros(1, 10)

    renderer = MeshRenderer(image_size=384, camera_distance=3.0)

    fig, axes = plt.subplots(3, 2, figsize=(8, 12))

    for i, gender in enumerate(genders):
        try:
            print(f"\nLoading {gender} model...")
            star = STARLayer(gender=gender, num_betas=10)
            vertices, joints = star(betas)
            faces = star.get_faces()

            # Render
            front_img = renderer.render_view(vertices, faces, view='front')
            back_img = renderer.render_view(vertices, faces, view='back')

            # Plot
            axes[i, 0].imshow(front_img)
            axes[i, 0].set_title(f"{gender.capitalize()} - Front", fontweight='bold')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(back_img)
            axes[i, 1].set_title(f"{gender.capitalize()} - Back", fontweight='bold')
            axes[i, 1].axis('off')

            height = (vertices[0, :, 1].max() - vertices[0, :, 1].min()).item()
            print(f"  Height: {height*100:.1f}cm")

        except Exception as e:
            print(f"  ⚠️  Could not load {gender} model: {e}")
            axes[i, 0].text(0.5, 0.5, f"{gender}\nNot Available",
                          ha='center', va='center', fontsize=12)
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')

    fig.suptitle("Gender Model Comparison (Beta = 0)", fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = "outputs/renders/gender_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")

    plt.show()


def main():
    """Run all rendering tests"""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  STAR Body Model - 2D Rendering Tests".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)

    print("\nThis test suite will:")
    print("  1. Render basic front/back views of average body")
    print("  2. Render multiple body shape variations")
    print("  3. Create comparison grid")
    print("  4. Test different camera distances")
    print("  5. Compare gender models")

    # Create output directory
    os.makedirs('outputs/renders', exist_ok=True)

    try:
        # Run tests
        test_basic_rendering()
        test_shape_variations()
        test_comparison_grid()
        test_camera_distances()
        test_gender_comparison()

        # Summary
        print("\n" + "#"*70)
        print("#" + " "*68 + "#")
        print("#" + "  All Rendering Tests Completed! ✓".center(68) + "#")
        print("#" + " "*68 + "#")
        print("#"*70)

        print("\n📁 Output images saved in: ./outputs/renders/")
        print("\nGenerated images:")
        print("  - Individual front/back views: *_front.png, *_back.png")
        print("  - Comparison grid: comparison_grid.png")
        print("  - Camera distance test: camera_distances.png")
        print("  - Gender comparison: gender_comparison.png")

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
