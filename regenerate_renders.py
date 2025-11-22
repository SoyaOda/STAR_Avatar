"""Regenerate all rendered images with corrected orientation"""
import torch
import numpy as np
from models.star_layer import STARLayer
from visualizations.renderer import MeshRenderer
import os

def main():
    print("\n" + "="*70)
    print("Regenerating Rendered Images with Corrected Orientation")
    print("="*70)

    # Initialize STAR model
    print("\nInitializing STAR model...")
    star = STARLayer(gender='neutral', num_betas=10)
    faces = star.get_faces()

    # Create renderer
    renderer = MeshRenderer(image_size=512, camera_distance=3.0)

    # Output directory
    output_dir = "outputs/renders"
    os.makedirs(output_dir, exist_ok=True)

    # Define test cases
    test_cases = [
        {
            'name': 'average',
            'betas': torch.zeros(1, 10),
            'title': 'Average Body (174.7 cm)'
        },
        {
            'name': 'tall_thin',
            'betas': torch.tensor([[2.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            'title': 'Tall & Thin Body (195.0 cm)'
        },
        {
            'name': 'short_heavy',
            'betas': torch.tensor([[-2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            'title': 'Short & Heavy Body (153.3 cm)'
        }
    ]

    # Generate individual views
    print("\nGenerating individual front/side views...")
    for case in test_cases:
        vertices, _ = star(case['betas'])
        save_prefix = os.path.join(output_dir, case['name'])

        renderer.render_front_side(vertices, faces, save_prefix=save_prefix)

    # Generate comparison views
    print("\nGenerating comparison views...")
    for case in test_cases:
        vertices, _ = star(case['betas'])
        save_path = os.path.join(output_dir, f"{case['name']}_front_side.png")

        renderer.render_multi_view_figure(
            vertices, faces,
            title=case['title'],
            views=['front', 'side'],
            save_path=save_path
        )

    # Generate grid comparison
    print("\nGenerating grid comparison...")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 18))

    for row, case in enumerate(test_cases):
        vertices, _ = star(case['betas'])

        # Front view
        front_img = renderer.render_view(vertices, faces, view='front')
        axes[row, 0].imshow(front_img)
        axes[row, 0].set_title(f"{case['title']}\nFront View", fontsize=12, fontweight='bold')
        axes[row, 0].axis('off')

        # Side view
        side_img = renderer.render_view(vertices, faces, view='side')
        axes[row, 1].imshow(side_img)
        axes[row, 1].set_title(f"{case['title']}\nSide View", fontsize=12, fontweight='bold')
        axes[row, 1].axis('off')

    plt.suptitle("STAR Body Model - Shape Variations (Front + Side)", fontsize=16, fontweight='bold')
    plt.tight_layout()

    grid_path = os.path.join(output_dir, "comparison_front_side.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {grid_path}")
    plt.close()

    print("\n" + "="*70)
    print("✓ All images regenerated successfully!")
    print("="*70)
    print(f"\nImages saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  Individual views:")
    print("    - average_front.png, average_side.png")
    print("    - tall_thin_front.png, tall_thin_side.png")
    print("    - short_heavy_front.png, short_heavy_side.png")
    print("  Comparison views:")
    print("    - average_front_side.png")
    print("    - tall_and_thin_front_side.png")
    print("    - short_and_heavy_front_side.png")
    print("    - comparison_front_side.png")

if __name__ == "__main__":
    main()
