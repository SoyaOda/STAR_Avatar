"""Regenerate comparison views only"""
import torch
import numpy as np
from models.star_layer import STARLayer
from visualizations.renderer import MeshRenderer
import matplotlib.pyplot as plt
import os

def main():
    print("Regenerating comparison views...")

    # Initialize STAR model
    star = STARLayer(gender='neutral', num_betas=10)
    faces = star.get_faces()

    # Create renderer
    renderer = MeshRenderer(image_size=512, camera_distance=3.0)

    # Output directory
    output_dir = "outputs/renders"

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

    print("\n✓ Comparison views regenerated successfully!")

if __name__ == "__main__":
    main()
