"""Regenerate individual front and side views"""
import torch
from models.star_layer import STARLayer
from visualizations.renderer import MeshRenderer
import os

def main():
    print("Regenerating individual views...")

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
            'betas': torch.zeros(1, 10)
        },
        {
            'name': 'tall_thin',
            'betas': torch.tensor([[2.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        },
        {
            'name': 'short_heavy',
            'betas': torch.tensor([[-2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        }
    ]

    # Generate individual views
    for case in test_cases:
        vertices, _ = star(case['betas'])
        save_prefix = os.path.join(output_dir, case['name'])

        renderer.render_front_side(vertices, faces, save_prefix=save_prefix)

    print("\n✓ Individual views regenerated successfully!")

if __name__ == "__main__":
    main()
