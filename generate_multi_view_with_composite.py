"""
Generate Multi-View Data with Transparent Background + Composite

Clean separation of concerns:
1. Render person with transparent background (RGBA)
2. Extract/prepare studio background (2D image)
3. Composite person + background

This approach gives complete control over background without HDRI coupling.

Usage:
    python generate_multi_view_with_composite.py --num_subjects 3 --views_per_subject 8
"""

import torch
import numpy as np
import os
import argparse
import json
from pathlib import Path
from PIL import Image

from models.star_layer import STARLayer
from visualizations.photorealistic_renderer import PhotorealisticRenderer
from utils.body_smoothing import BodySmoother
from utils.image_compositor import ImageCompositor, extract_studio_background


def rotate_vertices_y_axis(vertices: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate vertices around Y-axis."""
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    rotation_matrix = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])

    rotated = vertices @ rotation_matrix.T
    return rotated


def generate_camera_positions(
    num_views: int,
    distance: float = 3.0,
    height_range: tuple = (1.2, 1.6),
    azimuth_range: tuple = (0, 360)
) -> list:
    """Generate camera positions."""
    cameras = []
    for i in range(num_views):
        azimuth = azimuth_range[0] + (azimuth_range[1] - azimuth_range[0]) * i / num_views
        elevation_factor = np.sin(2 * np.pi * i / num_views) * 0.5 + 0.5
        height = height_range[0] + (height_range[1] - height_range[0]) * elevation_factor

        cameras.append({
            'azimuth': azimuth,
            'height': height,
            'distance': distance
        })
    return cameras


def generate_multi_view_subject_with_composite(
    star_model: STARLayer,
    renderer: PhotorealisticRenderer,
    compositor: ImageCompositor,
    smoother: BodySmoother,
    subject_idx: int,
    camera_positions: list,
    output_dir: str,
    beta_std: float = 0.5,
    smooth_genital: bool = True
):
    """Generate multi-view images with transparent background + composite."""
    print(f"\n{'='*70}")
    print(f"Subject {subject_idx}")
    print(f"{'='*70}")

    subject_dir = os.path.join(output_dir, f"subject_{subject_idx:04d}")
    os.makedirs(subject_dir, exist_ok=True)

    # 1. Sample beta parameters
    num_betas = star_model.num_betas
    betas = torch.randn(1, num_betas) * beta_std

    print(f"\n1. Beta parameters:")
    print(f"   Range: [{betas.min().item():.3f}, {betas.max().item():.3f}]")

    # 2. Generate shaped body
    vertices_torch, joints_torch = star_model(betas)
    v_shaped = vertices_torch[0].cpu().numpy()
    faces = star_model.get_faces()

    # 3. Smooth genital region (disabled by default to avoid unnatural appearance)
    if smooth_genital:
        print(f"\n2. Smoothing genital region...")
        v_shaped = smoother.smooth_genital_region(
            v_shaped, faces, method='flatten', flatten_amount=0.7
        )
        print(f"   ✓ Smoothed")
    else:
        print(f"\n2. Genital smoothing disabled (natural appearance)")

    print(f"\n3. Body shape: {v_shaped.shape}")

    # 4. Render from multiple viewpoints
    print(f"\n4. Rendering {len(camera_positions)} viewpoints...")

    images_info = []

    for view_idx, cam_pos in enumerate(camera_positions):
        print(f"\n   View {view_idx+1}/{len(camera_positions)}:")
        print(f"     Azimuth: {cam_pos['azimuth']:.1f}°")

        # Rotate person for this view
        rotated_vertices = rotate_vertices_y_axis(v_shaped, cam_pos['azimuth'])

        # Render with transparent background
        # PhotorealisticRenderer doesn't have built-in alpha support
        # We'll render with solid background and create alpha mask

        # Render RGB with natural skin color
        # Using lighter, more natural skin tone similar to reference
        skin_color = [0.85, 0.70, 0.60, 1.0]  # Lighter, peachy skin tone

        img_rgb = renderer.render(
            vertices=rotated_vertices,
            faces=faces,
            camera_distance=cam_pos['distance'],
            view='front',
            mesh_color=skin_color,
            background_color=[0, 0, 0, 0]  # Black background for masking
        )

        # Create alpha channel based on non-black pixels
        # This is a simple approach - assumes person is non-black
        alpha = np.any(img_rgb > 10, axis=2).astype(np.uint8) * 255

        # Create RGBA image
        person_rgba = np.dstack([img_rgb, alpha])

        # Composite with background
        final_image = compositor.composite(person_rgba)

        # Save image
        view_name = f"{int(cam_pos['azimuth']):03d}deg"
        img_filename = f"view_{view_idx:02d}_{view_name}.png"
        img_path = os.path.join(subject_dir, img_filename)
        Image.fromarray(final_image).save(img_path)

        images_info.append({
            'view_index': view_idx,
            'filename': img_filename,
            'azimuth': cam_pos['azimuth']
        })

        print(f"     ✓ Saved: {img_filename}")

    # 5. Save metadata
    metadata = {
        'subject_idx': subject_idx,
        'betas': betas.cpu().numpy().tolist(),
        'genital_smoothed': smooth_genital,
        'num_views': len(camera_positions),
        'views': images_info
    }

    metadata_path = os.path.join(subject_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n5. Saved metadata: {metadata_path}")
    print(f"\n{'='*70}")
    print(f"✓ Subject {subject_idx} complete!")
    print(f"{'='*70}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Generate multi-view data with transparent background + composite'
    )
    parser.add_argument('--num_subjects', type=int, default=3)
    parser.add_argument('--views_per_subject', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='outputs/multi_view_composite')
    parser.add_argument('--beta_std', type=float, default=0.5)
    parser.add_argument('--camera_distance', type=float, default=3.0)
    parser.add_argument('--studio_hdri', type=str, default='data/hdri_backgrounds/studio_small_03_1k.jpg')
    parser.add_argument('--smooth_genital', action='store_true',
                        help='Enable genital region smoothing (disabled by default)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("Generate Multi-View Data with Composite")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Subjects: {args.num_subjects}")
    print(f"  Views per subject: {args.views_per_subject}")
    print(f"  Output: {args.output_dir}")
    print(f"  Genital smoothing: {args.smooth_genital}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Prepare studio background
    print("\n" + "="*70)
    print("Preparing Studio Background")
    print("="*70)

    if os.path.exists(args.studio_hdri):
        studio_bg = extract_studio_background(
            args.studio_hdri,
            direction='back',  # Studio at 180°
            output_size=(1024, 1024)
        )
        print(f"✓ Extracted studio background: {studio_bg.shape}")

        # Save background for reference
        bg_path = os.path.join(args.output_dir, "studio_background.png")
        Image.fromarray(studio_bg).save(bg_path)
        print(f"✓ Saved background: {bg_path}")
    else:
        print(f"⚠️  Studio HDRI not found, using solid background")
        studio_bg = None

    # 2. Initialize compositor
    compositor = ImageCompositor(background=studio_bg)

    # 3. Initialize STAR model
    print("\n" + "="*70)
    print("Initializing STAR Model")
    print("="*70)
    star_model = STARLayer(gender='neutral', num_betas=10)
    print("✓ STAR model loaded")

    # 4. Initialize Body Smoother
    print("\n" + "="*70)
    print("Initializing Body Smoother")
    print("="*70)
    smoother = BodySmoother()
    print("✓ Body smoother initialized")

    # 5. Initialize Renderer
    print("\n" + "="*70)
    print("Initializing Photorealistic Renderer")
    print("="*70)
    renderer = PhotorealisticRenderer(image_size=1024)
    print("✓ Renderer initialized")

    # 6. Generate camera positions
    print("\n" + "="*70)
    print("Generating Camera Positions")
    print("="*70)
    camera_positions = generate_camera_positions(
        num_views=args.views_per_subject,
        distance=args.camera_distance
    )
    print(f"✓ Generated {len(camera_positions)} camera positions")

    # 7. Generate subjects
    print("\n" + "="*70)
    print(f"Generating {args.num_subjects} Subjects")
    print("="*70)

    all_metadata = []

    for i in range(args.num_subjects):
        metadata = generate_multi_view_subject_with_composite(
            star_model=star_model,
            renderer=renderer,
            compositor=compositor,
            smoother=smoother,
            subject_idx=i,
            camera_positions=camera_positions,
            output_dir=args.output_dir,
            beta_std=args.beta_std,
            smooth_genital=args.smooth_genital
        )
        all_metadata.append(metadata)

    # 8. Save summary
    print("\n" + "="*70)
    print("Saving Summary")
    print("="*70)

    summary = {
        'total_subjects': args.num_subjects,
        'views_per_subject': args.views_per_subject,
        'total_images': args.num_subjects * args.views_per_subject,
        'studio_background': args.studio_hdri,
        'subjects': all_metadata
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Saved summary: {summary_path}")

    print("\n" + "="*70)
    print("✓ Complete!")
    print("="*70)
    print(f"\nOutput: {os.path.abspath(args.output_dir)}")
    print(f"\nGenerated:")
    print(f"  - {args.num_subjects} subjects")
    print(f"  - {args.views_per_subject} views per subject")
    print(f"  - {args.num_subjects * args.views_per_subject} total images")
    print(f"  - Fixed studio background")
    print()


if __name__ == "__main__":
    main()
