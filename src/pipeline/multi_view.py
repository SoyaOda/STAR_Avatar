"""
Multi-View Pipeline Component
Complete pipeline for generating multi-view synthetic data with Sapiens processing
"""
import numpy as np
import os
import json
from pathlib import Path
from PIL import Image

# Import our components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.star_generator import STARGenerator
from src.rendering.renderer import Renderer, rotate_vertices_y_axis
from src.background.manager import BackgroundManager
from src.compositing.compositor import Compositor
from src.inference.sapiens import SapiensInference


class MultiViewPipeline:
    """
    Complete pipeline for multi-view synthetic data generation

    Pipeline stages:
    1. Generate 3D body with STAR model
    2. Render from multiple viewpoints
    3. Composite with studio background
    4. (Optional) Run Sapiens inference
    """

    def __init__(
        self,
        image_size=1024,
        num_betas=10,
        gender='neutral'
    ):
        """
        Initialize pipeline components

        Args:
            image_size: Output image resolution
            num_betas: Number of shape parameters
            gender: STAR model gender
        """
        print("\n" + "="*70)
        print("Initializing Multi-View Pipeline")
        print("="*70)

        # Initialize components
        self.star_generator = STARGenerator(gender=gender, num_betas=num_betas)
        self.renderer = Renderer(image_size=image_size)
        self.background_manager = BackgroundManager()
        self.compositor = Compositor()

        # Optional Sapiens inference
        self.sapiens = None

        print("\n" + "="*70)
        print("✓ Pipeline initialized")
        print("="*70)

    def setup_background(self, studio_index=0, direction='back'):
        """
        Setup studio background

        Args:
            studio_index: Studio background index (0 or 1)
            direction: Direction to extract from HDRI
        """
        print("\n" + "="*70)
        print("Setting up background")
        print("="*70)

        background = self.background_manager.load_studio_background(
            index=studio_index,
            direction=direction,
            output_size=(self.renderer.image_size, self.renderer.image_size)
        )

        self.compositor.set_background(background)

        return background

    def generate_camera_positions(
        self,
        num_views=4,
        distance=3.0,
        azimuth_range=(0, 360)
    ):
        """
        Generate camera positions for multi-view capture

        Args:
            num_views: Number of viewpoints
            distance: Camera distance from subject
            azimuth_range: Range of azimuth angles (start, end)

        Returns:
            List of camera position dicts
        """
        cameras = []

        for i in range(num_views):
            azimuth = azimuth_range[0] + (azimuth_range[1] - azimuth_range[0]) * i / num_views
            cameras.append({
                'azimuth': azimuth,
                'distance': distance,
                'view_index': i
            })

        return cameras

    def generate_subject(
        self,
        output_dir: str,
        subject_idx: int,
        camera_positions: list,
        beta_std: float = 0.5,
        save_intermediate: bool = True
    ):
        """
        Generate multi-view images for single subject

        Args:
            output_dir: Output directory
            subject_idx: Subject index
            camera_positions: List of camera positions
            beta_std: Standard deviation for random betas
            save_intermediate: Save intermediate outputs

        Returns:
            Dict with subject metadata and image paths
        """
        print(f"\n{'='*70}")
        print(f"Generating Subject {subject_idx}")
        print(f"{'='*70}")

        # Create subject directory
        subject_dir = Path(output_dir) / f"subject_{subject_idx:04d}"
        subject_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate 3D body
        print("\n1. Generating 3D body...")
        body = self.star_generator.generate_body(beta_std=beta_std)

        vertices = body['vertices']
        faces = body['faces']
        betas = body['betas']

        print(f"   Beta range: [{betas.min():.3f}, {betas.max():.3f}]")
        print(f"   Vertices: {vertices.shape}")

        # 2. Render multiple views
        print(f"\n2. Rendering {len(camera_positions)} views...")

        images_info = []

        for cam_pos in camera_positions:
            view_idx = cam_pos['view_index']
            azimuth = cam_pos['azimuth']

            print(f"\n   View {view_idx+1}/{len(camera_positions)}: {azimuth:.1f}°")

            # Rotate body for this view
            rotated_vertices = rotate_vertices_y_axis(vertices, azimuth)

            # Render with alpha
            person_rgba = self.renderer.render_with_alpha(
                vertices=rotated_vertices,
                faces=faces,
                camera_distance=cam_pos['distance'],
                view='front'
            )

            # Composite with background
            final_image = self.compositor.composite(person_rgba)

            # Save image
            view_name = f"{int(azimuth):03d}deg"
            img_filename = f"view_{view_idx:02d}_{view_name}.png"
            img_path = subject_dir / img_filename

            Image.fromarray(final_image).save(img_path)

            print(f"     ✓ Saved: {img_filename}")

            images_info.append({
                'view_index': view_idx,
                'filename': img_filename,
                'azimuth': azimuth
            })

        # 3. Save metadata
        metadata = {
            'subject_idx': subject_idx,
            'betas': betas.tolist(),
            'num_views': len(camera_positions),
            'views': images_info
        }

        metadata_path = subject_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n3. ✓ Saved metadata: {metadata_path.name}")

        return metadata

    def generate_dataset(
        self,
        output_dir: str,
        num_subjects: int,
        views_per_subject: int = 4,
        beta_std: float = 0.5,
        studio_index: int = 0
    ):
        """
        Generate complete multi-view dataset

        Args:
            output_dir: Output directory
            num_subjects: Number of subjects to generate
            views_per_subject: Views per subject
            beta_std: Beta standard deviation
            studio_index: Studio background index

        Returns:
            Dataset summary dict
        """
        print("\n" + "="*70)
        print(f"Generating Multi-View Dataset")
        print("="*70)
        print(f"Subjects: {num_subjects}")
        print(f"Views per subject: {views_per_subject}")

        # Setup output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup background
        background = self.setup_background(studio_index=studio_index, direction='back')

        # Save background
        bg_path = output_dir / 'studio_background.png'
        Image.fromarray(background).save(bg_path)
        print(f"\n✓ Saved background: {bg_path}")

        # Generate camera positions
        camera_positions = self.generate_camera_positions(
            num_views=views_per_subject,
            distance=3.0
        )

        print(f"\n✓ Generated {len(camera_positions)} camera positions")

        # Generate subjects
        subjects_metadata = []

        for subject_idx in range(num_subjects):
            metadata = self.generate_subject(
                output_dir=output_dir,
                subject_idx=subject_idx,
                camera_positions=camera_positions,
                beta_std=beta_std
            )

            subjects_metadata.append(metadata)

        # Save dataset summary
        summary = {
            'total_subjects': num_subjects,
            'views_per_subject': views_per_subject,
            'total_images': num_subjects * views_per_subject,
            'studio_background': str(bg_path),
            'subjects': subjects_metadata
        }

        summary_path = output_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "="*70)
        print("✓ Dataset generation complete!")
        print("="*70)
        print(f"\nOutput: {output_dir}")
        print(f"Subjects: {num_subjects}")
        print(f"Total images: {num_subjects * views_per_subject}")

        return summary

    def run_sapiens_on_dataset(
        self,
        dataset_dir: str,
        tasks: list = ['seg', 'normal', 'depth', 'pose']
    ):
        """
        Run Sapiens inference on entire dataset

        Args:
            dataset_dir: Path to dataset directory
            tasks: Sapiens tasks to run

        Returns:
            Sapiens results summary
        """
        if self.sapiens is None:
            self.sapiens = SapiensInference()

        dataset_dir = Path(dataset_dir)

        # Load dataset summary
        summary_path = dataset_dir / 'summary.json'
        with open(summary_path, 'r') as f:
            summary = json.load(f)

        print("\n" + "="*70)
        print(f"Running Sapiens on Dataset")
        print("="*70)
        print(f"Dataset: {dataset_dir}")
        print(f"Subjects: {summary['total_subjects']}")
        print(f"Total images: {summary['total_images']}")

        # Process each subject
        sapiens_results = []

        for subject_meta in summary['subjects']:
            subject_idx = subject_meta['subject_idx']
            subject_dir = dataset_dir / f"subject_{subject_idx:04d}"

            print(f"\n[Subject {subject_idx}]")

            for view_meta in subject_meta['views']:
                view_file = subject_dir / view_meta['filename']

                output_dir = subject_dir / f"sapiens_{view_meta['view_index']:02d}"

                print(f"  Processing: {view_meta['filename']}")

                outputs = self.sapiens.run_inference(
                    input_image=view_file,
                    output_dir=output_dir,
                    tasks=tasks
                )

                sapiens_results.append({
                    'subject_idx': subject_idx,
                    'view_index': view_meta['view_index'],
                    'input_image': str(view_file),
                    'outputs': {k: str(v) for k, v in outputs.items()}
                })

        # Save Sapiens results
        sapiens_summary_path = dataset_dir / 'sapiens_summary.json'
        with open(sapiens_summary_path, 'w') as f:
            json.dump(sapiens_results, f, indent=2)

        print("\n" + "="*70)
        print("✓ Sapiens processing complete!")
        print("="*70)

        return sapiens_results


if __name__ == "__main__":
    # Test
    print("\nTesting MultiViewPipeline...")

    # Create pipeline
    pipeline = MultiViewPipeline(
        image_size=1024,
        num_betas=10,
        gender='neutral'
    )

    print("\n✓ Pipeline components initialized")
    print("\nTo generate dataset, use:")
    print("  pipeline.generate_dataset(")
    print("      output_dir='outputs/test_dataset',")
    print("      num_subjects=5,")
    print("      views_per_subject=8")
    print("  )")
