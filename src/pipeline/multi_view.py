"""
Multi-View Pipeline Component
Complete pipeline for generating multi-view synthetic data with Sapiens processing
Supports both STAR and MHR body models

Camera presets:
- HMR standard (5000px focal length, square images)
- iPhone presets (1080p, 4K, photo modes)
"""
import numpy as np
import os
import json
from pathlib import Path
from PIL import Image

# Import our components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rendering.renderer import Renderer, rotate_vertices_y_axis
from src.rendering.camera import IPHONE_PRESETS
from src.background.manager import BackgroundManager
from src.compositing.compositor import Compositor
from src.inference.sapiens import SapiensInference


class MultiViewPipeline:
    """
    Complete pipeline for multi-view synthetic data generation

    Pipeline stages:
    1. Generate 3D body with STAR or MHR model
    2. Render from multiple viewpoints
    3. Composite with studio background
    4. (Optional) Run Sapiens inference

    Supports two body models:
    - STAR: Sparse Trained Articulated Human Body Regressor
    - MHR: Momentum Human Rig (Meta's parametric human model)

    Supports camera presets:
    - HMR standard (default)
    - iPhone 1080p, 4K, photo modes
    """

    SUPPORTED_MODELS = ['star', 'mhr']
    SUPPORTED_CAMERA_PRESETS = ['hmr'] + list(IPHONE_PRESETS.keys())

    def __init__(
        self,
        image_size=1024,
        num_betas=10,
        gender='neutral',
        model_type='star',
        camera_preset=None
    ):
        """
        Initialize pipeline components

        Args:
            image_size: Output image resolution (for HMR mode)
            num_betas: Number of shape parameters (STAR: betas, MHR: identity)
            gender: STAR model gender ('neutral', 'male', 'female')
                   (ignored for MHR which is gender-neutral)
            model_type: Body model type ('star' or 'mhr')
            camera_preset: Camera preset ('iphone_1080p', 'iphone_4k', 'iphone_photo_12mp', None for HMR)
        """
        print("\n" + "="*70)
        print("Initializing Multi-View Pipeline")
        print("="*70)

        self.model_type = model_type.lower()
        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {self.SUPPORTED_MODELS}")

        self.camera_preset = camera_preset

        # Initialize body generator based on model type
        if self.model_type == 'star':
            from src.models.star_generator import STARGenerator
            self.body_generator = STARGenerator(gender=gender, num_betas=num_betas)
            self.param_name = 'betas'
        elif self.model_type == 'mhr':
            from src.models.mhr_generator import MHRGenerator
            self.body_generator = MHRGenerator(num_identity=num_betas)
            self.param_name = 'identity'

        # Initialize rendering components
        if camera_preset is not None and camera_preset in IPHONE_PRESETS:
            self.renderer = Renderer(camera_preset=camera_preset)
            self.image_width = self.renderer.image_width
            self.image_height = self.renderer.image_height
        else:
            self.renderer = Renderer(image_width=image_size)
            self.image_width = image_size
            self.image_height = image_size

        self.background_manager = BackgroundManager()
        self.compositor = Compositor()

        # Optional Sapiens inference
        self.sapiens = None

        print("\n" + "="*70)
        print(f"✓ Pipeline initialized (model: {self.model_type.upper()})")
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
            output_size=(self.image_width, self.image_height)
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

    def _generate_body(self, param_std=0.5):
        """
        Generate 3D body using the configured model

        Args:
            param_std: Standard deviation for random parameters

        Returns:
            dict with vertices, faces, and model-specific parameters
        """
        if self.model_type == 'star':
            return self.body_generator.generate_body(beta_std=param_std)
        elif self.model_type == 'mhr':
            return self.body_generator.generate_body(identity_std=param_std)

    def _get_params(self, body):
        """Get shape parameters from body dict based on model type"""
        if self.model_type == 'star':
            return body['betas']
        elif self.model_type == 'mhr':
            return body['identity']

    def generate_subject(
        self,
        output_dir: str,
        subject_idx: int,
        camera_positions: list,
        param_std: float = 0.5,
        save_intermediate: bool = True
    ):
        """
        Generate multi-view images for single subject

        Args:
            output_dir: Output directory
            subject_idx: Subject index
            camera_positions: List of camera positions
            param_std: Standard deviation for random shape params
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
        body = self._generate_body(param_std=param_std)

        vertices = body['vertices']
        faces = body['faces']
        params = self._get_params(body)

        print(f"   {self.param_name.capitalize()} range: [{params.min():.3f}, {params.max():.3f}]")
        print(f"   Vertices: {vertices.shape}")

        # 2. Render multiple views
        print(f"\n2. Rendering {len(camera_positions)} views...")

        images_info = []

        for cam_pos in camera_positions:
            view_idx = cam_pos['view_index']
            azimuth = cam_pos['azimuth']
            distance = cam_pos['distance']

            print(f"\n   View {view_idx+1}/{len(camera_positions)}: {azimuth:.1f}°")

            # Render with azimuth and get camera params (SAM 3D Body format)
            person_rgba, camera_params = self.renderer.render_with_azimuth(
                vertices=vertices,
                faces=faces,
                camera_distance=distance,
                azimuth=azimuth,
                return_camera_params=True
            )

            # Composite with background
            final_image = self.compositor.composite(person_rgba)

            # Save image
            view_name = f"{int(azimuth):03d}deg"
            img_filename = f"view_{view_idx:02d}_{view_name}.png"
            img_path = subject_dir / img_filename

            Image.fromarray(final_image).save(img_path)

            print(f"     ✓ Saved: {img_filename}")

            # Store view info with SAM 3D Body compatible camera params
            images_info.append({
                'view_index': view_idx,
                'filename': img_filename,
                'azimuth': azimuth,
                'camera': camera_params
            })

        # 3. Save metadata (SAM 3D Body compatible format)
        metadata = {
            'subject_idx': subject_idx,
            'model_type': self.model_type,
            'num_views': len(camera_positions),
            'views': images_info
        }

        # Add model-specific parameters in SAM 3D Body format
        if self.model_type == 'mhr':
            # MHR parameters (45 identity, 204 pose, 72 expression)
            metadata['mhr_params'] = {
                'identity': params.tolist(),  # 45 parameters
                'pose': body.get('pose', [0.0] * 204) if isinstance(body.get('pose'), list) else (body['pose'].tolist() if body.get('pose') is not None else [0.0] * 204),
                'expression': body.get('expression', [0.0] * 72) if isinstance(body.get('expression'), list) else (body['expression'].tolist() if body.get('expression') is not None else [0.0] * 72),
            }
            if 'height_m' in body:
                metadata['mhr_params']['height_m'] = float(body['height_m'])
        else:
            # STAR parameters (betas)
            metadata['star_params'] = {
                'betas': params.tolist(),
                'pose': [0.0] * 72  # T-pose
            }

        # Legacy compatibility
        metadata[self.param_name] = params.tolist()

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
        param_std: float = 0.5,
        studio_index: int = 0
    ):
        """
        Generate complete multi-view dataset

        Args:
            output_dir: Output directory
            num_subjects: Number of subjects to generate
            views_per_subject: Views per subject
            param_std: Shape parameter standard deviation
            studio_index: Studio background index

        Returns:
            Dataset summary dict
        """
        print("\n" + "="*70)
        print(f"Generating Multi-View Dataset ({self.model_type.upper()})")
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
        # Calculate camera distance based on focal length and image size
        # To fit a ~1.7m tall human in ~70% of the image height
        # distance = (height * focal_length) / (image_height * display_ratio)

        if self.camera_preset is not None and self.camera_preset in IPHONE_PRESETS:
            # iPhone camera: wider FOV, shorter distance
            # iPhone focal length ~1500px at 1080p, FOV ~69°
            # distance = (1.7 * 1500) / (1080 * 0.7) ≈ 3.4m
            camera_distance = 3.5 if self.model_type == 'mhr' else 3.2
        else:
            # HMR standard: 5000px focal length
            # distance = (1.7 * 5000) / (1024 * 0.7) ≈ 11.9m
            if self.model_type == 'mhr':
                camera_distance = 12.0
            else:
                camera_distance = 11.5

        camera_positions = self.generate_camera_positions(
            num_views=views_per_subject,
            distance=camera_distance
        )

        print(f"\n✓ Generated {len(camera_positions)} camera positions")

        # Generate subjects
        subjects_metadata = []

        for subject_idx in range(num_subjects):
            metadata = self.generate_subject(
                output_dir=output_dir,
                subject_idx=subject_idx,
                camera_positions=camera_positions,
                param_std=param_std
            )

            subjects_metadata.append(metadata)

        # Save dataset summary (SAM 3D Body compatible format)
        # Get camera info from first subject's first view for global params
        first_cam = camera_positions[0]
        global_camera_info = self.renderer.get_camera_params(
            camera_distance=first_cam['distance'],
            azimuth=0.0
        )

        summary = {
            'model_type': self.model_type,
            'total_subjects': num_subjects,
            'views_per_subject': views_per_subject,
            'total_images': num_subjects * views_per_subject,
            'studio_background': str(bg_path),
            # Global camera parameters (SAM 3D Body / HMR format)
            'camera_config': {
                'preset': self.camera_preset or 'hmr_standard',
                'focal_length': global_camera_info['focal_length'],
                'image_width': global_camera_info['image_width'],
                'image_height': global_camera_info['image_height'],
                'camera_distance': global_camera_info['camera_distance'],
                'intrinsics': global_camera_info['intrinsics'],
            },
            'subjects': subjects_metadata
        }

        summary_path = output_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "="*70)
        print("✓ Dataset generation complete!")
        print("="*70)
        print(f"\nOutput: {output_dir}")
        print(f"Model: {self.model_type.upper()}")
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

    # Test with STAR
    print("\n--- Testing STAR model ---")
    pipeline_star = MultiViewPipeline(
        image_size=1024,
        num_betas=10,
        gender='neutral',
        model_type='star'
    )

    # Test with MHR
    print("\n--- Testing MHR model ---")
    pipeline_mhr = MultiViewPipeline(
        image_size=1024,
        num_betas=45,  # MHR uses 45 identity parameters
        model_type='mhr'
    )

    print("\n✓ Both pipeline types initialized")
    print("\nTo generate dataset with STAR, use:")
    print("  pipeline = MultiViewPipeline(model_type='star')")
    print("  pipeline.generate_dataset('outputs/star_dataset', num_subjects=5)")
    print("\nTo generate dataset with MHR, use:")
    print("  pipeline = MultiViewPipeline(model_type='mhr')")
    print("  pipeline.generate_dataset('outputs/mhr_dataset', num_subjects=5)")
