"""
Sapiens Inference Wrapper Component
Handles running Sapiens models for segmentation, normal, depth, and pose estimation
"""
import subprocess
import os
import json
from pathlib import Path
from typing import Optional, Union, Dict


class SapiensInference:
    """
    Wrapper for Sapiens inference

    Features:
    - Run segmentation, normal, depth, pose estimation
    - Support for single image or batch processing
    - Automatic output directory management
    """

    def __init__(
        self,
        sapiens_root: str = '/Users/moei/program/sapiens-main/lite',
        checkpoint_root: str = '~/sapiens_lite_host/torchscript'
    ):
        """
        Initialize Sapiens inference wrapper

        Args:
            sapiens_root: Path to Sapiens repository root
            checkpoint_root: Path to Sapiens checkpoints
        """
        self.sapiens_root = Path(sapiens_root)
        self.checkpoint_root = Path(os.path.expanduser(checkpoint_root))

        if not self.sapiens_root.exists():
            raise FileNotFoundError(f"Sapiens root not found: {self.sapiens_root}")

        print("="*70)
        print("Initializing Sapiens Inference")
        print("="*70)
        print(f"Sapiens root: {self.sapiens_root}")
        print(f"Checkpoint root: {self.checkpoint_root}")
        print("✓ Sapiens wrapper initialized")

    def run_inference(
        self,
        input_image: Union[str, Path],
        output_dir: Union[str, Path],
        tasks: list = ['seg', 'normal', 'depth', 'pose']
    ) -> Dict[str, Path]:
        """
        Run Sapiens inference on single image

        Args:
            input_image: Path to input image
            output_dir: Output directory
            tasks: List of tasks to run ['seg', 'normal', 'depth', 'pose']

        Returns:
            Dict mapping task names to output directories
        """
        input_image = Path(input_image)
        output_dir = Path(output_dir)

        if not input_image.exists():
            raise FileNotFoundError(f"Input image not found: {input_image}")

        # Use run_sapiens_single.sh script
        script_path = Path(__file__).parent.parent.parent / 'run_sapiens_single.sh'

        if not script_path.exists():
            raise FileNotFoundError(f"Sapiens script not found: {script_path}")

        print(f"\nRunning Sapiens inference on: {input_image.name}")
        print(f"Output directory: {output_dir}")

        # Run inference
        cmd = [str(script_path), str(input_image), str(output_dir)]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            # Parse output for success
            if "✅ Sapiens推論完了！" in result.stdout or result.returncode == 0:
                print("✓ Sapiens inference completed successfully")
            else:
                print("⚠️  Sapiens inference may have warnings")
                print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"❌ Sapiens inference failed: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise

        # Return output paths
        outputs = {}
        for task in tasks:
            task_dir = output_dir / task
            if task_dir.exists():
                outputs[task] = task_dir

        return outputs

    def run_batch(
        self,
        input_images: list,
        output_base_dir: Union[str, Path],
        tasks: list = ['seg', 'normal', 'depth', 'pose']
    ) -> list:
        """
        Run Sapiens inference on multiple images

        Args:
            input_images: List of input image paths
            output_base_dir: Base output directory
            tasks: List of tasks to run

        Returns:
            List of output dictionaries
        """
        output_base_dir = Path(output_base_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for i, input_image in enumerate(input_images):
            print(f"\n[{i+1}/{len(input_images)}] Processing: {Path(input_image).name}")

            output_dir = output_base_dir / f"image_{i:04d}"
            outputs = self.run_inference(input_image, output_dir, tasks)

            results.append({
                'input_image': str(input_image),
                'output_dir': str(output_dir),
                'outputs': {k: str(v) for k, v in outputs.items()}
            })

        return results

    def get_segmentation_output(self, output_dir: Union[str, Path], image_name: str):
        """
        Get path to segmentation output

        Args:
            output_dir: Sapiens output directory
            image_name: Original image name

        Returns:
            Path to segmentation output
        """
        seg_dir = Path(output_dir) / 'seg'
        # Sapiens outputs PNG files with same basename
        seg_file = seg_dir / Path(image_name).with_suffix('.png').name
        return seg_file if seg_file.exists() else None

    def get_normal_output(self, output_dir: Union[str, Path], image_name: str):
        """Get path to normal map output"""
        normal_dir = Path(output_dir) / 'normal'
        normal_file = normal_dir / Path(image_name).with_suffix('.png').name
        return normal_file if normal_file.exists() else None

    def get_depth_output(self, output_dir: Union[str, Path], image_name: str):
        """Get path to depth map output"""
        depth_dir = Path(output_dir) / 'depth'
        depth_file = depth_dir / Path(image_name).with_suffix('.png').name
        return depth_file if depth_file.exists() else None

    def get_pose_output(self, output_dir: Union[str, Path], image_name: str):
        """Get path to pose JSON output"""
        pose_dir = Path(output_dir) / 'pose'
        pose_file = pose_dir / Path(image_name).with_suffix('.json').name
        return pose_file if pose_file.exists() else None


if __name__ == "__main__":
    # Test
    print("\nTesting SapiensInference...")

    # Note: This test requires Sapiens to be installed and an input image
    # Just show the initialization for now

    inference = SapiensInference()

    print("\n✓ SapiensInference test complete")
    print("\nTo run full test, use:")
    print("  sapiens = SapiensInference()")
    print("  sapiens.run_inference('input.png', 'output_dir')")
