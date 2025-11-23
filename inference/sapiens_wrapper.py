"""
Sapiens Model Wrapper for STAR Avatar Integration

Provides inference interface for Sapiens models (normal, depth, pose, segmentation).
Supports both actual Sapiens models and fallback GT generation.
"""

import sys
import os
import torch
import numpy as np
import cv2
from pathlib import Path

# Add sapiens lite to path if available
SAPIENS_LITE_ROOT = Path("/Users/moei/program/sapiens/lite")
if SAPIENS_LITE_ROOT.exists():
    sys.path.insert(0, str(SAPIENS_LITE_ROOT))


class SapiensInference:
    """
    Wrapper for Sapiens inference.

    Falls back to GT generation if Sapiens models are not available.
    """

    def __init__(self, model_size='0.3b', device='cpu', use_mock=True):
        """
        Initialize Sapiens inference.

        Args:
            model_size: Model size ('0.3b', '0.6b', '1b', '2b')
            device: Device to run inference on
            use_mock: If True, use mock/GT generation instead of actual Sapiens
        """
        self.model_size = model_size
        self.device = torch.device(device)
        self.use_mock = use_mock

        self.models_loaded = False

        if not use_mock:
            self._load_sapiens_models()
        else:
            print("Using mock Sapiens (GT generation)")
            print("To use actual Sapiens:")
            print("  1. Download models from HuggingFace")
            print("  2. Set use_mock=False")

    def _load_sapiens_models(self):
        """Load actual Sapiens models from checkpoints."""
        checkpoint_root = Path("/Users/moei/program/sapiens_lite_host/torchscript")

        # Check if checkpoints exist
        tasks = ['normal', 'depth', 'pose', 'seg']
        self.models = {}

        for task in tasks:
            checkpoint_pattern = f"{task}/checkpoints/sapiens_{self.model_size}/*.pt2"
            checkpoints = list(checkpoint_root.glob(checkpoint_pattern))

            if len(checkpoints) == 0:
                print(f"Warning: No checkpoint found for {task}")
                print(f"  Expected: {checkpoint_root / checkpoint_pattern}")
                continue

            checkpoint_path = checkpoints[0]
            print(f"Loading {task} model from {checkpoint_path}")

            try:
                # Load TorchScript model
                self.models[task] = torch.jit.load(str(checkpoint_path), map_location=self.device)
                self.models[task].eval()
                print(f"  ✓ Loaded {task} model")
            except Exception as e:
                print(f"  ✗ Failed to load {task} model: {e}")

        self.models_loaded = len(self.models) > 0

        if not self.models_loaded:
            print("\nNo Sapiens models loaded. Using mock mode.")
            self.use_mock = True

    def infer(self, image_path, output_dir=None):
        """
        Run Sapiens inference on an image.

        Args:
            image_path: Path to input RGB image
            output_dir: Optional output directory for results

        Returns:
            dict with keys: 'normal', 'depth', 'pose', 'segmentation'
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.use_mock:
            return self._mock_inference(image_rgb, image_path, output_dir)
        else:
            return self._sapiens_inference(image_rgb, output_dir)

    def _sapiens_inference(self, image_rgb, output_dir):
        """Run actual Sapiens model inference."""
        results = {}

        # Preprocess image
        # Sapiens expects specific input format (H, W, 3) normalized
        # TODO: Implement proper preprocessing based on Sapiens specs

        # Normal estimation
        if 'normal' in self.models:
            # TODO: Implement normal inference
            pass

        # Depth estimation
        if 'depth' in self.models:
            # TODO: Implement depth inference
            pass

        # Pose estimation
        if 'pose' in self.models:
            # TODO: Implement pose inference
            pass

        # Segmentation
        if 'seg' in self.models:
            # TODO: Implement segmentation inference
            pass

        return results

    def _mock_inference(self, image_rgb, image_path, output_dir):
        """
        Mock inference using GT generation from 3D mesh.

        This is a placeholder that returns message to use generate_sapiens_style_outputs.py
        """
        print(f"\nMock inference for: {image_path}")
        print("Using GT generation mode.")
        print("\nFor actual Sapiens inference:")
        print("  1. Download models:")
        print("     https://huggingface.co/facebook/sapiens-normal-0.3b-torchscript")
        print("     https://huggingface.co/facebook/sapiens-depth-0.3b-torchscript")
        print("     https://huggingface.co/facebook/sapiens-pose-0.3b-torchscript")
        print("     https://huggingface.co/facebook/sapiens-seg-0.3b-torchscript")
        print("  2. Place .pt2 files in: /Users/moei/program/sapiens_lite_host/torchscript/<task>/checkpoints/sapiens_0.3b/")
        print("  3. Set use_mock=False")
        print("\nFor GT generation from 3D mesh, use:")
        print("  python generate_sapiens_style_outputs.py")

        return {
            'normal': None,
            'depth': None,
            'pose': None,
            'segmentation': None,
            'message': 'Use generate_sapiens_style_outputs.py for GT data'
        }


def test_sapiens():
    """Test Sapiens wrapper."""
    print("="*70)
    print("Testing Sapiens Wrapper")
    print("="*70)

    # Initialize wrapper (mock mode)
    sapiens = SapiensInference(model_size='0.3b', use_mock=True)

    # Test with photorealistic image
    image_path = "outputs/renders/average_photorealistic_front.png"

    if not Path(image_path).exists():
        print(f"\nError: Test image not found: {image_path}")
        print("Please run: python render_average_photorealistic.py")
        return

    print(f"\nTesting inference on: {image_path}")
    results = sapiens.infer(image_path, output_dir="outputs/sapiens_inference")

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)


if __name__ == "__main__":
    test_sapiens()
