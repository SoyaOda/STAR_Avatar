# Sapiens Integration Guide

## Overview

This document describes how to integrate the actual Sapiens model from Meta AI
for real-world image inference.

## Current Status

✅ **GT Generation**: Ground-truth normal/depth/pose/segmentation from 3D mesh
❌ **Sapiens Inference**: Not yet integrated (requires model download)

## Full Sapiens Integration Steps

### 1. Clone Sapiens Repository

```bash
cd /path/to/your/workspace
git clone https://github.com/facebookresearch/sapiens.git
export SAPIENS_ROOT=/path/to/sapiens
```

### 2. Install Dependencies (Lite Version Recommended)

```bash
cd $SAPIENS_ROOT/lite
pip install torch torchvision numpy opencv-python
```

### 3. Download Model Checkpoints

Download from HuggingFace: https://huggingface.co/facebook/sapiens

Required checkpoints:
- `sapiens_0.3b` or `sapiens_0.6b` (recommended for speed)
- Task-specific checkpoints: normal, depth, pose, seg

### 4. Run Inference

```python
# Example inference code (pseudocode)
from sapiens import SapiensModel

model = SapiensModel.load_pretrained('sapiens_0.6b')

# Input: RGB image [H, W, 3]
rgb_image = cv2.imread('image.png')

# Inference
outputs = model.infer(rgb_image)

normal_map = outputs['normal']  # [H, W, 3]
depth_map = outputs['depth']    # [H, W]
pose_2d = outputs['pose']       # [num_joints, 3] (x, y, confidence)
segmentation = outputs['seg']   # [H, W]
```

### 5. Integration with STAR Avatar

Modify `inference/sapiens_wrapper.py` to:
1. Load Sapiens model
2. Process input images
3. Extract features
4. Convert to ShapeEstimator input format

## Performance

- **Model Size**: 0.3B - 2B parameters
- **Inference Speed**: ~0.3s per image (GPU)
- **Resolution**: Supports up to 1024x1024
- **Accuracy**: State-of-the-art on human-centric benchmarks

## Alternative: Mock Sapiens Output

For testing without Sapiens installation, use the GT generator:

```bash
python generate_sapiens_style_outputs.py
```

This creates Sapiens-style outputs from known 3D mesh data.

## References

- GitHub: https://github.com/facebookresearch/sapiens
- Paper: https://arxiv.org/abs/2408.12569
- HuggingFace: https://huggingface.co/facebook/sapiens
