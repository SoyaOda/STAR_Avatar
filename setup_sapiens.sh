#!/bin/bash

# Sapiens Setup Script for STAR Avatar Integration
# This script sets up Sapiens-Lite for inference

set -e  # Exit on error

echo "========================================================================"
echo "Sapiens-Lite Setup for STAR Avatar"
echo "========================================================================"

# 1. Set environment variables
export SAPIENS_ROOT="/Users/moei/program/sapiens"
export SAPIENS_LITE_ROOT="$SAPIENS_ROOT/lite"
export SAPIENS_LITE_CHECKPOINT_ROOT="/Users/moei/program/sapiens_lite_host/torchscript"

echo ""
echo "Environment variables:"
echo "  SAPIENS_ROOT: $SAPIENS_ROOT"
echo "  SAPIENS_LITE_ROOT: $SAPIENS_LITE_ROOT"
echo "  SAPIENS_LITE_CHECKPOINT_ROOT: $SAPIENS_LITE_CHECKPOINT_ROOT"

# 2. Create checkpoint directory structure
echo ""
echo "Creating checkpoint directory structure..."
mkdir -p "$SAPIENS_LITE_CHECKPOINT_ROOT/normal/checkpoints/sapiens_0.3b"
mkdir -p "$SAPIENS_LITE_CHECKPOINT_ROOT/depth/checkpoints/sapiens_0.3b"
mkdir -p "$SAPIENS_LITE_CHECKPOINT_ROOT/pose/checkpoints/sapiens_0.3b"
mkdir -p "$SAPIENS_LITE_CHECKPOINT_ROOT/seg/checkpoints/sapiens_0.3b"

echo "✓ Directory structure created"

# 3. Check if conda environment exists
echo ""
echo "Checking conda environment..."
if conda env list | grep -q "sapiens_lite"; then
    echo "✓ conda environment 'sapiens_lite' already exists"
else
    echo "Creating conda environment 'sapiens_lite'..."
    echo "This will take a few minutes..."
    conda create -n sapiens_lite python=3.10 -y
    echo "✓ Conda environment created"
fi

# 4. Install dependencies
echo ""
echo "Installing dependencies..."
echo "Activating environment and installing packages..."
conda run -n sapiens_lite pip install opencv-python tqdm json-tricks pillow numpy --quiet

echo "✓ Dependencies installed"

# 5. Download models from HuggingFace
echo ""
echo "========================================================================"
echo "Model Download Instructions"
echo "========================================================================"
echo ""
echo "To download Sapiens models, visit HuggingFace:"
echo ""
echo "  Normal Estimation (0.3B):"
echo "  https://huggingface.co/facebook/sapiens-normal-0.3b-torchscript"
echo ""
echo "  Depth Estimation (0.3B):"
echo "  https://huggingface.co/facebook/sapiens-depth-0.3b-torchscript"
echo ""
echo "  Pose Estimation (0.3B):"
echo "  https://huggingface.co/facebook/sapiens-pose-0.3b-torchscript"
echo ""
echo "  Segmentation (0.3B):"
echo "  https://huggingface.co/facebook/sapiens-seg-0.3b-torchscript"
echo ""
echo "Download the .pt2 checkpoint files and place them in:"
echo "  $SAPIENS_LITE_CHECKPOINT_ROOT/<task>/checkpoints/sapiens_0.3b/"
echo ""
echo "========================================================================"
echo "Quick Download with Git LFS (if available):"
echo "========================================================================"
echo ""
echo "cd $SAPIENS_LITE_CHECKPOINT_ROOT/normal/checkpoints/sapiens_0.3b"
echo "git lfs install"
echo "git clone https://huggingface.co/facebook/sapiens-normal-0.3b-torchscript ."
echo ""
echo "Repeat for depth, pose, and seg tasks."
echo ""
echo "========================================================================"
echo "Setup Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "1. Download model checkpoints from HuggingFace"
echo "2. Run: conda activate sapiens_lite"
echo "3. Test with: python inference/sapiens_inference.py"
echo ""
