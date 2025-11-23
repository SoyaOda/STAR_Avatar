# STAR + Clothing Blender Rendering Pipeline

## Overview

This Blender-based pipeline generates photorealistic synthetic human data optimized for Sapiens inference. It combines STAR parametric body models with realistic clothing meshes, HDRI lighting, and multi-pass rendering to create training data that matches Sapiens' expected input distribution.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Blender Python Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ STAR Model   │───▶│   Skeleton   │───▶│   Skinning   │  │
│  │  (.npz)      │    │  (24 joints) │    │   Weights    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │          │
│         ▼                    ▼                    ▼          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Shape & Pose Application                   │  │
│  │  • Beta (10 params) → Body shape                     │  │
│  │  • Pose (72 params) → Joint rotations                │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Clothing & Materials                       │  │
│  │  • OBJ clothing meshes → Armature skinning           │  │
│  │  • Cloth physics simulation                          │  │
│  │  • Skin material (Fitzpatrick scale)                 │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Scene Setup & Rendering                    │  │
│  │  • HDRI environment lighting                          │  │
│  │  • Camera positioning (spherical coords)             │  │
│  │  • Cycles renderer (GPU acceleration)                │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Multi-Pass Output                          │  │
│  │  • RGB image (.png)                                   │  │
│  │  • Surface normals (.png)                             │  │
│  │  • Depth map (.exr)                                   │  │
│  │  • Segmentation mask (.png)                           │  │
│  │  • Parameters (.pkl)                                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Features

### 1. Full STAR Model Support

- **Shape Parameters**: 10-dimensional beta vector controlling body proportions
- **Pose Parameters**: 72-dimensional pose vector (24 joints × 3 axis-angle)
- **Complete Skeleton**: Full 24-joint SMPL-compatible skeleton hierarchy
- **LBS Skinning**: Linear Blend Skinning using STAR's native weights

### 2. Realistic Clothing

- **Multiple Garment Types**: Support for any OBJ clothing mesh
- **Automatic Skinning**: Clothing inherits body armature for deformation
- **Physics Simulation**: Cloth modifier for realistic draping
- **Collision Detection**: Body collision for cloth interaction

### 3. Photorealistic Materials

- **Skin Tones**: 6 presets based on Fitzpatrick scale (I-VI)
  - Light (0.95, 0.85, 0.80)
  - Medium (0.85, 0.70, 0.60)
  - Tan (0.75, 0.60, 0.50)
  - Brown (0.55, 0.40, 0.30)
  - Dark (0.35, 0.25, 0.20)
- **Subsurface Scattering**: SSS for realistic skin translucency
- **Random Variation**: Per-sample color jitter for diversity

### 4. HDRI Lighting

- **Environment Maps**: Random selection from HDRI directory
- **Realistic Backgrounds**: Matches Sapiens training distribution
- **Supported Formats**: .hdr, .exr, .jpg, .png

### 5. Multi-Pass Rendering

| Pass | Format | Description |
|------|--------|-------------|
| RGB | PNG | Photorealistic color image |
| Normal | PNG | Surface normal map |
| Depth | EXR | Depth/distance map (32-bit float) |
| Mask | PNG | Segmentation mask (optional) |

## Installation

### Prerequisites

```bash
# Blender 3.6+ required
# Check version:
blender --version

# Install Blender if needed (macOS):
brew install --cask blender

# Or download from:
# https://www.blender.org/download/
```

### Python Dependencies

The script uses Blender's built-in Python environment. Additional packages can be installed:

```bash
# Get Blender's Python path
blender --background --python-expr "import sys; print(sys.executable)"

# Install packages (example)
/Applications/Blender.app/Contents/Resources/3.6/python/bin/python3.10 -m pip install numpy
```

## Usage

### Basic Usage

```bash
blender --background --python blender/star_clothing_renderer.py -- \
    --star_model data/STAR/female/model.npz \
    --output_dir outputs/blender_renders \
    --num_samples 100
```

### With Clothing

```bash
blender --background --python blender/star_clothing_renderer.py -- \
    --star_model data/STAR/female/model.npz \
    --clothing_mesh clothing/sports_bra.obj clothing/shorts.obj \
    --output_dir outputs/blender_renders \
    --num_samples 100
```

### With HDRI Backgrounds

```bash
blender --background --python blender/star_clothing_renderer.py -- \
    --star_model data/STAR/female/model.npz \
    --clothing_mesh clothing/sports_bra.obj clothing/shorts.obj \
    --hdri_dir data/hdri_backgrounds \
    --output_dir outputs/blender_renders \
    --num_samples 100
```

## Command-Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--star_model` | str | Yes | Path to STAR .npz model file |
| `--clothing_mesh` | str[] | No | Path(s) to clothing .obj files |
| `--hdri_dir` | str | No | Directory containing HDRI backgrounds |
| `--output_dir` | str | No | Output directory (default: `../outputs/blender_renders`) |
| `--num_samples` | int | No | Number of samples to generate (default: 10) |

## STAR Model Format

The pipeline expects STAR .npz files with the following structure:

```python
{
    'v_template': [6890, 3],      # Template vertices
    'f': [13776, 3],               # Face indices
    'shapedirs': [6890, 3, 10],   # Shape blend shapes
    'posedirs': [6890, 3, 207],   # Pose blend shapes (optional)
    'J_regressor': [24, 6890],    # Joint regressor matrix
    'weights': [6890, 24],         # LBS weights
    'kintree_table': [2, 24],      # Kinematic tree (optional)
}
```

### Where to Get STAR Models

1. **Official STAR Repository**:
   ```bash
   # Download from: https://star.is.tue.mpg.de/
   # Registration required
   ```

2. **Convert from SMPL**:
   ```bash
   # STAR is SMPL-compatible, can adapt SMPL models
   # See: https://github.com/ahmedosman/STAR
   ```

## Clothing Mesh Requirements

### Format

- **File Format**: Wavefront OBJ (.obj)
- **Coordinate System**: Same as STAR (Y-up)
- **Scale**: Metric units (meters)
- **Topology**: Manifold mesh (watertight)

### Creating Clothing Meshes

#### Method 1: Manual Modeling (Blender)

```python
# 1. Import STAR model
# 2. Model clothing around body in T-pose
# 3. Ensure slight gap (0.5-1cm) to avoid intersection
# 4. Export as OBJ
```

#### Method 2: Use Existing Datasets

- **CLOTH3D**: https://chalearnlap.cvc.uab.cat/dataset/38/description/
- **CAPE**: https://cape.is.tue.mpg.de/
- **MGN**: https://github.com/jby1993/MGN (garment meshes)

#### Method 3: Procedural Generation

```python
# Use utils/simple_clothing.py for basic garments
from utils.simple_clothing import SimpleClothingGenerator

generator = SimpleClothingGenerator()
vertices_clothed = generator.add_clothing(
    vertices,
    clothing_types=['sports_bra', 'shorts']
)
# Export as OBJ for Blender
```

## Output Structure

```
outputs/blender_renders/
├── sample_0001_rgb.png          # RGB render
├── sample_0001_normal.png       # Normal map
├── sample_0001_depth.exr        # Depth map
├── sample_0001_params.pkl       # Parameters
├── sample_0002_rgb.png
├── sample_0002_normal.png
├── sample_0002_depth.exr
├── sample_0002_params.pkl
...
```

### Parameter File Structure

```python
import pickle

with open('sample_0001_params.pkl', 'rb') as f:
    params = pickle.load(f)

# params = {
#     'betas': [10],              # Shape parameters
#     'pose': [72],               # Pose parameters
#     'skin_tone': 'medium',      # Skin tone preset
#     'camera_distance': 3.2,
#     'camera_elevation': -5.0,
#     'camera_azimuth': 45.0,
# }
```

## Integration with Sapiens Pipeline

### 1. Generate Blender Renders

```bash
blender --background --python blender/star_clothing_renderer.py -- \
    --star_model data/STAR/female/model.npz \
    --clothing_mesh clothing/sports_bra.obj clothing/shorts.obj \
    --hdri_dir data/hdri_backgrounds \
    --output_dir outputs/blender_renders \
    --num_samples 1000
```

### 2. Run Sapiens Inference

```bash
cd inference
python run_inference.py \
    --input ../outputs/blender_renders/*_rgb.png \
    --output ../outputs/sapiens_blender \
    --tasks depth normal seg
```

### 3. Evaluate Accuracy

```bash
python evaluate_synthetic_data.py \
    --renders outputs/blender_renders \
    --predictions outputs/sapiens_blender \
    --metrics depth_error normal_error seg_iou
```

## Comparison: Pyrender vs Blender

| Feature | Pyrender | Blender |
|---------|----------|---------|
| **Speed** | Fast (~1s/render) | Slow (~30s/render) |
| **Quality** | Good | Photorealistic |
| **Clothing Physics** | No | Yes (cloth simulation) |
| **HDRI Support** | Post-process only | Native |
| **Material Complexity** | Basic | Advanced (SSS, etc.) |
| **Pose Deformation** | Manual | Automatic (armature) |
| **Best For** | Rapid prototyping | Production datasets |

## Performance Optimization

### GPU Acceleration

```python
# In star_clothing_renderer.py (already enabled):
scene.cycles.device = 'GPU'
scene.cycles.samples = 128  # Reduce for faster renders
```

### Batch Rendering

```bash
# Parallel rendering across multiple machines
# Machine 1:
blender --background --python blender/star_clothing_renderer.py -- \
    --star_model data/STAR/female/model.npz \
    --num_samples 500 \
    --output_dir outputs/batch_1

# Machine 2:
blender --background --python blender/star_clothing_renderer.py -- \
    --star_model data/STAR/female/model.npz \
    --num_samples 500 \
    --output_dir outputs/batch_2
```

### Headless Rendering (Server)

```bash
# Install Blender headless
sudo apt-get install blender

# Ensure GPU drivers installed
nvidia-smi

# Run without display
xvfb-run blender --background --python blender/star_clothing_renderer.py -- ...
```

## Troubleshooting

### Issue: "STAR model not found"

**Solution**: Ensure STAR .npz file path is correct:

```bash
ls -la data/STAR/female/model.npz
```

### Issue: "Clothing mesh import failed"

**Solution**: Validate OBJ file format:

```bash
# Check file exists
ls -la clothing/sports_bra.obj

# Validate OBJ structure
head -20 clothing/sports_bra.obj
# Should see: v x y z (vertices), f i1 i2 i3 (faces)
```

### Issue: "GPU not detected"

**Solution**: Check CUDA/OptiX availability:

```python
# In Blender Python console:
import bpy
print(bpy.context.preferences.addons['cycles'].preferences.devices)
```

### Issue: "Renders too dark"

**Solution**: Increase sample count or adjust HDRI:

```python
scene.cycles.samples = 256  # More samples = less noise
node_env.image.colorspace_settings.name = 'Linear'  # Adjust HDRI
```

### Issue: "Clothing intersects body"

**Solution**: Increase clothing expansion or enable collision:

```python
# In SimpleClothingGenerator:
'sports_bra': {'expansion': 1.050}  # Increase from 1.030

# Or ensure collision modifier is applied:
collision_mod.settings.thickness_outer = 0.02
```

## Best Practices for Sapiens-Ready Data

Based on Meta's Sapiens paper and SURREAL dataset methodology:

1. **HDRI Backgrounds**: Use 100+ diverse HDRI environments
2. **Pose Diversity**: Cover full range of natural human poses
3. **Shape Diversity**: Use beta parameters spanning real human variation
4. **Skin Tone Diversity**: Balance across Fitzpatrick scale
5. **Clothing Variation**: Multiple garment types, colors, fits
6. **Camera Angles**: Multiple viewpoints per subject
7. **Lighting Conditions**: Day/night, indoor/outdoor HDRIs

## References

- **STAR Model**: [https://star.is.tue.mpg.de/](https://star.is.tue.mpg.de/)
- **SURREAL Dataset**: [https://www.di.ens.fr/willow/research/surreal/](https://www.di.ens.fr/willow/research/surreal/)
- **Meshcapade SMPL Addon**: [https://github.com/Meshcapade/SMPL_blender_addon](https://github.com/Meshcapade/SMPL_blender_addon)
- **Sapiens**: [https://github.com/facebookresearch/sapiens](https://github.com/facebookresearch/sapiens)
- **Blender Python API**: [https://docs.blender.org/api/current/](https://docs.blender.org/api/current/)

## Future Enhancements

- [ ] Hair/head meshes integration
- [ ] Facial expressions (blend shapes)
- [ ] Hand pose refinement (fingers)
- [ ] Foot contact constraints
- [ ] Multi-person scenes
- [ ] Dynamic pose sequences (animation)
- [ ] Texture variation (wrinkles, logos)
- [ ] Automatic clothing color randomization
- [ ] Integration with motion capture data (BVH/FBX)

## License

This pipeline implementation is released under MIT License. STAR models require separate licensing from the authors.
