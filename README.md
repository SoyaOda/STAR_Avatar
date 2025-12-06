# STAR Avatar - Multi-View Synthetic Human Data Generator

Multi-view synthetic human body dataset generator using STAR and MHR body models with Sapiens integration.

## Features

- **STAR Body Model**: Generate diverse 3D human bodies with 30 shape parameters
- **MHR Body Model**: Meta's high-resolution parametric human model (Apache 2.0 license)
- **Photorealistic Rendering**: PBR-based rendering with natural skin tones
- **Multi-View Capture**: Generate images from multiple camera angles (0°, 90°, 180°, 270°)
- **Studio Background**: HDRI-based studio backgrounds optimized for Sapiens
- **Sapiens Integration**: Automatic segmentation, normal maps, depth maps, and pose estimation

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install torch numpy pillow pyrender trimesh

# Setup Sapiens (optional, for inference)
./setup_sapiens.sh
```

### 2. Generate Dataset

```python
python3 example_generate_dataset.py
```

This will generate:
- 2 subjects with different body shapes
- 4 views per subject (front, right, back, left)
- Output: `outputs/component_dataset/`

### 3. Run Sapiens Inference (Optional)

```bash
./run_sapiens_single.sh outputs/component_dataset/subject_0000/view_00_000deg.png
```

## Usage

### Basic Example

```python
from src.pipeline.multi_view import MultiViewPipeline

# Initialize pipeline
pipeline = MultiViewPipeline(
    image_size=1024,
    num_betas=30,
    gender='female'
)

# Generate dataset
pipeline.generate_dataset(
    output_dir='outputs/my_dataset',
    num_subjects=10,
    views_per_subject=8,
    beta_std=2.5,
    studio_index=0
)
```

### Advanced Usage

#### Custom Body Generation

```python
from src.models.star_generator import STARGenerator

# Create generator
generator = STARGenerator(gender='female', num_betas=30)

# Generate random body
body = generator.generate_body(beta_std=2.5)

# Or use specific beta parameters
import numpy as np
custom_betas = np.random.randn(30) * 2.0
body = generator.generate_body(betas=custom_betas)
```

#### Custom Rendering

```python
from src.rendering.renderer import Renderer

renderer = Renderer(image_size=1024)

# Render with alpha channel
person_rgba = renderer.render_with_alpha(
    vertices=body['vertices'],
    faces=body['faces'],
    camera_distance=3.0,
    view='front'
)
```

#### Background Management

```python
from src.background.manager import BackgroundManager

bg_manager = BackgroundManager()

# Load studio background
studio_bg = bg_manager.load_studio_background(
    index=0,  # 0 or 1
    direction='back',
    output_size=(1024, 1024)
)
```

## Configuration

### Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_size` | 1024 | Output image resolution |
| `num_betas` | 30 | Number of shape parameters (max: 300) |
| `gender` | 'female' | Body model gender ('female', 'male', 'neutral') |

### Dataset Generation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_subjects` | 2 | Number of different bodies |
| `views_per_subject` | 4 | Camera angles per subject |
| `beta_std` | 2.5 | Shape variation (1.0-3.0 recommended) |
| `studio_index` | 0 | Studio background (0 or 1) |

### Beta Standard Deviation Guide

- `beta_std=1.0`: Subtle variations
- `beta_std=1.5`: Moderate variations
- `beta_std=2.0`: Clear differences
- `beta_std=2.5`: Strong diversity (recommended)
- `beta_std=3.0`: Extreme variations

## Project Structure

```
STAR_Avatar/
├── src/                          # Core components
│   ├── models/                   # STAR & MHR model generators
│   │   ├── star_layer.py         # STAR model wrapper
│   │   ├── star_generator.py     # STAR body generator
│   │   ├── mhr_layer.py          # MHR model wrapper
│   │   └── mhr_generator.py      # MHR body generator
│   ├── rendering/                # Photorealistic renderer
│   ├── background/               # Background management
│   ├── compositing/              # Image compositing
│   ├── inference/                # Sapiens wrapper
│   └── pipeline/                 # Complete pipeline (STAR & MHR support)
│
├── tests/                        # Tests
│   └── test_pipeline_components.py
│
├── data/                         # Data files
│   ├── star_models/              # STAR model files (.npz)
│   ├── mhr_models/               # MHR model files
│   │   ├── assets/               # Downloaded MHR assets
│   │   └── mhr_mesh_lod1_fixed.npz  # Generated mesh mapping
│   └── hdri_backgrounds/         # Studio backgrounds
│
├── example_generate_dataset.py  # Usage example
├── run_sapiens_single.sh         # Sapiens inference
└── setup_sapiens.sh              # Setup script
```

## Output Structure

```
outputs/component_dataset/
├── studio_background.png         # Background used
├── summary.json                  # Dataset metadata
└── subject_0000/
    ├── metadata.json             # Subject parameters
    ├── view_00_000deg.png        # Front view (0°)
    ├── view_01_090deg.png        # Right view (90°)
    ├── view_02_180deg.png        # Back view (180°)
    └── view_03_270deg.png        # Left view (270°)
```

## Metadata Format

### Dataset Summary (`summary.json`)

```json
{
  "total_subjects": 2,
  "views_per_subject": 4,
  "total_images": 8,
  "studio_background": "outputs/component_dataset/studio_background.png",
  "subjects": [...]
}
```

### Subject Metadata (`metadata.json`)

```json
{
  "subject_idx": 0,
  "betas": [0.42, -0.08, ...],
  "num_views": 4,
  "views": [
    {
      "view_index": 0,
      "filename": "view_00_000deg.png",
      "azimuth": 0.0
    }
  ]
}
```

## Sapiens Integration

### Single Image Inference

```bash
./run_sapiens_single.sh <input_image> [output_dir]
```

Example:
```bash
./run_sapiens_single.sh outputs/component_dataset/subject_0000/view_00_000deg.png
```

Generates:
- Segmentation mask
- Normal map (2B model)
- Depth map (2B model)
- Pose keypoints

### Batch Processing

```python
from src.pipeline.multi_view import MultiViewPipeline

pipeline = MultiViewPipeline()
pipeline.generate_dataset('outputs/dataset', num_subjects=10, views_per_subject=8)

# Run Sapiens on entire dataset
pipeline.run_sapiens_on_dataset('outputs/dataset')
```

## Testing

Run all component tests:

```bash
python3 tests/test_pipeline_components.py
```

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- Pillow
- pyrender
- trimesh

Optional (for Sapiens):
- Sapiens models (see `setup_sapiens.sh`)

## STAR Model Setup

Download STAR models and place in `data/star_models/`:

```
data/star_models/
├── female/
│   └── model.npz
├── male/
│   └── model.npz
└── neutral/
    └── model.npz
```

## Studio Backgrounds

HDRI studio backgrounds are included in `data/hdri_backgrounds/`:
- Index 0: `studio_small_03_1k.jpg`
- Index 1: `studio_small_08_1k.jpg`

These are optimized for Sapiens segmentation (indoor studio lighting).

## Tips

### Increasing Diversity

1. **Increase `num_betas`**: Use 30-50 for more variation dimensions
2. **Increase `beta_std`**: Use 2.0-3.0 for stronger variations
3. **Multiple genders**: Generate separate datasets for each gender

### Best Practices

1. **Beta_std**: Start with 2.0-2.5 for good diversity
2. **Views**: Use 8 views for complete 360° coverage
3. **Studio backgrounds**: Index 0 and 1 work best with Sapiens
4. **Image size**: 1024x1024 recommended for high quality

### Troubleshooting

**Issue**: Bodies look the same
- **Solution**: Increase `beta_std` (try 2.5 or 3.0)

**Issue**: Sapiens segmentation fails
- **Solution**: Use studio backgrounds (index 0 or 1)

**Issue**: STAR model not found
- **Solution**: Check `data/star_models/[gender]/model.npz` exists

## MHR Model Support

STAR_Avatar now supports **MHR (Momentum Human Rig)** - Meta's parametric human body model.

### MHR vs STAR Comparison

| Feature | STAR | MHR |
|---------|------|-----|
| **Vertices** | 6,890 | 18,439 |
| **Faces** | ~13,000 | 36,874 |
| **Shape Parameters** | 10-300 (betas) | 45 (identity) |
| **Pose Parameters** | 72 | 204 |
| **Facial Expression** | No | 72 parameters |
| **Gender Models** | male / female / neutral | Unified (gender encoded in identity params) |
| **License** | Non-commercial only | Apache 2.0 (commercial OK) |

### MHR Setup

1. **Download MHR assets**:

```bash
cd /path/to/STAR_Avatar/data
curl -OL https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip
mkdir -p mhr_models && unzip -o assets.zip -d mhr_models
```

2. **Convert FBX to mesh data** (requires assimp):

```bash
# Install assimp
brew install assimp  # macOS
# apt-get install assimp-utils  # Ubuntu

# Convert FBX to OBJ
cd data/mhr_models/assets
assimp export lod1.fbx lod1.obj
```

3. **Generate mesh mapping** (run once):

```bash
cd /path/to/STAR_Avatar
python3 -c "
import torch
import numpy as np
import trimesh
from scipy.spatial import cKDTree

# Load MHR model vertices
model = torch.jit.load('data/mhr_models/assets/mhr_model.pt', map_location='cpu')
vertices, _ = model(torch.zeros(1, 45), torch.zeros(1, 204), torch.zeros(1, 72))
mhr_verts = vertices[0].numpy()

# Load OBJ mesh
mesh = trimesh.load('data/mhr_models/assets/lod1.obj', force='mesh')

# Map OBJ vertices to MHR vertices
tree = cKDTree(mhr_verts)
distances, obj_to_mhr = tree.query(mesh.vertices, k=1)

# Convert faces
mhr_faces = np.array([[obj_to_mhr[v] for v in face] for face in mesh.faces])

# Save
np.savez('data/mhr_models/mhr_mesh_lod1_fixed.npz', faces=mhr_faces, vertex_mapping=obj_to_mhr)
print('Mesh mapping saved!')
"
```

### MHR Usage

#### Basic Example

```python
from src.pipeline.multi_view import MultiViewPipeline

# Initialize with MHR model
pipeline = MultiViewPipeline(
    image_size=1024,
    num_betas=45,      # MHR uses 45 identity parameters
    model_type='mhr'   # Specify MHR model
)

# Generate dataset
pipeline.generate_dataset(
    output_dir='outputs/mhr_dataset',
    num_subjects=10,
    views_per_subject=4,
    param_std=1.0      # Identity parameter std (0.5-1.5 recommended)
)
```

#### Using MHR Generator Directly

```python
from src.models.mhr_generator import MHRGenerator

# Create generator
generator = MHRGenerator(num_identity=45)

# Generate random body
body = generator.generate_body(identity_std=1.0)

print(f"Vertices: {body['vertices'].shape}")  # (18439, 3)
print(f"Faces: {body['faces'].shape}")        # (36874, 3)
print(f"Height: {body['height_m']:.2f}m")
```

#### Custom Identity Parameters

```python
import numpy as np
from src.models.mhr_generator import MHRGenerator

generator = MHRGenerator(num_identity=45)

# Custom identity (45 parameters)
# First 20: body shape, Next 20: head shape, Last 5: hand shape
custom_identity = np.zeros(45)
custom_identity[0] = 2.0   # Adjust body shape
custom_identity[1] = -1.5  # Adjust another body dimension

body = generator.generate_body(identity=custom_identity)
```

### MHR Identity Parameter Guide

MHR's 45 identity parameters control:
- **Parameters 0-19**: Body shape (height, weight, proportions, etc.)
- **Parameters 20-39**: Head shape
- **Parameters 40-44**: Hand shape

Recommended ranges:
- `param_std=0.5`: Subtle variations
- `param_std=0.8`: Moderate variations
- `param_std=1.0`: Clear diversity (recommended)
- `param_std=1.5`: Strong variations

### MHR Output Structure

```
outputs/mhr_dataset/
├── studio_background.png
├── summary.json
└── subject_0000/
    ├── metadata.json          # Contains identity params, height_m
    ├── view_00_000deg.png
    ├── view_01_090deg.png
    ├── view_02_180deg.png
    └── view_03_270deg.png
```

### MHR Metadata Format

```json
{
  "subject_idx": 0,
  "model_type": "mhr",
  "identity": [0.42, -0.08, ...],
  "height_m": 1.72,
  "num_views": 4,
  "views": [...]
}
```

### MHR vs STAR: When to Use Which?

| Use Case | Recommended Model |
|----------|-------------------|
| Commercial projects | **MHR** (Apache 2.0 license) |
| Higher mesh detail | **MHR** (18K vertices vs 7K) |
| Gender-specific models | **STAR** (separate male/female/neutral) |
| Smaller file size | **STAR** |
| Academic research | Either (both are well-documented) |

### MHR References

- [MHR GitHub Repository](https://github.com/facebookresearch/MHR)
- [MHR Paper (arXiv)](https://arxiv.org/abs/2511.15586)

## License

See individual component licenses.

## Citation

If using STAR model:
```
@inproceedings{STAR:2020,
  title = {{STAR}: A Sparse Trained Articulated Human Body Regressor},
  author = {Osman, Ahmed A A and Bolkart, Timo and Black, Michael J.},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2020}
}
```

If using MHR model:
```
@article{MHR:2024,
  title = {{MHR}: Momentum Human Rig},
  author = {Meta Research},
  journal = {arXiv preprint arXiv:2511.15586},
  year = {2024}
}
```
