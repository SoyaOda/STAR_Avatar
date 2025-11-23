# STAR Avatar - Multi-View Synthetic Human Data Generator

Multi-view synthetic human body dataset generator using STAR body model with Sapiens integration.

## Features

- **STAR Body Model**: Generate diverse 3D human bodies with 30 shape parameters
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
│   ├── models/                   # STAR model & generator
│   ├── rendering/                # Photorealistic renderer
│   ├── background/               # Background management
│   ├── compositing/              # Image compositing
│   ├── inference/                # Sapiens wrapper
│   └── pipeline/                 # Complete pipeline
│
├── tests/                        # Tests
│   └── test_pipeline_components.py
│
├── data/                         # Data files
│   ├── star_models/              # STAR model files
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
