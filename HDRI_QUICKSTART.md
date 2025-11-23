# HDRI Background Quick Start Guide

## 🚀 3-Step Setup

### Step 1: Download HDRI Backgrounds (1 minute)

```bash
python scripts/download_hdri_backgrounds.py --category all --max-count 15
```

**What this does**: Downloads 15 curated HDRI backgrounds from Poly Haven (free, CC0 license)

**Output**: `data/hdri_backgrounds/` with 15 JPG backgrounds + preview montage

---

### Step 2: Test HDRI Rendering (30 seconds)

```bash
python test_hdri_rendering.py
```

**What this does**: Renders 4 body shapes with/without HDRI backgrounds for comparison

**Output**: `outputs/hdri_test/comparison_montage.png` shows the improvement

---

### Step 3: Generate Sapiens-Ready Data (5 minutes for 100 samples)

```bash
python generate_synthetic_data_with_hdri.py --num-samples 100 --use-hdri
```

**What this does**: Generates 100 photorealistic human images with HDRI backgrounds

**Output**: `outputs/sapiens_synthetic_data/sample_XXXX/front_rgb.png` (ready for Sapiens)

---

## ✅ Verify with Sapiens

```bash
# Test on generated image
python run_sapiens_on_image.py \
    outputs/sapiens_synthetic_data/sample_0001/front_rgb.png
```

Compare depth/normal/segmentation quality with vs without HDRI backgrounds!

---

## 📊 Expected Results

### Without HDRI Background
- Solid color background (gradient/gray)
- Looks synthetic
- Sapiens may have artifacts (noisy depth/normals)

### With HDRI Background
- Realistic environment (studio/street/indoor)
- Photorealistic appearance
- Better Sapiens accuracy (smoother depth/normals)

---

## 🔧 Advanced Usage

### Custom Number of Backgrounds

```bash
# Download more backgrounds for diversity
python scripts/download_hdri_backgrounds.py --max-count 50
```

### Different Categories

```bash
# Indoor only
python scripts/download_hdri_backgrounds.py --category indoor --max-count 10

# Outdoor only
python scripts/download_hdri_backgrounds.py --category outdoor --max-count 10
```

### Higher Resolution

```bash
# 2K resolution for even better quality
python generate_synthetic_data_with_hdri.py \
    --num-samples 10 \
    --image-size 2048 \
    --use-hdri
```

---

## 📖 Full Documentation

See [docs/HDRI_BACKGROUNDS.md](docs/HDRI_BACKGROUNDS.md) for:
- Architecture details
- Programmatic API usage
- Sapiens best practices
- Troubleshooting
- Technical specifications

---

## 💡 Quick Tips

1. **Start small**: Test with 5-10 samples before generating hundreds
2. **Check quality**: Review `comparison_montage.png` to see HDRI improvement
3. **Diverse backgrounds**: More HDRI backgrounds = better model generalization
4. **Sapiens validation**: Always validate with Sapiens to measure improvement

---

## ⚡ One-Liner Full Pipeline

```bash
# Download backgrounds → Test → Generate 100 samples → Validate
python scripts/download_hdri_backgrounds.py --max-count 15 && \
python test_hdri_rendering.py && \
python generate_synthetic_data_with_hdri.py --num-samples 100 --use-hdri && \
echo "✓ Done! Check outputs/sapiens_synthetic_data/"
```

---

## 🆘 Troubleshooting

**Issue**: `No HDRI backgrounds found`
```bash
python scripts/download_hdri_backgrounds.py --category all --max-count 15
```

**Issue**: Rendering fails
- Check pyrender installed: `pip install pyrender`
- Check trimesh installed: `pip install trimesh`

**Issue**: Slow download
- Poly Haven CDN may be slow, wait or reduce `--max-count`

---

## 📈 Performance

- Download 15 backgrounds: ~1 minute
- Render 1 image: ~2-3 seconds
- Generate 100 samples: ~5 minutes
- Storage: ~500 MB for 100 samples (1024px)

---

Enjoy photorealistic STAR Avatar rendering! 🎨
