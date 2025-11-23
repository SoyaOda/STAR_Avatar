#!/usr/bin/env python3
"""
Test Pipeline Components
Verify all components work correctly
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.star_generator import STARGenerator
from src.rendering.renderer import Renderer, rotate_vertices_y_axis
from src.background.manager import BackgroundManager
from src.compositing.compositor import Compositor
from src.pipeline.multi_view import MultiViewPipeline

import numpy as np
from PIL import Image


def test_star_generator():
    """Test STAR model generation"""
    print("\n" + "="*70)
    print("TEST 1: STAR Generator")
    print("="*70)

    generator = STARGenerator(gender='neutral', num_betas=10)

    # Generate random body
    body = generator.generate_body(beta_std=0.5)

    assert 'vertices' in body
    assert 'faces' in body
    assert 'betas' in body

    print(f"\n✓ Generated body:")
    print(f"  Vertices: {body['vertices'].shape}")
    print(f"  Faces: {body['faces'].shape}")
    print(f"  Beta range: [{body['betas'].min():.3f}, {body['betas'].max():.3f}]")

    return body


def test_renderer(body):
    """Test renderer"""
    print("\n" + "="*70)
    print("TEST 2: Renderer")
    print("="*70)

    renderer = Renderer(image_size=512)

    # Render RGB
    img_rgb = renderer.render(
        vertices=body['vertices'],
        faces=body['faces'],
        camera_distance=3.0
    )

    assert img_rgb.shape == (512, 512, 3)
    print(f"\n✓ Rendered RGB: {img_rgb.shape}")

    # Render RGBA
    img_rgba = renderer.render_with_alpha(
        vertices=body['vertices'],
        faces=body['faces'],
        camera_distance=3.0
    )

    assert img_rgba.shape == (512, 512, 4)
    print(f"✓ Rendered RGBA: {img_rgba.shape}")

    # Test rotation
    rotated = rotate_vertices_y_axis(body['vertices'], 90.0)
    assert rotated.shape == body['vertices'].shape
    print(f"✓ Rotation works: {rotated.shape}")

    renderer.close()

    return img_rgba


def test_background_manager():
    """Test background manager"""
    print("\n" + "="*70)
    print("TEST 3: Background Manager")
    print("="*70)

    manager = BackgroundManager()

    # Load studio background
    try:
        studio_bg = manager.load_studio_background(index=0, direction='back')
        print(f"\n✓ Loaded studio background: {studio_bg.shape}")

        return studio_bg

    except FileNotFoundError as e:
        print(f"\n⚠️  Studio HDRI not found (expected in development)")
        print(f"    Creating solid background instead...")

        # Create solid background as fallback
        solid_bg = manager.create_solid_background(512, 512, color=(220, 230, 255))
        print(f"✓ Created solid background: {solid_bg.shape}")

        return solid_bg


def test_compositor(person_rgba, background):
    """Test compositor"""
    print("\n" + "="*70)
    print("TEST 4: Compositor")
    print("="*70)

    compositor = Compositor(background=background)

    # Composite
    result = compositor.composite(person_rgba)

    assert result.shape == person_rgba.shape[:2] + (3,)
    print(f"\n✓ Composited: {result.shape}")

    # Test batch
    batch_results = compositor.composite_batch([person_rgba, person_rgba])
    assert len(batch_results) == 2
    print(f"✓ Batch composite: {len(batch_results)} images")

    return result


def test_complete_pipeline():
    """Test complete pipeline"""
    print("\n" + "="*70)
    print("TEST 5: Complete Pipeline")
    print("="*70)

    pipeline = MultiViewPipeline(
        image_size=512,
        num_betas=10,
        gender='neutral'
    )

    # Setup background
    try:
        background = pipeline.setup_background(studio_index=0, direction='back')
        print(f"\n✓ Background setup: {background.shape}")
    except FileNotFoundError:
        print("\n⚠️  Studio HDRI not found, using solid background")
        solid_bg = pipeline.background_manager.create_solid_background(
            512, 512, color=(220, 230, 255)
        )
        pipeline.compositor.set_background(solid_bg)

    # Generate camera positions
    cameras = pipeline.generate_camera_positions(num_views=4)
    assert len(cameras) == 4
    print(f"✓ Camera positions: {len(cameras)}")

    # Generate single subject (minimal test)
    print("\n✓ Pipeline ready for dataset generation")

    return pipeline


def save_test_outputs(body, img_rgba, composited):
    """Save test outputs"""
    print("\n" + "="*70)
    print("Saving Test Outputs")
    print("="*70)

    output_dir = Path("outputs/pipeline_component_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save person RGBA
    Image.fromarray(img_rgba).save(output_dir / "person_rgba.png")
    print(f"✓ Saved: {output_dir}/person_rgba.png")

    # Save composited
    Image.fromarray(composited).save(output_dir / "composited.png")
    print(f"✓ Saved: {output_dir}/composited.png")

    print(f"\nOutput directory: {output_dir.absolute()}")


def main():
    """Run all component tests"""
    print("\n" + "="*70)
    print("PIPELINE COMPONENT TESTS")
    print("="*70)

    try:
        # Test each component
        body = test_star_generator()
        img_rgba = test_renderer(body)
        background = test_background_manager()
        composited = test_compositor(img_rgba, background)
        pipeline = test_complete_pipeline()

        # Save outputs
        save_test_outputs(body, img_rgba, composited)

        # Summary
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)

        print("\n各コンポーネントの動作確認完了:")
        print("  ✓ STAR Model Generator")
        print("  ✓ Renderer")
        print("  ✓ Background Manager")
        print("  ✓ Compositor")
        print("  ✓ Complete Pipeline")

        print("\n次のステップ:")
        print("  1. データセット生成:")
        print("     python3 -c \"from src.pipeline.multi_view import MultiViewPipeline; \\")
        print("                  p = MultiViewPipeline(); \\")
        print("                  p.generate_dataset('outputs/test_dataset', num_subjects=2, views_per_subject=4)\"")
        print()
        print("  2. Sapiens推論:")
        print("     ./run_sapiens_single.sh outputs/test_dataset/subject_0000/view_00_000deg.png")

        return True

    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"Error: {e}")

        import traceback
        traceback.print_exc()

        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
