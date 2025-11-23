#!/usr/bin/env python3
"""
Example: Generate Multi-View Dataset using Components

This example demonstrates how to use the pipeline components to generate
synthetic multi-view data for Sapiens training.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.pipeline.multi_view import MultiViewPipeline


def main():
    """Generate example dataset"""

    print("\n" + "="*70)
    print("Multi-View Dataset Generation Example")
    print("="*70)

    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = MultiViewPipeline(
        image_size=1024,      # High resolution for Sapiens
        num_betas=30,         # Shape parameters (more for greater variation)
        gender='female'       # Female STAR model
    )

    # Generate dataset
    print("\n2. Generating dataset...")
    summary = pipeline.generate_dataset(
        output_dir='outputs/component_dataset',
        num_subjects=2,           # Number of different bodies
        views_per_subject=4,      # Multi-view captures (0°, 90°, 180°, 270°)
        beta_std=2.5,            # Shape variation (larger for more diversity)
        studio_index=0           # Studio background (best for Sapiens)
    )

    print("\n" + "="*70)
    print("✅ Dataset Generation Complete!")
    print("="*70)

    print(f"\nDataset Summary:")
    print(f"  Output: outputs/component_dataset")
    print(f"  Subjects: {summary['total_subjects']}")
    print(f"  Views per subject: {summary['views_per_subject']}")
    print(f"  Total images: {summary['total_images']}")

    print("\n次のステップ - Sapiens推論を実行:")
    print("\n  単一画像:")
    print("    ./run_sapiens_single.sh outputs/component_dataset/subject_0000/view_00_000deg.png")

    print("\n  全画像:")
    print("    for img in outputs/component_dataset/subject_*/view_*.png; do")
    print("      ./run_sapiens_single.sh \"$img\"")
    print("    done")

    print("\nまたは、Pythonから:")
    print("    pipeline.run_sapiens_on_dataset('outputs/component_dataset')")


if __name__ == "__main__":
    main()
