#!/usr/bin/env python3
"""
Download HDRI backgrounds from Poly Haven for photorealistic rendering.

This script downloads a curated set of HDRI backgrounds suitable for
human rendering, following Sapiens best practices.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.hdri_background_manager import HDRIBackgroundManager
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Download HDRI backgrounds from Poly Haven'
    )
    parser.add_argument(
        '--category',
        type=str,
        default='all',
        choices=['indoor', 'outdoor', 'neutral', 'all'],
        help='Category of backgrounds to download (default: all)'
    )
    parser.add_argument(
        '--max-count',
        type=int,
        default=15,
        help='Maximum number of backgrounds to download (default: 15)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='data/hdri_backgrounds',
        help='Directory to cache backgrounds (default: data/hdri_backgrounds)'
    )
    parser.add_argument(
        '--resolution',
        type=str,
        default='1k',
        choices=['1k', '2k', '4k'],
        help='Resolution to download (default: 1k)'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("HDRI Background Downloader")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Category: {args.category}")
    print(f"  Max count: {args.max_count}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Cache directory: {args.cache_dir}")
    print()

    # Initialize manager
    manager = HDRIBackgroundManager(
        cache_dir=args.cache_dir,
        image_size=1024
    )

    # Download backgrounds
    count = manager.download_recommended_set(
        category=args.category,
        max_count=args.max_count
    )

    # Show summary
    print("\n" + "="*70)
    print("Download Summary")
    print("="*70)
    print(f"  Successfully downloaded/cached: {count} backgrounds")
    print(f"  Total available: {manager.get_background_count()}")
    print(f"  Cache location: {args.cache_dir}")
    print()

    # List backgrounds
    manager.list_backgrounds()

    # Generate preview montage
    if manager.get_background_count() > 0:
        print("\n" + "="*70)
        print("Generating preview montage...")
        print("="*70)

        from PIL import Image
        import numpy as np

        # Create a grid of backgrounds (up to 12)
        preview_count = min(12, manager.get_background_count())
        rows = 3
        cols = 4
        thumb_size = 256

        montage = np.zeros((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8)

        for i in range(preview_count):
            row = i // cols
            col = i % cols

            # Load background
            bg = manager.load_background(
                manager.backgrounds[i],
                target_size=(thumb_size, thumb_size)
            )

            # Place in montage
            y_start = row * thumb_size
            y_end = y_start + thumb_size
            x_start = col * thumb_size
            x_end = x_start + thumb_size

            montage[y_start:y_end, x_start:x_end] = bg

        # Save montage
        montage_path = os.path.join(args.cache_dir, 'preview_montage.png')
        Image.fromarray(montage).save(montage_path)
        print(f"✓ Saved preview montage: {montage_path}")

    print("\n" + "="*70)
    print("✓ HDRI backgrounds ready for use!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Use these backgrounds with EnhancedPhotorealisticRenderer")
    print("  2. Generate synthetic training data with realistic backgrounds")
    print("  3. Test with Sapiens for improved accuracy")
    print()


if __name__ == "__main__":
    main()
