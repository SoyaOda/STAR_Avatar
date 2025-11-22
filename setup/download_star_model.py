"""
Download STAR model files from official source
"""
import os
import urllib.request
import sys

def download_file(url, destination):
    """Download a file with progress bar"""
    print(f"Downloading {url}...")
    print(f"Destination: {destination}")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f'\r[{"=" * (percent // 2)}{" " * (50 - percent // 2)}] {percent}%')
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, destination, progress_hook)
        print(f"\n✓ Downloaded successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Failed to download: {e}")
        return False

def main():
    # Create data directory
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'star_models')
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 60)
    print("STAR Model Download Instructions")
    print("=" * 60)
    print("\nThe STAR models need to be downloaded from the official website:")
    print("https://star.is.tue.mpg.de/")
    print("\nYou will need to:")
    print("1. Register an account")
    print("2. Download the model files:")
    print("   - STAR_NEUTRAL.npz (Gender-neutral model)")
    print("   - STAR_MALE.npz (Male model)")
    print("   - STAR_FEMALE.npz (Female model)")
    print(f"\n3. Place them in: {os.path.abspath(data_dir)}")
    print("\n" + "=" * 60)

    # Alternative: Download from GitHub release if available
    github_url = "https://github.com/ahmedosman/STAR"
    print(f"\nAlternatively, check the official GitHub repository:")
    print(github_url)
    print("\nFor this demo, we'll use a minimal implementation.")

    return data_dir

if __name__ == "__main__":
    data_dir = main()
    print(f"\nModel directory created at: {os.path.abspath(data_dir)}")
    print("\nNext steps:")
    print("1. Download STAR models manually from the official website")
    print("2. Or we'll create a minimal STAR implementation for testing")
