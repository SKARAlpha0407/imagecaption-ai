"""
02_download_data.py
Maps to original notebook cell IDs: [2]
Downloads Flickr8k dataset via kagglehub and verifies integrity.
"""

import os
import argparse
import kagglehub
from pathlib import Path


def download_flickr8k(output_dir: str = "/mnt/agents/data/flickr8k") -> str:
    """
    Download the Flickr8k dataset from Kaggle.
    Returns the resolved path where the dataset was downloaded.
    """
    print("[INFO] Downloading Flickr8k dataset via kagglehub...")
    path = kagglehub.dataset_download("adityajn105/flickr8k")
    print(f"[INFO] Kaggle download path: {path}")

    # Kagglehub may return a path like .../flickr8k/Images and captions.txt in parent
    # We expect BASE_DIR to contain both Images/ and captions.txt
    resolved = Path(path).resolve()
    return str(resolved)


def verify_dataset(base_dir: str) -> bool:
    """Verify that expected files and folders exist."""
    base = Path(base_dir)
    images_dir = base / "Images"
    captions_file = base / "captions.txt"

    ok = True
    if not images_dir.exists():
        print(f"[ERROR] Missing Images dir: {images_dir}")
        ok = False
    else:
        num_images = len(list(images_dir.glob("*.jpg")))
        print(f"[OK] Images directory found with {num_images} JPG files.")

    if not captions_file.exists():
        print(f"[ERROR] Missing captions file: {captions_file}")
        ok = False
    else:
        print(f"[OK] Captions file found: {captions_file}")

    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and verify Flickr8k")
    parser.add_argument("--output", type=str, default="/mnt/agents/data/flickr8k",
                        help="Target directory for dataset")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing dataset, do not download")
    args = parser.parse_args()

    if args.verify_only:
        print("[MODE] Verify-only mode")
        success = verify_dataset(args.output)
        exit(0 if success else 1)

    # Download
    downloaded_path = download_flickr8k(output_dir=args.output)
    print(f"[INFO] Dataset downloaded to: {downloaded_path}")

    # If kagglehub returns a nested path, you may symlink/copy to expected output
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)

    # Verify
    success = verify_dataset(args.output)
    if success:
        print("[SUCCESS] Flickr8k dataset is ready.")
    else:
        print("[WARN] Dataset structure unexpected. Inspect manually.")
