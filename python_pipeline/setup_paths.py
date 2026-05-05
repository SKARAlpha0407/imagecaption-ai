"""
01_setup_paths.py
Maps to original notebook cell IDs: [3]
Sets up BASE_DIR, WORKING_DIR, artifact paths, and environment handling.
"""

import os
import argparse
from pathlib import Path


def get_paths(base_dir: str = None, working_dir: str = None):
    """
    Resolve and return all project paths.
    
    Args:
        base_dir: Path to Flickr8k dataset (contains Images/ and captions.txt).
                  Defaults to /Users/shreyaskumarrai/Desktop/QWERTY/hjbg/app/python_pipeline/data or env FLICKR8K_BASE.
        working_dir: Path for pickles, models, plots.
                       Defaults to ./artifacts or env WORKING_DIR.
    
    Returns:
        dict with all resolved paths.
    """
    if base_dir is None:
        base_dir = os.environ.get("FLICKR8K_BASE", "./data/flickr8k")
    if working_dir is None:
        working_dir = os.environ.get("WORKING_DIR", "./artifacts")

    BASE_DIR = Path(base_dir).resolve()
    WORKING_DIR = Path(working_dir).resolve()
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "BASE_DIR": BASE_DIR,
        "WORKING_DIR": WORKING_DIR,
        "IMAGES_DIR": BASE_DIR / "Images",
        "CAPTIONS_FILE": BASE_DIR / "captions.txt",
        "FEATURES_PKL": WORKING_DIR / "features.pkl",
        "TOKENIZER_PKL": WORKING_DIR / "tokenizer.pkl",
        "MODEL_KERAS": WORKING_DIR / "best_model.keras",
        "MODEL_PLOT": WORKING_DIR / "model_architecture.png",
    }

    return paths


def validate_dataset_paths(paths: dict) -> bool:
    """Check that required dataset files exist."""
    ok = True
    if not paths["IMAGES_DIR"].exists():
        print(f"[ERROR] Images directory not found: {paths['IMAGES_DIR']}")
        ok = False
    if not paths["CAPTIONS_FILE"].exists():
        print(f"[ERROR] Captions file not found: {paths['CAPTIONS_FILE']}")
        ok = False
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup project paths")
    parser.add_argument("--base", type=str, default=None, help="Flickr8k dataset root")
    parser.add_argument("--working", type=str, default=None, help="Working/artifacts directory")
    parser.add_argument("--validate", action="store_true", help="Validate dataset paths exist")
    args = parser.parse_args()

    paths = get_paths(base_dir=args.base, working_dir=args.working)

    print("=" * 60)
    print("Project Paths Configured")
    print("=" * 60)
    for key, val in paths.items():
        print(f"  {key:<20} -> {val}")
    print("=" * 60)

    if args.validate:
        is_valid = validate_dataset_paths(paths)
        if is_valid:
            print("[OK] All dataset paths validated successfully.")
        else:
            print("[FAIL] Some dataset paths are missing. Run 02_download_data.py if needed.")
            exit(1)

    # Export paths for downstream scripts
    print("\n[NOTE] Import this module and call get_paths() in other scripts.")
