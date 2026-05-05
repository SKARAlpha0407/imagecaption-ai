"""
06_build_tokenizer.py
Maps to original notebook cell IDs: [14, 15, 16, 17, 18, 19]
Builds Keras Tokenizer from clean captions, computes vocab_size and max_length,
saves tokenizer.pkl for inference use.
"""

import pickle
import argparse
from pathlib import Path
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from _01_setup_paths import get_paths
# from _04_load_captions import load_captions
# from _05_clean_captions import clean_captions
# ✅ CHANGE to:
from setup_paths import get_paths


def build_tokenizer(all_captions: list):
    """Fit Keras Tokenizer on all captions."""
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"[INFO] Vocab size: {vocab_size}")
    return tokenizer, vocab_size


def compute_max_length(all_captions: list) -> int:
    """Get maximum caption length in words."""
    max_length = max(len(caption.split()) for caption in all_captions)
    print(f"[INFO] Max caption length: {max_length}")
    return max_length


def save_tokenizer(tokenizer, vocab_size: int, max_length: int, output_path: str):
    """Save tokenizer and metadata as a single pickle bundle."""
    bundle = {
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "max_length": max_length,
        "word_index": tokenizer.word_index,
    }
    with open(output_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[SUCCESS] Tokenizer bundle saved: {output_path}")


def load_tokenizer_bundle(tokenizer_pkl: str):
    """Load tokenizer bundle (for downstream scripts / inference)."""
    with open(tokenizer_pkl, "rb") as f:
        bundle = pickle.load(f)
    return bundle["tokenizer"], bundle["vocab_size"], bundle["max_length"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and save Keras Tokenizer")
    parser.add_argument("--base", type=str, default=None, help="Override BASE_DIR")
    parser.add_argument("--working", type=str, default=None, help="Override WORKING_DIR")
    parser.add_argument("--load-only", action="store_true", help="Load existing tokenizer bundle and print stats")
    args = parser.parse_args()

    paths = get_paths(base_dir=args.base, working_dir=args.working)

    if args.load_only:
        if not paths["TOKENIZER_PKL"].exists():
            print(f"[ERROR] Tokenizer not found: {paths['TOKENIZER_PKL']}")
            exit(1)
        tok, vs, ml = load_tokenizer_bundle(str(paths["TOKENIZER_PKL"]))
        print(f"[OK] Tokenizer loaded. vocab_size={vs}, max_length={ml}")
        exit(0)

    # # Build from captions
    # raw_mapping = load_captions(str(paths["CAPTIONS_FILE"]))
    # clean_mapping = clean_captions(raw_mapping)

    # ✅ LOAD PRE-CLEANED MAPPING FROM STEP 05
    with open(paths["WORKING_DIR"] / "mapping_clean.pkl", "rb") as f:
        clean_mapping = pickle.load(f)
    print(f"[INFO] Loaded {len(clean_mapping)} cleaned image-caption mappings")

    all_captions = []
    for key in clean_mapping:
        for caption in clean_mapping[key]:
            all_captions.append(caption)

    print(f"[INFO] Fitting tokenizer on {len(all_captions)} captions...")
    tokenizer, vocab_size = build_tokenizer(all_captions)
    max_length = compute_max_length(all_captions)

    save_tokenizer(tokenizer, vocab_size, max_length, str(paths["TOKENIZER_PKL"]))
