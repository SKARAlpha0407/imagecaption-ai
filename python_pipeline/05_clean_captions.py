"""
05_clean_captions.py
Maps to original notebook cell IDs: [12, 13]
Cleans captions: lowercase, regex remove special chars (fixed), collapse whitespace,
filter short words, add startseq/endseq.
FIXES: SyntaxWarning on invalid escape sequence '\s' by using raw string r'\s+'.
"""

import re
import pickle
import argparse
from pathlib import Path
import pickle
from setup_paths import get_paths  # Use the renamed file you created earlier


def clean_captions(mapping: dict) -> dict:
    """
    Clean all captions in mapping.
    Original logic preserved; regex fixed with raw string.
    """
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            # Convert to lowercase
            caption = caption.lower()
            # Remove digits and special characters (keep A-Z, a-z, spaces)
            # NOTE: Original used replace('[^A-Za-z]', '') which is wrong; we use re.sub
            caption = re.sub(r"[^a-z\s]", "", caption)
            # Collapse multiple spaces into single space (FIXED: raw string)
            caption = re.sub(r"\s+", " ", caption)
            # Add start/end tags, filter words with len > 1
            caption = "startseq " + " ".join([word for word in caption.split() if len(word) > 1]) + " endseq"
            captions[i] = caption
    return mapping


def save_clean_mapping(mapping: dict, path: str):
    with open(path, "wb") as f:
        pickle.dump(mapping, f)
    print(f"[INFO] Clean mapping saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean captions mapping")
    parser.add_argument("--base", type=str, default=None, help="Override BASE_DIR")
    parser.add_argument("--working", type=str, default=None, help="Override WORKING_DIR")
    parser.add_argument("--inspect", action="store_true", help="Print before/after sample")
    args = parser.parse_args()

    paths = get_paths(base_dir=args.base, working_dir=args.working)

    # # Load raw mapping
    # raw_mapping = load_captions(str(paths["CAPTIONS_FILE"]))
    # ✅ ADD this instead (load the .pkl file created by 04_load_captions.py):
    with open(paths["WORKING_DIR"] / "mapping_raw.pkl", "rb") as f:
        raw_mapping = pickle.load(f)
        sample_key = list(raw_mapping.keys())[0]

    if args.inspect:
        print("=" * 60)
        print("BEFORE cleaning (sample)")
        print("=" * 60)
        for c in raw_mapping[sample_key][:2]:
            print(f"  {c}")

    clean_mapping = clean_captions(raw_mapping)

    if args.inspect:
        print("=" * 60)
        print("AFTER cleaning (sample)")
        print("=" * 60)
        for c in clean_mapping[sample_key][:2]:
            print(f"  {c}")

    save_clean_mapping(clean_mapping, str(paths["WORKING_DIR"] / "mapping_clean.pkl"))

    all_captions = []
    for key in clean_mapping:
        for caption in clean_mapping[key]:
            all_captions.append(caption)
    print(f"[INFO] Total clean captions: {len(all_captions)}")
    print(f"[INFO] Sample: {all_captions[:3]}")
