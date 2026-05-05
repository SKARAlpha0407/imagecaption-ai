"""
04_load_captions.py
Maps to original notebook cell IDs: [8, 9, 10, 11]
Parses captions.txt, builds mapping dict: image_id -> list of raw captions.
"""

import argparse
from pathlib import Path
from _01_setup_paths import get_paths


def load_captions(captions_file: str):
    """
    Parse captions.txt and return mapping dict.
    Skips header line, handles CSV-like lines: image.jpg,caption text here
    """
    captions_file = Path(captions_file)
    with open(captions_file, "r", encoding="utf-8") as f:
        next(f)  # skip header
        captions_doc = f.read()

    mapping = {}
    lines = captions_doc.strip().split("\n")
    print(f"[INFO] Parsing {len(lines)} lines from {captions_file}...")

    for line in lines:
        if len(line.strip()) < 2:
            continue
        tokens = line.split(",")
        if len(tokens) < 2:
            continue
        image_id = tokens[0].split(".")[0]
        caption = ",".join(tokens[1:]).strip()

        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)

    return mapping


def print_mapping_stats(mapping: dict):
    """Print basic stats about the loaded caption mapping."""
    print("=" * 60)
    print("Caption Mapping Stats")
    print("=" * 60)
    print(f"  Total image IDs: {len(mapping)}")
    total_captions = sum(len(v) for v in mapping.values())
    print(f"  Total captions:  {total_captions}")
    sample_key = list(mapping.keys())[0]
    print(f"  Sample key:      {sample_key}")
    print(f"  Sample captions:")
    for c in mapping[sample_key][:3]:
        print(f"    - {c}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and parse captions.txt")
    parser.add_argument("--base", type=str, default=None, help="Override BASE_DIR")
    parser.add_argument("--working", type=str, default=None, help="Override WORKING_DIR")
    args = parser.parse_args()

    paths = get_paths(base_dir=args.base, working_dir=args.working)

    if not paths["CAPTIONS_FILE"].exists():
        print(f"[ERROR] Captions file not found: {paths['CAPTIONS_FILE']}")
        exit(1)

    mapping = load_captions(str(paths["CAPTIONS_FILE"]))
    print_mapping_stats(mapping)

    # Optionally save raw mapping for inspection
    import pickle
    raw_mapping_path = paths["WORKING_DIR"] / "mapping_raw.pkl"
    with open(raw_mapping_path, "wb") as f:
        pickle.dump(mapping, f)
    print(f"[INFO] Raw mapping saved to {raw_mapping_path}")
