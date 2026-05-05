"""
03_extract_features.py
Maps to original notebook cell IDs: [4, 5, 6, 7]
Loads VGG16 (top removed), extracts 4096-dim features for all images, saves features.pkl.
Supports both batch extraction and on-demand single-image extraction.
"""

import os
import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

from _01_setup_paths import get_paths


def build_vgg16_feature_extractor():
    """Load VGG16 and restructure to output fc2 (4096-dim) features."""
    print("[INFO] Loading VGG16 (ImageNet weights)...")
    base_model = VGG16()
    # Remove final classification layer; keep up to fc2
    model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
    print(f"[INFO] Feature extractor ready. Output shape: {model.output_shape}")
    return model


def extract_image_features(image_path: str, model) -> np.ndarray:
    """Extract 4096-dim feature vector for a single image path."""
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature


def extract_all_features(images_dir: str, model, output_pkl: str):
    """Extract features for every .jpg in images_dir and pickle them."""
    images_dir = Path(images_dir)
    features = {}
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg")])

    print(f"[INFO] Extracting features for {len(image_files)} images...")
    for img_path in tqdm(image_files, desc="Feature Extraction"):
        feature = extract_image_features(str(img_path), model)
        image_id = img_path.stem  # filename without extension
        features[image_id] = feature

    with open(output_pkl, "wb") as f:
        pickle.dump(features, f)

    print(f"[SUCCESS] Saved features to {output_pkl} ({len(features)} entries)")
    return features


def load_features(features_pkl: str):
    """Load precomputed features from pickle."""
    with open(features_pkl, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGG16 Feature Extraction")
    parser.add_argument("--mode", type=str, default="extract",
                        choices=["extract", "load"],
                        help="extract = build VGG16 and save features.pkl; load = just load existing")
    parser.add_argument("--base", type=str, default=None, help="Override BASE_DIR")
    parser.add_argument("--working", type=str, default=None, help="Override WORKING_DIR")
    parser.add_argument("--image", type=str, default=None,
                        help="Single image path for on-demand extraction (mode=extract)")
    args = parser.parse_args()

    paths = get_paths(base_dir=args.base, working_dir=args.working)

    if args.mode == "load":
        if not paths["FEATURES_PKL"].exists():
            print(f"[ERROR] Features pickle not found: {paths['FEATURES_PKL']}")
            exit(1)
        features = load_features(str(paths["FEATURES_PKL"]))
        print(f"[OK] Loaded {len(features)} features from pickle.")
        exit(0)

    # Extract mode
    model = build_vgg16_feature_extractor()

    if args.image:
        # On-demand single image
        if not Path(args.image).exists():
            print(f"[ERROR] Image not found: {args.image}")
            exit(1)
        feat = extract_image_features(args.image, model)
        print(f"[OK] Extracted feature shape: {feat.shape}")
        print(f"[SAMPLE] First 5 values: {feat[0, :5]}")
    else:
        # Batch extraction
        if not paths["IMAGES_DIR"].exists():
            print(f"[ERROR] Images directory missing: {paths['IMAGES_DIR']}")
            exit(1)
        extract_all_features(str(paths["IMAGES_DIR"]), model, str(paths["FEATURES_PKL"]))
