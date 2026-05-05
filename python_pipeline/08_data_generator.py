import numpy as np #type: ignore
import pickle
import argparse
from pathlib import Path

from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore
from tensorflow.keras.utils import to_categorical #type: ignore

import pickle
from setup_paths import get_paths 


def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    """
    Generator that yields batches of training data.
    Matches original notebook logic exactly.
    """
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    # CRITICAL: padding='post' must match inference
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding="post")[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)

            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield (X1, X2), y
                X1, X2, y = list(), list(), list()
                n = 0


def demo_generator():
    """Demonstrate generator output shapes with a tiny sample."""
    paths = get_paths()
    print(f"[INFO] Loading artifacts from {paths['WORKING_DIR']}...")

    # 1. Load features
    with open(paths["FEATURES_PKL"], "rb") as f:
        features = pickle.load(f)

    # 2. Load clean mapping (saved by 05_clean_captions.py)
    with open(paths["WORKING_DIR"] / "mapping_clean.pkl", "rb") as f:
        mapping = pickle.load(f)

    # 3. Load tokenizer bundle (saved by 06_build_tokenizer.py)
    with open(paths["TOKENIZER_PKL"], "rb") as f:
        bundle = pickle.load(f)
    tokenizer = bundle["tokenizer"]
    vocab_size = bundle["vocab_size"]
    max_length = bundle["max_length"]

    # 4. Run demo batch
    image_ids = list(mapping.keys())
    split = int(len(image_ids) * 0.90)
    train = image_ids[:split]

    batch_size = 2
    gen = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)

    (X1, X2), y = next(gen)
    print("=" * 60)
    print("Generator Demo (1 batch)")
    print("=" * 60)
    print(f"  X1 (image features) shape: {X1.shape}")
    print(f"  X2 (text sequences) shape: {X2.shape}")
    print(f"  y  (one-hot labels) shape: {y.shape}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data generator for training")
    parser.add_argument("--demo", action="store_true", help="Run a demo batch and print shapes")
    parser.add_argument("--working", type=str, default=None)
    args = parser.parse_args()

    if args.demo:
        demo_generator()
    else:
        print("[INFO] This script provides data_generator().")
        print("[INFO] Import it in training script (09_train_model.py).")
        print("[USAGE] python 08_data_generator.py --demo")
