
import os
import gc
import math
import pickle
import argparse
import numpy as np #type: ignore
import tensorflow as tf #type: ignore
import warnings

from setup_paths import get_paths
from model_architecture import build_model, BahdanauAttention   #type: ignore
from data_generator import data_generator                        #type: ignore

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(
    epochs: int = 30,
    batch_size: int = 32,
    working_dir: str = None,
    base_dir: str = None,
    use_attention: bool = False,
):
    paths = get_paths(base_dir=base_dir, working_dir=working_dir)

    # ── Load artifacts ────────────────────────────────────────────────
    print("[INFO] Loading artifacts...")
    with open(paths["FEATURES_PKL"], "rb") as f:
        features = pickle.load(f)

    with open(paths["WORKING_DIR"] / "mapping_clean.pkl", "rb") as f:
        mapping = pickle.load(f)

    with open(paths["TOKENIZER_PKL"], "rb") as f:
        bundle = pickle.load(f)

    tokenizer  = bundle["tokenizer"]
    vocab_size = bundle["vocab_size"]
    max_length = bundle["max_length"]

    # ── Train / val split ─────────────────────────────────────────────
    image_ids = list(mapping.keys())
    np.random.shuffle(image_ids)
    split      = int(len(image_ids) * 0.90)
    train_keys = image_ids[:split]

    print(f"[INFO] Train images : {len(train_keys)}")
    print(f"[INFO] Batch size   : {batch_size}")
    print(f"[INFO] Epochs       : {epochs}")
    print(f"[INFO] use_attention: {use_attention}")

    # ── Build model ───────────────────────────────────────────────────
    model = build_model(vocab_size, max_length, use_attention=use_attention)
    model.summary()

    # 5 captions per image → correct steps_per_epoch
    steps = math.ceil(len(train_keys) * 5 / batch_size)
    print(f"[INFO] steps_per_epoch = {steps}")

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(paths["MODEL_KERAS"]), save_best_only=False, verbose=1
        ),
    ]
    for i in range(epochs):
        print(f"\n{'='*40} Epoch {i+1}/{epochs} {'='*40}")

        generator = data_generator(
            train_keys.copy(), mapping, features,
            tokenizer, max_length, vocab_size, batch_size
        )

        model.fit(
            generator,
            epochs=1,
            steps_per_epoch=steps,
            verbose=1,
            callbacks=callbacks,
        )

        del generator
        gc.collect()

        # Periodic session clear to prevent TF memory fragmentation
        if (i + 1) % 5 == 0:
            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model(
                str(paths["MODEL_KERAS"]),
                custom_objects={"BahdanauAttention": BahdanauAttention},
            )
            print(f"  [INFO] Session cleared & model reloaded at epoch {i+1}")

    print(f"[SUCCESS] Training complete. Model saved to {paths['MODEL_KERAS']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the image captioning model")
    parser.add_argument("--epochs",     type=int,  default=30)
    parser.add_argument("--batch-size", type=int,  default=32)
    parser.add_argument("--base",       type=str,  default=None, help="Override BASE_DIR")
    parser.add_argument("--working",    type=str,  default=None, help="Override WORKING_DIR")
    parser.add_argument("--attention",  action="store_true", default=False,
                        help="Use Bahdanau attention in the decoder")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_dir=args.base,
        working_dir=args.working,
        use_attention=args.attention,
    )