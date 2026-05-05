"""
12_demo_inference.py
Maps to original notebook cell IDs: [30, 31]
Loads a trained model, tokenizer, and features; runs prediction on a single image;
displays actual vs predicted captions with matplotlib rendering.
"""

import os
import warnings
import pickle
import argparse
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from pathlib import Path
from PIL import Image # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from setup_paths import get_paths  # ✅ Fixed import

# Suppress harmless TF/Protobuf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


class BahdanauAttention(tf.keras.layers.Layer):
    """Bahdanau attention layer - needed for loading models trained with attention."""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)
        self.V  = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, query, values):
        query_expanded = tf.expand_dims(query, axis=1)
        score = tf.nn.tanh(self.W1(values) + self.W2(query_expanded))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context = tf.reduce_sum(attention_weights * values, axis=1)
        return context, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

def predict_caption(model, feature, tokenizer, max_length):
    """Greedy search caption generation (matches notebook cell 25/26)."""
    in_text = "startseq"
    # Ensure feature is 1D (4096,)
    feature = np.array(feature).reshape(1, -1) if hasattr(feature, 'shape') else feature
    
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding="post")
        yhat = model.predict([feature, sequence], verbose=0)[0]
        word_id = np.argmax(yhat)
        word = tokenizer.index_word.get(word_id, None)
        if word is None or word == "endseq" or word == "startseq":
            break
        in_text += " " + word
    return " ".join([w for w in in_text.split() if w not in ("startseq", "endseq")])

def generate_caption_for_image(image_file, model, mapping, features, tokenizer, max_length, base_dir, show_plot=True):
    # Extract just the filename (handles full paths passed via CLI)
    image_name = Path(image_file).name
    image_id = Path(image_name).stem
    img_path = Path(base_dir) / "Images" / image_name

    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}")
        return None

    image = Image.open(img_path)

    print("-" * 50)
    print("ACTUAL CAPTIONS")
    print("-" * 50)
    actual_captions = mapping.get(image_id, [])
    for caption in actual_captions:
        print(caption)

    # Use precomputed feature
    if image_id in features:
        image_feature = features[image_id]
    else:
        print(f"[WARN] Feature for {image_id} not found in features.pkl")
        return None

    y_pred = predict_caption(model, image_feature, tokenizer, max_length)
    print("-" * 50)
    print("PREDICTED CAPTION")
    print("-" * 50)
    print(y_pred)

    if show_plot:
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Predicted: {y_pred}", wrap=True)
        plt.tight_layout()
        plt.show()

    return y_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo inference on a single image")
    parser.add_argument("--image", type=str, required=True, help="Image path or filename")
    parser.add_argument("--base", type=str, default=None)
    parser.add_argument("--working", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true", help="Print only, skip matplotlib")
    args = parser.parse_args()

    paths = get_paths(base_dir=args.base, working_dir=args.working)

    # Load artifacts directly from disk
    model = tf.keras.models.load_model(str(paths["MODEL_KERAS"]),
                                       custom_objects={"BahdanauAttention": BahdanauAttention})
    
    with open(paths["FEATURES_PKL"], "rb") as f:
        features = pickle.load(f)
        
    with open(paths["WORKING_DIR"] / "mapping_clean.pkl", "rb") as f:
        mapping = pickle.load(f)
        
    with open(paths["TOKENIZER_PKL"], "rb") as f:
        bundle = pickle.load(f)
    tokenizer = bundle["tokenizer"]
    max_length = bundle["max_length"]

    generate_caption_for_image(
        image_file=args.image,
        model=model,
        mapping=mapping,
        features=features,
        tokenizer=tokenizer,
        max_length=max_length,
        base_dir=str(paths["BASE_DIR"]),
        show_plot=not args.no_plot,
    )