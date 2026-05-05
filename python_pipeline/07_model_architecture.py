
import os
import argparse
import pickle
from pathlib import Path

import tensorflow as tf # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add # type: ignore
from tensorflow.keras.utils import plot_model # type: ignore

import pickle
from setup_paths import get_paths  # Use the name you renamed it to earlier


class BahdanauAttention(tf.keras.layers.Layer):

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)   # projects LSTM states
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)   # projects query (image)
        self.V  = tf.keras.layers.Dense(1,     use_bias=False)   # scalar score per timestep

    def call(self, query, values):
        query_expanded = tf.expand_dims(query, axis=1)

        score = tf.nn.tanh(self.W1(values) + self.W2(query_expanded))

        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # Weighted sum of values → (batch, units)
        context = tf.reduce_sum(attention_weights * values, axis=1)

        return context, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_model(vocab_size: int, max_length: int, use_attention: bool = False) -> Model:
    # Encoder: image feature layers
    inputs1 = Input(shape=(4096,), name="image_input")
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation="relu", name="encoder_dense")(fe1)

    # Decoder: sequence feature layers
    inputs2 = Input(shape=(max_length,), name="text_input")
    se1 = Embedding(vocab_size, 256, mask_zero=True, name="embedding")(inputs2)
    se2 = Dropout(0.4)(se1)

    if use_attention:
        # return_sequences=True → shape (batch, max_length, 256)
        lstm_out = LSTM(256, return_sequences=True, name="lstm")(se2)

        # Attention: query=image features, values=all LSTM states
        attention = BahdanauAttention(units=256, name="bahdanau_attention")
        context, _ = attention(query=fe2, values=lstm_out)  # context: (batch, 256)

        # Merge: same as before — add image features + attended context
        decoder1 = add([fe2, context], name="merge")
    else:
        # Original path — LSTM returns last hidden state only
        se3 = LSTM(256, name="lstm")(se2)                   # (batch, 256)
        decoder1 = add([fe2, se3], name="merge")            # (batch, 256)

    # Output layers
    decoder2 = Dense(256, activation="relu", name="decoder_dense")(decoder1)
    outputs = Dense(vocab_size, activation="softmax", name="output")(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs, name="ImageCaptioner")
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and inspect model architecture")
    parser.add_argument("--working", type=str, default=None, help="Override WORKING_DIR")
    parser.add_argument("--plot", action="store_true", help="Save model architecture PNG")
    parser.add_argument("--attention", action="store_true", default=False,
                        help="Use Bahdanau attention in the decoder (default: False)")
    args = parser.parse_args()

    paths = get_paths(working_dir=args.working)

    # Load tokenizer bundle for vocab_size and max_length
    if not paths["TOKENIZER_PKL"].exists():
        print(f"[ERROR] Tokenizer not found: {paths['TOKENIZER_PKL']}")
        print("[HINT] Run 06_build_tokenizer.py first.")
        exit(1)

    with open(paths["TOKENIZER_PKL"], "rb") as f:
        bundle = pickle.load(f)
    vocab_size = bundle["vocab_size"]
    max_length = bundle["max_length"]
    # Note: `tokenizer` isn't actually used for building the architecture, 
    # only vocab_size and max_length are needed.
    print(f"[INFO] vocab_size={vocab_size}, max_length={max_length}")
    print(f"[INFO] use_attention={args.attention}")

    model = build_model(vocab_size, max_length, use_attention=args.attention)
    print("=" * 60)
    print("Model Architecture" + (" (with Bahdanau Attention)" if args.attention else ""))
    print("=" * 60)
    model.summary()

    if args.plot:
        plot_path = str(paths["MODEL_PLOT"])
        plot_model(model, to_file=plot_path, show_shapes=True, dpi=96)
        print(f"[SUCCESS] Model plot saved: {plot_path}")
