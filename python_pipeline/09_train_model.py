"""
09_train_model.py (Optimized for BLEU-1 >= 0.60)
Fixes: steps calculation, LR scheduling, data shuffling, batch size, callbacks
"""
import os
import gc
import math
import pickle
import argparse
import numpy as np # type: ignore
from pathlib import Path
import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from setup_paths import get_paths
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class BahdanauAttention(tf.keras.layers.Layer):
    """
    Bahdanau (additive) attention over LSTM sequence outputs.

    Args:
        units (int): Attention hidden size. Must equal the LSTM/Dense units (256).

    Call inputs:
        query:   shape (batch, units)         — the image feature vector fe2
        values:  shape (batch, T, units)      — all LSTM hidden states (return_sequences=True)

    Call output:
        context: shape (batch, units)         — weighted sum of LSTM states
        weights: shape (batch, T, 1)          — attention weights (useful for visualization)
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)   # projects LSTM states
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)   # projects query (image)
        self.V  = tf.keras.layers.Dense(1,     use_bias=False)   # scalar score per timestep

    def call(self, query, values):
        # query shape:  (batch, units)
        # values shape: (batch, T, units)

        # Expand query to broadcast over time steps → (batch, 1, units)
        query_expanded = tf.expand_dims(query, axis=1)

        # Additive score: tanh(W1(values) + W2(query)) → (batch, T, units)
        score = tf.nn.tanh(self.W1(values) + self.W2(query_expanded))

        # Project to scalar → (batch, T, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # Weighted sum of values → (batch, units)
        context = tf.reduce_sum(attention_weights * values, axis=1)

        return context, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

def build_model(vocab_size: int, max_length: int, use_attention: bool = False):
    from tensorflow.keras.models import Model # type: ignore
    from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add # type: ignore
    
    # Encoder: image feature layers
    inputs1 = Input(shape=(4096,), name="image_input")
    fe1 = Dropout(0.3)(inputs1)  # Reduced from 0.4 to prevent underfitting
    fe2 = Dense(256, activation="relu", name="encoder_dense")(fe1)
    
    # Decoder: sequence feature layers
    inputs2 = Input(shape=(max_length,), name="text_input")
    se1 = Embedding(vocab_size, 256, mask_zero=True, name="embedding")(inputs2)
    se2 = Dropout(0.3)(se1)
    
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
    
    decoder2 = Dense(256, activation="relu", name="decoder_dense")(decoder1)
    outputs = Dense(vocab_size, activation="softmax", name="output")(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs, name="ImageCaptioner")
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    return model

def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    """Yields ((X1, X2), y) batches. Shuffles keys at start of each epoch."""
    X1, X2, y = [], [], []
    n = 0
    while True:
        np.random.shuffle(data_keys)  # Critical for convergence
        for key in data_keys:
            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding="post")[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
                n += 1
                if n == batch_size:
                    yield (np.array(X1), np.array(X2)), np.array(y)
                    X1, X2, y = [], [], []
                    n = 0

def train(epochs: int = 30, batch_size: int = 32, working_dir: str = None, base_dir: str = None, use_attention: bool = False):
    paths = get_paths(base_dir=base_dir, working_dir=working_dir)
    
    print("[INFO] Loading artifacts...")
    with open(paths["FEATURES_PKL"], "rb") as f:
        features = pickle.load(f)
    with open(paths["WORKING_DIR"] / "mapping_clean.pkl", "rb") as f:
        mapping = pickle.load(f)
    with open(paths["TOKENIZER_PKL"], "rb") as f:
        bundle = pickle.load(f)
        
    tokenizer = bundle["tokenizer"]
    vocab_size = bundle["vocab_size"]
    max_length = bundle["max_length"]
    
    image_ids = list(mapping.keys())
    np.random.shuffle(image_ids)
    split = int(len(image_ids) * 0.90)
    train_keys = image_ids[:split]
    
    print(f"[INFO] Train: {len(train_keys)} images | Batch: {batch_size} | Epochs: {epochs}")
    print(f"[INFO] use_attention={use_attention}")
    
    model = build_model(vocab_size, max_length, use_attention=use_attention)
    steps = math.ceil(len(train_keys) * 5 / batch_size)  # 5 captions/image → correct steps
    print(f"[INFO] steps_per_epoch = {steps}")
    
    # Callbacks for convergence & BLEU optimization
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(paths["MODEL_KERAS"]), save_best_only=False, verbose=1)
    ]
    
    for i in range(epochs):
        print(f"\n{'='*40} Epoch {i+1}/{epochs} {'='*40}")
        generator = data_generator(train_keys.copy(), mapping, features, tokenizer, max_length, vocab_size, batch_size)
        
        model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1, callbacks=callbacks)
        
        del generator
        gc.collect()
        
        if (i + 1) % 5 == 0:
            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model(str(paths["MODEL_KERAS"]), 
                                               custom_objects={"BahdanauAttention": BahdanauAttention})
            print(f"  🔁 Session cleared & model reloaded at epoch {i+1}")
            
    print(f"[SUCCESS] Training complete. Saved to {paths['MODEL_KERAS']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base", type=str, default=None)
    parser.add_argument("--working", type=str, default=None)
    parser.add_argument("--attention", action="store_true", default=False,
                        help="Use Bahdanau attention in the decoder (default: False)")
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, base_dir=args.base, 
          working_dir=args.working, use_attention=args.attention)