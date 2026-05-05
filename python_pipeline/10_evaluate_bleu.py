"""
10_evaluate_bleu.py
Maps to original notebook cell IDs: [27]
Computes BLEU-1 and BLEU-2 scores on the test set using corpus_bleu.
Uses Beam Search + Length Normalization for best BLEU scores.
"""

import os
import warnings
import argparse
import pickle
import numpy as np # type: ignore
from pathlib import Path
from tqdm import tqdm # type: ignore
from nltk.translate.bleu_score import corpus_bleu # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from setup_paths import get_paths

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


def length_penalty(length, alpha=0.7):
    """
    Google's length penalty formula from beam search paper.
    Prevents beam search from favoring short captions.
    alpha=0.7 is the standard value used in most NLP papers.
    Higher alpha = stronger preference for longer sentences.
    """
    return ((5 + length) / 6) ** alpha


def predict_caption_beam(model, feature, tokenizer, max_length, beam_width=5):
    """
    Beam Search decoding with Length Normalization.

    Why beam search beats greedy:
      - Greedy picks the single best word at each step (short-sighted)
      - Beam keeps top-k full sequences alive, picks best complete sentence

    Why length normalization matters:
      - Without it, beam search prefers SHORT captions (lower total log-prob)
      - With it, we divide score by length penalty so longer captions compete fairly
    """
    feature = np.array(feature).reshape(1, -1)  # shape (1, 4096)

    # Start: one beam with just 'startseq', score 0.0
    sequences = [{'text': 'startseq', 'score': 0.0}]
    completed = []

    for _ in range(max_length):
        all_candidates = []

        for seq in sequences:
            words = seq['text'].split()

            # If this sequence already ended, preserve it
            if words[-1] == 'endseq':
                completed.append(seq)
                continue

            # Encode and pad current sequence
            seq_ids = tokenizer.texts_to_sequences([seq['text']])[0]
            padded = pad_sequences([seq_ids], maxlen=max_length, padding='post')

            # Get next-word probabilities from model
            preds = model.predict([feature, padded], verbose=0)[0]

            # Expand: take top beam_width next words
            top_indices = np.argsort(preds)[-beam_width:][::-1]

            for idx in top_indices:
                word = tokenizer.index_word.get(idx)
                if word is None:
                    continue

                new_text = f"{seq['text']} {word}"
                # Log probability accumulation (+ epsilon avoids log(0))
                new_score = seq['score'] + np.log(preds[idx] + 1e-8)
                all_candidates.append({'text': new_text, 'score': new_score})

        if not all_candidates:
            break

        # Keep top beam_width candidates by raw score
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        sequences = all_candidates[:beam_width]

        # Move finished sequences to completed list
        still_going = []
        for seq in sequences:
            if seq['text'].split()[-1] == 'endseq':
                completed.append(seq)
            else:
                still_going.append(seq)
        sequences = still_going

        # Stop early if all beams have ended
        if not sequences:
            break

    # If nothing completed, use whatever is left
    if not completed:
        completed = sequences if sequences else [{'text': 'startseq endseq', 'score': -999}]

    # ---- LENGTH NORMALIZATION ----
    # Without this, beam search unfairly prefers short captions.
    # We normalize each sequence's score by its length penalty.
    def normalized_score(seq):
        words = [w for w in seq['text'].split() if w not in ('startseq', 'endseq')]
        L = max(len(words), 1)
        return seq['score'] / length_penalty(L)

    best = sorted(completed, key=normalized_score, reverse=True)[0]

    # Clean and return
    result = [w for w in best['text'].split() if w not in ('startseq', 'endseq')]
    return ' '.join(result)


def evaluate_bleu(model, mapping, features, tokenizer, max_length, test_keys, beam_width=5):
    actual, predicted = [], []

    for key in tqdm(test_keys, desc=f"Evaluating (beam={beam_width}, length norm ON)"):
        captions = mapping[key]

        # Handle both flat (4096,) and nested [[...]] feature shapes
        feat = features[key]
        if isinstance(feat, (list, np.ndarray)):
            feat = np.array(feat)
            if feat.ndim > 1:
                feat = feat[0]

        y_pred = predict_caption_beam(model, feat, tokenizer, max_length, beam_width)

        actual.append([cap.split() for cap in captions])
        predicted.append(y_pred.split())

    bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    return bleu1, bleu2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model with BLEU scores")
    parser.add_argument("--base",       type=str, default=None)
    parser.add_argument("--working",    type=str, default=None)
    parser.add_argument("--beam-width", type=int, default=5,
                        help="Beam width (default 5). Try 3, 5, 7. Higher = better but slower.")
    args = parser.parse_args()

    paths = get_paths(base_dir=args.base, working_dir=args.working)

    # Load model
    model_path = paths["MODEL_KERAS"]
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        exit(1)
    print(f"[INFO] Loading model from {model_path}...")
    model = tf.keras.models.load_model(str(model_path), 
                                       custom_objects={"BahdanauAttention": BahdanauAttention})

    # Load features
    with open(paths["FEATURES_PKL"], "rb") as f:
        features = pickle.load(f)

    # Load clean mapping
    with open(paths["WORKING_DIR"] / "mapping_clean.pkl", "rb") as f:
        mapping = pickle.load(f)

    # Load tokenizer bundle
    with open(paths["TOKENIZER_PKL"], "rb") as f:
        bundle = pickle.load(f)
    tokenizer  = bundle["tokenizer"]
    max_length = bundle["max_length"]

    # 90/10 train/test split
    image_ids = list(mapping.keys())
    split      = int(len(image_ids) * 0.90)
    test       = image_ids[split:]

    print(f"[INFO] Beam width : {args.beam_width}")
    print(f"[INFO] Test images: {len(test)}")
    print(f"[INFO] Length normalization: ON (alpha=0.7)")
    print()

    bleu1, bleu2 = evaluate_bleu(
        model, mapping, features, tokenizer,
        max_length, test, beam_width=args.beam_width
    )

    print()
    print("=" * 60)
    print(f"  BLEU Scores  (beam={args.beam_width}, length_norm=ON)")
    print("=" * 60)
    print(f"  BLEU-1 : {bleu1:.6f}  ({bleu1*100:.2f}%)")
    print(f"  BLEU-2 : {bleu2:.6f}  ({bleu2*100:.2f}%)")
    print("=" * 60)