
import os
import warnings
import argparse
import pickle
import numpy as np #type: ignore
from tqdm import tqdm #type: ignore
from nltk.translate.bleu_score import corpus_bleu #type: ignore
import tensorflow as tf #type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore

from setup_paths import get_paths
from model_architecture import BahdanauAttention     #type: ignore

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


def length_penalty(length, alpha=0.7):

    return ((5 + length) / 6) ** alpha


def predict_caption_beam(model, feature, tokenizer, max_length, beam_width=5):

    feature = np.array(feature).reshape(1, -1)          # shape (1, 4096)
    sequences = [{'text': 'startseq', 'score': 0.0}]
    completed = []

    for _ in range(max_length):
        all_candidates = []

        for seq in sequences:
            words = seq['text'].split()

            if words[-1] == 'endseq':
                completed.append(seq)
                continue

            seq_ids = tokenizer.texts_to_sequences([seq['text']])[0]
            padded  = pad_sequences([seq_ids], maxlen=max_length, padding='post')
            preds   = model.predict([feature, padded], verbose=0)[0]

            for idx in np.argsort(preds)[-beam_width:][::-1]:
                word = tokenizer.index_word.get(idx)
                if word is None:
                    continue
                all_candidates.append({
                    'text':  f"{seq['text']} {word}",
                    'score': seq['score'] + np.log(preds[idx] + 1e-8),
                })

        if not all_candidates:
            break

        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        sequences = all_candidates[:beam_width]

        still_going = []
        for seq in sequences:
            (completed if seq['text'].split()[-1] == 'endseq' else still_going).append(seq)
        sequences = still_going

        if not sequences:
            break

    if not completed:
        completed = sequences if sequences else [{'text': 'startseq endseq', 'score': -999}]

    def normalized_score(seq):
        words = [w for w in seq['text'].split() if w not in ('startseq', 'endseq')]
        return seq['score'] / length_penalty(max(len(words), 1))

    best   = sorted(completed, key=normalized_score, reverse=True)[0]
    result = [w for w in best['text'].split() if w not in ('startseq', 'endseq')]
    return ' '.join(result)


# ── BLEU evaluation ───────────────────────────────────────────────────────────

def evaluate_bleu(model, mapping, features, tokenizer, max_length, test_keys, beam_width=5):
    actual, predicted = [], []

    for key in tqdm(test_keys, desc=f"Evaluating (beam={beam_width}, length norm ON)"):
        feat = np.array(features[key])
        if feat.ndim > 1:
            feat = feat[0]

        y_pred = predict_caption_beam(model, feat, tokenizer, max_length, beam_width)

        actual.append([cap.split() for cap in mapping[key]])
        predicted.append(y_pred.split())

    bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0,   0, 0))
    bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    return bleu1, bleu2


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model with BLEU scores")
    parser.add_argument("--base",       type=str, default=None)
    parser.add_argument("--working",    type=str, default=None)
    parser.add_argument("--beam-width", type=int, default=5,
                        help="Beam width (default 5). Try 3, 5, 7. Higher = better but slower.")
    args = parser.parse_args()

    paths = get_paths(base_dir=args.base, working_dir=args.working)

    # ── Load model ────────────────────────────────────────────────────
    if not paths["MODEL_KERAS"].exists():
        print(f"[ERROR] Model not found: {paths['MODEL_KERAS']}")
        exit(1)
    print(f"[INFO] Loading model from {paths['MODEL_KERAS']}...")
    model = tf.keras.models.load_model(
        str(paths["MODEL_KERAS"]),
        custom_objects={"BahdanauAttention": BahdanauAttention},
    )

    # ── Load artifacts ────────────────────────────────────────────────
    with open(paths["FEATURES_PKL"], "rb") as f:
        features = pickle.load(f)

    with open(paths["WORKING_DIR"] / "mapping_clean.pkl", "rb") as f:
        mapping = pickle.load(f)

    with open(paths["TOKENIZER_PKL"], "rb") as f:
        bundle = pickle.load(f)
    tokenizer  = bundle["tokenizer"]
    max_length = bundle["max_length"]

    # ── 90 / 10 split (same seed-free split as training) ─────────────
    image_ids = list(mapping.keys())
    test      = image_ids[int(len(image_ids) * 0.90):]

    print(f"[INFO] Beam width          : {args.beam_width}")
    print(f"[INFO] Test images         : {len(test)}")
    print(f"[INFO] Length normalization: ON (alpha=0.7)")
    print()

    bleu1, bleu2 = evaluate_bleu(
        model, mapping, features, tokenizer,
        max_length, test, beam_width=args.beam_width,
    )

    print()
    print("=" * 60)
    print(f"  BLEU Scores  (beam={args.beam_width}, length_norm=ON)")
    print("=" * 60)
    print(f"  BLEU-1 : {bleu1:.6f}  ({bleu1*100:.2f}%)")
    print(f"  BLEU-2 : {bleu2:.6f}  ({bleu2*100:.2f}%)")
    print("=" * 60)