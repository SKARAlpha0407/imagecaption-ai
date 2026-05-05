"""
11_predict_caption.py
Maps to original notebook cell IDs: [25, 26, 27 (predict_caption)]
Implements idx_to_word and predict_caption with greedy decoding.
CRITICAL FIX: padding='post' in predict_caption to match training.
"""

import numpy as np # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore


def idx_to_word(integer, tokenizer):
    """Convert token index back to word string."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image, tokenizer, max_length):
    """
    Generate a caption for an image using greedy decoding.
    MUST use padding='post' to match training generator.
    """
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # CRITICAL FIX: padding='post' must match training (08_data_generator.py)
        sequence = pad_sequences([sequence], maxlen=max_length, padding="post")
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    return in_text


if __name__ == "__main__":
    print("[INFO] This module provides:")
    print("  - idx_to_word(integer, tokenizer)")
    print("  - predict_caption(model, image, tokenizer, max_length)")
    print("[USAGE] Import into 12_demo_inference.py or the FastAPI service.")
