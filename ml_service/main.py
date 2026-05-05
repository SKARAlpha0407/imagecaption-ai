"""
FastAPI ML Inference Microservice
Provides /predict endpoint for image captioning.
Loads pre-trained artifacts: best_model.keras, tokenizer.pkl
Extracts features on-the-fly using VGG16.
Uses beam search (width=5) for better caption quality.
"""

import os
import io
import pickle
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np # type: ignore
import uvicorn # type: ignore
from fastapi import FastAPI, File, UploadFile, HTTPException # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from PIL import Image # type: ignore

import tensorflow as tf # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from tensorflow.keras.models import Model # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ml_service")

# Configuration from environment
WORKING_DIR = Path(os.environ.get("WORKING_DIR", "./artifacts")).resolve()
MODEL_PATH = WORKING_DIR / "best_model.keras"
TOKENIZER_PATH = WORKING_DIR / "tokenizer.pkl"
MAX_IMAGE_SIZE_MB = int(os.environ.get("MAX_IMAGE_SIZE_MB", "5"))

# Global model references (loaded on startup)
vgg_extractor = None
caption_model = None
tokenizer = None
vocab_size = None
max_length = None


def build_vgg16_feature_extractor():
    """Load VGG16 and remove final classification layer."""
    base = VGG16(weights="imagenet")
    model = Model(inputs=base.inputs, outputs=base.layers[-2].output)
    return model


def load_artifacts():
    """Load all pre-trained artifacts into global variables."""
    global vgg_extractor, caption_model, tokenizer, vocab_size, max_length

    logger.info(f"Loading artifacts from {WORKING_DIR}...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    # Load tokenizer bundle
    with open(TOKENIZER_PATH, "rb") as f:
        bundle = pickle.load(f)
    tokenizer = bundle["tokenizer"]
    vocab_size = bundle["vocab_size"]
    max_length = bundle["max_length"]

    # Load caption model
    caption_model = tf.keras.models.load_model(str(MODEL_PATH))

    # Load VGG16 feature extractor
    vgg_extractor = build_vgg16_feature_extractor()

    logger.info(f"Artifacts loaded. vocab_size={vocab_size}, max_length={max_length}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model artifacts. Shutdown: cleanup."""
    logger.info("[STARTUP] Loading ML artifacts...")
    try:
        load_artifacts()
        logger.info("[STARTUP] ML service ready.")
    except Exception as e:
        logger.error(f"[STARTUP] Failed to load artifacts: {e}")
        raise
    yield
    logger.info("[SHUTDOWN] Cleaning up...")
    tf.keras.backend.clear_session()


app = FastAPI(
    title="Image Captioning ML Service",
    description="FastAPI microservice for VGG16+LSTM image captioning inference",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def idx_to_word(integer, tokenizer):
    """Convert word index back to word string."""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image_feature, tokenizer, max_length, beam_width=5):
    """
    Beam search caption generation with post-padding (matches training).
    beam_width=5 gives significantly better results than greedy search.
    Each candidate: (sequence_of_word_indices, cumulative_log_score)
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

    start_idx = tokenizer.word_index.get("startseq", 1)
    end_idx = tokenizer.word_index.get("endseq", 2)

    # Start with [startseq], score 0.0
    sequences = [([start_idx], 0.0)]

    for _ in range(max_length):
        all_candidates = []

        for seq, score in sequences:
            # If this sequence already ended, keep it as-is
            if seq[-1] == end_idx:
                all_candidates.append((seq, score))
                continue

            # Pad and predict
            padded = pad_sequences([seq], maxlen=max_length, padding="post")
            preds = model.predict([image_feature, padded], verbose=0)[0]

            # Take top beam_width predictions
            top_k = np.argsort(preds)[-beam_width:]
            for word_idx in top_k:
                prob = preds[word_idx]
                new_score = score - np.log(prob + 1e-10)  # negative log prob (lower = better)
                all_candidates.append((seq + [int(word_idx)], new_score))

        # Keep only top beam_width candidates (sorted by score, lower is better)
        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

        # Early stop if all sequences have ended
        if all(seq[-1] == end_idx for seq, _ in sequences):
            break

    # Pick best sequence
    best_seq = sequences[0][0]

    # Convert indices to words, strip startseq/endseq
    words = []
    for idx in best_seq:
        word = idx_to_word(idx, tokenizer)
        if word and word not in ("startseq", "endseq"):
            words.append(word)

    caption = " ".join(words).strip()
    return caption


def extract_feature_from_upload(upload_file: UploadFile) -> np.ndarray:
    """Read uploaded image, preprocess, and extract VGG16 features."""
    contents = upload_file.file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large: {size_mb:.1f}MB (max {MAX_IMAGE_SIZE_MB}MB)",
        )

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))
    arr = img_to_array(image)
    arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
    arr = preprocess_input(arr)
    feature = vgg_extractor.predict(arr, verbose=0)
    return feature


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": caption_model is not None,
        "vocab_size": vocab_size,
        "max_length": max_length,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an image file and return predicted caption using beam search.
    """
    if caption_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    allowed = {"image/jpeg", "image/png", "image/webp"}
    content_type = file.content_type
    if content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Allowed: {allowed}",
        )

    try:
        feature = extract_feature_from_upload(file)
        caption = predict_caption(caption_model, feature, tokenizer, max_length, beam_width=5)

        # Confidence: based on meaningful word count (more words = more confident description)
        word_count = len(caption.split())
        confidence = round(min(word_count / 10.0, 1.0), 4)  # 10 words = 100%

        return {
            "success": True,
            "caption": caption,
            "confidence": confidence,
            "model": "vgg16_lstm",
            "vocab_size": vocab_size,
            "max_length": max_length,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Image Captioning ML Service", "docs": "/docs"}


if __name__ == "__main__":
    port = int(os.environ.get("ML_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)