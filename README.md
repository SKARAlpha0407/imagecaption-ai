# ImageCaption AI — Production-Ready MERN + VGG16+LSTM

A full-stack image captioning web application combining cutting-edge deep learning with modern web technologies. Built with **React + Vite** frontend, **Express.js** backend, **MongoDB** persistence, and **FastAPI** ML microservice running **VGG16+LSTM** on Flickr8k dataset. This project preserves 100% of the original research logic, splits it into modular Python scripts for academic demonstration, and deploys in a production-grade MERN stack.

---

## 1. Architecture & Data Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   React Frontend │────▶  Express Server  │────▶  FastAPI ML    │
│   (Vite + TW)   │     │  (Node.js)       │     │  (TensorFlow)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         ▲                       │                         │
         │                       ▼                         ▼
         │               ┌──────────────────┐     ┌─────────────────┐
         └───────────────│    MongoDB       │     │  Pre-trained    │
                        │   (Mongoose)     │     │  Artifacts      │
                        └──────────────────┘     │  .keras/.pkl    │
                                                  └─────────────────┘
```

**Request Flow:**
1. User uploads image via React drag-and-drop
2. Express receives multipart upload, validates (max 5MB, JPG/PNG/WebP)
3. Express proxies image to FastAPI `/predict`
4. FastAPI extracts VGG16 features, runs LSTM decoder, returns caption
5. Express stores result in MongoDB, returns JSON to React
6. React displays caption + confidence, updates history grid

---

## 2. Python Pipeline Structure (01–12)

Each script maps to original notebook cells and can be run independently for classroom demos.

| Script | Notebook Cell | Purpose | CLI Example |
|--------|---------------|---------|-------------|
| `01_setup_paths.py` | [3] | Configure BASE_DIR, WORKING_DIR, artifact paths | `python 01_setup_paths.py --validate` |
| `02_download_data.py` | [2] | Download Flickr8k via kagglehub | `python 02_download_data.py --output ./data` |
| `03_extract_features.py` | [4,5,6,7] | VGG16 feature extraction (4096-dim) | `python 03_extract_features.py --mode extract` |
| `04_load_captions.py` | [8,9,10,11] | Parse captions.txt into mapping dict | `python 04_load_captions.py` |
| `05_clean_captions.py` | [12,13] | Clean captions, fix `\s+` regex with raw string | `python 05_clean_captions.py --inspect` |
| `06_build_tokenizer.py` | [14-19] | Fit Tokenizer, save vocab_size, max_length | `python 06_build_tokenizer.py` |
| `07_model_architecture.py` | [22] | Define Encoder-Decoder, export plot | `python 07_model_architecture.py --plot` |
| `08_data_generator.py` | [21] | Batch generator yielding (X1,X2), y | `python 08_data_generator.py --demo` |
| `09_train_model.py` | [20,23,24] | 30-epoch loop, gc.collect, clear_session | `python 09_train_model.py --epochs 30` |
| `10_evaluate_bleu.py` | [27] | BLEU-1 & BLEU-2 scoring on test set | `python 10_evaluate_bleu.py` |
| `11_predict_caption.py` | [25,26] | Greedy inference with post-padding | Import module |
| `12_demo_inference.py` | [30,31] | Single-image prediction + matplotlib | `python 12_demo_inference.py --image img.jpg` |

### Quick Sequential Demo (for presentations)

```bash
cd python_pipeline
python 01_setup_paths.py --validate
python 02_download_data.py --verify-only
python 03_extract_features.py --mode extract
python 04_load_captions.py
python 05_clean_captions.py --inspect
python 06_build_tokenizer.py
python 07_model_architecture.py --plot
python 08_data_generator.py --demo
python 09_train_model.py --epochs 30
python 10_evaluate_bleu.py
python 12_demo_inference.py --image 1001773457_577c3a7d70.jpg --no-plot
```

---

## 3. Core Code Blocks

### 3.1 FastAPI Inference (`ml_service/main.py`)

```python
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    feature = extract_feature_from_upload(file)
    caption = predict_caption(caption_model, feature, tokenizer, max_length)
    return {"success": True, "caption": caption, "confidence": ...}
```

**Key fixes from notebook:**
- `padding='post'` in `predict_caption()` matches training generator
- Raw string `r"\s+"` fixes SyntaxWarning in cleaning step
- VGG16 feature extractor loaded once at startup (lifespan)

### 3.2 Express Routes (`server/routes/`)

- `POST /api/upload` — proxy to ML service, store in MongoDB
- `GET /api/captions` — paginated caption records
- `GET /api/history` — recent history with optional user filter
- `DELETE /api/history/:id` — remove record

### 3.3 Mongoose Model (`server/models/ImageRecord.js`)

```javascript
const ImageRecordSchema = new mongoose.Schema({
  imageUrl: String,
  predictedCaption: String,
  actualCaptions: [String],
  userId: String,
  confidence: Number,
  modelVersion: { type: String, default: 'vgg16_lstm' }
}, { timestamps: true });
```

### 3.4 React Components (`src/components/`)

- **Upload.tsx** — Drag-and-drop zone with validation, preview, loading state
- **CaptionDisplay.tsx** — Shows generated caption with confidence badge
- **History.tsx** — Grid of recent predictions with delete action

---

## 5. Quick Start (Development)

### Prerequisites
- **Node.js** 18+ 
- **Python** 3.10+
- **MongoDB** running locally or connection string in `.env`
- **Pre-trained artifacts** in `./artifacts/`:
  - `best_model.keras`
  - `tokenizer.pkl`

### Setup & Running (Local Development)

**Option 1: All-in-one (Recommended for first-time setup)**

```bash
# 1. Install frontend dependencies
npm install

# 2. Install server dependencies
cd server && npm install && cd ..

# 3. Install ML service dependencies
cd ml_service && pip install -r requirements.txt && cd ..

# 4. Create .env file in project root
cat > .env << EOF
MONGO_URI=mongodb://localhost:27017/imagecaption
ML_SERVICE_URL=http://localhost:8000
PORT=5001
NODE_ENV=development
EOF

# 5. Start MongoDB (in separate terminal)
# If using macOS with Homebrew:
brew services start mongodb-community
# or Docker:
docker run -d -p 27017:27017 mongo:latest

# 6. Start ML Service (Terminal 1)
cd ml_service
python main.py
# Server running on http://localhost:8000

# 7. Start Express Backend (Terminal 2)
cd server
npm run dev
# Server running on http://localhost:5001

# 8. Start React Frontend (Terminal 3)
npm run dev
# Frontend running on http://localhost:5173
```

**Option 2: Using Docker Compose (Recommended for production-like setup)**

```bash
docker-compose up --build
# All services will start:
# - Frontend:  http://localhost (via Nginx)
# - Backend:   http://localhost:5001
# - ML Service: http://localhost:8000
# - MongoDB:   localhost:27017
```

### Verify Services Are Running

```bash
# Frontend health
curl http://localhost:5173

# Backend health
curl http://localhost:5001/health

# ML Service health
curl http://localhost:8000/docs

# MongoDB connection
mongosh mongodb://localhost:27017/imagecaption
```

---

## 6. Testing the Application

### Via Web UI
1. Open http://localhost:5173 (or http://localhost if using Docker)
2. Drag and drop an image (JPG/PNG/WebP, <5MB) into the upload zone
3. Watch as the caption appears with confidence score
4. View history of predictions in the History tab

### Via API (cURL)

```bash
# Upload image and get caption
curl -X POST http://localhost:5001/api/upload \
  -F "file=@test_image.jpg" \
  -F "userId=user123"

# Get all captions
curl http://localhost:5001/api/captions

# Get user history
curl http://localhost:5001/api/history?userId=user123

# Delete a record
curl -X DELETE http://localhost:5001/api/history/{recordId}
```

### Via Python CLI (ML Service Direct)

```bash
# Single image prediction
cd python_pipeline
python 12_demo_inference.py --image path/to/image.jpg

# This will display the predicted caption and confidence score
```

---

## 7. Step-by-Step Demo & Presentation Guide

### For Instructors / Students

**Goal:** Demonstrate each neural network pipeline stage sequentially.

1. **Setup (01)** — Show directory structure, explain BASE_DIR vs WORKING_DIR
2. **Data Acquisition (02)** — Show Flickr8k download, discuss dataset size
3. **Feature Extraction (03)** — Run VGG16, explain 4096-dim vector, show `features.pkl`
4. **Caption Loading (04)** — Parse CSV, build mapping, show 5 captions per image
5. **Text Cleaning (05)** — Before/after comparison, highlight `re.sub(r"\s+", " ", ...)` fix
6. **Tokenizer (06)** — Show vocab_size (~8485), max_length (35), word index sample
7. **Model Architecture (07)** — Display encoder-decoder diagram, explain LSTM + merge
8. **Data Generator (08)** — Demo one batch: X1 shape (2,4096), X2 shape (2,35), y shape (2,8485)
9. **Training (09)** — Run 1-2 epochs live, show loss dropping, explain `clear_session()` every 5 epochs
10. **Evaluation (10)** — Show BLEU-1 (~0.55) and BLEU-2 (~0.33) scores
11. **Inference Demo (12)** — Pick any test image, compare actual vs predicted

### Web App Demo

1. Open React frontend at `http://localhost:5173` 
2. Drag-and-drop any image (JPG/PNG/WebP, <5MB)
3. Watch loading state → caption appears with confidence badge
4. Show History tab updating in real-time with new predictions
5. Show MongoDB records via `GET /api/history` API endpoint

---

## 8. Step-by-Step Demo & Presentation Guide

---

## 9. Project Directory Tree

```
.
├── docker-compose.yml
├── .env                      # Environment variables
├── package.json              # Frontend dependencies
├── vite.config.ts            # Vite configuration
├── tsconfig.json             # TypeScript config
├── README.md
│
├── python_pipeline/          # Academic demo scripts (01-12)
│   ├── 01_setup_paths.py
│   ├── 02_download_data.py
│   ├── 03_extract_features.py
│   ├── 04_load_captions.py
│   ├── 05_clean_captions.py
│   ├── 06_build_tokenizer.py
│   ├── 07_model_architecture.py
│   ├── 08_data_generator.py
│   ├── 09_train_model.py
│   ├── 10_evaluate_bleu.py
│   ├── 11_predict_caption.py
│   ├── 12_demo_inference.py
│   ├── artifacts/            # Trained model & tokenizer
│   │   ├── best_model.keras
│   │   ├── best_model.h5
│   │   └── tokenizer.pkl
│   └── data/
│       └── flickr8k/
│           ├── captions.txt
│           └── Images/
│
├── ml_service/               # FastAPI inference microservice
│   ├── main.py               # FastAPI app with /predict endpoint
│   ├── requirements.txt
│   └── artifacts/            # Symlink or copy of trained models
│       ├── best_model.keras
│       └── tokenizer.pkl
│
├── server/                   # Express.js backend
│   ├── server.js             # Entry point
│   ├── package.json
│   ├── .env
│   ├── models/
│   │   └── ImageRecord.js    # Mongoose schema
│   ├── routes/
│   │   ├── upload.js         # POST /api/upload
│   │   ├── captions.js       # GET /api/captions
│   │   └── history.js        # GET /api/history, DELETE /api/history/:id
│   ├── middleware/
│   │   ├── upload.js         # Multer configuration
│   │   ├── errorHandler.js   # Global error handler
│   │   └── rateLimiter.js    # Rate limiting
│   └── utils/
│       └── logger.js         # Winston logger
│
└── src/                      # React frontend (Vite + Tailwind + shadcn)
    ├── App.tsx
    ├── main.tsx
    ├── App.css
    ├── index.css
    ├── pages/
    │   └── Home.tsx          # Main page
    ├── components/
    │   ├── Upload.tsx        # Drag-and-drop upload
    │   ├── CaptionDisplay.tsx # Display generated caption
    │   ├── History.tsx       # View prediction history
    │   └── ui/               # shadcn UI components
    │       ├── button.tsx
    │       ├── card.tsx
    │       ├── input.tsx
    │       └── ... (20+ more UI components)
    ├── services/
    │   └── api.ts            # Axios client for Express backend
    ├── types/
    │   └── index.ts          # TypeScript interfaces
    ├── hooks/
    │   └── use-mobile.ts
    └── lib/
        └── utils.ts          # Utility functions
```

---

## 10. Constraints & Best Practices Observed

- ✅ Model files NEVER exposed in frontend bundle
- ✅ All secrets in `.env` (DB URIs, service URLs)
- ✅ Structured logging (Winston/Morgan for Node, Python logging)
- ✅ Error boundaries via Express errorHandler middleware
- ✅ Rate limiting on upload and API routes via express-rate-limit
- ✅ CORS enabled for development (ports 3000, 5173, localhost)
- ✅ Input validation: file type, file size (max 5MB), image dimensions
- ✅ Training fully separate from inference pipeline
- ✅ FastAPI startup loads artifacts once (lifespan context) for efficient reuse
- ✅ `padding='post'` fixed and consistent across training + inference
- ✅ TypeScript strict mode enabled for type safety
- ✅ MongoDB indexes on userId and timestamps for query performance

---

## 11. Troubleshooting

### "Cannot connect to ML Service" on backend startup
- Ensure FastAPI is running: `cd ml_service && python main.py`
- Check `ML_SERVICE_URL` in `.env` matches actual service address
- Verify no firewall blocking port 8000

### "Cannot find module" errors in Python
- Run `pip install -r ml_service/requirements.txt`
- Verify Python 3.10+ installed: `python --version`

### "MongoDB connection refused"
- Ensure MongoDB is running locally or update `MONGO_URI` in `.env`
- Test: `mongosh mongodb://localhost:27017`

### Frontend shows "Network Error" on upload
- Check backend is running: `curl http://localhost:5001/health`
- Check CORS settings in `server/server.js`
- Verify `ML_SERVICE_URL` environment variable is set

### Model inference is slow
- First request loads model (~2-3 seconds): this is normal
- Subsequent requests should be <500ms
- Check ML service logs: `cd ml_service && python main.py`

---

## 12. Tech Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 19, Vite, TypeScript, Tailwind CSS, shadcn/ui | Fast, type-safe UI with drag-and-drop |
| **Backend** | Express.js, Mongoose, MongoDB | RESTful API with persistent storage |
| **ML Service** | FastAPI, TensorFlow, Keras, VGG16+LSTM | Image captioning inference |
| **Data Pipeline** | Python, Flickr8k, Tokenizer, BLEU metrics | Training & evaluation (separate) |
| **DevOps** | Docker, Docker Compose, Nginx | Containerized multi-service deployment |
| **Tooling** | TypeScript, ESLint, Vite, Nodemon | Developer experience & code quality |

---

## 13. License

MIT — Academic and commercial use permitted. Attribution to original Flickr8k dataset and Kaggle authors appreciated.
