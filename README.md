# Action Recognition (Stanford-40, Image-Based)

**Developed by:**

* **Syed Wasif Ali Shah**
* **Sabbas Ahmad**

Image-based action recognition using a ResNet50 + LSTM model trained on the **Stanford-40** dataset (40 action classes). The app exposes a FastAPI backend and a static frontend for single-image predictions.

## Overview

* **Model:** ResNet50 (frozen) encoder + LSTM (2048→512) + Linear head
* **Dataset:** Stanford-40 Actions (40 classes)
* **Input:** Single RGB image (224×224 resize + ImageNet normalization)
* **Output:** Predicted action label with confidence scores and an annotated preview
* **APIs:** `/health`, `/labels`, `/predict_image` (single image); `/predict` also accepts sequences but routes to single-image flow when only one frame is provided

---

## Model Files (Required)

Download the pretrained model files from Google Drive and place them **exactly** in the `backend/weights/` folder.

**Google Drive link:**
👉 [https://drive.google.com/drive/folders/11Op-fIIRXtikAxVtSakA9YMC0fiTFmeY?usp=sharing](https://drive.google.com/drive/folders/11Op-fIIRXtikAxVtSakA9YMC0fiTFmeY?usp=sharing)

### Files to download

* `best_cnn_lstm.pth` – Trained Stanford-40 weights (`state_dict`)
* `best_cnn_lstm_scripted.pt` – TorchScript export (optional, kept for reference)
* `classes.txt` – Class names (one per line)
* `class_index.json` – Same class list in JSON format

### Expected directory structure

```
backend/
└── weights/
    ├── best_cnn_lstm.pth
    ├── best_cnn_lstm_scripted.pt
    ├── classes.txt
    └── class_index.json
```

> ⚠️ The backend will fail to start if `best_cnn_lstm.pth` or the class files are missing or placed elsewhere.

---

## Project Structure

```
+-- backend/
│   +-- app/
│   │   +-- main.py          # FastAPI endpoints
│   │   +-- model.py         # ResNet50+LSTM model wrapper
│   +-- weights/
│   │   +-- best_cnn_lstm.pth         # Trained Stanford-40 weights
│   │   +-- best_cnn_lstm_scripted.pt # TorchScript export (reference)
│   │   +-- classes.txt               # Class names
│   │   +-- class_index.json          # Class names (JSON)
│   +-- requirements.txt
+-- frontend/
│   +-- index.html
│   +-- script.js
│   +-- style.css
+-- stanford_40_training.py   # Reference training script (ResNet50 + LSTM)
+-- README.md
```

---

## Setup

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux / macOS

pip install -r requirements.txt

# Run API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
python -m http.server 5173
```

Open **[http://localhost:5173](http://localhost:5173)** and set the API base to **[http://localhost:8000](http://localhost:8000)**.

---

## Usage

1. Start the backend and frontend.
2. In the web UI, confirm the API base URL.
3. Upload an image (JPG/PNG/etc.). The app resizes it to 224×224 and applies ImageNet normalization.
4. View the top predicted action, confidence score, and annotated preview.

---

## API Endpoints

* `GET /health` – Health check
* `GET /labels` – List of the 40 Stanford-40 action classes
* `POST /predict_image` – Multipart form upload (`file` field)

  * Returns: predicted label, confidence, per-class scores, base64 preview image
* `POST /predict` – Accepts sequences; single-frame input is routed to the same image pipeline

---

## Model Details

* **Encoder:** ResNet50 pretrained on ImageNet, all layers frozen, final FC replaced with `Identity`
* **LSTM:** Input size 2048, hidden size 512, `batch_first=True`
* **Classifier Head:** Linear(512 → 40)
* **Preprocessing:**

  * Resize to 224×224
  * Convert to RGB
  * Normalize using ImageNet mean/std
* **Weights loading:**

  * Default: `backend/weights/best_cnn_lstm.pth`
  * Override using `MODEL_WEIGHTS_PATH` environment variable if needed

---

## Notes

* Although the model includes an LSTM, **single-image inference** works by passing a sequence length of 1.
* The TorchScript model is not required for inference but is included for experimentation or deployment reference.
* The training script is provided for reproducibility and experimentation.
