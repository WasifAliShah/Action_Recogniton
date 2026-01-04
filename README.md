# Action Recognition (Stanford-40, Image-Based)

Image-based action recognition using a ResNet50 + LSTM model trained on the **Stanford-40** dataset (40 action classes). The app exposes a FastAPI backend and a static frontend for single-image predictions.

## Overview

- **Model:** ResNet50 (frozen) encoder + LSTM (2048->512) + Linear head
- **Dataset:** Stanford-40 Actions (40 classes)
- **Input:** Single RGB image (224x224 resize + ImageNet normalization)
- **Output:** Predicted action label with confidence scores and an annotated preview
- **APIs:** `/health`, `/labels`, `/predict_image` (single image); `/predict` also accepts sequences but routes to single-image flow when only one frame is provided

## Project Structure

```
+-- backend/
�   +-- app/
�   �   +-- main.py          # FastAPI endpoints
�   �   +-- model.py         # ResNet50+LSTM model wrapper
�   +-- weights/
�   �   +-- best_cnn_lstm.pth         # Trained Stanford-40 weights (state_dict)
�   �   +-- best_cnn_lstm_scripted.pt # TorchScript export (kept for reference)
�   �   +-- classes.txt               # Class names (one per line)
�   �   +-- class_index.json          # Same classes as JSON array
�   +-- requirements.txt
+-- frontend/
�   +-- index.html
�   +-- script.js
�   +-- style.css
+-- stanford_40_training.py   # Reference training script (ResNet50 + LSTM)
+-- README.md
```

## Setup

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# run API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
python -m http.server 5173
```

Open http://localhost:5173 and set API base to http://localhost:8000.

## Usage

1. Start backend and frontend.
2. In the web UI, pick the API base (default http://localhost:8000).
3. Upload an image (common formats work). The app resizes to 224x224 and normalizes with ImageNet stats.
4. View the top prediction, confidence, and the annotated preview.

## API Endpoints

- `GET /health` � health check
- `GET /labels` � list of the 40 Stanford-40 action classes
- `POST /predict_image` � multipart form with one image file (`file` field); returns label, score, per-class scores, and a base64 preview
- `POST /predict` � accepts sequences; when a single frame is provided, it uses the same single-image path

## Model Details

- **Encoder:** ResNet50 pretrained on ImageNet, all layers frozen, final FC replaced with Identity
- **LSTM:** input 2048, hidden 512, batch_first=True
- **Head:** Linear(512 -> 40)
- **Preprocessing:** Resize to 224x224, convert to RGB, divide by 255, ImageNet mean/std normalization
- **Weights:** Loaded from `backend/weights/best_cnn_lstm.pth` by default (override with `MODEL_WEIGHTS_PATH` if needed)

## Files of Interest

- Backend model + loader: `backend/app/model.py`
- API definitions: `backend/app/main.py`
- Weights and class names: `backend/weights`
- Frontend UI: `frontend/index.html`, `frontend/script.js`
- Training reference: `stanford_40_training.py`

