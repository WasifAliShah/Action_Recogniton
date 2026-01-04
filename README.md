# Action Recognition Demo

Lightweight demo that exposes a REST API (FastAPI) running a CNN+LSTM action recognition model and a static web UI for uploading an image and receiving an annotated preview.

## Project layout

- backend/: FastAPI app with a lightweight CNN+LSTM model stub and REST endpoints.
- frontend/: Static HTML/CSS/JS client that calls the API and renders predictions and the annotated image.
- .github/copilot-instructions.md: Workflow checklist for this project.

## Prerequisites

- Python 3.11+
- Node.js not required unless you want to serve the frontend with a bundler; the provided static files work via a simple file server.

## Backend (API)

1. Install dependencies:
   ```bash
   cd backend
   python -m venv .venv
   .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
2. Add real weights and labels (choose one):
   - Place your PyTorch checkpoint at `backend/weights/model.pt` (default path), and optional class names in `backend/weights/classes.txt` (one class per line, order matches checkpoint).
   - Or set environment variables:
     - `MODEL_WEIGHTS_PATH=path/to/your/model.pt`
     - `MODEL_CLASSES=walking,running,jumping` (comma-separated) or `MODEL_CLASSES_PATH=path/to/classes.txt`
3. Run the API:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. Test:
   - Health: `GET http://localhost:8000/health`
   - Predict: `POST http://localhost:8000/predict` with form-data field `file` containing an image.

Notes:
- If weights are found, the API response includes a `note` stating the source; otherwise it warns that random initialization is being used.
- The API returns an `annotated_image_base64` string you can render in an `<img>` tag.

## Training (HMDB51 or custom dataset)

Expected layout: `data_root/class_name/clip_name/frame.jpg` (at least `seq_len` frames per clip). Example:

```
data_root/
  walking/
    clip1/frame_0001.jpg
    clip1/frame_0002.jpg
  running/
    clip5/frame_0001.jpg
```

Train (full set, resnet18 backbone):

```bash
cd backend
.\.venv\Scripts\activate
python train.py ..\HMDB51 --epochs 10 --batch-size 8 --seq-len 16 --image-size 128 --lr 5e-4 --encoder resnet18 --feature-dim 256 --hidden-dim 128 --weight-decay 1e-4
```

Faster subset (example walking/running/jump/sit):

```bash
python train.py ..\HMDB51 --epochs 8 --batch-size 8 --seq-len 16 --classes walking running jump sit --encoder resnet18
```

Outputs (default):
- `backend/weights/model.pt` (best val)
- `backend/weights/classes.txt`

Restart the API after training so it loads the new checkpoint.

### No dataset? Generate a tiny synthetic one

From backend:

```bash
cd backend
.\.venv\Scripts\activate
python generate_synthetic.py ..\data_synth --classes walking running jumping --clips-per-class 12 --frames-per-clip 16 --image-size 128
python train.py ..\data_synth --epochs 5 --batch-size 4 --seq-len 8 --image-size 128
```

This will create moving-blob clips for the given classes, train the CNN+LSTM, and write weights/classes into backend/weights/. Restart uvicorn afterward.

## Frontend (static)

You can open `frontend/index.html` directly or serve it locally to avoid CORS quirks:
```bash
cd frontend
python -m http.server 5173
```
Then browse to http://localhost:5173 (the page calls the API at http://localhost:8000 by default).

## Training hook (optional)

The CNN+LSTM architecture lives in `backend/app/model.py`. You can train it elsewhere and load weights into the `ActionRecognitionModel`. Keep the class ordering consistent with `class_names`.

## Troubleshooting

- If predictions fail, ensure the backend is running and the API base matches the frontend (`http://localhost:8000` by default).
- Torch may download CPU wheels; if you want GPU support, install a CUDA build from PyTorch’s site.
