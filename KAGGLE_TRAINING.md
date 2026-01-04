# Kaggle Training Setup

Upload these files to your Kaggle notebook environment:

1. **This training script**: `kaggle_train.py` (or run cells in a notebook)
2. **Dataset**: Upload HMDB51 folder or use Kaggle's HMDB51 dataset if available
3. **Dependencies**: Already available on Kaggle (torch, torchvision, PIL, tqdm)

## Quick Start

In a Kaggle notebook:

```python
# Cell 1: Copy model definition and training code from kaggle_train.py

# Cell 2: Set paths and run training
DATA_ROOT = "/kaggle/input/hmdb51/HMDB51"  # adjust to your dataset path
OUTPUT_DIR = "/kaggle/working/weights"

train_model(
    data_root=DATA_ROOT,
    save_dir=OUTPUT_DIR,
    epochs=10,
    batch_size=8,
    seq_len=16,
    image_size=128,
    lr=5e-4,
    encoder="resnet18",
    feature_dim=256,
    hidden_dim=128,
    classes=["walk", "run", "jump", "sit"],  # optional subset, or None for all
)
```

## After Training

Download from Kaggle output:
- `weights/model.pt`
- `weights/classes.txt`

Place them in `backend/weights/` and restart your API:
```bash
cd backend
.\.venv\Scripts\Activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
