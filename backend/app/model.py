import io
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:  # Optional torchvision for stronger backbones
    import torchvision.models as tv_models
except Exception:  # noqa: BLE001
    tv_models = None


LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CLASS_NAMES = ["walking", "running", "jumping", "sitting"]


class CNNLSTM(nn.Module):
    """CNN encoder + LSTM temporal head. Supports tiny conv stack or resnet18 backbone."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 128,
        hidden_dim: int = 64,
        encoder: str = "tiny",
    ) -> None:
        super().__init__()
        self.encoder_name = encoder
        if encoder == "resnet18":
            if tv_models is None:
                raise RuntimeError("torchvision not available for resnet18 encoder")
            backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
            modules = list(backbone.children())[:-1]  # remove fc
            self.encoder = nn.Sequential(*modules)
            self.encoder_out = 512
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.encoder_out = 64

        self.proj = nn.Linear(self.encoder_out, feature_dim)
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, channels, height, width)
        batch, time, channels, height, width = x.shape
        x = x.view(batch * time, channels, height, width)
        feats = self.encoder(x).view(batch * time, -1)
        feats = self.proj(feats)
        feats = feats.view(batch, time, -1)
        outputs, _ = self.lstm(feats)
        logits = self.head(outputs[:, -1, :])
        return logits


class ActionRecognitionModel:
    """Wrapper to run CNN+LSTM inference and keep IO helpers together."""

    def __init__(self, class_names: List[str] | None = None, device: str | None = None) -> None:
        self.class_names = class_names or _load_class_names()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(7)
        encoder = os.getenv("MODEL_ENCODER", "resnet18")
        feature_dim = int(os.getenv("MODEL_FEATURE_DIM", "256"))
        hidden_dim = int(os.getenv("MODEL_HIDDEN_DIM", "128"))
        self.seq_len = int(os.getenv("MODEL_SEQ_LEN", "16"))
        self.image_size = int(os.getenv("MODEL_IMAGE_SIZE", "128"))

        self.model = CNNLSTM(
            num_classes=len(self.class_names),
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            encoder=encoder,
        ).to(self.device)
        self.weights_loaded, self.weights_path = _load_weights(self.model, self.device)
        self.model.eval()

    def preprocess(self, image: Image.Image, seq_len: int | None = None, image_size: int | None = None) -> torch.Tensor:
        seq_len = seq_len or self.seq_len
        image_size = image_size or self.image_size
        image = image.convert("RGB").resize((image_size, image_size))
        arr = np.array(image).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
        frame = torch.from_numpy(arr)
        frames = frame.unsqueeze(0).repeat(seq_len, 1, 1, 1)  # (T, C, H, W)
        batch_frames = frames.unsqueeze(0)  # (1, T, C, H, W)
        return batch_frames.to(self.device)

    def predict(self, image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        tensor = self.preprocess(image)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        top_idx = int(np.argmax(probs))
        top_label = self.class_names[top_idx]
        top_score = float(probs[top_idx])
        scores = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
        return top_label, top_score, scores

    def predict_sequence(self, images: List[Image.Image]) -> Tuple[str, float, Dict[str, float]]:
        """Predict action from a sequence of frame images."""
        # Resize and normalize each frame
        frames = []
        for img in images:
            img = img.convert("RGB").resize((self.image_size, self.image_size))
            arr = np.array(img).astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
            frames.append(torch.from_numpy(arr))
        
        # Pad or trim to seq_len
        seq_len = self.seq_len
        if len(frames) < seq_len:
            # Repeat last frame to reach seq_len
            while len(frames) < seq_len:
                frames.append(frames[-1])
        elif len(frames) > seq_len:
            # Take evenly spaced frames
            indices = np.linspace(0, len(frames) - 1, seq_len, dtype=int)
            frames = [frames[i] for i in indices]
        
        # Stack into batch: (1, T, C, H, W)
        batch_frames = torch.stack(frames).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(batch_frames)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        top_idx = int(np.argmax(probs))
        top_label = self.class_names[top_idx]
        top_score = float(probs[top_idx])
        scores = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
        return top_label, top_score, scores

    def export_state_dict(self) -> bytes:
        """Serialize weights to bytes (handy for saving from a notebook)."""
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        return buffer.getvalue()

    def load_state_dict_bytes(self, data: bytes) -> None:
        buffer = io.BytesIO(data)
        state_dict = torch.load(buffer, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.weights_loaded = True
        self.weights_path = "bytes"


# Single, lazy-initialized model instance for reuse in the app.
_model: ActionRecognitionModel | None = None


def get_model() -> ActionRecognitionModel:
    global _model
    if _model is None:
        _model = ActionRecognitionModel()
    return _model


def _load_class_names() -> List[str]:
    from_env = os.getenv("MODEL_CLASSES")
    if from_env:
        names = [n.strip() for n in from_env.split(",") if n.strip()]
        if names:
            LOGGER.info("Loaded class names from MODEL_CLASSES env var: %s", names)
            return names

    path = os.getenv("MODEL_CLASSES_PATH") or BASE_DIR.parent / "weights" / "classes.txt"
    try:
        if Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                names = [line.strip() for line in f.readlines() if line.strip()]
            if names:
                LOGGER.info("Loaded class names from %s", path)
                return names
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to read class names from %s: %s", path, exc)

    LOGGER.warning("Falling back to default class names: %s", DEFAULT_CLASS_NAMES)
    return DEFAULT_CLASS_NAMES


def _load_weights(model: nn.Module, device: torch.device) -> tuple[bool, str | None]:
    path = os.getenv("MODEL_WEIGHTS_PATH") or BASE_DIR.parent / "weights" / "model.pt"
    try:
        if not Path(path).exists():
            LOGGER.warning("No weights found at %s; using random initialization.", path)
            return False, None
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        LOGGER.info("Loaded weights from %s", path)
        return True, str(path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to load weights from %s: %s", path, exc)
        return False, None
