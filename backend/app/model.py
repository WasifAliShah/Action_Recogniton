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
        if encoder == "stanford40":
            # Stanford-40 architecture: ResNet50 (frozen) + LSTM
            if tv_models is None:
                raise RuntimeError("torchvision not available for stanford40 encoder")
            self.encoder = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
            # Freeze all parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
            # Replace final FC with identity
            self.encoder.fc = nn.Identity()
            self.encoder_out = 2048
            # No projection layer for Stanford-40
            self.proj = None
            self.lstm = nn.LSTM(2048, 512, batch_first=True)
            self.head = nn.Linear(512, num_classes)
        elif encoder == "resnet50":
            if tv_models is None:
                raise RuntimeError("torchvision not available for resnet50 encoder")
            backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
            modules = list(backbone.children())[:-1]  # remove fc
            self.encoder = nn.Sequential(*modules)
            self.encoder_out = 2048
            self.proj = nn.Linear(self.encoder_out, feature_dim)
            self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, num_classes)
        elif encoder == "resnet18":
            if tv_models is None:
                raise RuntimeError("torchvision not available for resnet18 encoder")
            backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
            modules = list(backbone.children())[:-1]  # remove fc
            self.encoder = nn.Sequential(*modules)
            self.encoder_out = 512
            self.proj = nn.Linear(self.encoder_out, feature_dim)
            self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, num_classes)
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
        
        if self.encoder_name == "stanford40":
            # ResNet50 feature extraction
            feats = self.encoder(x)  # (batch*time, 2048)
            feats = feats.view(batch, time, -1)
        else:
            feats = self.encoder(x).view(batch * time, -1)
            if self.proj is not None:
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
        
        # Resolve weights path (prefer explicit env, then best_cnn_lstm.pth, then others)
        env_path = os.getenv("MODEL_WEIGHTS_PATH")
        weights_dir = BASE_DIR.parent / "weights"
        candidates = [
            weights_dir / "best_cnn_lstm.pth",
            weights_dir / "model.pth",
            weights_dir / "model.pt",
            weights_dir / "best_cnn_lstm_scripted.pt",
        ]
        path = Path(env_path) if env_path else next((p for p in candidates if p.exists()), candidates[0])
        self.is_torchscript = False
        self.weights_loaded = False
        self.weights_path = None
        
        try:
            if Path(path).exists():
                # Try loading state_dict from .pth file first
                try:
                    raw_state = torch.load(str(path), map_location=self.device)
                    state_dict = raw_state.get("state_dict", raw_state)

                    # Remap keys from training script (cnn.*, fc.*) to our module names (encoder.*, head.*)
                    remapped: Dict[str, torch.Tensor] = {}
                    for k, v in state_dict.items():
                        new_key = k
                        if k.startswith("cnn."):
                            new_key = k.replace("cnn.", "encoder.", 1)
                        elif k.startswith("fc."):
                            new_key = k.replace("fc.", "head.", 1)
                        # Drop running stats/buffers for num_batches_tracked to avoid strict mismatches
                        if new_key.endswith("num_batches_tracked"):
                            continue
                        # Skip encoder.fc.* since we replaced it with Identity
                        if new_key.startswith("encoder.fc."):
                            continue
                        remapped[new_key] = v

                    encoder = os.getenv("MODEL_ENCODER", "stanford40")
                    self.model = CNNLSTM(
                        num_classes=len(self.class_names),
                        feature_dim=2048,
                        hidden_dim=512,
                        encoder=encoder,
                    ).to(self.device)

                    missing, unexpected = self.model.load_state_dict(remapped, strict=False)
                    if missing:
                        LOGGER.warning("Missing keys when loading: %s", missing)
                    if unexpected:
                        LOGGER.warning("Unexpected keys when loading: %s", unexpected)

                    self.weights_loaded = True
                    self.weights_path = str(path)
                    self.model.eval()
                    LOGGER.info("Loaded state_dict weights from %s (with key remap)", path)
                except Exception as e:
                    LOGGER.warning("State dict load failed: %s, trying TorchScript", e)
                    # Fallback to TorchScript
                    self.model = torch.jit.load(str(path), map_location=self.device)
                    self.is_torchscript = True
                    self.weights_loaded = True
                    self.weights_path = str(path)
                    self.model.eval()
                    LOGGER.info("Loaded TorchScript model from %s", path)
            else:
                # No weights, create default model
                LOGGER.warning("No weights found at %s; using random initialization.", path)
                encoder = os.getenv("MODEL_ENCODER", "stanford40")
                
                self.model = CNNLSTM(
                    num_classes=len(self.class_names),
                    feature_dim=2048,
                    hidden_dim=512,
                    encoder=encoder,
                ).to(self.device)
        except Exception as exc:
            LOGGER.error("Failed to load model: %s", exc)
            # Fallback to default model
            encoder = os.getenv("MODEL_ENCODER", "stanford40")
            
            self.model = CNNLSTM(
                num_classes=len(self.class_names),
                feature_dim=2048,
                hidden_dim=512,
                encoder=encoder,
            ).to(self.device)
        
        self.seq_len = int(os.getenv("MODEL_SEQ_LEN", "1"))
        self.image_size = int(os.getenv("MODEL_IMAGE_SIZE", "224"))
        self.model.eval()

    def preprocess(self, image: Image.Image, seq_len: int | None = None, image_size: int | None = None) -> torch.Tensor:
        seq_len = seq_len or self.seq_len
        image_size = image_size or self.image_size
        image = image.convert("RGB").resize((image_size, image_size))
        arr = np.array(image).astype(np.float32) / 255.0
        
        # Apply ImageNet normalization (TorchScript or ResNet models)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
        arr = (arr - mean) / std
        
        frame = torch.from_numpy(arr.astype(np.float32))
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
            
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
            arr = (arr - mean) / std
            
            frames.append(torch.from_numpy(arr.astype(np.float32)))
        
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
        
        # Try loading as TorchScript first (for Stanford-40 scripted model)
        try:
            scripted_model = torch.jit.load(str(path), map_location=device)
            # For TorchScript models, replace the wrapper model with the loaded one
            if hasattr(model, '__dict__'):
                model.__class__ = scripted_model.__class__
                model.__dict__.update(scripted_model.__dict__)
            LOGGER.info("Loaded TorchScript model from %s", path)
            return True, str(path)
        except Exception:
            # Fall back to state_dict loading
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            LOGGER.info("Loaded state_dict weights from %s", path)
            return True, str(path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to load weights from %s: %s", path, exc)
        return False, None
