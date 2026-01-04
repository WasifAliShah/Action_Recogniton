"""
Evaluate the trained CNN+LSTM action recognition model on HMDB51.
Calculates accuracy, per-class metrics, and confusion matrix.
"""

import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models
from torchvision import transforms
from tqdm import tqdm


# ========== Model Definition ==========
class CNNLSTM(nn.Module):
    """CNN encoder + LSTM temporal head."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        encoder: str = "resnet18",
    ) -> None:
        super().__init__()
        self.encoder_name = encoder
        if encoder == "resnet18":
            backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
            modules = list(backbone.children())[:-1]
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
        batch, time, channels, height, width = x.shape
        x = x.view(batch * time, channels, height, width)
        feats = self.encoder(x).view(batch * time, -1)
        feats = self.proj(feats)
        feats = feats.view(batch, time, -1)
        outputs, _ = self.lstm(feats)
        logits = self.head(outputs[:, -1, :])
        return logits


# ========== Dataset ==========
class FrameSequenceDataset(Dataset):
    """Loads sequences of frames from folder structure: root/class_name/clip_name/frame.jpg"""

    def __init__(
        self,
        root: str | Path,
        seq_len: int = 16,
        image_size: int = 128,
        class_filter: list[str] | None = None,
        transform = None,
    ) -> None:
        self.root = Path(root)
        self.seq_len = seq_len
        self.image_size = image_size
        self.samples: list[tuple[Path, int]] = []
        all_classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        if class_filter:
            class_filter_set = set(class_filter)
            self.class_names = [c for c in all_classes if c in class_filter_set]
        else:
            self.class_names = all_classes
        if not self.class_names:
            raise ValueError("No classes selected; check class_filter")
        class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        for class_dir in self.root.iterdir():
            if not class_dir.is_dir() or class_dir.name not in class_to_idx:
                continue
            for clip_dir in class_dir.iterdir():
                if not clip_dir.is_dir():
                    continue
                frames = sorted(clip_dir.glob("*.jpg")) + sorted(clip_dir.glob("*.png"))
                if len(frames) >= seq_len:
                    self.samples.append((clip_dir, class_to_idx[class_dir.name]))
        if not self.samples:
            raise ValueError("No clips found. Expect root/class/clip/frame.jpg with at least seq_len frames.")

        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        clip_dir, label = self.samples[idx]
        frames = sorted(clip_dir.glob("*.jpg")) + sorted(clip_dir.glob("*.png"))
        if len(frames) < self.seq_len:
            raise IndexError("Not enough frames in clip")
        start = random.randint(0, len(frames) - self.seq_len)
        selected = frames[start : start + self.seq_len]
        imgs = [self.transform(Image.open(f).convert("RGB")) for f in selected]
        seq = torch.stack(imgs, dim=0)
        return seq, label


# ========== Evaluation ==========
def evaluate_model(
    data_root: str,
    weights_path: str,
    classes_path: str,
    seq_len: int = 16,
    image_size: int = 128,
    batch_size: int = 16,
    encoder: str = "resnet18",
) -> None:
    """
    Evaluate the trained model on test data.
    
    Args:
        data_root: Path to dataset (class/clip/frame.jpg structure)
        weights_path: Path to model.pt
        classes_path: Path to classes.txt
        seq_len: Number of frames per sequence
        image_size: Input image size
        batch_size: Batch size for evaluation
        encoder: "resnet18" or "tiny"
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load class names
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Classes: {class_names}")

    # Create dataset
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    dataset = FrameSequenceDataset(
        data_root,
        seq_len=seq_len,
        image_size=image_size,
        class_filter=class_names,
        transform=val_transform,
    )
    print(f"Dataset: {len(dataset)} clips")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Load model
    model = CNNLSTM(
        num_classes=len(class_names),
        feature_dim=256,
        hidden_dim=128,
        encoder=encoder,
    ).to(device)

    print(f"Loading weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded successfully.")

    # Evaluate
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(dataloader, desc="Evaluating", leave=True)
    with torch.no_grad():
        for seq, labels in pbar:
            seq = seq.to(device)
            labels = labels.to(device)
            logits = model(seq)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    overall_acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataset)

    print("\n" + "=" * 60)
    print(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print(f"Average Loss: {avg_loss:.4f}")
    print("=" * 60)

    # Per-class metrics
    print("\nPer-Class Metrics:")
    print("-" * 60)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    # Confusion matrix
    print("\nConfusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Print confusion matrix in a readable format
    print("\nConfusion Matrix (readable):")
    print("-" * 60)
    print("Predicted →")
    print("Actual ↓    " + "  ".join([f"{c:>8}" for c in class_names]))
    for i, class_name in enumerate(class_names):
        print(f"{class_name:>10} " + "  ".join([f"{cm[i, j]:>8}" for j in range(len(class_names))]))


if __name__ == "__main__":
    DATA_ROOT = "../HMDB51"
    WEIGHTS_PATH = "./weights/model.pt"
    CLASSES_PATH = "./weights/classes.txt"

    evaluate_model(
        data_root=DATA_ROOT,
        weights_path=WEIGHTS_PATH,
        classes_path=CLASSES_PATH,
        seq_len=16,
        image_size=128,
        batch_size=16,
        encoder="resnet18",
    )
