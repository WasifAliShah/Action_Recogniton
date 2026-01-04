"""
Quick evaluation script without sklearn.
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models
from torchvision import transforms
from tqdm import tqdm


class CNNLSTM(nn.Module):
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


class FrameSequenceDataset(Dataset):
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
            raise ValueError("No classes selected")
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

        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        clip_dir, label = self.samples[idx]
        frames = sorted(clip_dir.glob("*.jpg")) + sorted(clip_dir.glob("*.png"))
        start = random.randint(0, len(frames) - self.seq_len)
        selected = frames[start : start + self.seq_len]
        imgs = [self.transform(Image.open(f).convert("RGB")) for f in selected]
        seq = torch.stack(imgs, dim=0)
        return seq, label


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load class names
    classes_path = Path("./weights/classes.txt")
    with open(classes_path, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    print(f"Classes: {class_names}")

    # Create dataset
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    dataset = FrameSequenceDataset(
        "../HMDB51",
        seq_len=16,
        image_size=128,
        class_filter=class_names,
        transform=val_transform,
    )
    print(f"Evaluating on {len(dataset)} clips\n")

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    # Load model
    model = CNNLSTM(
        num_classes=len(class_names),
        feature_dim=256,
        hidden_dim=128,
        encoder="resnet18",
    ).to(device)

    state_dict = torch.load("./weights/model.pt", map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded.\n")

    # Evaluate
    all_preds = []
    all_labels = []
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for seq, labels in tqdm(dataloader, desc="Evaluating"):
            seq = seq.to(device)
            labels = labels.to(device)
            logits = model(seq)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = total_correct / total_samples

    print("\n" + "=" * 60)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Correct: {total_correct}/{total_samples}")
    print("=" * 60)

    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    print("-" * 60)
    for i, class_name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == i).sum() / mask.sum()
            count = mask.sum()
            print(f"{class_name:>15} {class_acc:.4f} ({class_acc*100:.2f}%) [{int(count)} samples]")

    # Confusion matrix
    print("\nConfusion Matrix:")
    print("-" * 60)
    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true, pred] += 1
    
    print("Predicted →")
    print("Actual ↓    " + "  ".join([f"{c:>8}" for c in class_names]))
    for i, class_name in enumerate(class_names):
        print(f"{class_name:>10} " + "  ".join([f"{cm[i, j]:>8}" for j in range(len(class_names))]))


if __name__ == "__main__":
    evaluate()
