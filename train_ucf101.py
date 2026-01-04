"""
Train CNN+LSTM on the UCF101 videos dataset (reads videos directly).
- Expects directory layout: root/class_name/*.avi|*.mp4|*.mkv
- Samples seq_len frames per video using OpenCV, applies transforms, and trains.

Requirements:
- pip install opencv-python torch torchvision tqdm

Outputs:
- Saves weights to ./backend/weights/model.pt
- Saves class list to ./backend/weights/classes.txt
"""

from pathlib import Path
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models
from torchvision import transforms
from tqdm import tqdm

try:
    import cv2  # type: ignore
except Exception as exc:
    raise RuntimeError("OpenCV (opencv-python) is required for UCF101 video loading. Install with: pip install opencv-python") from exc


# ========== Model Definition ==========
class CNNLSTM(nn.Module):
    """CNN encoder + LSTM temporal head. Supports resnet18 backbone (recommended)."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        encoder: str = "resnet18",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder
        if encoder == "resnet18":
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
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True, dropout=dropout if hidden_dim > 1 else 0)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, channels, height, width)
        batch, time, channels, height, width = x.shape
        x = x.view(batch * time, channels, height, width)
        feats = self.encoder(x).view(batch * time, -1)
        feats = self.proj(feats)
        feats = self.dropout(feats)
        feats = feats.view(batch, time, -1)
        outputs, _ = self.lstm(feats)
        outputs = self.dropout(outputs)
        logits = self.head(outputs[:, -1, :])
        return logits


# ========== Dataset ==========
class VideoSequenceDataset(Dataset):
    """Loads sequences from UCF101 videos: root/class_name/*.avi|*.mp4|*.mkv"""

    def __init__(
        self,
        root: str | Path,
        seq_len: int = 16,
        image_size: int = 128,
        class_filter: List[str] | None = None,
        transform = None,
        min_frames: int = 16,
        extensions: Tuple[str, ...] = (".avi", ".mp4", ".mkv"),
    ) -> None:
        self.root = Path(root)
        self.seq_len = seq_len
        self.image_size = image_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        all_classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        if class_filter:
            cf = set(class_filter)
            self.class_names = [c for c in all_classes if c in cf]
        else:
            self.class_names = all_classes
        if not self.class_names:
            raise ValueError("No classes selected; check class_filter")
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.samples: List[Tuple[Path, int]] = []
        for class_dir in self.root.iterdir():
            if not class_dir.is_dir() or class_dir.name not in self.class_to_idx:
                continue
            for video_path in class_dir.glob("**/*"):
                if video_path.suffix.lower() in extensions:
                    # Quickly check frame count
                    cap = cv2.VideoCapture(str(video_path))
                    if cap.isOpened():
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                        cap.release()
                        if frame_count >= min_frames:
                            self.samples.append((video_path, self.class_to_idx[class_dir.name]))
        if not self.samples:
            raise ValueError("No videos found. Expect root/class/*.avi|*.mp4|*.mkv with at least seq_len frames.")

    def __len__(self) -> int:
        return len(self.samples)

    def _read_frames(self, video_path: Path, num_frames: int) -> List[Image.Image]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            cap.release()
            raise RuntimeError(f"Video has no frames: {video_path}")
        # Select evenly spaced indices
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
        frames: List[Image.Image] = []
        last_img: Image.Image | None = None
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                # Fallback to last good frame
                if last_img is not None:
                    frames.append(last_img.copy())
                    continue
                else:
                    # Try reading next frame sequentially
                    ok2, frame2 = cap.read()
                    if not ok2 or frame2 is None:
                        continue
                    frame = frame2
            # BGR -> RGB, to PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            last_img = img
            frames.append(img)
        cap.release()
        if len(frames) == 0:
            raise RuntimeError(f"No readable frames in {video_path}")
        # Pad if short
        while len(frames) < num_frames:
            frames.append(frames[-1])
        return frames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        imgs = self._read_frames(video_path, self.seq_len)
        # Apply transform
        tensor_frames = [self.transform(img) for img in imgs]
        seq = torch.stack(tensor_frames, dim=0)  # (T, C, H, W)
        return seq, label


# ========== Training Function ==========

def train_model(
    data_root: str,
    save_dir: str = "./backend/weights",
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 5e-4,
    seq_len: int = 16,
    image_size: int = 128,
    feature_dim: int = 256,
    hidden_dim: int = 128,
    encoder: str = "resnet18",
    weight_decay: float = 1e-4,
    val_split: float = 0.1,
    classes: List[str] | None = None,
) -> None:
    """Train CNN+LSTM on UCF101 videos."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms (train with mild augmentation; val clean)
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    full_train = VideoSequenceDataset(
        data_root,
        seq_len=seq_len,
        image_size=image_size,
        class_filter=classes,
        transform=train_transform,
        min_frames=seq_len,
    )
    full_val = VideoSequenceDataset(
        data_root,
        seq_len=seq_len,
        image_size=image_size,
        class_filter=classes,
        transform=val_transform,
        min_frames=seq_len,
    )

    print(f"Dataset: {len(full_train)} videos, {len(full_train.class_names)} classes")
    print(f"Classes: {full_train.class_names}")

    # Train/val split
    indices = list(range(len(full_train)))
    random.shuffle(indices)
    val_size = max(1, int(len(indices) * val_split))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if not train_indices:
        raise ValueError("Not enough data after split; reduce val_split or add data")

    train_ds = torch.utils.data.Subset(full_train, train_indices)
    val_ds = torch.utils.data.Subset(full_val, val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = CNNLSTM(
        num_classes=len(full_train.class_names),
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        encoder=encoder,
        dropout=0.5,
    ).to(device)

    print(f"Model: {encoder} encoder, feature_dim={feature_dim}, hidden_dim={hidden_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights to mitigate imbalance
    class_counts = {c: 0 for c in full_train.class_names}
    for _, label in full_train.samples:
        c = full_train.class_names[label]
        class_counts[c] += 1
    total_samples = sum(class_counts.values())
    class_weights = torch.tensor([
        total_samples / (len(class_counts) * max(1, class_counts[class_name]))
        for class_name in full_train.class_names
    ], dtype=torch.float32).to(device)
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {dict(zip(full_train.class_names, class_weights.cpu().numpy()))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 6

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    weights_path = save_path / "model.pt"
    classes_path = save_path / "classes.txt"

    for epoch in range(epochs):
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for seq, labels in train_pbar:
            seq = seq.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(seq)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(seq)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.3f}'})
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ", leave=False)
        with torch.no_grad():
            for seq, labels in val_pbar:
                seq = seq.to(device)
                labels = labels.to(device)
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = model(seq)
                        loss = criterion(logits, labels)
                else:
                    logits = model(seq)
                    loss = criterion(logits, labels)
                v_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{v_correct/max(1, v_total):.3f}'})
        val_loss = v_loss / max(1, v_total)
        val_acc = v_correct / max(1, v_total)

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"train_loss: {train_loss:.4f} train_acc: {train_acc:.3f} - "
            f"val_loss: {val_loss:.4f} val_acc: {val_acc:.3f} - "
            f"lr: {current_lr:.6f}"
        )

        # Save best and early stop
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), weights_path)
            classes_path.write_text("\n".join(full_train.class_names), encoding="utf-8")
            print(f"  → Saved checkpoint (val_acc={val_acc:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping after {epoch+1} epochs (patience={max_patience})")
                break

    print("\nTraining complete!")
    print(f"Best val_acc: {best_val_acc:.3f}")
    print(f"Weights saved to: {weights_path}")
    print(f"Classes saved to: {classes_path}")


# ========== Example Usage ==========
if __name__ == "__main__":
    # Adjust for your local dataset path
    DATA_ROOT = "./UCF101"  # e.g., C:/datasets/UCF101
    OUTPUT_DIR = "./backend/weights"

    # Train (optionally set classes=None for all 101 classes)
    train_model(
        data_root=DATA_ROOT,
        save_dir=OUTPUT_DIR,
        epochs=15,
        batch_size=8,
        seq_len=16,
        image_size=128,
        lr=5e-4,
        encoder="resnet18",
        feature_dim=256,
        hidden_dim=128,
        classes=None,  # or a subset like ["Walking", "Running", ...] matching folder names
    )
