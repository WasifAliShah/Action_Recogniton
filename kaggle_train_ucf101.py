"""
Self-contained UCF101 training script for Kaggle.
Reads videos with OpenCV, samples seq_len frames, trains CNN+LSTM.

Usage on Kaggle:
- Upload this file to your Kaggle Notebook environment
- Add the UCF101 dataset under /kaggle/input (as a Dataset input)
- Optionally run: !pip install -q opencv-python-headless
- Then: !python kaggle_train_ucf101.py

Outputs:
- /kaggle/working/weights/model.pt
- /kaggle/working/weights/classes.txt
"""

import os
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
    raise RuntimeError("OpenCV is required. In Kaggle, run: !pip install -q opencv-python-headless") from exc


# ========== Model ==========
class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        encoder: str = "resnet18",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
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
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True, dropout=dropout if hidden_dim > 1 else 0)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.encoder(x).view(b * t, -1)
        feats = self.proj(feats)
        feats = self.dropout(feats)
        feats = feats.view(b, t, -1)
        outputs, _ = self.lstm(feats)
        outputs = self.dropout(outputs)
        logits = self.head(outputs[:, -1, :])
        return logits


# ========== Dataset ==========
class VideoSequenceDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        seq_len: int = 16,
        image_size: int = 128,
        class_filter: List[str] | None = None,
        transform = None,
        min_frames: int = 16,
        extensions: Tuple[str, ...] = (".avi", ".mp4", ".mkv"),
        samples: List[Tuple[Path, int]] | None = None,
        class_names_override: List[str] | None = None,
    ) -> None:
        # Resolve root to the directory that actually contains class folders.
        raw_root = Path(root)
        resolved_root = raw_root
        try:
            subdirs = [d for d in raw_root.iterdir() if d.is_dir()]
            # Prefer nested 'UCF-101' folder if present (common Kaggle layout: UCF101/UCF-101/<classes>)
            for d in subdirs:
                if d.name.lower() in ("ucf-101", "ucf101"):
                    inner = [x for x in d.iterdir() if x.is_dir()]
                    if len(inner) >= 20:
                        resolved_root = d
                        break
            else:
                # If there's a single subfolder with many class dirs, use it
                if len(subdirs) == 1:
                    inner = subdirs[0]
                    inner_subdirs = [x for x in inner.iterdir() if x.is_dir()]
                    if len(inner_subdirs) >= 20:
                        resolved_root = inner
        except Exception:
            pass

        self.root = resolved_root
        self.seq_len = seq_len
        self.image_size = image_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        # If explicit samples provided (from official splits), use them directly.
        if samples is not None and len(samples) > 0:
            self.samples = samples
            # Use provided class names if available; else derive from sample labels
            if class_names_override is not None:
                self.class_names = class_names_override
            else:
                max_label = max(lbl for _, lbl in samples)
                self.class_names = [str(i) for i in range(max_label + 1)]
            self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        else:
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
                        cap = cv2.VideoCapture(str(video_path))
                        if cap.isOpened():
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                            cap.release()
                            if frame_count >= min_frames:
                                self.samples.append((video_path, self.class_to_idx[class_dir.name]))
            if not self.samples:
                raise ValueError("No videos found under classes.")

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
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
        frames: List[Image.Image] = []
        last_img: Image.Image | None = None
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                if last_img is not None:
                    frames.append(last_img.copy())
                    continue
                ok2, frame2 = cap.read()
                if not ok2 or frame2 is None:
                    continue
                frame = frame2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            last_img = img
            frames.append(img)
        cap.release()
        if len(frames) == 0:
            raise RuntimeError(f"No readable frames in {video_path}")
        while len(frames) < num_frames:
            frames.append(frames[-1])
        return frames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        imgs = self._read_frames(video_path, self.seq_len)
        tensor_frames = [self.transform(img) for img in imgs]
        seq = torch.stack(tensor_frames, dim=0)
        return seq, label


# ========== Training ==========

def train_model(
    data_root: str,
    save_dir: str = "/kaggle/working/weights",
    epochs: int = 12,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tx = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_tx = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Try official train/test split files if present
    use_official = True
    try:
        splits_root = _auto_ucf_splits_root()
    except Exception:
        use_official = False

    if use_official:
        split_id = int(os.environ.get("SPLIT_ID", "1"))
        train_samples, val_samples, class_names = _load_ucf_splits(splits_root, data_root, split_id)

        full_train = VideoSequenceDataset(
            data_root,
            seq_len=seq_len,
            image_size=image_size,
            transform=train_tx,
            min_frames=seq_len,
            samples=train_samples,
            class_names_override=class_names,
        )
        full_val = VideoSequenceDataset(
            data_root,
            seq_len=seq_len,
            image_size=image_size,
            transform=val_tx,
            min_frames=seq_len,
            samples=val_samples,
            class_names_override=class_names,
        )

        train_ds = full_train
        val_ds = full_val
        print(f"Using official split {split_id}")
    else:
        full_train = VideoSequenceDataset(
            data_root,
            seq_len=seq_len,
            image_size=image_size,
            class_filter=classes,
            transform=train_tx,
            min_frames=seq_len,
        )
        full_val = VideoSequenceDataset(
            data_root,
            seq_len=seq_len,
            image_size=image_size,
            class_filter=classes,
            transform=val_tx,
            min_frames=seq_len,
        )

        indices = list(range(len(full_train)))
        random.shuffle(indices)
        val_size = max(1, int(len(indices) * val_split))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        if not train_indices:
            raise ValueError("Not enough data after split")

        train_ds = torch.utils.data.Subset(full_train, train_indices)
        val_ds = torch.utils.data.Subset(full_val, val_indices)

    print(f"Train videos: {len(train_ds)} | Val videos: {len(val_ds)}")
    print(f"Classes: {full_train.class_names}")

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

    class_counts = {c: 0 for c in full_train.class_names}
    for _, label in full_train.samples:
        c = full_train.class_names[label]
        class_counts[c] += 1
    total_samples = sum(class_counts.values())
    class_weights = torch.tensor([
        total_samples / (len(class_counts) * max(1, class_counts[c]))
        for c in full_train.class_names
    ], dtype=torch.float32).to(device)
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {dict(zip(full_train.class_names, class_weights.cpu().numpy()))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Some Kaggle environments use a PyTorch version without the 'verbose' arg.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 6

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    weights_path = save_path / "model.pt"
    classes_path = save_path / "classes.txt"

    for epoch in range(epochs):
        model.train()
        run_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for seq, labels in pbar:
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
            run_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/max(1,total):.3f}'})
        train_loss = run_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Val
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ", leave=False)
        with torch.no_grad():
            for seq, labels in pbar:
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
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{v_correct/max(1,v_total):.3f}'})
        val_loss = v_loss / max(1, v_total)
        val_acc = v_correct / max(1, v_total)

        scheduler.step(val_acc)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} train_acc: {train_acc:.3f} - val_loss: {val_loss:.4f} val_acc: {val_acc:.3f} - lr: {lr_now:.6f}")

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


def _auto_ucf_root() -> str:
    """Try to auto-detect UCF101 root under /kaggle/input."""
    candidates = [
        "/kaggle/input/ucf101/UCF101",
        "/kaggle/input/UCF101/UCF101",
        "/kaggle/input/ucf-101/UCF101",
        "/kaggle/input/ucf101",
        "/kaggle/input/UCF101",
    ]
    for p in candidates:
        path = Path(p)
        if path.exists():
            # Prefer nested 'UCF-101' subfolder if present
            nested = path / "UCF-101"
            if nested.exists():
                class_dirs = [d for d in nested.iterdir() if d.is_dir()]
                if len(class_dirs) >= 20:
                    return str(nested)
            # Otherwise check direct
            class_dirs = [d for d in path.iterdir() if d.is_dir()]
            if len(class_dirs) >= 20:
                return p
    # Fallback: search /kaggle/input
    root = Path("/kaggle/input")
    for sub in root.iterdir():
        if sub.is_dir():
            nested = sub / "UCF-101"
            if nested.exists():
                class_dirs = [d for d in nested.iterdir() if d.is_dir()]
                if len(class_dirs) >= 20:
                    return str(nested)
            class_dirs = [d for d in sub.iterdir() if d.is_dir()]
            if len(class_dirs) >= 20:
                return str(sub)
    raise RuntimeError("Could not auto-detect UCF101 root. Set DATA_ROOT to your dataset path.")


def _auto_ucf_splits_root() -> str:
    """Try to find UCF101 official split files under /kaggle/input."""
    candidates = [
        "/kaggle/input/ucf101traintestsplits-recognition/ucfTrainTestlist",
        "/kaggle/input/UCF101TrainTestSplits-Recognition/ucfTrainTestlist",
        "/kaggle/input/ucf101TraiNTest/ucfTrainTestlist",
    ]
    for p in candidates:
        path = Path(p)
        if path.exists():
            # Expect classInd.txt + trainlist/testlist files
            ci = path / "classInd.txt"
            if ci.exists():
                return str(path)
    # Fallback scan
    root = Path("/kaggle/input")
    for sub in root.iterdir():
        if sub.is_dir():
            ci = sub / "ucfTrainTestlist" / "classInd.txt"
            if ci.exists():
                return str(sub / "ucfTrainTestlist")
    raise RuntimeError("UCF101 split files not found under /kaggle/input")


def _load_ucf_splits(splits_root: str, ucf_root: str, split_id: int) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[str]]:
    """Load official train/test splits.

    Returns:
        train_samples: list of (video_path, label)
        val_samples: list of (video_path, label)
        class_names: list of class names ordered by classInd.txt
    """
    sr = Path(splits_root)
    ur_raw = Path(ucf_root)
    # Resolve nested class dir as earlier
    ur = ur_raw
    nested = ur_raw / "UCF-101"
    if nested.exists():
        ur = nested

    # Parse classInd.txt: lines like "1 ApplyEyeMakeup"
    class_map: dict[str, int] = {}
    class_names: List[str] = []
    with open(sr / "classInd.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                idx = int(parts[0])
                name = parts[1]
                class_map[name] = idx - 1  # zero-based
    # Preserve order by index
    class_names = [n for n, _ in sorted(class_map.items(), key=lambda kv: kv[1])]

    def parse_list(list_path: Path, is_train: bool) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                rel = parts[0]  # e.g., ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi
                label: int
                if is_train and len(parts) >= 2:
                    # trainlist includes label index (1-based)
                    label = int(parts[1]) - 1
                else:
                    # testlist does not include label; derive from class name prefix
                    class_name = rel.split("/")[0]
                    label = class_map.get(class_name, -1)
                    if label < 0:
                        continue
                video_path = ur / rel
                samples.append((video_path, label))
        return samples

    train_file = sr / f"trainlist0{split_id}.txt"
    test_file = sr / f"testlist0{split_id}.txt"
    if not train_file.exists() or not test_file.exists():
        raise RuntimeError(f"Split files not found for split {split_id} at {sr}")

    train_samples = parse_list(train_file, is_train=True)
    val_samples = parse_list(test_file, is_train=False)
    return train_samples, val_samples, class_names


if __name__ == "__main__":
    DATA_ROOT = os.environ.get("DATA_ROOT") or _auto_ucf_root()
    OUTPUT_DIR = "/kaggle/working/weights"

    train_model(
        data_root=DATA_ROOT,
        save_dir=OUTPUT_DIR,
        epochs=12,
        batch_size=8,
        seq_len=16,
        image_size=128,
        lr=5e-4,
        encoder="resnet18",
        feature_dim=256,
        hidden_dim=128,
        classes=None,
    )
