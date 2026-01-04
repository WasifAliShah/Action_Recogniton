"""
Self-contained training script for Kaggle.
Upload this file + HMDB51 dataset to Kaggle, then run train_model().
"""

import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tv_models
from torchvision import transforms
from tqdm import tqdm


# ========== Model Definition ==========
class CNNLSTM(nn.Module):
    """CNN encoder + LSTM temporal head. Supports tiny conv stack or resnet18 backbone."""

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
        self.proj_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True, dropout=dropout if hidden_dim > 1 else 0)
        self.lstm_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, channels, height, width)
        batch, time, channels, height, width = x.shape
        x = x.view(batch * time, channels, height, width)
        feats = self.encoder(x).view(batch * time, -1)
        feats = self.proj(feats)
        feats = self.proj_dropout(feats)
        feats = feats.view(batch, time, -1)
        outputs, _ = self.lstm(feats)
        outputs = self.lstm_dropout(outputs)
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

        # Use provided transform or default
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        clip_dir, label = self.samples[idx]
        frames = sorted(clip_dir.glob("*.jpg")) + sorted(clip_dir.glob("*.png"))
        if len(frames) < self.seq_len:
            raise IndexError("Not enough frames in clip")
        start = random.randint(0, len(frames) - self.seq_len)
        selected = frames[start : start + self.seq_len]
        imgs = [self.transform(Image.open(f).convert("RGB")) for f in selected]
        seq = torch.stack(imgs, dim=0)  # (T, C, H, W)
        return seq, label


# ========== Training Function ==========
def train_model(
    data_root: str,
    save_dir: str = "./weights",
    epochs: int = 10,
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
    """
    Train CNN+LSTM action recognition model.
    
    Args:
        data_root: Path to dataset (class/clip/frame.jpg structure)
        save_dir: Where to save model.pt and classes.txt
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        seq_len: Number of frames per sequence
        image_size: Input image size (square)
        feature_dim: Feature dimension after encoder
        hidden_dim: LSTM hidden dimension
        encoder: "resnet18" or "tiny"
        weight_decay: L2 regularization
        val_split: Validation fraction (0-1)
        classes: Optional list of class names to filter (None = all)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Separate transforms for train (augmented) and val (clean)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Create datasets with appropriate transforms
    train_dataset = FrameSequenceDataset(
        data_root,
        seq_len=seq_len,
        image_size=image_size,
        class_filter=classes,
        transform=train_transform,
    )
    val_dataset = FrameSequenceDataset(
        data_root,
        seq_len=seq_len,
        image_size=image_size,
        class_filter=classes,
        transform=val_transform,
    )
    print(f"Dataset: {len(train_dataset)} clips, {len(train_dataset.class_names)} classes")
    print(f"Classes: {train_dataset.class_names}")

    # Train/val split
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    val_size = max(1, int(len(indices) * val_split))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if not train_indices:
        raise ValueError("Not enough data after split; reduce val_split or add data")

    train_ds = torch.utils.data.Subset(train_dataset, train_indices)
    val_ds = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = CNNLSTM(
        num_classes=len(train_dataset.class_names),
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        encoder=encoder,
    ).to(device)
    
    print(f"Model: {encoder} encoder, feature_dim={feature_dim}, hidden_dim={hidden_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Calculate class weights to handle any imbalance
    class_counts = {}
    for _, label in train_dataset.samples:
        class_name = train_dataset.class_names[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    total_samples = sum(class_counts.values())
    class_weights = torch.tensor([
        total_samples / (len(class_counts) * class_counts[class_name])
        for class_name in train_dataset.class_names
    ], dtype=torch.float32).to(device)
    
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {dict(zip(train_dataset.class_names, class_weights.cpu().numpy()))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 5
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
            
            # Mixed precision training
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
        train_loss = running_loss / total
        train_acc = correct / total if total else 0.0

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
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{v_correct/v_total:.3f}'})
        val_loss = v_loss / v_total if v_total else 0.0
        val_acc = v_correct / v_total if v_total else 0.0

        # Update learning rate
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"train_loss: {train_loss:.4f} train_acc: {train_acc:.3f} - "
            f"val_loss: {val_loss:.4f} val_acc: {val_acc:.3f} - "
            f"lr: {current_lr:.6f}"
        )

        # Save best model and early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), weights_path)
            classes_path.write_text("\n".join(train_dataset.class_names), encoding="utf-8")
            print(f"  → Saved checkpoint (val_acc={val_acc:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs (patience={max_patience})")
                break

    print(f"\nTraining complete!")
    print(f"Best val_acc: {best_val_acc:.3f}")
    print(f"Weights saved to: {weights_path}")
    print(f"Classes saved to: {classes_path}")


# ========== Example Usage ==========
if __name__ == "__main__":
    # Adjust paths for your Kaggle environment
    DATA_ROOT = "/kaggle/input/hmdb51/HMDB51"  # or your uploaded dataset path
    OUTPUT_DIR = "/kaggle/working/weights"
    
    # Train on balanced subset with class weights
    train_model(
        data_root=DATA_ROOT,
        save_dir=OUTPUT_DIR,
        epochs=15,  # More epochs for balanced data
        batch_size=8,
        seq_len=16,
        image_size=128,
        lr=5e-4,
        encoder="resnet18",
        feature_dim=256,
        hidden_dim=128,
        classes=["jump", "run", "sit", "walk"],  # Balanced subset
    )
