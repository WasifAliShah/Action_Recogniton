# =========================================================
# CNN + LSTM Action Recognition (Stanford-40, Colab Ready)
# =========================================================

import os
import json
import shutil
from copy import deepcopy
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
DATASET_ROOT = "/content/stanford40_dataset"
FULL_DATASET = os.path.join(DATASET_ROOT, "train_FUll")
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "test")

BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Saving paths
SAVE_DIR = "./saved"
os.makedirs(SAVE_DIR, exist_ok=True)

BEST_WEIGHTS_PATH = os.path.join(SAVE_DIR, "best_cnn_lstm.pth")
TORCHSCRIPT_PATH = os.path.join(SAVE_DIR, "best_cnn_lstm_scripted.pt")
CLASS_INDEX_PATH = os.path.join(SAVE_DIR, "class_index.json")

# -----------------------------
# Create train / test split (only once)
# -----------------------------
def has_classes(path):
    return os.path.exists(path) and any(
        os.path.isdir(os.path.join(path, d)) for d in os.listdir(path)
    )

if not has_classes(TRAIN_DIR):
    print("Creating train / test split...")

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    classes = os.listdir(FULL_DATASET)
    for cls in classes:
        cls_path = os.path.join(FULL_DATASET, cls)
        if not os.path.isdir(cls_path):
            continue

        images = os.listdir(cls_path)
        train_imgs, val_imgs = train_test_split(
            images, test_size=0.2, random_state=42
        )

        os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)

        for img in train_imgs:
            shutil.copy(
                os.path.join(cls_path, img),
                os.path.join(TRAIN_DIR, cls, img)
            )

        for img in val_imgs:
            shutil.copy(
                os.path.join(cls_path, img),
                os.path.join(VAL_DIR, cls, img)
            )

    print("Train / test split completed.")
else:
    print("Train / test split already exists and is valid. Skipping split.")

# -----------------------------
# Data Transformations
# -----------------------------
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# Datasets & DataLoaders
# -----------------------------
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=2)

num_classes = len(train_dataset.classes)
print(f"Number of classes: {num_classes}")

with open(CLASS_INDEX_PATH, "w") as f:
    json.dump(train_dataset.classes, f)

# -----------------------------
# Model Definition
# -----------------------------
class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        # Freeze CNN backbone
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.cnn.fc = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=512,
            batch_first=True
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.cnn(x)   # (B, 2048)

        features = features.unsqueeze(1)  # (B, 1, 2048)
        lstm_out, _ = self.lstm(features)
        return self.fc(lstm_out[:, -1, :])

model = CNN_LSTM_Model(num_classes).to(DEVICE)

# -----------------------------
# Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

# -----------------------------
# Training Loop
# -----------------------------
best_acc = 0.0
best_state = None

print("Starting training (Frozen CNN Backbone)...")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Val Acc={val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        best_state = deepcopy(model.state_dict())
        torch.save(best_state, BEST_WEIGHTS_PATH)
        print("New best model saved.")

print(f"Training complete. Best Val Accuracy: {best_acc:.2f}%")

# -----------------------------
# Export TorchScript
# -----------------------------
print("Exporting TorchScript model...")
model.load_state_dict(best_state)
model.eval()

example_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
traced_model = torch.jit.trace(model, example_input)
traced_model.save(TORCHSCRIPT_PATH)

print(f"TorchScript saved to: {TORCHSCRIPT_PATH}")
