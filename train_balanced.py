"""
Train on the balanced HMDB51_balanced dataset locally.
Run this after creating the balanced dataset with balance_dataset.py
"""

import sys
sys.path.insert(0, './backend')

from kaggle_train import train_model

if __name__ == "__main__":
    # Train on locally balanced dataset
    train_model(
        data_root="./HMDB51_balanced",
        save_dir="./backend/weights",
        epochs=20,  # More epochs since data is smaller
        batch_size=8,
        seq_len=16,
        image_size=128,
        lr=5e-4,
        encoder="resnet18",
        feature_dim=256,
        hidden_dim=128,
        classes=["jump", "run", "sit", "walk"],
        val_split=0.15,  # 15% validation
    )
