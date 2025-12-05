"""
Baseline CNN for Humpback Whale Identification
- Handles class imbalance (new_whale has 9664 samples)
- Uses preprocessed data with proper augmentation
- Tracks metrics with Weights & Biases
- GPU acceleration
"""

import os
import random
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

import wandb

# -------------------------
# Configuration
# -------------------------

SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
IMG_SIZE = 224

# Data paths
DATA_DIR = Path("Data")
TRAIN_DIR = DATA_DIR / "train"
PREPROCESSED_DIR = Path("preprocessed_data")
TRAIN_CSV = PREPROCESSED_DIR / "train_clean.csv"

# Output
OUTPUT_DIR = Path("results/baseline_cnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# W&B Configuration
WANDB_PROJECT = "whale-identification"
WANDB_ENTITY = "orisin-ben-gurion-university-of-the-negev"

# Class imbalance handling
# Exclude new_whale class as it has 9664 samples and would dominate
HANDLE_NEW_WHALE = "exclude"  # Options: "exclude", "downsample", "weighted"
NEW_WHALE_SAMPLES = 500  # Max samples to keep if downsampling (not used if exclude)

# -------------------------
# Reproducibility
# -------------------------

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# -------------------------
# Device Setup
# -------------------------

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("⚠ Using CPU (GPU not available)")
    return device


DEVICE = get_device()


# -------------------------
# Dataset
# -------------------------

class WhaleDataset(Dataset):
    """Dataset for whale images."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path,
        transform=None,
        label_to_idx: Dict = None
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
        # Create label mapping if not provided
        if label_to_idx is None:
            unique_labels = sorted(df['Id'].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx
        
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image']
        label = row['Id']
        
        # Load image
        img_path = self.img_dir / img_name
        img = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Get label index
        label_idx = self.label_to_idx[label]
        
        return img, label_idx


# -------------------------
# Data Transforms
# -------------------------

def get_transforms(train: bool = True):
    """Get data augmentation transforms."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# -------------------------
# Data Preparation
# -------------------------

def prepare_data():
    """Load and prepare training data with class balancing."""
    print("\n" + "="*60)
    print("PREPARING DATA")
    print("="*60 + "\n")
    
    # Load data
    df = pd.read_csv(TRAIN_CSV)
    print(f"Total samples: {len(df)}")
    
    # Analyze class distribution
    class_counts = df['Id'].value_counts()
    print(f"Total classes: {len(class_counts)}")
    print(f"\nClass distribution:")
    print(f"  new_whale: {class_counts.get('new_whale', 0)} samples")
    print(f"  Other classes: {len(class_counts) - 1} classes")
    print(f"  Min samples per class: {class_counts[class_counts.index != 'new_whale'].min()}")
    print(f"  Max samples per class (excl. new_whale): {class_counts[class_counts.index != 'new_whale'].max()}")
    print(f"  Mean samples per class (excl. new_whale): {class_counts[class_counts.index != 'new_whale'].mean():.2f}")
    
    # Handle new_whale class - EXCLUDE it completely
    print(f"\n→ Excluding 'new_whale' class (9664 samples)")
    df = df[df['Id'] != 'new_whale'].reset_index(drop=True)
    print(f"  Remaining samples: {len(df)}")
    print(f"  Remaining classes: {df['Id'].nunique()}")
    
    # Show class distribution (keeping ALL classes, even those with 1-2 samples)
    final_class_counts = df['Id'].value_counts()
    print(f"\nClass distribution after removing new_whale:")
    print(f"  Total classes: {len(final_class_counts)}")
    print(f"  Min samples per class: {final_class_counts.min()}")
    print(f"  Max samples per class: {final_class_counts.max()}")
    print(f"  Mean samples per class: {final_class_counts.mean():.2f}")
    print(f"  Median samples per class: {final_class_counts.median():.1f}")
    print(f"  Classes with 1 sample: {(final_class_counts == 1).sum()}")
    print(f"  Classes with 2 samples: {(final_class_counts == 2).sum()}")
    print(f"  Classes with 3-5 samples: {((final_class_counts >= 3) & (final_class_counts <= 5)).sum()}")
    print(f"  Classes with >5 samples: {(final_class_counts > 5).sum()}")
    
    # For stratified train/val split, we need at least 3 samples per class
    # (2 for train, 1 for val with 85/15 split)
    # This keeps classes with 2+ samples and avoids the stratification error
    min_samples_per_class = 3
    class_counts = df['Id'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    df = df[df['Id'].isin(valid_classes)].reset_index(drop=True)
    
    print(f"\nAfter filtering classes with < {min_samples_per_class} samples (for stratified split):")
    print(f"  Samples: {len(df)}")
    print(f"  Classes: {df['Id'].nunique()}")
    
    final_class_counts = df['Id'].value_counts()
    print(f"  Min samples per class: {final_class_counts.min()}")
    print(f"  Max samples per class: {final_class_counts.max()}")
    print(f"  Mean samples per class: {final_class_counts.mean():.2f}")
    
    # Train/Val split (stratified by class)
    train_df, val_df = train_test_split(
        df,
        test_size=0.15,  # 85/15 split
        random_state=SEED,
        stratify=df['Id']
    )
    
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    return train_df, val_df


# -------------------------
# Model Architecture
# -------------------------

class BaselineCNN(nn.Module):
    """Baseline CNN architecture for whale identification."""
    
    def __init__(self, num_classes: int):
        super(BaselineCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 224 -> 112
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 112 -> 56
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)  # 56 -> 28
        
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)  # 28 -> 14
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Conv blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.pool4(x)
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.dropout(x)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# -------------------------
# Training Functions
# -------------------------

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({"loss": loss.item()})
    
    # Calculate metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_f1


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Metrics
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({"loss": loss.item()})
    
    # Calculate metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    epoch_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall


# -------------------------
# Main Training Loop
# -------------------------

def main():
    """Main training function."""
    
    print("\n" + "="*60)
    print("BASELINE CNN TRAINING - WHALE IDENTIFICATION")
    print("="*60 + "\n")
    
    # Initialize W&B
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config={
            "architecture": "Baseline CNN",
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "img_size": IMG_SIZE,
            "optimizer": "Adam",
            "new_whale_handling": "excluded",
            "min_samples_per_class": 3,
            "val_split": 0.15,
            "data_source": "preprocessed_clean_csv",
        },
        name=f"baseline_cnn_no_new_whale"
    )
    
    # Prepare data
    train_df, val_df = prepare_data()
    
    # Create datasets
    train_dataset = WhaleDataset(
        train_df,
        TRAIN_DIR,
        transform=get_transforms(train=True)
    )
    
    val_dataset = WhaleDataset(
        val_df,
        TRAIN_DIR,
        transform=get_transforms(train=False),
        label_to_idx=train_dataset.label_to_idx
    )
    
    print(f"\nNumber of classes: {train_dataset.num_classes}")
    
    # Create dataloaders
    if HANDLE_NEW_WHALE == "weighted":
        # Calculate class weights
        class_counts = train_df['Id'].value_counts()
        class_weights = 1.0 / class_counts
        sample_weights = train_df['Id'].map(class_weights).values
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    model = BaselineCNN(num_classes=train_dataset.num_classes).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60 + "\n")
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_precision, val_recall = validate(
            model, val_loader, criterion, DEVICE
        )
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")
        print(f"Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Log to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "train/f1_score": train_f1,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "val/f1_score": val_f1,
            "val/precision": val_precision,
            "val/recall": val_recall,
            "learning_rate": current_lr,
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'label_to_idx': train_dataset.label_to_idx,
            }, OUTPUT_DIR / 'best_model.pth')
            print(f"✓ Saved best model (Val Acc: {val_acc:.4f})")
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Validation F1 Score: {best_val_f1:.4f}")
    print(f"\nModel saved to: {OUTPUT_DIR / 'best_model.pth'}")
    
    # Log final summary to W&B
    wandb.summary["best_val_accuracy"] = best_val_acc
    wandb.summary["best_val_f1_score"] = best_val_f1
    
    wandb.finish()


if __name__ == "__main__":
    main()
