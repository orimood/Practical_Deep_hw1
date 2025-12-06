"""
Quick Triplet Loss Model Test - Small Subsample
- Trains on a small subset to verify the approach works
- Faster iterations for debugging and validation
- Same architecture as full training
- W&B tracking
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import wandb

# -------------------------
# Configuration
# -------------------------

SEED = 42
BATCH_SIZE = 16  # Smaller batch for quick testing
NUM_EPOCHS = 10  # Fewer epochs for quick validation
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 2
IMG_SIZE = 224  # Smaller images for faster processing

# Triplet loss parameters
MARGIN = 0.3
EMBEDDING_DIM = 512

# QUICK TEST PARAMETERS
MAX_CLASSES = 100  # Only use 100 classes
MIN_SAMPLES_PER_CLASS = 5  # Ensure enough samples for triplet mining

# Data paths
DATA_DIR = Path("Data")
TRAIN_DIR = DATA_DIR / "train"
PREPROCESSED_DIR = Path("preprocessed_data")
TRAIN_CSV = PREPROCESSED_DIR / "train_clean.csv"

# Output
OUTPUT_DIR = Path("results/triplet_loss_quick_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# W&B Configuration
WANDB_PROJECT = "whale-identification"
WANDB_ENTITY = "orisin-ben-gurion-university-of-the-negev"

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
# Dataset - Triplet Mining
# -------------------------

class TripletDataset(Dataset):
    """Dataset that returns (anchor, positive, negative) triplets."""
    
    def __init__(self, df: pd.DataFrame, img_dir: Path, transform=None, is_training: bool = True):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_training = is_training
        
        # Reset index to ensure sequential indexing (critical for iloc)
        self.df = self.df.reset_index(drop=True)
        
        # Create label mapping
        unique_labels = sorted(df['Id'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        
        # Group images by label for triplet mining
        self.label_to_images = {}
        for idx in range(len(self.df)):
            label = self.df.iloc[idx]['Id']
            if label not in self.label_to_images:
                self.label_to_images[label] = []
            self.label_to_images[label].append(idx)
        
        # Get labels with multiple samples for proper triplet mining
        self.valid_labels = [label for label, imgs in self.label_to_images.items() if len(imgs) >= 2]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get anchor
        anchor_row = self.df.iloc[idx]
        anchor_label = anchor_row['Id']
        anchor_img = self._load_image(anchor_row['Image'])
        
        if self.is_training:
            # Sample positive (same whale)
            positive_idx = idx
            while positive_idx == idx:
                positive_idx = random.choice(self.label_to_images[anchor_label])
            positive_row = self.df.iloc[positive_idx]
            positive_img = self._load_image(positive_row['Image'])
            
            # Sample negative (different whale)
            negative_label = anchor_label
            while negative_label == anchor_label:
                negative_label = random.choice(self.valid_labels)
            negative_idx = random.choice(self.label_to_images[negative_label])
            negative_row = self.df.iloc[negative_idx]
            negative_img = self._load_image(negative_row['Image'])
            
            # Get label index
            label_idx = self.label_to_idx[anchor_label]
            
            return anchor_img, positive_img, negative_img, label_idx
        else:
            # For validation, just return anchor and label
            label_idx = self.label_to_idx[anchor_label]
            return anchor_img, label_idx
    
    def _load_image(self, img_name: str) -> torch.Tensor:
        """Load and transform image."""
        img_path = self.img_dir / img_name
        img = Image.open(img_path)
        
        # Convert grayscale to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img


# -------------------------
# Model Architecture
# -------------------------

class SEResNetTriplet(nn.Module):
    """
    SEResNet50 backbone for triplet loss
    Combines global and local features
    """
    
    def __init__(self, embedding_dim: int = 512, num_classes: int = None, pretrained: bool = True):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Load pretrained SEResNet50
        self.backbone = timm.create_model('seresnet50', pretrained=pretrained, num_classes=0)
        backbone_out_features = self.backbone.num_features  # 2048 for resnet50
        
        # Global and local feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = nn.AdaptiveMaxPool2d(1)
        
        # Feature dimension after concatenation
        combined_features = backbone_out_features * 2
        
        # Embedding projection
        self.embedding = nn.Sequential(
            nn.BatchNorm1d(combined_features),
            nn.Dropout(0.5),
            nn.Linear(combined_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Optional classification head for combined loss
        if num_classes:
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            self.classifier = None
    
    def forward(self, x, return_embedding: bool = True):
        # Extract features from backbone
        features = self.backbone.forward_features(x)  # [B, 2048, H, W]
        
        # Global and local pooling
        global_feat = self.global_pool(features).flatten(1)  # [B, 2048]
        local_feat = self.local_pool(features).flatten(1)    # [B, 2048]
        
        # Concatenate
        combined = torch.cat([global_feat, local_feat], dim=1)  # [B, 4096]
        
        # Project to embedding space
        embedding = self.embedding(combined)  # [B, embedding_dim]
        
        # L2 normalize for triplet loss
        embedding_norm = F.normalize(embedding, p=2, dim=1)
        
        if return_embedding:
            if self.classifier is not None:
                logits = self.classifier(embedding_norm)
                return embedding_norm, logits
            return embedding_norm
        else:
            return embedding


# -------------------------
# Loss Functions
# -------------------------

class OnlineHardTripletLoss(nn.Module):
    """
    Online hard triplet loss.
    Selects hardest positive and hardest negative within batch.
    """
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, embedding_dim] L2-normalized embeddings
            labels: [B] class labels
        """
        # Compute pairwise distances
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)  # [B, B]
        
        # For each anchor, find hardest positive and hardest negative
        losses = []
        for i in range(len(labels)):
            anchor_label = labels[i]
            
            # Mask for same class (positive) and different class (negative)
            positive_mask = (labels == anchor_label) & (torch.arange(len(labels), device=labels.device) != i)
            negative_mask = labels != anchor_label
            
            if positive_mask.sum() == 0 or negative_mask.sum() == 0:
                continue
            
            # Hardest positive: farthest same-class sample
            hardest_positive_dist = pairwise_dist[i][positive_mask].max()
            
            # Hardest negative: closest different-class sample
            hardest_negative_dist = pairwise_dist[i][negative_mask].min()
            
            # Triplet loss
            loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return torch.stack(losses).mean()


def top_k_bce_loss(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Top-k BCE loss for multi-label classification.
    Only considers top-k predictions.
    """
    batch_size = logits.size(0)
    num_classes = logits.size(1)
    
    # Create one-hot labels
    labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
    
    # BCE loss
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels_one_hot, reduction='none')
    
    # Take top-k
    topk_loss, _ = torch.topk(bce_loss, k=min(k, num_classes), dim=1)
    
    return topk_loss.mean()


# -------------------------
# Data Preparation
# -------------------------

def prepare_data():
    """Load and prepare data - SMALL SUBSAMPLE VERSION."""
    print("\n" + "=" * 60)
    print("PREPARING SMALL SUBSAMPLE FOR QUICK TEST")
    print("=" * 60 + "\n")
    
    # Load data
    df = pd.read_csv(TRAIN_CSV)
    print(f"Total samples: {len(df)}")
    
    # Exclude new_whale
    df = df[df['Id'] != 'new_whale'].copy()
    print(f"\n→ Excluding 'new_whale' class")
    print(f"  Remaining samples: {len(df)}")
    print(f"  Remaining classes: {df['Id'].nunique()}")
    
    # Filter to classes with enough samples
    class_counts = df['Id'].value_counts()
    valid_classes = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index.tolist()
    df = df[df['Id'].isin(valid_classes)].copy()
    
    print(f"\nAfter filtering classes with < {MIN_SAMPLES_PER_CLASS} samples:")
    print(f"  Samples: {len(df)}")
    print(f"  Classes: {df['Id'].nunique()}")
    
    # SELECT RANDOM SUBSET OF CLASSES
    all_classes = df['Id'].unique()
    np.random.seed(SEED)
    selected_classes = np.random.choice(all_classes, size=min(MAX_CLASSES, len(all_classes)), replace=False)
    df = df[df['Id'].isin(selected_classes)].copy()
    
    print(f"\n→ Using random subset of {MAX_CLASSES} classes")
    print(f"  Samples: {len(df)}")
    print(f"  Classes: {df['Id'].nunique()}")
    
    class_counts = df['Id'].value_counts()
    print(f"  Min samples per class: {class_counts.min()}")
    print(f"  Max samples per class: {class_counts.max()}")
    print(f"  Mean samples per class: {class_counts.mean():.2f}")
    print(f"  Median samples per class: {class_counts.median()}")
    
    # Split data
    train_df, val_df = train_test_split(
        df, 
        test_size=0.15, 
        random_state=SEED,
        stratify=df['Id']
    )
    
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    return train_df, val_df


# -------------------------
# Training & Validation
# -------------------------

def train_epoch(model, dataloader, optimizer, triplet_criterion, device):
    """Train for one epoch."""
    model.train()
    
    total_triplet_loss = 0.0
    total_bce_loss = 0.0
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for anchor, positive, negative, labels in pbar:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Get embeddings and logits
        anchor_emb, anchor_logits = model(anchor)
        positive_emb, _ = model(positive)
        negative_emb, _ = model(negative)
        
        # Combine all embeddings for triplet loss
        all_embeddings = torch.cat([anchor_emb, positive_emb, negative_emb], dim=0)
        all_labels = torch.cat([labels, labels, labels], dim=0)
        
        # Compute triplet loss
        triplet_loss = triplet_criterion(all_embeddings, all_labels)
        
        # Compute classification loss
        bce_loss = top_k_bce_loss(anchor_logits, labels, k=5)
        
        # Combined loss
        loss = triplet_loss + 0.5 * bce_loss
        
        loss.backward()
        optimizer.step()
        
        total_triplet_loss += triplet_loss.item()
        total_bce_loss += bce_loss.item()
        total_loss += loss.item()
        
        pbar.set_postfix({
            'triplet': f'{triplet_loss.item():.4f}',
            'bce': f'{bce_loss.item():.4f}',
            'total': f'{loss.item():.4f}'
        })
    
    avg_triplet = total_triplet_loss / len(dataloader)
    avg_bce = total_bce_loss / len(dataloader)
    avg_total = total_loss / len(dataloader)
    
    return avg_triplet, avg_bce, avg_total


def validate(model, dataloader, device):
    """Validate the model using embedding distances."""
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            embeddings, _ = model(images)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute pairwise distances
    distances = torch.cdist(all_embeddings, all_embeddings, p=2)
    
    # For each sample, find nearest neighbor (excluding itself)
    correct = 0
    for i in range(len(all_labels)):
        # Mask out self
        dist_row = distances[i].clone()
        dist_row[i] = float('inf')
        
        # Find nearest neighbor
        nearest_idx = dist_row.argmin()
        
        # Check if same class
        if all_labels[i] == all_labels[nearest_idx]:
            correct += 1
    
    accuracy = correct / len(all_labels)
    return accuracy


# -------------------------
# Main Training Loop
# -------------------------

def main():
    print("\n" + "=" * 60)
    print("QUICK TRIPLET LOSS TEST - WHALE IDENTIFICATION")
    print("=" * 60 + "\n")
    
    # Initialize W&B
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="triplet_loss_quick_test",
        config={
            "architecture": "SEResNet50",
            "dataset": "whale-identification-quick",
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "embedding_dim": EMBEDDING_DIM,
            "margin": MARGIN,
            "img_size": IMG_SIZE,
            "max_classes": MAX_CLASSES,
            "min_samples_per_class": MIN_SAMPLES_PER_CLASS,
        }
    )
    
    # Prepare data
    train_df, val_df = prepare_data()
    num_classes = train_df['Id'].nunique()
    
    print(f"\nNumber of classes: {num_classes}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = TripletDataset(train_df, TRAIN_DIR, transform=train_transform, is_training=True)
    val_dataset = TripletDataset(val_df, TRAIN_DIR, transform=val_transform, is_training=False)
    
    # Create dataloaders
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
    model = SEResNetTriplet(
        embedding_dim=EMBEDDING_DIM,
        num_classes=num_classes,
        pretrained=True
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Loss and optimizer
    triplet_criterion = OnlineHardTripletLoss(margin=MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60 + "\n")
    
    best_val_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_triplet_loss, train_bce_loss, train_total_loss = train_epoch(
            model, train_loader, optimizer, triplet_criterion, DEVICE
        )
        
        # Validate
        val_acc = validate(model, val_loader, DEVICE)
        
        # Update scheduler
        scheduler.step(train_total_loss)
        
        # Log metrics
        metrics = {
            "epoch": epoch + 1,
            "train/triplet_loss": train_triplet_loss,
            "train/bce_loss": train_bce_loss,
            "train/total_loss": train_total_loss,
            "val/accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        wandb.log(metrics)
        
        # Print summary
        print(f"\nTrain - Triplet: {train_triplet_loss:.4f}, BCE: {train_bce_loss:.4f}, Total: {train_total_loss:.4f}")
        print(f"Val   - Accuracy: {val_acc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, OUTPUT_DIR / 'best_model.pth')
            print(f"✓ Saved best model (val_acc: {val_acc:.4f})")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {OUTPUT_DIR / 'best_model.pth'}")
    
    wandb.finish()


if __name__ == "__main__":
    main()
