"""
Triplet Loss Model for Humpback Whale Identification
- Uses metric learning instead of classification
- Inspired by top Kaggle solutions
- SENet-based feature extractor
- Combined triplet + BCE loss
- Preprocessed data with augmentation
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
BATCH_SIZE = 32  # Smaller for triplet mining
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
IMG_SIZE = 256  # Width
IMG_HEIGHT = 512  # Height (like Kaggle solution)

# Triplet loss parameters
MARGIN = 0.3
EMBEDDING_DIM = 512

# Data paths
DATA_DIR = Path("Data")
TRAIN_DIR = DATA_DIR / "train"
PREPROCESSED_DIR = Path("preprocessed_data")
TRAIN_CSV = PREPROCESSED_DIR / "train_clean.csv"

# Output
OUTPUT_DIR = Path("results/triplet_loss")
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
# Dataset
# -------------------------

class TripletDataset(Dataset):
    """Dataset for triplet loss training."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path,
        transform=None,
        is_training: bool = True
    ):
        self.df = df.reset_index(drop=True)
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
            
            label_idx = self.label_to_idx[anchor_label]
            
            return anchor_img, positive_img, negative_img, label_idx
        else:
            label_idx = self.label_to_idx[anchor_label]
            return anchor_img, label_idx
    
    def _load_image(self, img_name):
        """Load and transform image."""
        img_path = self.img_dir / img_name
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img


# -------------------------
# Data Transforms
# -------------------------

def get_transforms(train: bool = True):
    """Get data augmentation transforms."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# -------------------------
# Data Preparation
# -------------------------

def prepare_data():
    """Load and prepare training data."""
    print("\n" + "="*60)
    print("PREPARING DATA FOR TRIPLET LOSS")
    print("="*60 + "\n")
    
    # Load data
    df = pd.read_csv(TRAIN_CSV)
    print(f"Total samples: {len(df)}")
    
    # Exclude new_whale
    print(f"\n→ Excluding 'new_whale' class")
    df = df[df['Id'] != 'new_whale'].reset_index(drop=True)
    print(f"  Remaining samples: {len(df)}")
    print(f"  Remaining classes: {df['Id'].nunique()}")
    
    # For triplet loss with stratified split, we need at least 3 samples per class
    # (minimum 2 in train, 1 in val for 85/15 split)
    min_samples = 3
    class_counts = df['Id'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    df = df[df['Id'].isin(valid_classes)].reset_index(drop=True)
    
    print(f"\nAfter filtering classes with < {min_samples} samples:")
    print(f"  Samples: {len(df)}")
    print(f"  Classes: {df['Id'].nunique()}")
    
    final_class_counts = df['Id'].value_counts()
    print(f"  Min samples per class: {final_class_counts.min()}")
    print(f"  Max samples per class: {final_class_counts.max()}")
    print(f"  Mean samples per class: {final_class_counts.mean():.2f}")
    print(f"  Median samples per class: {final_class_counts.median():.1f}")
    
    # Train/Val split (stratified)
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
# Model Architecture
# -------------------------

class TripletNet(nn.Module):
    """
    Triplet Network with metric learning.
    Inspired by Kaggle winning solutions using SENet + global/local features.
    """
    
    def __init__(self, num_classes: int, embedding_dim: int = EMBEDDING_DIM):
        super(TripletNet, self).__init__()
        
        # Use SE-ResNet as backbone (similar to senet154 but smaller)
        self.backbone = timm.create_model('seresnet50', pretrained=True, num_classes=0)
        
        # Get backbone output dimension
        backbone_out_dim = self.backbone.num_features
        
        # Global feature branch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_bn = nn.BatchNorm1d(backbone_out_dim)
        
        # Local feature branch (max pooling)
        self.local_pool = nn.AdaptiveMaxPool2d(1)
        self.local_bn = nn.BatchNorm1d(backbone_out_dim)
        
        # Combined features
        combined_dim = backbone_out_dim * 2
        
        # Embedding layer (normalized features for triplet loss)
        self.embedding = nn.Sequential(
            nn.Linear(combined_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        
        # Classification head (for auxiliary BCE loss)
        self.fc = nn.Linear(embedding_dim, num_classes, bias=False)
        
    def forward(self, x, return_embedding=False):
        # Extract features from backbone
        features = self.backbone.forward_features(x)
        
        # Global branch
        global_feat = self.global_pool(features).flatten(1)
        global_feat = self.global_bn(global_feat)
        
        # Local branch
        local_feat = self.local_pool(features).flatten(1)
        local_feat = self.local_bn(local_feat)
        
        # Combine features
        combined = torch.cat([global_feat, local_feat], dim=1)
        
        # Get embedding (L2 normalized)
        embedding = self.embedding(combined)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        if return_embedding:
            return embedding
        
        # Classification output (without bias, for metric learning)
        logits = self.fc(embedding)
        
        return embedding, logits


# -------------------------
# Loss Functions
# -------------------------

class TripletLoss(nn.Module):
    """Triplet loss with hard negative mining."""
    
    def __init__(self, margin: float = MARGIN):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        # Calculate distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss
        losses = F.relu(pos_dist - neg_dist + self.margin)
        
        return losses.mean()


# -------------------------
# Training Functions
# -------------------------

def train_epoch(model, dataloader, triplet_criterion, bce_criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_triplet_loss = 0.0
    running_bce_loss = 0.0
    running_total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for anchor, positive, negative, labels in pbar:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass for triplets
        anchor_emb, anchor_logits = model(anchor)
        positive_emb, _ = model(positive)
        negative_emb, _ = model(negative)
        
        # Triplet loss
        triplet_loss = triplet_criterion(anchor_emb, positive_emb, negative_emb)
        
        # BCE loss (auxiliary classification loss)
        bce_loss = bce_criterion(anchor_logits, labels)
        
        # Combined loss (weighted)
        total_loss = triplet_loss + 0.5 * bce_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Metrics
        running_triplet_loss += triplet_loss.item() * anchor.size(0)
        running_bce_loss += bce_loss.item() * anchor.size(0)
        running_total_loss += total_loss.item() * anchor.size(0)
        
        pbar.set_postfix({
            "triplet": f"{triplet_loss.item():.3f}",
            "bce": f"{bce_loss.item():.3f}",
            "total": f"{total_loss.item():.3f}"
        })
    
    epoch_triplet_loss = running_triplet_loss / len(dataloader.dataset)
    epoch_bce_loss = running_bce_loss / len(dataloader.dataset)
    epoch_total_loss = running_total_loss / len(dataloader.dataset)
    
    return epoch_triplet_loss, epoch_bce_loss, epoch_total_loss


def validate(model, dataloader, triplet_criterion, bce_criterion, device):
    """Validate the model using embedding similarity."""
    model.eval()
    running_triplet_loss = 0.0
    running_bce_loss = 0.0
    running_total_loss = 0.0
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for anchor, positive, negative, labels in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            labels = labels.to(device)
            
            # Forward pass
            anchor_emb, anchor_logits = model(anchor)
            positive_emb, _ = model(positive)
            negative_emb, _ = model(negative)
            
            # Losses
            triplet_loss = triplet_criterion(anchor_emb, positive_emb, negative_emb)
            bce_loss = bce_criterion(anchor_logits, labels)
            total_loss = triplet_loss + 0.5 * bce_loss
            
            # Metrics
            running_triplet_loss += triplet_loss.item() * anchor.size(0)
            running_bce_loss += bce_loss.item() * anchor.size(0)
            running_total_loss += total_loss.item() * anchor.size(0)
            
            # Store embeddings for metric calculation
            all_embeddings.append(anchor_emb.cpu())
            all_labels.append(labels.cpu())
            
            pbar.set_postfix({
                "triplet": f"{triplet_loss.item():.3f}",
                "bce": f"{bce_loss.item():.3f}"
            })
    
    epoch_triplet_loss = running_triplet_loss / len(dataloader.dataset)
    epoch_bce_loss = running_bce_loss / len(dataloader.dataset)
    epoch_total_loss = running_total_loss / len(dataloader.dataset)
    
    # Calculate embedding quality metric (average distance to same class)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate average intra-class distance
    intra_class_dist = 0.0
    count = 0
    unique_labels = torch.unique(all_labels)
    for label in unique_labels:
        mask = all_labels == label
        if mask.sum() > 1:
            class_embeddings = all_embeddings[mask]
            # Calculate pairwise distances within class
            dists = torch.cdist(class_embeddings, class_embeddings, p=2)
            # Take upper triangle (exclude diagonal)
            intra_class_dist += dists.triu(diagonal=1).sum().item()
            count += dists.triu(diagonal=1).count_nonzero().item()
    
    avg_intra_class_dist = intra_class_dist / count if count > 0 else 0.0
    
    return epoch_triplet_loss, epoch_bce_loss, epoch_total_loss, avg_intra_class_dist


# -------------------------
# Main Training Loop
# -------------------------

def main():
    """Main training function."""
    
    print("\n" + "="*60)
    print("TRIPLET LOSS TRAINING - WHALE IDENTIFICATION")
    print("="*60 + "\n")
    
    # Initialize W&B
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config={
            "architecture": "SEResNet50 + Triplet Loss",
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "img_size": f"{IMG_HEIGHT}x{IMG_SIZE}",
            "embedding_dim": EMBEDDING_DIM,
            "margin": MARGIN,
            "optimizer": "Adam",
            "backbone": "seresnet50",
            "loss": "triplet + bce",
            "data_source": "preprocessed_no_new_whale",
        },
        name="triplet_loss_seresnet50"
    )
    
    # Prepare data
    train_df, val_df = prepare_data()
    
    # Create datasets
    train_dataset = TripletDataset(
        train_df,
        TRAIN_DIR,
        transform=get_transforms(train=True),
        is_training=True
    )
    
    val_dataset = TripletDataset(
        val_df,
        TRAIN_DIR,
        transform=get_transforms(train=False),
        is_training=True
    )
    
    print(f"\nNumber of classes: {train_dataset.num_classes}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    model = TripletNet(num_classes=train_dataset.num_classes).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Loss functions
    triplet_criterion = TripletLoss(margin=MARGIN)
    bce_criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60 + "\n")
    
    best_val_loss = float('inf')
    best_intra_class_dist = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_triplet_loss, train_bce_loss, train_total_loss = train_epoch(
            model, train_loader, triplet_criterion, bce_criterion, optimizer, DEVICE
        )
        
        # Validate
        val_triplet_loss, val_bce_loss, val_total_loss, avg_intra_class_dist = validate(
            model, val_loader, triplet_criterion, bce_criterion, DEVICE
        )
        
        # Learning rate scheduling
        scheduler.step(val_total_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"\nTrain - Triplet: {train_triplet_loss:.4f} | BCE: {train_bce_loss:.4f} | Total: {train_total_loss:.4f}")
        print(f"Val   - Triplet: {val_triplet_loss:.4f} | BCE: {val_bce_loss:.4f} | Total: {val_total_loss:.4f}")
        print(f"Val   - Avg Intra-Class Distance: {avg_intra_class_dist:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Log to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train/triplet_loss": train_triplet_loss,
            "train/bce_loss": train_bce_loss,
            "train/total_loss": train_total_loss,
            "val/triplet_loss": val_triplet_loss,
            "val/bce_loss": val_bce_loss,
            "val/total_loss": val_total_loss,
            "val/intra_class_distance": avg_intra_class_dist,
            "learning_rate": current_lr,
        })
        
        # Save best model (based on validation loss)
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_intra_class_dist = avg_intra_class_dist
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total_loss,
                'intra_class_dist': avg_intra_class_dist,
                'label_to_idx': train_dataset.label_to_idx,
            }, OUTPUT_DIR / 'best_model.pth')
            print(f"✓ Saved best model (Val Loss: {val_total_loss:.4f}, Intra-Class Dist: {avg_intra_class_dist:.4f})")
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nBest Validation Loss: {best_val_loss:.4f}")
    print(f"Best Intra-Class Distance: {best_intra_class_dist:.4f}")
    print(f"\nModel saved to: {OUTPUT_DIR / 'best_model.pth'}")
    
    # Log final summary to W&B
    wandb.summary["best_val_loss"] = best_val_loss
    wandb.summary["best_intra_class_distance"] = best_intra_class_dist
    
    wandb.finish()


if __name__ == "__main__":
    main()
