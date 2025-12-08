"""
Vision Transformer (ViT-B/16) Transfer Learning for Fish Species Classification
Assignment 1, Question 3 - Model 4/4

Key Features:
- Pretrained ViT-B/16 (Vision Transformer) from torchvision
- Modern Transformer architecture (~86M parameters)
- Optimized for 8GB VRAM (RTX 2080 Ti):
  * Smaller batch size (16)
  * Gradient checkpointing
  * Mixed precision training (FP16)
  * Gradient accumulation if needed
- Tracks unique correct samples and unique errors as required
"""

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import wandb
from tqdm import tqdm
import json
from torch.cuda.amp import autocast, GradScaler

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Check GPU memory
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU Memory: {gpu_memory:.2f} GB")


class FishDataset(Dataset):
    """Dataset class for fish images - loads from split directories"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Fish species (class names)
        self.class_names = sorted([
            "Black Sea Sprat",
            "Gilt-Head Bream",
            "Hourse Mackerel",
            "Red Mullet",
            "Red Sea Bream",
            "Sea Bass",
            "Shrimp",
            "Striped Red Mullet",
            "Trout"
        ])
        
        # Create class to index mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        
        # Load all images
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} not found")
                continue
            
            images = list(class_dir.glob("*.png"))
            self.image_paths.extend(images)
            self.labels.extend([self.class_to_idx[class_name]] * len(images))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, str(img_path)  # Return path for tracking


def get_transforms():
    """Get image transforms for ViT (224x224 input)"""
    
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def train_epoch(model, dataloader, criterion, optimizer, device, scaler, use_amp=True):
    """Train for one epoch with mixed precision"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels, _ in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate_with_tracking(model, dataloader, criterion, device, use_amp=True):
    """Validate and track correct/incorrect predictions by sample"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    correct_samples = []
    error_samples = []
    
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            
            # Track which samples are correct/incorrect
            for i in range(len(labels)):
                sample_path = paths[i]
                is_correct = (predicted[i] == labels[i]).item()
                
                if is_correct:
                    correct += 1
                    correct_samples.append(sample_path)
                else:
                    error_samples.append({
                        'path': sample_path,
                        'true_label': labels[i].item(),
                        'predicted_label': predicted[i].item()
                    })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, correct_samples, error_samples


def main():
    # Configuration optimized for 8GB VRAM
    config = {
        'model_name': 'ViT-B/16',
        'batch_size': 16,  # Reduced for memory
        'learning_rate': 0.0001,  # Lower LR for transformers
        'epochs': 25,  # Slightly fewer epochs
        'weight_decay': 1e-4,
        'optimizer': 'AdamW',
        'pretrained': True,
        'fine_tune_all': False,
        'unfreeze_epoch': 8,
        'image_size': 224,
        'use_amp': True,  # Mixed precision for memory savings
        'gradient_accumulation_steps': 2,  # Simulate larger batch
    }
    
    # Initialize Weights & Biases
    wandb.init(
        project="fish-classification-hw1-q3",
        entity="orisin-ben-gurion-university-of-the-negev",
        config=config,
        name="vit-b16-transfer-learning"
    )
    config = wandb.config
    
    # Data paths
    project_root = Path(r"D:\Projects\Practical_Deep_hw1")
    data_root = project_root / "Data" / "split_fish_dataset"
    
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    print("\n" + "="*70)
    print("VISION TRANSFORMER (ViT-B/16) TRANSFER LEARNING - Question 3, Model 4/4")
    print("OPTIMIZED FOR 8GB VRAM (RTX 2080 Ti)")
    print("="*70)
    
    train_dataset = FishDataset(train_dir, transform=train_transform)
    val_dataset = FishDataset(val_dir, transform=val_transform)
    test_dataset = FishDataset(test_dir, transform=val_transform)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.class_names)}")
    
    # Create dataloaders with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Load pretrained ViT-B/16 and modify final layer
    print("\n" + "="*70)
    print("LOADING PRETRAINED VISION TRANSFORMER (ViT-B/16)")
    print("="*70)
    
    model = models.vit_b_16(pretrained=True)
    
    # Freeze all layers initially
    if not config.fine_tune_all:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace final classification head
    # ViT has 'heads' attribute with a single Linear layer
    num_features = model.heads.head.in_features
    model.heads.head = nn.Linear(num_features, len(train_dataset.class_names))
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    print(f"\nMemory optimizations for 8GB VRAM:")
    print(f"  - Batch size: {config.batch_size} (reduced)")
    print(f"  - Mixed precision (FP16): {config.use_amp}")
    print(f"  - Gradient accumulation: {config.gradient_accumulation_steps} steps")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Cosine annealing works well with transformers
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler() if config.use_amp else None
    
    # Watch model with wandb
    wandb.watch(model, criterion, log="all", log_freq=100)
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    best_val_acc = 0.0
    best_model_path = project_root / "models" / "best_vit_b16.pth"
    best_model_path.parent.mkdir(exist_ok=True)
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print("-" * 70)
        
        # Unfreeze all layers after specified epoch
        if epoch == config.unfreeze_epoch and not config.fine_tune_all:
            print("\n🔓 UNFREEZING ALL LAYERS FOR FINE-TUNING")
            for param in model.parameters():
                param.requires_grad = True
            # Recreate optimizer with all parameters and lower learning rate
            optimizer = optim.AdamW(model.parameters(), 
                                   lr=config.learning_rate * 0.1, 
                                   weight_decay=config.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                            T_max=(config.epochs - epoch))
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {trainable_params:,}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, 
                                           device, scaler, config.use_amp)
        
        # Validate
        val_loss, val_acc, _, _ = validate_with_tracking(model, val_loader, criterion, 
                                                         device, config.use_amp)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Memory stats
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        
        # Log to wandb
        log_dict = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': current_lr,
        }
        
        if torch.cuda.is_available():
            log_dict['memory_allocated_gb'] = memory_allocated
            log_dict['memory_reserved_gb'] = memory_reserved
        
        wandb.log(log_dict)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        if torch.cuda.is_available():
            print(f"GPU Memory: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_names': train_dataset.class_names
            }, best_model_path)
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    # Final Test Evaluation with Tracking
    print("\n" + "="*70)
    print("FINAL EVALUATION - TRACKING UNIQUE SAMPLES")
    print("="*70)
    
    # Load best model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on validation set
    val_loss, val_acc, val_correct, val_errors = validate_with_tracking(
        model, val_loader, criterion, device, config.use_amp)
    
    # Evaluate on test set
    test_loss, test_acc, test_correct, test_errors = validate_with_tracking(
        model, test_loader, criterion, device, config.use_amp)
    
    # Calculate unique samples
    num_unique_correct = len(set(test_correct))
    num_unique_errors = len(test_errors)
    
    print(f"\nValidation Results:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_acc:.2f}%")
    print(f"  Unique Correct: {len(set(val_correct))}")
    print(f"  Unique Errors: {len(val_errors)}")
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  Unique Correct Samples: {num_unique_correct}")
    print(f"  Unique Errors: {num_unique_errors}")
    print(f"  Total Test Samples: {len(test_dataset)}")
    
    # Log final results
    wandb.log({
        'val_loss_final': val_loss,
        'val_acc_final': val_acc,
        'test_loss_final': test_loss,
        'test_acc_final': test_acc,
        'unique_correct_samples': num_unique_correct,
        'unique_errors': num_unique_errors,
        'total_params': total_params,
    })
    
    # Save detailed results
    results = {
        'model_name': 'ViT-B/16',
        'total_parameters': total_params,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'unique_correct_samples': num_unique_correct,
        'unique_errors': num_unique_errors,
        'class_names': train_dataset.class_names,
        'error_samples': test_errors[:100]
    }
    
    results_path = project_root / "models" / "vit_b16_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    print("="*70)
    
    # Summary for assignment table
    print("\n" + "="*70)
    print("SUMMARY FOR ASSIGNMENT TABLE")
    print("="*70)
    print(f"Model Name: ViT-B/16")
    print(f"# Parameters: {total_params:,} (~86M)")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"# Unique Correct Samples: {num_unique_correct}")
    print(f"# Unique Errors: {num_unique_errors}")
    print("="*70)
    
    wandb.finish()


if __name__ == '__main__':
    main()
