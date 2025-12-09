"""
Basic CNN for Fish Species Classification - Version 3
Using physically separated train/val/test directories to prevent data leakage
"""

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import wandb
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class FishDataset(Dataset):
    """Dataset class for fish images - loads from split directories"""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Path to directory containing species subdirectories
            transform: Image transforms to apply
        """
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
        
        return image, label


class BasicCNN(nn.Module):
    """Basic CNN architecture with regularization"""
    
    def __init__(self, num_classes=9, dropout_rate=0.3):
        super(BasicCNN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=7, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_metric):
        score = -val_metric if self.mode == 'min' else val_metric
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def get_transforms(augment=True):
    """Get image transforms for training and validation"""
    
    if augment:
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Validation/test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def main():
    # Configuration
    config = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 30,
        'weight_decay': 1e-3,
        'dropout_rate': 0.3,
        'augmentation': True,
        'optimizer': 'Adam',
        'architecture': 'BasicCNN',
        'image_size': 224,
        'early_stopping_patience': 10,
        'data_split': 'physical_separation'
    }
    
    # Initialize Weights & Biases
    wandb.init(
        project="fish-classification-hw1",
        entity="orisin-ben-gurion-university-of-the-negev",
        config=config,
        name="basic-cnn-no-leakage"
    )
    config = wandb.config
    
    # Data paths - using physically separated directories
    project_root = Path(r"D:\Projects\Practical_Deep_hw1")
    data_root = project_root / "Data" / "split_fish_dataset"
    
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"
    
    # Verify directories exist
    for dir_path, name in [(train_dir, "Train"), (val_dir, "Val"), (test_dir, "Test")]:
        if not dir_path.exists():
            raise FileNotFoundError(f"{name} directory not found: {dir_path}")
    
    # Get transforms
    train_transform, val_transform = get_transforms(augment=config.augmentation)
    
    # Create datasets from separate directories
    print("\n" + "="*60)
    print("LOADING DATA FROM PHYSICALLY SEPARATED DIRECTORIES")
    print("="*60)
    print(f"Train: {train_dir}")
    print(f"Val:   {val_dir}")
    print(f"Test:  {test_dir}")
    
    train_dataset = FishDataset(train_dir, transform=train_transform)
    val_dataset = FishDataset(val_dir, transform=val_transform)
    test_dataset = FishDataset(test_dir, transform=val_transform)
    
    print(f"\nDataset sizes:")
    print(f"  Training:   {len(train_dataset):5d} images")
    print(f"  Validation: {len(val_dataset):5d} images")
    print(f"  Test:       {len(test_dataset):5d} images")
    print(f"  Total:      {len(train_dataset) + len(val_dataset) + len(test_dataset):5d} images")
    print(f"  Classes:    {len(train_dataset.class_names)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    model = BasicCNN(num_classes=len(train_dataset.class_names), 
                    dropout_rate=config.dropout_rate).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=5)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode='min')
    
    # Watch model with wandb
    wandb.watch(model, criterion, log="all", log_freq=100)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING (No Data Leakage - Physical Separation)")
    print("="*60)
    
    best_val_acc = 0.0
    best_model_path = project_root / "models" / "best_basic_cnn.pth"
    best_model_path.parent.mkdir(exist_ok=True)
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Check early stopping
        early_stopping(val_loss)
        
        # Log to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': current_lr,
            'train_val_gap': train_acc - val_acc
        })
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Train-Val Gap: {train_acc - val_acc:+.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'class_names': train_dataset.class_names
            }, best_model_path)
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
            wandb.save(str(best_model_path))
        
        # Early stopping check
        if early_stopping.early_stop:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            print(f"No improvement in validation loss for {early_stopping.patience} epochs")
            break
    
    # Test evaluation
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION (Completely Separate Test Set)")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Generalization Gap (Val-Test): {best_val_acc - test_acc:+.2f}%")
    
    wandb.log({
        'test_loss': test_loss,
        'test_acc': test_acc,
        'val_test_gap': best_val_acc - test_acc
    })
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Model saved to: {best_model_path}")
    print("\n✓ No data leakage - train/val/test physically separated!")
    print("="*60)
    
    wandb.finish()


if __name__ == '__main__':
    main()
