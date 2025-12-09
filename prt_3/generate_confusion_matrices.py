"""
Generate Confusion Matrices from Saved Model Results
Creates beautiful confusion matrix visualizations using seaborn
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class FishDataset(Dataset):
    """Dataset class for fish images"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
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
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
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


def generate_confusion_matrix(model_name, model_path, model_loader_fn):
    """Generate and save confusion matrix for a model"""
    
    print(f"\n{'='*70}")
    print(f"Generating Confusion Matrix: {model_name}")
    print(f"{'='*70}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path)
    class_names = checkpoint['class_names']
    
    # Create model
    model = model_loader_fn()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Prepare test data
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dir = Path(r"D:\Projects\Practical_Deep_hw1\Data\split_fish_dataset\test")
    test_dataset = FishDataset(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Generate predictions
    all_preds = []
    all_labels = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate accuracy
    accuracy = 100 * np.trace(cm) / np.sum(cm)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'},
                linewidths=0.5,
                linecolor='gray',
                ax=ax)
    
    # Styling
    ax.set_title(f'{model_name} - Confusion Matrix\nTest Accuracy: {accuracy:.2f}%', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    
    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(r"D:\Projects\Practical_Deep_hw1\prt_3\plots") / f"{model_name.lower().replace(' ', '_').replace('-', '_')}_confusion_matrix.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Confusion matrix saved to {output_path}")
    print(f"  Test Accuracy: {accuracy:.2f}%")
    print(f"  Total Errors: {np.sum(cm) - np.trace(cm):.0f}")
    
    return cm, accuracy


def main():
    project_root = Path(r"D:\Projects\Practical_Deep_hw1")
    models_dir = project_root / "models"
    
    # Model configurations
    model_configs = [
        {
            'name': 'ResNet-50',
            'path': models_dir / 'best_resnet50.pth',
            'loader': lambda: models.resnet50(pretrained=False)
        },
        {
            'name': 'EfficientNet-B0',
            'path': models_dir / 'best_efficientnet_b0.pth',
            'loader': lambda: models.efficientnet_b0(pretrained=False)
        },
        {
            'name': 'MobileNet-V3-Large',
            'path': models_dir / 'best_mobilenet_v3_large.pth',
            'loader': lambda: models.mobilenet_v3_large(pretrained=False)
        },
        {
            'name': 'ViT-B-16',
            'path': models_dir / 'best_vit_b16.pth',
            'loader': lambda: models.vit_b_16(pretrained=False)
        }
    ]
    
    print("\n" + "="*70)
    print("CONFUSION MATRIX GENERATION FOR ALL MODELS")
    print("="*70)
    
    results = {}
    
    for config in model_configs:
        if not config['path'].exists():
            print(f"\n⚠ Model not found: {config['path']}")
            continue
        
        try:
            # Modify model architecture for fish classification
            model = config['loader']()
            
            # Adjust final layer based on model type
            if 'ResNet' in config['name']:
                num_features = model.fc.in_features
                model.fc = nn.Linear(num_features, 9)
            elif 'EfficientNet' in config['name']:
                num_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(num_features, 9)
            elif 'MobileNet' in config['name']:
                num_features = model.classifier[3].in_features
                model.classifier[3] = nn.Linear(num_features, 9)
            elif 'ViT' in config['name']:
                num_features = model.heads.head.in_features
                model.heads.head = nn.Linear(num_features, 9)
            
            # Generate confusion matrix
            cm, acc = generate_confusion_matrix(
                config['name'],
                config['path'],
                lambda: model
            )
            
            results[config['name']] = {
                'confusion_matrix': cm,
                'accuracy': acc
            }
            
        except Exception as e:
            print(f"✗ Error processing {config['name']}: {str(e)}")
            continue
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, result in results.items():
        print(f"{name:25s} - Accuracy: {result['accuracy']:.2f}%")
    
    print("\n✓ All confusion matrices generated successfully!")
    print(f"   Saved to: {project_root / 'prt_3' / 'plots'}")


if __name__ == '__main__':
    main()
