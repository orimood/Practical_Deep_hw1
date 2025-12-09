"""
Q2c Improvement 1: Transfer Learning (ResNet50)
- Pretrained ResNet50 backbone with frozen early layers
- Fine-tune last 1–2 blocks; replace classifier for 9 classes
- Expected accuracy uplift: ~10-15%

Usage (sketch):
    python q2c_transfer_learning.py

Notes:
- Reuses FishDataset and load_data from train_basic_cnn.py
- Match baseline epochs (5) with lower lr for head (1e-3) and backbone (1e-4)
- Use cosine scheduler or ReduceLROnPlateau
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from train_basic_cnn import (
    FishDataset,
    get_transforms,
    evaluate,
)


def build_model(num_classes: int, freeze_until: str = "layer3"):
    """Create a transfer-learning model based on ResNet50."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Freeze early layers
    freeze = True
    for name, param in model.named_parameters():
        if freeze:
            param.requires_grad = False
        if freeze_until in name:
            freeze = False
    # Replace classifier
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def train_once(model, train_loader, val_loader, device, lr_head=1e-3, lr_backbone=1e-4, epochs=5):
    """Minimal training loop placeholder."""
    # Separate params for differential learning rates
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "fc" in n]
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "fc" not in n]
    optimizer = optim.Adam([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    final_val_loss, final_val_acc = None, None
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)[:2]
        final_val_loss, final_val_acc = val_loss, val_acc
        print(f"Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}")
    return model, final_val_loss, final_val_acc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = Path(__file__).parent.parent
    data_root = project_root / "data" / "fish_split"
    output_dir = project_root / "results" / "q2c_transfer_learning"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data - load from split directories
    train_tfm, val_tfm = get_transforms(augment=True)
    train_ds = FishDataset(data_root / "train", transform=train_tfm)
    val_ds = FishDataset(data_root / "val", transform=val_tfm)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = build_model(num_classes=len(train_ds.class_names))
    # Train (placeholder loop; adjust epochs/hyperparams as needed)
    model, val_loss, val_acc = train_once(model, train_loader, val_loader, device)

    # Persist simple results summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Q2c Transfer Learning (ResNet50)\n")
        f.write(f"Epochs: 5\n")
        f.write(f"Final Val Acc: {val_acc:.2f}%\n")
        f.write(f"Final Val Loss: {val_loss:.4f}\n")

    torch.save(model.state_dict(), output_dir / "resnet50_finetuned.pth")


if __name__ == "__main__":
    main()
