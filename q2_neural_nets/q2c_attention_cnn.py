"""
Q2c Improvement 3: Attention Mechanisms (Squeeze-and-Excitation blocks)
- Adds channel attention to help model focus on discriminative features
- Lightweight (~2% parameter overhead) with potential +1-3% accuracy

Usage (sketch):
    python q2c_attention_cnn.py

Notes:
- Defines SEBasicCNN (BasicCNN + SE blocks) reusing data utilities from train_basic_cnn.py
"""
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from train_basic_cnn import (
    FishDataset,
    get_transforms,
    evaluate,
)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBasicCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.conv1 = self._block(3, 32)
        self.conv2 = self._block(32, 64)
        self.conv3 = self._block(64, 128)
        self.conv4 = self._block(128, 256)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            SEBlock(out_ch),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.classifier(x)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = Path(__file__).parent.parent
    data_root = project_root / "data" / "fish_split"

    # Data - load from split directories
    train_tfm, val_tfm = get_transforms(augment=True)
    train_ds = FishDataset(data_root / "train", transform=train_tfm)
    val_ds = FishDataset(data_root / "val", transform=val_tfm)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = SEBasicCNN(num_classes=len(train_ds.class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)[:2]
        print(f"Epoch {epoch+1}/{epochs} | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
