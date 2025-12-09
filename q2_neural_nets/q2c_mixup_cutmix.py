"""
Q2c Improvement 2: Advanced Augmentation (MixUp + CutMix)
- Reduces overfitting and improves robustness
- Typically yields +1-3% accuracy on medium-sized datasets

Usage (sketch):
    python q2c_mixup_cutmix.py

Notes:
- Reuses FishDataset, load_data, evaluate, and BasicCNN from train_basic_cnn.py
- Toggle mixup/cutmix probabilities via config
"""
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from train_basic_cnn import (
    FishDataset,
    get_transforms,
    BasicCNN,
    evaluate,
)


def mixup_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = random.betavariate(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = random.betavariate(alpha, alpha)
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size, device=x.device)

    cut_rat = (1 - lam) ** 0.5
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = random.randint(0, w)
    cy = random.randint(0, h)

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    y2 = min(cy + cut_h // 2, h)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    y_a, y_b = y, y[index]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    return x, y_a, y_b, lam


def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_with_mixups(model, loader, criterion, optimizer, device, alpha_mixup=0.4, alpha_cutmix=0.0):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        use_cutmix = alpha_cutmix > 0 and random.random() < 0.5
        if use_cutmix:
            images, y_a, y_b, lam = cutmix_data(images, labels, alpha_cutmix)
        else:
            images, y_a, y_b, lam = mixup_data(images, labels, alpha_mixup)

        optimizer.zero_grad()
        outputs = model(images)
        loss = mixed_criterion(criterion, outputs, y_a, y_b, lam)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        # Approx accuracy using primary label
        correct += (preds == y_a).sum().item()
        total += labels.size(0)
    return running_loss / total, 100.0 * correct / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = Path(__file__).parent.parent
    data_root = project_root / "data" / "fish_split"

    # Data - load from split directories
    output_dir = project_root / "results" / "q2c_mixup_cutmix"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_tfm, val_tfm = get_transforms(augment=True)
    train_ds = FishDataset(data_root / "train", transform=train_tfm)
    val_ds = FishDataset(data_root / "val", transform=val_tfm)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = BasicCNN(num_classes=len(train_ds.class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Train few epochs with mixup/cutmix
    epochs = 5
    final_val_loss, final_val_acc = None, None
    for epoch in range(epochs):
        train_loss, train_acc = train_with_mixups(model, train_loader, criterion, optimizer, device,
                                                  alpha_mixup=0.4, alpha_cutmix=0.4)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)[:2]
        final_val_loss, final_val_acc = val_loss, val_acc
        print(f"Epoch {epoch+1}/{epochs} | Train Acc (approx): {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}")

    # Persist simple results summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Q2c Advanced Augmentation (MixUp + CutMix)\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Final Val Acc: {final_val_acc:.2f}%\n")
        f.write(f"Final Val Loss: {final_val_loss:.4f}\n")

    torch.save(model.state_dict(), output_dir / "basiccnn_mixup_cutmix.pth")


if __name__ == "__main__":
    main()
