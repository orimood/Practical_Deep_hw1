"""
Baseline fish classifier for Assignment 1 – section 2.a

- Uses RGB images only (no masks)
- Stratified 5-fold cross validation
- Small CNN built from components similar to class walkthroughs
- Prints per-fold metrics table
- Plots + saves confusion matrix of out-of-fold predictions

Run from project root:
    python train_baseline_fish.py
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from torchvision import transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Configuration
# -------------------------

SEED = 42
NUM_FOLDS = 5
NUM_EPOCHS = 15           # you can increase if training is fast enough
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_WORKERS = 4           # set to 0 on Windows if you get issues
IMAGE_SIZE = 224          # resize shorter side to this
PATIENCE = 3              # early stopping patience (in epochs)

# Fix randomness for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# -------------------------
# Dataset
# -------------------------

class FishRGBDataset(Dataset):
    """
    Dataset that loads RGB images only, without masks.
    Assumes directory structure:

    Data/2/Fish_Dataset/Fish_Dataset/
        Black Sea Sprat/
            Black Sea Sprat/*.png          <-- RGB
            Black Sea Sprat GT/*.png       <-- masks (ignored here)
        Gilt-Head Bream/
            Gilt-Head Bream/*.png
            Gilt-Head Bream GT/*.png
        ...

    Each class folder has a subfolder with the same name containing RGB images.
    """

    def __init__(self, root_dir: Path, transform=None):
        """
        :param root_dir: Path to Fish_Dataset (the one that contains the 9 classes)
        :param transform: torchvision transforms to apply to PIL image
        """
        self.root_dir = root_dir
        self.transform = transform

        self.class_names = sorted(
            [
                d.name
                for d in root_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
        )

        # Map class name to integer label
        self.class_to_idx: Dict[str, int] = {
            cls_name: idx for idx, cls_name in enumerate(self.class_names)
        }

        # Build list of (image_path, label)
        self.samples: List[Tuple[Path, int]] = []
        for cls_name in self.class_names:
            rgb_folder = root_dir / cls_name / cls_name  # e.g. "Black Sea Sprat/Black Sea Sprat"
            if not rgb_folder.is_dir():
                raise RuntimeError(f"Expected RGB folder {rgb_folder} not found")

            for img_path in sorted(rgb_folder.glob("*.png")):
                label = self.class_to_idx[cls_name]
                self.samples.append((img_path, label))

        print(f"Found {len(self.samples)} images across {len(self.class_names)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# -------------------------
# Model
# -------------------------

class SimpleCNN(nn.Module):
    """
    Small CNN using components similar to class walkthroughs:
    [Conv -> ReLU -> MaxPool] x 3 -> Global AvgPool -> FC -> ReLU -> FC(num_classes)
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28
        )

        # Global average pooling to make FC independent of image size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


# -------------------------
# Training / Evaluation utils
# -------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer,
) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    :return: (loss, preds, probs) over the dataset
    """
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            all_logits.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    logits = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = probs.argmax(axis=1)

    return epoch_loss, preds, probs, labels_np


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix – Out-of-fold predictions")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()


# -------------------------
# Main K-Fold training
# -------------------------

def main():
    project_root = Path(__file__).resolve().parent
    dataset_root = project_root / "Data" / "2" / "Fish_Dataset" / "Fish_Dataset"

    # Transforms: mild augmentations for train, plain resize for val
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )

    # Base dataset (we will swap transform when creating subsets)
    base_dataset = FishRGBDataset(root_dir=dataset_root, transform=None)
    num_classes = len(base_dataset.class_names)

    # Prepare labels array for StratifiedKFold
    all_labels = np.array([label for _, label in base_dataset.samples])
    all_indices = np.arange(len(all_labels))

    skf = StratifiedKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    fold_metrics = []

    # These will store out-of-fold predictions to build a "global test" view
    oof_preds = np.zeros_like(all_labels)
    oof_probs = np.zeros((len(all_labels), num_classes))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels), start=1):
        print(f"\n===== Fold {fold_idx}/{NUM_FOLDS} =====")

        # Create per-fold datasets by changing transform
        train_dataset = Subset(base_dataset, train_idx)
        val_dataset = Subset(base_dataset, val_idx)

        # We need to set transforms on the underlying dataset objects.
        # Easiest: wrap subsets in new Dataset classes that apply transforms.
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

        model = SimpleCNN(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val_acc = 0.0
        best_state_dict = None
        epochs_without_improvement = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_preds, _, val_labels = evaluate(model, val_loader, criterion)

            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average="macro")

            print(
                f"Fold {fold_idx} | Epoch {epoch:02d} "
                f"| train_loss={train_loss:.4f} "
                f"| val_loss={val_loss:.4f} "
                f"| val_acc={val_acc:.4f} "
                f"| val_macro_f1={val_f1:.4f}"
            )

            # Early stopping on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    print("Early stopping triggered.")
                    break

        # Load best weights for this fold
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        # Final evaluation on validation (for this fold)
        val_loss, val_preds, val_probs, val_labels = evaluate(model, val_loader, criterion)

        fold_acc = accuracy_score(val_labels, val_preds)
        fold_macro_f1 = f1_score(val_labels, val_preds, average="macro")

        print(f"Fold {fold_idx} final val accuracy: {fold_acc:.4f}")
        print(f"Fold {fold_idx} final val macro F1: {fold_macro_f1:.4f}")

        fold_metrics.append(
            {
                "fold": fold_idx,
                "val_accuracy": fold_acc,
                "val_macro_f1": fold_macro_f1,
            }
        )

        # Store out-of-fold predictions/probabilities
        oof_preds[val_idx] = val_preds
        oof_probs[val_idx] = val_probs

    # -------------------------
    # Metrics table
    # -------------------------

    metrics_df = pd.DataFrame(fold_metrics)
    print("\nPer-fold validation metrics:")
    print(metrics_df.to_string(index=False))

    print("\nMean metrics over folds:")
    print(metrics_df[["val_accuracy", "val_macro_f1"]].mean())

    # -------------------------
    # Global out-of-fold report & confusion matrix
    # -------------------------

    print("\nClassification report (out-of-fold predictions):")
    print(
        classification_report(
            all_labels,
            oof_preds,
            target_names=base_dataset.class_names,
            digits=4,
        )
    )

    cm = confusion_matrix(all_labels, oof_preds)
    plots_dir = project_root / "plots"
    cm_path = plots_dir / "confusion_matrix_baseline_oof.png"
    plot_confusion_matrix(cm, base_dataset.class_names, cm_path)
    print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
