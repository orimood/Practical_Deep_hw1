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

import wandb

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

class WhaleDataset(Dataset):
    """
    Dataset that loads RGB images from a flat 'train' folder using a CSV file
    with labels.

    Expected:

        root_dir/
            0a0c1df99.jpg
            0a00c7a0f.jpg
            ...

        labels_csv (e.g. Data/train.csv) with columns:
            Image, Id
    """

    def __init__(
        self,
        root_dir: Path,
        labels_csv: Path,
        image_col: str = "Image",
        label_col: str = "Id",
        transform=None,
    ):
        self.root_dir = root_dir
        self.transform = transform

        if not self.root_dir.is_dir():
            raise RuntimeError(f"Dataset root directory does not exist: {self.root_dir}")

        if not labels_csv.is_file():
            raise RuntimeError(f"Labels CSV not found at: {labels_csv}")

        df = pd.read_csv(labels_csv)

        if image_col not in df.columns or label_col not in df.columns:
            raise RuntimeError(
                f"CSV {labels_csv} must contain columns '{image_col}' and '{label_col}'. "
                f"Found columns: {list(df.columns)}"
            )

        # Class names from Id column
        self.class_names = sorted(df[label_col].unique().tolist())

        # Map class name to integer label index
        self.class_to_idx: Dict[str, int] = {
            cls_name: idx for idx, cls_name in enumerate(self.class_names)
        }

        # Build list of (image_path, label_idx)
        self.samples: List[Tuple[Path, int]] = []
        for _, row in df.iterrows():
            img_name = str(row[image_col])
            cls_name = row[label_col]

            img_path = self.root_dir / img_name
            if not img_path.is_file():
                raise RuntimeError(f"Image file listed in CSV not found: {img_path}")

            label_idx = self.class_to_idx[cls_name]
            self.samples.append((img_path, label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Found 0 valid samples in {labels_csv} / {self.root_dir}. "
                "Check that the CSV filenames match the files in the train folder."
            )

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
import wandb

def main():
    # -----------------------------
    # Init W&B
    # -----------------------------
    wandb.login()   # optional, if not logged in
    wandb.init(
        project="whale_classification_kfold",
        entity="orisin-ben-gurion-university-of-the-negev",
        settings=wandb.Settings(_disable_stats=True),
        name="kfold_training",
        config={
            "image_size": IMAGE_SIZE,
            "num_folds": NUM_FOLDS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "patience": PATIENCE,
            "seed": SEED,
        }
    )

    project_root = Path(__file__).resolve().parent
    dataset_root = project_root / "Data" / "train"
    labels_csv = project_root / "Data" / "train.csv"

    base_dataset = WhaleDataset(
        root_dir=dataset_root,
        labels_csv=labels_csv,
        image_col="Image",
        label_col="Id",
        transform=None,
    )


    plots_dir = project_root / "plots"

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    num_classes = len(base_dataset.class_names)

    all_labels = np.array([label for _, label in base_dataset.samples])
    all_indices = np.arange(len(all_labels))

    skf = StratifiedKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    fold_metrics = []
    oof_preds = np.zeros_like(all_labels)
    oof_probs = np.zeros((len(all_labels), num_classes))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels), start=1):
        print(f"\n===== Fold {fold_idx}/{NUM_FOLDS} =====")

        # Track fold separately
        wandb.run.name = f"fold_{fold_idx}"

        # Per-fold history for plotting
        train_losses_history: List[float] = []
        val_losses_history: List[float] = []
        val_acc_history: List[float] = []

        train_dataset = Subset(base_dataset, train_idx)
        val_dataset = Subset(base_dataset, val_idx)

        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        model = SimpleCNN(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val_acc = 0.0
        best_state_dict = None
        epochs_without_improvement = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_preds, _, val_labels = evaluate(model, val_loader, criterion)

            # record histories
            train_losses_history.append(float(train_loss))
            val_losses_history.append(float(val_loss))

            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average="macro")

            val_acc_history.append(float(val_acc))

            print(
                f"Fold {fold_idx} | Epoch {epoch:02d} "
                f"| train_loss={train_loss:.4f} "
                f"| val_loss={val_loss:.4f} "
                f"| val_acc={val_acc:.4f} "
                f"| val_macro_f1={val_f1:.4f}"
            )

            # -----------------------------
            # Log to W&B
            # -----------------------------
            wandb.log({
                f"fold_{fold_idx}/train_loss": train_loss,
                f"fold_{fold_idx}/val_loss": val_loss,
                f"fold_{fold_idx}/val_accuracy": val_acc,
                f"fold_{fold_idx}/val_macro_f1": val_f1,
                "epoch": epoch,
            })

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    print("Early stopping triggered.")
                    break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        # Evaluate on training set for comparison (final model for this fold)
        train_loader_eval = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        train_loss, train_preds, train_probs, train_labels_eval = evaluate(model, train_loader_eval, criterion)
        train_acc = accuracy_score(train_labels_eval, train_preds)
        train_f1 = f1_score(train_labels_eval, train_preds, average="macro")

        # Final fold evaluation
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
                "val_loss": float(val_loss),
                "train_accuracy": float(train_acc),
                "train_macro_f1": float(train_f1),
                "train_loss": float(train_loss),
            }
        )

        # Store OOF predictions
        oof_preds[val_idx] = val_preds
        oof_probs[val_idx] = val_probs

        # Log fold-level aggregated metrics
        wandb.log({
            f"fold_{fold_idx}/final_accuracy": fold_acc,
            f"fold_{fold_idx}/final_macro_f1": fold_macro_f1,
            f"fold_{fold_idx}/train_accuracy": train_acc,
        })

    # -------------------------
    # Metrics table
    # -------------------------

    metrics_df = pd.DataFrame(fold_metrics)
    print("\nPer-fold validation metrics:")
    print(metrics_df.to_string(index=False))

    mean_acc = metrics_df["val_accuracy"].mean()
    mean_f1 = metrics_df["val_macro_f1"].mean()

    print("\nMean metrics over folds:")
    print({"accuracy": mean_acc, "macro_f1": mean_f1})

    # Log aggregated metrics
    wandb.log({
        "mean_accuracy": mean_acc,
        "mean_macro_f1": mean_f1,
    })

    # -------------------------
    # Global out-of-fold report
    # -------------------------

    print("\nClassification report (OOF):")
    print(
        classification_report(
            all_labels,
            oof_preds,
            target_names=base_dataset.class_names,
            digits=4,
        )
    )

    # Compare aggregated OOF metrics to mean of per-fold metrics
    agg_acc = accuracy_score(all_labels, oof_preds)
    agg_f1 = f1_score(all_labels, oof_preds, average="macro")
    print("Aggregated OOF metrics:")
    print({"accuracy": agg_acc, "macro_f1": agg_f1})

    print("\nComparison: mean-per-fold vs aggregated OOF")
    print({"mean_val_accuracy": mean_acc, "mean_val_macro_f1": mean_f1, "oof_accuracy": agg_acc, "oof_macro_f1": agg_f1})

    # Save fold metrics and aggregated metrics
    metrics_out = project_root / "plots" / "fold_metrics_summary.csv"
    metrics_df.to_csv(metrics_out, index=False)
    try:
        with open(project_root / "plots" / "oof_summary.txt", "w") as fh:
            fh.write(f"OOF accuracy: {agg_acc}\n")
            fh.write(f"OOF macro_f1: {agg_f1}\n")
            fh.write(f"Mean val accuracy: {mean_acc}\n")
            fh.write(f"Mean val macro_f1: {mean_f1}\n")
    except Exception:
        pass

    # Log confusion matrix (OOF)
    cm = confusion_matrix(all_labels, oof_preds)

    plots_dir = project_root / "plots"
    cm_path = plots_dir / "confusion_matrix_baseline_oof.png"
    plot_confusion_matrix(cm, base_dataset.class_names, cm_path)
    print(f"Confusion matrix saved to: {cm_path}")

    wandb.log({"confusion_matrix_oof": wandb.Image(str(cm_path))})

    wandb.finish()

if __name__ == "__main__":
    main()
