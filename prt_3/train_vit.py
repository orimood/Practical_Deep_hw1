"""
Vision Transformer (ViT-B/32) Transfer Learning - Fish Classification

- Transformer-based architecture
- 5-fold cross validation
- Logs metrics to W&B

Run from prt_3 directory:
    python train_vit.py
"""

import os
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

warnings.filterwarnings("ignore")

import wandb
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
plt.ioff()

# Try to import timm
try:
    from timm import create_model
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("ERROR: timm not installed. Install with: pip install timm")
    exit(1)

# -------------------------
# Configuration
# -------------------------

SEED = 42
NUM_FOLDS = 5
NUM_EPOCHS = 15
BATCH_SIZE = 16  # Smaller batch for ViT
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
IMAGE_SIZE = 224
PATIENCE = 4

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'='*70}")
print(f"VISION TRANSFORMER (ViT-B/32) - FISH CLASSIFICATION")
print(f"{'='*70}")
print(f"Device: {device}", end="")
if torch.cuda.is_available():
    print(f" | GPU: {torch.cuda.get_device_name(0)}")
else:
    print()
print(f"{'='*70}\n")

torch.backends.cudnn.benchmark = True

# -------------------------
# Dataset
# -------------------------

class FishRGBDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.class_names = sorted(
            [d.name for d in root_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        )

        self.class_to_idx: Dict[str, int] = {
            cls_name: idx for idx, cls_name in enumerate(self.class_names)
        }

        self.samples: List[Tuple[Path, int]] = []
        for cls_name in self.class_names:
            rgb_folder = root_dir / cls_name / cls_name
            if not rgb_folder.is_dir():
                raise RuntimeError(f"Expected RGB folder {rgb_folder} not found")

            for img_path in sorted(rgb_folder.glob("*.png")):
                label = self.class_to_idx[cls_name]
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, str(img_path)


# -------------------------
# Model
# -------------------------

def load_vit(num_classes: int) -> nn.Module:
    model = create_model('vit_base_patch32_224', pretrained=True, num_classes=num_classes)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# -------------------------
# Training / Evaluation
# -------------------------

def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion, optimizer) -> float:
    model.train()
    running_loss = 0.0
    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def evaluate(model: nn.Module, dataloader: DataLoader, criterion):
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_labels = []
    all_image_paths = []

    with torch.no_grad():
        for images, labels, img_paths in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            all_logits.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_image_paths.extend(img_paths)

    epoch_loss = running_loss / len(dataloader.dataset)
    logits = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = probs.argmax(axis=1)

    return epoch_loss, preds, probs, labels_np, all_image_paths


def save_per_sample_predictions(preds, labels, probs, image_paths, class_names, save_path: Path, fold_idx, model_name):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    predictions_list = []
    for i, (pred, label, prob, img_path) in enumerate(zip(preds, labels, probs, image_paths)):
        is_correct = pred == label
        predictions_list.append({
            "image_path": img_path,
            "true_label": class_names[label],
            "predicted_label": class_names[pred],
            "is_correct": is_correct,
            "confidence": float(prob[pred]),
        })
    
    df = pd.DataFrame(predictions_list)
    df.to_csv(save_path, index=False)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True label", fontsize=12)
    plt.xlabel("Predicted label", fontsize=12)
    plt.title("Confusion Matrix – ViT-B/32 OOF")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# -------------------------
# Main
# -------------------------

def main():
    project_root = Path(__file__).resolve().parent.parent
    dataset_root = project_root / "Data" / "2" / "Fish_Dataset" / "Fish_Dataset"
    plots_dir = project_root / "plots" / "ViT-B32"
    results_dir = project_root / "results" / "ViT-B32"

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    base_dataset = FishRGBDataset(root_dir=dataset_root, transform=None)
    num_classes = len(base_dataset.class_names)

    all_labels = np.array([label for _, label in base_dataset.samples])
    all_indices = np.arange(len(all_labels))

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    wandb.init(
        project="fish_classification_transfer",
        entity="orisin-ben-gurion-university-of-the-negev",
        name="ViT-B32_kfold",
        config={
            "model": "ViT-B/32",
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
        },
        reinit=True,
    )

    fold_metrics = []
    oof_preds = np.zeros_like(all_labels)
    oof_probs = np.zeros((len(all_labels), num_classes))
    all_oof_image_paths = [""] * len(all_labels)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels), start=1):
        print(f"\n--- Fold {fold_idx}/{NUM_FOLDS} ---")

        train_dataset = Subset(base_dataset, train_idx)
        val_dataset = Subset(base_dataset, val_idx)

        base_dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        model = load_vit(num_classes).to(device)
        num_params = count_parameters(model)
        print(f"Model parameters: {num_params:,}")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val_acc = 0.0
        best_state_dict = None
        epochs_without_improvement = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            train_loss_eval, train_preds, _, train_labels, _ = evaluate(model, train_loader, criterion)
            train_acc = accuracy_score(train_labels, train_preds)
            
            val_loss, val_preds, _, val_labels, _ = evaluate(model, val_loader, criterion)
            val_acc = accuracy_score(val_labels, val_preds)

            print(f"Fold {fold_idx} | Epoch {epoch:02d} | TrL={train_loss:.4f} TrAcc={train_acc:.4f} | VaL={val_loss:.4f} VaAcc={val_acc:.4f}")

            wandb.log({
                f"fold_{fold_idx}/train_loss": train_loss,
                f"fold_{fold_idx}/train_accuracy": train_acc,
                f"fold_{fold_idx}/val_loss": val_loss,
                f"fold_{fold_idx}/val_accuracy": val_acc,
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        base_dataset.transform = val_transform
        train_loader_eval = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        train_loss, train_preds, train_probs, train_labels_eval, _ = evaluate(model, train_loader_eval, criterion)
        train_acc = accuracy_score(train_labels_eval, train_preds)

        val_loss, val_preds, val_probs, val_labels, val_image_paths = evaluate(model, val_loader, criterion)
        fold_acc = accuracy_score(val_labels, val_preds)

        fold_metrics.append({
            "fold": fold_idx,
            "model": "ViT-B/32",
            "num_parameters": num_params,
            "val_accuracy": fold_acc,
            "val_loss": float(val_loss),
            "train_accuracy": float(train_acc),
        })

        oof_preds[val_idx] = val_preds
        oof_probs[val_idx] = val_probs
        for i, path in enumerate(val_image_paths):
            all_oof_image_paths[val_idx[i]] = path

        predictions_file = results_dir / f"fold_{fold_idx}_predictions.csv"
        save_per_sample_predictions(val_preds, val_labels, val_probs, val_image_paths, base_dataset.class_names, predictions_file, fold_idx, "ViT")

        cm = confusion_matrix(val_labels, val_preds)
        cm_path = plots_dir / f"fold_{fold_idx}_confusion_matrix.png"
        plot_confusion_matrix(cm, base_dataset.class_names, cm_path)

    metrics_df = pd.DataFrame(fold_metrics)
    print(f"\nViT-B/32 – Metrics:\n{metrics_df.to_string(index=False)}")

    agg_acc = accuracy_score(all_labels, oof_preds)
    agg_f1 = f1_score(all_labels, oof_preds, average="macro")

    print(f"\nOOF Accuracy: {agg_acc:.4f}, F1: {agg_f1:.4f}")

    metrics_out = results_dir / "fold_metrics_summary.csv"
    metrics_df.to_csv(metrics_out, index=False)

    oof_predictions_file = results_dir / "oof_predictions.csv"
    save_per_sample_predictions(oof_preds, all_labels, oof_probs, all_oof_image_paths, base_dataset.class_names, oof_predictions_file, 0, "ViT")

    cm = confusion_matrix(all_labels, oof_preds)
    cm_path = plots_dir / "confusion_matrix_oof.png"
    plot_confusion_matrix(cm, base_dataset.class_names, cm_path)

    wandb.finish()
    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
