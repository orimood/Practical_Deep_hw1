"""
ResNet50 Transfer Learning - Fish Classification

- Simplified training (less augmentation)
- 5-fold cross validation
- Logs metrics to W&B
- Saves per-sample predictions

Run from prt_3 directory:
    python train_resnet50.py
"""

import os
import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import wandb
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet50

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
plt.ioff()

# -------------------------
# Configuration
# -------------------------

SEED = 42
NUM_FOLDS = 5
NUM_EPOCHS = 15  # Fewer epochs
BATCH_SIZE = 32
LEARNING_RATE = 5e-5  # Lower learning rate
NUM_WORKERS = 0
IMAGE_SIZE = 224
PATIENCE = 3

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'='*70}")
print(f"RESNET50 - FISH CLASSIFICATION")
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
    """Dataset that loads RGB images only."""

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

def load_resnet50(num_classes: int) -> nn.Module:
    """Load ResNet50 with new classification head."""
    model = resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def freeze_backbone(model: nn.Module, freeze: bool = True):
    """Freeze or unfreeze backbone parameters."""
    for param in model.fc.parameters():
        param.requires_grad = True
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = not freeze


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

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model: nn.Module, dataloader: DataLoader, criterion) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, List[str]]:
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
    """Save detailed per-sample predictions to CSV."""
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
            "true_label_prob": float(prob[label]),
        })
        
        for class_idx, class_name in enumerate(class_names):
            predictions_list[-1][f"prob_{class_name}"] = float(prob[class_idx])
    
    df = pd.DataFrame(predictions_list)
    df.to_csv(save_path, index=False)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar_kws={"label": "Count"})
    plt.ylabel("True label", fontsize=12)
    plt.xlabel("Predicted label", fontsize=12)
    plt.title("Confusion Matrix – ResNet50 OOF Predictions")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# -------------------------
# Main K-Fold training
# -------------------------

def main():
    project_root = Path(__file__).resolve().parent.parent
    dataset_root = project_root / "Data" / "2" / "Fish_Dataset" / "Fish_Dataset"
    plots_dir = project_root / "plots" / "ResNet50"
    results_dir = project_root / "results" / "ResNet50"

    # Minimal augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
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

    # Initialize W&B
    wandb.init(
        project="fish_classification_transfer",
        entity="orisin-ben-gurion-university-of-the-negev",
        name="ResNet50_simplified",
        config={
            "model": "ResNet50",
            "image_size": IMAGE_SIZE,
            "num_folds": NUM_FOLDS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "patience": PATIENCE,
            "seed": SEED,
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

        model = load_resnet50(num_classes).to(device)
        num_params = count_parameters(model)
        print(f"Model parameters: {num_params:,}")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=False)

        best_val_acc = 0.0
        best_state_dict = None
        epochs_without_improvement = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            if epoch == 8:
                print(f"Unfreezing backbone at epoch {epoch}")
                freeze_backbone(model, freeze=False)

            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            train_loss_eval, train_preds, _, train_labels, _ = evaluate(model, train_loader, criterion)
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average="macro")
            
            val_loss, val_preds, _, val_labels, _ = evaluate(model, val_loader, criterion)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average="macro")

            print(
                f"Fold {fold_idx} | Epoch {epoch:02d} | "
                f"TrL={train_loss:.4f} TrAcc={train_acc:.4f} | "
                f"VaL={val_loss:.4f} VaAcc={val_acc:.4f}"
            )

            wandb.log({
                f"fold_{fold_idx}/train_loss": train_loss,
                f"fold_{fold_idx}/train_accuracy": train_acc,
                f"fold_{fold_idx}/val_loss": val_loss,
                f"fold_{fold_idx}/val_accuracy": val_acc,
                "epoch": epoch,
            })

            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        base_dataset.transform = val_transform
        train_loader_eval = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        train_loss, train_preds, train_probs, train_labels_eval, _ = evaluate(model, train_loader_eval, criterion)
        train_acc = accuracy_score(train_labels_eval, train_preds)

        val_loss, val_preds, val_probs, val_labels, val_image_paths = evaluate(model, val_loader, criterion)
        fold_acc = accuracy_score(val_labels, val_preds)
        fold_macro_f1 = f1_score(val_labels, val_preds, average="macro")

        print(f"Fold {fold_idx} final val accuracy: {fold_acc:.4f}")

        fold_metrics.append({
            "fold": fold_idx,
            "model": "ResNet50",
            "num_parameters": num_params,
            "val_accuracy": fold_acc,
            "val_macro_f1": fold_macro_f1,
            "val_loss": float(val_loss),
            "train_accuracy": float(train_acc),
            "train_loss": float(train_loss),
        })

        oof_preds[val_idx] = val_preds
        oof_probs[val_idx] = val_probs
        for i, path in enumerate(val_image_paths):
            all_oof_image_paths[val_idx[i]] = path

        predictions_file = results_dir / f"fold_{fold_idx}_predictions.csv"
        save_per_sample_predictions(val_preds, val_labels, val_probs, val_image_paths, base_dataset.class_names, predictions_file, fold_idx, "ResNet50")

        cm = confusion_matrix(val_labels, val_preds)
        cm_path = plots_dir / f"fold_{fold_idx}_confusion_matrix.png"
        plot_confusion_matrix(cm, base_dataset.class_names, cm_path)

    # -------------------------
    # Aggregate results
    # -------------------------

    metrics_df = pd.DataFrame(fold_metrics)
    print(f"\nResNet50 – Per-fold metrics:")
    print(metrics_df.to_string(index=False))

    mean_acc = metrics_df["val_accuracy"].mean()
    std_acc = metrics_df["val_accuracy"].std()

    agg_acc = accuracy_score(all_labels, oof_preds)
    agg_f1 = f1_score(all_labels, oof_preds, average="macro")

    print(f"\nResNet50 – OOF aggregated metrics:")
    print(f"  Accuracy: {agg_acc:.4f}")
    print(f"  Macro F1: {agg_f1:.4f}")

    metrics_out = results_dir / "fold_metrics_summary.csv"
    metrics_df.to_csv(metrics_out, index=False)

    with open(results_dir / "oof_summary.txt", "w") as fh:
        fh.write(f"Model: ResNet50\n")
        fh.write(f"Number of parameters: {metrics_df['num_parameters'].iloc[0]:,}\n")
        fh.write(f"Mean val accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n")
        fh.write(f"OOF accuracy: {agg_acc:.4f}\n")
        fh.write(f"OOF macro_f1: {agg_f1:.4f}\n")

    oof_predictions_file = results_dir / "oof_predictions.csv"
    save_per_sample_predictions(oof_preds, all_labels, oof_probs, all_oof_image_paths, base_dataset.class_names, oof_predictions_file, 0, "ResNet50")

    cm = confusion_matrix(all_labels, oof_preds)
    cm_path = plots_dir / "confusion_matrix_oof.png"
    plot_confusion_matrix(cm, base_dataset.class_names, cm_path)

    with open(results_dir / "classification_report.txt", "w") as fh:
        fh.write(classification_report(all_labels, oof_preds, target_names=base_dataset.class_names, digits=4))

    wandb.log({
        "mean_oof_accuracy": agg_acc,
        "mean_oof_macro_f1": agg_f1,
        "confusion_matrix": wandb.Image(str(cm_path)),
    })

    wandb.finish()

    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
