"""
Transfer Learning Models for Fish Classification – Assignment 1 Section 3

- Tests 4 pretrained architectures: ResNet50, EfficientNet-B4, MobileNetV3-Large, ViT-B/32
- Uses 5-fold cross validation
- Logs metrics to W&B
- Saves per-sample predictions with correct/incorrect status
- Generates confusion matrices and detailed reports

Run from project root:
    python train_transfer_models.py
"""

import os
import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

# Suppress warnings
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
from torchvision.models import resnet50, efficientnet_b4, mobilenet_v3_large

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns

# Suppress matplotlib output
plt.ioff()

# Try to import ViT from timm, fallback if not available
try:
    from timm import create_model
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

# -------------------------
# Configuration
# -------------------------

SEED = 42
NUM_FOLDS = 5
NUM_EPOCHS = 20
BATCH_SIZE = 32  # Increase if GPU has >8GB VRAM
LEARNING_RATE = 1e-4  # Lower for fine-tuning
NUM_WORKERS = 4  # Increase for faster data loading (set to 0 if issues)
IMAGE_SIZE = 224
PATIENCE = 5
UNFREEZE_EPOCH = 10  # Unfreeze backbone after this epoch

# GPU optimization
CUDNN_BENCHMARK = True  # Auto-tune CUDA kernels for speed
MIXED_PRECISION = False  # Set to True for faster training if GPU supports it

# Models to train
MODELS_CONFIG = {
    "ResNet50": {
        "loader": lambda num_classes: load_resnet50(num_classes),
        "input_size": 224,
    },
    "EfficientNet-B4": {
        "loader": lambda num_classes: load_efficientnet_b4(num_classes),
        "input_size": 224,
    },
    "MobileNetV3-Large": {
        "loader": lambda num_classes: load_mobilenetv3(num_classes),
        "input_size": 224,
    },
}

# Add ViT if timm is available
if HAS_TIMM:
    MODELS_CONFIG["ViT-B/32"] = {
        "loader": lambda num_classes: load_vit(num_classes),
        "input_size": 224,
    }

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print startup info only once



# Apply GPU optimization settings
torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
if MIXED_PRECISION and torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler
    SCALER = GradScaler()
else:
    SCALER = None


# -------------------------
# Model Loaders
# -------------------------

def load_resnet50(num_classes: int) -> nn.Module:
    """Load ResNet50 with new classification head."""
    model = resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_efficientnet_b4(num_classes: int) -> nn.Module:
    """Load EfficientNet-B4 with new classification head."""
    model = efficientnet_b4(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_mobilenetv3(num_classes: int) -> nn.Module:
    """Load MobileNetV3-Large with new classification head."""
    model = mobilenet_v3_large(pretrained=True)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def load_vit(num_classes: int) -> nn.Module:
    """Load Vision Transformer from timm with new head."""
    model = create_model('vit_base_patch32_224', pretrained=True, num_classes=num_classes)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


# -------------------------
# Dataset
# -------------------------

class FishRGBDataset(Dataset):
    """Dataset that loads RGB images only."""

    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.class_names = sorted(
            [
                d.name
                for d in root_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
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
        return img, label, str(img_path)  # Include image path for tracking


# -------------------------
# Training / Evaluation utils
# -------------------------

def freeze_backbone(model: nn.Module, freeze: bool = True):
    """Freeze or unfreeze backbone parameters (keep head trainable)."""
    # Identify backbone vs head based on model type
    if hasattr(model, 'fc'):  # ResNet
        for param in model.fc.parameters():
            param.requires_grad = True
        for param in list(model.parameters())[:-1]:
            param.requires_grad = not freeze
    elif hasattr(model, 'classifier'):  # EfficientNet, MobileNet
        for param in model.classifier.parameters():
            param.requires_grad = True
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = not freeze
    elif hasattr(model, 'head'):  # ViT
        for param in model.head.parameters():
            param.requires_grad = True
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = not freeze


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer,
) -> float:
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


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    :return: (loss, preds, probs, labels, image_paths)
    """
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


def save_per_sample_predictions(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    image_paths: List[str],
    class_names: List[str],
    save_path: Path,
    fold_idx: int,
    model_name: str,
):
    """Save detailed per-sample predictions to CSV."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    predictions_list = []
    for i, (pred, label, prob, img_path) in enumerate(
        zip(preds, labels, probs, image_paths)
    ):
        is_correct = pred == label
        predictions_list.append({
            "image_path": img_path,
            "true_label": class_names[label],
            "predicted_label": class_names[pred],
            "is_correct": is_correct,
            "confidence": float(prob[pred]),
            "true_label_prob": float(prob[label]),
        })
        
        # Add probabilities for all classes
        for class_idx, class_name in enumerate(class_names):
            predictions_list[-1][f"prob_{class_name}"] = float(prob[class_idx])
    
    df = pd.DataFrame(predictions_list)
    df.to_csv(save_path, index=False)
    print(f"Saved per-sample predictions to: {save_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.ylabel("True label", fontsize=12)
    plt.xlabel("Predicted label", fontsize=12)
    plt.title("Confusion Matrix – Out-of-fold predictions")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# -------------------------
# Main K-Fold training
# -------------------------

def train_model_kfold(model_name: str, model_loader, num_classes: int, base_dataset: Dataset):
    """Train a single model using K-fold cross validation."""
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    # Initialize W&B for this model
    wandb.init(
        project="fish_classification_transfer",
        entity="orisin-ben-gurion-university-of-the-negev",
        name=f"{model_name}_kfold",
        config={
            "model": model_name,
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

    project_root = Path(__file__).resolve().parent.parent
    plots_dir = project_root / "plots" / model_name
    results_dir = project_root / "results" / model_name

    # Transforms (ImageNet normalization)
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

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
    all_oof_image_paths = [""] * len(all_labels)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels), start=1):
        print(f"\n--- Fold {fold_idx}/{NUM_FOLDS} ---")

        train_dataset = Subset(base_dataset, train_idx)
        val_dataset = Subset(base_dataset, val_idx)

        # Apply transforms
        base_dataset.transform = train_transform
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

        model = model_loader(num_classes).to(device)
        num_params = count_parameters(model)
        print(f"Model parameters: {num_params:,}")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True,
        )

        best_val_acc = 0.0
        best_state_dict = None
        epochs_without_improvement = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            # Unfreeze backbone after certain epoch
            if epoch == UNFREEZE_EPOCH:
                print(f"Unfreezing backbone at epoch {epoch}")
                freeze_backbone(model, freeze=False)
                # Reset optimizer to include all parameters
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)

            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            
            # Get training metrics
            train_loss_eval, train_preds, _, train_labels, _ = evaluate(model, train_loader, criterion)
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average="macro")
            
            # Get validation metrics
            val_loss, val_preds, _, val_labels, _ = evaluate(model, val_loader, criterion)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average="macro")

            print(
                f"Fold {fold_idx} | Epoch {epoch:02d} | "
                f"TrL={train_loss:.4f} TrAcc={train_acc:.4f} TrF1={train_f1:.4f} | "
                f"VaL={val_loss:.4f} VaAcc={val_acc:.4f} VaF1={val_f1:.4f}"
            )

            wandb.log({
                f"fold_{fold_idx}/train_loss": train_loss,
                f"fold_{fold_idx}/train_accuracy": train_acc,
                f"fold_{fold_idx}/train_macro_f1": train_f1,
                f"fold_{fold_idx}/val_loss": val_loss,
                f"fold_{fold_idx}/val_accuracy": val_acc,
                f"fold_{fold_idx}/val_macro_f1": val_f1,
                "epoch": epoch,
            })

            scheduler.step(val_loss)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        # Load best model
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        # Final evaluation
        base_dataset.transform = val_transform
        train_loader_eval = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        train_loss, train_preds, train_probs, train_labels_eval, _ = evaluate(
            model,
            train_loader_eval,
            criterion,
        )
        train_acc = accuracy_score(train_labels_eval, train_preds)
        train_f1 = f1_score(train_labels_eval, train_preds, average="macro")

        val_loss, val_preds, val_probs, val_labels, val_image_paths = evaluate(
            model,
            val_loader,
            criterion,
        )
        fold_acc = accuracy_score(val_labels, val_preds)
        fold_macro_f1 = f1_score(val_labels, val_preds, average="macro")

        print(f"Fold {fold_idx} final val accuracy: {fold_acc:.4f}")

        fold_metrics.append({
            "fold": fold_idx,
            "model": model_name,
            "num_parameters": num_params,
            "val_accuracy": fold_acc,
            "val_macro_f1": fold_macro_f1,
            "val_loss": float(val_loss),
            "train_accuracy": float(train_acc),
            "train_macro_f1": float(train_f1),
            "train_loss": float(train_loss),
        })

        # Store OOF predictions
        oof_preds[val_idx] = val_preds
        oof_probs[val_idx] = val_probs
        for i, path in enumerate(val_image_paths):
            all_oof_image_paths[val_idx[i]] = path

        wandb.log({
            f"fold_{fold_idx}/final_accuracy": fold_acc,
            f"fold_{fold_idx}/final_macro_f1": fold_macro_f1,
        })

        # Save per-sample predictions for this fold
        predictions_file = results_dir / f"fold_{fold_idx}_predictions.csv"
        save_per_sample_predictions(
            val_preds,
            val_labels,
            val_probs,
            val_image_paths,
            base_dataset.class_names,
            predictions_file,
            fold_idx,
            model_name,
        )

        # Save confusion matrix for this fold
        cm = confusion_matrix(val_labels, val_preds)
        cm_path = plots_dir / f"fold_{fold_idx}_confusion_matrix.png"
        plot_confusion_matrix(cm, base_dataset.class_names, cm_path)

    # -------------------------
    # Aggregate results
    # -------------------------

    metrics_df = pd.DataFrame(fold_metrics)
    print(f"\n{model_name} – Per-fold metrics:")
    print(metrics_df.to_string(index=False))

    mean_acc = metrics_df["val_accuracy"].mean()
    mean_f1 = metrics_df["val_macro_f1"].mean()
    std_acc = metrics_df["val_accuracy"].std()

    print(f"\n{model_name} – Mean metrics:")
    print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Macro F1: {mean_f1:.4f}")

    # OOF aggregated metrics
    agg_acc = accuracy_score(all_labels, oof_preds)
    agg_f1 = f1_score(all_labels, oof_preds, average="macro")

    print(f"\n{model_name} – OOF aggregated metrics:")
    print(f"  Accuracy: {agg_acc:.4f}")
    print(f"  Macro F1: {agg_f1:.4f}")

    # Save aggregated metrics
    metrics_out = results_dir / "fold_metrics_summary.csv"
    metrics_df.to_csv(metrics_out, index=False)

    with open(results_dir / "oof_summary.txt", "w") as fh:
        fh.write(f"Model: {model_name}\n")
        fh.write(f"Number of parameters: {metrics_df['num_parameters'].iloc[0]:,}\n")
        fh.write(f"\nTRAINING METRICS (Mean across folds):\n")
        fh.write(f"  Train Accuracy: {metrics_df['train_accuracy'].mean():.4f} ± {metrics_df['train_accuracy'].std():.4f}\n")
        fh.write(f"  Train Macro F1: {metrics_df['train_macro_f1'].mean():.4f} ± {metrics_df['train_macro_f1'].std():.4f}\n")
        fh.write(f"  Train Loss: {metrics_df['train_loss'].mean():.4f} ± {metrics_df['train_loss'].std():.4f}\n")
        fh.write(f"\nVALIDATION METRICS (Mean across folds):\n")
        fh.write(f"  Val Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n")
        fh.write(f"  Val Macro F1: {mean_f1:.4f}\n")
        fh.write(f"  Val Loss: {metrics_df['val_loss'].mean():.4f} ± {metrics_df['val_loss'].std():.4f}\n")
        fh.write(f"\nTEST METRICS (OOF Aggregated):\n")
        fh.write(f"  Test Accuracy: {agg_acc:.4f}\n")
        fh.write(f"  Test Macro F1: {agg_f1:.4f}\n")
        fh.write(f"\nPER-FOLD BREAKDOWN:\n")
        for idx, row in metrics_df.iterrows():
            fh.write(f"\nFold {int(row['fold'])}:\n")
            fh.write(f"  Train - Acc: {row['train_accuracy']:.4f}, F1: {row['train_macro_f1']:.4f}, Loss: {row['train_loss']:.4f}\n")
            fh.write(f"  Val   - Acc: {row['val_accuracy']:.4f}, F1: {row['val_macro_f1']:.4f}, Loss: {row['val_loss']:.4f}\n")

    # Save global OOF predictions
    oof_predictions_file = results_dir / "oof_predictions.csv"
    save_per_sample_predictions(
        oof_preds,
        all_labels,
        oof_probs,
        all_oof_image_paths,
        base_dataset.class_names,
        oof_predictions_file,
        0,
        model_name,
    )

    # Global confusion matrix
    cm = confusion_matrix(all_labels, oof_preds)
    cm_path = plots_dir / "confusion_matrix_oof.png"
    plot_confusion_matrix(cm, base_dataset.class_names, cm_path)

    # Classification report
    with open(results_dir / "classification_report.txt", "w") as fh:
        fh.write(
            classification_report(
                all_labels,
                oof_preds,
                target_names=base_dataset.class_names,
                digits=4,
            )
        )

    wandb.log({
        "mean_oof_accuracy": agg_acc,
        "mean_oof_macro_f1": agg_f1,
        "confusion_matrix": wandb.Image(str(cm_path)),
    })

    wandb.finish()

    return {
        "model": model_name,
        "num_parameters": metrics_df["num_parameters"].iloc[0],
        "train_accuracy": metrics_df["train_accuracy"].mean(),
        "train_accuracy_std": metrics_df["train_accuracy"].std(),
        "train_loss": metrics_df["train_loss"].mean(),
        "val_accuracy": mean_acc,
        "val_accuracy_std": std_acc,
        "val_loss": metrics_df["val_loss"].mean(),
        "test_accuracy": agg_acc,
        "test_macro_f1": agg_f1,
    }


# -------------------------
# Main
# -------------------------

def main():
    project_root = Path(__file__).resolve().parent.parent
    dataset_root = project_root / "Data" / "2" / "Fish_Dataset" / "Fish_Dataset"

    base_dataset = FishRGBDataset(root_dir=dataset_root, transform=None)
    num_classes = len(base_dataset.class_names)

    results_summary = []

    for model_name, config in MODELS_CONFIG.items():
        try:
            result = train_model_kfold(
                model_name=model_name,
                model_loader=config["loader"],
                num_classes=num_classes,
                base_dataset=base_dataset,
            )
            results_summary.append(result)
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # -------------------------
    # Summary comparison table
    # -------------------------

    summary_df = pd.DataFrame(results_summary)
    print("\n" + "="*80)
    print("TRANSFER LEARNING MODELS SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))

    summary_path = project_root / "results" / "transfer_learning_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
