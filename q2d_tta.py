"""
Q2d: Inference-Time Augmentation (TTA) on existing fold models.
- Loads the 5 trained fold models from results/basic_cnn_results/fold_*/best_model.pth
- Applies several test-time augmentations per image and averages predictions
- Averages across folds for final ensemble prediction
- Saves metrics to results/q2d_tta/summary.txt (and optional confusion matrix)

Run:
    ./.venv/Scripts/python.exe q2d_tta.py

Notes:
- Uses same 5 epochs baseline models (already trained) to avoid retraining.
- Baseline metrics are read from results/basic_cnn_results/test_results_summary.txt if present
  to report delta versus prior ensemble.
"""

from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from train_basic_cnn import BasicCNN, load_data, get_transforms


def load_baseline_metrics(summary_path: Path):
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        metrics = {}
        for line in lines:
            if "Ensemble Accuracy" in line:
                metrics["acc"] = float(line.split(":")[-1].strip().strip("%"))
            if "Ensemble F1" in line:
                metrics["f1"] = float(line.split(":")[-1].strip())
            if "Ensemble Precision" in line:
                metrics["precision"] = float(line.split(":")[-1].strip())
            if "Ensemble Recall" in line:
                metrics["recall"] = float(line.split(":")[-1].strip())
        return metrics if metrics else None
    except Exception:
        return None


def tta_transforms():
    base_tfm, _ = get_transforms(augment=False)
    # Define a few light TTA variants
    tfm_list = [
        base_tfm,
        get_transforms(augment=True)[0],  # mild augment from training
    ]
    return tfm_list


def tta_predict_image(model, image, tfms, device):
    with torch.no_grad():
        probs = []
        for tfm in tfms:
            img_t = tfm(image).unsqueeze(0).to(device)
            logits = model(img_t)
            prob = F.softmax(logits, dim=1)
            probs.append(prob.cpu())
        probs = torch.stack(probs, dim=0).mean(dim=0)  # (1, num_classes)
    return probs.squeeze(0)


def evaluate_tta(models, X_paths, y_true, tfms, device, class_names):
    all_probs = []
    for model in models:
        model.eval()
        fold_probs = []
        for path in X_paths:
            image = Image.open(path).convert("RGB")
            prob = tta_predict_image(model, image, tfms, device)
            fold_probs.append(prob.numpy())
        all_probs.append(np.stack(fold_probs, axis=0))  # (N, C)
    # Average across folds
    avg_probs = np.mean(np.stack(all_probs, axis=0), axis=0)  # (N, C)
    preds = np.argmax(avg_probs, axis=1)
    acc = accuracy_score(y_true, preds) * 100
    f1 = f1_score(y_true, preds, average="weighted")
    precision = precision_score(y_true, preds, average="weighted")
    recall = recall_score(y_true, preds, average="weighted")
    cm = confusion_matrix(y_true, preds, labels=list(range(len(class_names))))
    return {
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }


def load_fold_models(folds_dir: Path, num_classes: int, device):
    models = []
    for fold in range(1, 6):
        model_path = folds_dir / f"fold_{fold}" / "best_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model = BasicCNN(num_classes=num_classes).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)
    return models


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).parent
    data_root = project_root / "Data" / "2" / "Fish_Dataset" / "Fish_Dataset"
    folds_dir = project_root / "results" / "basic_cnn_results"
    output_dir = project_root / "results" / "q2d_tta"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    X_train, y_train, X_test, y_test, class_names = load_data(data_root, test_size=0.2)

    # Models
    models = load_fold_models(folds_dir, num_classes=len(class_names), device=device)

    # TTA transforms
    tfms = tta_transforms()

    # Evaluate with TTA
    tta_metrics = evaluate_tta(models, X_test, y_test, tfms, device, class_names)

    # Load baseline metrics if available
    baseline = load_baseline_metrics(folds_dir / "test_results_summary.txt")

    # Save summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Q2d Inference-Time Augmentation (TTA)\n")
        f.write(f"Device: {device}\n")
        f.write(f"Test Samples: {len(X_test)}\n")
        f.write(f"Transforms used: {len(tfms)} variants\n")
        f.write(f"TTA Accuracy: {tta_metrics['acc']:.2f}%\n")
        f.write(f"TTA F1 (weighted): {tta_metrics['f1']:.4f}\n")
        f.write(f"TTA Precision (weighted): {tta_metrics['precision']:.4f}\n")
        f.write(f"TTA Recall (weighted): {tta_metrics['recall']:.4f}\n")
        if baseline:
            f.write("\nBaseline (ensemble, no TTA) from prior run:\n")
            if "acc" in baseline:
                f.write(f"Baseline Accuracy: {baseline['acc']:.2f}%\n")
                f.write(f"Accuracy Δ: {tta_metrics['acc'] - baseline['acc']:.2f} pts\n")
            if "f1" in baseline:
                f.write(f"Baseline F1: {baseline['f1']:.4f}\n")
                f.write(f"F1 Δ: {tta_metrics['f1'] - baseline['f1']:.4f}\n")
            if "precision" in baseline:
                f.write(f"Baseline Precision: {baseline['precision']:.4f}\n")
                f.write(f"Precision Δ: {tta_metrics['precision'] - baseline['precision']:.4f}\n")
            if "recall" in baseline:
                f.write(f"Baseline Recall: {baseline['recall']:.4f}\n")
                f.write(f"Recall Δ: {tta_metrics['recall'] - baseline['recall']:.4f}\n")

    # Save confusion matrix
    cm_path = output_dir / "confusion_matrix.npy"
    np.save(cm_path, tta_metrics["confusion_matrix"])

    print(f"TTA results saved to {summary_path}")
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
