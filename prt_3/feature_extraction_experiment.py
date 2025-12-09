"""
Feature Extraction Experiment - Part 3d
Assignment 1, Question 3 - Feature Extraction Comparison

Key Features:
- Extract features from ResNet-50 (remove last layer)
- Use classical ML algorithms (SVM, Random Forest) on extracted features
- Compare performance to fine-tuned transfer learning results
- Report runtime, loss, metrics, and parameter settings
"""

import os
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class FishDataset(Dataset):
    """Dataset class for fish images - loads from split directories"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Fish species (class names) - MUST match training scripts
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
        
        # Create class to index mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        
        # Load image paths and labels
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist")
                continue
            
            for img_path in class_dir.glob('*.jpg'):
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


class FeatureExtractor(nn.Module):
    """ResNet-50 with last layer removed for feature extraction"""
    
    def __init__(self, pretrained=True):
        super(FeatureExtractor, self).__init__()
        resnet50 = models.resnet50(pretrained=pretrained)
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten to (batch_size, 2048)
        return x


def extract_features(model, dataloader, dataset_name=""):
    """Extract features from all samples in the dataloader"""
    model.eval()
    features_list = []
    labels_list = []
    
    print(f"Extracting features from {dataset_name}...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Extracting {dataset_name}"):
            images = images.to(device)
            batch_features = model(images)
            features_list.append(batch_features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    
    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)
    
    print(f"Extracted {len(features)} samples with {features.shape[1]} features")
    return features, labels


def train_and_evaluate_classifier(clf, train_features, train_labels, 
                                   val_features, val_labels,
                                   test_features, test_labels,
                                   class_names, model_name=""):
    """Train classical ML classifier and evaluate on validation and test sets"""
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    clf.fit(train_features, train_labels)
    train_time = time.time() - start_time
    
    # Validation predictions
    print("Making validation predictions...")
    val_pred = clf.predict(val_features)
    val_accuracy = accuracy_score(val_labels, val_pred)
    val_precision = precision_score(val_labels, val_pred, average='weighted', zero_division=0)
    val_recall = recall_score(val_labels, val_pred, average='weighted', zero_division=0)
    val_f1 = f1_score(val_labels, val_pred, average='weighted', zero_division=0)
    
    # Test predictions
    print("Making test predictions...")
    test_pred = clf.predict(test_features)
    test_accuracy = accuracy_score(test_labels, test_pred)
    test_precision = precision_score(test_labels, test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(test_labels, test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(test_labels, test_pred, average='weighted', zero_division=0)
    
    # Calculate unique correct and errors on test set
    unique_correct = np.sum(test_pred == test_labels)
    unique_errors = np.sum(test_pred != test_labels)
    
    results = {
        'model': model_name,
        'training_time': train_time,
        'num_parameters': _count_parameters(clf),
        'validation_accuracy': val_accuracy,
        'validation_precision': val_precision,
        'validation_recall': val_recall,
        'validation_f1': val_f1,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'unique_correct': unique_correct,
        'unique_errors': unique_errors,
        'test_predictions': test_pred,
        'test_labels': test_labels,
        'val_predictions': val_pred,
        'val_labels': val_labels,
    }
    
    # Print results
    print(f"\nResults for {model_name}:")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"\nValidation Metrics:")
    print(f"  Accuracy:  {val_accuracy:.4f}")
    print(f"  Precision: {val_precision:.4f}")
    print(f"  Recall:    {val_recall:.4f}")
    print(f"  F1-Score:  {val_f1:.4f}")
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    print(f"\nTest Results:")
    print(f"  Unique Correct: {unique_correct}")
    print(f"  Unique Errors:  {unique_errors}")
    
    return results


def _count_parameters(clf):
    """Estimate parameter count for sklearn classifier"""
    try:
        if hasattr(clf, 'n_estimators'):  # RandomForest
            return clf.n_estimators * 100  # Rough estimate
        elif hasattr(clf, 'coef_'):  # SVM
            return clf.coef_.size + (clf.intercept_.size if hasattr(clf, 'intercept_') else 0)
    except:
        pass
    return 0


def create_comparison_summary(results_list, transfer_learning_results):
    """Create a summary table comparing feature extraction vs transfer learning"""
    
    print(f"\n{'='*100}")
    print("COMPARISON SUMMARY: Feature Extraction vs Transfer Learning")
    print(f"{'='*100}\n")
    
    # Create DataFrame for easy comparison
    comparison_data = {
        'Method': [],
        'Model': [],
        'Runtime (s)': [],
        'Test Accuracy': [],
        'Test Precision': [],
        'Test Recall': [],
        'Test F1': [],
        'Unique Correct': [],
        'Unique Errors': []
    }
    
    # Add feature extraction results
    for result in results_list:
        comparison_data['Method'].append('Feature Extraction')
        comparison_data['Model'].append(result['model'])
        comparison_data['Runtime (s)'].append(f"{result['training_time']:.2f}")
        comparison_data['Test Accuracy'].append(f"{result['test_accuracy']:.4f}")
        comparison_data['Test Precision'].append(f"{result['test_precision']:.4f}")
        comparison_data['Test Recall'].append(f"{result['test_recall']:.4f}")
        comparison_data['Test F1'].append(f"{result['test_f1']:.4f}")
        comparison_data['Unique Correct'].append(result['unique_correct'])
        comparison_data['Unique Errors'].append(result['unique_errors'])
    
    # Add transfer learning results for comparison
    if transfer_learning_results:
        for tl_result in transfer_learning_results:
            comparison_data['Method'].append('Transfer Learning')
            comparison_data['Model'].append(tl_result['model'])
            comparison_data['Runtime (s)'].append(f"{tl_result.get('training_time', 'N/A')}")
            comparison_data['Test Accuracy'].append(f"{tl_result.get('test_accuracy', 0):.4f}")
            comparison_data['Test Precision'].append(f"{tl_result.get('test_precision', 0):.4f}")
            comparison_data['Test Recall'].append(f"{tl_result.get('test_recall', 0):.4f}")
            comparison_data['Test F1'].append(f"{tl_result.get('test_f1', 0):.4f}")
            comparison_data['Unique Correct'].append(tl_result.get('unique_correct', 'N/A'))
            comparison_data['Unique Errors'].append(tl_result.get('unique_errors', 'N/A'))
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    print()
    
    return comparison_df


def plot_confusion_matrices(results, class_names, output_dir="models"):
    """Plot confusion matrices for all classifiers"""
    
    for result in results:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Validation confusion matrix
        val_cm = confusion_matrix(result['val_labels'], result['val_predictions'])
        sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=class_names, yticklabels=class_names)
        axes[0].set_title(f"{result['model']} - Validation Confusion Matrix")
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # Test confusion matrix
        test_cm = confusion_matrix(result['test_labels'], result['test_predictions'])
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                    xticklabels=class_names, yticklabels=class_names)
        axes[1].set_title(f"{result['model']} - Test Confusion Matrix")
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        filename = f"{output_dir}/{result['model'].replace(' ', '_').lower()}_confusion_matrices.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix plot to {filename}")
        plt.close()


def main():
    # Configuration
    data_split_dir = "data/fish_split"
    model_checkpoint = "models/best_resnet50.pth"
    output_dir = "models"
    
    # Verify data exists
    if not Path(data_split_dir).exists():
        print(f"Error: {data_split_dir} not found")
        print("Please ensure the fish data is split into train/val/test directories")
        return
    
    # Verify model checkpoint exists
    if not Path(model_checkpoint).exists():
        print(f"Error: {model_checkpoint} not found")
        print("Please train ResNet-50 first using train_resnet50.py")
        return
    
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = FishDataset(f"{data_split_dir}/train", transform=transform)
    val_dataset = FishDataset(f"{data_split_dir}/val", transform=transform)
    test_dataset = FishDataset(f"{data_split_dir}/test", transform=transform)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load feature extractor
    print("\nLoading feature extractor...")
    feature_extractor = FeatureExtractor(pretrained=True).to(device)
    
    # Load trained weights (optional - for better features)
    try:
        checkpoint = torch.load(model_checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Adapt state dict if needed (remove fc layer)
            state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        else:
            state_dict = checkpoint
        
        # Load the feature extraction part
        resnet_state = {k.replace('features.', ''): v for k, v in state_dict.items() 
                       if 'features' in k}
        if resnet_state:
            feature_extractor.features.load_state_dict(resnet_state, strict=False)
            print("Loaded pretrained ResNet-50 weights")
    except Exception as e:
        print(f"Could not load checkpoint ({e}), using ImageNet pretrained weights")
    
    # Extract features
    print("\nExtracting features...")
    train_features, train_labels = extract_features(feature_extractor, train_loader, "training set")
    val_features, val_labels = extract_features(feature_extractor, val_loader, "validation set")
    test_features, test_labels = extract_features(feature_extractor, test_loader, "test set")
    
    print(f"\nFeature extraction complete!")
    print(f"Feature dimension: {train_features.shape[1]}")
    
    # Train classifiers
    results = []
    
    # 1. Support Vector Machine (SVM)
    print("\n" + "="*60)
    print("Training Support Vector Machine (SVM)...")
    svm_clf = SVC(kernel='rbf', C=10, gamma='scale')
    svm_result = train_and_evaluate_classifier(
        svm_clf, train_features, train_labels,
        val_features, val_labels,
        test_features, test_labels,
        train_dataset.class_names,
        model_name="SVM (RBF)"
    )
    results.append(svm_result)
    
    # 2. Random Forest
    print("\n" + "="*60)
    print("Training Random Forest...")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_result = train_and_evaluate_classifier(
        rf_clf, train_features, train_labels,
        val_features, val_labels,
        test_features, test_labels,
        train_dataset.class_names,
        model_name="Random Forest"
    )
    results.append(rf_result)
    
    # 3. Linear SVM
    print("\n" + "="*60)
    print("Training Linear SVM...")
    linear_svm_clf = SVC(kernel='linear', C=1.0)
    linear_svm_result = train_and_evaluate_classifier(
        linear_svm_clf, train_features, train_labels,
        val_features, val_labels,
        test_features, test_labels,
        train_dataset.class_names,
        model_name="SVM (Linear)"
    )
    results.append(linear_svm_result)
    
    # Load transfer learning results for comparison
    transfer_learning_results = []
    try:
        with open('models/resnet50_results.json', 'r') as f:
            tl_data = json.load(f)
            transfer_learning_results.append({
                'model': 'ResNet-50 (Transfer Learning)',
                'test_accuracy': tl_data.get('test_accuracy', 0),
                'test_precision': tl_data.get('test_precision', 0),
                'test_recall': tl_data.get('test_recall', 0),
                'test_f1': tl_data.get('test_f1', 0),
                'unique_correct': tl_data.get('unique_correct', 'N/A'),
                'unique_errors': tl_data.get('unique_errors', 'N/A'),
            })
    except FileNotFoundError:
        print("Warning: Transfer learning results not found for comparison")
    
    # Create comparison summary
    comparison_df = create_comparison_summary(results, transfer_learning_results)
    
    # Plot confusion matrices
    print("\nPlotting confusion matrices...")
    plot_confusion_matrices(results, train_dataset.class_names, output_dir)
    
    # Save results
    print("\nSaving results...")
    results_to_save = {
        'feature_extraction_results': [
            {
                'model': r['model'],
                'training_time': r['training_time'],
                'validation_accuracy': float(r['validation_accuracy']),
                'validation_f1': float(r['validation_f1']),
                'test_accuracy': float(r['test_accuracy']),
                'test_precision': float(r['test_precision']),
                'test_recall': float(r['test_recall']),
                'test_f1': float(r['test_f1']),
                'unique_correct': int(r['unique_correct']),
                'unique_errors': int(r['unique_errors']),
                'feature_dimension': train_features.shape[1],
                'training_algorithm': 'sklearn',
            }
            for r in results
        ],
        'comparison_summary': comparison_df.to_dict()
    }
    
    with open(f'{output_dir}/feature_extraction_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Save comparison table
    comparison_df.to_csv(f'{output_dir}/feature_extraction_comparison.csv', index=False)
    
    print(f"\nResults saved to {output_dir}/feature_extraction_results.json")
    print(f"Comparison table saved to {output_dir}/feature_extraction_comparison.csv")
    
    # Print final summary
    print("\n" + "="*100)
    print("FEATURE EXTRACTION EXPERIMENT COMPLETE")
    print("="*100)
    print("\nKey Findings:")
    best_fe_model = max(results, key=lambda x: x['test_accuracy'])
    print(f"  Best Feature Extraction Model: {best_fe_model['model']}")
    print(f"  Test Accuracy: {best_fe_model['test_accuracy']:.4f}")
    print(f"\nParameter Settings:")
    print(f"  Feature Extractor: ResNet-50 (pretrained on ImageNet)")
    print(f"  Feature Dimension: {train_features.shape[1]}")
    print(f"  Training Set Size: {len(train_features)}")
    print(f"  Number of Classes: {len(train_dataset.class_names)}")
    print(f"\nProcessing Changes:")
    print(f"  - Removed last FC layer from ResNet-50")
    print(f"  - Extracted 2048-dimensional features")
    print(f"  - Applied classical ML algorithms on fixed features")
    print(f"  - No fine-tuning (features remain fixed from ImageNet pretraining)")


if __name__ == "__main__":
    main()
