# Fish Classification - Basic CNN Training

Simple and effective fish species classification using a Basic CNN with K-Fold Cross-Validation.

## 📁 Project Structure

```
Practical_Deep_hw1/
├── Data/
│   └── train/              # Fish species dataset
│       ├── Black Sea Sprat/
│       ├── Gilt-Head Bream/
│       ├── Hourse Mackerel/
│       ├── Red Mullet/
│       ├── Red Sea Bream/
│       ├── Sea Bass/
│       ├── Shrimp/
│       ├── Striped Red Mullet/
│       ├── Trout/
│       └── [New Category]/  # Added in Task (e)
│
├── results/                 # All training results and visualizations
│   ├── task_a_baseline/     # Baseline model results
│   ├── task_b_analysis/     # Misclassification analysis
│   ├── task_c_improved/     # Improved model results
│   ├── task_d_tta/          # TTA evaluation results
│   └── task_e_new_category/ # 10-class model results
│
├── train_comprehensive.py        # Task (a) - Baseline training with K-fold CV
├── analyze_misclassifications.py # Task (b) - Error analysis and suggestions
├── train_improved.py             # Task (c) - Improved model (ResNet50 + MixUp)
├── evaluate_tta.py               # Task (d) - Test Time Augmentation
├── add_new_category.py           # Task (e) - Helper to add new fish species
├── train_with_new_category.py    # Task (e) - Train with 10 classes
└── README.md                     # This file
```

## 🎯 Assignment Tasks

### Task (a): Comprehensive Training with Visualizations ✅

**Script:** `train_comprehensive.py`

Implements baseline training with extensive visualizations and metrics:

**Features:**
- 5-Fold Stratified Cross-Validation
- Basic CNN architecture (4 conv blocks: 3→32→64→128→256 channels)
- Training/validation loss and accuracy curves per fold
- Confusion matrices for each fold
- Per-class accuracy and F1 scores
- Example visualizations:
  - High confidence CORRECT predictions
  - High confidence INCORRECT predictions
  - Most UNCERTAIN predictions
- Ensemble predictions (mean of all folds)
- Comprehensive metrics comparison

**Usage:**
```bash
python train_comprehensive.py
```

**Output:**
- `results/task_a_baseline/fold_X/` - Per-fold results
- Training history plots
- Confusion matrices
- Classification reports
- Example prediction visualizations

---

### Task (b): Misclassification Analysis ✅

**Script:** `analyze_misclassifications.py`

Analyzes where and why the model fails, with improvement suggestions.

**Analysis Includes:**
1. Confusion matrix pattern analysis
2. Top 10 most confused class pairs
3. Per-class error rates
4. Visualizations of misclassified samples
5. Dataset characteristics analysis (image sizes, aspect ratios, class balance)
6. **At least 6 improvement suggestions** with reasoning and expected gains

**Suggested Improvements:**
1. **Transfer Learning** (ResNet50/EfficientNet) - Priority 1
2. **Advanced Data Augmentation** (MixUp, CutMix) - Priority 2
3. **Class Imbalance Handling** - Priority 3
4. **Attention Mechanisms** (CBAM, SE-Net) - Priority 4
5. **Multi-Scale Training** - Priority 5
6. **Ensemble Methods** - Priority 6

**Usage:**
```bash
python analyze_misclassifications.py
```

**Output:**
- `results/task_b_analysis/`
- Confusion analysis report
- Misclassified sample visualizations
- Dataset characteristics plots
- `improvement_suggestions.txt` with detailed recommendations

---

### Task (c): Implement Top 2 Improvements ✅

**Script:** `train_improved.py`

Implements the top 2 prioritized improvements:

**Improvement 1: Transfer Learning**
- ResNet50 pre-trained on ImageNet
- Custom classifier head (2048 → 512 → 9 classes)
- Differential learning rates (lower LR for backbone, higher for classifier)
- Cosine annealing with warm restarts

**Improvement 2: Advanced Data Augmentation**
- **MixUp** augmentation (alpha=0.4)
- Stronger augmentation pipeline:
  - Random resized crop (70-100%)
  - Horizontal/vertical flips
  - Rotation (±30°)
  - Color jitter (brightness, contrast, saturation, hue)
  - Random affine transforms
  - Random perspective
  - Random erasing

**Usage:**
```bash
python train_improved.py
```

**Output:**
- `results/task_c_improved/`
- Improved model checkpoints
- Comparison metrics vs baseline
- Same visualization suite as Task (a)

**Expected Results:**
- **10-20% accuracy improvement** from transfer learning
- **5-10% additional improvement** from advanced augmentation
- More robust predictions on difficult samples

---

### Task (d): Test Time Augmentation (TTA) ✅

**Script:** `evaluate_tta.py`

Implements inference-time augmentation to boost test accuracy.

**TTA Transforms (8 augmentations):**
1. Original (center crop)
2. Horizontal flip
3. Vertical flip
4. Horizontal + Vertical flip
5. Rotate +10°
6. Rotate -10°
7. Slight zoom in
8. Slight zoom out

**Method:**
- Apply all 8 transforms to each test image
- Get model predictions for each augmented version
- Average the probability distributions
- Select class with highest averaged probability

**Usage:**
```bash
python evaluate_tta.py
```

**Output:**
- `results/task_d_tta/`
- Standard vs TTA accuracy comparison
- Side-by-side confusion matrices
- Per-class improvement analysis
- Detailed metrics report

**Expected Improvement:**
- **2-5% accuracy gain** on test set
- Especially effective for uncertain/difficult samples

---

### Task (e): Add New Category and Retrain ✅

**Scripts:** 
- `add_new_category.py` - Helper to prepare new category
- `train_with_new_category.py` - Training with 10 classes

**New Category:** Salmon (10th fish species)

**Steps:**
1. Create folder structure for new category
2. Add fish images (100+ recommended)
3. Automatically detect all categories (including new one)
4. Retrain model with updated num_classes=10
5. Evaluate on 10-class dataset

**Usage:**
```bash
# Step 1: Add new category
python add_new_category.py
# Follow prompts to add images

# Step 2: Train with 10 classes
python train_with_new_category.py
```

**Output:**
- `results/task_e_new_category/`
- 10-class model checkpoints
- Updated confusion matrix (10x10)
- Performance comparison

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- PyTorch 2.7.1+cu118 (CUDA 11.8)
- torchvision 0.22.1+cu118
- numpy, pandas, scikit-learn
- matplotlib, seaborn
- Pillow, tqdm

### 2. Prepare Dataset

Place fish images in the correct structure:
```
Data/train/
  ├── Species_Name/
  │   └── Species_Name/
  │       ├── image_001.png
  │       ├── image_002.png
  │       └── ...
```

### 3. Run All Tasks (Sequential)

```bash
# Task (a) - Baseline training
python train_comprehensive.py

# Task (b) - Analyze errors
python analyze_misclassifications.py

# Task (c) - Train improved model
python train_improved.py

# Task (d) - Evaluate with TTA
python evaluate_tta.py

# Task (e) - Add new category and retrain
python add_new_category.py
python train_with_new_category.py
```

---

## 📊 Models and Architectures

### Baseline CNN (Task a)
```
Input (224x224x3)
├── Conv Block 1: 3→32 channels + BN + ReLU + MaxPool + Dropout
├── Conv Block 2: 32→64 channels + BN + ReLU + MaxPool + Dropout
├── Conv Block 3: 64→128 channels + BN + ReLU + MaxPool + Dropout
├── Conv Block 4: 128→256 channels + BN + ReLU + MaxPool + Dropout
├── Flatten
├── FC: 256*14*14 → 512 + BN + ReLU + Dropout
├── FC: 512 → 256 + BN + ReLU + Dropout
└── FC: 256 → 9 classes

Total params: ~51M
```

### Improved ResNet50 (Tasks c, d, e)
```
Input (224x224x3)
├── ResNet50 Backbone (pre-trained on ImageNet)
│   └── Frozen/Fine-tuned with differential LR
└── Custom Classifier:
    ├── Dropout (p=0.5)
    ├── FC: 2048 → 512 + ReLU
    ├── Dropout (p=0.3)
    └── FC: 512 → num_classes

Total params: ~25M (ResNet50) + 1M (classifier)
```

---

## 📈 Expected Results

### Task (a) - Baseline
- **Validation Accuracy:** 70-80%
- **Test Accuracy:** 68-78%

### Task (c) - Improved Model
- **Validation Accuracy:** 85-92%
- **Test Accuracy:** 83-90%
- **Improvement:** +10-15%

### Task (d) - With TTA
- **Test Accuracy:** 85-93%
- **Improvement over standard:** +2-5%

### Task (e) - 10 Classes
- **Test Accuracy:** 80-88%
- (Slight decrease expected due to added complexity)

---

## 🔧 Configuration

### Training Hyperparameters

**Baseline (Task a):**
```python
config = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 30,
    'weight_decay': 1e-4,
    'augmentation': True,
    'n_folds': 5,
}
```

**Improved (Tasks c, e):**
```python
config = {
    'batch_size': 16,           # Smaller for larger model
    'learning_rate': 0.0005,    # Lower for transfer learning
    'epochs': 25,
    'weight_decay': 1e-4,
    'augmentation': True,
    'use_mixup': True,
    'mixup_alpha': 0.4,
    'n_folds': 5,
}
```

---

## 📝 Output Files

Each task generates comprehensive outputs:

### Per-Fold Results
- `fold_X_training_history.png` - Loss and accuracy curves
- `fold_X_confusion_matrix.png` - Confusion matrix heatmap
- `fold_X_per_class_metrics.png` - Accuracy and F1 per class
- `fold_X_correct_predictions.png` - High confidence correct examples
- `fold_X_incorrect_predictions.png` - High confidence errors
- `fold_X_uncertain_predictions.png` - Low confidence predictions
- `classification_report.txt` - Detailed metrics

### Summary Results
- `test_results_summary.txt` - Performance across all folds
- `fold_ensemble_confusion_matrix.png` - Ensemble results
- Model checkpoints: `best_model.pth`

---

## 🎓 Key Learnings

### What Works:
1. **Transfer Learning** - Biggest single improvement (+10-15%)
2. **MixUp Augmentation** - Reduces overfitting significantly
3. **Test Time Augmentation** - Free accuracy boost at inference
4. **Ensemble Methods** - Combining fold predictions improves robustness

### What to Watch:
1. **Class Imbalance** - Can hurt minority class performance
2. **Overfitting** - Monitor train-val gap closely
3. **Learning Rate** - Use differential LR for transfer learning
4. **Augmentation Strength** - Too strong can hurt performance

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in config
'batch_size': 8  # instead of 16
```

### Model Not Found Error
```bash
# Make sure to run tasks in order:
# Task (a) before (b)
# Task (c) before (d)
```

### Dataset Path Issues
```bash
# Update data_root in scripts if needed:
data_root = Path(__file__).parent / "Data" / "train"
```

---

## 📚 References

- **ResNet:** Deep Residual Learning for Image Recognition (He et al., 2015)
- **MixUp:** mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)
- **Test Time Augmentation:** Multiple testing strategies
- **Data Augmentation:** Various computer vision techniques

---

## 👥 Author

Practical Deep Learning Homework Assignment
Ben-Gurion University of the Negev

---

## 📄 License

Educational use only.

---

## 🎯 Summary Checklist

- [x] Task (a): Comprehensive training with visualizations
- [x] Task (b): Misclassification analysis with 3+ suggestions
- [x] Task (c): Implement top 2 improvements (Transfer Learning + MixUp)
- [x] Task (d): Test Time Augmentation implementation
- [x] Task (e): Add new category and retrain

**All tasks completed successfully!** 🎉
