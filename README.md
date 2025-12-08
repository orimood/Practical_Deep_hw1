# Fish Classification - Deep Learning Project

Fish species classification using deep learning with multiple approaches (baseline CNN, transfer learning, advanced augmentation, and test-time augmentation).

## 📁 Project Files

### Core Training & Data

| File | Purpose |
|------|---------|
| **train_basic_cnn.py** | Main baseline training script. Trains a basic 4-layer CNN with 5-fold stratified cross-validation, caching support, and comprehensive results/visualizations. Outputs trained models, confusion matrices, and conclusions to `results/basic_cnn_results/`. |
| **Download_data.py** | Downloads the fish dataset from Kaggle Hub and copies it to the local `Data/` directory. Run once to set up the dataset. |

### Q2c: Model Improvements (3 Approaches)

| File | Purpose |
|------|---------|
| **q2c_transfer_learning.py** | ResNet50 transfer learning approach. Uses a pretrained ResNet50 backbone with frozen early layers, fine-tunes the last blocks with differential learning rates (head: 1e-3, backbone: 1e-4). Trains for 5 epochs and saves results to `results/q2c_transfer_learning/`. Expected improvement: +10-15% accuracy. |
| **q2c_mixup_cutmix.py** | Advanced augmentation approach using MixUp and CutMix. Alternates between mixing images and mixing patches during training to reduce overfitting. Uses BasicCNN architecture, trains for 5 epochs. Saves results to `results/q2c_mixup_cutmix/`. Expected improvement: +1-3% accuracy. |
| **q2c_attention_cnn.py** | Attention mechanism approach using Squeeze-and-Excitation (SE) blocks. Adds channel attention to the BasicCNN to help the model focus on discriminative features. Lightweight (~2% parameter overhead), trains for 5 epochs. Saves results to `results/q2c_attention_cnn/`. Expected improvement: +1-3% accuracy. |

### Q2d: Inference Optimization

| File | Purpose |
|------|---------|
| **q2d_tta.py** | Test-Time Augmentation (TTA) evaluation. Loads the 5 trained fold models from baseline training, applies multiple augmentations per test image, and averages predictions for improved robustness. Saves metrics and comparison with baseline to `results/q2d_tta/summary.txt`. |

### Utilities & Notebooks

| File | Purpose |
|------|---------|
| **plot_fish.ipynb** | Jupyter notebook for dataset exploration and visualization. Shows dataset statistics, class distribution, sample images, and supports multi-extension image discovery with flexible folder structures. |

---

## 🚀 Quick Start

### 1. Setup Dataset
```bash
python Download_data.py
```

### 2. Train Baseline Model
```bash
python train_basic_cnn.py              # First run (trains all 5 folds)
python train_basic_cnn.py --retrain    # Force retrain if needed
```

### 3. Run Model Improvements (Q2c)
```bash
python q2c_transfer_learning.py    # Transfer learning approach
python q2c_mixup_cutmix.py         # Advanced augmentation approach
python q2c_attention_cnn.py        # Attention mechanism approach
```

### 4. Evaluate with Test-Time Augmentation (Q2d)
```bash
python q2d_tta.py
```

---

## 📊 Results

All results are automatically saved to the `results/` directory:

- **`results/basic_cnn_results/`** - Baseline CNN (5-fold CV)
  - `fold_1/` to `fold_5/` - Per-fold models, confusion matrices, metrics
  - `CONCLUSIONS.txt` - Summary and analysis
  - `test_results_summary.txt` - Ensemble test metrics

- **`results/q2c_transfer_learning/`** - Transfer learning results
- **`results/q2c_mixup_cutmix/`** - Augmentation results
- **`results/q2c_attention_cnn/`** - Attention results
- **`results/q2d_tta/`** - TTA evaluation metrics

---

## 🔧 Architecture Details

### Baseline CNN (train_basic_cnn.py)
- **Layers:** 4 convolutional blocks with batch normalization
- **Channels:** 3 → 32 → 64 → 128 → 256
- **Dropout:** 0.25 in conv layers, 0.5 in dense layers
- **Parameters:** ~51 million
- **Training:** Adam optimizer (lr=0.001), ReduceLROnPlateau scheduler, 5 epochs per fold

### ResNet50 Transfer Learning (q2c_transfer_learning.py)
- **Backbone:** Pretrained ImageNet weights
- **Fine-tuning:** Last 2 blocks + classifier
- **Learning rates:** Head 1e-3, Backbone 1e-4 (differential)
- **Expected gain:** +10-15% accuracy

### Advanced Augmentation (q2c_mixup_cutmix.py)
- **MixUp:** α=0.4 (probabilistic image blending)
- **CutMix:** α=0.4 (probabilistic region mixing)
- **Expected gain:** +1-3% accuracy

### SE-Block Attention (q2c_attention_cnn.py)
- **Channel attention:** Adaptive pooling + FC layers
- **Reduction ratio:** 16
- **Parameter overhead:** ~2%
- **Expected gain:** +1-3% accuracy

---

## 📈 Dataset

- **Source:** Kaggle - A Large Scale Fish Dataset
- **Classes:** 10 fish species (including newly added Gold Fish)
- **Total images:** ~9,206
- **Train/Test split:** ~80/20 with stratification
- **Image formats:** PNG, JPG, JPEG, BMP, WebP
- **Folder structure:** Supports nested (`Class/Class/`) or single-level (`Class/`) layouts

---

## 📝 Notes

- **Caching:** Baseline training caches fold results to avoid retraining. Delete `results/basic_cnn_results/` to force retraining.
- **GPU:** Automatically uses CUDA if available (RTX 3060 tested)
- **Reproducibility:** All scripts set random seeds for reproducible results
- **Encoding:** All file operations use UTF-8 encoding

---

## 📋 Assignment Tasks

### Task (a): Comprehensive Training with Visualizations ✅
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
