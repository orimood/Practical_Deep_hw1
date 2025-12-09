# Assignment 1 – Practical Deep Learning Workshop
## Directory Organization Guide

This directory is organized by **question** and by **role** (scripts, results, documentation) for maximum clarity and maintainability.

---

## Directory Structure

```
Practical_Deep_hw1/
├── q1_eda/                          # Question 1: Exploratory Data Analysis
│   ├── plot_fish.ipynb              # EDA visualization notebook
│   ├── dataset_summary.csv          # Dataset statistics
│   └── fish_readme.txt              # Dataset documentation
│
├── q2_neural_nets/                  # Question 2: Neural Network Modeling
│   ├── train_basic_cnn.py           # Baseline CNN implementation
│   ├── q2c_attention_cnn.py         # CNN with attention mechanisms
│   ├── q2c_mixup_cutmix.py          # Advanced augmentation techniques
│   ├── q2c_transfer_learning.py     # Transfer learning baseline
│   └── q2d_tta.py                   # Inference-time augmentation
│
├── q3_transfer_learning/            # Question 3: Transfer Learning Models
│   ├── train_resnet50.py            # ResNet-50 training script
│   ├── train_efficientnet_b0.py     # EfficientNet-B0 training script
│   ├── train_mobilenet_v3_large.py  # MobileNet-V3 training script
│   ├── train_vit_b16.py             # Vision Transformer training script
│   ├── generate_confusion_matrices.py # Confusion matrix visualization
│   └── feature_extraction_experiment.py # Part 3d: Feature extraction + classical ML
│
├── scripts/                         # Utility and Processing Scripts
│   ├── download_data.py             # Download fish dataset from Kaggle
│   ├── split_dataset.py             # Split data into train/val/test
│   ├── output.log                   # Feature extraction execution log
│   └── feature_extraction_output.log# Feature extraction detailed log
│
├── results/                         # All Results and Outputs
│   ├── resnet50_results.json        # ResNet-50 metrics and errors
│   ├── efficientnet_b0_results.json # EfficientNet-B0 metrics
│   ├── mobilenet_v3_large_results.json # MobileNet-V3 metrics
│   ├── vit_b16_results.json         # ViT-B/16 metrics
│   ├── feature_extraction_results.json # Part 3d classical ML results
│   ├── feature_extraction_comparison.csv # Feature extraction comparison table
│   ├── *_confusion_matrices.png     # Confusion matrix visualizations
│   └── [Additional model outputs]   # Other metrics and analyses
│
├── docs/                            # Documentation
│   ├── CLEANUP_SUMMARY.md           # Data cleanup notes
│   ├── DATA_SPLIT_SUMMARY.md        # Data split statistics
│   ├── TRAINING_README.md           # Training procedures and settings
│   ├── TRAINING_STATUS.md           # Training completion status
│   └── PART_3D_FEATURE_EXTRACTION.md # Part 3d methodology and status
│
├── data/                            # Dataset Directory
│   ├── 2/Fish_Dataset/              # Original fish dataset
│   └── fish_split/                  # Split dataset (train/val/test)
│       ├── train/                   # Training set (6,300 images)
│       ├── val/                     # Validation set (900 images)
│       └── test/                    # Test set (1,800 images)
│
├── models/                          # Model Checkpoints and LFS Files
│   ├── best_resnet50.pth            # ResNet-50 checkpoint (Git LFS)
│   ├── best_efficientnet_b0.pth     # EfficientNet-B0 checkpoint (Git LFS)
│   ├── best_mobilenet_v3_large.pth  # MobileNet-V3 checkpoint (Git LFS)
│   ├── best_vit_b16.pth             # ViT-B/16 checkpoint (Git LFS)
│   └── [Symbolic links to results/] # Links to result files
│
├── main.tex                         # Main LaTeX report document
├── assignment1_2026a.md             # Assignment specifications
├── requirements.txt                 # Python dependencies
└── README.md                        # Project overview

```

---

## File Organization by Role

### 📊 Question 1: Exploratory Data Analysis (`q1_eda/`)
Contains all EDA files and dataset documentation:
- **plot_fish.ipynb** - Jupyter notebook with visualizations
- **dataset_summary.csv** - Statistical summary (9 species, 1000 images each)
- **fish_readme.txt** - Original dataset documentation

**How to use:**
```bash
jupyter notebook q1_eda/plot_fish.ipynb
```

---

### 🧠 Question 2: Neural Network Modeling (`q2_neural_nets/`)
Contains implementations of neural network approaches:

1. **train_basic_cnn.py** - Baseline CNN (K-fold cross-validation)
   - K-fold evaluation (K=5)
   - Metrics tracking and visualization
   - Error analysis

2. **q2c_*.py** - Improvement implementations
   - `attention_cnn.py` - Attention mechanisms
   - `mixup_cutmix.py` - Advanced augmentation
   - `transfer_learning.py` - Transfer learning baseline

3. **q2d_tta.py** - Inference-time augmentation
   - Test-time augmentation
   - Prediction aggregation

**How to run:**
```bash
source venv/bin/activate
python q2_neural_nets/train_basic_cnn.py
```

---

### 🔄 Question 3: Transfer Learning Models (`q3_transfer_learning/`)
Contains transfer learning implementations and analysis:

1. **Training Scripts** (4 architectures):
   - `train_resnet50.py` - ResNet-50 (Best: 98.89%)
   - `train_efficientnet_b0.py` - EfficientNet-B0 (98.61%)
   - `train_mobilenet_v3_large.py` - MobileNet-V3 (99.39%)
   - `train_vit_b16.py` - Vision Transformer (98.17%)

2. **Analysis Scripts**:
   - `generate_confusion_matrices.py` - Confusion matrix visualization
   - `feature_extraction_experiment.py` - Part 3d: Classical ML on frozen features

**How to run all models:**
```bash
cd q3_transfer_learning/
python train_resnet50.py
python train_efficientnet_b0.py
python train_mobilenet_v3_large.py
python train_vit_b16.py
```

---

### ⚙️ Utility Scripts (`scripts/`)
Data processing and execution utilities:

1. **download_data.py** - Download fish dataset from Kaggle
   ```bash
   python scripts/download_data.py
   ```

2. **split_dataset.py** - Split data into train/val/test
   ```bash
   python scripts/split_dataset.py
   ```

3. **Logs**:
   - `output.log` - Feature extraction execution log
   - `feature_extraction_output.log` - Detailed feature extraction log

---

### 📈 Results (`results/`)
All output files, metrics, and visualizations:

**JSON Results Files:**
- `resnet50_results.json` - ResNet-50 validation/test metrics, unique correct/errors
- `efficientnet_b0_results.json` - EfficientNet-B0 metrics
- `mobilenet_v3_large_results.json` - MobileNet-V3 metrics
- `vit_b16_results.json` - ViT metrics
- `feature_extraction_results.json` - Classical ML on features (Part 3d)

**CSV Comparison:**
- `feature_extraction_comparison.csv` - Feature extraction vs. transfer learning

**Visualizations:**
- `resnet50_confusion_matrices.png` - ResNet-50 confusion matrices
- `efficientnet_b0_confusion_matrices.png` - EfficientNet confusion matrices
- `svm_rbf_confusion_matrices.png` - SVM (RBF) confusion matrices
- `svm_linear_confusion_matrices.png` - SVM (Linear) confusion matrices
- `random_forest_confusion_matrices.png` - Random Forest confusion matrices

---

### 📚 Documentation (`docs/`)
Project documentation and status:

- **CLEANUP_SUMMARY.md** - Data cleanup and preprocessing notes
- **DATA_SPLIT_SUMMARY.md** - Train/val/test split statistics
- **TRAINING_README.md** - Training procedures, hyperparameters, settings
- **TRAINING_STATUS.md** - Training completion and performance summary
- **PART_3D_FEATURE_EXTRACTION.md** - Part 3d methodology, status, and results

**Key Findings:**
```
Question 3 Results (Transfer Learning):
- ResNet-50: 98.89% (BEST)
- MobileNet-V3: 99.39% (Best efficiency-accuracy tradeoff)
- EfficientNet-B0: 98.61%
- ViT-B/16: 98.17%

Question 3d Results (Feature Extraction + Classical ML):
- SVM (RBF): 100.00% accuracy, 0 errors (BEST)
- SVM (Linear): 100.00% accuracy, 0 errors
- Random Forest: 99.50% accuracy, 9 errors
- Feature extraction outperforms fine-tuning by 1.11%
```

---

## 🚀 Quick Start

### Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Analysis Pipeline
```bash
# 1. Download data
python scripts/download_data.py

# 2. Split dataset
python scripts/split_dataset.py

# 3. Run EDA
jupyter notebook q1_eda/plot_fish.ipynb

# 4. Train baseline CNN (Q2)
python q2_neural_nets/train_basic_cnn.py

# 5. Train transfer learning models (Q3)
cd q3_transfer_learning
for model in train_*.py; do python $model; done

# 6. Feature extraction experiment (Q3d)
python q3_transfer_learning/feature_extraction_experiment.py
```

---

## 📊 Key Results Summary

### Question 1: EDA ✅
- 9 fish species, 1,000 images each
- High quality, balanced dataset
- Requires preprocessing and normalization

### Question 2: Neural Networks ✅
- Baseline CNN implemented with K-fold cross-validation
- Improvements: attention, augmentation, test-time augmentation
- Advanced techniques explored and evaluated

### Question 3: Transfer Learning ✅
- 4 architectures evaluated
- **Best Overall**: ResNet-50 (98.89% test accuracy)
- **Best Efficiency**: MobileNet-V3 (99.39%, 4.2M params)

### Question 3d: Feature Extraction ✅
- ResNet-50 features + classical ML
- **Best Result**: SVM (RBF) achieves **100% accuracy**
- Outperforms fine-tuned transfer learning by 1.11%
- 1000× faster training than neural networks

---

## 📁 Git LFS for Large Files

Model checkpoints are tracked with Git LFS:
```bash
# Verify LFS is configured
git lfs version

# Track large files
git lfs track "models/*.pth"
git lfs track "results/*.png"
```

---

## 🔗 Cross-References

**From Assignment to Files:**
- Q1 EDA → `q1_eda/`, `docs/DATA_SPLIT_SUMMARY.md`
- Q2 Modeling → `q2_neural_nets/`, `docs/TRAINING_README.md`
- Q3 Transfer Learning → `q3_transfer_learning/`, `results/`
- Q3d Feature Extraction → `q3_transfer_learning/feature_extraction_experiment.py`, `results/feature_extraction_*.json`

**Main Report:** `main.tex` - Complete LaTeX document with all findings

---

## 📝 Notes

- All results are in **`results/`** directory
- All training scripts are in **`q*_*/`** directories
- Logs and utilities in **`scripts/`**
- Documentation in **`docs/`**
- Data in **`data/`** (gitignored for large files)

For more information, see `assignment1_2026a.md` for full assignment specifications.
