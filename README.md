# Fish Species Classification - Deep Learning Assignment

## Project Overview
Classification of 9 fish/seafood species using deep learning on RGB images.

## Dataset
- **Total Images**: 9,000 (1,000 per species)
- **Classes**: 9 species (Black Sea Sprat, Gilt-Head Bream, Hourse Mackerel, Red Mullet, Red Sea Bream, Sea Bass, Shrimp, Striped Red Mullet, Trout)
- **Image Size**: 590×445 (resized to 224×224 for training)
- **Split**: Train 70% (6,300) | Val 10% (900) | Test 20% (1,800)
- **Data Source**: Large-Scale Fish Dataset (Ulucan et al., 2020)

## Project Structure
```
Practical_Deep_hw1/
├── train_basic_cnn.py          # Main training script (with physical data separation)
├── split_dataset.py            # Script to split dataset into train/val/test
├── plot_fish.ipynb            # Exploratory data analysis notebook
├── download_data.py           # Dataset download script
├── main.tex                   # LaTeX report
├── Data/
│   ├── split_fish_dataset/    # Physically separated train/val/test
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── 2/Fish_Dataset/        # Original dataset
├── models/                    # Saved model checkpoints
├── plots/                     # Generated visualizations
├── docs/                      # Documentation and summaries
└── wandb/                     # Weights & Biases logs
```

## Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
# Split dataset into train/val/test (one-time operation)
python split_dataset.py
```

### 3. Train Model
```bash
# Train Basic CNN with early stopping
python train_basic_cnn.py
```

### 4. Monitor Training
Visit WandB dashboard: https://wandb.ai/orisin-ben-gurion-university-of-the-negev/fish-classification-hw1

## Model Architecture

**BasicCNN** - Custom CNN with 27M parameters:
- 4 Conv blocks (32→64→128→256 filters)
- Batch normalization + Dropout (0.3)
- 3 FC layers (512→256→9)
- Early stopping (patience=10)

## Training Configuration
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-3)
- **Batch Size**: 32
- **Max Epochs**: 30 (with early stopping)
- **Augmentation**: Random flip, rotation, color jitter, affine
- **Normalization**: ImageNet mean/std
- **Loss**: Cross-entropy

## Key Features
✓ **No Data Leakage** - Physical separation of train/val/test  
✓ **Early Stopping** - Prevents overfitting  
✓ **WandB Tracking** - Complete experiment logging  
✓ **Reproducible** - Fixed random seeds (42)  
✓ **Best Model Saving** - Based on validation accuracy  

## Expected Results
- **Target**: 85-95% accuracy
- **Baseline**: 11.1% (random guessing)

## Files Description

### Scripts
- `train_basic_cnn.py` - Main training with physical data separation
- `split_dataset.py` - One-time dataset splitting
- `download_data.py` - Download dataset from Kaggle

### Notebooks
- `plot_fish.ipynb` - EDA with visualizations

### Documentation (docs/)
- `DATA_SPLIT_SUMMARY.md` - Dataset splitting details
- `TRAINING_README.md` - Training guide
- `TRAINING_STATUS.md` - Training progress

### Reports
- `main.tex` - LaTeX assignment report
- `assignment1_2026a.md` - Assignment instructions

## Output Files
- `models/best_basic_cnn.pth` - Best model checkpoint
- `plots/*.png` - Generated visualizations
- `dataset_summary.csv` - Dataset statistics

## Notes
- GPU highly recommended (training takes ~30-60 min on GPU)
- Original dataset preserved in `Data/2/Fish_Dataset/`
- Split dataset uses ~9GB disk space
- All experiments tracked on Weights & Biases

## Citation
```
O. Ulucan, D. Karakaya, and M. Turkan.
"A Large-Scale Dataset for Fish Segmentation and Classification."
2020 Innovations in Intelligent Systems and Applications Conference (ASYU), IEEE, 2020.
```
