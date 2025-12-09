# Fish Classification - Basic CNN Training

## Overview
This project trains a basic CNN from scratch to classify 9 species of fish and seafood from RGB images.

## Model Architecture

### BasicCNN
- **Input**: 224x224 RGB images
- **Architecture**:
  - 4 Convolutional blocks (32, 64, 128, 256 filters)
  - Each block: 2x Conv2d + BatchNorm + ReLU + MaxPool + Dropout
  - 3 Fully connected layers (512, 256, 9)
  - Batch normalization and dropout for regularization
- **Parameters**: ~12M trainable parameters

## Dataset
- **Total samples**: ~9,000 images
- **Classes**: 9 fish/seafood species
- **Split**: 70% train, 10% validation, 20% test
- **Image size**: Resized to 224x224
- **Normalization**: ImageNet mean and std

## Training Configuration

```python
config = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50,
    'weight_decay': 1e-4,
    'augmentation': True,
    'optimizer': 'Adam',
    'image_size': 224,
}
```

## Data Augmentation (Training only)
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine transforms (translation, scale)

## Training Features
- ✓ Weights & Biases integration for experiment tracking
- ✓ Learning rate scheduling (ReduceLROnPlateau)
- ✓ Best model checkpointing based on validation accuracy
- ✓ Progress bars with tqdm
- ✓ Gradient and parameter tracking with wandb.watch()

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python train_basic_cnn.py
```

### Monitor training
Visit your Weights & Biases dashboard:
- Project: `fish-classification-hw1`
- Entity: `orisin-ben-gurion-university-of-the-negev`
- URL: https://wandb.ai/orisin-ben-gurion-university-of-the-negev/fish-classification-hw1

## Expected Performance
Based on similar fine-grained classification tasks:
- **Target**: 70-85% accuracy (basic CNN from scratch)
- **Baseline**: Random guessing = 11.1% (9 classes)

## Model Saving
- Best model saved to: `models/best_basic_cnn.pth`
- Checkpoint includes:
  - Model state dict
  - Optimizer state dict
  - Validation accuracy
  - Class names
  - Epoch number

## Files
- `train_basic_cnn.py` - Main training script
- `models/` - Saved model checkpoints
- `plots/` - Visualization outputs from EDA
- `plot_fish.ipynb` - Exploratory data analysis notebook

## Classes
1. Black Sea Sprat
2. Gilt-Head Bream
3. Hourse Mackerel (typo in original dataset)
4. Red Mullet
5. Red Sea Bream
6. Sea Bass
7. Shrimp
8. Striped Red Mullet
9. Trout

## Notes
- Training runs on GPU if available (CUDA), otherwise CPU
- Model uses ImageNet normalization for potential transfer learning compatibility
- Validation-based early stopping via learning rate scheduler
- All random seeds set to 42 for reproducibility
