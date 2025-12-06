# Basic CNN Training - Fish Classification

## Status: Training Started ✓

### Training Configuration
- **Model**: BasicCNN (27M parameters)
- **Device**: CUDA (GPU)
- **Dataset Split**:
  - Training: 6,300 images
  - Validation: 900 images  
  - Test: 1,800 images
- **Total Classes**: 9 fish species
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Epochs**: 50
- **Optimizer**: Adam with weight decay (1e-4)
- **LR Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)

### Data Augmentation
Training images undergo:
- Resize to 224x224
- Random horizontal flip (50%)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine transforms
- ImageNet normalization

### WandB Tracking
**Project URL**: https://wandb.ai/orisin-ben-gurion-university-of-the-negev/fish-classification-hw1

The training is being tracked with:
- Training/validation loss and accuracy per epoch
- Learning rate changes
- Model gradients and parameters
- Best model checkpoints

### Model Architecture
```
BasicCNN(
  4 Conv Blocks:
    - Block 1: 2x[Conv2d(32) + BN + ReLU] + MaxPool + Dropout
    - Block 2: 2x[Conv2d(64) + BN + ReLU] + MaxPool + Dropout  
    - Block 3: 2x[Conv2d(128) + BN + ReLU] + MaxPool + Dropout
    - Block 4: 2x[Conv2d(256) + BN + ReLU] + MaxPool + Dropout
  
  3 FC Layers:
    - FC1: 256*14*14 -> 512 (+ BN + ReLU + Dropout)
    - FC2: 512 -> 256 (+ BN + ReLU + Dropout)
    - FC3: 256 -> 9 (output)
)
```

### Expected Results
- **Target Accuracy**: 70-85% (typical for CNN from scratch)
- **Baseline**: 11.1% (random guessing, 9 classes)

### Output Files
- Best model: `models/best_basic_cnn.pth`
- WandB logs: `wandb/run-*`
- Training progress: Real-time in terminal with tqdm

### Fish Species (Classes)
1. Black Sea Sprat
2. Gilt-Head Bream
3. Hourse Mackerel
4. Red Mullet
5. Red Sea Bream
6. Sea Bass
7. Shrimp
8. Striped Red Mullet
9. Trout

### Training Features
✓ GPU acceleration (CUDA)
✓ Batch normalization for stability
✓ Dropout for regularization
✓ Learning rate scheduling
✓ Best model checkpointing
✓ Progress bars with tqdm
✓ WandB experiment tracking

---

**Note**: Training will take approximately 30-60 minutes depending on GPU. Monitor progress at the WandB URL above.
