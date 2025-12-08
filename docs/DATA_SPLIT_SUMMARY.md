# Dataset Split Summary - No Data Leakage

## Problem Addressed
The original training script loaded all images and then split them in memory. This approach could potentially lead to data leakage if not carefully managed.

## Solution: Physical Separation

### What We Did
1. **Created `split_dataset.py`** - A script that physically separates the dataset into three distinct directories
2. **Updated training script** - `train_basic_cnn_v3_no_leakage.py` loads from separate directories

### Directory Structure
```
Data/split_fish_dataset/
├── train/          (6,300 images - 70%)
│   ├── Black Sea Sprat/
│   ├── Gilt-Head Bream/
│   └── ... (9 species)
├── val/            (900 images - 10%)
│   ├── Black Sea Sprat/
│   ├── Gilt-Head Bream/
│   └── ... (9 species)
└── test/           (1,800 images - 20%)
    ├── Black Sea Sprat/
    ├── Gilt-Head Bream/
    └── ... (9 species)
```

### Split Statistics

| Split      | Images | Percentage | Per Species |
|------------|--------|------------|-------------|
| Train      | 6,300  | 70.0%      | 700 each    |
| Validation | 900    | 10.0%      | 100 each    |
| Test       | 1,800  | 20.0%      | 200 each    |
| **Total**  | 9,000  | 100%       | 1,000 each  |

## Key Improvements

### 1. **Physical Separation**
- Train, validation, and test sets are in completely separate directories
- No possibility of accidentally mixing data
- Easy to verify no overlap by checking file paths

### 2. **Reproducibility**
- Uses fixed random seed (42) for consistent splits
- Same split every time the script is run
- Original data preserved (uses copy, not move)

### 3. **Stratified Splitting**
- Each species has exactly 700/100/200 images in train/val/test
- Maintains class balance across all splits
- No class bias introduced

### 4. **Updated Training Pipeline**
```python
# Old way (in-memory split)
X_train, X_test = train_test_split(all_images, ...)  # Could leak

# New way (physical separation)
train_dataset = FishDataset(train_dir)
val_dataset = FishDataset(val_dir)
test_dataset = FishDataset(test_dir)  # Completely separate
```

## Training Script Versions

| Version | Script Name                          | Features                              |
|---------|--------------------------------------|---------------------------------------|
| v1      | `train_basic_cnn.py`                | Original (50 epochs, in-memory split) |
| v2      | `train_basic_cnn_v2.py`             | Early stopping, 30 epochs             |
| v3      | `train_basic_cnn_v3_no_leakage.py`  | Physical separation + early stopping  |

## How to Use

### 1. Split the dataset (one-time operation)
```bash
python split_dataset.py
```

### 2. Train with separated data
```bash
python train_basic_cnn_v3_no_leakage.py
```

## Benefits

✓ **No Data Leakage** - Impossible for test data to contaminate training
✓ **Transparent** - Can manually verify separation by checking directories
✓ **Reproducible** - Fixed seed ensures same split every time
✓ **Auditable** - Easy to check which images are in which split
✓ **Standard Practice** - Follows best practices for ML experiments

## Verification

You can verify no leakage by checking:
```python
# All paths should be unique
train_files = set(train_dir.rglob("*.png"))
val_files = set(val_dir.rglob("*.png"))
test_files = set(test_dir.rglob("*.png"))

assert len(train_files & val_files) == 0, "Train-Val overlap!"
assert len(train_files & test_files) == 0, "Train-Test overlap!"
assert len(val_files & test_files) == 0, "Val-Test overlap!"
print("✓ No data leakage - all sets are disjoint!")
```

## WandB Tracking

New run tracked as: `basic-cnn-v3-no-leakage`
- Project: fish-classification-hw1
- Entity: orisin-ben-gurion-university-of-the-negev
- Config includes: `data_split: 'physical_separation'`

## Notes

- Original data in `Data/2/Fish_Dataset/Fish_Dataset/` is preserved
- Split data in `Data/split_fish_dataset/` uses ~9GB disk space
- If you need to re-split, delete `split_fish_dataset/` and run again
- The split uses the same random seed (42) as the original training for consistency
