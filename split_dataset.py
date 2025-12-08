"""
Physically split the fish dataset into train/val/test directories
This ensures complete separation and prevents any data leakage
"""

import os
import shutil
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

def split_dataset(
    source_root,
    output_root,
    test_size=0.2,
    val_size=0.1,
    copy_files=True
):
    """
    Split dataset into physically separate train/val/test directories
    
    Args:
        source_root: Path to original dataset
        output_root: Path where split datasets will be created
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation set
        copy_files: If True, copy files; if False, move files
    """
    
    fish_species = [
        "Black Sea Sprat",
        "Gilt-Head Bream",
        "Hourse Mackerel",
        "Red Mullet",
        "Red Sea Bream",
        "Sea Bass",
        "Shrimp",
        "Striped Red Mullet",
        "Trout"
    ]
    
    # Create output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for species in fish_species:
            (split_dir / species).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("SPLITTING DATASET INTO TRAIN/VAL/TEST")
    print("="*70)
    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print(f"Test size: {test_size*100:.0f}%")
    print(f"Val size: {val_size*100:.0f}% of remaining")
    print(f"Train size: {(1-test_size)*(1-val_size)*100:.0f}%")
    print("="*70 + "\n")
    
    total_stats = {'train': 0, 'val': 0, 'test': 0}
    species_stats = {}
    
    # Process each species
    for species_idx, species in enumerate(fish_species, 1):
        print(f"[{species_idx}/{len(fish_species)}] Processing: {species}")
        
        species_path = source_root / species / species
        if not species_path.exists():
            print(f"  ⚠ Warning: Path not found - {species_path}")
            continue
        
        # Get all images for this species
        all_images = list(species_path.glob("*.png"))
        
        if len(all_images) == 0:
            print(f"  ⚠ Warning: No images found")
            continue
        
        # Create labels (all same for stratification)
        labels = np.zeros(len(all_images), dtype=int)
        
        # First split: train+val vs test
        trainval_imgs, test_imgs, _, _ = train_test_split(
            all_images,
            labels,
            test_size=test_size,
            random_state=42,
            stratify=None  # Not needed since all same class
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train_imgs, val_imgs, _, _ = train_test_split(
            trainval_imgs,
            np.zeros(len(trainval_imgs), dtype=int),
            test_size=val_ratio,
            random_state=42,
            stratify=None
        )
        
        species_stats[species] = {
            'train': len(train_imgs),
            'val': len(val_imgs),
            'test': len(test_imgs),
            'total': len(all_images)
        }
        
        # Copy/move files to appropriate directories
        operation = shutil.copy2 if copy_files else shutil.move
        
        print(f"  Train: {len(train_imgs):4d} | Val: {len(val_imgs):4d} | Test: {len(test_imgs):4d} | Total: {len(all_images):4d}")
        
        # Train set
        for img_path in tqdm(train_imgs, desc="  → train", leave=False):
            dest = output_root / 'train' / species / img_path.name
            operation(str(img_path), str(dest))
        total_stats['train'] += len(train_imgs)
        
        # Val set
        for img_path in tqdm(val_imgs, desc="  → val  ", leave=False):
            dest = output_root / 'val' / species / img_path.name
            operation(str(img_path), str(dest))
        total_stats['val'] += len(val_imgs)
        
        # Test set
        for img_path in tqdm(test_imgs, desc="  → test ", leave=False):
            dest = output_root / 'test' / species / img_path.name
            operation(str(img_path), str(dest))
        total_stats['test'] += len(test_imgs)
    
    # Print summary
    print("\n" + "="*70)
    print("SPLIT SUMMARY")
    print("="*70)
    print(f"\nTotal images per split:")
    print(f"  Train:      {total_stats['train']:5d} ({total_stats['train']/sum(total_stats.values())*100:.1f}%)")
    print(f"  Validation: {total_stats['val']:5d} ({total_stats['val']/sum(total_stats.values())*100:.1f}%)")
    print(f"  Test:       {total_stats['test']:5d} ({total_stats['test']/sum(total_stats.values())*100:.1f}%)")
    print(f"  Total:      {sum(total_stats.values()):5d}")
    
    print("\n" + "="*70)
    print("Per-species breakdown:")
    print("="*70)
    print(f"{'Species':<25} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>6}")
    print("-"*70)
    for species, stats in species_stats.items():
        print(f"{species:<25} {stats['train']:>6} {stats['val']:>6} {stats['test']:>6} {stats['total']:>6}")
    
    print("\n" + "="*70)
    print("✓ Dataset split complete!")
    print("="*70)
    print(f"\nSplit directories created at:")
    print(f"  Train: {output_root / 'train'}")
    print(f"  Val:   {output_root / 'val'}")
    print(f"  Test:  {output_root / 'test'}")
    print("\nThese directories are now completely separate - no data leakage possible!")
    print("="*70 + "\n")


def main():
    # Paths
    project_root = Path(r"D:\Projects\Practical_Deep_hw1")
    source_root = project_root / "Data" / "2" / "Fish_Dataset" / "Fish_Dataset"
    output_root = project_root / "Data" / "split_fish_dataset"
    
    # Check if source exists
    if not source_root.exists():
        print(f"Error: Source directory not found: {source_root}")
        return
    
    # Check if output already exists
    if output_root.exists():
        print(f"\n⚠ Warning: Output directory already exists: {output_root}")
        response = input("Do you want to overwrite it? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
        print("Removing existing directory...")
        shutil.rmtree(output_root)
    
    # Split the dataset (copy files to preserve original)
    split_dataset(
        source_root=source_root,
        output_root=output_root,
        test_size=0.2,
        val_size=0.1,
        copy_files=True  # Set to False to move instead of copy
    )


if __name__ == '__main__':
    main()
