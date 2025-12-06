"""
Data Preprocessing and Analysis for Humpback Whale Identification
- Analyzes image properties (sizes, color modes, distributions)
- Applies preprocessing transformations
- Creates cleaned/processed datasets
- Generates visualizations of data characteristics
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# -------------------------
# Configuration
# -------------------------

DATA_DIR = Path("Data")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
TRAIN_CSV = DATA_DIR / "train.csv"

OUTPUT_DIR = Path("preprocessed_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Target image size for preprocessing
TARGET_SIZE = (224, 224)

# -------------------------
# Image Analysis Functions
# -------------------------

def analyze_images(image_dir: Path, csv_path: Path = None, max_samples: int = None) -> Dict:
    """
    Analyze image properties in the dataset.
    
    Args:
        image_dir: Directory containing images
        csv_path: Optional CSV with image labels
        max_samples: Limit analysis to first N images (None = all)
        
    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing images in: {image_dir}")
    print(f"{'='*60}\n")
    
    # Load metadata if available
    df = None
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"Loaded metadata: {len(df)} samples\n")
    
    # Get list of images
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    if max_samples:
        image_files = image_files[:max_samples]
    
    print(f"Found {len(image_files)} images")
    
    # Initialize counters
    sizes = []
    widths = []
    heights = []
    modes = []
    aspect_ratios = []
    file_sizes = []
    corrupted = []
    
    # Analyze each image
    for img_path in tqdm(image_files, desc="Analyzing images"):
        try:
            # Get file size
            file_sizes.append(img_path.stat().st_size / 1024)  # KB
            
            # Load image
            img = Image.open(img_path)
            
            # Get properties
            width, height = img.size
            widths.append(width)
            heights.append(height)
            sizes.append((width, height))
            modes.append(img.mode)
            aspect_ratios.append(width / height)
            
            img.close()
            
        except Exception as e:
            print(f"Error loading {img_path.name}: {e}")
            corrupted.append(img_path.name)
    
    # Compile statistics
    results = {
        'total_images': len(image_files),
        'corrupted': corrupted,
        'num_corrupted': len(corrupted),
        'widths': widths,
        'heights': heights,
        'sizes': sizes,
        'modes': modes,
        'aspect_ratios': aspect_ratios,
        'file_sizes': file_sizes,
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("IMAGE ANALYSIS SUMMARY")
    print(f"{'='*60}\n")
    
    print(f"Total images analyzed: {len(image_files)}")
    print(f"Corrupted images: {len(corrupted)}")
    
    print(f"\nImage Dimensions:")
    print(f"  Width  - min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.1f}")
    print(f"  Height - min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.1f}")
    
    print(f"\nAspect Ratios:")
    print(f"  Min: {min(aspect_ratios):.2f}")
    print(f"  Max: {max(aspect_ratios):.2f}")
    print(f"  Mean: {np.mean(aspect_ratios):.2f}")
    
    print(f"\nColor Modes:")
    mode_counts = Counter(modes)
    for mode, count in mode_counts.most_common():
        print(f"  {mode}: {count} ({100*count/len(modes):.1f}%)")
    
    print(f"\nMost Common Resolutions:")
    size_counts = Counter(sizes)
    for size, count in size_counts.most_common(10):
        print(f"  {size[0]}x{size[1]}: {count} images")
    
    print(f"\nFile Sizes (KB):")
    print(f"  Min: {min(file_sizes):.1f}")
    print(f"  Max: {max(file_sizes):.1f}")
    print(f"  Mean: {np.mean(file_sizes):.1f}")
    
    # Analyze labels if available
    if df is not None:
        print(f"\n{'='*60}")
        print("LABEL ANALYSIS")
        print(f"{'='*60}\n")
        
        label_counts = df['Id'].value_counts()
        
        print(f"Total unique whale IDs: {len(label_counts)}")
        print(f"\nSamples per whale ID:")
        print(f"  Min: {label_counts.min()}")
        print(f"  Max: {label_counts.max()}")
        print(f"  Mean: {label_counts.mean():.2f}")
        print(f"  Median: {label_counts.median():.1f}")
        
        # Check for new_whale
        if 'new_whale' in label_counts.index:
            print(f"\n'new_whale' samples: {label_counts['new_whale']}")
        
        print(f"\nTop 10 most common whale IDs:")
        for whale_id, count in label_counts.head(10).items():
            print(f"  {whale_id}: {count} samples")
        
        # Distribution statistics
        print(f"\nDistribution of samples per ID:")
        print(f"  IDs with 1 sample: {(label_counts == 1).sum()}")
        print(f"  IDs with 2 samples: {(label_counts == 2).sum()}")
        print(f"  IDs with 3-5 samples: {((label_counts >= 3) & (label_counts <= 5)).sum()}")
        print(f"  IDs with 6-10 samples: {((label_counts >= 6) & (label_counts <= 10)).sum()}")
        print(f"  IDs with >10 samples: {(label_counts > 10).sum()}")
        
        results['label_counts'] = label_counts
        results['df'] = df
    
    return results


def visualize_analysis(results: Dict, output_dir: Path):
    """
    Create visualizations of the analysis results.
    
    Args:
        results: Dictionary from analyze_images()
        output_dir: Directory to save plots
    """
    print(f"\n{'='*60}")
    print("Creating Visualizations")
    print(f"{'='*60}\n")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Width distribution
    ax = axes[0, 0]
    ax.hist(results['widths'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(results['widths']), color='red', linestyle='--', 
               label=f"Mean: {np.mean(results['widths']):.0f}")
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Frequency')
    ax.set_title('Image Width Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Height distribution
    ax = axes[0, 1]
    ax.hist(results['heights'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(np.mean(results['heights']), color='red', linestyle='--',
               label=f"Mean: {np.mean(results['heights']):.0f}")
    ax.set_xlabel('Height (pixels)')
    ax.set_ylabel('Frequency')
    ax.set_title('Image Height Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Aspect ratio distribution
    ax = axes[0, 2]
    ax.hist(results['aspect_ratios'], bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(np.mean(results['aspect_ratios']), color='red', linestyle='--',
               label=f"Mean: {np.mean(results['aspect_ratios']):.2f}")
    ax.set_xlabel('Aspect Ratio (width/height)')
    ax.set_ylabel('Frequency')
    ax.set_title('Aspect Ratio Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Color mode distribution
    ax = axes[1, 0]
    mode_counts = Counter(results['modes'])
    modes_list = list(mode_counts.keys())
    counts_list = list(mode_counts.values())
    ax.bar(modes_list, counts_list, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Color Mode')
    ax.set_ylabel('Count')
    ax.set_title('Color Mode Distribution')
    ax.grid(alpha=0.3, axis='y')
    
    # 5. File size distribution
    ax = axes[1, 1]
    ax.hist(results['file_sizes'], bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax.set_xlabel('File Size (KB)')
    ax.set_ylabel('Frequency')
    ax.set_title('File Size Distribution')
    ax.grid(alpha=0.3)
    
    # 6. Label distribution (if available)
    ax = axes[1, 2]
    if 'label_counts' in results:
        label_counts = results['label_counts']
        # Plot histogram of samples per ID
        samples_per_id = label_counts.values
        ax.hist(samples_per_id[samples_per_id <= 50], bins=50, 
                edgecolor='black', alpha=0.7, color='red')
        ax.set_xlabel('Samples per Whale ID')
        ax.set_ylabel('Number of IDs')
        ax.set_title('Label Distribution (≤50 samples)')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No label data', ha='center', va='center')
        ax.axis('off')
    
    plt.tight_layout()
    plot_path = output_dir / 'dataset_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {plot_path}")
    plt.close()
    
    # Additional plot: Top resolution sizes
    if len(results['sizes']) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        size_counts = Counter(results['sizes'])
        top_sizes = size_counts.most_common(20)
        
        labels = [f"{w}x{h}" for (w, h), _ in top_sizes]
        counts = [count for _, count in top_sizes]
        
        ax.barh(labels, counts, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Number of Images')
        ax.set_ylabel('Resolution (WxH)')
        ax.set_title('Top 20 Most Common Image Resolutions')
        ax.grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        plot_path = output_dir / 'top_resolutions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization: {plot_path}")
        plt.close()


def check_image_quality(image_path: Path) -> Tuple[bool, str]:
    """
    Check if an image can be properly loaded and converted.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        img = Image.open(image_path)
        
        # Try to convert to RGB
        img_rgb = img.convert('RGB')
        
        # Check if image has valid size
        if img.size[0] == 0 or img.size[1] == 0:
            return False, "Invalid dimensions"
        
        img.close()
        return True, ""
        
    except Exception as e:
        return False, str(e)


def create_preprocessed_dataset_info(df: pd.DataFrame, train_dir: Path, output_dir: Path):
    """
    Create a CSV with preprocessing information for each image.
    
    Args:
        df: DataFrame with image metadata
        train_dir: Directory containing training images
        output_dir: Directory to save output
    """
    print(f"\n{'='*60}")
    print("Creating Preprocessed Dataset Info")
    print(f"{'='*60}\n")
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        img_name = row['Image']
        img_path = train_dir / img_name
        whale_id = row['Id']
        
        if not img_path.exists():
            results.append({
                'Image': img_name,
                'Id': whale_id,
                'exists': False,
                'valid': False,
                'error': 'File not found'
            })
            continue
        
        # Check image validity
        is_valid, error = check_image_quality(img_path)
        
        # Get image properties
        try:
            img = Image.open(img_path)
            width, height = img.size
            mode = img.mode
            img.close()
        except:
            width, height, mode = 0, 0, 'unknown'
        
        results.append({
            'Image': img_name,
            'Id': whale_id,
            'exists': True,
            'valid': is_valid,
            'error': error if not is_valid else '',
            'width': width,
            'height': height,
            'mode': mode,
            'needs_rgb_conversion': mode != 'RGB'
        })
    
    # Create DataFrame
    result_df = pd.DataFrame(results)
    
    # Print summary
    print(f"\nPreprocessing Summary:")
    print(f"  Total images: {len(result_df)}")
    print(f"  Valid images: {result_df['valid'].sum()}")
    print(f"  Invalid images: {(~result_df['valid']).sum()}")
    print(f"  Need RGB conversion: {result_df['needs_rgb_conversion'].sum()}")
    
    # Save to CSV
    output_path = output_dir / 'preprocessing_info.csv'
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved preprocessing info to: {output_path}")
    
    return result_df


def display_sample_images(df: pd.DataFrame, train_dir: Path, n_samples: int = 12):
    """
    Display a grid of sample images from the dataset.
    
    Args:
        df: DataFrame with image metadata
        train_dir: Directory containing images
        n_samples: Number of samples to display
    """
    print(f"\n{'='*60}")
    print("Displaying Sample Images")
    print(f"{'='*60}\n")
    
    # Sample random images
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=42)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        if idx >= n_samples:
            break
            
        img_path = train_dir / row['Image']
        
        try:
            img = Image.open(img_path)
            img_rgb = img.convert('RGB')
            
            axes[idx].imshow(img_rgb)
            axes[idx].set_title(f"{row['Id']}\n{img.size[0]}x{img.size[1]}", fontsize=8)
            axes[idx].axis('off')
            
            img.close()
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Error loading\n{row['Image']}", 
                          ha='center', va='center')
            axes[idx].axis('off')
    
    plt.suptitle('Sample Images from Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'sample_images.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved sample images to: {output_path}")
    plt.close()


# -------------------------
# Main Execution
# -------------------------

def main():
    """Main preprocessing pipeline."""
    
    print("\n" + "="*60)
    print("HUMPBACK WHALE DATA PREPROCESSING")
    print("="*60 + "\n")
    
    # Step 1: Analyze training images
    if TRAIN_DIR.exists() and TRAIN_CSV.exists():
        train_results = analyze_images(TRAIN_DIR, TRAIN_CSV, max_samples=None)
        
        # Step 2: Create visualizations
        visualize_analysis(train_results, OUTPUT_DIR)
        
        # Step 3: Display sample images
        if 'df' in train_results:
            display_sample_images(train_results['df'], TRAIN_DIR, n_samples=12)
        
        # Step 4: Create preprocessing info
        if 'df' in train_results:
            prep_df = create_preprocessed_dataset_info(
                train_results['df'], 
                TRAIN_DIR, 
                OUTPUT_DIR
            )
            
            # Save clean dataset (only valid images)
            clean_df = prep_df[prep_df['valid']].copy()
            clean_df = clean_df[['Image', 'Id']]
            clean_path = OUTPUT_DIR / 'train_clean.csv'
            clean_df.to_csv(clean_path, index=False)
            print(f"\nSaved clean training CSV to: {clean_path}")
            print(f"  Original: {len(train_results['df'])} samples")
            print(f"  Clean: {len(clean_df)} samples")
    else:
        print(f"Error: Training data not found!")
        print(f"  Expected CSV: {TRAIN_CSV}")
        print(f"  Expected directory: {TRAIN_DIR}")
    
    # Step 5: Analyze test images (if available)
    if TEST_DIR.exists():
        print("\n" + "="*60)
        test_results = analyze_images(TEST_DIR, csv_path=None, max_samples=None)
        
        # Create test visualizations
        test_output_dir = OUTPUT_DIR / "test"
        test_output_dir.mkdir(exist_ok=True)
        visualize_analysis(test_results, test_output_dir)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"\nAll outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\nRecommended preprocessing steps for training:")
    print("  1. Convert all images to RGB")
    print("  2. Resize to 224x224 (or 256x256)")
    print("  3. Normalize with ImageNet mean/std")
    print("  4. Apply augmentations: horizontal flip, rotation, color jitter")
    print("\nRecommended approach:")
    print("  - Use metric learning (triplet loss, ArcFace)")
    print("  - Pretrained backbones (ResNet50, EfficientNet)")
    print("  - Heavy augmentation due to class imbalance")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
