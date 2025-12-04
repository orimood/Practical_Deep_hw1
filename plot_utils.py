import os
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def main():
    # ---------------------------------------------------------
    # 0. Paths & basic config
    # ---------------------------------------------------------
    project_root = Path(__file__).resolve().parent if "__file__" in globals() else Path(".").resolve()
    data_root = project_root / "Data"

    train_csv_path = data_root / "train.csv"
    train_images_dir = data_root / "train"
    test_images_dir = data_root / "test"   # May be useful later

    rng = np.random.default_rng(seed=42)   # for reproducibility

    # ---------------------------------------------------------
    # 1. Load metadata (train.csv) and basic sizes
    # ---------------------------------------------------------
    if not train_csv_path.is_file():
        raise FileNotFoundError(f"Could not find train.csv at: {train_csv_path}")

    train_df = pd.read_csv(train_csv_path)

    # Basic info
    num_samples = len(train_df)
    unique_whales = train_df["Id"].nunique()
    num_new_whale = (train_df["Id"] == "new_whale").sum()

    print("==== Basic Dataset Info ====")
    print(f"# Train samples:          {num_samples}")
    print(f"# Unique whale Ids:       {unique_whales}")
    print(f"# 'new_whale' samples:    {num_new_whale}")
    print(f"Train image folder:       {train_images_dir}")
    print(f"Test image folder:        {test_images_dir}")
    print()

    # ---------------------------------------------------------
    # 2. Inspect image dimensions / channels
    #    (sample a subset so it's fast)
    # ---------------------------------------------------------
    print("==== Image Dimensions & Channels (Sample) ====")

    # Take up to 200 random images for dimension sampling
    sample_size = min(200, num_samples)
    sample_indices = rng.choice(num_samples, size=sample_size, replace=False)
    sampled_files = train_df.iloc[sample_indices]["Image"].values

    sizes_counter = Counter()
    mode_counter = Counter()  # RGB / L / etc.

    for fname in sampled_files:
        img_path = train_images_dir / fname
        if not img_path.is_file():
            continue
        with Image.open(img_path) as img:
            sizes_counter[img.size] += 1  # (width, height)
            mode_counter[img.mode] += 1   # e.g., 'RGB'

    # Show most common sizes & modes in the sample
    print("Most common image sizes in sample:")
    for (w, h), count in sizes_counter.most_common(5):
        print(f"  {w}x{h}  (count={count})")

    print("\nImage modes (channels) in sample:")
    for mode, count in mode_counter.most_common():
        print(f"  mode={mode}, count={count}")
    print()

    # Quick suggestion about preprocessing (printed to console)
    print("Suggested preprocessing:")
    print("- Convert all images to RGB (3 channels).")
    print("- Resize / center-crop to a fixed size (e.g., 224x224 or 256x256) for CNNs.")
    print("- Normalize pixel values using ImageNet mean/std if using pretrained backbones.")
    print()

    # ---------------------------------------------------------
    # 3. Class balance: how many images per whale Id?
    # ---------------------------------------------------------
    print("==== Class Balance ====")
    counts_per_id = train_df["Id"].value_counts()

    print(f"Min images per Id: {counts_per_id.min()}")
    print(f"Max images per Id: {counts_per_id.max()}")
    print(f"Mean images per Id: {counts_per_id.mean():.2f}")
    print(f"Median images per Id: {counts_per_id.median():.2f}")

    # Show a few head/tail of the distribution
    print("\nTop 10 most frequent Ids:")
    print(counts_per_id.head(10))

    print("\nBottom 10 least frequent Ids:")
    print(counts_per_id.tail(10))
    print()

    # Histogram of images per Id (clipped for visualization)
    plt.figure(figsize=(10, 5))
    max_clip = 20  # clip at 20 to avoid very long tail
    clipped_counts = np.clip(counts_per_id.values, 0, max_clip)
    plt.hist(clipped_counts, bins=np.arange(1, max_clip + 2) - 0.5)
    plt.xlabel("# Images per Id (clipped at 20)")
    plt.ylabel("# Whale Ids")
    plt.title("Class Frequency Histogram (Number of Images per Whale Id)")
    plt.xticks(range(1, max_clip + 1))
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 4. 3×3 grid: 3 random whales × 3 images each
    # ---------------------------------------------------------
    print("==== 3×3 Sample Grid: 3 random whales × 3 images each ====")

    plot_random_whale_grid(
        train_df=train_df,
        images_dir=train_images_dir,
        rng=rng,
        num_whales=3,
        imgs_per_whale=3,
        figsize=(10, 10),
    )


def plot_random_whale_grid(
    train_df: pd.DataFrame,
    images_dir: Path,
    rng: np.random.Generator,
    num_whales: int = 3,
    imgs_per_whale: int = 3,
    figsize=(10, 10),
):
    """
    Plot a num_whales x imgs_per_whale grid of images:
    - Each row corresponds to one whale Id.
    - Each row shows imgs_per_whale images of that whale.

    This is the "3-3 table" requested: 3 random whales, each with 3 images (total 9 cells).
    """

    # Filter ids that have at least imgs_per_whale samples
    id_counts = train_df["Id"].value_counts()
    eligible_ids = id_counts[id_counts >= imgs_per_whale].index.tolist()

    if len(eligible_ids) < num_whales:
        raise ValueError(
            f"Not enough whale Ids with at least {imgs_per_whale} images. "
            f"Found only {len(eligible_ids)} eligible ids."
        )

    # Sample random whale ids
    chosen_ids = rng.choice(eligible_ids, size=num_whales, replace=False)

    fig, axes = plt.subplots(
        nrows=num_whales,
        ncols=imgs_per_whale,
        figsize=figsize,
        squeeze=False
    )
    fig.suptitle("3×3 Sample Grid: 3 Random Whales × 3 Images Each", fontsize=16)
    fig.subplots_adjust(hspace=0.3, wspace=0.1)

    for row_idx, whale_id in enumerate(chosen_ids):
        whale_rows = train_df[train_df["Id"] == whale_id]
        # Randomly sample imgs_per_whale rows for this id
        whale_samples = whale_rows.sample(n=imgs_per_whale, random_state=int(rng.integers(0, 1e9)))

        for col_idx, (_, row) in enumerate(whale_samples.iterrows()):
            fname = row["Image"]
            img_path = images_dir / fname

            ax = axes[row_idx, col_idx]
            ax.axis("off")

            if not img_path.is_file():
                ax.set_title(f"{whale_id}\n(Missing: {fname})", fontsize=8)
                continue

            with Image.open(img_path) as img:
                img = img.convert("RGB")
                ax.imshow(img)

            # First column: write whale Id as row title
            if col_idx == 0:
                ax.set_ylabel(f"Id: {whale_id}", fontsize=10)
            ax.set_title(fname, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    plt.show()


if __name__ == "__main__":
    main()
