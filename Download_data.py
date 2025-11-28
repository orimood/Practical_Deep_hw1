import os
import shutil
from pathlib import Path

import kagglehub

# 1. Download to KaggleHub's cache (this is how it's meant to be used)
cache_path = kagglehub.dataset_download("crowww/a-large-scale-fish-dataset")
print("KaggleHub cache path:", cache_path)

# 2. Define your desired local root directory
target_root = Path(r"D:\Practical_deep_hw1\Data")
target_root.mkdir(parents=True, exist_ok=True)

# 3. Put the dataset under that root (e.g. D:\Practical_deep_hw1\Data\a-large-scale-fish-dataset)
cache_path = Path(cache_path)
target_path = target_root / cache_path.name

# If you want to overwrite any previous copy, uncomment the rmtree:
if target_path.exists():
    print(f"Removing existing folder: {target_path}")
    shutil.rmtree(target_path)

print(f"Copying dataset to: {target_path}")
shutil.copytree(cache_path, target_path)

print("Final dataset location:", target_path)
