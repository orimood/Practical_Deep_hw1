import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------
# CONFIG
# -------------------------
DATA_ROOT = r"Data/2/Fish_Dataset/Fish_Dataset"
OUTPUT_DIR = "plots"
SAMPLES_PER_CLASS = 5
DPI = 500  # High-resolution export

os.makedirs(OUTPUT_DIR, exist_ok=True)

# collect only class folders (no README.txt, license, etc.)
CLASS_FOLDERS = [
    d for d in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, d))
]
CLASS_FOLDERS = sorted(CLASS_FOLDERS)
print("Class folders:", CLASS_FOLDERS)


# -------------------------
# HELPER: find RGB and GT
# -------------------------
def get_rgb_and_gt_dirs(class_name):
    """
    Finds the subfolder containing images and the subfolder containing GT masks.
    """
    outer = os.path.join(DATA_ROOT, class_name)
    subdirs = [
        s for s in os.listdir(outer)
        if os.path.isdir(os.path.join(outer, s))
    ]

    rgb_sub = [s for s in subdirs if not s.endswith("GT")]
    gt_sub  = [s for s in subdirs if s.endswith("GT")]

    if not rgb_sub or not gt_sub:
        raise RuntimeError(f"Could not find RGB/GT subfolders for {class_name}. Subdirs: {subdirs}")

    return (
        os.path.join(outer, rgb_sub[0]),
        os.path.join(outer, gt_sub[0])
    )


# -------------------------
# COLLECT IMAGES RECURSIVELY
# -------------------------
def collect_images_recursively(folder, n):
    """
    Returns n random .png images recursively from folder.
    """
    paths = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".png"):
                paths.append(os.path.join(root, f))
    if len(paths) < n:
        raise RuntimeError(f"Folder '{folder}' has only {len(paths)} images, but {n} requested.")
    return [Image.open(p).convert("RGB") for p in random.sample(paths, n)]


# -------------------------
# LOAD SAMPLES
# -------------------------
rgb_samples = {}
gt_samples = {}

for cls in CLASS_FOLDERS:
    rgb_dir, gt_dir = get_rgb_and_gt_dirs(cls)

    rgb_imgs = collect_images_recursively(rgb_dir, SAMPLES_PER_CLASS)
    gt_imgs = [
        Image.open(p).convert("L")
        for p in random.sample(
            [
                os.path.join(root, f)
                for root, dirs, files in os.walk(gt_dir)
                for f in files if f.lower().endswith(".png")
            ],
            SAMPLES_PER_CLASS
        )
    ]

    rgb_samples[cls] = rgb_imgs
    gt_samples[cls] = gt_imgs


# -------------------------
# FUNCTION TO PLOT & SAVE
# -------------------------
def plot_and_save_table(samples_dict, out_path, title, cmap=None):
    rows = len(samples_dict)
    cols = SAMPLES_PER_CLASS

    fig, ax = plt.subplots(
        rows, cols,
        figsize=(cols * 3.0, rows * 3.0),  # large figure
        dpi=DPI
    )
    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    if rows == 1:
        ax = [ax]

    for i, (cls, imgs) in enumerate(samples_dict.items()):
        for j in range(cols):
            ax[i][j].imshow(imgs[j], cmap=cmap)
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            if j == 0:
                ax[i][j].set_ylabel(
                    cls,
                    fontsize=12,
                    rotation=0,
                    labelpad=50
                )

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")


# -------------------------
# SAVE RGB TABLE
# -------------------------
plot_and_save_table(
    rgb_samples,
    os.path.join(OUTPUT_DIR, "rgb_table.png"),
    "RGB Images – 5 Random Samples per Class",
    cmap=None
)

# -------------------------
# SAVE GT TABLE
# -------------------------
plot_and_save_table(
    gt_samples,
    os.path.join(OUTPUT_DIR, "gt_table.png"),
    "GT Masks – 5 Random Samples per Class",
    cmap="gray"
)
