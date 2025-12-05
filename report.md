# Assignment 1 – Practical Deep Learning Workshop
**Ori Sinvani – 325770824**  
**Harel Brener – 214179012**

---

## Abstract
Your abstract.

---

## Introduction

---

## Question 1 – Exploratory Data Analysis

### a – Size of Data

The Humpback Whale Identification dataset contains thousands of labeled images of whale flukes.  
According to the loaded training metadata:

- **Train samples:** ≈ 25,000 images  
- **Unique whale IDs:** Over 3,000 individuals  
- A special label **`new_whale`** contains:  
  **9,664 samples**, representing whales the model has not seen.

Thus, the dataset is large in total, but extremely sparse per identity.

---

### b – What Each Sample Contains

Each training sample consists of:

- A **color image** of a whale fluke.
- Image shapes vary significantly. The most common resolutions observed were:  
  - 1050×700  
  - 1050×450  
  - 1050×600  
  - 700×500  

Most images are stored in **RGB (3 channels)**, though some are grayscale.

Because image sizes and color modes are inconsistent, preprocessing is necessary.

---

### Preprocessing Suggestions

Based on dataset characteristics:

- Convert all images to **RGB**.  
- Resize or center-crop to a uniform size (e.g., **224×224** or **256×256**) for CNN training.  
- Normalize using ImageNet mean/std when using pretrained backbones.

Useful augmentations:

- Horizontal flip (flukes are symmetric)  
- Random rotations  
- Color jitter  
- Random zoom/crop  

These transformations maintain identity cues while improving generalization.

---

### c – Is the Data Balanced?

No — the dataset is **extremely imbalanced**.

Sample distribution per whale ID:

- **Minimum per ID:** 1  
- **Median per ID:** 2  
- **Mean per ID:** 5.07  
- **Maximum per ID:** 9,664 (`new_whale`)  

Most whales appear only once or twice.  
Because of this, strong methods are needed:

- Metric learning (triplet loss, arcface)  
- Embedding-based retrieval instead of softmax  
- Heavy augmentation  

---

### d – Benchmark Results (Related Work)

Insights from Kaggle solutions and related research:

- Simple CNN classifiers struggle due to extreme imbalance.
- **Metric learning** methods significantly outperform softmax classification.
- Top solutions use:
  - Pretrained backbones: ResNet50, Inception, EfficientNet  
  - Very heavy augmentation  
  - Embedding extraction + k-NN retrieval  

State-of-the-art models achieve high accuracy by generating deep embeddings and performing nearest-neighbor search rather than directly predicting classes.

---

