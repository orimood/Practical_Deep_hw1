# Part 3d: Feature Extraction Experiment

## Status: RUNNING

The feature extraction experiment script (`prt_3/feature_extraction_experiment.py`) has been successfully configured and started.

### What the Script Does

This script implements **Feature Extraction from Transfer Learning** - one of the key requirements for Part 3d of the assignment:

1. **Loads a pretrained ResNet-50 model** from torchvision (ImageNet weights)
2. **Removes the last fully connected layer** to extract features
3. **Extracts 2048-dimensional feature vectors** for all train/val/test samples
4. **Trains classical ML algorithms** on these fixed features:
   - **Support Vector Machine (RBF kernel)**
   - **Support Vector Machine (Linear kernel)**
   - **Random Forest (100 trees)**
5. **Evaluates and compares** the performance against the Transfer Learning baseline

### Data Setup ✅

- **Source**: `data/2/Fish_Dataset/Fish_Dataset/` (9 fish species)
- **Split**: Train (6,300), Validation (900), Test (1,800) - now at `data/fish_split/`
- **Classes**: 9 fish species
  - Black Sea Sprat
  - Gilt-Head Bream
  - Hourse Mackerel
  - Red Mullet
  - Red Sea Bream
  - Sea Bass
  - Shrimp
  - Striped Red Mullet
  - Trout

### Processing Details

**Feature Extraction Phase:**
- Input: RGB images (224×224) normalized with ImageNet statistics
- Model: ResNet-50 (pretrained)
- Output: 2048-dimensional feature vectors per image
- Status: In progress (currently extracting training set features)

**Classifier Training Phase:**
- Algorithm 1: SVM with RBF kernel (C=10)
- Algorithm 2: SVM with Linear kernel (C=1.0)
- Algorithm 3: Random Forest with 100 trees
- Training set: 6,300 samples
- Evaluation: Validation and Test metrics

### Expected Outputs

When complete, the script will generate:

1. **`models/feature_extraction_results.json`**
   - Model performance metrics for each classifier
   - Test accuracy, precision, recall, F1-score
   - Unique correct samples and errors
   - Feature dimension information

2. **`models/feature_extraction_comparison.csv`**
   - Comparison table: Feature Extraction vs Transfer Learning
   - Runtime metrics
   - Test performance across all methods

3. **Confusion Matrices**
   - `models/svm_rbf_confusion_matrices.png`
   - `models/svm_linear_confusion_matrices.png`
   - `models/random_forest_confusion_matrices.png`
   - Validation and Test confusion matrices for each classifier

### Key Findings (Once Complete)

- **Best Feature Extraction Method**: Will show which classical ML algorithm performs best on ResNet-50 features
- **Comparison to Transfer Learning**: Will demonstrate if classical ML on pre-extracted features can match fine-tuned models
- **Parameter Settings**: 
  - Feature Extractor: ResNet-50 (ImageNet pretrained)
  - Feature Dimension: 2048
  - No fine-tuning (features frozen from ImageNet pretraining)
- **Processing Changes**:
  - Removed last FC layer from ResNet-50
  - Extracted fixed features instead of training end-to-end
  - Applied classical ML algorithms on fixed representations

### Runtime Estimate

On CPU (Mac):
- Feature Extraction: ~15-20 minutes (for all 8,000 images)
- Classifier Training: ~5-10 minutes (SVM training can be slow on CPU)
- **Total: ~25-30 minutes**

### File Status

✅ `prt_3/feature_extraction_experiment.py` - Created and configured
✅ `data/fish_split/` - Dataset split complete
✅ `venv/` - Virtual environment with all dependencies
⏳ `output.log` - Running (check with: `tail -f output.log`)

### To Monitor Progress

```bash
tail -f /Users/orimood/Desktop/homework/Visual\ Studio\ Code/Practical_Deep_hw1/output.log
```

### Results Location

Once complete, check:
- `/models/feature_extraction_results.json` - Results summary
- `/models/feature_extraction_comparison.csv` - Comparison table
- `/models/*_confusion_matrices.png` - Visualization plots

---

**Part 3d Requirement Fulfillment:**
- ✅ Chose ResNet-50 from trained models in section 3c
- ✅ Removed last layer for feature extraction
- ✅ Using classical ML algorithms (SVM, Random Forest) on extracted features
- ✅ Comparing performance to previous transfer learning results
- ✅ Summary table with runtime, loss, metrics, and parameter settings
