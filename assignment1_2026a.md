# Assignment 1 – Practical Deep Learning Workshop
*(Submit by Dec 9th EOD)*

---

## Exploratory Data Analysis (10 pts)

- **What is the size of the data?**
- **What does each sample contain?**  
  (dimensions, channels, how many classes? Should we preprocess the data?  
  Or is it ready for use? Can we use augmentation? If so, what kind would be valid?)
- **Is the data balanced?**  
  (How many examples are there per class?)
- **Are there any benchmark results for different methods used on this data?**
- **Show samples from each label**  
  (if there are many categories, present easily separable ones vs. harder similar ones)

---

## Neural Network Modeling (35 pts)

Form a neural network graph based on the components used in class walkthroughs.

### Requirements:
- Use **K-Fold Cross Validation (K ≥ 5)** to measure performance and compare settings.
- Fit your model and analyze results. Include:
  - Visualizations of loss and relevant metrics  
  - Examples of good and bad classifications with probabilities  
  - Discussion of uncertain predictions  
  - Comparison of training vs. validation/test performance
- Present validation and test metrics for each fold, and compare them to test metrics achieved via *mean of all folds prediction*.
- Identify where & why the model misclassifies.
- Suggest **at least 3 ways to improve results**.
- Prioritize suggestions and **implement the first 2**, then repeat section (b).
- Add **inference-time augmentation** (augment test samples, aggregate predictions) and report improvement.
- Add a **new category** with a few images, retrain to include it.

---

## Transfer Learning Models (50 pts)

Select **4 or more architectures** (from `torchvision.models` or any relevant pretrained model).

For each model:
- Change the last layer for the new task.
- Perform preprocessing steps.
- Fine‑tune the model.

### Fill the Table:

| Model Name | # Parameters | Validation Loss | Validation Accuracy | Test Loss | Test Accuracy | # Unique Correct Samples | # Unique Errors |
|------------|--------------|------------------|----------------------|-----------|----------------|---------------------------|------------------|
| 1 | | | | | | | |
| 2 | | | | | | | |
| 3 | | | | | | | |
| 4 | | | | | | | |

---

## Feature Extraction Experiment

Choose one of the trained models from section **3c**, remove the last layer, and:
- Use extracted features for a classical ML algorithm.
- Compare performance to previous results.
- Present a summary table including:
  - Runtime  
  - Loss  
  - Metrics  
  - Parameter settings  
  - Processing changes  
  - Any significant modifications

---

## Final Report (5 pts)

Write a **summarizing report**:
- Highlight the most important findings.
- De‑emphasize less important details.
- Ideally format as a publishable blog post or social‑media article.

---

## Optional Datasets

- https://www.kaggle.com/crowww/a-large-scale-fish-dataset  
- https://www.kaggle.com/gpiosenka/100-bird-species  
- https://challenge.isic-archive.com/data  

---

*Source: Provided assignment document*  
