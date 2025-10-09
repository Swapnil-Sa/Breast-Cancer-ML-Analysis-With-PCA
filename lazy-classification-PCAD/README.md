# Breast Cancer Feature Selection, PCA, and Classification

This repository contains a comprehensive machine learning pipeline for breast cancer diagnosis based on the Wisconsin Breast Cancer Dataset. The workflow includes feature selection, dimensionality reduction using Principal Component Analysis (PCA), exploratory data analysis, multiple classification models, and classification threshold tuning to optimize performance.

## Project Overview

Breast cancer is one of the most common cancers worldwide. Accurate and early diagnosis significantly improves patient outcomes. This project leverages machine learning to identify key features for diagnosis, reduce dimensionality, visualize data, and build predictive classification models.

Key steps include:

- Selecting important features using Random Forest feature importance and Recursive Feature Elimination (RFE).
- Applying PCA to reduce feature dimensionality to 2 principal components.
- Visualizing data distributions with box plots, scatter plots, and heatmaps.
- Using LazyPredict to quickly evaluate multiple classification models.
- Fine-tuning classification thresholds to optimize metrics like accuracy, precision, recall, and F1-score.

---

## Dataset

The dataset used is the [Wisconsin Breast Cancer Dataset (WBC)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) containing features computed from digitized images of fine needle aspirate (FNA) of breast masses. It includes:

- 30 numeric features (e.g., radius, texture, perimeter, area, smoothness).
- Diagnosis label: Malignant (M) or Benign (B).

---

## Features and Target

- **Features:** Various tumor measurements (mean, worst, concavity, etc.).
- **Target:** Diagnosis mapped to binary values — Malignant = 1, Benign = 0.

---

## Methodology

### Feature Selection

- Standardize features using `StandardScaler`.
- Train a `RandomForestClassifier` to compute feature importances.
- Use Recursive Feature Elimination (RFE) with Random Forest to select the top 10 most important features.
- Save the reduced dataset for PCA.

### Principal Component Analysis (PCA)

- Impute missing values if present.
- Use selected features to perform PCA, reducing dimensionality to 2 principal components.
- Create a new dataset containing the two principal components and the diagnosis label.
- Save PCA results to CSV.

### Exploratory Data Analysis (EDA)

- Box plots to compare principal components between benign and malignant cases.
- Scatter plot to visualize separation of classes in PCA space.
- Heatmap to visualize correlation between principal components and diagnosis.

### Classification with LazyPredict

- Use `LazyClassifier` to quickly benchmark various classification models on the PCA dataset.
- Select the top 4 performing models.
- Train, evaluate, and visualize confusion matrices for the selected models.

### Classification Threshold Tuning

- Use a calibrated Linear Support Vector Classifier (`LinearSVC`) for probabilistic predictions.
- Evaluate model performance across thresholds from 0 to 1.
- Plot metrics (accuracy, precision, recall, F1-score) against threshold values.
- Determine the best threshold for each metric.

---

## Results

- Top 2 features selected based on Random Forest RFE.
- PCA reduces the dataset to two components explaining a significant portion of variance.
- Visualizations demonstrate clear class separation.
- LazyPredict identifies best-performing classifiers quickly.
- Threshold tuning improves classification decision boundary performance.

---
