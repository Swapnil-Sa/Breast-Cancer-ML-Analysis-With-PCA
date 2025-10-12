# 🩺 Breast Cancer Classification using PCA, K-Means, and Lazy Classifier

This project applies **Principal Component Analysis (PCA)**, **Unsupervised Clustering (K-Means)**, and **Supervised Classification (Lazy Predict & Threshold Tuning)** to the **Wisconsin Breast Cancer Dataset (WBCD)**.  
It forms part of a research project exploring dimensionality reduction, cluster-based diagnosis grouping, and model performance optimization.

---

## 📚 Overview

The workflow performs:
1. **Data Cleaning and Preprocessing**
2. **Principal Component Analysis (PCA)**
3. **K-Means Clustering on PCA-transformed Data**
4. **Model Evaluation using Lazy Predict**
5. **Classification Threshold Tuning**
6. **Performance Visualization and Analysis**

---

## 🧩 Steps in Detail

### 1. Data Preprocessing
- Removed unwanted/unnamed columns, duplicates, and missing values.
- Standardized numeric features using `StandardScaler`.

### 2. Principal Component Analysis (PCA)
- Computed covariance matrix, eigenvalues, and eigenvectors manually.
- Selected top 10 principal components.
- Visualized:
  - **Feature Loadings (PC1–PC10)**
  - **Feature Loadings Heatmap**
  - **Explained Variance and Cumulative Variance**

Outputs:
- `pca_results.csv`
- Feature loading plots

### 3. K-Means Clustering
- Applied `KMeans` with:
  - 2 clusters (Benign, Malignant)
  - 1500 initializations (`n_init`)
  - 500 max iterations
- Calculated **Silhouette Score** to assess cluster quality.
- Saved cluster results as `kmeans_pca_results.csv`.

### 4. Lazy Classification
Benchmarked the following classifiers:
- `LinearSVC`
- `Logistic Regression`
- `Random Forest`
- `AdaBoost`

Each model produced:
- **Confusion Matrix** (visualized via Seaborn)
- **Classification Report** (Precision, Recall, F1, Accuracy)

### 5. Threshold Tuning
- Trained an `AdaBoostClassifier` on PCA-based features.
- Tuned thresholds (0.0–1.0) to analyze performance trade-offs.
- Calculated:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
  - **ROC-AUC Score**
- Plotted metrics vs threshold.

---

## 📊 Key Outputs
| File | Description |
|------|--------------|
| `pca_results.csv` | Eigenvectors & principal component values |
| `kmeans_pca_results.csv` | PCA data with cluster labels |
| `boxplots.png`, `heatmap.png` | Feature visualization (if added) |
| `threshold_tuning_plot.png` | Model performance across thresholds |

---

## ⚙️ Tech Stack
- **Python**  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `lazypredict`, `tabulate`

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/breast-cancer-pca.git
   cd breas

