# Breast Cancer Classification with PCA 

This project applies **Machine Learning techniques** to the **Wisconsin Breast Cancer Dataset (WBCD)**, focusing on **exploratory data analysis (EDA)**, **model comparison using Lazy Predict**, and **classification threshold tuning** for performance optimization.  

---

## 📂 Project Structure


---

## 📊 Dataset
- **Name:** Wisconsin Breast Cancer Dataset (WBCD)  
- **Samples:** 569  
- **Features:** 30 (mean, worst, etc. of radius, texture, perimeter, concavity, etc.)  
- **Target variable:** `diagnosis` → Malignant (M) or Benign (B)  

---

## 🔎 Workflow

### 1. Data Exploration & Cleaning
- Checked dataset shape, missing values, and unique counts  
- Converted target (`diagnosis`) into numerical format (M = 1, B = 0)  
- Distribution of classes (Malignant vs Benign)  

### 2. Automated EDA with **Sweetviz**
- Generated an interactive EDA report  
- Visualized data distributions, correlations, and target relationships  

### 3. Visualization
- **Boxplots**: Compared distributions of selected important features  
- **Scatter Plots (Pair Plot)**: Feature relationships colored by diagnosis  
- **Heatmap**: Correlation matrix of selected features  

### 4. Lazy Predict (LazyClassifier)
- Applied `LazyClassifier` to quickly benchmark multiple models  
- Compared accuracy and performance across classifiers  
- Selected **Top 4 models** for detailed evaluation  

### 5. Model Evaluation
- Trained and tested top-performing models  
- Generated **confusion matrices** and **classification reports**  
- Visualized confusion matrices with Seaborn heatmaps  

### 6. Threshold Tuning
- Used **SGDClassifier + CalibratedClassifierCV** to get probability estimates  
- Tuned decision thresholds (0.0 → 1.0)  
- Compared Accuracy, Precision, Recall, and F1-score across thresholds  
- Found best thresholds for each metric  
- Calculated **ROC-AUC score** independent of threshold  

---

## ⚙️ Tech Stack
- **Python**  
- **Libraries:** Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Sweetviz, LazyPredict  

---

## 📈 Results
- Original dataset yielded **high accuracy (~94–96%)** across classifiers  
- Lazy Predict helped identify top models (e.g., Random Forest, KNN, Logistic Regression)  
- Threshold tuning showed trade-offs between **Precision, Recall, and F1**  
- ROC-AUC score indicated strong classification performance  

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/Swapnil-Sa/breast-cancer-ml.git
   cd lazy-classification-wbcd

