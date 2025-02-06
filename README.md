# Breast Cancer Prediction - Exploratory Data Analysis (EDA)

## 📌 Project Overview

This project focuses on Exploratory Data Analysis (EDA) of the Breast Cancer Prediction Dataset to identify patterns and insights in the data. The dataset contains various biometric measurements of tumors and a diagnosis label (Benign or Malignant). The analysis aims to help understand the dataset's structure, preprocess the data, and explore potential predictive patterns.

## 📊 Data Source

This dataset originates from the University of Wisconsin Hospitals, Madison and was created by Dr. William H. Wolberg. The dataset is publicly available on Kaggle:

🔗 Breast Cancer Prediction Dataset - Kaggle : https://www.kaggle.com/datasets/merishnasuwal/breast-cancer-prediction-dataset

⚠️ Note: The dataset is NOT included in this repository. Please download it from the Kaggle link above before running the analysis.

## 🔬 Methods & Analysis

The following techniques are applied to analyze and preprocess the dataset:

### 📊 Step 1: Exploratory Data Analysis (EDA)

- Reading the dataset (pandas)

- Checking for missing values (none in this case)

- Handling categorical data (mapping 0 = Malignant, 1 = Benign)

- Understanding feature distributions

- 📍 **Notebook:** [01_EDA.ipynb](notebooks/01_EDA.ipynb)

### 📊 Step 2: Comparing Dimensionality Reduction Techniques

| **Method** | **Components** | **Used For** |
|------------|--------------|--------------|
| PCA | 2D | Train-Test Splitting & Visualization |
| MDS | 2D,3D | Visualization |
| t-SNE | 2D,3D | Visualization |
| UMAP | 2D,3D | Visualization |

- 📍 **Notebook:** [02_All_Dim_Reduction.ipynb](notebooks/02_All_Dim_Reduction.ipynb)
- **Includes:** Scatter plots & visual comparisons of all techniques.

### 📊 Step 3: Dimensionality reduction - Model Comparison

- The **best dimensionality reduction techniques** are selected based on:
  - **Silhouette Score**
  - **Stress (for MDS)**
  - **Correlation with original distances**
  - **Classification Accuracy (Random Forest)**
- The final **Random Forest classification model** is trained on:
  - **PCA (2D) → Performed Poorly**
  - **MDS (3D) → Best Performance**
  - **t-SNE (2D) → Near-optimal Performance**
  - **UMAP (3D) → Strong Alternative**

### **📊 Final Accuracy Results:**
| **Method** | **Classification Accuracy (%)** |
|------------|-------------------------------|
| **PCA (2D)** | **54.4%** ❌ |
| **t-SNE (2D)** | **92.9%** ✅ |
| **UMAP (3D)** | **91.8%** ✅ |
| **MDS (3D)** | **93.5%** ✅ |

📍 **Notebook:** [03_Best_Models_DR.ipynb](notebooks/03_Best_Models_DR.ipynb)


## 📌 Best Dimensionality Reduction: MDS (3D → 2D)
Below is the final decision boundary for MDS (3D reduced to 2D), which gave the highest classification accuracy.

![MDS Decision Boundary](images/MDS_3D_to_2D_Decision_Boundary.png)

---

## **🛠 How to Run This Project**

### **1️⃣ Install Requirements**

```bash
pip install -r requirements.txt
```



1. Download Dataset

- Go to Kaggle

- Download the dataset and place it in the project directory

2. Run the Analysis

- Ensure you have the necessary dependencies installed:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Load dataset
data = pd.read_csv("breast_cancer.csv")

# Display first 5 rows
print(data.head())
```

## 📜 License

This repository is licensed under the Apache 2 License. You are free to use and modify the code but must provide proper attribution.

## 👨‍💻 Author & Acknowledgments

This project was developed by Henry Serpa, as part of practicing Data Science concepts.

## 🙏 Special thanks to:

- Dr. William H. Wolberg (Dataset Creator)

- Kaggle Community for hosting the dataset

- Open-source contributors of Python libraries (pandas, seaborn, matplotlib, etc.)

## ✅ Future Updates

📌 Classification models

📌 Machine Learning Models (Logistic Regression, Random Forest, etc.)

Feel free to ⭐ star this repository if you found it helpful!


