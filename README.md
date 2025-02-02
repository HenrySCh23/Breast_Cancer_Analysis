# Breast Cancer Prediction - Exploratory Data Analysis (EDA)

## 📌 Project Overview

This project focuses on Exploratory Data Analysis (EDA) of the Breast Cancer Prediction Dataset to identify patterns and insights in the data. The dataset contains various biometric measurements of tumors and a diagnosis label (Benign or Malignant). The analysis aims to help understand the dataset's structure, preprocess the data, and explore potential predictive patterns.

## 📊 Data Source

This dataset originates from the University of Wisconsin Hospitals, Madison and was created by Dr. William H. Wolberg. The dataset is publicly available on Kaggle:

🔗 Breast Cancer Prediction Dataset - Kaggle : https://www.kaggle.com/datasets/merishnasuwal/breast-cancer-prediction-dataset

⚠️ Note: The dataset is NOT included in this repository. Please download it from the Kaggle link above before running the analysis.

## 🔬 Methods & Analysis

The following techniques are applied to analyze and preprocess the dataset:

### 1️⃣ Data Exploration & Cleaning

- Reading the dataset (pandas)

- Checking for missing values (none in this case)

- Handling categorical data (mapping 0 = Malignant, 1 = Benign)

- Understanding feature distributions

### 2️⃣ Exploratory Data Analysis (EDA)

📈 Visualizations Used:

- Bar Plots: Diagnosis distribution

- Histograms: Feature distributions

- Box Plots: Outlier detection

- Grouped Bar Plots: Comparing features across diagnoses

- Heatmaps: Feature correlation analysis

- Cluster Maps: Hierarchical relationships in data

- Pair Plots: Feature interaction visualization

### 3️⃣ Feature Engineering & Preprocessing (Upcoming)

- Standardization (Scaling Features)

- Dimensionality Reduction (PCA - Principal Component Analysis)

## 🚀 How to Use

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

📌 Feature Scaling

📌 Dimensionality Reduction (PCA)

📌 Machine Learning Models (Logistic Regression, Random Forest, etc.)

Feel free to ⭐ star this repository if you found it helpful!


