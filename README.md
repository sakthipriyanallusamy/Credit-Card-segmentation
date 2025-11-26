# 💳 Credit Card Customer Segmentation App  
### *Interactive Machine Learning App Built with Streamlit*

This project is an interactive Streamlit application that performs **customer segmentation** using machine learning techniques. It allows users to upload credit-card datasets, analyze spending behaviors, and visualize clusters created using **K-Means**, **Silhouette Scores**, **Elbow Method**, and **PCA**.

---

## 🌟 Features

### 🔹 **1. Upload or Use Sample Dataset**
- Upload your own CSV file  
- Or use the included `CC_General.csv` dataset  

### 🔹 **2. Automated & Manual Clustering**
- Automatic K selection using **Silhouette Score**
- Manual selection option for choosing number of clusters (k)

### 🔹 **3. Interactive Visualizations**
The app generates:
- 📊 **Correlation Heatmap**
- 📈 **Elbow Curve**
- 🧮 **Silhouette Score Plot**
- 🎨 **PCA Scatter Plot** (2D visualization of clusters)
- 📋 **Cluster Profiles Table**
- 📥 **Downloadable Cluster Results (CSV)**

### 🔹 **4. Data Cleaning & Feature Scaling**
- Removes missing values using mean imputation  
- Converts numeric columns  
- Drops duplicates  
- Scales features using **StandardScaler**

---

## 🧠 Machine Learning Pipeline

1. **Load →**
2. **Clean →**
3. **Scale →**
4. **Find Best K →**
5. **Run K-Means →**
6. **Visualize Clusters →**
7. **Download Results**

---

## 📂 Project Structure

