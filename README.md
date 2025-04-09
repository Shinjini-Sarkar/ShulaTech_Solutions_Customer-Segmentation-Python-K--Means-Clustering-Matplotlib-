# 🧠 Customer Segmentation using K-Means Clustering

This project performs **Customer Segmentation** using **Unsupervised Machine Learning** techniques. The main goal is to group customers into distinct clusters based on their purchasing behavior and demographics.

## 📌 Technologies & Concepts Used

- Python
- K-Means Clustering (Unsupervised Learning)
- PCA (Principal Component Analysis)
- Matplotlib & Seaborn (Visualization)
- Scikit-learn
- Pandas & NumPy

---

## 📂 Dataset

**Dataset used:** [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial)

**Features:**
- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1–100)

---

## 🚀 Project Flow

1. **Import Libraries**  
   Load essential libraries like `pandas`, `matplotlib`, `seaborn`, `sklearn`.

2. **Load Dataset**  
   Load data from CSV and explore it using `.head()` and `.info()`.

3. **Preprocessing**
   - Drop unnecessary columns
   - Encode categorical features (e.g., Gender) using `map()`
   - Scale the data using `StandardScaler`

4. **Determine Optimal K**
   - Use the **Elbow Method** to find the best value for `k` using **WCSS**

5. **Apply K-Means Clustering**
   - Fit the model using `KMeans`
   - Predict and label the clusters

6. **Dimensionality Reduction (PCA)**
   - Apply **PCA** to reduce dimensions (2D/3D) for visualization
   - Transform both data points and centroids

7. **Visualization**
   - Create 2D and 3D plots using `matplotlib` and `seaborn`
   - Color the clusters
   - Highlight cluster centroids with stars
   - Add legends and axis labels

---

## 📊 Visual Output

### 2D Cluster Visualization  
- Using PCA (2 Components)  
- Clear separation of clusters with color-coded points  

### 3D Cluster Visualization  
- Interactive 3D view of clusters  
- Centroids marked as black stars  
- Axis labels: Component 1, 2, 3  

---

## 📁 Directory Structure

```bash
📦Customer-Segmentation
 ┣ 📜main.py
 ┣ 📊Mall_Customers.csv
 ┣ 📈images/
 ┃ ┣ 📷2d_customer_segments.png
 ┃ ┣ 📷3d_customer_segments.png
 ┣ 📄README.md
