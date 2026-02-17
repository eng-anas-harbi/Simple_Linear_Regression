# Salary Prediction using Simple Linear Regression

## 📌 Project Overview

This project implements **Simple Linear Regression** to predict salary based on years of experience.

The model is built using `scikit-learn` and visualized using `matplotlib`.  
The objective is to demonstrate a fundamental supervised learning workflow in machine learning.

---

## 📂 Dataset

File used: `Salary_Data.csv`

The dataset contains two columns:

- **Years of Experience** (Independent Variable - X)
- **Salary** (Dependent Variable - y)

The goal is to model the linear relationship between experience and salary.

---

## 🧠 Machine Learning Workflow

The project follows these steps:

1. Load the dataset using `pandas`
2. Separate features (X) and target variable (y)
3. Split the dataset into:
   - 80% Training Set
   - 20% Test Set
4. Create a Linear Regression model
5. Train the model on the training data
6. Predict salaries using the test data
7. Visualize the results

---

## 📊 Visualization

- **Red dots** → Actual data points  
- **Blue line** → Regression line (trained model)

The plot illustrates the linear relationship between years of experience and salary.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
