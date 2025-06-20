# 🎓 Student Performance Prediction Using Machine Learning

This project predicts a student's math performance based on various academic and demographic factors using regression models. It demonstrates the complete lifecycle of a machine learning project — from data ingestion to model deployment using Flask.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Workflow Pipeline](#workflow-pipeline)
- [Model Performance](#model-performance)
- [How to Run](#how-to-run)
- [Web App Demo](#web-app-demo)
- [Screenshots](#screenshots)
- [Author](#author)

---

## 📘 Overview

This ML project uses a dataset containing students' gender, ethnicity, parental education, lunch type, test preparation status, reading score, and writing score to predict their **math score**.

📊 The goal: Train multiple regression models, tune them, and deploy the best one using Flask.

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Pandas, NumPy, Scikit-learn**
- **CatBoost, XGBoost**
- **Flask (Web Deployment)**
- **VS Code / Jupyter**
- **GridSearchCV for hyperparameter tuning**

---

## 🗂️ Project Structure

ML-Project/
├── artifacts/               # Stored models & transformed data
├── notebook/                # Raw dataset
├── src/                     # Source code
│   ├── components/          # Data ingestion, transformation, training
│   ├── pipeline/            # Prediction logic
│   ├── exception.py         # Custom exception handling
│   ├── logger.py            # Logging
│   └── utils.py             # Utility functions
├── app.py                   # Flask web app
├── templates/               # HTML templates
├── requirements.txt
└── README.md                # Project documentation

🔁 Workflow Pipeline
1.Data Ingestion
Load and split raw data (stud.csv) into train and test sets.

2.Data Transformation

Impute missing values

Encode categorical features

Scale numerical values

Save preprocessor.pkl

3.Model Training

Evaluate 8+ regression models

Tune hyperparameters with GridSearchCV

Save best model as model.pkl

4.Prediction

Load model & transformer

Take user input

Predict math score

✅ Model Performance
Best Model: Linear Regression

R² Score on Test Data: ~0.88

🚀 How to Run
🔧 Step 1: Install Requirements
pip install -r requirements.txt
📊 Step 2: Train the Model
python src/components/data_ingestion.py
🌐 Step 3: Run the Web App
python app.py
Then open: http://127.0.0.1:5000

🌍 Web App Demo
Users can fill a form with academic details, and the app will predict their math score in real-time.
🖼️ Screenshots
Home Page
![image alt](https://github.com/LAXMAN7795/Student-Performance-Prediction-Using-Machine-Learning/blob/da3a2e974b4eccd482d4ce71b75f777aef53f191/templates/HomePage.png)

Prediction Page
![image alt](https://github.com/LAXMAN7795/Student-Performance-Prediction-Using-Machine-Learning/blob/050c6184086dabe77d3362a42830f06b004f5266/templates/InputPage.png)

Result Page
![image alt](https://github.com/LAXMAN7795/Student-Performance-Prediction-Using-Machine-Learning/blob/b88af60d0c1868d411a5b3a211189c578838ded7/templates/PredictionPage.png)
👤 Author
Name: Laxman Gouda
Email: laxman.sg0104@gmail.com
GitHub: @LAXMAN7795
