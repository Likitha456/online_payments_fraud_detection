# 🛡️ Online Payments Fraud Detection System
## 📌 Project Overview
This project presents a **Machine Learning–based Online Payments Fraud Detection System** designed to classify transactions as **Fraudulent** or **Not Fraudulent**.
The system leverages transaction attributes, a trained ML model, and a **Flask web application** to provide real-time fraud predictions.
---
## 🎯 Objectives
* Analyze online payment data to identify fraud patterns
* Perform data preprocessing and feature engineering
* Train and evaluate machine learning models
* Deploy the selected model using Flask
* Provide a simple web interface for fraud prediction
---
## 🧠 Technologies Used
**Programming & Tools**
* Python 3.x
* VS Code
* Git & GitHub
**Libraries**
* pandas, numpy
* matplotlib, seaborn
* scikit-learn
* Flask
---
## 📊 Dataset Description
* **Source:** Kaggle – Online Payments Fraud Dataset
* **Size:** ~6.3 million records
* **Target Column:** `isFraud`
**Note:**
Due to the large size of the dataset, it is **not included in this repository**.
Exploratory Data Analysis (EDA) and model training were performed locally.
---
## 📈 Exploratory Data Analysis (EDA)
EDA was carried out to understand transaction behavior and fraud patterns.
**Key Insights**
* The dataset is highly imbalanced
* Fraud occurs predominantly in **CASH_OUT** and **TRANSFER** transactions
* High-value transactions have a higher likelihood of fraud
**EDA Implementation**
* EDA code is available in the `EDA/` folder
* Visualizations were generated locally
* Screenshots of key plots are uploaded separately for reference
---
## ⚙️ Data Preprocessing
* Removed irrelevant ID columns (`nameOrig`, `nameDest`)
* One-hot encoded the transaction `type` feature
* Separated features (`X`) and target (`y`)
* Ensured feature consistency for Flask-based prediction
---
## 🤖 Model Training & Selection
**Models Analyzed**
* Decision Tree Classifier
* Random Forest Classifier
* Support Vector Machine (analysis only)
* Extra Trees Classifier (analysis only)
* XGBoost (analysis only)
**Final Model Selected:** **Decision Tree Classifier**
**Reason for Selection**
* High fraud recall
* Fast prediction time
* Low computational cost
* Suitable for real-time deployment
---
## 🌐 Flask Web Application
**Features**
* Home page with project overview
* Transaction input form
* Result page displaying fraud prediction
**Routes**
* `/` → Home
* `/predict` → Transaction input
* `/result` → Prediction output
---
## 🧪 How to Run the Project Locally
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Likitha456/online_payments_fraud_detection.git
cd online_payments_fraud_detection
```
### 2️⃣ Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn flask
```
### 3️⃣ Run the Application
```bash
python flask/app.py
```
Open your browser and visit:
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)
---
## 🔄 Prediction Flow
* User enters transaction details
* Input is transformed into model-compatible features
* The model predicts fraud status
**Output**
* Fraud Transaction
* Not a Fraud Transaction
---
## 👥 Team Contributions
* **Sama Pavithra** – UI/UX design and frontend development
* **Velakaturi Lekhya Sreeya** – Exploratory Data Analysis (EDA), dataset analysis, model training support
* **Likitha Puttareddy** – Model development, Flask backend integration, deployment
---
## 📌 Version Control
* GitHub used for collaboration and final submission
* Dataset excluded due to size constraints
---
