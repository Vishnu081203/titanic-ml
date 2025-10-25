# 🚢 Titanic Survival Prediction (AI & ML Placement Project)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)

### 🔗 Live App:
👉 [**Click to Open App**](https://titanic-ml-3zkhevafpkumwqbuhsykdd.streamlit.app/)

### 💻 GitHub Repository:
👉 [**https://github.com/Vishnu081203/titanic-ml**](https://github.com/Vishnu081203/titanic-ml)

---

## 🧠 Project Overview

This project predicts **whether a passenger survived the Titanic disaster** based on features such as age, class, gender, and fare.

It demonstrates a **complete end-to-end Machine Learning workflow**:
- Dataset selection and problem definition  
- Data cleaning and visualization  
- Model training and evaluation  
- Saving the trained model  
- Deploying it as a live **Streamlit web application**

---

## 🧩 Dataset

**Source:** [Kaggle – Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)

| Feature | Description |
|----------|-------------|
| PassengerId | Unique ID |
| Survived | 0 = No, 1 = Yes |
| Pclass | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| Sex | Gender |
| Age | Age in years |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Fare | Passenger fare |
| Embarked | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

**Target Variable:** `Survived`

---

## 🧹 Data Cleaning & EDA (src/eda_and_cleaning.py)

- Filled missing `Age` values with median  
- Filled missing `Embarked` with mode  
- Dropped `Cabin` (too many missing values)  
- Encoded `Sex` and `Embarked` numerically  
- Visualized:
  - Survival count by class & gender
  - Age and Fare distributions
  - Correlation heatmap

Cleaned dataset saved as:

---

## 🤖 Model Building (src/train_model.py)

### Algorithms Tested:
1. Logistic Regression  
2. Random Forest Classifier ✅ (Best performing)

### Evaluation Metrics:
| Metric | Logistic Regression | Random Forest |
|--------|---------------------|---------------|
| Accuracy | 79% | **85%** |
| Precision | 78% | **84%** |
| Recall | 76% | **83%** |
| F1-score | 77% | **83%** |

Saved best model:

---

## 🌐 Deployment (src/streamlit_app.py)

Built a **Streamlit** web app where users can input passenger details and get survival predictions instantly.

**Features:**
- Input fields for all relevant features (Pclass, Sex, Age, etc.)
- Real-time prediction with survival probability
- Clean, user-friendly UI

---

## ⚙️ Installation Guide (Run Locally)

'''bash
# Clone the repo
git clone https://github.com/Vishnu081203/titanic-ml.git
cd titanic-ml

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run src/streamlit_app.py
---

Then open http://localhost:8501

☁️ Deployment

Deployed live on Streamlit Cloud directly from GitHub.

Steps:

Go to https://share.streamlit.io

Connect GitHub → Vishnu081203/titanic-ml

Branch → main

File path → src/streamlit_app.py

Click Deploy

📊 Tools & Libraries

Python 3.10

Pandas, NumPy – data manipulation

Matplotlib, Seaborn – visualization

Scikit-learn – machine learning

Joblib – model serialization

Streamlit – deployment UI

🧾 Project Structure

titanic-ml/
│
├── data/
│   ├── train.csv
│   └── cleaned_titanic.csv
│
├── model/
│   └── model.joblib
│
├── src/
│   ├── eda_and_cleaning.py
│   ├── train_model.py
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md

🎯 Key Learnings

Data preprocessing and EDA are critical before modeling

Random Forest performed best due to handling mixed data well

Learned to deploy ML models with Streamlit Cloud

Full-cycle ML project development from raw data to live app

👨‍💻 Author

Vishnu Vardhan
AI & ML Placement Project — Titanic Survival Prediction
📧 Email: vishnu40500@gmail.com
🌐 GitHub: Vishnu081203

⭐ Don’t forget to star this repo if you found it useful!

---

## 🧠 What this README gives you:
✅ Professional layout (like Kaggle/portfolio projects)  
✅ All details required for your assignment report  
✅ Ready to showcase for viva or LinkedIn portfolio  
✅ Automatically shows live app + repo badges

---
