import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model/model.joblib")

model = load_model()

st.title("🚢 Titanic Survival Prediction App")
st.write("Predict whether a passenger survived the Titanic disaster.")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Parents/Children Aboard", 0, 6, 0)
fare = st.slider("Ticket Fare (£)", 0.0, 500.0, 32.2)
embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

sex = 0 if sex == "male" else 1
embarked = {"C": 0, "Q": 1, "S": 2}[embarked]

X = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

if st.button("Predict Survival"):
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    if prediction == 1:
        st.success(f"✅ The passenger survived! (Probability: {proba:.2f})")
    else:
        st.error(f"❌ The passenger did not survive. (Probability: {proba:.2f})")
