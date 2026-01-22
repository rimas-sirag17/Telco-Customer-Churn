# ===========================
# Streamlit App (WORKING VERSION)
# ===========================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn")

# -------------------------
# Load model & files
# -------------------------
try:
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("model_features.pkl")
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# -------------------------
# User Inputs (ONLY trained features)
# -------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total = tenure * monthly

# -------------------------
# Build input dataframe
# -------------------------
input_data = {
    "gender": 1 if gender == "Male" else 0,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": 1 if partner == "Yes" else 0,
    "Dependents": 1 if dependents == "Yes" else 0,
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}
features = joblib.load("model_features.pkl")
df_input = pd.DataFrame([input_data])
df_input = pd.get_dummies(df_input)
df_input = df_input.reindex(columns=features,fill_value=0)
df_input_scaled = scaler.transform(df_input)

if st.button("🔮 Predict Churn"):
    prob = model.predict_proba(df_input_scaled)[0][1]

    if prob >= 0.4:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is not likely to churn")

    st.write(f"**Churn Probability:** {prob:.2%}")

# save files
joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "model_features.pkl")

print("✅ الملفات اتولدت بنجاح!")
