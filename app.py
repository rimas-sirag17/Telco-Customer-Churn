# ===========================
# Streamlit App (DEPLOY VERSION)
# ===========================

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# ===========================
# Train Model (runs once)
# ===========================
@st.cache_resource
def train_model():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df.drop("customerID", axis=1, inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    df["Churn"] = df["Churn"].map({"Yes":1,"No":0})

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, scaler, X.columns.tolist()


model, scaler, features = train_model()

# ===========================
# UI
# ===========================
st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn")

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total = tenure * monthly

# ===========================
# Prepare Input
# ===========================
input_dict = {
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "gender_Male": 1 if gender == "Male" else 0,
    "Partner_Yes": 1 if partner == "Yes" else 0,
    "Dependents_Yes": 1 if dependents == "Yes" else 0,
    "PhoneService_Yes": 1 if phone_service == "Yes" else 0,
    "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
    "InternetService_No": 1 if internet_service == "No" else 0,
    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,
}

df_input = pd.DataFrame([input_dict])

# نخلي نفس ترتيب الأعمدة زي التدريب
df_input = df_input.reindex(columns=features, fill_value=0)

df_input_scaled = scaler.transform(df_input)

# ===========================
# Prediction
# ===========================
if st.button("🔮 Predict Churn"):
    prob = model.predict_proba(df_input_scaled)[0][1]

    if prob >= 0.4:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is not likely to churn")

    st.write(f"**Churn Probability:** {prob:.2%}")
