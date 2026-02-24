import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("fraud_model.pkl")
expected_features = joblib.load("model_features.pkl")

st.title("💳 AI Fraud Detection System")

# Inputs
step = st.number_input("Step", min_value=1, value=1)
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=5000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=4000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=1000.0)
isFlaggedFraud = st.selectbox("Is Flagged Fraud?", [0, 1])
txn_type = st.selectbox("Transaction Type", ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])

# Prepare input
input_data = {
    "step": step,
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest,
    "isFlaggedFraud": isFlaggedFraud
}

# Handle one-hot encoded type columns
for col in expected_features:
    if col.startswith("type_"):
        input_data[col] = 0

if f"type_{txn_type}" in expected_features:
    input_data[f"type_{txn_type}"] = 1

input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=expected_features, fill_value=0)

# Predict
if st.button("Check Fraud"):
    prob = model.predict_proba(input_df)[0][1]
    result = "Fraud" if prob >= 0.5 else "Not Fraud"

    if result == "Fraud":
        st.error(f"⚠️ Fraud Detected! Probability: {prob:.4f}")
    else:
        st.success(f"✅ Not Fraud. Probability: {prob:.4f}")

    st.progress(float(prob))