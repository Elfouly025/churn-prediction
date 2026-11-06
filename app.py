import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Churn Predictor", page_icon="📉")

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.title("Customer Churn Predictor")
st.write("Enter a customer's details to estimate their risk of leaving.")

col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.slider("Monthly charges ($)", 18, 120, 70)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
with col2:
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.checkbox("Paperless billing", value=True)
    security = st.checkbox("Online security")
    support = st.checkbox("Tech support")

# Start with every column at 0, then fill in the user's choices
row = pd.DataFrame(0, index=[0], columns=columns)

def set_col(name, value=1):
    if name in row.columns:
        row[name] = value

set_col("tenure", tenure)
set_col("MonthlyCharges", monthly)
set_col("TotalCharges", tenure * monthly)

if contract != "Month-to-month":
    set_col(f"Contract_{contract}")

if internet == "No":
    set_col("InternetService_No")
    for service in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]:
        set_col(f"{service}_No internet service")
else:
    if internet == "Fiber optic":
        set_col("InternetService_Fiber optic")
    if security:
        set_col("OnlineSecurity_Yes")
    if support:
        set_col("TechSupport_Yes")

if payment == "Electronic check":
    set_col("PaymentMethod_Electronic check")
elif payment == "Mailed check":
    set_col("PaymentMethod_Mailed check")
elif payment == "Credit card (automatic)":
    set_col("PaymentMethod_Credit card (automatic)")

if paperless:
    set_col("PaperlessBilling_Yes")

score = model.predict_proba(row[columns])[0][1]

st.subheader("Result")
st.metric("Churn risk score", f"{score:.2f}")
st.progress(min(float(score), 1.0))

if score >= 0.6:
    st.error("High risk: contact this customer with a retention offer.")
elif score >= 0.4:
    st.warning("Medium risk: keep an eye on this customer.")
else:
    st.success("Low risk.")

st.caption("The score ranks customers by risk. It is not an exact probability, "
           "because the model was trained with balanced class weights.")