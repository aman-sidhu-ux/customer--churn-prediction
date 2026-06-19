import streamlit as st
import numpy as np
import joblib

model = joblib.load("churn_model_final.pkl")

st.title("📊 Customer Churn Prediction App")

age = st.number_input("Age", 18, 100, 30)
monthly_income = st.number_input("Monthly Income", 0, 200000, 30000)
monthly_bill = st.number_input("Monthly Bill", 0, 10000, 1000)
internet_usage_gb = st.number_input("Internet Usage (GB)", 0, 500, 50)
call_minutes = st.number_input("Call Minutes", 0, 2000, 200)
support_tickets = st.number_input("Support Tickets", 0, 20, 2)
customer_feedback = st.number_input("Customer Feedback (1-5)", 1, 5, 3)

if st.button("Predict Churn"):

    input_data = np.array([[
        age,
        monthly_income,
        monthly_bill,
        internet_usage_gb,
        call_minutes,
        support_tickets,
        customer_feedback
    ]])

    # ✅ FIRST predict
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.write("📊 Churn Probability:", f"{prob:.2%}")

    # ✅ THEN risk logic
    if prob < 0.4:
        st.success("🟢 Low Risk - Customer will NOT churn")
    elif prob < 0.6:
        st.warning("🟡 Medium Risk - Uncertain customer")
    else:
        st.error("🔴 High Risk - Customer WILL churn")

    # ✅ Final result
    if prediction == 1:
        st.error(f"⚠️ YES - Customer WILL CHURN ({prob:.0%})")
    else:
        st.success(f"✅ NO - Customer will NOT CHURN ({(1-prob):.0%})")