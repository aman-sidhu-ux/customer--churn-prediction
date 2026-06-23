import streamlit as st
import joblib
import numpy as np


# Page setup
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📡",
    layout="wide"
)


# Load model
try:
    model = joblib.load("churn_model_final.pkl")

except Exception as e:
    st.error("Model loading failed")
    st.write(e)
    st.stop()



# Fixed decision threshold
threshold = 0.5


# Title
st.title("📡 Telecom Customer Churn Prediction")
st.write("ML Based Customer Retention System")


st.subheader("👤 Customer Details")


age = st.number_input(
    "Age",
    min_value=18,
    max_value=80,
    value=25
)


gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


monthly_income = st.number_input(
    "Monthly Income (INR)",
    min_value=10000,
    max_value=400000,
    value=30000
)


monthly_bill = st.number_input(
    "Monthly Bill",
    min_value=200,
    max_value=10000,
    value=500
)


internet_usage = st.number_input(
    "Internet Usage (GB)",
    min_value=0,
    max_value=50,
    value=10
)


call_minutes = st.number_input(
    "Call Minutes",
    min_value=0,
    max_value=700,
    value=200
)


contract = st.selectbox(
    "Contract Type (Year)",
    [0, 1, 2]
)


support = st.number_input(
    "Support Ticket (No. of complaints)",
    min_value=0,
    max_value=5,
    value=0
)



# Gender encoding
gender_value = 1 if gender == "Male" else 0



if st.button("🔍 Predict Churn"):


    input_data = np.array([
        age,
        gender_value,
        monthly_income,
        monthly_bill,
        internet_usage,
        call_minutes,
        contract,
        support
    ]).reshape(1, -1)


    try:

        probability = model.predict_proba(input_data)[0][1]


        st.divider()


        if probability >= threshold:

            st.error("⚠ HIGH CHURN RISK CUSTOMER")

            st.write(
                "Recommended Action: Contact customer and offer retention plan."
            )


        else:

            st.success("✅ CUSTOMER LIKELY TO STAY")

            st.write(
                "Customer has lower churn probability."
            )


        st.metric(
            "Churn Probability",
            f"{probability*100:.2f}%"
        )


        st.progress(float(probability))


    except Exception as e:

        st.error("Prediction failed")
        st.write(e)