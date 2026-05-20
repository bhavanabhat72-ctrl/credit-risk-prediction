import streamlit as st
import pickle
import numpy as np

# =========================
# Load Model and Scaler
# =========================

model = pickle.load(open('credit_risk_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="wide"
)

# =========================
# Title Section
# =========================

st.title("💳 Credit Risk Prediction System")

st.write(
    "Predict whether a borrower is likely to default on a loan using Machine Learning."
)

st.write("---")

# =========================
# Borrower Input Section
# =========================

st.header("Enter Borrower Details")

col1, col2 = st.columns(2)

with col1:

    person_age = st.number_input(
        "Person Age",
        min_value=18,
        max_value=100,
        value=25
    )

    person_income = st.number_input(
        "Annual Income",
        min_value=0,
        value=50000
    )

    person_emp_length = st.number_input(
        "Employment Length (Years)",
        min_value=0,
        value=5
    )

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=0,
        value=10000
    )

with col2:

    loan_int_rate = st.number_input(
        "Interest Rate",
        min_value=0.0,
        value=12.5
    )

    loan_percent_income = st.number_input(
        "Loan Percent Income",
        min_value=0.0,
        value=0.2
    )

    cb_person_cred_hist_length = st.number_input(
        "Credit History Length",
        min_value=0,
        value=5
    )

# =========================
# Prediction Button
# =========================

if st.button("Predict Credit Risk"):

    input_data = np.array([[
    person_age,
    person_income,
    person_emp_length,
    loan_amnt,
    loan_int_rate,
    loan_percent_income,
    cb_person_cred_hist_length,
    0,
    0,
    0,
    0
]])
    # Scale input data
    scaled_data = scaler.transform(input_data)

    # Prediction
    probability = model.predict_proba(scaled_data)

default_probability = probability[0][1] * 100

if default_probability > 40:
    prediction = [1]
else:
    prediction = [0]

    # Probability
    probability = model.predict_proba(scaled_data)

    repay_probability = probability[0][0] * 100
    default_probability = probability[0][1] * 100

    st.write("---")

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.error("⚠️ High Risk Borrower — Likely to Default")

    else:

        st.success("✅ Low Risk Borrower — Likely to Repay Loan")

    # =========================
    # Probability Section
    # =========================

    st.subheader("Prediction Probability")

    col3, col4 = st.columns(2)

    with col3:
        st.metric(
            label="Repayment Probability",
            value=f"{repay_probability:.2f}%"
        )

    with col4:
        st.metric(
            label="Default Probability",
            value=f"{default_probability:.2f}%"
        )

    # =========================
    # Risk Meter
    # =========================

    st.subheader("Risk Meter")

    st.progress(int(default_probability))

    # =========================
    # Loan Decision
    # =========================

    st.subheader("Loan Recommendation")

    if default_probability > 60:

        st.error("❌ Recommendation: Reject Loan")

    elif default_probability > 40:

        st.warning("⚠️ Recommendation: Review Carefully")

    else:

        st.success("✅ Recommendation: Approve Loan")

# =========================
# Footer
# =========================

st.write("---")

st.caption(
    "Final Year Project — Credit Risk Prediction using Machine Learning"
)
