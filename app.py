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
    page_title="Credit Risk Prediction System",
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
# Input Section
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

    previous_loan_defaults = st.selectbox(
        "Previous Loan Defaults",
        ["No", "Yes"]
    )

# =========================
# Encode Previous Defaults
# =========================

if previous_loan_defaults == "Yes":
    default_history = 1
else:
    default_history = 0

# =========================
# Prediction Button
# =========================

if st.button("Predict Credit Risk"):

    # =========================
    # Model Input
    # =========================

    input_data = np.array([[
        person_age,
        person_income,
        person_emp_length,
        loan_amnt,
        loan_int_rate,
        loan_percent_income,
        cb_person_cred_hist_length,
        default_history,
        0,
        0,
        0
    ]])

    # =========================
    # Scale Data
    # =========================

    scaled_data = scaler.transform(input_data)

    # =========================
    # ML Prediction
    # =========================

    probability = model.predict_proba(scaled_data)

    ml_repay_probability = probability[0][0] * 100
    ml_default_probability = probability[0][1] * 100

    # =========================
    # Balanced Manual Risk Logic
    # =========================

    risk_score = 0

    # Income Risk
    if person_income < 15000:
        risk_score += 25
    elif person_income < 35000:
        risk_score += 12

    # Loan Amount Risk
    if loan_amnt > 40000:
        risk_score += 20
    elif loan_amnt > 20000:
        risk_score += 10

    # Interest Rate Risk
    if loan_int_rate > 22:
        risk_score += 18
    elif loan_int_rate > 14:
        risk_score += 10

    # Loan Percent Income Risk
    if loan_percent_income > 0.7:
        risk_score += 20
    elif loan_percent_income > 0.4:
        risk_score += 10

    # Credit History Risk
    if cb_person_cred_hist_length < 2:
        risk_score += 15
    elif cb_person_cred_hist_length < 5:
        risk_score += 8

    # Employment Risk
    if person_emp_length < 1:
        risk_score += 10
    elif person_emp_length < 3:
        risk_score += 5

    # Previous Loan Default Risk
    if default_history == 1:
        risk_score += 20

    # =========================
    # Final Probability
    # =========================

    default_probability = (
        (ml_default_probability * 0.4) +
        (risk_score * 0.6)
    )

    if default_probability > 100:
        default_probability = 100

    repay_probability = 100 - default_probability

    # =========================
    # Risk Classification
    # =========================

    if default_probability >= 70:

        risk_level = "High"

    elif default_probability >= 40:

        risk_level = "Medium"

    else:

        risk_level = "Low"

    st.write("---")

    # =========================
    # Prediction Result
    # =========================

    st.subheader("Prediction Result")

    if risk_level == "High":

        st.error(
            "⚠️ High Risk Borrower — Likely to Default"
        )

    elif risk_level == "Medium":

        st.warning(
            "⚠️ Medium Risk Borrower — Needs Careful Review"
        )

    else:

        st.success(
            "✅ Low Risk Borrower — Likely to Repay Loan"
        )

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
    # Loan Recommendation
    # =========================

    st.subheader("Loan Recommendation")

    if risk_level == "High":

        st.error(
            "❌ Recommendation: Reject Loan"
        )

    elif risk_level == "Medium":

        st.warning(
            "⚠️ Recommendation: Review Carefully"
        )

    else:

        st.success(
            "✅ Recommendation: Approve Loan"
        )

# =========================
# Footer
# =========================

st.write("---")

st.caption(
    "Final Year Project — Credit Risk Prediction using Machine Learning"
)
