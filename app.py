import streamlit as st
import pickle
import numpy as np
import pandas as pd

# =============================
# Load Trained Model and Scaler
# =============================

model = pickle.load(open('credit_risk_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# =============================
# Page Configuration
# =============================

st.set_page_config(
    page_title='Credit Risk Prediction System',
    page_icon='💳',
    layout='wide'
)

# =============================
# Custom CSS Styling
# =============================

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }

    .title {
        font-size:40px;
        font-weight:bold;
        color:#003366;
        text-align:center;
    }

    .subtext {
        font-size:18px;
        text-align:center;
        color:#444444;
    }

    .result-box {
        padding:20px;
        border-radius:10px;
        text-align:center;
        font-size:24px;
        font-weight:bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# Title Section
# =============================

st.markdown('<p class="title">Credit Risk Prediction System</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="subtext">Predict whether a borrower is likely to default on a loan using Machine Learning.</p>',
    unsafe_allow_html=True
)

st.write('---')

# =============================
# Input Section
# =============================
st.caption('Final Year Project — Credit Risk Prediction using Machine Learning')