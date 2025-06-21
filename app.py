# app.py
import streamlit as st
from home import run_home
from doctor_dashboard import run_doctor_dashboard

st.set_page_config(page_title="SymptoDiag", layout="wide")

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Choose a page:", ["Home & Prediction", "Doctor Dashboard"])

if page == "Home & Prediction":
    run_home()
elif page == "Doctor Dashboard":
    run_doctor_dashboard()
