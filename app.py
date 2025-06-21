# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

st.set_page_config(page_title="SymptoDiag: Disease Diagnosis", layout="wide")

@st.cache_data
def load_data():
    diabetes_df = pd.read_csv("diabetes.csv")
    heart_df = pd.read_csv("heart.csv")
    return diabetes_df, heart_df

diabetes_df, heart_df = load_data()

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Home", "EDA", "Model Training", "Predict"])

if app_mode == "Home":
    st.title("SymptoDiag: Disease Diagnosis from Symptoms")
    st.image("logo.png", width=120)
    st.markdown("Predict your risk of **Diabetes** or **Heart Disease** based on key health indicators.")

elif app_mode == "EDA":
    st.title("Exploratory Data Analysis")
    dataset = st.selectbox("Choose Dataset", ["Diabetes", "Heart Disease"])

    if dataset == "Diabetes":
        st.subheader("Diabetes Dataset Preview")
        st.write(diabetes_df.head())
        st.write(diabetes_df.describe())
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(diabetes_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.subheader("Heart Disease Dataset Preview")
        st.write(heart_df.head())
        st.write(heart_df.describe())
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(heart_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

elif app_mode == "Model Training":
    st.title("Train Models")
    selected = st.selectbox("Select Disease", ["Diabetes", "Heart Disease"])

    if selected == "Diabetes":
        df = diabetes_df.copy()
        target = 'Outcome'
    else:
        df = heart_df.copy()
        target = 'target'

    df.replace(0, np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)

    X = df.drop(columns=[target])
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    st.subheader("Random Forest Results")
    st.write("Accuracy:", accuracy_score(y_test, rf_preds))
    st.write("ROC-AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, rf_preds))

    st.subheader("XGBoost Results")
    st.write("Accuracy:", accuracy_score(y_test, xgb_preds))
    st.write("ROC-AUC:", roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1]))
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, xgb_preds))

    st.subheader("SHAP Summary Plot for XGBoost")
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(X_test)
    fig, ax = plt.subplots()
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(fig)

elif app_mode == "Predict":
    st.title("Predict Disease Risk")
    disease = st.selectbox("Choose Disease", ["Diabetes", "Heart Disease"])

    def user_input_features_diabetes():
        pregnancies = st.number_input("Pregnancies", 0, 20)
        glucose = st.slider("Glucose", 50, 200, 120)
        bp = st.slider("Blood Pressure", 40, 122, 80)
        skin = st.slider("Skin Thickness", 7, 99, 20)
        insulin = st.slider("Insulin", 15, 846, 80)
        bmi = st.slider("BMI", 15.0, 50.0, 25.0)
        dpf = st.slider("Diabetes Pedigree Function", 0.1, 2.5, 0.5)
        age = st.slider("Age", 20, 100, 33)
        return pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
                            columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])

    def user_input_features_heart():
        age = st.slider("Age", 29, 77, 50)
        sex = st.selectbox("Sex", [0, 1])
        cp = st.selectbox("Chest Pain Type", [0,1,2,3])
        trestbps = st.slider("Resting BP", 90, 200, 120)
        chol = st.slider("Cholesterol", 100, 600, 240)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
        restecg = st.selectbox("Resting ECG", [0,1,2])
        thalach = st.slider("Max Heart Rate", 70, 210, 150)
        exang = st.selectbox("Exercise Induced Angina", [0,1])
        oldpeak = st.slider("ST depression", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope", [0,1,2])
        ca = st.selectbox("Number of major vessels", [0,1,2,3])
        thal = st.selectbox("Thalassemia", [0,1,2,3])
        return pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                            columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'])

    if disease == "Diabetes":
        input_df = user_input_features_diabetes()
        X = diabetes_df.drop(columns=['Outcome'])
        y = diabetes_df['Outcome']
    else:
        input_df = user_input_features_heart()
        X = heart_df.drop(columns=['target'])
        y = heart_df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    input_scaled = scaler.transform(input_df)

    model = RandomForestClassifier()
    model.fit(X_scaled, y)
    pred = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")
    if pred[0] == 1:
        st.error(f"High Risk Detected! Probability: {proba*100:.2f}%")
    else:
        st.success(f"Low Risk Detected. Probability: {proba*100:.2f}%")
