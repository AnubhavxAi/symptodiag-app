# home.py
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
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def run_home():
    st.title("SymptoDiag: Disease Diagnosis from Symptoms")

    @st.cache_data
    def load_data():
        diabetes_df = pd.read_csv("diabetes.csv")
        heart_df = pd.read_csv("heart.csv")
        return diabetes_df, heart_df

    diabetes_df, heart_df = load_data()

    st.sidebar.title("Tools")
    app_mode = st.sidebar.radio("Choose Mode", ["Home", "EDA", "Model Training", "Predict"])

    if app_mode == "Home":
        st.image("logo.png", width=120)
        st.markdown("Predict your risk of **Diabetes** or **Heart Disease** based on key health indicators.")

    elif app_mode == "EDA":
        st.subheader("Exploratory Data Analysis")
        dataset = st.selectbox("Choose Dataset", ["Diabetes", "Heart Disease"])

        df = diabetes_df if dataset == "Diabetes" else heart_df
        st.write(df.head())
        st.write(df.describe())

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    elif app_mode == "Model Training":
        st.subheader("Train Machine Learning Models")
        selected = st.selectbox("Select Dataset", ["Diabetes", "Heart Disease"])
        df = diabetes_df.copy() if selected == "Diabetes" else heart_df.copy()
        target = 'Outcome' if selected == "Diabetes" else 'target'

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

        st.write("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
        st.write("XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))

        st.subheader("SHAP Explainability for XGBoost")
        explainer = shap.Explainer(xgb_model)
        shap_values = explainer(X_test)
        fig, ax = plt.subplots()
        shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(fig)

    elif app_mode == "Predict":
        st.subheader("Make a Prediction")
        disease = st.selectbox("Choose Disease", ["Diabetes", "Heart Disease"])

        def user_input_diabetes():
            return pd.DataFrame([[
                st.number_input("Pregnancies", 0, 20),
                st.slider("Glucose", 50, 200, 120),
                st.slider("Blood Pressure", 40, 122, 80),
                st.slider("Skin Thickness", 7, 99, 20),
                st.slider("Insulin", 15, 846, 80),
                st.slider("BMI", 15.0, 50.0, 25.0),
                st.slider("Diabetes Pedigree Function", 0.1, 2.5, 0.5),
                st.slider("Age", 20, 100, 33)
            ]], columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])

        def user_input_heart():
            return pd.DataFrame([[
                st.slider("Age", 29, 77, 50),
                st.selectbox("Sex", [0, 1]),
                st.selectbox("Chest Pain Type", [0,1,2,3]),
                st.slider("Resting BP", 90, 200, 120),
                st.slider("Cholesterol", 100, 600, 240),
                st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1]),
                st.selectbox("Resting ECG", [0,1,2]),
                st.slider("Max Heart Rate", 70, 210, 150),
                st.selectbox("Exercise Induced Angina", [0,1]),
                st.slider("ST depression", 0.0, 6.0, 1.0),
                st.selectbox("Slope", [0,1,2]),
                st.selectbox("Number of major vessels", [0,1,2,3]),
                st.selectbox("Thalassemia", [0,1,2,3])
            ]], columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'])

        if disease == "Diabetes":
            input_df = user_input_diabetes()
            X = diabetes_df.drop(columns=['Outcome'])
            y = diabetes_df['Outcome']
        else:
            input_df = user_input_heart()
            X = heart_df.drop(columns=['target'])
            y = heart_df['target']

        scaler = StandardScaler()
        model = RandomForestClassifier()
        model.fit(scaler.fit_transform(X), y)
        prediction = model.predict(scaler.transform(input_df))
        proba = model.predict_proba(scaler.transform(input_df))[0][1]

        if prediction[0] == 1:
            st.error(f"⚠️ High Risk Detected! Probability: {proba:.2%}")
        else:
            st.success(f"✅ Low Risk Detected. Probability: {proba:.2%}")
