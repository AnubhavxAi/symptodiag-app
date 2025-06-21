# doctor_dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_doctor_dashboard():
    st.title("👩‍⚕️ Doctor Dashboard: Patient Insights")

    uploaded_file = st.file_uploader("📤 Upload Patient History (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        st.subheader("📋 Raw Data Preview")
        st.dataframe(df.head())

        if 'disease' not in df.columns or 'prediction' not in df.columns:
            st.error("❌ CSV must contain 'disease' and 'prediction' columns (e.g., diabetes, heart, 0/1 prediction).")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📈 Disease Distribution")
                fig, ax = plt.subplots()
                sns.countplot(data=df, x="disease", palette="Set2", ax=ax)
                st.pyplot(fig)

            with col2:
                st.subheader("📊 Prediction Outcomes")
                fig, ax = plt.subplots()
                sns.countplot(data=df, x="prediction", hue="disease", palette="Set1", ax=ax)
                ax.set_xticklabels(["Negative", "Positive"])
                st.pyplot(fig)

            st.subheader("🧠 Filter by Disease")
            disease_filter = st.multiselect("Choose disease type(s)", df["disease"].unique())

            if disease_filter:
                filtered_df = df[df["disease"].isin(disease_filter)]
                st.dataframe(filtered_df)
                st.download_button("⬇️ Download Filtered Data", filtered_df.to_csv(index=False), file_name="filtered_patients.csv")

    else:
        st.info("💡 Upload a CSV containing previous patient predictions to view analytics. Columns must include: disease, prediction, and input features.")
