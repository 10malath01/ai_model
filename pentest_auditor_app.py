import streamlit as st
import pandas as pd
from preprocess import preprocess_data
from compliance import compliance_score
from model import train_model, load_model

st.set_page_config(page_title="AI Compliance Auditor", layout="wide")

st.title("🔐 AI Cyber Threat Compliance Auditor")

st.markdown("Upload a cybersecurity dataset to train and evaluate compliance using AI.")

uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Dataset")
    st.dataframe(df.head())

    # Preprocess
    df = preprocess_data(df)

    # Apply compliance formula
    df["compliance_score"] = df.apply(compliance_score, axis=1)

    st.subheader("📈 Compliance Scores (Rule-Based)")
    st.dataframe(df[["text", "compliance_score"]].head())

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧠 Train AI Model"):
            with st.spinner("Training model..."):
                train_model(df)
            st.success("✅ Model trained and saved!")

    with col2:
        if st.button("⚡ Run AI Assessment"):
            try:
                model = load_model()

                df["ai_score"] = model.predict(df["text"])

                # Hybrid scoring
                df["final_score"] = (
                    0.6 * df["ai_score"] +
                    0.4 * df["compliance_score"]
                )

                st.subheader("🎯 Final AI Compliance Scores")
                st.dataframe(df[["text", "final_score"]])

                st.subheader("📊 Score Distribution")
                st.bar_chart(df["final_score"])

            except:
                st.error("❌ Train the model first!")

    # Download results
    if "final_score" in df.columns:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results",
            csv,
            "compliance_results.csv",
            "text/csv"
        )
