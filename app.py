from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
import joblib

# -------------------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Twitter Sentiment Intelligence",
    page_icon="💼",
    layout="wide",
)

st.title("Twitter Sentiment Intelligence")
st.caption("Streamlit front-end for the Deloitte-ready Twitter Sentiment Intelligence dashboard.")

try:
    # -------------------------------------------------------------------------
    # Download model from Hugging Face Hub (no local artifacts needed)
    # -------------------------------------------------------------------------
    MODEL_REPO = "vishnu-coder/twitter-sentiment-model"
    MODEL_FILENAME = "sentiment_pipeline.joblib"

    @st.cache_resource(show_spinner=False)
    def load_model():
        """Download and load the trained model from Hugging Face Hub."""
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        pipeline = joblib.load(model_path)
        return pipeline

    pipeline = load_model()

    # -------------------------------------------------------------------------
    # Helper function for predictions
    # -------------------------------------------------------------------------
    def predict_sentiment(text: str) -> tuple[str, Dict[str, float]]:
        """Predict sentiment and confidence scores."""
        probs = pipeline.predict_proba([text])[0]
        classes = pipeline.classes_
        label = classes[probs.argmax()]
        probabilities = dict(zip(classes, probs))
        return label, probabilities

    def format_probabilities(probabilities: Dict[str, float]) -> pd.DataFrame:
        """Convert probabilities to styled DataFrame."""
        return (
            pd.DataFrame.from_dict(probabilities, orient="index", columns=["confidence"])
            .sort_values("confidence", ascending=False)
            .style.format({"confidence": "{:.2%}"})
        )

    # -------------------------------------------------------------------------
    # Streamlit UI
    # -------------------------------------------------------------------------
    def main():
        st.sidebar.header("📊 Model Snapshot")
        st.sidebar.write("**Source:**", MODEL_REPO)
        st.sidebar.success("✅ Loaded model from Hugging Face Hub")

        st.subheader("🔮 Real-Time Sentiment Analysis")
        user_input = st.text_area("Enter a tweet or comment:", height=150)

        if st.button("Analyze", type="primary"):
            if not user_input.strip():
                st.warning("⚠️ Please enter text to analyze.")
            else:
                label, probabilities = predict_sentiment(user_input)
                st.success(f"Predicted Sentiment: **{label.title()}**")
                st.dataframe(format_probabilities(probabilities), use_container_width=True)

        st.markdown("---")
        st.caption("© 2025 Deloitte-aligned Sentiment Analytics Accelerator")

    if __name__ == "__main__":
        main()

except Exception as e:
    st.error(f"Startup failed: {e}")