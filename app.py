from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict
import pandas as pd
import streamlit as st

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
    # Path setup (✅ Fixed for root-level app.py)
    # -------------------------------------------------------------------------
    ROOT = Path(__file__).resolve().parents[0]
    SRC_PATH = ROOT / "src"
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))

    # Import from src/twitter_sentiment/
    from twitter_sentiment.config import load_config
    from twitter_sentiment.predictor import load_artifacts, predict_with_threshold

    # -------------------------------------------------------------------------
    # Cached dependencies
    # -------------------------------------------------------------------------
    @st.cache_resource(show_spinner=False)
    def _load_dependencies():
        """Load configuration, trained pipeline, and metrics from artifacts."""
        config = load_config()
        pipeline, metrics = load_artifacts(config)
        return config, pipeline, metrics

    # -------------------------------------------------------------------------
    # Format probabilities helper
    # -------------------------------------------------------------------------
    def format_probabilities(probabilities: Dict[str, float]) -> pd.DataFrame:
        """Convert prediction probabilities to a styled DataFrame for display."""
        return (
            pd.DataFrame.from_dict(probabilities, orient="index", columns=["confidence"])
            .sort_values("confidence", ascending=False)
            .style.format({"confidence": "{:.2%}"})
        )

    # -------------------------------------------------------------------------
    # Main Streamlit Application
    # -------------------------------------------------------------------------
    def main() -> None:
        """Render the Deloitte-ready Twitter Sentiment Intelligence Dashboard."""
        config, pipeline, metrics = _load_dependencies()

        # ---------------------- Sidebar ----------------------
        with st.sidebar:
            st.header("📊 Model Snapshot")
            st.write("**Classes:**", ", ".join(pipeline.classes_))
            if metrics:
                st.metric("Macro F1", f"{metrics.get('f1_macro', 0.0):.2f}")
                st.metric("Accuracy", f"{metrics.get('accuracy', 0.0):.2f}")
            else:
                st.info("Run `python scripts/train.py` to generate metrics.")
            st.download_button(
                label="⬇️ Download Metrics JSON",
                data=json.dumps(metrics or {}, indent=2).encode("utf-8"),
                file_name="metrics.json",
                mime="application/json",
            )
            st.info(
                "🚀 Tip: integrate Oracle Autonomous Database by updating `config/settings.yaml`."
            )

        # ---------------------- Tabs ----------------------
        tab_predict, tab_metrics = st.tabs(["🔮 Predict", "⚙️ Model Governance"])

        # ---------------------- Prediction Tab ----------------------
        with tab_predict:
            st.subheader("Real-Time Sentiment Assessment")
            user_input = st.text_area("Enter a tweet or customer comment:", height=150)
            if st.button("Run Analysis", type="primary"):
                if not user_input.strip():
                    st.warning("⚠️ Please enter text to analyse.")
                else:
                    label, probabilities = predict_with_threshold(user_input, config)
                    st.success(f"Predicted Sentiment: **{label.title()}**")
                    st.dataframe(format_probabilities(probabilities), use_container_width=True)

        # ---------------------- Metrics Tab ----------------------
        with tab_metrics:
            st.subheader("Operational Metrics")
            if metrics:
                metrics_df = (
                    pd.DataFrame(metrics, index=["score"])
                    .T.rename(columns={"score": "value"})
                )
                st.dataframe(metrics_df, use_container_width=True)
            else:
                st.info("Metrics will appear after the first training run (see `scripts/train.py`).")

        # ---------------------- Footer ----------------------
        st.markdown("---")
        st.caption("© 2025 Deloitte-aligned Sentiment Analytics Accelerator")

    # -------------------------------------------------------------------------
    # Entry Point
    # -------------------------------------------------------------------------
    if __name__ == "__main__":
        main()

except Exception as e:
    st.error(f"Startup failed: {e}")