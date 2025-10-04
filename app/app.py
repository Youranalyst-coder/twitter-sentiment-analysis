"""Streamlit front-end for the Deloitte-ready Twitter sentiment analyser."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from twitter_sentiment.config import load_config
from twitter_sentiment.predictor import load_artifacts, predict_with_threshold

st.set_page_config(page_title="Twitter Sentiment Intelligence", page_icon="💼", layout="wide")

@st.cache_resource(show_spinner=False)
def _load_dependencies():
    config = load_config()
    pipeline, metrics = load_artifacts(config)
    return config, pipeline, metrics


def format_probabilities(probabilities: Dict[str, float]) -> pd.DataFrame:
    return (
        pd.DataFrame.from_dict(probabilities, orient="index", columns=["confidence"])
        .sort_values("confidence", ascending=False)
        .style.format({"confidence": "{:.2%}"})
    )


def main() -> None:
    config, pipeline, metrics = _load_dependencies()

    st.title("Twitter Sentiment Intelligence Dashboard")
    st.caption(
        "Built to demonstrate Deloitte India Oracle Analyst competencies across data engineering, analytics, and automation."
    )

    with st.sidebar:
        st.header("Model Snapshot")
        st.write("**Classes:**", ", ".join(pipeline.classes_))
        if metrics:
            st.metric("Macro F1", f"{metrics.get('f1_macro', 0.0):.2f}")
            st.metric("Accuracy", f"{metrics.get('accuracy', 0.0):.2f}")
        st.download_button(
            "Download Metrics JSON",
            data=json.dumps(metrics or {}, indent=2).encode("utf-8"),
            file_name="metrics.json",
            mime="application/json",
        )
        st.info(
            "🚀 Tip: integrate with Oracle Autonomous Database by updating `config/settings.yaml`."
        )

    tab_predict, tab_metrics = st.tabs(["Predict", "Model Governance"])

    with tab_predict:
        st.subheader("Real-time Sentiment Assessment")
        user_input = st.text_area("Enter a tweet or customer verbatim", height=150)
        if st.button("Run Analysis", type="primary"):
            if not user_input.strip():
                st.warning("Please enter text to analyse.")
            else:
                label, probabilities = predict_with_threshold(user_input, config)
                st.success(f"Predicted sentiment: **{label.title()}**")
                st.dataframe(format_probabilities(probabilities), use_container_width=True)

    with tab_metrics:
        st.subheader("Operational Metrics")
        if metrics:
            metrics_df = pd.DataFrame(metrics, index=["score"]).T.rename(columns={"score": "value"})
            st.dataframe(metrics_df, use_container_width=True)
        else:
            st.info("Metrics will appear after the first training run (see `scripts/train.py`).")

    st.markdown("---")
    st.caption("© 2024 Deloitte-aligned Sentiment Analytics Accelerator")


if __name__ == "__main__":
    main()
