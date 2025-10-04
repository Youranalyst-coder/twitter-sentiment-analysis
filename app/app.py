"""Streamlit front-end for the Deloitte-ready Twitter Sentiment Intelligence dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

# -------------------------------------------------------------------------
# Path setup to include the local src/ package for imports
# -------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from twitter_sentiment.config import load_config
from twitter_sentiment.predictor import load_artifacts, predict_with_threshold

# -------------------------------------------------------------------------
# Streamlit App Configuration
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Twitter Sentiment Intelligence",
    page_icon="💼",
    layout="wide",
)

# -------------------------------------------------------------------------
# Cached resource loading (config, pipeline, metrics)
# -------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_dependencies():
    """Load configuration, trained pipeline, and metrics from artifacts."""
    config = load_config()
    pipeline, metrics = load_artifacts(config)
    return config, pipeline, metrics


# -------------------------------------------------------------------------
# Helper function to format prediction probabilities
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

    # -----------------------------------------------------
