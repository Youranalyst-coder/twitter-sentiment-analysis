from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_artifacts() -> Tuple[object, object]:
    """Load the trained sentiment model and TF-IDF vectorizer."""
    artifacts_dir = Path(__file__).resolve().parent
    model = joblib.load(artifacts_dir / "logistic_model.pkl")
    vectorizer = joblib.load(artifacts_dir / "tfidf_vectorizer.pkl")
    return model, vectorizer


def predict_sentiment(text: str) -> Tuple[str, float, Dict[str, float]]:
    """Predict the sentiment label and probability distribution for the text."""
    model, vectorizer = load_artifacts()
    features = vectorizer.transform([text])
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_
    class_probabilities = {label: prob for label, prob in zip(classes, probabilities)}
    top_label = max(class_probabilities, key=class_probabilities.get)
    top_score = class_probabilities[top_label]
    return top_label, top_score, class_probabilities


SENTIMENT_DISPLAY = {
    "positive": {
        "title": "Positive",
        "emoji": "😊",
        "description": "The tweet expresses a favourable or uplifting sentiment.",
    },
    "neutral": {
        "title": "Neutral",
        "emoji": "😐",
        "description": "The tweet is balanced without a clear positive or negative leaning.",
    },
    "negative": {
        "title": "Negative",
        "emoji": "😞",
        "description": "The tweet carries criticism, frustration, or an unfavourable view.",
    },
}


st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="💬")
st.title("💬 Twitter Sentiment Analysis")
st.write(
    "Enter a tweet below and the model will classify it as positive, neutral, or negative."
)

user_input = st.text_area(
    "Tweet text",
    placeholder="I love how friendly everyone is on Twitter today!",
    height=150,
)

if st.button("Predict Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        label, score, probabilities = predict_sentiment(user_input)
        details = SENTIMENT_DISPLAY.get(label, {"title": label.title(), "emoji": "", "description": ""})

        st.success(
            f"{details['emoji']} **{details['title']}** sentiment detected with a confidence of {score:.1%}."
        )
        if details.get("description"):
            st.caption(details["description"])

        sorted_probabilities = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        probability_frame = pd.DataFrame(
            {
                "Sentiment": [SENTIMENT_DISPLAY[label]["title"] for label, _ in sorted_probabilities],
                "Probability": [score for _, score in sorted_probabilities],
            }
        )
        probability_frame["Probability"] = probability_frame["Probability"].map(lambda p: f"{p:.1%}")
        st.subheader("Class Probabilities")
        st.dataframe(probability_frame, hide_index=True, width='stretch')
else:
    st.info("Click **Predict Sentiment** to get a prediction.")
