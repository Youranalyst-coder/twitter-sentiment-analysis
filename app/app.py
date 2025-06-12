import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already available
nltk.download('stopwords')

# Load model and vectorizer
with open("model/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit App UI
st.set_page_config(page_title="Twitter Sentiment Analysis", layout="centered")

st.title("📊 Twitter Sentiment Analysis")
st.write("Enter a tweet below to predict whether it's **Positive**, **Negative**, or **Neutral**.")

tweet = st.text_area("✍️ Enter Tweet Text")

if st.button("Analyze Sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocessing
        input_vector = vectorizer.transform([tweet])
        prediction = model.predict(input_vector)[0]

        st.success(f"🔍 Sentiment: **{prediction.capitalize()}**")

