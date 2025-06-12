import streamlit as st
import joblib
import os

# Load model and vectorizer once
model_path = os.path.join(os.getcwd(), "app", "logistic_model.pkl")
vectorizer_path = os.path.join(os.getcwd(), "app", "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

st.title("Twitter Sentiment Analysis")

user_input = st.text_area("Enter your tweet or sentence here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Transform the input text
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]

        sentiment = "Positive" if prediction == 1 else "Negative"
        st.write(f"Predicted sentiment: **{sentiment}**")
