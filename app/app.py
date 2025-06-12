import streamlit as st
import joblib

# Load vectorizer and model
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
model = joblib.load('model/logistic_model.pkl')

st.title("Twitter Sentiment Analysis")

user_input = st.text_area("Enter text to analyze sentiment:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)
        sentiment = "Positive 😊" if prediction[0] == 1 else "Negative 😞"
        st.success(f"Predicted sentiment: {sentiment}")
