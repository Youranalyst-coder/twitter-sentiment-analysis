# 💬 Twitter Sentiment Analyzer

A machine learning project that predicts whether a tweet is **positive**, **neutral**, or **negative** using natural language processing and logistic regression.

## 🔧 Tech Stack

- Python, Pandas, Scikit-learn, NLTK
- TF-IDF Vectorization
- Multiclass Logistic Regression
- Streamlit (Web App UI)

## 🚀 Features

- Text preprocessing and model training notebooks
- Trained logistic regression pipeline for three-way sentiment detection
- Streamlit UI that displays the predicted class and class probabilities

## ✅ Try it

Deploy the `app/app.py` Streamlit interface or use the provided pickle files directly in your own project.

## 📂 How to Run

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## 🧠 Training Artifacts

The repository contains pretrained artifacts inside the `app/` and root directories:

- `app/logistic_model.pkl` – Multiclass logistic regression estimator
- `app/tfidf_vectorizer.pkl` – Fitted TF-IDF vectorizer compatible with the model
- `sentiment_model.pkl` – End-to-end Scikit-learn pipeline (vectorizer + classifier)

These are generated from the notebook in `notebooks/sentiment_model.ipynb`.
