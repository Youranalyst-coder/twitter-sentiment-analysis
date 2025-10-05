# create_dummy_model.py
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Create artifacts directory if it doesn't exist
os.makedirs("artifacts", exist_ok=True)

# Simple pipeline for demo purposes
pipeline = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", LogisticRegression())
])

# Train the dummy model on small sample data
texts = [
    "I love this product",
    "This is great",
    "Amazing experience",
    "I hate this",
    "This is bad",
    "Terrible quality",
]
labels = ["positive", "positive", "positive", "negative", "negative", "negative"]
pipeline.fit(texts, labels)

# Save model artifact
joblib.dump(pipeline, "artifacts/sentiment_pipeline.joblib")
print("✅ Dummy sentiment model created at artifacts/sentiment_pipeline.joblib")
