import os
import joblib
import urllib.request

# ⚙️ Public raw URL of your uploaded model on GitHub
MODEL_URL = "https://raw.githubusercontent.com/Youranalyst-coder/twitter-sentiment-analysis/main/artifacts/sentiment_pipeline.joblib"
MODEL_PATH = "artifacts/sentiment_pipeline.joblib"

def get_model():
    """Download and cache the model if not available."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from remote repository...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return joblib.load(MODEL_PATH)
