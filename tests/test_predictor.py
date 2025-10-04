from __future__ import annotations

from twitter_sentiment.config import load_config
from twitter_sentiment.predictor import load_artifacts, predict_with_threshold


def test_load_artifacts_returns_pipeline():
    config = load_config()
    pipeline, metrics = load_artifacts(config)
    assert hasattr(pipeline, "predict")
    assert isinstance(metrics, dict)


def test_predict_with_threshold_returns_label_and_probabilities():
    config = load_config()
    pipeline, _ = load_artifacts(config)
    label, probabilities = predict_with_threshold("I love this experience", config)
    assert label in {"positive", "negative", "neutral"}
    assert set(probabilities.keys()) == set(pipeline.classes_)
