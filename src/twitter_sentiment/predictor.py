"""Utilities to load persisted artifacts and score new text."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib

from .config import Config, load_config
from .logging_utils import get_logger

LOGGER = get_logger(__name__)


def load_artifacts(config: Config | None = None):
    if config is None:
        config = load_config()

    artifact_dir = Path(config.model.get("artifact_dir", "artifacts"))
    artifact_path = artifact_dir / config.model.get("pipeline_filename", "sentiment_pipeline.joblib")

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {artifact_path!s}. Train the model by running `python scripts/train.py`."
        )

    data = joblib.load(artifact_path)
    pipeline = data["pipeline"] if isinstance(data, dict) and "pipeline" in data else data
    metrics: Dict[str, float] = data.get("metrics", {}) if isinstance(data, dict) else {}

    LOGGER.info("Loaded pipeline from %s", artifact_path)
    return pipeline, metrics


def predict(texts: List[str], config: Config | None = None) -> Tuple[List[str], List[Dict[str, float]]]:
    pipeline, _ = load_artifacts(config)
    probabilities = pipeline.predict_proba(texts)
    labels = pipeline.classes_

    predicted_labels = pipeline.predict(texts)
    probability_dicts: List[Dict[str, float]] = []
    for prob in probabilities:
        probability_dicts.append({label: float(score) for label, score in zip(labels, prob)})

    return predicted_labels.tolist(), probability_dicts


def predict_with_threshold(text: str, config: Config | None = None) -> Tuple[str, Dict[str, float]]:
    config = config or load_config()
    labels, probabilities = predict([text], config)
    probability = probabilities[0]

    thresholds = config.training.get("probability_thresholds", {})
    positive_threshold = thresholds.get("positive", 0.5)
    negative_threshold = thresholds.get("negative", 0.5)

    selected_label = labels[0]
    if probability.get("positive", 0.0) >= positive_threshold:
        selected_label = "positive"
    elif probability.get("negative", 0.0) >= negative_threshold:
        selected_label = "negative"
    else:
        selected_label = "neutral"

    return selected_label, probability


__all__ = ["load_artifacts", "predict", "predict_with_threshold"]
