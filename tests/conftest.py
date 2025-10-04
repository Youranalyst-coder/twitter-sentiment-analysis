from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest

from twitter_sentiment.config import load_config
from twitter_sentiment.modeling import persist_artifacts, train_and_evaluate


@pytest.fixture(scope="session", autouse=True)
def ensure_artifacts():
    """Train and persist the pipeline if artifacts are absent."""

    config = load_config()
    artifact_dir = Path(config.model.get("artifact_dir", "artifacts"))
    artifact_path = artifact_dir / config.model.get("pipeline_filename", "sentiment_pipeline.joblib")

    if artifact_path.exists():
        return

    pipeline, metrics = train_and_evaluate(config)
    persist_artifacts(pipeline, config, metrics)

