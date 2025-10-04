"""Command line utility for training the sentiment analysis pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from twitter_sentiment.config import load_config
from twitter_sentiment.modeling import persist_artifacts, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Deloitte-aligned Twitter sentiment model")
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--metrics",
        default="artifacts/metrics.json",
        help="Optional path to persist evaluation metrics as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    pipeline, metrics_summary = train_and_evaluate(config)
    artifact_path = persist_artifacts(pipeline, config, metrics_summary)

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    print(f"Artifacts saved to {artifact_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
