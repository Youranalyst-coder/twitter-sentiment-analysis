"""CLI helper for scoring ad-hoc text snippets."""

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
from twitter_sentiment.predictor import predict_with_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a tweet or message with the trained sentiment model")
    parser.add_argument("text", help="The text to analyse")
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    label, probabilities = predict_with_threshold(args.text, config)
    print(json.dumps({"label": label, "probabilities": probabilities}, indent=2))


if __name__ == "__main__":
    main()
