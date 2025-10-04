"""Utility for verifying the deployed Streamlit app is reachable."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Optional

import requests

from src.twitter_sentiment.config import load_config


@dataclass
class LiveCheckResult:
    url: str
    status_code: Optional[int]
    elapsed_seconds: Optional[float]
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.status_code == 200 and self.error is None


def check_live_app(url: str, timeout: float = 10.0) -> LiveCheckResult:
    start = time.perf_counter()
    try:
        response = requests.get(url, timeout=timeout)
        elapsed = time.perf_counter() - start
        return LiveCheckResult(
            url=url,
            status_code=response.status_code,
            elapsed_seconds=elapsed,
        )
    except requests.RequestException as exc:  # pragma: no cover - network errors vary
        elapsed = time.perf_counter() - start
        return LiveCheckResult(
            url=url,
            status_code=getattr(exc.response, "status_code", None),
            elapsed_seconds=elapsed,
            error=str(exc),
        )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        required=False,
        help="Public Streamlit app URL to validate. Defaults to config live_app.streamlit_url",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to the project configuration file",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the server response (default: 10)",
    )
    return parser.parse_args(argv)


def _resolve_target_url(url_argument: Optional[str], config_path: str) -> str:
    if url_argument:
        return url_argument

    config = load_config(config_path)
    target_url = config.live_app.get("streamlit_url")

    if not target_url or "<" in target_url or "your-app" in target_url:
        raise ValueError(
            "No live Streamlit URL configured. Provide --url or update live_app.streamlit_url in the config."
        )

    return target_url


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        target_url = _resolve_target_url(args.url, args.config)
    except ValueError as error:
        print(f"⚠️ {error}", file=sys.stderr)
        return 1

    result = check_live_app(target_url, timeout=args.timeout)

    if result.is_success:
        print(
            f"✅ Live app reachable at {result.url} (status {result.status_code}, "
            f"latency {result.elapsed_seconds:.2f}s)"
        )
        return 0

    message = (
        f"❌ Unable to confirm app availability at {result.url}."
        f" Status: {result.status_code or 'N/A'}."
    )
    if result.error:
        message += f" Error: {result.error}"
    elif result.status_code == 404:
        message += (
            " The server returned 404. Verify your Vercel redirect or update the"
            " Streamlit deployment link as described in deployment/vercel_redirect.md."
        )
    print(message, file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
