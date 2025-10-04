"""Core package for the Deloitte-aligned Twitter sentiment analysis solution."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("twitter-sentiment-analysis")
except PackageNotFoundError:  # pragma: no cover - best effort
    __version__ = "0.1.0"

__all__ = ["__version__"]
