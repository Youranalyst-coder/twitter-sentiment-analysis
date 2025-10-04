"""Reusable text cleaning utilities."""

from __future__ import annotations

import re
import string
from typing import Iterable

import pandas as pd

from .config import Config
from .logging_utils import get_logger

LOGGER = get_logger(__name__)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_PATTERN = re.compile(r"#[A-Za-z0-9_]+")


def clean_text(text: str, config: Config) -> str:
    settings = config.preprocessing
    processed = text

    if settings.get("lowercase", True):
        processed = processed.lower()

    if settings.get("strip_urls", True):
        processed = URL_PATTERN.sub("", processed)

    if settings.get("strip_mentions", True):
        processed = MENTION_PATTERN.sub("", processed)

    if settings.get("strip_hashtags", False):
        processed = HASHTAG_PATTERN.sub("", processed)

    if settings.get("remove_punctuation", True):
        processed = processed.translate(str.maketrans("", "", string.punctuation))

    if settings.get("normalize_whitespace", True):
        processed = re.sub(r"\s+", " ", processed).strip()

    return processed


def preprocess_dataframe(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Apply text cleaning and reorder target labels."""

    data_settings = config.data
    text_column = data_settings.get("text_column", "text")
    target_column = data_settings.get("target_column", "sentiment")

    df = df.copy()
    df[text_column] = df[text_column].astype(str).apply(lambda text: clean_text(text, config))

    class_order: Iterable[str] = data_settings.get("class_order") or config.model.get("class_order")
    if class_order:
        df[target_column] = pd.Categorical(df[target_column], categories=list(class_order), ordered=True)

    LOGGER.info("Completed preprocessing for %d records", len(df))
    return df


__all__ = ["clean_text", "preprocess_dataframe"]
