from __future__ import annotations

import pandas as pd

from twitter_sentiment.config import load_config, Config
from twitter_sentiment.preprocessing import clean_text, preprocess_dataframe


def test_clean_text_removes_urls_mentions(tmp_path):
    config = load_config()
    result = clean_text("Check this https://example.com @user", config)
    assert "http" not in result
    assert "@" not in result


def test_preprocess_dataframe_returns_expected_columns(tmp_path):
    config = load_config()
    df = pd.DataFrame({
        config.data["text_column"]: ["Great flight!"],
        config.data["target_column"]: ["positive"],
    })
    processed = preprocess_dataframe(df, config)
    assert processed.shape == (1, 2)
    assert processed[config.data["text_column"]].iloc[0] == "great flight"
