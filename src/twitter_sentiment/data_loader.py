"""Data loading utilities supporting CSV and Oracle Autonomous DB sources."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import Config
from .logging_utils import get_logger

LOGGER = get_logger(__name__)


def _load_from_csv(csv_path: Path, text_column: str, target_column: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found at {csv_path!s}")

    df = pd.read_csv(csv_path)
    if text_column not in df.columns or target_column not in df.columns:
        raise KeyError(
            f"CSV file must contain '{text_column}' and '{target_column}' columns. Found: {df.columns.tolist()}"
        )

    LOGGER.info("Loaded %d records from %s", len(df), csv_path)
    return df[[text_column, target_column]].dropna()


def _load_from_oracle(settings: dict) -> pd.DataFrame:
    spec = importlib.util.find_spec("oracledb")
    if spec is None:
        raise ModuleNotFoundError(
            "oracledb package is required for Oracle Autonomous Database ingestion. Install via `pip install oracledb`."
        )

    oracledb = importlib.import_module("oracledb")
    wallet_location = settings.get("wallet_location")
    user = settings.get("user")
    dsn = settings.get("dsn")
    query = settings.get("sql_query")

    if not all([wallet_location, user, dsn, query]):
        raise ValueError("Oracle configuration must define wallet_location, user, dsn and sql_query")

    connection = oracledb.connect(user=user, dsn=dsn, config_dir=wallet_location)
    try:
        df = pd.read_sql(query, con=connection)
    finally:
        connection.close()

    LOGGER.info("Loaded %d records from Oracle Autonomous Database", len(df))
    return df


def load_dataset(config: Config, limit: Optional[int] = None) -> pd.DataFrame:
    """Load dataset based on configuration.

    If the oracle integration is enabled the function attempts to load data from
    Oracle Autonomous Database, otherwise it falls back to CSV ingestion.
    """

    data_config = config.data
    oracle_config = config.oracle_integration

    if oracle_config.get("enabled"):
        df = _load_from_oracle(oracle_config)
    else:
        csv_path = Path(data_config.get("path", ""))
        df = _load_from_csv(csv_path, data_config.get("text_column", "text"), data_config.get("target_column", "sentiment"))

    if limit:
        df = df.head(limit)
        LOGGER.info("Sampling first %d records for experimentation", limit)

    return df


__all__ = ["load_dataset"]
