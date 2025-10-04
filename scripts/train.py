"""Configuration loader that supports both YAML and JSON formats."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import yaml


class Config(dict):
    """Wrapper class to access config like attributes."""
    def __getattr__(self, item):
        return self.get(item)


def load_config(path: str | Path = "config/settings.yaml") -> Config:
    """
    Load configuration from YAML or JSON.
    Automatically detects file type based on extension.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        if path.suffix.lower() in [".yaml", ".yml"]:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        elif path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    except Exception as e:
        raise ValueError(f"Failed to parse config file {path}: {e}")

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config structure in {path}")

    return Config(data)
