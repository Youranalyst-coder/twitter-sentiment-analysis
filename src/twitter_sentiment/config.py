"""Utilities for loading and validating project configuration."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

_yaml_spec = importlib.util.find_spec("yaml")
_yaml_module = importlib.import_module("yaml") if _yaml_spec else None


@dataclass
class Config:
    """Typed representation of the project configuration."""

    raw: Dict[str, Any]

    @property
    def data(self) -> Dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def preprocessing(self) -> Dict[str, Any]:
        return self.raw.get("preprocessing", {})

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw.get("training", {})

    @property
    def project(self) -> Dict[str, Any]:
        return self.raw.get("project", {})

    @property
    def monitoring(self) -> Dict[str, Any]:
        return self.raw.get("monitoring", {})

    @property
    def oracle_integration(self) -> Dict[str, Any]:
        return self.raw.get("oracle_integration", {})

    @property
    def live_app(self) -> Dict[str, Any]:
        return self.raw.get("live_app", {})


def _deserialize_config(content: str) -> Dict[str, Any]:
    if _yaml_module:
        return _yaml_module.safe_load(content)
    return json.loads(content)


def load_config(path: str | Path = "config/settings.yaml") -> Config:
    """Load configuration from YAML (or JSON) file."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path!s}")

    content = config_path.read_text(encoding="utf-8")
    raw_config = _deserialize_config(content)

    if not isinstance(raw_config, dict):
        raise ValueError("Configuration file must define a dictionary at the top level")

    return Config(raw=raw_config)


__all__ = ["Config", "load_config"]
