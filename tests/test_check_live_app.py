from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.check_live_app import LiveCheckResult, check_live_app, main


class DummyResponse(SimpleNamespace):
    def __init__(self, status_code: int):
        super().__init__(status_code=status_code)


def test_check_live_app_success(monkeypatch):
    def fake_get(url, timeout):
        assert url == "https://example.com"
        assert timeout == 5
        return DummyResponse(status_code=200)

    monkeypatch.setattr("scripts.check_live_app.requests.get", fake_get)

    result = check_live_app("https://example.com", timeout=5)
    assert isinstance(result, LiveCheckResult)
    assert result.is_success
    assert result.status_code == 200
    assert result.error is None


def test_check_live_app_failure(monkeypatch):
    class DummyException(requests.RequestException):
        def __init__(self, message: str):
            super().__init__(message)
            self.response = DummyResponse(status_code=503)

    def fake_get(url, timeout):  # pragma: no cover - executed via exception
        raise DummyException("service unavailable")

    monkeypatch.setattr("scripts.check_live_app.requests.get", fake_get)

    result = check_live_app("https://example.com")
    assert not result.is_success
    assert result.status_code == 503
    assert "service unavailable" in result.error


def test_main_success(monkeypatch, capsys):
    def fake_check(url, timeout):
        return LiveCheckResult(url=url, status_code=200, elapsed_seconds=0.12, error=None)

    monkeypatch.setattr("scripts.check_live_app.check_live_app", fake_check)

    exit_code = main(["--url", "https://example.com"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "✅" in captured.out


def test_main_failure(monkeypatch, capsys):
    def fake_check(url, timeout):
        return LiveCheckResult(url=url, status_code=500, elapsed_seconds=0.12, error="boom")

    monkeypatch.setattr("scripts.check_live_app.check_live_app", fake_check)

    exit_code = main(["--url", "https://example.com"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "❌" in captured.err


def test_main_uses_config_when_url_missing(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps({"live_app": {"streamlit_url": "https://config-app.streamlit.app"}}))

    def fake_check(url, timeout):
        assert url == "https://config-app.streamlit.app"
        return LiveCheckResult(url=url, status_code=200, elapsed_seconds=0.05, error=None)

    monkeypatch.setattr("scripts.check_live_app.check_live_app", fake_check)

    exit_code = main(["--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "✅" in captured.out


def test_main_warns_when_config_missing_url(tmp_path, capsys):
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps({"live_app": {"streamlit_url": "https://<your-app>.streamlit.app"}}))

    exit_code = main(["--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "⚠️" in captured.err
