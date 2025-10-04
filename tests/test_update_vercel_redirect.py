from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.update_vercel_redirect import main


def test_main_generates_redirect(tmp_path, capsys):
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps({"live_app": {"streamlit_url": "https://demo.streamlit.app"}}))
    output_path = tmp_path / "index.html"

    exit_code = main(["--config", str(config_path), "--output", str(output_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "✅" in captured.out
    html = output_path.read_text(encoding="utf-8")
    assert "https://demo.streamlit.app" in html
    assert "window.location.replace" in html


def test_main_errors_with_placeholder_url(tmp_path, capsys):
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps({"live_app": {"streamlit_url": "https://<your-app>.streamlit.app"}}))
    output_path = tmp_path / "index.html"

    exit_code = main(["--config", str(config_path), "--output", str(output_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "⚠️" in captured.out
    assert not output_path.exists()
