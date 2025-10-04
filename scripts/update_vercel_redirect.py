"""Generate a Vercel-friendly redirect page that points to the live Streamlit app."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from src.twitter_sentiment.config import load_config

REDIRECT_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <title>Redirecting to Streamlit dashboard</title>
    <meta http-equiv=\"refresh\" content=\"0; url={target}\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <style>
      body {{
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        display: flex;
        min-height: 100vh;
        align-items: center;
        justify-content: center;
        background: #0f172a;
        color: #f8fafc;
        padding: 2rem;
        text-align: center;
      }}
      a {{ color: #38bdf8; }}
    </style>
  </head>
  <body>
    <main>
      <h1>Taking you to the Deloitte-ready sentiment dashboard…</h1>
      <p>If you are not redirected automatically, <a href=\"{target}\">click here to open the Streamlit app</a>.</p>
    </main>
    <script>
      window.location.replace('{target}');
    </script>
  </body>
</html>
"""


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to the configuration file containing live_app settings",
    )
    parser.add_argument(
        "--output",
        default="deployment/vercel/index.html",
        help="HTML file path to write (defaults to deployment/vercel/index.html)",
    )
    parser.add_argument(
        "--url",
        help="Override URL for the redirect. If omitted, use live_app.streamlit_url from the config",
    )
    return parser.parse_args(argv)


def resolve_streamlit_url(args: argparse.Namespace) -> str:
    if args.url:
        return args.url

    config = load_config(args.config)
    streamlit_url = config.live_app.get("streamlit_url")
    if not streamlit_url or "<" in streamlit_url or "your-app" in streamlit_url:
        raise ValueError(
            "Configure live_app.streamlit_url in config/settings.yaml or pass --url to set the redirect target."
        )

    return streamlit_url


def write_redirect_html(output_path: str, target_url: str) -> Path:
    html_content = REDIRECT_TEMPLATE.format(target=target_url)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html_content, encoding="utf-8")
    return destination


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    try:
        url = resolve_streamlit_url(args)
    except ValueError as error:
        print(f"⚠️ {error}")
        return 1

    destination = write_redirect_html(args.output, url)
    print(f"✅ Wrote Vercel redirect page pointing to {url} at {destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
