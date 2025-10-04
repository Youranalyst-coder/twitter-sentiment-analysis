# Streamlit Community Cloud Launch Guide

Get a production-style link for the Deloitte-ready dashboard in under 10 minutes. Once deployed, the URL will look like `https://<your-app>.streamlit.app` and is reachable from any phone, tablet, or laptop.

## 1. Prepare the repository

1. Fork or push this project to your personal GitHub account.
2. Ensure the latest model artifact exists at `artifacts/sentiment_pipeline.joblib` (run `python scripts/train.py` if needed).
3. Commit and push any changes so Streamlit Cloud can clone the repo.

## 2. Create the Streamlit app

1. Navigate to [https://share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app** and select the repository and branch containing this project.
3. Set the main file path to `app/app.py` and the Python version to **3.10** (or higher).

## 3. Configure secrets & environment variables

Add the following environment settings under **Advanced settings → Secrets**:

```toml
MODEL_ARTIFACT_PATH = "artifacts/sentiment_pipeline.joblib"
CONFIG_PATH = "config/settings.yaml"
```

If you plan to stream data from Oracle Autonomous Database, create an Object Storage bucket or GitHub release that hosts the wallet zip and reference it via an additional variable, for example:

```toml
ORACLE_WALLET_ZIP_URL = "https://objectstorage.<region>.oraclecloud.com/.../wallet.zip"
```

> ℹ️ Secrets are encrypted by Streamlit Cloud and can be rotated without redeploying the app.

## 4. Manage Python dependencies

1. Keep `requirements.txt` in the project root. Streamlit Cloud will automatically install the listed packages.
2. If you add Oracle-specific drivers such as `oracledb`, ensure the package is listed in `requirements.txt` and that the wallet files are accessible via the secret or public URL.

## 5. Deploy and validate

1. Click **Deploy**. The platform will build the environment and serve the Streamlit app.
2. Copy the generated URL (e.g., `https://deloitte-sentiment.streamlit.app`).
3. Open the link on a mobile phone to validate responsive behaviour and confirm the app loads the cached model artifact.
4. Update the **Live resources** table in [`README.md`](../README.md) with the new URL.

## 6. Keep the link fresh

- When you push updates to `main`, Streamlit Cloud redeploys automatically.
- Use feature branches for new features and merge into `main` once CI passes so the live link always reflects production-ready code.
- Capture screenshots of the deployed app (mobile and desktop) and store them in `docs/screenshots/` for interview-ready collateral.

## 7. Optional: Custom domain & analytics

- Map a custom domain via Streamlit Cloud's **Domain management** settings to share a memorable URL during interviews.
- Integrate Google Analytics or Microsoft Clarity to showcase user engagement metrics. Add the script via `st.components.v1.html` in `app/app.py`.

With this setup you can hand over a polished, always-on demo link that demonstrates cloud readiness and Deloitte-style delivery excellence.
