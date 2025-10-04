# GitHub Connection & Live App Validation Guide

This runbook walks you through connecting the local project to your GitHub repository, wiring the repo to Streamlit Community Cloud (or another host), and verifying that the live dashboard is reachable from any device.

## 1. Connect the local repo to GitHub

1. Create a new **empty** repository on GitHub (no README or licence yet) – e.g. `deloitte-twitter-sentiment`.
2. From your local workstation, authenticate with GitHub if you have not already:
   ```bash
gh auth login  # or use `git config credential.helper` if you prefer HTTPS tokens
```
3. Add the remote origin and push the current branch:
   ```bash
git remote add origin git@github.com:<your-handle>/deloitte-twitter-sentiment.git
# or use https://github.com/<your-handle>/deloitte-twitter-sentiment.git if you prefer HTTPS
git push -u origin work
```
4. (Optional) Protect the `main` branch and enable required status checks (`pytest`, `python -m compileall`) from the **Settings ▸ Branches** tab for interview-ready governance.

## 2. Enable Streamlit Community Cloud deployment

1. Visit [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Grant Streamlit access to the new repository when prompted.
3. Configure the deployment with the following options:
   - **Repository**: `<your-handle>/deloitte-twitter-sentiment`
   - **Branch**: `work` (or `main` after merge)
   - **Main file path**: `app/app.py`
   - **Python version**: 3.10+
   - **Secrets** (optional): Add Oracle wallet paths, API keys, etc.
4. Click **Deploy** and wait for the build log to show `🎈 You can now view your Streamlit app` with the generated `https://<slug>.streamlit.app` URL.

## 3. Validate the live URL locally

Once the deployment completes, store the URL in an environment variable or `.env` file:

```bash
export STREAMLIT_APP_URL="https://your-app.streamlit.app"
```

Then run the verification helper included in this repository:

```bash
python scripts/check_live_app.py --url "$STREAMLIT_APP_URL"
# or equivalently
make verify-live URL=$STREAMLIT_APP_URL
```

The script sends a quick HTTP request, checks for a `200 OK` response, and reports latency. This makes it easy to confirm that the dashboard is reachable from phones or tablets before an interview.

## 4. Update README quick links

1. Edit the **Live resources & quick links** table in `README.md` and replace the placeholder domain with your actual Streamlit URL.
2. Commit and push the change:
   ```bash
git commit -am "docs: add live app URL" && git push
```
3. Verify the README renders correctly on GitHub and that the link opens on mobile.

## 5. Keep the deployment healthy

- Re-deploy automatically by enabling **App settings ▸ Watch for changes** in Streamlit Cloud, which rebuilds when you push to the configured branch.
- Periodically run `make test` and `make train` locally so the CI workflow and Streamlit Cloud dependency cache stay fresh.
- Monitor the Streamlit Cloud logs (⚙️ ▸ App settings ▸ Logs) for errors after each push.

Following this sequence guarantees the repo is connected to GitHub, the cloud deployment is online, and you can confidently demonstrate the live app during interviews.
