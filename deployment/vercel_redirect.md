# Vercel redirect setup for the Streamlit dashboard

The Streamlit experience in this repository is designed to run as a long-lived Python process. Vercel is optimised for static
content and serverless functions, so deploying Streamlit directly on Vercel typically results in a **404 Not Found** error like
the one shown in the screenshot you shared.

To turn your Vercel link into a reliable launchpad for interviews, use it as a lightweight redirect to the publicly hosted
Streamlit application (for example, a Streamlit Community Cloud deployment). The repository now ships with automation to
generate that redirect.

## 1. Configure the Streamlit URL

Update `live_app.streamlit_url` inside [`config/settings.yaml`](../config/settings.yaml) once your Streamlit deployment is
live. This value is used for both CLI validation and the Vercel redirect generator.

```json
"live_app": {
  "streamlit_url": "https://your-sentiment-app.streamlit.app",
  "vercel_redirect_domain": "https://twitter-sentiment.vercel.app"
}
```

## 2. Generate the redirect artefact

Run the helper script to create `deployment/vercel/index.html` with a zero-second redirect to your Streamlit dashboard:

```bash
python scripts/update_vercel_redirect.py
# or override explicitly
python scripts/update_vercel_redirect.py --url https://your-sentiment-app.streamlit.app
```

Commit the generated file so Vercel serves it as the project root.

## 3. Deploy on Vercel

1. Create (or update) a Vercel project pointing to this GitHub repository.
2. Set the **Framework Preset** to **Other** so Vercel treats the repo as static content.
3. Add an environment variable `STREAMLIT_PUBLIC_URL` with the Streamlit link (optional if you bake it into the config).
4. Deploy. Vercel will host the generated `deployment/vercel/index.html`. If you prefer the URL to live at the root, set the
   project output directory to `deployment/vercel` under **Build & Output Settings**.

Visiting your Vercel domain will now redirect immediately to the live Streamlit experience—perfect for resume QR codes or
mobile demos.

## 4. Validate the setup

After deployment, run the live check CLI without arguments. It now reads from the config and confirms the Streamlit app is
reachable:

```bash
python scripts/check_live_app.py
```

If the CLI reports a 404, double-check that the generated HTML is committed and that Vercel is configured to serve the
`deployment/vercel` directory. The CLI surfaces this hint directly when a 404 occurs.

---

With these changes in place, both the Streamlit Cloud link and your Vercel vanity URL work seamlessly from any phone or
browser.
