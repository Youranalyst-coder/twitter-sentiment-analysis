# Oracle Cloud Infrastructure Deployment Runbook

This guide describes how to host the Streamlit dashboard and model artifacts on Oracle Cloud Infrastructure (OCI) in a way that resonates with Deloitte India Oracle Analyst expectations.

## 1. Prerequisites

- OCI tenancy with permissions for Container Registry, Container Instances, and Object Storage.
- Oracle Autonomous Database (ATP/ADW) with wallet credentials downloaded.
- GitHub repository with this project (recommended branching: `main`, `develop`, feature branches).

## 2. Build the container image

```bash
otc login  # OCI CLI authentication
export OCIR_REGION=iad.ocir.io   # replace with your region
export OCIR_NAMESPACE=<tenancy-namespace>
export IMAGE_NAME=twitter-sentiment

# Build and tag the image
podman build -t $IMAGE_NAME -f deployment/Dockerfile .
podman tag $IMAGE_NAME $OCIR_REGION/$OCIR_NAMESPACE/$IMAGE_NAME:latest

# Push to OCI Registry
podman push $OCIR_REGION/$OCIR_NAMESPACE/$IMAGE_NAME:latest
```

> Tip: Use GitHub Actions to automate image builds on every tagged release.

## 3. Provision infrastructure

1. **Container Instance** – create via OCI Console, referencing the pushed image.
2. **Networking** – open HTTPS ingress on the public subnet and map to port `8501`.
3. **Object Storage bucket** – store model artifacts (`artifacts/sentiment_pipeline.joblib`) and metrics.
4. **Vault** – manage secrets for database credentials and wallet passwords.

## 4. Configure environment variables

| Variable | Description |
| --- | --- |
| `MODEL_ARTIFACT_URI` | OCI Object Storage pre-authenticated URL to the pipeline joblib |
| `CONFIG_PATH` | Path to mounted configuration file (e.g., `/app/config/settings.yaml`) |
| `ORACLE_WALLET_DIR` | Mount point for wallet files to enable database connectivity |
| `STREAMLIT_SERVER_PORT` | Set to `8501` |

Mount the wallet as a secret volume in the container instance for secure connections.

## 5. Automate retraining

- Schedule `scripts/train.py` via OCI Data Science jobs or GitHub Actions workflow.
- Upload the refreshed `sentiment_pipeline.joblib` to Object Storage.
- Trigger a rolling restart of the container instance to pick up the new artifact.

## 6. Observability & governance

- Enable OCI Logging for container stdout/err to capture Streamlit logs.
- Connect to OCI Application Performance Monitoring for latency insights.
- Track model performance drift by comparing live predictions vs. labelled feedback in Autonomous Database.

## 7. Extend to Deloitte storytelling

- Embed the public Streamlit URL in Oracle APEX or Deloitte's internal portals.
- Prepare a KPI dashboard using Oracle Analytics Cloud for executive presentations.
- Document the runbook and operational SLAs in Confluence or Deloitte knowledge base.

## 8. Multi-cloud notes

- **Azure App Service** – deploy the same container, store artifacts in Azure Blob Storage, and manage secrets in Azure Key Vault.
- **AWS App Runner** – push the container image to ECR, configure environment variables, and mount AWS Secrets Manager for database credentials.

With these steps you can confidently demonstrate cloud deployment proficiency, automation best practices, and Oracle ecosystem fluency during interviews.
