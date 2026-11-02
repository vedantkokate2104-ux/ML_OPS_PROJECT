# ⚡ Project Apex – EventOracle

**Event-Driven Indian Financial Market Prediction System**

![CI/CD](https://github.com/vedantkokate2104-ux/ML_OPS_PROJECT/actions/workflows/ci-cd.yml/badge.svg)

**🔴 Live app:** https://mlopsprojectgit-dugwuoy4mszdzveah5az7x.streamlit.app

EventOracle detects market-moving events from news, macroeconomic releases (RBI, Fed, CPI, GDP, IIP),
institutional flows (FII/DII) and social sentiment, and predicts their impact on 10 assets:
NIFTY, BANKNIFTY, NIFTY IT, NIFTY Metal, INR/USD, Brent, Gold, Silver, Platinum and Bitcoin.

The front end is a **Streamlit** dashboard (`streamlit_app.py`) that reads the pipeline outputs from
**MongoDB Atlas** and runs the trained **two-stage LightGBM** models directly. No separate Flask server is needed.

---

## Architecture

```
 Data pipelines (batch / cron)                    MongoDB Atlas (eventoracle)        Presentation
 ───────────────────────────────                  ───────────────────────────        ─────────────────────────
 market_pipeline.py   (yfinance)        ──►  prices
 news_pipeline.py     (RSS + VADER)     ──►  news
 events_pipeline.py   (FRED, RBI, gov)  ──►  events          ┐
 flows_pipeline.py    (NSE FII/DII)     ──►  flows           ├─►  streamlit_app.py ──► Streamlit Cloud
 cleaning_pipeline.py (merge/features)  ──►  features        │      ├ data_access.py     Docker / K8s / Azure
 social_anomaly.py    (IsolationForest) ──►  social_anomaly  │      ├ modeling.py  (live LightGBM inference)
 modeling.py          (LightGBM + SHAP) ──►  predictions     ┘      ├ hedging.py
                                                                    ├ market_agent.py (Gemini)
                                                                    └ app_metrics.py ──► Prometheus ──► Grafana
 api_server.py (legacy Flask REST API + static/index.html, still works, now exposes /metrics)
```

### Dashboard pages

| Page | What it shows | Source module |
|---|---|---|
| Executive Dashboard | Latest prices/returns for all assets, rebased performance chart, FII/DII, news sentiment, technical indicators | `data_access.py` |
| Market Prediction | **Live LightGBM inference** (BUY/SELL/HOLD, class probabilities, magnitude, regime), stored prediction history, backtest equity curve, LightGBM↔LSTM agreement, Gemini AI agent | `modeling.py`, `market_agent.py` |
| Event Intelligence | RBI/Fed policy rates, CPI/GDP/IIP series, FII/DII flows, event-day vs non-event-day return impact | `events`, `flows`, `features` |
| News Sentiment | VADER sentiment mix, daily sentiment, event-type breakdown, headline table, live VADER tester | `news_pipeline.py` |
| Social Anomaly | Isolation Forest anomaly score, flagged points, market stress, social sentiment/buzz | `social_anomaly` collection |
| Hedging Strategy | Regime, beta proxy, VaR 95%, hedge ratio, institutional flow, options skew, strategy | `hedging.py` |
| Model Information | Pipeline architecture, 40 SHAP-selected features with importances, hyper-parameters, per-asset backtest Sharpe / win rate | `models/*.pkl` |
| System Status | MongoDB/Gemini/metrics status, collection counts, cache reset | |

---

## Repository layout

```
├── streamlit_app.py            # Streamlit entry point
├── data_access.py              # Shared MongoDB query layer
├── app_metrics.py              # Prometheus instrumentation (optional)
├── mlflow_tracking.py          # MLflow experiment tracking + model registry
├── api_server.py               # Legacy Flask API (+ /metrics, /healthz)
├── modeling.py, hedging.py, market_agent.py, social_anomaly.py, *_pipeline.py
├── models/                     # Trained LightGBM artefacts (committed, required at runtime)
├── static/                     # Legacy HTML frontend
├── tests/                      # pytest: model + page smoke tests
├── requirements.txt
├── Dockerfile, .dockerignore, docker-compose.yml
├── .streamlit/config.toml      # dark theme
├── .streamlit/secrets.toml.example
├── .env.example
├── .github/workflows/ci-cd.yml
├── k8s/                        # deployment, service, configmap
├── terraform/                  # Azure Container Apps IaC
└── monitoring/                 # prometheus.yml, grafana-dashboard.json, provisioning
```

---

## Configuration

| Variable | Required | Used by |
|---|---|---|
| `MONGO_URI` | **Yes** | Everything (MongoDB Atlas connection string) |
| `GEMINI_API_KEY` | Optional | AI Market Agent tab |
| `FRED_API_KEY`, `DATA_GOV_KEY`, `GNEWS_API_KEY` | Optional | Data pipelines only |
| `ENABLE_METRICS`, `METRICS_PORT` | Optional | Prometheus endpoint (default off; `true` in Docker) |

The app reads values from (in order) environment variables, a local `.env`, or **Streamlit Secrets**
(`st.secrets` values are copied into the environment at startup, so existing modules work unchanged).

> **Never commit `.env` or `.streamlit/secrets.toml`.** Both are in `.gitignore`.

---

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # fill in MONGO_URI (+ GEMINI_API_KEY)
streamlit run streamlit_app.py                         # http://localhost:8501
pytest -q                                              # tests
```

Refreshing data (optional, needs pipeline API keys):

```bash
python market_pipeline.py && python news_pipeline.py && python events_pipeline.py
python flows_pipeline.py && python cleaning_pipeline.py && python social_anomaly.py
python modeling.py --predict
```

---

## 🚀 Deploy on Streamlit Community Cloud (primary target)

1. Push the code to GitHub (see the workflow below). Make sure `models/*.pkl` is committed.
2. Open **https://share.streamlit.io/** and sign in with GitHub.
3. Click **Create app → Deploy a public app from GitHub** and pick the repository.
4. Select the branch (e.g. `main`).
5. Set **Main file path** to `streamlit_app.py`.
6. Under **Advanced settings**, choose Python **3.11**, then paste into **Secrets**:
   ```toml
   MONGO_URI = "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?appName=Cluster0"
   GEMINI_API_KEY = "your-gemini-api-key"
   ```
7. Click **Deploy**.

In MongoDB Atlas, go to **Network Access** and allow connections from Streamlit Cloud (`0.0.0.0/0`, or
Streamlit's published egress IPs). Without it the app shows "MongoDB not connected".

**Continuous deployment:** Streamlit Community Cloud watches the connected branch and redeploys automatically
on every push. It has no official deploy API or GitHub Action, so the CI pipeline validates the code and
Streamlit Cloud handles the deployment.

---

## GitHub workflow

```bash
git init
git add .
git status                     # confirm .env and secrets.toml are NOT listed
git commit -m "Streamlit app + DevOps setup"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

Day to day: `git checkout -b feature/x` → commit → push → open a PR. CI runs on the PR, then you merge to
`main` and Streamlit Cloud redeploys.

## ✅ CI/CD – GitHub Actions (`.github/workflows/ci-cd.yml`)

Runs on every push and pull request. It needs **no production secrets**: without `MONGO_URI`, every page must
still render and show a friendly message, and the tests check exactly that.

| Job | Steps |
|---|---|
| `test` | checkout → Python 3.11 → install deps → `compileall` syntax check → `ruff` (syntax errors / undefined names) → import check of the app modules → `pytest` (model artefacts load and predict; all 8 Streamlit pages render via `streamlit.testing.AppTest`) |
| `docker` | builds the image, runs it and polls `/_stcore/health` |
| `config` | parses the K8s/monitoring YAML and Grafana JSON, then runs `terraform fmt -check` and `terraform validate` |

---

## 🐳 Docker

```bash
docker build -t project-apex:latest .
docker run -d -p 8501:8501 -p 9100:9100 --env-file .env --name apex project-apex:latest
curl http://localhost:8501/_stcore/health          # → ok
curl http://localhost:9100/metrics                  # Prometheus metrics
```

The image uses `python:3.11-slim`, runs as a non-root user, has a `HEALTHCHECK` on `/_stcore/health`, and binds
Streamlit to `0.0.0.0:8501`. Secrets are injected at runtime and never baked into the image (`.env` is excluded
by `.dockerignore`).

## ☸️ Kubernetes (`k8s/`)

```bash
# Local cluster example (Docker Desktop K8s / minikube / kind): make the image available to the cluster first
kubectl create secret generic project-apex-secrets \
  --from-literal=MONGO_URI="mongodb+srv://..." \
  --from-literal=GEMINI_API_KEY="..."
kubectl apply -f k8s/configmap.yaml -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods,svc -l app=project-apex
kubectl port-forward svc/project-apex 8501:80       # → http://localhost:8501
```

The Deployment runs 2 replicas with startup, readiness and liveness probes on `/_stcore/health`. Each pod
requests 250m CPU / 512Mi and is limited to 1 CPU / 1.5Gi. Config comes from a ConfigMap and credentials from a
Secret, and the pods carry Prometheus scrape annotations. The Service uses `ClientIP` session affinity because
Streamlit sessions use websockets.

## ☁️ Terraform – Azure Container Apps (`terraform/`)

Creates a Resource Group, Azure Container Registry, Log Analytics workspace, Container Apps Environment, and a
Container App with external HTTPS ingress on port 8501, health probes and secret-backed env vars.

```bash
az login
cd terraform
export TF_VAR_mongo_uri="mongodb+srv://..."  TF_VAR_gemini_api_key="..."
terraform init
terraform plan
terraform apply -target=azurerm_container_registry.acr      # 1) create the registry first
az acr login --name projectapexacr
docker tag project-apex:latest projectapexacr.azurecr.io/project-apex:latest
docker push projectapexacr.azurecr.io/project-apex:latest   # 2) push the image
terraform apply                                             # 3) create the Container App
terraform output app_url
terraform destroy                                           # tear down
```

**Extending to AKS:** swap the Container Apps resources for `azurerm_kubernetes_cluster` plus an `AcrPull` role
assignment, then `kubectl apply -f k8s/` with the image set to the ACR path.

## 🧪 Experiment Tracking – MLflow (`mlflow_tracking.py`)

MLflow tracks the two-stage LightGBM models and acts as the model registry.

| Logged | Details |
|---|---|
| **Parameters** | All Stage 1 and Stage 2 LightGBM hyper-parameters, neutral threshold, transaction cost, CV splits, SHAP drop % |
| **Metrics** | Per-asset nested runs: accuracy, macro-F1, majority-class baseline, magnitude MAE, mean confidence, backtest return, Sharpe and win rate; portfolio-level aggregates |
| **Artifacts** | Feature-importance chart and CSV, the 40 SHAP-selected features, class map, asset encoder, per-asset metrics table |
| **Models** | Both models logged with input examples and registered as `eventoracle-stage1-impact-classifier` and `eventoracle-stage2-magnitude-regressor` |

Evaluation compares stored predictions with realised outcomes. Labels come from `modeling.label_events` and P&L
from `modeling.backtest`, the same definitions used in training. The stored predictions may include the training
period, so treat these as historical-fit metrics. `python modeling.py --train` also logs cross-validation macro-F1
and MAE to MLflow whenever `MLFLOW_TRACKING_URI` is set.

```bash
pip install -r requirements-mlops.txt
docker compose up -d mlflow                      # MLflow UI → http://localhost:5000
export MLFLOW_TRACKING_URI=http://localhost:5000
python mlflow_tracking.py                        # log + register current models
python modeling.py --train                       # (optional) retrain – tracked automatically
```

## 📊 Monitoring – Prometheus & Grafana (`monitoring/`)

**How it works:** Streamlit owns its HTTP server, so `app_metrics.py` uses `prometheus_client` to serve
`/metrics` on a separate port (9100) when `ENABLE_METRICS=true`. Prometheus scrapes that endpoint every 15 s.
It also scrapes the Flask API's `/metrics` (HTTP request count and latency) and **cAdvisor** (container
CPU/memory). Grafana queries Prometheus and loads the provisioned dashboard.

Metrics exported:

| Metric | Meaning |
|---|---|
| `eventoracle_start_time_seconds` | process start, used for the **uptime** panel |
| `eventoracle_page_views_total{page}` | dashboard page renders |
| `eventoracle_data_load_seconds{operation}` | MongoDB, inference, hedging and Gemini **latency** |
| `eventoracle_errors_total{operation}` | application **errors** |
| `eventoracle_mongo_up` | database health |
| `eventoracle_http_requests_total`, `eventoracle_http_request_seconds` | Flask API request **count/latency** |
| `process_cpu_seconds_total`, `process_resident_memory_bytes` | app **CPU/memory** (Linux) |
| `container_cpu_usage_seconds_total`, `container_memory_usage_bytes` | **container** resources (cAdvisor) |

```bash
docker compose up -d --build
# Dashboard :8501 · Prometheus :9090 · Grafana :3000 (admin/admin) · MLflow :5000
```

The Grafana datasource and the *EventOracle – Application & Container Health* dashboard are provisioned
automatically. You can also import `monitoring/grafana-dashboard.json` by hand.

> Streamlit Community Cloud cannot run Prometheus/Grafana sidecars, so monitoring runs only in the
> Docker Compose, Kubernetes or Azure deployments.

---

## Known limitations

- The `labels` collection is empty. Training CV scores from `modeling.py --train` are logged but not saved, so
  the Model page reports backtest metrics computed from stored predictions instead.
- Live inference detects regimes (KMeans) per asset. The batch pipeline fits them on all assets together, so
  regime labels can differ slightly.
- Some FII/DII rows come from the flows pipeline's placeholder fallback when the NSE API is unreachable. The UI
  labels these rows.
- The model pickles were created with scikit-learn 1.8, and newer versions show a harmless
  `InconsistentVersionWarning`.
- Research and education only. This is not investment advice.
