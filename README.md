# Loan Approval Prediction System

MLOps end-term project (STDE 301) — a machine learning system that predicts
whether a loan application should be approved or rejected, covering the full
MLOps lifecycle: data versioning, experiment tracking, testing,
containerization, CI/CD, deployment, and monitoring.

Dataset: Kaggle ["Loan Prediction Problem Dataset"](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
(614 labeled applications). The dataset's own "test" file has no target
column, so the labeled file is split ourselves into train/val/test.

## Architecture

```
Data Source -> DVC Versioning -> Data Validation -> Feature Engineering
  -> Model Training -> MLflow Tracking -> Pytest Validation -> Docker Build
  -> GitHub Actions CI/CD -> Kubernetes Deployment -> FastAPI Service
  -> Prometheus -> Grafana Dashboard
```

## Repository structure

```
├── data/
│   ├── raw/              # DVC-tracked source data
│   └── processed/        # DVC-tracked: train/val/test splits, preprocessor, model
├── models/                # git-tracked copy of the winning model + preprocessor —
│                          # what Docker/Kubernetes/CI actually serve, so a plain
│                          # `git clone` builds without needing a reachable DVC remote
├── notebooks/             # exploration only, not part of the pipeline
├── src/
│   ├── ingestion/         # load raw data
│   ├── validation/        # schema/null/range checks
│   ├── transformation/    # imputation, scaling, encoding, train/val/test split
│   ├── training/          # model training + MLflow logging
│   ├── prediction/        # inference logic used by the API
│   └── utils/             # shared config (columns, paths)
├── tests/                 # pytest suite, synthetic fixtures only — never touches
│                          # the real dataset, so it passes in CI with no DVC pull
├── deployment/
│   └── kubernetes/        # Deployment + Service manifests
├── monitoring/
│   ├── prometheus.yml
│   ├── docker-compose.yml # Prometheus + Grafana, provisioned automatically
│   └── grafana/           # datasource + dashboard provisioning
├── .github/workflows/ci.yml
├── dvc.yaml               # DVC pipeline: validate -> transform -> train
├── requirements.in / .txt         # full dev environment (dvc, mlflow, pytest, ...)
├── requirements-serving.in / .txt # slim runtime deps baked into the Docker image
├── Dockerfile
└── app.py                 # FastAPI entrypoint
```

## Setup

```powershell
conda create -n loan-mlops python=3.12 -y
conda activate loan-mlops
pip install uv
uv pip install -r requirements.txt
```

> Always recompile with `uv pip compile --universal requirements.in -o requirements.txt`,
> never a plain compile — a plain compile pins Windows-only transitive deps
> (e.g. `pywin32`) with no platform marker, which breaks `pip install` on the
> Linux GitHub Actions runner.

## Running the pipeline

```powershell
dvc repro          # validate -> transform -> train, writes reports/*.json
mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1
```

`--workers 1` is required — multi-worker uvicorn can't share a socket on
Windows. MLflow 3.x also refuses a bare `./mlruns` file store, hence the
SQLite backend.

## Running the API

**Bare metal:**
```powershell
uvicorn app:app --reload
```

**Docker:**
```powershell
docker build -t loan-approval-api:local .
docker run -d --name loan-api -p 8000:8000 loan-approval-api:local
```

**Kubernetes** (Docker Desktop's built-in cluster — enable it under
Settings → Kubernetes first):
```powershell
kubectl apply -f deployment/kubernetes/
kubectl get pods
```
The Service is `NodePort` on `30080`, so the API is reachable at
`http://localhost:30080` with no port-forwarding needed.

Either way: `GET /health`, `POST /predict`, `GET /metrics`.

## Monitoring

```powershell
docker compose -f monitoring/docker-compose.yml up -d
```
- Prometheus: `http://localhost:9090` (scrapes the Kubernetes NodePort service)
- Grafana: `http://localhost:3000` (`admin` / `admin`) — the **"Loan Approval API"**
  dashboard is auto-provisioned: request rate, p95 latency, error rate, total
  requests, and an ML-specific panel tracking the Approved/Rejected prediction
  split over time (watches the *model's* behavior, not just the service's).

## Testing

```powershell
pytest -q
```
All 18 tests use synthetic fixtures — none touch the real DVC-tracked data,
so the suite passes identically on a clean CI runner with no dataset present.

## Rubric (40 marks) — status

| Component | Marks | Status |
|---|---|---|
| Git Practices | 4 | ✅ |
| DVC Usage | 4 | ✅ |
| Data Pipeline | 4 | ✅ |
| Pytest Coverage | 4 | ✅ |
| MLflow Tracking | 4 | ✅ |
| FastAPI Service | 4 | ✅ |
| Dockerization | 4 | ✅ |
| GitHub Actions | 4 | ✅ |
| Kubernetes Deployment | 4 | ✅ |
| Monitoring & Dashboards | 4 | ✅ |
