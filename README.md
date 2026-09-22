# Loan Approval Prediction System

MLOps end-term project (STDE 301) — a machine learning system that predicts
whether a loan application should be approved or rejected, covering the full
MLOps lifecycle: data versioning, experiment tracking, testing,
containerization, CI/CD, deployment, and monitoring.

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
│   ├── raw/            # DVC-tracked source data
│   └── processed/      # DVC-tracked, feature-engineered data
├── notebooks/           # exploration only, not part of the pipeline
├── src/
│   ├── ingestion/       # load raw data
│   ├── validation/      # schema/null/range checks
│   ├── transformation/  # imputation, encoding, feature engineering
│   ├── training/        # model training + MLflow logging
│   ├── prediction/      # inference logic used by the API
│   └── utils/           # shared helpers
├── tests/               # pytest suite, synthetic fixtures only
├── deployment/
│   ├── docker/          # Dockerfile(s)
│   └── kubernetes/      # Deployment/Service manifests
├── monitoring/          # Prometheus + Grafana config
├── .github/workflows/   # CI/CD pipelines
├── dvc.yaml             # DVC pipeline stages
├── requirements.txt
├── Dockerfile
└── app.py               # FastAPI entrypoint
```

## Status

Scaffold only — pipeline stages are being built in order (see rubric below).

## Rubric (40 marks)

| Component | Marks |
|---|---|
| Git Practices | 4 |
| DVC Usage | 4 |
| Data Pipeline | 4 |
| Pytest Coverage | 4 |
| MLflow Tracking | 4 |
| FastAPI Service | 4 |
| Dockerization | 4 |
| GitHub Actions | 4 |
| Kubernetes Deployment | 4 |
| Monitoring & Dashboards | 4 |
