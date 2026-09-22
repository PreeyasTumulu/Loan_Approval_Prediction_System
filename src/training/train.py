"""Train candidate models on the processed splits, track every run in MLflow,
register the best one, and export it as a plain artifact for serving.

MLflow 3.x refuses a bare './mlruns' file store for the tracking backend, so
we use a SQLite backend (also required for the Model Registry).
"""
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

from src.utils.config import ID_COL, PROCESSED_DIR, RANDOM_SEED, REPORTS_DIR, TARGET_COL

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "loan-approval"
REGISTERED_MODEL_NAME = "loan-approval-model"

CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=5000, random_state=RANDOM_SEED),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED),
}


def _xy(df: pd.DataFrame):
    return df.drop(columns=[ID_COL, TARGET_COL]), df[TARGET_COL]


def evaluate(model, X, y) -> dict:
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }


def run(train_path=None, val_path=None) -> dict:
    train_path = train_path or PROCESSED_DIR / "train.csv"
    val_path = val_path or PROCESSED_DIR / "val.csv"

    X_train, y_train = _xy(pd.read_csv(train_path))
    X_val, y_val = _xy(pd.read_csv(val_path))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = {}
    best_name, best_f1, best_model_uri, best_model_obj = None, -1.0, None, None

    for name, model in CANDIDATES.items():
        with mlflow.start_run(run_name=name) as run_ctx:
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_val, y_val)

            mlflow.log_param("model_type", name)
            mlflow.log_params({k: v for k, v in model.get_params().items()})
            mlflow.log_metrics(metrics)
            model_info = mlflow.sklearn.log_model(
                model, name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

            results[name] = {**metrics, "run_id": run_ctx.info.run_id}

            if metrics["f1"] > best_f1:
                best_name, best_f1 = name, metrics["f1"]
                best_model_uri, best_model_obj = model_info.model_uri, model

    registered = mlflow.register_model(best_model_uri, REGISTERED_MODEL_NAME)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model_obj, PROCESSED_DIR / "model.joblib")

    results["best_model"] = best_name
    results["registered_model_name"] = REGISTERED_MODEL_NAME
    results["registered_version"] = registered.version

    with open(REPORTS_DIR / "train_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    print(run())
