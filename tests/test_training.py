import joblib
import numpy as np
import pandas as pd

from src.training.train import run
from src.utils.config import ID_COL, TARGET_COL


def _make_processed_df(n, seed):
    rng = np.random.default_rng(seed)
    target = np.array([0] * (n // 2) + [1] * (n - n // 2))
    rng.shuffle(target)
    df = pd.DataFrame({
        ID_COL: [f"LP{i:03d}" for i in range(n)],
        "num__ApplicantIncome": rng.uniform(1000, 10000, n),
        "num__LoanAmount": rng.uniform(50, 400, n),
        "cat__Property_Area_Urban": rng.integers(0, 2, n).astype(float),
    })
    df[TARGET_COL] = target
    return df


def test_run_trains_tracks_and_exports_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # isolates mlflow.db / mlruns from the real project

    train_df = _make_processed_df(30, seed=1)
    val_df = _make_processed_df(12, seed=2)
    train_path = tmp_path / "train.csv"
    val_path = tmp_path / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    joblib.dump({"placeholder": True}, tmp_path / "preprocessor.joblib")

    models_dir = tmp_path / "models"
    monkeypatch.setattr("src.training.train.PROCESSED_DIR", tmp_path)
    monkeypatch.setattr("src.training.train.REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr("src.training.train.MODELS_DIR", models_dir)

    results = run(train_path=train_path, val_path=val_path)

    assert results["best_model"] in {"logistic_regression", "random_forest"}
    for name in ("logistic_regression", "random_forest"):
        assert 0.0 <= results[name]["f1"] <= 1.0
        assert 0.0 <= results[name]["roc_auc"] <= 1.0

    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "reports" / "train_metrics.json").exists()
    assert (models_dir / "model.joblib").exists()
    assert (models_dir / "preprocessor.joblib").exists()
