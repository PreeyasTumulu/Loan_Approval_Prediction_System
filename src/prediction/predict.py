"""Load the trained model + preprocessor once and run inference on new applicants."""
import functools

import joblib
import pandas as pd

from src.utils.config import CATEGORICAL_COLS, MODELS_DIR, NUMERIC_COLS


@functools.lru_cache(maxsize=1)
def _load_artifacts():
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / "model.joblib")
    return preprocessor, model


def predict_one(applicant: dict) -> dict:
    preprocessor, model = _load_artifacts()
    row = pd.DataFrame([applicant])[NUMERIC_COLS + CATEGORICAL_COLS]
    features = pd.DataFrame(
        preprocessor.transform(row), columns=preprocessor.get_feature_names_out(),
    )
    approval_probability = float(model.predict_proba(features)[0, 1])
    loan_status = "Approved" if approval_probability >= 0.5 else "Rejected"
    return {"loan_status": loan_status, "approval_probability": round(approval_probability, 4)}
