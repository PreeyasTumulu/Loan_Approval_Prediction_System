import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.prediction.predict import _load_artifacts, predict_one
from src.transformation.features import build_preprocessor
from src.utils.config import CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL


def test_predict_one_returns_valid_response(sample_raw_df_large, tmp_path, monkeypatch):
    df = sample_raw_df_large
    X_raw = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = (df[TARGET_COL] == "Y").astype(int)

    preprocessor = build_preprocessor()
    X = pd.DataFrame(
        preprocessor.fit_transform(X_raw), columns=preprocessor.get_feature_names_out(),
    )
    model = LogisticRegression(max_iter=1000).fit(X, y)

    joblib.dump(preprocessor, tmp_path / "preprocessor.joblib")
    joblib.dump(model, tmp_path / "model.joblib")

    monkeypatch.setattr("src.prediction.predict.PROCESSED_DIR", tmp_path)
    _load_artifacts.cache_clear()

    applicant = {
        "Gender": "Male", "Married": "Yes", "Dependents": "0",
        "Education": "Graduate", "Self_Employed": "No",
        "ApplicantIncome": 5000, "CoapplicantIncome": 0,
        "LoanAmount": 128, "Loan_Amount_Term": 360,
        "Credit_History": 1, "Property_Area": "Urban",
    }
    result = predict_one(applicant)

    assert result["loan_status"] in {"Approved", "Rejected"}
    assert 0.0 <= result["approval_probability"] <= 1.0

    _load_artifacts.cache_clear()  # don't leak into other tests
