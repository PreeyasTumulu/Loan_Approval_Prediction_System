from fastapi.testclient import TestClient

from app import PREDICTION_COUNTER, app

client = TestClient(app)

VALID_APPLICANT = {
    "Gender": "Male", "Married": "Yes", "Dependents": "0",
    "Education": "Graduate", "Self_Employed": "No",
    "ApplicantIncome": 5000, "CoapplicantIncome": 0,
    "LoanAmount": 128, "Loan_Amount_Term": 360,
    "Credit_History": 1, "Property_Area": "Urban",
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_endpoint_exposed():
    response = client.get("/metrics")
    assert response.status_code == 200


def test_predict_valid_payload(monkeypatch):
    monkeypatch.setattr(
        "app.predict_one",
        lambda applicant: {"loan_status": "Approved", "approval_probability": 0.87},
    )
    before = PREDICTION_COUNTER.labels(status="Approved")._value.get()

    response = client.post("/predict", json=VALID_APPLICANT)

    assert response.status_code == 200
    assert response.json() == {"loan_status": "Approved", "approval_probability": 0.87}
    after = PREDICTION_COUNTER.labels(status="Approved")._value.get()
    assert after == before + 1


def test_predict_rejects_invalid_category():
    bad_applicant = {**VALID_APPLICANT, "Gender": "Other"}
    response = client.post("/predict", json=bad_applicant)
    assert response.status_code == 422


def test_predict_missing_required_field():
    incomplete = {k: v for k, v in VALID_APPLICANT.items() if k != "Property_Area"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422
