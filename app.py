"""FastAPI entrypoint for the loan approval prediction service."""
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from src.prediction.predict import predict_one

app = FastAPI(title="Loan Approval Prediction API")
Instrumentator().instrument(app).expose(app)  # GET /metrics for Prometheus

# ML-specific signal: watches the MODEL's behavior (is it drifting toward
# approving/rejecting everything?), not just whether the service is up.
PREDICTION_COUNTER = Counter(
    "loan_predictions_total", "Loan prediction outcomes by status", ["status"],
)


class LoanApplicationRequest(BaseModel):
    Gender: Literal["Male", "Female"]
    Married: Literal["Yes", "No"]
    Dependents: Literal["0", "1", "2", "3+"]
    Education: Literal["Graduate", "Not Graduate"]
    Self_Employed: Literal["Yes", "No"]
    ApplicantIncome: float = Field(ge=0)
    CoapplicantIncome: float = Field(ge=0)
    LoanAmount: Optional[float] = Field(default=None, ge=0)
    Loan_Amount_Term: Optional[float] = Field(default=None, ge=0)
    Credit_History: Optional[float] = Field(default=None, ge=0, le=1)
    Property_Area: Literal["Urban", "Semiurban", "Rural"]


class LoanApplicationResponse(BaseModel):
    loan_status: Literal["Approved", "Rejected"]
    approval_probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=LoanApplicationResponse)
def predict(applicant: LoanApplicationRequest):
    try:
        result = predict_one(applicant.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found — run the training pipeline first.",
        ) from exc
    PREDICTION_COUNTER.labels(status=result["loan_status"]).inc()
    return result
