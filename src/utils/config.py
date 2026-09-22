"""Central column/path config shared across the pipeline stages."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_TRAIN_PATH = ROOT / "data" / "raw" / "loan_train.csv"
RAW_UNLABELED_PATH = ROOT / "data" / "raw" / "loan_test_unlabeled.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
# Git-tracked (not DVC-tracked) copy of the winning model + preprocessor —
# this is what Docker/Kubernetes/CI actually serve, so a plain `git clone`
# is enough to build the image without needing a reachable DVC remote.
MODELS_DIR = ROOT / "models"

ID_COL = "Loan_ID"
TARGET_COL = "Loan_Status"

CATEGORICAL_COLS = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area",
]
NUMERIC_COLS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History",
]
ALL_FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

ALLOWED_CATEGORIES = {
    "Gender": {"Male", "Female"},
    "Married": {"Yes", "No"},
    "Dependents": {"0", "1", "2", "3+"},
    "Education": {"Graduate", "Not Graduate"},
    "Self_Employed": {"Yes", "No"},
    "Property_Area": {"Urban", "Semiurban", "Rural"},
}
ALLOWED_TARGET_VALUES = {"Y", "N"}

RANDOM_SEED = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15
