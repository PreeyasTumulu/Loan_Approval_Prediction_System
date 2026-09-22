"""Schema/quality checks the raw data must pass before it enters the pipeline."""
import json
import sys

import pandas as pd

from src.ingestion.load_data import load_raw_train
from src.utils.config import (
    ALL_FEATURE_COLS, ALLOWED_CATEGORIES, ALLOWED_TARGET_VALUES,
    ID_COL, NUMERIC_COLS, REPORTS_DIR, TARGET_COL,
)


class DataValidationError(Exception):
    pass


def validate_raw_data(df: pd.DataFrame) -> dict:
    issues = []

    required_cols = [ID_COL, TARGET_COL] + ALL_FEATURE_COLS
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"missing required columns: {missing_cols}")
        # can't check anything else meaningfully without the columns
        return {"valid": False, "issues": issues, "n_rows": len(df)}

    if df[ID_COL].duplicated().any():
        issues.append(f"{df[ID_COL].duplicated().sum()} duplicate {ID_COL} values")

    if df[TARGET_COL].isna().any():
        issues.append(f"{df[TARGET_COL].isna().sum()} rows with missing target ({TARGET_COL})")
    bad_target = set(df[TARGET_COL].dropna().unique()) - ALLOWED_TARGET_VALUES
    if bad_target:
        issues.append(f"unexpected {TARGET_COL} values: {bad_target}")

    for col, allowed in ALLOWED_CATEGORIES.items():
        observed = set(df[col].dropna().unique())
        unexpected = observed - allowed
        if unexpected:
            issues.append(f"unexpected values in {col}: {unexpected}")

    for col in NUMERIC_COLS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        non_numeric = df[col].notna() & coerced.isna()
        if non_numeric.any():
            issues.append(f"{col} has {non_numeric.sum()} non-numeric values")
        elif (coerced.dropna() < 0).any():
            issues.append(f"{col} has negative values")

    fully_empty_rows = df.drop(columns=[ID_COL]).isna().all(axis=1).sum()
    if fully_empty_rows:
        issues.append(f"{fully_empty_rows} fully empty rows")

    return {"valid": len(issues) == 0, "issues": issues, "n_rows": len(df)}


def validate_or_raise(df: pd.DataFrame) -> dict:
    report = validate_raw_data(df)
    if not report["valid"]:
        raise DataValidationError("; ".join(report["issues"]))
    return report


def main() -> int:
    df = load_raw_train()
    report = validate_raw_data(df)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "data_validation.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
