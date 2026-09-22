import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df():
    """Small, clean synthetic dataset matching the raw schema (10 rows)."""
    return pd.DataFrame({
        "Loan_ID": [f"LP{i:03d}" for i in range(10)],
        "Gender": ["Male", "Female", "Male", "Female", "Male",
                   "Female", "Male", "Female", "Male", "Female"],
        "Married": ["Yes", "No", "Yes", "Yes", "No",
                    "Yes", "No", "Yes", "Yes", "No"],
        "Dependents": ["0", "1", "3+", "0", "2", "0", "1", "3+", "0", "1"],
        "Education": ["Graduate", "Not Graduate", "Graduate", "Graduate", "Not Graduate",
                      "Graduate", "Graduate", "Not Graduate", "Graduate", "Graduate"],
        "Self_Employed": ["No", "No", "Yes", "No", "No",
                          "No", "No", "Yes", "No", "No"],
        "ApplicantIncome": [5849, 4583, 3000, 2583, 6000, 5417, 2333, 3036, 4006, 12841],
        "CoapplicantIncome": [0, 1508, 0, 2358, 0, 4196, 1516, 2504, 1526, 10968],
        "LoanAmount": [150.0, 128, 66, 120, 141, 267, 95, 158, 168, 349],
        "Loan_Amount_Term": [360, 360, 360, 360, 360, 360, 360, 360, 360, 180],
        "Credit_History": [1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
        "Property_Area": ["Urban", "Rural", "Urban", "Urban", "Urban",
                          "Urban", "Urban", "Semiurban", "Urban", "Semiurban"],
        "Loan_Status": ["Y", "N", "Y", "Y", "Y", "Y", "Y", "N", "Y", "N"],
    })


@pytest.fixture
def sample_raw_df_large():
    """Larger synthetic dataset (40 rows, balanced classes, some NaNs) for
    split/imputation tests, where tiny fixtures would break stratification."""
    rng = np.random.default_rng(0)
    n = 40
    target = np.array(["Y"] * 24 + ["N"] * 16)
    rng.shuffle(target)

    df = pd.DataFrame({
        "Loan_ID": [f"LP{i:03d}" for i in range(n)],
        "Gender": rng.choice(["Male", "Female"], n),
        "Married": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["0", "1", "2", "3+"], n),
        "Education": rng.choice(["Graduate", "Not Graduate"], n),
        "Self_Employed": rng.choice(["Yes", "No"], n),
        "ApplicantIncome": rng.integers(1500, 10000, n),
        "CoapplicantIncome": rng.integers(0, 5000, n),
        "LoanAmount": rng.integers(50, 400, n).astype(float),
        "Loan_Amount_Term": rng.choice([360, 180, 120], n),
        "Credit_History": rng.choice([0, 1], n).astype(float),
        "Property_Area": rng.choice(["Urban", "Semiurban", "Rural"], n),
        "Loan_Status": target,
    })

    # inject missing values to exercise the imputers
    df.loc[df.index[:3], "LoanAmount"] = np.nan
    df.loc[df.index[3:5], "Credit_History"] = np.nan
    df.loc[df.index[5:7], "Gender"] = np.nan
    df.loc[df.index[7:9], "Self_Employed"] = np.nan
    return df
