import pytest

from src.validation.schema import DataValidationError, validate_or_raise, validate_raw_data


def test_valid_data_passes(sample_raw_df):
    report = validate_raw_data(sample_raw_df)
    assert report["valid"] is True
    assert report["issues"] == []
    assert report["n_rows"] == 10


def test_missing_target_detected(sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[0, "Loan_Status"] = None
    report = validate_raw_data(df)
    assert report["valid"] is False
    assert any("missing target" in issue for issue in report["issues"])


def test_unexpected_category_detected(sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[0, "Property_Area"] = "Suburb"
    report = validate_raw_data(df)
    assert report["valid"] is False
    assert any("Property_Area" in issue for issue in report["issues"])


def test_missing_required_column_detected(sample_raw_df):
    df = sample_raw_df.drop(columns=["Credit_History"])
    report = validate_raw_data(df)
    assert report["valid"] is False
    assert any("missing required columns" in issue for issue in report["issues"])


def test_duplicate_id_detected(sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[1, "Loan_ID"] = df.loc[0, "Loan_ID"]
    report = validate_raw_data(df)
    assert report["valid"] is False
    assert any("duplicate" in issue for issue in report["issues"])


def test_validate_or_raise_raises_on_bad_data(sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[0, "Loan_Status"] = "Maybe"
    with pytest.raises(DataValidationError):
        validate_or_raise(df)


def test_validate_or_raise_passes_on_good_data(sample_raw_df):
    validate_or_raise(sample_raw_df)  # should not raise
