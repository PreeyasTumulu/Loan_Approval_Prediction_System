import numpy as np
import pandas as pd

from src.transformation.features import build_preprocessor, split_data, run
from src.utils.config import CATEGORICAL_COLS, ID_COL, NUMERIC_COLS, TARGET_COL


def test_split_data_covers_all_rows_without_overlap(sample_raw_df_large):
    train_df, val_df, test_df = split_data(sample_raw_df_large)

    train_ids = set(train_df[ID_COL])
    val_ids = set(val_df[ID_COL])
    test_ids = set(test_df[ID_COL])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(sample_raw_df_large)
    # roughly 70/15/15 given a 40-row set
    assert len(train_df) > len(val_df)
    assert len(train_df) > len(test_df)


def test_split_data_preserves_class_balance(sample_raw_df_large):
    train_df, val_df, test_df = split_data(sample_raw_df_large)
    for split_df in (train_df, val_df, test_df):
        assert set(split_df[TARGET_COL].unique()) <= {"Y", "N"}
        assert split_df[TARGET_COL].nunique() == 2


def test_preprocessor_removes_all_nulls(sample_raw_df_large):
    train_df, val_df, _ = split_data(sample_raw_df_large)
    preprocessor = build_preprocessor()
    preprocessor.fit(train_df[NUMERIC_COLS + CATEGORICAL_COLS])

    transformed = preprocessor.transform(val_df[NUMERIC_COLS + CATEGORICAL_COLS])
    assert not np.isnan(transformed).any()


def test_run_writes_processed_splits_and_preprocessor(sample_raw_df_large, tmp_path, monkeypatch):
    monkeypatch.setattr("src.transformation.features.load_raw_train", lambda: sample_raw_df_large)
    monkeypatch.setattr("src.transformation.features.PROCESSED_DIR", tmp_path)

    counts = run()

    assert sum(counts.values()) == len(sample_raw_df_large)
    for name in ("train", "val", "test"):
        assert (tmp_path / f"{name}.csv").exists()
    assert (tmp_path / "preprocessor.joblib").exists()

    train_out = pd.read_csv(tmp_path / "train.csv")
    assert set(train_out[TARGET_COL].unique()) <= {0, 1}
    assert not train_out.isna().any().any()
