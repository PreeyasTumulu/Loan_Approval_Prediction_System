"""Split + impute + encode the validated raw data into model-ready CSVs.

The preprocessing pipeline (imputers + encoder) is fit on the train split only
and reused to transform val/test, then persisted so the training stage and the
serving API apply the exact same transform to new data.
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ingestion.load_data import load_raw_train
from src.utils.config import (
    CATEGORICAL_COLS, ID_COL, NUMERIC_COLS, PROCESSED_DIR,
    RANDOM_SEED, TARGET_COL, TEST_SIZE, VAL_SIZE,
)
from src.validation.schema import validate_or_raise


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_COLS),
        ("cat", categorical_pipe, CATEGORICAL_COLS),
    ])


def split_data(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df, test_size=VAL_SIZE + TEST_SIZE, random_state=RANDOM_SEED,
        stratify=df[TARGET_COL],
    )
    relative_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_size, random_state=RANDOM_SEED,
        stratify=temp_df[TARGET_COL],
    )
    return train_df, val_df, test_df


def _transform_split(preprocessor: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    features = preprocessor.transform(df[NUMERIC_COLS + CATEGORICAL_COLS])
    out = pd.DataFrame(features, columns=preprocessor.get_feature_names_out(), index=df.index)
    out[TARGET_COL] = (df[TARGET_COL] == "Y").astype(int).values
    out.insert(0, ID_COL, df[ID_COL].values)
    return out


def run() -> dict:
    df = load_raw_train()
    validate_or_raise(df)

    train_df, val_df, test_df = split_data(df)

    preprocessor = build_preprocessor()
    preprocessor.fit(train_df[NUMERIC_COLS + CATEGORICAL_COLS])

    splits = {
        "train": _transform_split(preprocessor, train_df),
        "val": _transform_split(preprocessor, val_df),
        "test": _transform_split(preprocessor, test_df),
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        split_df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    joblib.dump(preprocessor, PROCESSED_DIR / "preprocessor.joblib")

    return {name: len(d) for name, d in splits.items()}


if __name__ == "__main__":
    print(run())
