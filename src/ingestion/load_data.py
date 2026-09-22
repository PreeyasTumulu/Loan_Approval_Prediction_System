"""Raw data loading — the only place that touches the CSV on disk."""
import pandas as pd

from src.utils.config import RAW_TRAIN_PATH


def load_raw_train(path=RAW_TRAIN_PATH) -> pd.DataFrame:
    return pd.read_csv(path)
