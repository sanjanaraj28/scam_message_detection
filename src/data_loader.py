"""
data_loader.py
Loads the raw dataset and performs a structural inspection. The raw CSV file
is never modified - all cleaning happens on an in-memory copy that is later
persisted to data/processed/cleaned_dataset.csv.
"""

import pandas as pd
from src.utils import RAW_DATA_PATH, get_logger

logger = get_logger(__name__)


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw, untouched dataset from disk."""
    df = pd.read_csv(path)
    logger.info(f"Loaded raw dataset from {path} -> shape={df.shape}")
    return df


def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Produce a structural summary of the dataset: shape, dtypes, missing
    values, duplicates and label distribution. Used both by the pipeline
    scripts and the EDA notebook so the numbers reported everywhere come
    from the exact same computation.
    """
    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isnull().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if "text" in df.columns:
        summary["unique_text_count"] = int(df["text"].nunique())
    if "label" in df.columns:
        summary["label_distribution"] = {
            int(k): int(v) for k, v in df["label"].value_counts().items()
        }
    logger.info(f"Dataset inspection: {summary}")
    return summary


if __name__ == "__main__":
    data = load_raw_data()
    inspect_dataset(data)
