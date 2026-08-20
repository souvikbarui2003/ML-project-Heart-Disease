"""
Data loading and initial inspection for the Heart Disease dataset.
"""
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import (
    DATASET_FILENAME,
    RANDOM_SEED,
    RAW_DATA_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.utils.logging import logger


def load_dataset(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Load the heart disease dataset from CSV.

    Args:
        data_dir: Directory containing the raw dataset.

    Returns:
        DataFrame with the loaded dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset is empty.
    """
    file_path = data_dir / DATASET_FILENAME

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. "
            f"Please ensure {DATASET_FILENAME} exists in {data_dir}"
        )

    logger.info(f"Loading dataset from {file_path}")

    # Handle BOM-encoded CSV if present
    df = pd.read_csv(file_path, encoding="utf-8-sig")

    if df.empty:
        raise ValueError(f"Dataset at {file_path} is empty")

    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")

    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with dataset information.
    """
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "target_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
        "target_positive_ratio": float(df[TARGET_COLUMN].mean()),
    }

    logger.info(f"Dataset shape: {info['shape']}")
    logger.info(f"Missing values: {sum(info['missing_values'].values())}")
    logger.info(f"Duplicate rows: {info['duplicates']}")
    logger.info(
        f"Target distribution: "
        f"{info['target_distribution']}"
    )

    return info
