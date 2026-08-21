"""
Data validation for the Heart Disease dataset.
"""
from typing import List

import pandas as pd

from ml.config import (
    COLUMN_RANGES,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    TARGET_COLUMN,
)
from ml.utils.logging import logger


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


def validate_dataset(df: pd.DataFrame) -> List[str]:
    """
    Validate the dataset for quality and consistency.

    Checks:
    - Expected columns present
    - Data types correct
    - Missing values
    - Duplicate rows
    - Invalid values (out of range)
    - Target column values

    Args:
        df: Input DataFrame to validate.

    Returns:
        List of warning messages (non-fatal issues).

    Raises:
        DataValidationError: If critical validation issues are found.
    """
    warnings: List[str] = []
    errors: List[str] = []

    # Check expected columns
    expected_cols = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN]
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    extra_cols = set(df.columns) - set(expected_cols)
    if extra_cols:
        warnings.append(f"Unexpected columns found: {extra_cols}")

    if errors:
        raise DataValidationError(
            f"Critical validation errors: {'; '.join(errors)}"
        )

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        missing_info = missing[missing > 0].to_dict()
        warnings.append(f"Missing values detected: {missing_info}")

    # Check for duplicates
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        warnings.append(
            f"{n_duplicates} duplicate row(s) detected. "
            f"These will be removed during preprocessing."
        )

    # Check numerical column ranges
    for col in NUMERICAL_COLUMNS + [TARGET_COLUMN]:
        if col in df.columns and col in COLUMN_RANGES:
            min_val, max_val = COLUMN_RANGES[col]
            actual_min = df[col].min()
            actual_max = df[col].max()
            if actual_min < min_val:
                warnings.append(
                    f"Column '{col}': minimum value {actual_min} "
                    f"is below expected range [{min_val}, {max_val}]"
                )
            if actual_max > max_val:
                warnings.append(
                    f"Column '{col}': maximum value {actual_max} "
                    f"exceeds expected range [{min_val}, {max_val}]"
                )

    # Check target column values
    target_values = set(df[TARGET_COLUMN].unique())
    if not target_values.issubset({0, 1}):
        errors.append(
            f"Target column contains unexpected values: {target_values}. "
            f"Expected only 0 and 1."
        )

    # Check categorical columns
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            unique_vals = sorted(df[col].unique())
            if col in COLUMN_RANGES:
                min_val, max_val = COLUMN_RANGES[col]
                out_of_range = [
                    v for v in unique_vals if v < min_val or v > max_val
                ]
                if out_of_range:
                    warnings.append(
                        f"Column '{col}' has values outside expected range "
                        f"[{min_val}, {max_val}]: {out_of_range}"
                    )

    if errors:
        raise DataValidationError(
            f"Critical validation errors: {'; '.join(errors)}"
        )

    # Log warnings
    for warning in warnings:
        logger.warning(f"Data validation: {warning}")

    if not warnings:
        logger.info("Dataset validation passed with no warnings")

    return warnings
