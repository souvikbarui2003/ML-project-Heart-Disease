"""
Preprocessing pipeline for the Heart Disease dataset.

This module builds a sklearn Pipeline and ColumnTransformer that:
- Handles missing values (SimpleImputer)
- Encodes categorical features (OneHotEncoder)
- Scales numerical features (StandardScaler)
- Removes duplicate rows
- Preserves all transformations in a single fitted object

IMPORTANT: This pipeline must be fit ONLY on training data to prevent
data leakage. The fit_transform / transform pattern enforces this.
"""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.utils.logging import logger


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with duplicates removed.
    """
    n_before = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df_clean)
    if n_removed > 0:
        logger.info(f"Removed {n_removed} duplicate row(s)")
    return df_clean


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test sets with stratification.

    Args:
        df: Input DataFrame.
        test_size: Proportion of test data.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )
    logger.info(
        f"Train/test split: {len(train_df)} train, {len(test_df)} test"
    )
    return train_df, test_df


def build_preprocessing_pipeline() -> Tuple[Pipeline, ColumnTransformer]:
    """
    Build the preprocessing pipeline.

    The pipeline consists of:
    1. SimpleImputer for missing values
    2. StandardScaler for numerical features
    3. OneHotEncoder for categorical features

    Returns:
        Tuple of (full_pipeline, preprocessor).
    """
    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    drop="first",  # Avoid multicollinearity
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )

    logger.info("Preprocessing pipeline created")
    return preprocessor


def prepare_data(
    df: pd.DataFrame,
    remove_dupes: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Prepare data for modeling: clean, split, and preprocess.

    IMPORTANT: Preprocessing is fit ONLY on training data to prevent
    data leakage.

    Args:
        df: Raw DataFrame.
        remove_dupes: Whether to remove duplicate rows.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names).
    """
    # Step 1: Remove duplicates
    if remove_dupes:
        df = remove_duplicates(df)

    # Step 2: Split FIRST (before any fitting)
    train_df, test_df = split_data(df)

    # Step 3: Separate features and target
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].values
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].values

    # Step 4: Build and fit preprocessing pipeline on training data ONLY
    preprocessor = build_preprocessing_pipeline()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Step 5: Get feature names after preprocessing
    feature_names = _get_feature_names(preprocessor, X_train.columns)

    logger.info(f"Training features shape: {X_train_processed.shape}")
    logger.info(f"Test features shape: {X_test_processed.shape}")
    logger.info(f"Number of features after preprocessing: {len(feature_names)}")

    return X_train_processed, X_test_processed, y_train, y_test, feature_names


def _get_feature_names(
    preprocessor: ColumnTransformer,
    original_columns: pd.Index,
) -> list:
    """
    Extract feature names from the fitted preprocessor.

    Args:
        preprocessor: Fitted ColumnTransformer.
        original_columns: Original DataFrame column names.

    Returns:
        List of feature names after preprocessing.
    """
    feature_names = []

    # Get names from each transformer
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue

        if hasattr(transformer, "named_steps"):
            # Pipeline
            last_step = list(transformer.named_steps.values())[-1]
        else:
            last_step = transformer

        if hasattr(last_step, "get_feature_names_out"):
            names = last_step.get_feature_names_out()
            feature_names.extend(names)
        else:
            feature_names.extend(columns)

    return list(feature_names)
