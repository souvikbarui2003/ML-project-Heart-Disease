"""
Unit tests for data loading and validation.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    TARGET_COLUMN,
)
from src.data.loader import get_dataset_info, load_dataset
from src.data.validation import DataValidationError, validate_dataset


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "age": np.random.randint(29, 78, n_samples),
        "sex": np.random.randint(0, 2, n_samples),
        "cp": np.random.randint(0, 4, n_samples),
        "trestbps": np.random.randint(94, 201, n_samples),
        "chol": np.random.randint(126, 565, n_samples),
        "fbs": np.random.randint(0, 2, n_samples),
        "restecg": np.random.randint(0, 3, n_samples),
        "thalach": np.random.randint(71, 203, n_samples),
        "exang": np.random.randint(0, 2, n_samples),
        "oldpeak": np.random.uniform(0, 6.2, n_samples),
        "slope": np.random.randint(0, 3, n_samples),
        "ca": np.random.randint(0, 4, n_samples),
        "thal": np.random.randint(0, 4, n_samples),
        "target": np.random.randint(0, 2, n_samples),
    }
    return pd.DataFrame(data)


class TestDataLoader:
    """Tests for data loading functions."""

    def test_load_dataset_returns_dataframe(self, sample_df, tmp_path):
        """Test that load_dataset returns a DataFrame."""
        # Save sample data
        csv_path = tmp_path / "heart.csv"
        sample_df.to_csv(csv_path, index=False)

        # Load and verify
        df = load_dataset(tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_df)

    def test_load_dataset_missing_file(self, tmp_path):
        """Test that load_dataset raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path)

    def test_dataset_info(self, sample_df):
        """Test that get_dataset_info returns expected keys."""
        info = get_dataset_info(sample_df)
        assert "shape" in info
        assert "columns" in info
        assert "missing_values" in info
        assert "duplicates" in info
        assert "target_distribution" in info


class TestDataValidation:
    """Tests for data validation functions."""

    def test_valid_dataset(self, sample_df):
        """Test validation passes for valid dataset."""
        warnings = validate_dataset(sample_df)
        assert isinstance(warnings, list)

    def test_missing_column(self, sample_df):
        """Test validation catches missing columns."""
        df_incomplete = sample_df.drop(columns=["age"])
        with pytest.raises(DataValidationError):
            validate_dataset(df_incomplete)

    def test_invalid_target_values(self, sample_df):
        """Test validation catches invalid target values."""
        df_bad = sample_df.copy()
        df_bad[TARGET_COLUMN] = df_bad[TARGET_COLUMN] * 2  # Values will be 0 or 2
        with pytest.raises(DataValidationError):
            validate_dataset(df_bad)

    def test_out_of_range_values(self, sample_df):
        """Test validation warns about out-of-range values."""
        df_bad = sample_df.copy()
        df_bad["age"] = 150  # Above expected range
        warnings = validate_dataset(df_bad)
        assert any("age" in w for w in warnings)

    def test_missing_values_detected(self, sample_df):
        """Test validation detects missing values."""
        df_missing = sample_df.copy()
        df_missing.loc[0, "age"] = np.nan
        warnings = validate_dataset(df_missing)
        assert any("Missing values" in w for w in warnings)
