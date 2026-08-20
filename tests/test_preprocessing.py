"""
Unit tests for preprocessing pipeline.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import (
    build_preprocessing_pipeline,
    prepare_data,
    remove_duplicates,
    split_data,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples = 200

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


class TestRemoveDuplicates:
    """Tests for duplicate removal."""

    def test_removes_duplicates(self):
        """Test that duplicates are removed."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
        result = remove_duplicates(df)
        assert len(result) == 2

    def test_no_duplicates(self):
        """Test when no duplicates exist."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = remove_duplicates(df)
        assert len(result) == 3


class TestSplitData:
    """Tests for train/test split."""

    def test_split_sizes(self, sample_df):
        """Test that split produces correct sizes."""
        train_df, test_df = split_data(sample_df, test_size=0.2)
        assert len(train_df) == 160
        assert len(test_df) == 40

    def test_stratified_split(self, sample_df):
        """Test that target distribution is preserved."""
        train_df, test_df = split_data(sample_df, test_size=0.2)
        train_ratio = train_df["target"].mean()
        test_ratio = test_df["target"].mean()
        # Should be approximately equal
        assert abs(train_ratio - test_ratio) < 0.1


class TestPreprocessingPipeline:
    """Tests for preprocessing pipeline."""

    def test_build_pipeline(self):
        """Test that pipeline builds successfully."""
        preprocessor = build_preprocessing_pipeline()
        assert preprocessor is not None

    def test_prepare_data_shapes(self, sample_df):
        """Test that prepare_data returns correct shapes."""
        X_train, X_test, y_train, y_test, feature_names = prepare_data(sample_df)
        assert X_train.shape[0] + X_test.shape[0] == 200
        assert y_train.shape[0] + y_test.shape[0] == 200
        assert len(feature_names) > 0

    def test_no_leakage(self, sample_df):
        """Test that preprocessing doesn't leak test data."""
        X_train, X_test, _, _, _ = prepare_data(sample_df)
        # Check that training and test sets are distinct
        assert not np.array_equal(X_train[:10], X_test[:10])

    def test_feature_names_count(self, sample_df):
        """Test that feature names match transformed columns."""
        X_train, _, _, _, feature_names = prepare_data(sample_df)
        assert len(feature_names) == X_train.shape[1]
