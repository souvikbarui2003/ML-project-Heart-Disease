"""
Unit tests for model training and evaluation.
"""
import numpy as np
import pytest

from src.evaluation.metrics import calculate_metrics, evaluate_all_models
from src.models.trainer import get_model_configs, select_best_model, train_all_models


@pytest.fixture
def sample_data():
    """Create sample training/test data."""
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(50, 10)
    y_test = np.random.randint(0, 2, 50)
    return X_train, X_test, y_train, y_test


class TestModelConfigs:
    """Tests for model configurations."""

    def test_get_configs_returns_dict(self):
        """Test that get_model_configs returns a dictionary."""
        configs = get_model_configs()
        assert isinstance(configs, dict)
        assert len(configs) >= 6

    def test_configs_have_required_keys(self):
        """Test that each config has required keys."""
        configs = get_model_configs()
        for name, config in configs.items():
            assert "model" in config
            assert "params" in config


class TestModelTraining:
    """Tests for model training."""

    def test_train_all_models(self, sample_data):
        """Test that all models train successfully."""
        X_train, _, y_train, _, = sample_data
        feature_names = [f"feature_{i}" for i in range(10)]

        best_models, best_params, training_results = train_all_models(
            X_train, y_train, feature_names, tune_hyperparameters=False
        )

        assert len(best_models) >= 6
        assert len(training_results) >= 6

    def test_select_best_model(self):
        """Test best model selection."""
        training_results = [
            {"model_name": "A", "cv_score": 0.85, "training_time": 1.0},
            {"model_name": "B", "cv_score": 0.90, "training_time": 2.0},
            {"model_name": "C", "cv_score": 0.80, "training_time": 0.5},
        ]
        best = select_best_model(training_results)
        assert best == "B"


class TestModelEvaluation:
    """Tests for model evaluation."""

    def test_calculate_metrics(self):
        """Test metric calculation."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        y_prob = np.array([0.1, 0.6, 0.9, 0.8, 0.2, 0.4, 0.15, 0.85])

        metrics = calculate_metrics(y_true, y_pred, y_prob, "test_model")

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall_sensitivity" in metrics
        assert "specificity" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_evaluate_all_models(self, sample_data):
        """Test evaluation of multiple models."""
        X_train, X_test, y_train, y_test = sample_data
        feature_names = [f"feature_{i}" for i in range(10)]

        # Train a simple model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        models = {"RandomForest": model}
        all_metrics, all_curves = evaluate_all_models(models, X_test, y_test)

        assert "RandomForest" in all_metrics
        assert "RandomForest" in all_curves
