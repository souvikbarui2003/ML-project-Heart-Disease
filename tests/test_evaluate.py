"""
Tests for model evaluation metrics.
"""
import numpy as np
import pytest

from src.evaluation.metrics import calculate_metrics, get_classification_report


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_calculate_metrics(self):
        """Test metric calculation with known values."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.6, 0.8, 0.7, 0.2, 0.4, 0.3, 0.9, 0.75, 0.35])

        metrics = calculate_metrics(y_true, y_pred, y_prob, "test")

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall_sensitivity" in metrics
        assert "specificity" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert "confusion_matrix" in metrics

        # Verify accuracy
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.1, 0.9, 1.0])

        metrics = calculate_metrics(y_true, y_pred, y_prob, "test")

        assert metrics["accuracy"] == 1.0
        assert metrics["recall_sensitivity"] == 1.0
        assert metrics["specificity"] == 1.0

    def test_classification_report(self):
        """Test classification report generation."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])

        report = get_classification_report(y_true, y_pred, "test")
        assert isinstance(report, str)
        assert "No Disease" in report
        assert "Disease" in report
