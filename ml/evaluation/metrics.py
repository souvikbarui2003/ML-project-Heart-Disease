"""
Comprehensive model evaluation metrics for healthcare applications.
"""
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    average_precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml.utils.logging import logger


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "model",
) -> Dict[str, Any]:
    """
    Calculate comprehensive healthcare-relevant metrics.

    For a heart disease screening system, we emphasize:
    - Recall/Sensitivity (catching positive cases)
    - Specificity (correctly identifying negatives)
    - False negative rate (missing disease cases)

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities.
        model_name: Name of the model.

    Returns:
        Dictionary of calculated metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall_sensitivity": float(recall_score(y_true, y_pred)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "f1_score": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
    }

    logger.info(
        f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
        f"ROC-AUC: {metrics['roc_auc']:.4f}, "
        f"Recall: {metrics['recall_sensitivity']:.4f}, "
        f"Specificity: {metrics['specificity']:.4f}"
    )

    return metrics


def evaluate_all_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Evaluate all trained models on test data.

    Args:
        models: Dictionary of trained models.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Tuple of (all_metrics, all_curves) where all_curves maps model names
        to (fpr, tpr, thresholds) for ROC curves.
    """
    all_metrics = {}
    all_curves = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, y_pred, y_prob, name)
        all_metrics[name] = metrics

        # ROC curve data
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        all_curves[name] = (fpr, tpr, thresholds)

    return all_metrics, all_curves


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
) -> str:
    """
    Generate a detailed classification report.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        model_name: Name of the model.

    Returns:
        Classification report string.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=["No Disease (0)", "Disease (1)"],
        digits=4,
    )
    logger.info(f"\nClassification Report for {model_name}:\n{report}")
    return report
