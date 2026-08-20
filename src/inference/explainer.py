"""
Model explainability using SHAP and feature importance.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import shap

from src.config import FEATURE_DESCRIPTIONS, FIGURES_DIR
from src.utils.logging import logger


def get_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Get feature importance from a tree-based model.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names: List of feature names.
        top_n: Number of top features to return.

    Returns:
        List of feature importance dictionaries.
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not support feature_importances_")
        return []

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    importance_list = []
    for i in indices[:top_n]:
        importance_list.append({
            "feature": feature_names[i],
            "importance": float(importances[i]),
            "description": FEATURE_DESCRIPTIONS.get(
                feature_names[i].split("_")[0]
                if "_" in feature_names[i]
                else feature_names[i],
                "Unknown feature",
            ),
        })

    return importance_list


def get_shap_explanation(
    model: Any,
    X_train: np.ndarray,
    X_sample: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Generate SHAP explanation for a single prediction.

    Args:
        model: Trained model.
        X_train: Training data for background.
        X_sample: Single sample to explain.
        feature_names: List of feature names.
        top_n: Number of top features to include.

    Returns:
        Dictionary with SHAP explanation details.
    """
    try:
        # Use a background sample for efficiency
        background_size = min(50, len(X_train))
        background = shap.sample(X_train, background_size, random_state=42)

        explainer = shap.KernelExplainer(
            model.predict_proba, background
        )

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample.reshape(1, -1))

        # For binary classification, shap_values is a list [class_0, class_1]
        if isinstance(shap_values, list):
            values = shap_values[1][0]  # Class 1 (disease)
        else:
            values = shap_values[0]

        # Sort by absolute importance
        indices = np.argsort(np.abs(values))[::-1]

        top_features = []
        for i in indices[:top_n]:
            feature_key = (
                feature_names[i].split("_")[0]
                if "_" in feature_names[i]
                else feature_names[i]
            )
            top_features.append({
                "feature": feature_names[i],
                "shap_value": float(values[i]),
                "absolute_impact": float(abs(values[i])),
                "direction": "increases" if values[i] > 0 else "decreases",
                "description": FEATURE_DESCRIPTIONS.get(
                    feature_key, "Unknown feature"
                ),
            })

        return {
            "top_contributing_features": top_features,
            "model_type": type(model).__name__,
            "disclaimer": (
                "SHAP values indicate how each feature contributes to the "
                "model's prediction. They do not imply causal relationships."
            ),
        }

    except Exception as e:
        logger.error(f"SHAP explanation failed: {str(e)}")
        return {
            "error": str(e),
            "top_contributing_features": [],
            "disclaimer": "SHAP explanation was unavailable for this prediction.",
        }


def get_global_feature_importance(
    model: Any,
    X_train: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Get global feature importance using SHAP.

    Args:
        model: Trained model.
        X_train: Training data.
        feature_names: List of feature names.

    Returns:
        Dictionary with global feature importance.
    """
    try:
        background_size = min(50, len(X_train))
        background = shap.sample(X_train, background_size, random_state=42)

        explainer = shap.KernelExplainer(
            model.predict_proba, background
        )

        # Use a subset for efficiency
        sample_size = min(100, len(X_train))
        X_summary = X_train[:sample_size]
        shap_values = explainer.shap_values(X_summary)

        if isinstance(shap_values, list):
            values = shap_values[1]  # Class 1
        else:
            values = shap_values

        # Average absolute SHAP values
        mean_abs_shap = np.mean(np.abs(values), axis=0)
        indices = np.argsort(mean_abs_shap)[::-1]

        importance_list = []
        for i in indices:
            feature_key = (
                feature_names[i].split("_")[0]
                if "_" in feature_names[i]
                else feature_names[i]
            )
            importance_list.append({
                "feature": feature_names[i],
                "mean_abs_shap": float(mean_abs_shap[i]),
                "description": FEATURE_DESCRIPTIONS.get(
                    feature_key, "Unknown feature"
                ),
            })

        return {
            "global_importance": importance_list,
            "n_samples": sample_size,
            "disclaimer": (
                "Global feature importance based on SHAP values. "
                "These indicate average contribution magnitude, not causation."
            ),
        }

    except Exception as e:
        logger.error(f"Global SHAP analysis failed: {str(e)}")
        return {"error": str(e), "global_importance": []}
