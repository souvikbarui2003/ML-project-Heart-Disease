"""
Model inference and prediction serving.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import joblib

from src.config import MODEL_DIR, MODEL_VERSION, RISK_THRESHOLDS, TARGET_COLUMN
from src.utils.logging import logger


def load_model(
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load a persisted model and its metadata.

    Args:
        model_path: Path to the model file. If None, loads the latest model.

    Returns:
        Dictionary containing model, preprocessor, and metadata.

    Raises:
        FileNotFoundError: If no model is found.
    """
    if model_path is None:
        model_path = MODEL_DIR / "best_model.joblib"
        metadata_path = MODEL_DIR / "model_metadata.json"
    else:
        metadata_path = model_path.parent / "model_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. "
            f"Please train a model first: python -m src.train"
        )

    logger.info(f"Loading model from {model_path}")

    artifact = joblib.load(model_path)

    # Load metadata
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    artifact["metadata"] = metadata
    logger.info(
        f"Model loaded: {metadata.get('model_name', 'unknown')} "
        f"v{metadata.get('model_version', 'unknown')}"
    )

    return artifact


def predict(
    features: np.ndarray,
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Make a prediction with the loaded model.

    Args:
        features: Preprocessed feature array.
        model_path: Optional path to a specific model file.

    Returns:
        Dictionary with prediction, probability, and risk category.
    """
    artifact = load_model(model_path)
    model = artifact["model"]
    metadata = artifact.get("metadata", {})

    # Make prediction
    prediction = int(model.predict(features.reshape(1, -1))[0])
    probability = float(model.predict_proba(features.reshape(1, -1))[0][1])

    # Determine risk category
    risk_category = _get_risk_category(probability)

    result = {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_category": risk_category,
        "model_version": metadata.get("model_version", MODEL_VERSION),
        "model_name": metadata.get("model_name", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "disclaimer": (
            "This is a model-estimated risk score, not a medical diagnosis. "
            "Consult a qualified healthcare professional for medical evaluation."
        ),
    }

    logger.info(
        f"Prediction: {prediction}, Probability: {probability:.4f}, "
        f"Risk: {risk_category}"
    )

    return result


def predict_batch(
    features_batch: np.ndarray,
    model_path: Optional[Path] = None,
) -> list:
    """
    Make predictions for a batch of samples.

    Args:
        features_batch: Preprocessed feature array of shape (n_samples, n_features).
        model_path: Optional path to a specific model file.

    Returns:
        List of prediction dictionaries.
    """
    artifact = load_model(model_path)
    model = artifact["model"]
    metadata = artifact.get("metadata", {})

    predictions = model.predict(features_batch)
    probabilities = model.predict_proba(features_batch)[:, 1]

    results = []
    for i in range(len(predictions)):
        result = {
            "prediction": int(predictions[i]),
            "probability": round(float(probabilities[i]), 4),
            "risk_category": _get_risk_category(float(probabilities[i])),
            "model_version": metadata.get("model_version", MODEL_VERSION),
            "model_name": metadata.get("model_name", "unknown"),
        }
        results.append(result)

    return results


def _get_risk_category(probability: float) -> str:
    """
    Convert a probability to a risk category.

    NOTE: These thresholds are NOT clinically validated. They are
    model-estimated risk levels for research/educational purposes only.

    Args:
        probability: Predicted probability.

    Returns:
        Risk category string.
    """
    if probability < RISK_THRESHOLDS["low"]:
        return "lower predicted risk"
    elif probability < RISK_THRESHOLDS["moderate"]:
        return "moderate predicted risk"
    elif probability < RISK_THRESHOLDS["high"]:
        return "higher predicted risk"
    else:
        return "elevated predicted risk"
