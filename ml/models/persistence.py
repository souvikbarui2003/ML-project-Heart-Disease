"""
Model persistence: save and load complete ML pipelines.
"""
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_DIR, MODEL_VERSION
from src.utils.logging import logger


def save_model(
    model: Any,
    model_name: str,
    feature_names: list,
    metrics: Dict[str, Any],
    training_info: Dict[str, Any],
    dataset_hash: Optional[str] = None,
) -> Path:
    """
    Save the complete trained pipeline (model + metadata).

    Args:
        model: Trained sklearn model.
        model_name: Name of the model.
        feature_names: List of feature names.
        metrics: Dictionary of evaluation metrics.
        training_info: Training information (params, time, etc.).
        dataset_hash: Hash of the training dataset.

    Returns:
        Path to the saved model file.
    """
    import sys
    import sklearn

    model_path = MODEL_DIR / "best_model.joblib"
    metadata_path = MODEL_DIR / "model_metadata.json"

    # Create artifact
    artifact = {
        "model": model,
        "feature_names": feature_names,
    }

    # Save model
    joblib.dump(artifact, model_path)
    logger.info(f"Model saved to {model_path}")

    # Create and save metadata
    metadata = {
        "model_name": model_name,
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now().isoformat(),
        "dataset_hash": dataset_hash,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "metrics": metrics,
        "training_info": training_info,
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
        "disclaimer": (
            "This model is a machine-learning research/educational tool. "
            "It is not a medical diagnostic device."
        ),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Metadata saved to {metadata_path}")

    return model_path


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """
    Compute a hash of the dataset for versioning.

    Args:
        df: Dataset DataFrame.

    Returns:
        SHA256 hash of the dataset.
    """
    data_bytes = pd.util.hash_pandas_object(df).values.tobytes()
    return hashlib.sha256(data_bytes).hexdigest()[:16]


def get_model_info() -> Dict[str, Any]:
    """
    Get information about the currently saved model.

    Returns:
        Dictionary with model information.
    """
    metadata_path = MODEL_DIR / "model_metadata.json"

    if not metadata_path.exists():
        return {"status": "no_model_found"}

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata
