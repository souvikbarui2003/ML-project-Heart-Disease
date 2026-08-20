"""
Model explainability using SHAP values.
"""
import numpy as np
from typing import Any, Dict, List, Tuple
from ml.utils.logging import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")


def compute_shap_values(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    n_samples: int = 100,
) -> Dict[str, Any]:
    """
    Compute SHAP values for model interpretability.
    """
    if not SHAP_AVAILABLE:
        return {"error": "SHAP not installed"}

    try:
        # Use appropriate explainer based on model type
        model_name = type(model).__name__

        if model_name in ["RandomForestClassifier", "GradientBoostingClassifier",
                          "ExtraTreesClassifier", "DecisionTreeClassifier"]:
            explainer = shap.TreeExplainer(model)
        else:
            # Use KernelExplainer for other models
            background = shap.sample(X, min(n_samples, len(X)))
            explainer = shap.KernelExplainer(model.predict_proba, background)

        shap_values = explainer.shap_values(X[:n_samples])

        return {
            "shap_values": shap_values,
            "feature_names": feature_names,
            "expected_value": explainer.expected_value,
        }

    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        return {"error": str(e)}


def get_feature_importance(
    model: Any,
    feature_names: List[str],
    method: str = "auto",
) -> Dict[str, float]:
    """
    Get feature importance from model.
    """
    importance = {}

    # Try different methods based on model type
    model_name = type(model).__name__

    if hasattr(model, "feature_importances_"):
        # Tree-based models
        importances = model.feature_importances_
        for name, imp in zip(feature_names, importances):
            importance[name] = float(imp)

    elif hasattr(model, "coef_"):
        # Linear models
        coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        for name, coef in zip(feature_names, coefs):
            importance[name] = float(abs(coef))

    else:
        # Fallback: use permutation importance
        try:
            from sklearn.inspection import permutation_importance
            result = permutation_importance(
                model, np.zeros((10, len(feature_names))), np.zeros(10),
                n_repeats=5, random_state=42
            )
            for name, imp in zip(feature_names, result.importances_mean):
                importance[name] = float(abs(imp))
        except Exception:
            logger.warning(f"Could not compute feature importance for {model_name}")

    return importance


def explain_prediction(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    feature_values: Dict[str, float],
) -> Dict[str, Any]:
    """
    Explain a single prediction.
    """
    explanation = {
        "prediction": None,
        "probability": None,
        "contributing_factors": [],
        "protective_factors": [],
    }

    try:
        # Get prediction
        prob = model.predict_proba(X.reshape(1, -1))[0]
        prediction = model.predict(X.reshape(1, -1))[0]

        explanation["prediction"] = int(prediction)
        explanation["probability"] = float(prob[1])

        # Get feature importance
        importance = get_feature_importance(model, feature_names)

        # Categorize factors
        for name, value in zip(feature_names, X):
            if name in importance:
                imp = importance[name]
                # Normalize value contribution
                contribution = imp * value

                factor = {
                    "feature": name,
                    "value": float(value),
                    "importance": float(imp),
                    "contribution": float(contribution),
                }

                if contribution > 0:
                    explanation["contributing_factors"].append(factor)
                else:
                    explanation["protective_factors"].append(factor)

        # Sort by absolute contribution
        explanation["contributing_factors"].sort(
            key=lambda x: abs(x["contribution"]), reverse=True
        )
        explanation["protective_factors"].sort(
            key=lambda x: abs(x["contribution"]), reverse=True
        )

    except Exception as e:
        logger.error(f"Prediction explanation failed: {e}")
        explanation["error"] = str(e)

    return explanation
