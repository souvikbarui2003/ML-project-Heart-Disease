"""
Enhanced prediction module with calibrated probabilities and explanations.
"""
import numpy as np
from typing import Any, Dict, List, Optional
from ml.utils.logging import logger


class HeartDiseasePredictor:
    """Enhanced predictor with calibration and explainability."""

    def __init__(self):
        self.model = None
        self.calibrated_model = None
        self.model_name = None
        self.model_version = "2.0.0"
        self.feature_names = None
        self.metrics = {}
        self.feature_importance = {}

    def load_model(self, model_path: str = None):
        """Load trained model from disk."""
        import joblib
        from pathlib import Path
        from ml.config import MODEL_DIR

        if model_path is None:
            # Find most recent model
            model_files = list(MODEL_DIR.glob("*.joblib"))
            if not model_files:
                raise FileNotFoundError("No model found in models directory")
            model_path = max(model_files, key=lambda f: f.stat().st_mtime)

        logger.info(f"Loading model from {model_path}")
        model_data = joblib.load(model_path)

        self.model = model_data["model"]
        self.model_name = model_data.get("model_name", "Unknown")
        self.feature_names = model_data.get("feature_names", [])
        self.metrics = model_data.get("metrics", {})
        self.feature_importance = model_data.get("feature_importance", {})

        logger.info(f"Loaded model: {self.model_name}")

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make prediction with enhanced output.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Use calibrated model if available, otherwise use raw model
        model = self.calibrated_model if self.calibrated_model else self.model

        # Get predictions
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)

        return {
            "prediction": prediction,
            "probability": probabilities[:, 1],
            "all_probabilities": probabilities,
        }

    def explain_prediction(
        self,
        features: np.ndarray,
        feature_values: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Explain a single prediction.
        """
        from ml.inference.explainer import explain_prediction, get_feature_importance

        explanation = explain_prediction(
            self.model,
            features[0] if features.ndim > 1 else features,
            self.feature_names,
            feature_values,
        )

        # Add model info
        explanation["model_name"] = self.model_name
        explanation["model_version"] = self.model_version
        explanation["feature_importance"] = self.feature_importance

        return explanation

    def get_prediction_response(
        self,
        features: np.ndarray,
        feature_values: Dict[str, float],
        include_explanation: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive prediction response.
        """
        # Make prediction
        result = self.predict(features)

        prob = float(result["probability"][0])
        prediction = int(result["prediction"][0])

        # Determine risk category
        if prob < 0.3:
            risk_category = "lower predicted risk"
            risk_color = "green"
        elif prob < 0.5:
            risk_category = "moderate predicted risk"
            risk_color = "amber"
        elif prob < 0.7:
            risk_category = "higher predicted risk"
            risk_color = "orange"
        else:
            risk_category = "elevated predicted risk"
            risk_color = "red"

        response = {
            "prediction": prediction,
            "probability": prob,
            "risk_category": risk_category,
            "risk_color": risk_color,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "confidence": float(max(prob, 1 - prob)),
            "disclaimer": "This is a model-estimated risk score, not a medical diagnosis.",
        }

        # Add explanation if requested
        if include_explanation:
            explanation = self.explain_prediction(features, feature_values)
            response["explanation"] = explanation

        return response
