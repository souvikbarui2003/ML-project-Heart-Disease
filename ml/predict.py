"""
CLI module for making predictions with a trained model.

Usage:
    python -m src.predict --age 55 --sex 1 --cp 2 --trestbps 130 --chol 240 --fbs 0 --restecg 1 --thalach 160 --exang 0 --oldpeak 1.0 --slope 2 --ca 0 --thal 2
"""
import argparse
import json
import sys

import numpy as np

from ml.features.preprocessing import build_preprocessing_pipeline
from ml.inference.explainer import get_feature_importance, get_shap_explanation
from ml.inference.predictor import load_model, predict
from ml.utils.logging import logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Heart Disease Prediction - CLI Prediction"
    )
    parser.add_argument("--age", type=float, required=True, help="Age in years")
    parser.add_argument("--sex", type=int, required=True, help="Sex (1=male, 0=female)")
    parser.add_argument("--cp", type=int, required=True, help="Chest pain type (0-3)")
    parser.add_argument("--trestbps", type=float, required=True, help="Resting blood pressure (mm Hg)")
    parser.add_argument("--chol", type=float, required=True, help="Serum cholesterol (mg/dl)")
    parser.add_argument("--fbs", type=int, required=True, help="Fasting blood sugar > 120 mg/dl (1/0)")
    parser.add_argument("--restecg", type=int, required=True, help="Resting ECG results (0-2)")
    parser.add_argument("--thalach", type=float, required=True, help="Maximum heart rate achieved")
    parser.add_argument("--exang", type=int, required=True, help="Exercise induced angina (1/0)")
    parser.add_argument("--oldpeak", type=float, required=True, help="ST depression induced by exercise")
    parser.add_argument("--slope", type=int, required=True, help="Slope of peak exercise ST segment (0-2)")
    parser.add_argument("--ca", type=int, required=True, help="Number of major vessels (0-4)")
    parser.add_argument("--thal", type=int, required=True, help="Thalassemia (0-3)")

    return parser.parse_args()


def main() -> None:
    """Main prediction pipeline."""
    logger.info("Heart Disease Prediction - CLI")
    args = parse_args()

    # Create feature array
    features = np.array([
        args.age, args.sex, args.cp, args.trestbps, args.chol,
        args.fbs, args.restecg, args.thalach, args.exang,
        args.oldpeak, args.slope, args.ca, args.thal,
    ])

    # Get prediction
    result = predict(features)

    # Display result
    print("\n" + "=" * 50)
    print("HEART DISEASE RISK PREDICTION")
    print("=" * 50)
    print(f"Prediction:    {'Higher Risk' if result['prediction'] == 1 else 'Lower Risk'}")
    print(f"Probability:   {result['probability']:.2%}")
    print(f"Risk Category: {result['risk_category']}")
    print(f"Model:         {result['model_name']} v{result['model_version']}")
    print(f"Time:          {result['timestamp']}")
    print(f"\n⚠️  {result['disclaimer']}")
    print("=" * 50)

    # JSON output
    print(f"\nJSON Response:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
