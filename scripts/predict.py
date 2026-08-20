#!/usr/bin/env python
"""
Prediction script for heart disease prediction.

Usage:
    python scripts/predict.py --input data/patient.csv
    python scripts/predict.py --age 63 --sex 1 --cp 3 --trestbps 145
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.inference.predictor import HeartDiseasePredictor
from src.config import RISK_THRESHOLDS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict heart disease risk"
    )
    parser.add_argument("--input", type=str, help="CSV file with patient data")
    parser.add_argument("--age", type=int, help="Age in years")
    parser.add_argument("--sex", type=int, choices=[0, 1], help="Sex (1=male, 0=female)")
    parser.add_argument("--cp", type=int, help="Chest pain type (0-3)")
    parser.add_argument("--trestbps", type=int, help="Resting blood pressure")
    parser.add_argument("--chol", type=int, help="Serum cholesterol")
    parser.add_argument("--fbs", type=int, help="Fasting blood sugar > 120")
    parser.add_argument("--restecg", type=int, help="Resting ECG results")
    parser.add_argument("--thalach", type=int, help="Max heart rate")
    parser.add_argument("--exang", type=int, help="Exercise induced angina")
    parser.add_argument("--oldpeak", type=float, help="ST depression")
    parser.add_argument("--slope", type=int, help="Slope of peak exercise ST")
    parser.add_argument("--ca", type=int, help="Major vessels colored")
    parser.add_argument("--thal", type=int, help="Thalassemia type")
    return parser.parse_args()


def main():
    """Main prediction function."""
    args = parse_args()

    # Load model
    predictor = HeartDiseasePredictor()
    predictor.load_model()

    if args.input:
        # Batch prediction from CSV
        import pandas as pd
        df = pd.read_csv(args.input)
        features = df.drop(columns=["target"], errors="ignore").values
        results = predictor.predict(features)

        for i, (pred, prob) in enumerate(zip(results["prediction"], results["probability"])):
            risk = "HIGH" if prob >= RISK_THRESHOLDS["high"] else \
                   "MODERATE" if prob >= RISK_THRESHOLDS["moderate"] else "LOW"
            print(f"Patient {i+1}: Prediction={pred}, Probability={prob:.4f}, Risk={risk}")
    else:
        # Single prediction from arguments
        if args.age is None:
            print("Error: --input or --age required")
            sys.exit(1)

        features = np.array([[
            args.age, args.sex, args.cp, args.trestbps, args.chol,
            args.fbs, args.restecg, args.thalach, args.exang,
            args.oldpeak, args.slope, args.ca, args.thal
        ]])

        results = predictor.predict(features)
        pred = results["prediction"][0]
        prob = results["probability"][0]

        risk = "HIGH" if prob >= RISK_THRESHOLDS["high"] else \
               "MODERATE" if prob >= RISK_THRESHOLDS["moderate"] else "LOW"

        print(f"\nPrediction: {'Heart Disease' if pred == 1 else 'No Heart Disease'}")
        print(f"Probability: {prob:.4f}")
        print(f"Risk Level: {risk}")


if __name__ == "__main__":
    main()
