"""
FastAPI application for heart disease prediction.

Provides REST API for:
- Health check
- Single prediction
- Batch prediction
- Model information
"""
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from src.config import FEATURE_DESCRIPTIONS, MODEL_DIR, RISK_THRESHOLDS
from src.inference.predictor import HeartDiseasePredictor

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="ML-based heart disease risk prediction service",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[HeartDiseasePredictor] = None


class PredictionRequest(BaseModel):
    """Input schema for heart disease prediction."""
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1=male, 0=female)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type")
    trestbps: int = Field(..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=50, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results")
    thalach: int = Field(..., ge=40, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Major vessels colored by fluoroscopy")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia type")

    class Config:
        schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1,
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for heart disease prediction."""
    prediction: int = Field(..., description="Predicted class (1=disease, 0=no disease)")
    probability: float = Field(..., description="Probability of heart disease")
    risk_level: str = Field(..., description="Risk category (low/moderate/high)")
    confidence: float = Field(..., description="Prediction confidence")
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Model version")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    instances: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    count: int


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    model_version: str
    feature_names: List[str]
    metrics: Dict[str, float]


@app.on_event("startup")
async def load_model():
    """Load the trained model on startup."""
    global predictor
    try:
        predictor = HeartDiseasePredictor()
        predictor.load_model()
        print(f"Model loaded: {predictor.model_name}")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Predictions will not be available until model is loaded.")
        predictor = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
    }


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        feature_names=predictor.feature_names,
        metrics=predictor.metrics,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict heart disease risk for a single patient.

    Returns prediction, probability, risk level, and confidence.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert request to array
        features = np.array([[
            request.age,
            request.sex,
            request.cp,
            request.trestbps,
            request.chol,
            request.fbs,
            request.restecg,
            request.thalach,
            request.exang,
            request.oldpeak,
            request.slope,
            request.ca,
            request.thal,
        ]])

        # Get prediction
        result = predictor.predict(features)

        # Determine risk level
        prob = result["probability"][0]
        if prob < RISK_THRESHOLDS["low"]:
            risk_level = "low"
        elif prob < RISK_THRESHOLDS["moderate"]:
            risk_level = "moderate"
        elif prob < RISK_THRESHOLDS["high"]:
            risk_level = "elevated"
        else:
            risk_level = "high"

        # Calculate confidence
        confidence = max(prob, 1 - prob)

        return PredictionResponse(
            prediction=int(result["prediction"][0]),
            probability=float(prob),
            risk_level=risk_level,
            confidence=float(confidence),
            model_name=predictor.model_name,
            model_version=predictor.model_version,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict heart disease risk for multiple patients.

    Accepts a list of patient records and returns predictions for each.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert requests to array
        features = np.array([[
            inst.age,
            inst.sex,
            inst.cp,
            inst.trestbps,
            inst.chol,
            inst.fbs,
            inst.restecg,
            inst.thalach,
            inst.exang,
            inst.oldpeak,
            inst.slope,
            inst.ca,
            inst.thal,
        ] for inst in request.instances])

        # Get predictions
        result = predictor.predict(features)

        # Build responses
        predictions = []
        for i in range(len(features)):
            prob = result["probability"][i]

            if prob < RISK_THRESHOLDS["low"]:
                risk_level = "low"
            elif prob < RISK_THRESHOLDS["moderate"]:
                risk_level = "moderate"
            elif prob < RISK_THRESHOLDS["high"]:
                risk_level = "elevated"
            else:
                risk_level = "high"

            predictions.append(PredictionResponse(
                prediction=int(result["prediction"][i]),
                probability=float(prob),
                risk_level=risk_level,
                confidence=float(max(prob, 1 - prob)),
                model_name=predictor.model_name,
                model_version=predictor.model_version,
            ))

        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/features")
async def get_features():
    """Get feature descriptions and valid ranges."""
    return {
        "features": FEATURE_DESCRIPTIONS,
        "numerical_columns": [
            "age", "trestbps", "chol", "thalach", "oldpeak"
        ],
        "categorical_columns": [
            "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
