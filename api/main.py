"""
Comprehensive FastAPI application for heart disease prediction.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.config import RISK_THRESHOLDS, FEATURE_DESCRIPTIONS
from ml.inference.predictor import HeartDiseasePredictor

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="ML-based heart disease risk prediction with explainability",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
predictor: Optional[HeartDiseasePredictor] = None
prediction_count = 0
error_count = 0


class PredictionRequest(BaseModel):
    """Input schema for heart disease prediction."""
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1=male, 0=female)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type")
    trestbps: int = Field(..., ge=50, le=300, description="Resting blood pressure (mmHg)")
    chol: int = Field(..., ge=50, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results")
    thalach: int = Field(..., ge=40, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Major vessels colored by fluoroscopy")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia type")


class PredictionResponse(BaseModel):
    """Output schema for heart disease prediction."""
    prediction: int
    probability: float
    risk_category: str
    risk_color: str
    confidence: float
    model_name: str
    model_version: str
    disclaimer: str


class ExplainRequest(BaseModel):
    """Request for prediction explanation."""
    features: PredictionRequest


class ExplainResponse(BaseModel):
    """Output schema for prediction explanation."""
    prediction: int
    probability: float
    contributing_factors: List[Dict[str, Any]]
    protective_factors: List[Dict[str, Any]]
    feature_importance: Dict[str, float]


class ScenarioRequest(BaseModel):
    """Request for scenario analysis."""
    base_features: PredictionRequest
    modified_features: Dict[str, Any]


class ScenarioResponse(BaseModel):
    """Output schema for scenario analysis."""
    base_prediction: Dict[str, Any]
    scenario_prediction: Dict[str, Any]
    difference: Dict[str, Any]


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    model_version: str
    feature_names: List[str]
    metrics: Dict[str, float]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    prediction_count: int
    error_count: int
    uptime: str


# Track startup time
startup_time = datetime.now()


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
        predictor = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = str(datetime.now() - startup_time)
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        prediction_count=prediction_count,
        error_count=error_count,
        uptime=uptime,
    )


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
    """Make a heart disease risk prediction."""
    global prediction_count, error_count

    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        import numpy as np

        features = np.array([[
            request.age, request.sex, request.cp, request.trestbps,
            request.chol, request.fbs, request.restecg, request.thalach,
            request.exang, request.oldpeak, request.slope, request.ca, request.thal,
        ]])

        result = predictor.get_prediction_response(
            features,
            request.dict(),
            include_explanation=False,
        )

        prediction_count += 1

        return PredictionResponse(**result)

    except Exception as e:
        error_count += 1
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    """Get explanation for a prediction."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        import numpy as np

        features = np.array([[
            request.features.age, request.features.sex, request.features.cp,
            request.features.trestbps, request.features.chol, request.features.fbs,
            request.features.restecg, request.features.thalach, request.features.exang,
            request.features.oldpeak, request.features.slope, request.features.ca,
            request.features.thal,
        ]])

        explanation = predictor.explain_prediction(
            features,
            request.features.dict(),
        )

        return ExplainResponse(
            prediction=explanation["prediction"],
            probability=explanation["probability"],
            contributing_factors=explanation["contributing_factors"][:5],
            protective_factors=explanation["protective_factors"][:5],
            feature_importance=explanation.get("feature_importance", {}),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


@app.post("/scenario", response_model=ScenarioResponse)
async def scenario_analysis(request: ScenarioRequest):
    """Perform scenario analysis - compare base vs modified features."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        import numpy as np

        # Base prediction
        base_features = np.array([[
            request.base_features.age, request.base_features.sex, request.base_features.cp,
            request.base_features.trestbps, request.base_features.chol, request.base_features.fbs,
            request.base_features.restecg, request.base_features.thalach, request.base_features.exang,
            request.base_features.oldpeak, request.base_features.slope, request.base_features.ca,
            request.base_features.thal,
        ]])

        base_result = predictor.get_prediction_response(
            base_features,
            request.base_features.dict(),
            include_explanation=False,
        )

        # Modified prediction
        modified_data = request.base_features.dict()
        modified_data.update(request.modified_features)

        modified_features = np.array([[
            modified_data["age"], modified_data["sex"], modified_data["cp"],
            modified_data["trestbps"], modified_data["chol"], modified_data["fbs"],
            modified_data["restecg"], modified_data["thalach"], modified_data["exang"],
            modified_data["oldpeak"], modified_data["slope"], modified_data["ca"],
            modified_data["thal"],
        ]])

        scenario_result = predictor.get_prediction_response(
            modified_features,
            modified_data,
            include_explanation=False,
        )

        # Calculate difference
        prob_diff = scenario_result["probability"] - base_result["probability"]

        return ScenarioResponse(
            base_prediction=base_result,
            scenario_prediction=scenario_result,
            difference={
                "probability_change": float(prob_diff),
                "direction": "increased" if prob_diff > 0 else "decreased",
                "magnitude": float(abs(prob_diff)),
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scenario analysis error: {str(e)}")


@app.get("/features")
async def get_features():
    """Get feature descriptions and valid ranges."""
    return {
        "features": FEATURE_DESCRIPTIONS,
        "numerical_columns": ["age", "trestbps", "chol", "thalach", "oldpeak"],
        "categorical_columns": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
    }


@app.get("/data-sources")
async def get_data_sources():
    """Get information about data sources."""
    return {
        "sources": [
            {
                "name": "Cleveland Heart Disease Dataset",
                "records": 303,
                "features": 14,
                "license": "CC BY 4.0",
                "usage": "Primary training dataset",
            }
        ],
        "total_records": 303,
        "feature_count": 14,
    }


@app.get("/metrics")
async def get_metrics():
    """Get model performance metrics."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": predictor.model_name,
        "metrics": predictor.metrics,
        "feature_importance": predictor.feature_importance,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
