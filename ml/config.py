"""
Configuration constants for the Heart Disease Prediction project.
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# Dataset configuration
DATASET_FILENAME = "heart.csv"
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Column definitions
NUMERICAL_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COLUMN = "target"

# Expected column ranges
COLUMN_RANGES = {
    "age": (29, 77),
    "sex": (0, 1),
    "cp": (0, 3),
    "trestbps": (94, 200),
    "chol": (126, 564),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (71, 202),
    "exang": (0, 1),
    "oldpeak": (0.0, 6.2),
    "slope": (0, 2),
    "ca": (0, 4),
    "thal": (0, 3),
    "target": (0, 1),
}

# Feature descriptions
FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)",
    "trestbps": "Resting blood pressure (mm Hg on admission)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting electrocardiographic results (0, 1, or 2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of the peak exercise ST segment (0, 1, or 2)",
    "ca": "Number of major vessels colored by fluoroscopy (0-3)",
    "thal": "Thalassemia (0 = normal, 1 = fixed defect, 2 = reversable defect, 3 = unknown)",
    "target": "Heart disease presence (1 = disease, 0 = no disease)",
}

# Risk thresholds (not clinically validated)
RISK_THRESHOLDS = {
    "low": 0.3,
    "moderate": 0.5,
    "high": 0.7,
}

# Model version
MODEL_VERSION = "1.0.0"

# Ensure directories exist
for dir_path in [PROCESSED_DATA_DIR, MODEL_DIR, FIGURES_DIR, METRICS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
