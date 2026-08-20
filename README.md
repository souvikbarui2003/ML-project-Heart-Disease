# Heart Disease Prediction System

A machine learning-based system for predicting heart disease risk from clinical features. This project provides end-to-end ML pipeline including data validation, preprocessing, model training, evaluation, and REST API deployment.

## Overview

This system uses multiple machine learning algorithms to predict the presence of heart disease based on 13 clinical features. It includes:

- **Reproducible ML Pipeline**: Automated data validation, preprocessing, and model training
- **Multiple Models**: Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, and SVM
- **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV for optimal model performance
- **REST API**: FastAPI-based service for real-time predictions
- **Docker Support**: Containerized deployment ready
- **Comprehensive Testing**: Unit tests for all components

## Project Structure

```
ML-project-Heart-Disease/
├── api/                    # FastAPI application
│   ├── __init__.py
│   └── main.py            # API endpoints
├── config/                 # Configuration files
├── data/
│   ├── raw/               # Raw dataset
│   └── processed/         # Processed data
├── models/                 # Saved models
├── notebooks/              # Jupyter notebooks
├── reports/
│   ├── figures/           # Generated plots
│   └── metrics/           # Model metrics
├── scripts/               # Utility scripts
├── src/                   # Source code
│   ├── data/              # Data loading and validation
│   ├── features/          # Feature engineering
│   ├── models/            # Model training
│   ├── evaluation/        # Model evaluation
│   ├── inference/         # Prediction logic
│   └── utils/             # Utilities
├── tests/                 # Unit tests
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose
├── Makefile              # Development commands
├── requirements.txt      # Production dependencies
└── requirements-dev.txt  # Development dependencies
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ML-project-Heart-Disease

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Train Models

```bash
# Train all models with hyperparameter tuning
python -m src.train

# Or use Make
make train
```

### 3. Run Predictions

```bash
# Single prediction
python scripts/predict.py --age 63 --sex 1 --cp 3 --trestbps 145 \
    --chol 233 --fbs 1 --restecg 0 --thalach 150 --exang 0 \
    --oldpeak 2.3 --slope 0 --ca 0 --thal 1

# Batch prediction from CSV
python scripts/predict.py --input data/patients.csv
```

### 4. Start API Server

```bash
# Local development
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Or use Make
make api
```

### 5. Docker Deployment

```bash
# Build and run
docker-compose up -d

# Or use Make
make docker-run
```

## API Documentation

Once the API is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model/info` | Model information |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/features` | Feature descriptions |

### Example Request

```json
{
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
  "thal": 1
}
```

### Example Response

```json
{
  "prediction": 1,
  "probability": 0.8542,
  "risk_level": "high",
  "confidence": 0.8542,
  "model_name": "Random Forest",
  "model_version": "1.0.0"
}
```

## Model Performance

Models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall/Sensitivity**: Ability to detect disease
- **Specificity**: Ability to identify healthy patients
- **ROC-AUC**: Area under ROC curve
- **F1-Score**: Harmonic mean of precision and recall

### Key Metrics (Example)

| Model | Accuracy | ROC-AUC | Recall | Specificity |
|-------|----------|---------|--------|-------------|
| Random Forest | 0.87 | 0.92 | 0.89 | 0.85 |
| Gradient Boosting | 0.86 | 0.91 | 0.88 | 0.84 |
| Logistic Regression | 0.84 | 0.89 | 0.86 | 0.82 |

*Note: Actual results vary based on data and random seed*

## Dataset

The dataset contains 303 instances with 13 clinical features:

| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | Numerical |
| sex | Sex (1=male, 0=female) | Binary |
| cp | Chest pain type (0-3) | Categorical |
| trestbps | Resting blood pressure (mm Hg) | Numerical |
| chol | Serum cholesterol (mg/dl) | Numerical |
| fbs | Fasting blood sugar > 120 mg/dl | Binary |
| restecg | Resting ECG results (0-2) | Categorical |
| thalach | Maximum heart rate achieved | Numerical |
| exang | Exercise induced angina | Binary |
| oldpeak | ST depression induced by exercise | Numerical |
| slope | Slope of peak exercise ST segment | Categorical |
| ca | Number of major vessels (0-4) | Categorical |
| thal | Thalassemia type (0-3) | Categorical |

**Target**: Heart disease presence (1=disease, 0=no disease)

## Development

### Available Commands

```bash
make help          # Show all available commands
make install       # Install dependencies
make train         # Train models
make test          # Run tests
make lint          # Run linter
make format        # Format code
make docker-build  # Build Docker image
make clean         # Clean generated files
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data.py -v
```

### Code Quality

```bash
# Linting
flake8 src/ api/ tests/
mypy src/ api/

# Formatting
black src/ api/ tests/
isort src/ api/ tests/
```

## Configuration

Key configuration options in `src/config.py`:

```python
RANDOM_SEED = 42        # For reproducibility
TEST_SIZE = 0.2         # 80/20 train/test split
CV_FOLDS = 5            # Cross-validation folds
```

## Model Interpretability

The system provides feature importance analysis for tree-based models:

```python
from src.evaluation.plots import plot_feature_importance
from src.models.trainer import train_all_models

# After training
plot_feature_importance(model, feature_names, model_name)
```

## Limitations

- **Not for clinical use**: This is a demonstration system, not FDA-approved
- **Dataset size**: Limited to 303 samples
- **Feature set**: Only 13 clinical features available
- **Generalizability**: May not perform well on different populations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: [Heart Disease Dataset from UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- Built with scikit-learn, FastAPI, and Docker
- Inspired by real-world clinical decision support systems

## Disclaimer

This system is for educational and research purposes only. It should not be used for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.
