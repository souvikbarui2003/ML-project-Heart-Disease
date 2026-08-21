# Heart Disease Prediction System v2.0

**Author/Developer:** Souvik Barui

A comprehensive machine learning-based heart disease risk prediction system with explainability, calibration, and a smart user interface.

## Overview

This system transforms a basic Jupyter notebook into a production-ready ML research platform with:

- **Multiple ML Models**: 10+ algorithms with hyperparameter tuning
- **Probability Calibration**: Isotonic regression for reliable probabilities
- **Explainability**: Feature importance and prediction explanations
- **Smart User Interface**: Guided questionnaire with contextual help
- **Comprehensive Evaluation**: Cross-validation, multiple metrics, calibration analysis
- **Production Ready**: API, Docker, tests, documentation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/souvikbarui2003/ML-project-Heart-Disease.git
cd ML-project-Heart-Disease

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
npm install
```

### Training

```bash
# Train all models with calibration
PYTHONPATH=. python3 -m ml.train
```

### Running the Application

```bash
# Start the API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Start the frontend
npm run dev
```

## Model Performance

### Cross-Validated Results (5-Fold Stratified)

| Model | CV ROC-AUC | Test Accuracy | Test ROC-AUC | Recall | Specificity | F1 | Brier Score |
|-------|------------|---------------|--------------|--------|-------------|-----|-------------|
| **Logistic Regression** | **0.9109** | **83.61%** | **0.8885** | **87.88%** | 78.57% | 0.86 | 0.12 |
| SVM | 0.9102 | 78.69% | 0.8864 | 87.88% | 67.86% | 0.82 | 0.14 |
| KNN | 0.8953 | 75.41% | 0.8555 | 84.85% | 64.29% | 0.80 | 0.16 |
| Random Forest | 0.8882 | 72.13% | 0.8463 | 81.82% | 60.71% | 0.77 | 0.18 |
| Gradient Boosting | 0.8737 | 73.77% | 0.8506 | 81.82% | 64.29% | 0.78 | 0.17 |
| Decision Tree | 0.8274 | 73.77% | 0.7495 | 84.85% | 60.71% | 0.79 | 0.25 |

### Key Metrics Explained

- **ROC-AUC**: Overall discriminative ability (higher = better)
- **Recall**: Ability to detect disease (critical for screening)
- **Specificity**: Ability to identify healthy patients
- **Brier Score**: Probability calibration (lower = better calibrated)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with monitoring |
| `/model/info` | GET | Model information and metrics |
| `/predict` | POST | Make a risk prediction |
| `/explain` | POST | Get prediction explanation |
| `/scenario` | POST | Scenario analysis (what-if) |
| `/features` | GET | Feature descriptions |
| `/data-sources` | GET | Dataset information |
| `/metrics` | GET | Model performance metrics |

### Example Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55, "sex": 1, "cp": 2, "trestbps": 130,
    "chol": 240, "fbs": 0, "restecg": 1, "thalach": 160,
    "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2
  }'
```

## Smart User Interface

The frontend provides a guided questionnaire experience:

### Step 1: About You
- Age
- Sex

### Step 2: General Health
- Blood pressure
- Cholesterol
- Fasting blood sugar

### Step 3: Heart Tests
- Resting ECG results
- Maximum heart rate
- Exercise-induced angina

### Step 4: Exercise Response
- ST depression
- ST segment slope

### Step 5: Advanced Tests (Optional)
- Major vessels
- Thalassemia

**Features:**
- Contextual help for each field
- Input validation
- "I don't know" handling
- Mobile-friendly design

## Feature Importance

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | Exercise-Induced Angina | 14.2% | Chest pain during activity |
| 2 | Thalassemia | 12.8% | Blood disorder |
| 3 | Major Vessels | 12.4% | Blocked arteries |
| 4 | Chest Pain Type | 11.8% | Pain pattern |
| 5 | ST Depression | 10.9% | ECG change |
| 6 | Max Heart Rate | 9.8% | Exercise capacity |
| 7 | ST Slope | 8.7% | ECG pattern |
| 8 | Sex | 6.5% | Risk factor |
| 9 | Age | 5.2% | Primary factor |
| 10 | Resting BP | 3.5% | Hypertension |

## Project Structure

```
ML-project-Heart-Disease/
├── api/                    # FastAPI application
│   └── main.py
├── data/
│   ├── raw/               # Raw datasets
│   ├── sources.yaml       # Data sources registry
│   └── feature_schema.yaml # Clinical feature definitions
├── ml/                    # Python ML package
│   ├── config.py
│   ├── data/              # Data loading & validation
│   ├── features/          # Preprocessing
│   ├── models/            # Model training
│   ├── evaluation/        # Metrics & plots
│   ├── inference/         # Prediction & explainability
│   └── utils/             # Utilities
├── src/                   # React frontend
│   ├── pages/
│   │   ├── LandingPage.tsx
│   │   ├── PredictPage.tsx
│   │   └── DashboardPage.tsx
│   └── ...
├── tests/                 # Unit tests
├── reports/
│   └── RESEARCH_REPORT.md # Comprehensive research report
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Dataset

### Cleveland Heart Disease Dataset

| Attribute | Value |
|-----------|-------|
| Source | UCI Machine Learning Repository |
| Records | 303 patients |
| Features | 14 (13 input + 1 target) |
| Target | Heart disease presence (0/1) |
| License | CC BY 4.0 |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| age | Numeric | Age in years |
| sex | Binary | 0=Female, 1=Male |
| cp | Categorical | Chest pain type (0-3) |
| trestbps | Numeric | Resting blood pressure (mmHg) |
| chol | Numeric | Serum cholesterol (mg/dl) |
| fbs | Binary | Fasting blood sugar > 120 |
| restecg | Categorical | Resting ECG results |
| thalach | Numeric | Maximum heart rate |
| exang | Binary | Exercise-induced angina |
| oldpeak | Numeric | ST depression |
| slope | Categorical | ST segment slope |
| ca | Categorical | Major vessels (0-4) |
| thal | Categorical | Thalassemia type |

## Development

### Available Commands

```bash
# Python
PYTHONPATH=. python3 -m ml.train          # Train models
PYTHONPATH=. python3 -m ml.predict        # Run prediction

# Frontend
npm run dev                               # Start dev server
npm run build                             # Build for production

# Testing
pytest tests/ -v                          # Run tests

# Docker
docker-compose up -d                      # Start all services
```

### Code Quality

```bash
# Linting
flake8 ml/ api/ tests/

# Type checking
mypy ml/ api/

# Formatting
black ml/ api/ tests/
```

## Limitations

### Dataset Limitations
- Small sample size (303 patients)
- Single geographic source (Cleveland)
- Data from 1988 - may not reflect modern populations
- Limited demographic diversity

### Model Limitations
- Binary classification only (no severity levels)
- No temporal dynamics (single time point)
- No imaging data integration
- Limited to available features

### Clinical Limitations
- **Not a diagnostic device**
- Research/educational purposes only
- Should not replace professional medical advice
- Requires validation on external datasets

## Ethical Considerations

- Model trained on historically biased dataset
- May perform differently across demographics
- No personal information collected
- All processing is client-side
- Clear disclaimers about limitations

## Future Work

1. **Larger datasets**: Integrate multiple UCI subsets
2. **External validation**: Test on independent cohorts
3. **Feature engineering**: Add interaction terms
4. **Ensemble methods**: Stack top models
5. **Deep learning**: Explore neural architectures
6. **Clinical integration**: EHR data pipelines

## Documentation

### Academic Report

- [Project Report (Markdown)](docs/PROJECT_REPORT.md) - Comprehensive academic documentation
- [Model Comparison](reports/model-comparison.csv) - Detailed experimental results
- [Presentation Outline](presentation/HEART_DISEASE_ML_PRESENTATION.md) - PowerPoint structure

### Visualizations

- [Model Comparison Chart](reports/figures/model_comparison.png)
- [Feature Importance](reports/figures/feature_importance.png)
- [Confusion Matrix](reports/figures/confusion_matrix.png)
- [Class Distribution](reports/figures/class_distribution.png)
- [Brier Scores](reports/figures/brier_scores.png)

### Data Documentation

- [Data Sources Registry](data/sources.yaml)
- [Feature Schema](data/feature_schema.yaml)
- [Research Report](reports/RESEARCH_REPORT.md)

---

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- UCI Machine Learning Repository
- Cleveland Clinic Foundation
- scikit-learn contributors
- FastAPI and React communities

---

## Author

**Souvik Barui** — Research & Development

This project was designed, developed, and documented by Souvik Barui as a comprehensive machine learning research platform for heart disease risk prediction.

---

**Disclaimer:** This application provides a machine-learning-based risk estimate for research and educational purposes. It is **not** a medical diagnosis and should not replace evaluation by a qualified healthcare professional.
