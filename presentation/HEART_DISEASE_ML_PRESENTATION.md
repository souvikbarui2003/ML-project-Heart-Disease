# Heart Disease Prediction System
## Machine Learning-Based Heart Disease Risk Prediction

### Presentation for Academic Evaluation

---

## Slide 1: Title Slide

# Heart Disease Prediction System

## An Explainable Machine Learning Framework for Heart Disease Risk Prediction

**Student:** Souvik Barui
**Department:** [Department]
**University:** [University]
**Academic Year:** 2025-2026
**Supervisor:** [Supervisor Name]

---

## Slide 2: The Problem

# Cardiovascular Disease: A Global Challenge

- **17.9 million** deaths annually worldwide (WHO, 2021)
- Leading cause of death globally
- **$274 billion** annual healthcare cost in US alone
- Early detection saves lives

> "Half of all cardiovascular deaths could be prevented with early intervention"

---

## Slide 3: Why Machine Learning?

# Why ML for Heart Disease Prediction?

| Challenge | ML Solution |
|-----------|-------------|
| Complex feature interactions | Non-linear models |
| Large feature spaces | Automated feature selection |
| Pattern recognition | Ensemble methods |
| Risk calibration | Probability calibration |
| Interpretability | Explainable AI |

---

## Slide 4: Project Objectives

# What We Built

1. ✅ Reproducible ML pipeline
2. ✅ Data validation & preprocessing
3. ✅ Multiple ML algorithms (10 models)
4. ✅ Hyperparameter optimization
5. ✅ Probability calibration
6. ✅ Feature importance analysis
7. ✅ User-friendly interface
8. ✅ REST API
9. ✅ Comprehensive documentation

---

## Slide 5: Dataset

# Cleveland Heart Disease Dataset

| Attribute | Value |
|-----------|-------|
| Source | UCI Machine Learning Repository |
| Records | **303 patients** |
| Features | **14** (13 input + 1 target) |
| Target | Heart disease presence (0/1) |
| Class Distribution | 165 disease, 138 no disease |
| License | CC BY 4.0 |

---

## Slide 6: Features

# Clinical Features

| Category | Features |
|----------|----------|
| **Demographics** | Age, Sex |
| **Symptoms** | Chest pain type, Exercise angina |
| **Vitals** | Resting BP, Max heart rate |
| **Lab Tests** | Cholesterol, Fasting blood sugar |
| **ECG** | Resting ECG, ST depression, ST slope |
| **Imaging** | Major vessels, Thalassemia |

---

## Slide 7: Data Pipeline

# Preprocessing Pipeline

```
Raw Data → Validation → Deduplication → Split (80/20)
    ↓
Train Set → Fit Preprocessor → Transform
    ↓
Test Set → Transform (no fit!)
    ↓
Ready for ML Training
```

**Key:** No data leakage - split BEFORE preprocessing!

---

## Slide 8: ML Models

# Algorithms Evaluated

| Category | Models |
|----------|--------|
| **Linear** | Logistic Regression |
| **Distance** | K-Nearest Neighbors |
| **Tree** | Decision Tree, Random Forest, Extra Trees |
| **Boosting** | Gradient Boosting, AdaBoost |
| **Kernel** | Support Vector Machine |
| **Probabilistic** | Naive Bayes |
| **Neural** | Multi-Layer Perceptron |

**Total:** 10 algorithms with hyperparameter tuning

---

## Slide 9: Training Methodology

# Rigorous Evaluation

- **5-fold Stratified Cross-Validation**
- **Hyperparameter Tuning:** GridSearchCV / RandomizedSearchCV
- **Scoring:** ROC-AUC (primary)
- **Random Seed:** 42 (reproducible)

```
Training Data (241 samples)
    ↓
5-Fold CV → Best Hyperparameters
    ↓
Retrain on Full Training Set
    ↓
Evaluate on Held-Out Test Set (61 samples)
```

---

## Slide 10: Results

# Model Performance

| Model | CV ROC-AUC | Test Accuracy | Recall |
|-------|------------|---------------|--------|
| **Logistic Regression** | **0.9109** | **83.61%** | **87.88%** |
| SVM | 0.9102 | 78.69% | 87.88% |
| KNN | 0.8953 | 75.41% | 84.85% |
| Random Forest | 0.8882 | 72.13% | 81.82% |
| Gradient Boosting | 0.8737 | 73.77% | 81.82% |
| Decision Tree | 0.8274 | 73.77% | 84.85% |

**Best Model:** Logistic Regression (ROC-AUC: 0.8885)

---

## Slide 11: Confusion Matrix

# Logistic Regression - Confusion Matrix

|  | Predicted No Disease | Predicted Disease |
|--|---------------------|-------------------|
| **Actual No Disease** | 22 (TN) | 6 (FP) |
| **Actual Disease** | 4 (FN) | 29 (TP) |

- **Recall:** 87.88% (catches most disease cases)
- **Specificity:** 78.57% (identifies most healthy patients)
- **4 False Negatives** - missed cases (critical for screening)

---

## Slide 12: Feature Importance

# What Matters Most?

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Exercise-Induced Angina | 14.2% |
| 2 | Thalassemia | 12.8% |
| 3 | Major Vessels | 12.4% |
| 4 | Chest Pain Type | 11.8% |
| 5 | ST Depression | 10.9% |

**Top 5 features account for 62.1% of prediction**

---

## Slide 13: Probability Calibration

# Why Calibration Matters

**Brier Score Comparison:**

| Model | Brier Score |
|-------|-------------|
| Logistic Regression | **0.12** |
| SVM | 0.14 |
| KNN | 0.16 |
| Gradient Boosting | 0.17 |
| Random Forest | 0.18 |
| Decision Tree | 0.25 |

**Calibration Method:** Isotonic Regression

**Result:** 68% predicted probability ≈ 68% actual frequency

---

## Slide 14: Explainability

# Model Explanations

**Global:** Feature importance ranks

**Local:** Individual prediction explanations

Example Prediction:
- **Risk:** Higher predicted risk (68%)
- **Contributing:** Age > 60, Chest pain type, Exercise angina
- **Protective:** Young age, Good heart rate

> "The model contributed to this prediction" (not "caused")

---

## Slide 15: User Interface

# Smart Questionnaire

**5-Step Guided Process:**

1. **About You** - Age, Sex
2. **General Health** - BP, Cholesterol, Blood sugar
3. **Heart Tests** - ECG, Max HR, Angina
4. **Exercise Response** - ST depression, ST slope
5. **Advanced Tests** - Vessels, Thalassemia (optional)

**Features:**
- Contextual help for each field
- Input validation
- "I don't know" option

---

## Slide 16: Prediction Experience

# Results Presentation

**Output Includes:**
- ✅ Risk probability (e.g., 68%)
- ✅ Risk category (Lower/Moderate/Higher)
- ✅ Contributing factors with severity
- ✅ Protective factors
- ✅ Model explanation
- ✅ Medical disclaimer

**Design:** Clear, non-alarming, professional

---

## Slide 17: API Endpoints

# RESTful API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System status |
| `/predict` | POST | Make prediction |
| `/explain` | POST | Get explanation |
| `/scenario` | POST | What-if analysis |
| `/features` | GET | Feature descriptions |
| `/metrics` | GET | Model performance |

**Technology:** FastAPI + Pydantic validation

---

## Slide 18: System Architecture

# Technical Stack

```
┌─────────────────────────────────┐
│     React + TypeScript          │
│     (Frontend UI)               │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│     FastAPI                     │
│     (REST API)                  │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│     scikit-learn                │
│     (ML Pipeline)               │
└───────────────┬─────────────────┘
                │
┌───────────────▼─────────────────┐
│     Model Registry              │
│     (Joblib)                    │
└─────────────────────────────────┘
```

---

## Slide 19: Technology Stack

# Technologies Used

| Layer | Technology |
|-------|------------|
| **Frontend** | React 18, TypeScript, Tailwind CSS |
| **Build** | Vite 5 |
| **API** | FastAPI, Pydantic |
| **ML** | scikit-learn 1.3, pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Deployment** | Docker, Uvicorn |

---

## Slide 20: Results Summary

# Key Findings

✅ **Best Model:** Logistic Regression
- CV ROC-AUC: 0.9109
- Test ROC-AUC: 0.8885
- Recall: 87.88%

✅ **Calibration:** Brier score 0.12

✅ **Interpretability:** Feature importance + explanations

✅ **Accessibility:** User-friendly questionnaire

⚠️ **Limitations:** Small dataset (303), single source

---

## Slide 21: Limitations

# Honest Assessment

| Limitation | Impact |
|------------|--------|
| Small dataset (303) | Limited statistical power |
| Single source (Cleveland) | May not generalize |
| Historical data (1988) | Modern populations differ |
| No clinical validation | Research only |
| Binary classification | No severity levels |

**This is a RESEARCH tool, not a diagnostic device**

---

## Slide 22: Ethics & Privacy

# Responsible AI

- ✅ Clear medical disclaimers
- ✅ No PII collected
- ✅ No data stored
- ✅ Encourages professional consultation
- ✅ Transparent about limitations
- ✅ No treatment recommendations

**Priority:** Patient safety over model performance

---

## Slide 23: Future Work

# Roadmap

**Data:**
- Integrate multiple datasets
- Add modern EHR data
- Include imaging features

**Models:**
- Deep learning architectures
- Survival analysis
- Federated learning

**Clinical:**
- Prospective validation
- EHR integration
- Multi-center studies

---

## Slide 24: Conclusion

# What We Achieved

✅ Built explainable ML framework for heart disease prediction

✅ Evaluated 10 algorithms with rigorous methodology

✅ Achieved 88.85% ROC-AUC with calibrated probabilities

✅ Created accessible user interface

✅ Provided prediction explanations

✅ Documented all limitations

**Next step:** External validation on larger datasets

---

## Slide 25: Thank You

# Thank You

## Questions?

**Project Repository:** https://github.com/souvikbarui2003/ML-project-Heart-Disease

**Live Demo:** [URL if deployed]

---

**Disclaimer:** This system is for research and educational purposes only. It is not a medical device and should not be used for clinical diagnosis.

---

## Presentation Notes

### Design Guidelines

- **Color Scheme:** Blue (primary), Green (success), Red (warning)
- **Fonts:** Clean, professional (e.g., Inter, Roboto)
- **Layout:** Minimal text, maximum visual impact
- **Charts:** Generate from actual data where possible
- **Icons:** Use consistent icon set

### Diagrams to Include

1. System architecture diagram
2. Data flow diagram
3. Model comparison bar chart
4. ROC curves
5. Confusion matrices
6. Feature importance bar chart
7. Calibration curve
8. User interface screenshots

### Animation Suggestions

- Sequential reveal for lists
- Chart build-up for comparisons
- Pipeline flow animation
- Fade transitions between sections

### Image Sources (Royalty-Free)

- Healthcare/medical icons from Lucide or Heroicons
- Charts generated from matplotlib/seaborn
- Architecture diagrams created with Mermaid or draw.io
