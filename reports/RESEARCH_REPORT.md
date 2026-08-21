# Heart Disease Prediction System - Research Report

**Author:** Souvik Barui
**Date:** August 2026
**Version:** 2.0.0

## Executive Summary

This report documents the transformation of a basic ML notebook project into a comprehensive, production-ready heart disease risk prediction system. The enhanced system includes multiple ML models, probability calibration, explainability, a smart user interface, and comprehensive documentation.

---

## 1. Dataset Overview

### Primary Dataset: Cleveland Heart Disease

| Attribute | Value |
|-----------|-------|
| **Source** | UCI Machine Learning Repository |
| **Records** | 303 patients |
| **Features** | 14 (13 input + 1 target) |
| **Target** | Heart disease presence (0/1) |
| **Class Distribution** | 165 disease, 138 no disease |
| **Missing Values** | None (after handling '?' markers) |
| **Duplicates** | 1 row removed |

### Feature Categories

| Category | Features | Type |
|----------|----------|------|
| **Demographics** | age, sex | Numeric, Binary |
| **Symptoms** | cp (chest pain), exang (exercise angina) | Categorical |
| **Vitals** | trestbps (BP), thalach (max HR) | Numeric |
| **Lab Tests** | chol (cholesterol), fbs (fasting sugar) | Numeric, Binary |
| **ECG** | restecg, oldpeak, slope | Categorical, Numeric |
| **Imaging** | ca (vessels), thal (thalassemia) | Categorical |

---

## 2. Baseline Model Performance

### Original Notebook Results

| Model | Accuracy |
|-------|----------|
| Random Forest | 95.08% |
| Logistic Regression | 85.25% |
| Naive Bayes | 85.25% |
| XGBoost | 85.25% |
| Linear SVM | 81.97% |
| Decision Tree | 81.97% |
| KNN | 67.21% |

**Issues with Original:**
- Single train/test split (no cross-validation)
- No hyperparameter tuning
- No probability calibration
- No explainability
- No data leakage checks

---

## 3. Enhanced Pipeline Results

### Cross-Validated Performance (5-Fold Stratified)

| Model | CV ROC-AUC | Test Accuracy | Test ROC-AUC | Recall | Specificity | F1 | Brier Score |
|-------|------------|---------------|--------------|--------|-------------|-----|-------------|
| Logistic Regression | 0.9109 | 83.61% | 0.8885 | 87.88% | 78.57% | 0.86 | 0.12 |
| SVM | 0.9102 | 78.69% | 0.8864 | 87.88% | 67.86% | 0.82 | 0.14 |
| KNN | 0.8953 | 75.41% | 0.8555 | 84.85% | 64.29% | 0.80 | 0.16 |
| Random Forest | 0.8882 | 72.13% | 0.8463 | 81.82% | 60.71% | 0.77 | 0.18 |
| Gradient Boosting | 0.8737 | 73.77% | 0.8506 | 81.82% | 64.29% | 0.78 | 0.17 |
| Decision Tree | 0.8274 | 73.77% | 0.7495 | 84.85% | 60.71% | 0.79 | 0.25 |

### Key Improvements Over Baseline

1. **Cross-validation**: All metrics now use 5-fold stratified CV
2. **Hyperparameter tuning**: GridSearchCV/RandomizedSearchCV
3. **Probability calibration**: Isotonic regression
4. **Multiple metrics**: Accuracy, ROC-AUC, Recall, Specificity, F1, Brier Score
5. **Data leakage prevention**: Split before preprocessing

---

## 4. Feature Importance Analysis

### Top Features by Importance

| Rank | Feature | Importance | Clinical Relevance |
|------|---------|------------|-------------------|
| 1 | exang (exercise angina) | 0.142 | Strong predictor of coronary disease |
| 2 | thal (thalassemia) | 0.128 | Indicates blood flow abnormalities |
| 3 | ca (major vessels) | 0.124 | Direct measure of blockages |
| 4 | cp (chest pain type) | 0.118 | Primary symptom of heart disease |
| 5 | oldpeak (ST depression) | 0.109 | ECG indicator of ischemia |
| 6 | thalach (max heart rate) | 0.098 | Exercise capacity indicator |
| 7 | slope (ST segment) | 0.087 | ECG pattern during exercise |
| 8 | sex | 0.065 | Known risk factor |
| 9 | age | 0.052 | Primary risk factor |
| 10 | trestbps (resting BP) | 0.035 | Hypertension indicator |

---

## 5. Model Selection Rationale

### Final Model: Logistic Regression

**Selected for:**
- Highest CV ROC-AUC (0.9109)
- Strong recall (87.88%) - critical for screening
- Good interpretability
- Fast inference
- Well-calibrated probabilities

**Trade-offs:**
- Lower specificity than some models
- Assumes linear relationships
- May miss complex non-linear patterns

---

## 6. Calibration Analysis

### Brier Score Comparison

| Model | Brier Score | Interpretation |
|-------|-------------|----------------|
| Logistic Regression | 0.12 | Good calibration |
| SVM | 0.14 | Good calibration |
| KNN | 0.16 | Moderate calibration |
| Gradient Boosting | 0.17 | Moderate calibration |
| Random Forest | 0.18 | Moderate calibration |
| Decision Tree | 0.25 | Poor calibration |

**Note:** Lower Brier score = better calibrated probabilities

---

## 7. Limitations

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

---

## 8. Ethical Considerations

### Fairness
- Model trained on historically biased dataset
- May perform differently across demographics
- Should be validated before clinical deployment

### Privacy
- No personal information collected in predictions
- All processing is client-side
- No data stored or transmitted

### Safety
- Clear disclaimers about limitations
- Encourages professional consultation
- Does not provide medical advice

---

## 9. Future Work

### Recommended Improvements
1. **Larger datasets**: Integrate multiple UCI subsets
2. **External validation**: Test on independent cohorts
3. **Temporal validation**: Test on newer data
4. **Feature engineering**: Add interaction terms
5. **Ensemble methods**: Stack top models
6. **Deep learning**: Explore neural architectures
7. **Clinical integration**: EHR data pipelines

### Research Directions
- Multi-task learning (severity prediction)
- Survival analysis (time-to-event)
- Federated learning (privacy-preserving)
- Causal inference (treatment effects)

---

## 10. Conclusion

The enhanced heart disease prediction system represents a significant improvement over the original notebook implementation. Key achievements:

- **Rigorous methodology**: Cross-validation, proper train/test splits
- **Comprehensive evaluation**: Multiple metrics, calibration analysis
- **Explainability**: Feature importance, prediction explanations
- **User experience**: Smart questionnaire, risk visualization
- **Production readiness**: API, Docker, documentation

The system is designed for research and educational purposes, not clinical deployment. All predictions should be interpreted by qualified healthcare professionals.

---

*Report generated by Heart Disease Prediction System v2.0.0*
*Date: August 2026*
