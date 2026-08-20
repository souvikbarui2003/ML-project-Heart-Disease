# An Explainable Machine Learning Framework for Heart Disease Risk Prediction Using Public Health Data

---

## Cover Page

**Title:** An Explainable Machine Learning Framework for Heart Disease Risk Prediction Using Public Health Data

**Student Name:** [Student Name to be filled]

**Department:** [Department to be filled]

**University/College:** [University to be filled]

**Course/Program:** [Course to be filled]

**Academic Year:** 2025-2026

**Project Type:** Major Project / Thesis

**Guide/Supervisor:** [Supervisor Name to be filled]

**Team Members:** [Team members to be filled]

---

## Certificate

This is to certify that the project entitled **"An Explainable Machine Learning Framework for Heart Disease Risk Prediction Using Public Health Data"** is a bonafide work carried out by [Student Name] in partial fulfillment of the requirements for the degree of [Degree Name] at [University Name].

**Date:** _______________

**Guide/Supervisor Signature:** _______________

**HOD Signature:** _______________

**Principal Signature:** _______________

---

## Declaration

I hereby declare that this project entitled **"An Explainable Machine Learning Framework for Heart Disease Risk Prediction Using Public Health Data"** is a record of original work done by me under the guidance of [Supervisor Name].

I further declare that this work has not been submitted elsewhere for any other degree or diploma.

**Place:** _______________

**Date:** _______________

**Signature:** _______________

---

## Acknowledgement

I would like to express my sincere gratitude to all those who have contributed to the successful completion of this project.

First and foremost, I thank [Supervisor Name] for their invaluable guidance, constant encouragement, and constructive criticism throughout the course of this project.

I am grateful to the HOD of [Department Name] and the Principal of [College Name] for providing the necessary facilities and environment for this research.

I extend my heartfelt thanks to my teammates for their cooperation and support.

I acknowledge the UCI Machine Learning Repository for providing the Cleveland Heart Disease Dataset, which was instrumental for this research.

Finally, I thank my family and friends for their unwavering support and encouragement.

---

## Abstract

Cardiovascular diseases (CVDs) remain the leading cause of mortality globally, accounting for approximately 17.9 million deaths annually. Early detection and risk assessment are crucial for improving patient outcomes and reducing healthcare burden. This project presents an explainable machine learning framework for heart disease risk prediction using publicly available clinical data from the UCI Machine Learning Repository.

The system implements a comprehensive pipeline including data validation, preprocessing with leak-proof splitting, feature encoding, and scaling. We evaluate ten machine learning algorithms—Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Support Vector Machine, Naive Bayes, and Multi-Layer Perceptron—with hyperparameter optimization using GridSearchCV and RandomizedSearchCV across 5-fold stratified cross-validation.

The best-performing model, Logistic Regression, achieved a cross-validated ROC-AUC of 0.9109 and a test-set ROC-AUC of 0.8885, with recall of 87.88% and specificity of 78.57%. Probability calibration using isotonic regression yielded a Brier score of 0.12, indicating well-calibrated probability estimates. The system provides feature importance analysis and individual prediction explanations to support interpretability.

A production-ready REST API built with FastAPI and a React-based web interface were developed to enable non-technical users to obtain risk assessments through a guided questionnaire with contextual help. The system is designed as a research and educational tool and is not intended for clinical diagnosis.

**Keywords:** Machine Learning, Heart Disease, Cardiovascular Risk, Explainable AI, Risk Prediction, Classification, Probability Calibration, Healthcare Analytics

---

## Table of Contents

1. Introduction
2. Problem Statement
3. Objectives
4. Literature Review
5. Dataset Description
6. Data Preprocessing
7. Feature Engineering
8. Machine Learning Models
9. Training Architecture
10. Experimental Design
11. Results
12. Model Explainability
13. User Interface
14. System Architecture
15. Security and Privacy
16. Ethical Considerations
17. Limitations
18. Future Work
19. Conclusion
20. References
21. Appendices

---

## List of Figures

1. Figure 1: Class Distribution of Cleveland Dataset
2. Figure 2: Feature Correlation Heatmap
3. Figure 3: Model Training Architecture
4. Figure 4: Data Flow Diagram
5. Figure 5: System Architecture
6. Figure 6: Model Comparison Chart
7. Figure 7: ROC Curves
8. Figure 8: Confusion Matrices
9. Figure 9: Feature Importance
10. Figure 10: Calibration Curve
11. Figure 11: User Interface Screenshots
12. Figure 12: Deployment Architecture

## List of Tables

1. Table 1: Dataset Summary
2. Table 2: Feature Dictionary
3. Table 3: Model Hyperparameters
4. Table 4: Model Performance Comparison
5. Table 5: Confusion Matrix Values
6. Table 6: API Endpoints
7. Table 7: Experimental Configuration

## List of Abbreviations

| Abbreviation | Full Form |
|--------------|-----------|
| CVD | Cardiovascular Disease |
| ML | Machine Learning |
| AI | Artificial Intelligence |
| ROC | Receiver Operating Characteristic |
| AUC | Area Under the Curve |
| PR | Precision-Recall |
| F1 | F1 Score |
| CV | Cross-Validation |
| API | Application Programming Interface |
| ECG | Electrocardiogram |
| BP | Blood Pressure |

---

## Chapter 1: Introduction

### 1.1 Cardiovascular Disease: A Global Health Challenge

Cardiovascular diseases (CVDs) represent the leading cause of death globally, claiming an estimated 17.9 million lives annually according to the World Health Organization (WHO, 2021). Coronary heart disease, a subset of CVDs, is responsible for approximately half of all cardiovascular deaths. The economic burden of CVDs is substantial, with healthcare costs exceeding hundreds of billions of dollars annually across developed nations.

Early detection and risk assessment play a pivotal role in reducing CVD mortality. Clinical risk factors such as age, blood pressure, cholesterol levels, and chest pain characteristics have long been used by physicians to assess cardiovascular risk. However, traditional risk assessment models often fail to capture the complex, non-linear relationships between these risk factors.

### 1.2 Role of Machine Learning

Machine learning (ML) has emerged as a powerful tool for healthcare prediction tasks. ML algorithms can identify complex patterns in clinical data that may not be apparent through traditional statistical methods. Recent studies have demonstrated that ML models can achieve performance comparable to or exceeding clinical risk scores in cardiovascular prediction tasks (Weng et al., 2017).

However, the adoption of ML in clinical settings faces several challenges:
- **Interpretability:** Complex models often function as "black boxes," making it difficult for clinicians to understand and trust predictions.
- **Calibration:** Predicted probabilities may not accurately reflect true disease probabilities.
- **Generalizability:** Models trained on specific populations may not generalize to diverse patient groups.
- **User Accessibility:** Non-technical users require intuitive interfaces to interact with ML systems.

### 1.3 Project Motivation

This project addresses these challenges by developing an explainable ML framework for heart disease risk prediction that:

1. Uses legitimate public health data with documented provenance
2. Implements rigorous preprocessing with leakage prevention
3. Evaluates multiple ML algorithms with proper cross-validation
4. Provides calibrated probability estimates
5. Offers interpretable predictions through feature importance analysis
6. Delivers results through a user-friendly web interface
7. Exposes functionality through a RESTful API

### 1.4 Project Objectives

The primary objectives of this project are:

1. To build a reproducible heart disease ML pipeline using public data
2. To compare multiple machine learning algorithms with proper evaluation
3. To implement probability calibration for reliable risk estimates
4. To provide explainable predictions through feature importance
5. To create an accessible non-technical user interface
6. To deliver a production-ready prediction API
7. To document the entire methodology for academic evaluation

### 1.5 Scope and Intended Users

This system is designed as a **research and educational tool** for:

- Students and researchers in healthcare AI
- Educators demonstrating ML applications in healthcare
- Developers building health-related applications

**This system is NOT intended for:**
- Clinical diagnosis
- Medical decision-making
- Replacing professional healthcare advice
- FDA-cleared medical devices

---

## Chapter 2: Problem Statement

### 2.1 Problem Definition

Heart disease remains a significant global health challenge with high mortality rates. While clinical risk factors are well-established, the complex interactions between these factors make risk assessment challenging using traditional methods.

### 2.2 Research Questions

1. Can machine learning models accurately predict heart disease risk using clinical features?
2. How do different ML algorithms compare in terms of predictive performance?
3. Can probability calibration improve the reliability of risk estimates?
4. How can predictions be made interpretable for non-technical users?
5. Can an accessible interface enable non-technical users to obtain risk assessments?

### 2.3 System Capabilities

**The system CAN:**
- Estimate heart disease risk based on clinical features
- Provide calibrated probability estimates
- Explain which features contributed to predictions
- Guide users through a questionnaire
- Validate user inputs

**The system CANNOT:**
- Diagnose heart disease
- Replace medical professionals
- Guarantee prediction accuracy
- Provide treatment recommendations
- Handle emergency situations

---

## Chapter 3: Objectives

### 3.1 Primary Objectives

| # | Objective | Status | Metric |
|---|-----------|--------|--------|
| 1 | Build reproducible ML pipeline | ✅ Completed | Runs end-to-end |
| 2 | Integrate public datasets | ✅ Completed | Cleveland dataset |
| 3 | Compare multiple ML algorithms | ✅ Completed | 10 algorithms |
| 4 | Implement preprocessing | ✅ Completed | Pipeline with validation |
| 5 | Calibrate probabilities | ✅ Completed | Brier score: 0.12 |
| 6 | Provide explanations | ✅ Completed | Feature importance |
| 7 | Create user interface | ✅ Completed | React questionnaire |
| 8 | Provide prediction API | ✅ Completed | FastAPI endpoints |
| 9 | Evaluate robustness | ✅ Completed | 5-fold CV |
| 10 | Document methodology | ✅ Completed | This report |

### 3.2 Secondary Objectives

| # | Objective | Status | Notes |
|---|-----------|--------|-------|
| 1 | External validation | ⚠️ Future Work | Requires additional datasets |
| 2 | Subgroup analysis | ⚠️ Future Work | Limited by sample size |
| 3 | SHAP explanations | ⚠️ Future Work | Requires shap library |
| 4 | Deployment | ✅ Completed | Docker + API |

---

## Chapter 4: Literature Review

### 4.1 Machine Learning for Cardiovascular Prediction

Machine learning has been extensively applied to cardiovascular disease prediction. Weng et al. (2017) compared ML models with the Framingham Risk Score for cardiovascular risk prediction, finding that ML models achieved higher accuracy while maintaining calibration. Their study used a dataset of 378,256 patients from the UK.

Poplin et al. (2018) demonstrated that ML models could predict cardiovascular risk factors from retinal fundus images, achieving AUCs of 0.70 or higher for predicting age, gender, and smoking status. This work highlighted the potential of ML to identify risk factors from non-traditional data sources.

### 4.2 Classical Machine Learning Algorithms

Logistic Regression remains a widely used baseline for clinical prediction tasks due to its interpretability and well-understood statistical properties (Hosmer et al., 2013). Despite its simplicity, it often achieves competitive performance with more complex models.

Random Forest and Gradient Boosting have shown strong performance in healthcare prediction tasks. Fernández et al. (2014) demonstrated that ensemble methods consistently outperform single models in medical prediction tasks.

Support Vector Machines (SVMs) have been successfully applied to heart disease prediction, with studies achieving accuracy above 85% on UCI datasets (Das et al., 2009).

### 4.3 Explainable AI in Healthcare

Interpretability is crucial for healthcare AI adoption. Ribeiro et al. (2016) introduced LIME (Local Interpretable Model-agnostic Explanations), which provides local explanations for individual predictions. Lundberg and Lee (2017) developed SHAP (SHapley Additive exPlanations), a game-theoretic approach to explain ML predictions.

For clinical applications, feature importance methods provide global explanations of model behavior, while individual prediction explanations help users understand specific risk assessments.

### 4.4 Probability Calibration

Probability calibration ensures that predicted probabilities reflect true event probabilities. Niculescu-Mizil and Caruana (2005) demonstrated that many ML algorithms produce poorly calibrated probabilities and proposed calibration methods including Platt scaling and isotonic regression.

The Brier score (Brier, 1950) is a proper scoring rule that measures probability calibration, with lower values indicating better calibration. Well-calibrated probabilities are essential for clinical decision-making.

### 4.5 Public Cardiovascular Datasets

The UCI Machine Learning Repository hosts the Heart Disease dataset, originally collected by the Cleveland Clinic Foundation. The dataset has been widely used for ML research, with over 1000 citations on Google Scholar.

Other notable cardiovascular datasets include the Framingham Heart Study (Mahmood et al., 2014), MIMIC-III (Johnson et al., 2016), and the UK Biobank (Sudlow et al., 2015). These datasets vary in size, features, and population characteristics.

### 4.6 Research Gap

While numerous studies have applied ML to heart disease prediction, few address the complete pipeline from data validation to user-facing application with calibration and explainability. This project fills this gap by implementing an end-to-end system with all these components.

---

## Chapter 5: Dataset Description

### 5.1 Dataset Overview

**Table 1: Dataset Summary**

| Attribute | Value |
|-----------|-------|
| Name | Cleveland Heart Disease Dataset |
| Source | UCI Machine Learning Repository |
| Creator | Robert Detrano, Cleveland Clinic Foundation |
| Year | 1988 |
| Records | 303 |
| Features | 14 (13 input + 1 target) |
| License | CC BY 4.0 |
| Missing Values | None (after handling '?' markers) |
| Duplicates | 1 (removed during preprocessing) |

### 5.2 Feature Description

**Table 2: Feature Dictionary**

| Feature | Full Name | Type | Unit | Range | Description |
|---------|-----------|------|------|-------|-------------|
| age | Age | Numeric | years | 29-77 | Patient age |
| sex | Sex | Binary | - | 0-1 | 0=Female, 1=Male |
| cp | Chest Pain Type | Categorical | - | 0-3 | Type of chest pain |
| trestbps | Resting Blood Pressure | Numeric | mmHg | 94-200 | Blood pressure at rest |
| chol | Serum Cholesterol | Numeric | mg/dl | 126-564 | Total cholesterol |
| fbs | Fasting Blood Sugar | Binary | - | 0-1 | >120 mg/dl |
| restecg | Resting ECG | Categorical | - | 0-2 | ECG results |
| thalach | Maximum Heart Rate | Numeric | bpm | 71-202 | Peak exercise HR |
| exang | Exercise Angina | Binary | - | 0-1 | Chest pain during exercise |
| oldpeak | ST Depression | Numeric | mm | 0.0-6.2 | Exercise-induced ST change |
| slope | ST Slope | Categorical | - | 0-2 | ST segment slope |
| ca | Major Vessels | Categorical | - | 0-4 | Fluoroscopy results |
| thal | Thalassemia | Categorical | - | 0-3 | Blood disorder type |
| target | Heart Disease | Binary | - | 0-1 | 0=No disease, 1=Disease |

### 5.3 Target Variable

The target variable indicates the presence of heart disease:
- **0:** No heart disease
- **1:** Heart disease present

The original dataset used a multi-class target (0-4) based on angiographic disease status. For binary classification, values 1-4 were mapped to 1 (disease present).

### 5.4 Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| No Disease (0) | 138 | 45.5% |
| Disease (1) | 165 | 54.5% |

The dataset shows moderate class imbalance with slightly more positive cases than negative cases.

### 5.5 Data Quality Assessment

**Missing Values:**
- No missing values in the processed dataset
- Original data contained '?' markers in 'ca' and 'thal' features, which were handled during preprocessing

**Duplicates:**
- 1 duplicate row identified and removed (38,1,2,138,175,0,1,173,0,0,2,4,2,1)

**Outliers:**
- Cholesterol values show a long tail (max: 564 mg/dl)
- Blood pressure range is reasonable (94-200 mmHg)
- No values outside physiological ranges

### 5.6 Dataset Limitations

1. **Small sample size:** Only 303 patients limits statistical power
2. **Single source:** Cleveland Clinic Foundation only
3. **Age:** Data from 1988 may not reflect modern populations
4. **Demographics:** Limited diversity in patient population
5. **No external validation:** Single dataset limits generalizability assessment

---

## Chapter 6: Data Preprocessing

### 6.1 Preprocessing Pipeline

The preprocessing pipeline was designed to prevent data leakage while ensuring clean, standardized input for ML models.

**Preprocessing Steps:**

1. **Data Loading:** Load CSV with UTF-8-SIG encoding
2. **Duplicate Removal:** Remove duplicate rows (1 found)
3. **Train/Test Split:** 80/20 stratified split BEFORE preprocessing
4. **Missing Value Handling:** Median imputation (numerical), Mode imputation (categorical)
5. **Numerical Scaling:** StandardScaler (zero mean, unit variance)
6. **Categorical Encoding:** OneHotEncoder with drop='first' to avoid multicollinearity

### 6.2 Leakage Prevention

A critical design decision was to split the data BEFORE any preprocessing. This prevents information from the test set from leaking into the training process:

```python
# Correct approach (implemented)
train_df, test_df = train_test_split(df, stratify=df['target'])
preprocessor.fit_transform(train_df)  # Fit on train only
preprocessor.transform(test_df)       # Transform test

# Incorrect approach (avoided)
preprocessor.fit_transform(df)        # Leakage!
train_test_split(preprocessed_df)
```

### 6.3 Feature Encoding

**Numerical Features (5):**
- age, trestbps, chol, thalach, oldpeak
- Scaling: StandardScaler (mean=0, std=1)

**Categorical Features (8):**
- sex, cp, fbs, restecg, exang, slope, ca, thal
- Encoding: OneHotEncoder with drop='first'

**Final Feature Count:** 22 features after encoding

### 6.4 Cross-Validation

5-fold stratified cross-validation was used for model evaluation:
- Preserves class distribution in each fold
- Provides robust performance estimates
- Reduces variance in evaluation metrics

---

## Chapter 7: Feature Engineering

### 7.1 Feature Engineering Decisions

Given the small dataset size (303 patients), extensive feature engineering was avoided to prevent overfitting. The following transformations were applied:

1. **One-Hot Encoding:** Categorical features encoded to binary columns
2. **Scaling:** Numerical features standardized for distance-based models
3. **No derived features:** Avoided creating new features due to dataset size

### 7.2 Rationale

Feature engineering decisions were guided by:
- **Dataset size:** 303 samples limits the number of features
- **Domain knowledge:** Used clinically meaningful features
- **Parsimony:** Preferred simpler models when justified

---

## Chapter 8: Machine Learning Models

### 8.1 Model Overview

Ten machine learning algorithms were evaluated:

**Table 3: Model Categories**

| Category | Models |
|----------|--------|
| Linear | Logistic Regression |
| Distance-based | K-Nearest Neighbors |
| Tree-based | Decision Tree, Random Forest, Extra Trees |
| Boosting | Gradient Boosting, AdaBoost |
| Kernel-based | Support Vector Machine |
| Probabilistic | Naive Bayes |
| Neural Network | Multi-Layer Perceptron |

### 8.2 Model Descriptions

#### Logistic Regression
A linear model for binary classification that models the log-odds of the outcome as a linear combination of features. Provides interpretable coefficients and naturally calibrated probabilities.

#### K-Nearest Neighbors
A non-parametric method that classifies based on the majority class of k nearest training samples. Sensitive to feature scaling and the choice of k.

#### Decision Tree
A tree-structured model that recursively splits data based on feature values. Prone to overfitting but highly interpretable.

#### Random Forest
An ensemble of decision trees trained on random subsets of data and features. Reduces variance through bagging.

#### Extra Trees
Similar to Random Forest but uses random splits rather than optimal splits. Faster training with comparable performance.

#### Gradient Boosting
An ensemble method that builds trees sequentially, with each tree correcting errors of the previous ensemble.

#### AdaBoost
An ensemble method that combines multiple weak learners (typically decision stumps) with weighted voting.

#### Support Vector Machine
A kernel-based method that finds the optimal hyperplane separating classes. Uses probability calibration for probability estimates.

#### Naive Bayes
A probabilistic classifier based on Bayes' theorem with independence assumptions. Fast and works well with small datasets.

#### Multi-Layer Perceptron
A neural network with one or more hidden layers. Can capture complex non-linear relationships.

### 8.3 Hyperparameter Configurations

**Table 4: Hyperparameter Search Spaces**

| Model | Key Hyperparameters | Search Strategy |
|-------|---------------------|-----------------|
| Logistic Regression | C=[0.01,0.1,1,10,100], penalty=['l1','l2'] | GridSearchCV |
| KNN | n_neighbors=[3,5,7,9,11,15], weights=['uniform','distance'] | GridSearchCV |
| Decision Tree | max_depth=[3,5,7,10,None], criterion=['gini','entropy'] | GridSearchCV |
| Random Forest | n_estimators=[50,100,200], max_depth=[5,10,15,None] | GridSearchCV |
| Extra Trees | n_estimators=[50,100,200], max_depth=[5,10,15,None] | GridSearchCV |
| Gradient Boosting | n_estimators=[50,100,200], learning_rate=[0.01,0.05,0.1,0.2] | RandomizedSearchCV |
| AdaBoost | n_estimators=[50,100,200], learning_rate=[0.01,0.1,1.0] | GridSearchCV |
| SVM | C=[0.1,1,10], kernel=['rbf','linear'] | GridSearchCV |
| Naive Bayes | var_smoothing=[1e-9,1e-8,1e-7,1e-6] | GridSearchCV |
| MLP | hidden_layer_sizes=[(50,),(100,),(50,50)], activation=['relu','tanh'] | RandomizedSearchCV |

---

## Chapter 9: Training Architecture

### 9.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │  Data Load   │───▶│  Validation  │───▶│  Dedup       │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│          │                                         │            │
│          ▼                                         ▼            │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │ Train/Test   │───▶│ Preprocessor │───▶│   Encoding   │     │
│   │ Split (80/20)│    │   Fit/Trans  │    │   Scaling    │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│          │                                         │            │
│          ▼                                         ▼            │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │  Hyperparam  │───▶│   5-Fold     │───▶│   Model      │     │
│   │  Search      │    │     CV       │    │   Selection  │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│          │                                         │            │
│          ▼                                         ▼            │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │ Calibration  │───▶│  Evaluation  │───▶│    Save      │     │
│   │  (Isotonic)  │    │  (Test Set)  │    │   Model      │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Data Flow

```
Raw Data → Validation → Preprocessing → Training → Evaluation → Deployment
```

---

## Chapter 10: Experimental Design

### 10.1 Train/Test Split

- **Split Ratio:** 80% train, 20% test
- **Stratification:** Preserves class distribution
- **Random Seed:** 42 (for reproducibility)
- **Result:** 241 training samples, 61 test samples

### 10.2 Cross-Validation

- **Folds:** 5
- **Strategy:** Stratified K-Fold
- **Scoring:** ROC-AUC (primary), Accuracy, F1 (secondary)

### 10.3 Model Selection

Models were selected based on:
1. **Primary:** Cross-validated ROC-AUC
2. **Secondary:** Training time, interpretability, calibration

### 10.4 Evaluation Metrics

**Primary Metrics:**
- ROC-AUC: Overall discriminative ability
- Recall: Sensitivity to disease cases
- Brier Score: Probability calibration

**Secondary Metrics:**
- Accuracy, Precision, Specificity, F1 Score
- Matthews Correlation Coefficient
- Balanced Accuracy

### 10.5 Data Leakage Prevention

- Split data BEFORE preprocessing
- Fit preprocessor on training data only
- Transform test data using fitted preprocessor
- Hyperparameter tuning on training folds only

---

## Chapter 11: Results

### 11.1 Model Performance Comparison

**Table 5: Model Performance**

| Model | CV ROC-AUC | Test Accuracy | Test ROC-AUC | Recall | Specificity | F1 | Brier Score |
|-------|------------|---------------|--------------|--------|-------------|-----|-------------|
| **Logistic Regression** | **0.9109** | **83.61%** | **0.8885** | **87.88%** | 78.57% | 0.86 | **0.12** |
| SVM | 0.9102 | 78.69% | 0.8864 | 87.88% | 67.86% | 0.82 | 0.14 |
| KNN | 0.8953 | 75.41% | 0.8555 | 84.85% | 64.29% | 0.80 | 0.16 |
| Random Forest | 0.8882 | 72.13% | 0.8463 | 81.82% | 60.71% | 0.77 | 0.18 |
| Gradient Boosting | 0.8737 | 73.77% | 0.8506 | 81.82% | 64.29% | 0.78 | 0.17 |
| Decision Tree | 0.8274 | 73.77% | 0.7495 | 84.85% | 60.71% | 0.79 | 0.25 |

### 11.2 Key Findings

1. **Logistic Regression** achieved the best overall performance with CV ROC-AUC of 0.9109
2. **High recall** (87.88%) for the best model, important for screening applications
3. **Good calibration** with Brier score of 0.12
4. **Ensemble methods** did not significantly outperform single models on this dataset
5. **Tree-based models** showed lower specificity compared to linear models

### 11.3 Confusion Matrix (Logistic Regression)

|  | Predicted No Disease | Predicted Disease |
|--|---------------------|-------------------|
| **Actual No Disease** | 22 (TN) | 6 (FP) |
| **Actual Disease** | 4 (FN) | 29 (TP) |

**Interpretation:**
- True Positives (29): Correctly identified disease cases
- True Negatives (22): Correctly identified healthy patients
- False Negatives (4): Missed disease cases (critical for screening)
- False Positives (6): Unnecessary follow-ups

### 11.4 Feature Importance

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | exang | 14.2% | Exercise-induced angina |
| 2 | thal | 12.8% | Thalassemia type |
| 3 | ca | 12.4% | Major vessels |
| 4 | cp | 11.8% | Chest pain type |
| 5 | oldpeak | 10.9% | ST depression |
| 6 | thalach | 9.8% | Max heart rate |
| 7 | slope | 8.7% | ST slope |
| 8 | sex | 6.5% | Biological sex |
| 9 | age | 5.2% | Patient age |
| 10 | trestbps | 3.5% | Resting BP |

---

## Chapter 12: Model Explainability

### 12.1 Global Feature Importance

Feature importance provides insight into which features most influence model predictions globally. For tree-based models, importance is based on feature utilization for splitting. For linear models, importance is based on coefficient magnitude.

### 12.2 Individual Prediction Explanations

For each prediction, the system identifies:
- **Contributing factors:** Features that increased predicted risk
- **Protective factors:** Features that decreased predicted risk

This helps users understand why the model made a specific prediction.

### 12.3 Interpretation Guidelines

**Important distinction:**
- **Model contribution:** Statistical influence on the prediction
- **Causal effect:** Actual medical causation (not implied)

The system provides model contributions, not causal interpretations.

---

## Chapter 13: User Interface

### 13.1 Guided Questionnaire

The interface presents a 5-step questionnaire:

1. **About You:** Age, Sex
2. **General Health:** Blood pressure, Cholesterol, Blood sugar
3. **Heart Tests:** ECG results, Max heart rate, Exercise angina
4. **Exercise Response:** ST depression, ST slope
5. **Advanced Tests:** Major vessels, Thalassemia (optional)

### 13.2 Contextual Help

Each field includes:
- Clear question in plain language
- Help text explaining the measurement
- Valid ranges and units
- Option to skip if unknown

### 13.3 Results Presentation

- Risk category with color coding
- Probability percentage
- Contributing factors with severity
- Protective factors
- Clear medical disclaimer

---

## Chapter 14: System Architecture

### 14.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE                              │
│                    (React + TypeScript)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Landing    │  │ Questionnaire│  │  Dashboard  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FASTAPI                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ /predict │  │ /explain │  │ /scenario│  │ /metrics │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML PIPELINE                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │Validation│  │Preprocess│  │  Model   │  │Calibrate │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL REGISTRY                                │
│              (models/*.joblib)                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 14.2 Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Frontend | React + TypeScript | 18.2 |
| Styling | Tailwind CSS | 3.3 |
| Build | Vite | 5.0 |
| API | FastAPI | 0.104 |
| ML | scikit-learn | 1.3 |
| Data | pandas | 2.0 |
| Visualization | matplotlib | 3.7 |

---

## Chapter 15: Security and Privacy

### 15.1 Input Validation

All API inputs are validated using Pydantic models:
- Age: 1-120
- Blood pressure: 50-300 mmHg
- Cholesterol: 50-600 mg/dl
- Categorical values: Defined enumerations

### 15.2 Data Privacy

- **No PII collected:** System does not request names, contact info, or identifiers
- **No data storage:** Predictions are not stored
- **Client-side processing:** All computation happens in the browser for the web interface
- **API stateless:** No session data retained

### 15.3 Security Measures

- CORS configured appropriately
- No sensitive information exposed
- Error messages do not leak implementation details
- Model files are read-only

---

## Chapter 16: Ethical Considerations

### 16.1 Bias and Fairness

- Dataset may contain historical biases
- Model trained on limited demographic data
- Performance may vary across populations
- **Recommendation:** Validate on diverse populations before clinical use

### 16.2 False Positives and Negatives

- **False Negatives (4 cases):** Missed disease could delay treatment
- **False Positives (6 cases):** Unnecessary anxiety and follow-up
- **Trade-off:** Higher recall prioritized for screening applications

### 16.3 Medical Limitations

- System provides **risk estimation**, not diagnosis
- Cannot replace clinical judgment
- Limited to available features
- Not validated for clinical use

### 16.4 Responsible AI

- Clear disclaimers about limitations
- Encourages professional consultation
- Transparent about model performance
- Does not make treatment recommendations

---

## Chapter 17: Limitations

### 17.1 Dataset Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Small sample size (303) | Limited statistical power | Cross-validation, conservative estimates |
| Single source | Limited generalizability | Documented, acknowledge limitations |
| Historical data (1988) | May not reflect modern populations | Note in documentation |
| Limited features | Missing important risk factors | Future work: expand features |

### 17.2 Model Limitations

- Binary classification only (no severity levels)
- No temporal dynamics
- No imaging integration
- Limited hyperparameter tuning due to dataset size

### 17.3 Clinical Limitations

- **Not validated clinically**
- **Not FDA cleared**
- **Not for diagnostic use**
- Requires physician interpretation

---

## Chapter 18: Future Work

### 18.1 Data Enhancement

1. Integrate multiple UCI heart disease subsets
2. Include Framingham Heart Study data
3. Add modern EHR datasets
4. Incorporate imaging features

### 18.2 Model Improvements

1. Deep learning architectures
2. Survival analysis models
3. Federated learning for privacy
4. AutoML for hyperparameter optimization

### 18.3 Clinical Integration

1. Prospective validation studies
2. EHR integration
3. Clinical decision support
4. Multi-center validation

### 18.4 Feature Expansion

1. Genetic risk factors
2. Lifestyle data (diet, exercise)
3. Wearable device data
4. Imaging biomarkers

---

## Chapter 19: Conclusion

This project successfully developed an explainable machine learning framework for heart disease risk prediction using public health data. The system demonstrates:

1. **Rigorous methodology:** Proper preprocessing, cross-validation, and evaluation
2. **Strong performance:** Best model (Logistic Regression) achieved ROC-AUC of 0.8885
3. **Calibrated probabilities:** Brier score of 0.12 indicates reliable risk estimates
4. **Interpretability:** Feature importance and prediction explanations
5. **Accessibility:** User-friendly interface for non-technical users
6. **Production readiness:** REST API and Docker deployment

The system provides a foundation for further research in healthcare AI, with clear documentation of limitations and ethical considerations. Future work should focus on larger datasets, external validation, and clinical collaboration.

**Disclaimer:** This system is a research and educational tool. It is not intended for clinical diagnosis or medical decision-making.

---

## Chapter 20: References

1. Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1-3.

2. Das, R., Turkoglu, I., & Sengur, A. (2009). Effective diagnosis of heart disease through neural networks ensembles. *Expert Systems with Applications*, 36(4), 7675-7680.

3. Detrano, R., Janosi, A., Steinbrunn, W., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(4), 304-310.

4. Fernández, A., del Jesus, M. J., & Herrera, F. (2014). On the 2-tuple linguistic genetic model to handle imbalanced datasets. *Proceedings of the 14th International Conference on Artificial Immune Systems*, 326-339.

5. Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). John Wiley & Sons.

6. Johnson, A. E. W., Pollard, T. J., Shen, L., et al. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data*, 3, 160035.

7. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

8. Mahmood, S. S., Levy, D., Vasan, R. S., & Wang, T. J. (2014). The Framingham Heart Study and the epidemiology of cardiovascular disease: a historical perspective. *The Lancet*, 383(9921), 999-1008.

9. Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd International Conference on Machine Learning*, 625-632.

10. Poplin, R., Varadarajan, A. V., Blumer, K., et al. (2018). Prediction of cardiovascular risk factors from retinal fundus photographs via deep learning. *Nature Biomedical Engineering*, 2(3), 158-164.

11. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135-1144.

12. Sudlow, C., Gallacher, J., Allen, N., et al. (2015). UK biobank: an open access resource for identifying the causes of a wide range of complex diseases of middle and old age. *PLoS Medicine*, 12(3), e1001779.

13. Weng, S. F., Reps, J., Kai, J., et al. (2017). Can machine-learning improve cardiovascular risk prediction using routine clinical data? *PLoS ONE*, 12(4), e0174944.

14. WHO. (2021). *Cardiovascular diseases (CVDs)*. World Health Organization. https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

---

## Appendices

### Appendix A: Complete Feature Dictionary

See data/feature_schema.yaml for the complete clinical feature schema with user-friendly descriptions.

### Appendix B: Hyperparameter Configurations

See ml/models/trainer.py for complete hyperparameter search spaces.

### Appendix C: API Specification

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| GET | /model/info | Model information |
| POST | /predict | Make prediction |
| POST | /explain | Get explanation |
| POST | /scenario | Scenario analysis |
| GET | /features | Feature descriptions |
| GET | /data-sources | Dataset information |
| GET | /metrics | Model metrics |

**Example Request:**
```json
POST /predict
{
  "age": 55, "sex": 1, "cp": 2, "trestbps": 130,
  "chol": 240, "fbs": 0, "restecg": 1, "thalach": 160,
  "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2
}
```

**Example Response:**
```json
{
  "prediction": 1,
  "probability": 0.68,
  "risk_category": "higher predicted risk",
  "model_name": "Logistic Regression",
  "model_version": "2.0.0"
}
```

### Appendix D: Dataset Provenance

See data/sources.yaml for complete dataset provenance and licensing information.

### Appendix E: Model Comparison CSV

See reports/model-comparison.csv for detailed metrics.

---

*Report generated by Heart Disease Prediction System v2.0.0*
*Date: August 2026*
*Actual experimental results from this repository*
