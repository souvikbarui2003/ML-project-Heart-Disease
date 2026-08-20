"""
Enhanced model training with calibration, explainability, and ensembling.
"""
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml.config import CV_FOLDS, RANDOM_SEED
from ml.utils.logging import logger


def get_enhanced_model_configs() -> Dict[str, Dict[str, Any]]:
    """Get enhanced model configurations with hyperparameter search spaces."""
    return {
        "Logistic Regression": {
            "model": LogisticRegression(
                random_state=RANDOM_SEED,
                max_iter=1000,
            ),
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"],
            },
        },
        "K-Nearest Neighbors": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7, 9, 11, 15],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"],
            },
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=RANDOM_SEED),
            "params": {
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy"],
            },
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=RANDOM_SEED),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "Extra Trees": {
            "model": ExtraTreesClassifier(random_state=RANDOM_SEED),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
            },
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=RANDOM_SEED),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5, 10],
            },
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(random_state=RANDOM_SEED),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 1.0],
            },
        },
        "SVM": {
            "model": SVC(probability=True, random_state=RANDOM_SEED),
            "params": {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"],
            },
        },
        "Naive Bayes": {
            "model": GaussianNB(),
            "params": {
                "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
            },
        },
        "MLP Neural Network": {
            "model": MLPClassifier(
                random_state=RANDOM_SEED,
                max_iter=1000,
                early_stopping=True,
            ),
            "params": {
                "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "activation": ["relu", "tanh"],
                "alpha": [0.0001, 0.001, 0.01],
            },
        },
    }


def train_with_calibration(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    cv_folds: int = CV_FOLDS,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Train all models with calibration and comprehensive metrics.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    best_models = {}
    calibrated_models = {}
    best_params = {}
    training_results = []

    configs = get_enhanced_model_configs()

    for name, config in configs.items():
        logger.info(f"Training {name}...")
        start_time = time.time()

        try:
            # Hyperparameter tuning
            param_count = np.prod([len(v) for v in config["params"].values()])

            if param_count <= 20:
                search = GridSearchCV(
                    config["model"],
                    config["params"],
                    cv=cv,
                    scoring="roc_auc",
                    n_jobs=-1,
                    refit=True,
                )
            else:
                search = RandomizedSearchCV(
                    config["model"],
                    config["params"],
                    n_iter=min(50, int(param_count)),
                    cv=cv,
                    scoring="roc_auc",
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                    refit=True,
                )

            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params[name] = search.best_params_
            cv_score = search.best_score_

            # Calibrate the model
            try:
                calibrated = CalibratedClassifierCV(
                    best_model, method="isotonic", cv=3
                )
                calibrated.fit(X_train, y_train)
                calibrated_models[name] = calibrated
            except Exception as e:
                logger.warning(f"Calibration failed for {name}: {e}")
                calibrated_models[name] = best_model

            # Calculate metrics
            y_prob = best_model.predict_proba(X_test)[:, 1]
            y_pred = best_model.predict(X_test)

            from sklearn.metrics import (
                accuracy_score,
                brier_score_loss,
                f1_score,
                matthews_corrcoef,
                precision_score,
                recall_score,
                roc_auc_score,
            )

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred)),
                "recall": float(recall_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred)),
                "roc_auc": float(roc_auc_score(y_test, y_prob)),
                "brier_score": float(brier_score_loss(y_test, y_prob)),
                "matthews_corrcoef": float(matthews_corrcoef(y_test, y_pred)),
            }

            training_time = time.time() - start_time

            result = {
                "model_name": name,
                "cv_score": float(cv_score),
                "training_time": float(training_time),
                "best_params": best_params.get(name, {}),
                "test_metrics": metrics,
            }
            training_results.append(result)
            best_models[name] = best_model

            logger.info(
                f"{name} - CV ROC-AUC: {cv_score:.4f}, "
                f"Test ROC-AUC: {metrics['roc_auc']:.4f}, "
                f"Brier: {metrics['brier_score']:.4f} "
                f"(Time: {training_time:.2f}s)"
            )

        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
            training_time = time.time() - start_time
            training_results.append({
                "model_name": name,
                "cv_score": 0.0,
                "training_time": float(training_time),
                "error": str(e),
            })

    return best_models, calibrated_models, best_params, training_results


def create_ensemble(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    voting: str = "soft",
) -> Any:
    """Create an ensemble of top models."""
    # Select top 3 models by CV score
    sorted_models = sorted(
        [(name, m) for name, m in models.items()],
        key=lambda x: x[1].score(X_train, y_train) if hasattr(x[1], 'score') else 0,
        reverse=True
    )[:3]

    estimator_list = [(name, model) for name, model in sorted_models]

    ensemble = VotingClassifier(
        estimators=estimator_list,
        voting=voting,
    )
    ensemble.fit(X_train, y_train)

    logger.info(f"Created ensemble with: {[name for name, _ in estimator_list]}")
    return ensemble


def create_stacking_ensemble(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    """Create a stacking ensemble."""
    # Use top 3 base models
    sorted_models = sorted(
        models.items(),
        key=lambda x: x[1].score(X_train, y_train) if hasattr(x[1], 'score') else 0,
        reverse=True
    )[:3]

    estimator_list = [(name, model) for name, model in sorted_models]

    stacking = StackingClassifier(
        estimators=estimator_list,
        final_estimator=LogisticRegression(random_state=RANDOM_SEED),
        cv=3,
    )
    stacking.fit(X_train, y_train)

    logger.info(f"Created stacking ensemble with: {[name for name, _ in estimator_list]}")
    return stacking
