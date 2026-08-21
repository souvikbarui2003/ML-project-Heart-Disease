"""
Model training with multiple algorithms, cross-validation, and hyperparameter tuning.
"""
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml.config import CV_FOLDS, RANDOM_SEED
from ml.utils.logging import logger


def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get model configurations with hyperparameter search spaces.

    Returns:
        Dictionary mapping model names to their configurations.
    """
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
                "metric": ["euclidean", "manhattan"],
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
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=RANDOM_SEED),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5, 10],
            },
        },
        "SVM": {
            "model": SVC(
                probability=True,
                random_state=RANDOM_SEED,
            ),
            "params": {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"],
            },
        },
    }


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    tune_hyperparameters: bool = True,
    cv_folds: int = CV_FOLDS,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Train all models with cross-validation and optional hyperparameter tuning.

    Args:
        X_train: Training features.
        y_train: Training labels.
        feature_names: List of feature names.
        tune_hyperparameters: Whether to perform hyperparameter tuning.
        cv_folds: Number of cross-validation folds.

    Returns:
        Tuple of (best_models, best_params, training_results).
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    best_models = {}
    best_params = {}
    training_results = []

    configs = get_model_configs()

    for name, config in configs.items():
        logger.info(f"Training {name}...")
        start_time = time.time()

        try:
            if tune_hyperparameters and config["params"]:
                # Use GridSearchCV for small search spaces, RandomizedSearchCV for larger
                param_count = np.prod(
                    [len(v) for v in config["params"].values()]
                )

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
            else:
                best_model = config["model"]
                best_model.fit(X_train, y_train)
                cv_scores = cross_val_score(
                    best_model,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="roc_auc",
                )
                cv_score = cv_scores.mean()
                best_params[name] = config.get("params", {})

            training_time = time.time() - start_time

            result = {
                "model_name": name,
                "cv_score": float(cv_score),
                "training_time": float(training_time),
                "best_params": best_params.get(name, {}),
            }
            training_results.append(result)

            best_models[name] = best_model
            logger.info(
                f"{name} - CV ROC-AUC: {cv_score:.4f} "
                f"(Training time: {training_time:.2f}s)"
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

    return best_models, best_params, training_results


def select_best_model(
    training_results: List[Dict[str, Any]],
) -> str:
    """
    Select the best model based on CV score.

    The selection considers:
    - Predictive performance (primary)
    - Training time (tie-breaker)

    Args:
        training_results: List of training result dictionaries.

    Returns:
        Name of the best model.
    """
    valid_results = [r for r in training_results if "error" not in r]

    if not valid_results:
        raise ValueError("No models trained successfully")

    # Sort by CV score (descending), then by training time (ascending)
    valid_results.sort(
        key=lambda x: (-x["cv_score"], x["training_time"])
    )

    best = valid_results[0]
    logger.info(
        f"Selected best model: {best['model_name']} "
        f"(CV ROC-AUC: {best['cv_score']:.4f})"
    )

    return best["model_name"]
