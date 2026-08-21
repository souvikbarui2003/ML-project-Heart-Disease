"""
Evaluation visualization module for model comparison and analysis.
"""
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from ml.config import FIGURES_DIR
from ml.utils.logging import logger


# Consistent style
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10


def plot_roc_curves(
    all_curves: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    all_metrics: Dict[str, Dict[str, Any]],
    save_path: Path = FIGURES_DIR / "roc_curves.png",
) -> None:
    """
    Plot ROC curves for all models.

    Args:
        all_curves: Dictionary mapping model names to (fpr, tpr, thresholds).
        all_metrics: Dictionary mapping model names to metrics.
        save_path: Path to save the figure.
    """
    plt.figure(figsize=(10, 8))

    for name, (fpr, tpr, _) in all_curves.items():
        auc = all_metrics[name]["roc_auc"]
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title("ROC Curves - Model Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curves saved to {save_path}")


def plot_confusion_matrices(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Path = FIGURES_DIR / "confusion_matrices.png",
) -> None:
    """
    Plot confusion matrices for all models.

    Args:
        models: Dictionary of trained models.
        X_test: Test features.
        y_test: True labels.
        save_path: Path to save the figure.
    """
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"],
            ax=axes[idx],
            cbar=False,
        )
        axes[idx].set_title(name, fontsize=11, fontweight="bold")
        axes[idx].set_ylabel("True Label")
        axes[idx].set_xlabel("Predicted Label")

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "Confusion Matrices", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrices saved to {save_path}")


def plot_pr_curves(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Path = FIGURES_DIR / "pr_curves.png",
) -> None:
    """
    Plot Precision-Recall curves for all models.

    Args:
        models: Dictionary of trained models.
        X_test: Test features.
        y_test: True labels.
        save_path: Path to save the figure.
    """
    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_test, y_prob)
        plt.plot(recall, precision, linewidth=2, label=f"{name} (AP = {ap:.4f})")

    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"PR curves saved to {save_path}")


def plot_feature_importance(
    model: Any,
    feature_names: list,
    model_name: str = "model",
    top_n: int = 15,
    save_path: Path = FIGURES_DIR / "feature_importance.png",
) -> None:
    """
    Plot feature importance for tree-based models.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names: List of feature names.
        model_name: Name of the model.
        top_n: Number of top features to show.
        save_path: Path to save the figure.
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning(
            f"{model_name} does not support feature_importance_. Skipping."
        )
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    plt.figure(figsize=(10, 6))
    plt.barh(
        range(len(indices)),
        importances[indices],
        align="center",
        color=sns.color_palette("husl", len(indices)),
    )
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.title(
        f"Top {top_n} Feature Importances - {model_name}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance plot saved to {save_path}")


def plot_model_comparison(
    training_results: List[Dict[str, Any]],
    save_path: Path = FIGURES_DIR / "model_comparison.png",
) -> None:
    """
    Plot model comparison bar chart.

    Args:
        training_results: List of training result dictionaries.
        save_path: Path to save the figure.
    """
    valid_results = [r for r in training_results if "error" not in r]
    names = [r["model_name"] for r in valid_results]
    scores = [r["cv_score"] for r in valid_results]
    times = [r["training_time"] for r in valid_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # CV Score comparison
    colors = sns.color_palette("husl", len(names))
    bars1 = ax1.barh(names, scores, color=colors)
    ax1.set_xlabel("CV ROC-AUC Score")
    ax1.set_title("Model Performance Comparison", fontweight="bold")
    ax1.set_xlim(0, 1)
    for bar, score in zip(bars1, scores):
        ax1.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va="center",
            fontsize=10,
        )

    # Training time comparison
    bars2 = ax2.barh(names, times, color=colors)
    ax2.set_xlabel("Training Time (seconds)")
    ax2.set_title("Training Time Comparison", fontweight="bold")
    for bar, time_val in zip(bars2, times):
        ax2.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{time_val:.2f}s",
            va="center",
            fontsize=10,
        )

    plt.suptitle(
        "Model Comparison Dashboard", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Model comparison plot saved to {save_path}")
