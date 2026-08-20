"""
CLI module for evaluating a trained model.

Usage:
    python -m src.evaluate
"""
import json
import sys

from src.config import METRICS_DIR
from src.data.loader import load_dataset
from src.features.preprocessing import prepare_data
from src.evaluation.metrics import (
    evaluate_all_models,
    get_classification_report,
)
from src.evaluation.plots import (
    plot_confusion_matrices,
    plot_pr_curves,
    plot_roc_curves,
)
from src.inference.predictor import load_model
from src.inference.explainer import get_global_feature_importance
from src.utils.logging import logger


def main() -> None:
    """Main evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("Heart Disease Prediction - Model Evaluation")
    logger.info("=" * 60)

    # Load model
    try:
        artifact = load_model()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    model = artifact["model"]
    model_name = artifact.get("metadata", {}).get("model_name", "unknown")

    # Load and prepare data
    logger.info("Loading and preparing data...")
    df = load_dataset()
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)

    # Evaluate
    logger.info(f"\nEvaluating {model_name}...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Classification report
    report = get_classification_report(y_test, y_pred, model_name)

    # Full evaluation
    all_metrics, all_curves = evaluate_all_models(
        {model_name: model}, X_test, y_test
    )

    # Save results
    results_path = METRICS_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "metrics": all_metrics[model_name],
                "classification_report": report,
            },
            f,
            indent=2,
            default=str,
        )

    logger.info(f"\nResults saved to {results_path}")
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
