"""
CLI module for training heart disease prediction models.

Usage:
    python -m src.train
"""
import json
import sys
from pathlib import Path

from src.config import (
    METRICS_DIR,
    MODEL_DIR,
    RANDOM_SEED,
)
from src.data.loader import get_dataset_info, load_dataset
from src.features.preprocessing import prepare_data
from src.data.validation import validate_dataset
from src.evaluation.metrics import evaluate_all_models
from src.evaluation.plots import (
    plot_confusion_matrices,
    plot_feature_importance,
    plot_model_comparison,
    plot_pr_curves,
    plot_roc_curves,
)
from src.models.persistence import (
    compute_dataset_hash,
    get_model_info,
    save_model,
)
from src.models.trainer import select_best_model, train_all_models
from src.utils.logging import logger


def main() -> None:
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Heart Disease Prediction - Model Training")
    logger.info("=" * 60)

    # Step 1: Load data
    logger.info("\n[Step 1] Loading dataset...")
    df = load_dataset()
    dataset_info = get_dataset_info(df)

    # Step 2: Validate data
    logger.info("\n[Step 2] Validating dataset...")
    try:
        warnings = validate_dataset(df)
        if warnings:
            logger.warning(f"Validation warnings: {len(warnings)}")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

    # Step 3: Preprocess data (NO LEAKAGE - split before preprocessing)
    logger.info("\n[Step 3] Preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)

    # Step 4: Train all models
    logger.info("\n[Step 4] Training models...")
    best_models, best_params, training_results = train_all_models(
        X_train, y_train, feature_names
    )

    # Step 5: Select best model
    logger.info("\n[Step 5] Selecting best model...")
    best_model_name = select_best_model(training_results)
    best_model = best_models[best_model_name]

    # Step 6: Evaluate on test set
    logger.info("\n[Step 6] Final evaluation on test set...")
    all_metrics, all_curves = evaluate_all_models(best_models, X_test, y_test)

    # Step 7: Generate plots
    logger.info("\n[Step 7] Generating evaluation plots...")
    plot_roc_curves(all_curves, all_metrics)
    plot_confusion_matrices(best_models, X_test, y_test)
    plot_pr_curves(best_models, X_test, y_test)
    plot_model_comparison(training_results)

    # Feature importance for best model (if supported)
    plot_feature_importance(
        best_model, feature_names, best_model_name
    )

    # Step 8: Save best model
    logger.info("\n[Step 8] Saving best model...")
    dataset_hash = compute_dataset_hash(df)
    model_path = save_model(
        model=best_model,
        model_name=best_model_name,
        feature_names=feature_names,
        metrics=all_metrics[best_model_name],
        training_info={
            "training_results": training_results,
            "best_params": best_params.get(best_model_name, {}),
            "dataset_info": {
                "shape": dataset_info["shape"],
                "hash": dataset_hash,
            },
        },
        dataset_hash=dataset_hash,
    )

    # Step 9: Save results
    results_path = METRICS_DIR / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "training_results": training_results,
                "all_metrics": all_metrics,
                "best_model": best_model_name,
                "dataset_info": dataset_info,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f"\nResults saved to {results_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best model: {best_model_name}")
    logger.info(
        f"Test metrics: Accuracy={all_metrics[best_model_name]['accuracy']:.4f}, "
        f"ROC-AUC={all_metrics[best_model_name]['roc_auc']:.4f}, "
        f"Recall={all_metrics[best_model_name]['recall_sensitivity']:.4f}, "
        f"Specificity={all_metrics[best_model_name]['specificity']:.4f}"
    )
    logger.info(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
