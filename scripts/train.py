#!/usr/bin/env python
"""
Training script for heart disease prediction models.

Usage:
    python scripts/train.py
    python scripts/train.py --no-tune
    python scripts/train.py --cv-folds 10
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import main as train_main
from src.utils.logging import logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train heart disease prediction models"
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip hyperparameter tuning",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Starting training with {args.cv_folds} CV folds")
    train_main()
