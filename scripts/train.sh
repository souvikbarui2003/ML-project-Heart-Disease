#!/bin/bash
# Heart Disease Prediction - Training Script
# Usage: bash scripts/train.sh

set -e

echo "=========================================="
echo "Heart Disease Prediction - Training"
echo "=========================================="

# Check Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

# Train model
echo "Starting training..."
python -m src.train

echo ""
echo "Training complete!"
echo "Model saved to: models/"
echo "Reports saved to: reports/"
