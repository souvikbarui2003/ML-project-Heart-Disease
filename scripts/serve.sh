#!/bin/bash
# Heart Disease Prediction - API Server
# Usage: bash scripts/serve.sh

set -e

echo "=========================================="
echo "Heart Disease Prediction API"
echo "=========================================="

PORT=${API_PORT:-8000}
HOST=${API_HOST:-0.0.0.0}

echo "Starting API on $HOST:$PORT..."

# Check if model exists
if [ ! -f "models/best_model.joblib" ]; then
    echo "No trained model found. Training first..."
    python -m src.train
fi

python -m uvicorn api.main:app --host "$HOST" --port "$PORT"
