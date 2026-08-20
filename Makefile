.PHONY: help install train predict test lint format docker-build docker-run clean

# Default target
help:
	@echo "Heart Disease Prediction - Available Commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make train       - Train all models"
	@echo "  make predict     - Run prediction (use ARGS='--input data.csv')"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linter"
	@echo "  make format      - Format code with black"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run  - Run Docker container"
	@echo "  make api         - Start API server"
	@echo "  make clean       - Clean generated files"

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# Train models
train:
	python -m src.train

# Run prediction
predict:
	python scripts/predict.py $(ARGS)

# Run tests
test:
	pytest tests/ -v --tb=short

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=src --cov-report=html

# Lint code
lint:
	flake8 src/ api/ tests/
	mypy src/ api/

# Format code
format:
	black src/ api/ tests/
	isort src/ api/ tests/

# Build Docker image
docker-build:
	docker-compose build

# Run with Docker
docker-run:
	docker-compose up -d

# Stop Docker
docker-stop:
	docker-compose down

# Start API server locally
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Clean generated files
clean:
	rm -rf __pycache__ src/__pycache__ src/**/__pycache__
	rm -rf .pytest_cache htmlcov .coverage
	rm -rf models/*.pkl models/*.json
	rm -rf reports/figures/*.png reports/metrics/*.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Generate reports
reports:
	python -m src.train
	@echo "Reports generated in reports/"

# Validate data
validate:
	python -c "from src.data.loader import load_dataset; from src.data.validation import validate_dataset; validate_dataset(load_dataset())"

# Show project structure
structure:
	@find . -type f -name "*.py" | head -30
	@echo "..."
	@tree -I '__pycache__|*.pyc|.git' -L 3 2>/dev/null || find . -type d -not -path '*/\.*' -not -path '*__pycache__*' | head -20
