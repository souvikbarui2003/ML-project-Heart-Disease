"""
Tests for FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient

# NOTE: API tests require a trained model to be loaded.
# Run `python -m src.train` before running these tests.


@pytest.fixture
def client():
    """Create a test client."""
    from api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns response."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_valid_prediction(self, client):
        """Test valid prediction request."""
        response = client.post(
            "/predict",
            json={
                "age": 55, "sex": 1, "cp": 2, "trestbps": 130,
                "chol": 240, "fbs": 0, "restecg": 1, "thalach": 160,
                "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "risk_category" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["probability"] <= 1

    def test_invalid_prediction(self, client):
        """Test prediction with invalid input is rejected."""
        response = client.post(
            "/predict",
            json={"age": 55},  # Missing required fields
        )
        assert response.status_code == 422  # Validation error

    def test_invalid_field_range(self, client):
        """Test prediction with out-of-range values."""
        response = client.post(
            "/predict",
            json={
                "age": 200,  # Out of range
                "sex": 1, "cp": 2, "trestbps": 130,
                "chol": 240, "fbs": 0, "restecg": 1, "thalach": 160,
                "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2,
            },
        )
        assert response.status_code == 422  # Validation error

    def test_batch_prediction(self, client):
        """Test batch prediction."""
        response = client.post(
            "/predict/batch",
            json=[
                {
                    "age": 55, "sex": 1, "cp": 2, "trestbps": 130,
                    "chol": 240, "fbs": 0, "restecg": 1, "thalach": 160,
                    "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2,
                },
                {
                    "age": 35, "sex": 0, "cp": 1, "trestbps": 120,
                    "chol": 200, "fbs": 0, "restecg": 0, "thalach": 180,
                    "exang": 0, "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2,
                },
            ],
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert data["count"] == 2
