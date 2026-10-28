
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_prediction_endpoint():
    payload = {
        "step": 100,
        "type": "TRANSFER",
        "amount": 5000,
        "oldbalanceOrg": 10000,
        "newbalanceOrig": 5000,
        "oldbalanceDest": 2000,
        "newbalanceDest": 7000
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert "threshold" in data
    assert "decision" in data
    assert "model" in data
    assert 0 <= data["fraud_probability"] <= 1
    assert data["model"] == "XGBoost"


def test_invalid_transaction_type():
    payload = {
        "step": 100,
        "type": "INVALID_TYPE",
        "amount": 5000,
        "oldbalanceOrg": 10000,
        "newbalanceOrig": 5000,
        "oldbalanceDest": 2000,
        "newbalanceDest": 7000
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
