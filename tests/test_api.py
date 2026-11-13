
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["model_loaded"] is True
    assert data["input_features"] == 13


def test_predict_endpoint():
    payload = {
        "features": [0.0] * 13
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "anomaly_score" in data
    assert "threshold" in data
    assert "is_anomaly" in data

    assert isinstance(data["anomaly_score"], float)
    assert isinstance(data["is_anomaly"], bool)

    assert data["anomaly_score"] >= 0
    assert data["threshold"] > 0



def test_predict_raw_endpoint():
    measurements = []

    for hour in range(6):
        measurements.append({
            "timestamp": f"2025-01-01T{hour:02d}:00:00",
            "oil_rate": 100 + hour,
            "gas_rate": 200 + hour,
            "water_rate": 20,
            "tubing_pressure": 3000 - 5 * hour,
            "casing_pressure": 3100 - 5 * hour,
            "temperature": 80,
            "choke_size": 40
        })

    payload = {
        "well_id": "WELL_001",
        "measurements": measurements
    }

    response = client.post("/predict-raw", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 1

    prediction = data["predictions"][0]

    assert prediction["well_id"] == "WELL_001"
    assert prediction["timestamp"] == "2025-01-01T05:00:00"

    assert isinstance(prediction["anomaly_score"], float)
    assert isinstance(prediction["is_anomaly"], bool)

    assert prediction["anomaly_score"] >= 0
    assert prediction["threshold"] > 0



def test_predict_invalid_feature_count():
    payload = {
        "features": [0.0] * 5
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
