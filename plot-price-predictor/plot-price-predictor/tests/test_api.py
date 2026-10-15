import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


PAYLOAD = {
    "acres": 2.5, "location": "Lakeview", "zoning": "residential",
    "road_access": "paved", "corner_plot": True,
    "distance_to_city_km": 4.0, "frontage_m": 30.0,
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["models_loaded"] is True


def test_models_endpoint(client):
    r = client.get("/models")
    assert r.status_code == 200
    assert "neural_net" in r.json()["available"]


def test_predict_all_models(client):
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert set(body["predictions"]) == {"ridge", "random_forest",
                                        "gradient_boosting", "neural_net"}
    assert body["estimated_price"] > 0
    assert body["price_range"]["low"] < body["estimated_price"] < body["price_range"]["high"]


def test_invalid_acres_rejected(client):
    bad = dict(PAYLOAD, acres=-1)
    assert client.post("/predict", json=bad).status_code == 422
