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


def test_saved_comparison_batch_and_session_isolation():
    with TestClient(app) as a, TestClient(app) as b:
        saved=a.post('/runs',json=PAYLOAD).json()
        assert saved['kind']=='estimate'
        assert b.get('/runs/'+saved['id']+'/report.csv').status_code==404
        assert a.get('/runs/'+saved['id']+'/report.csv').status_code==200
        compared=a.post('/compare',json={'baseline':PAYLOAD,'alternative':dict(PAYLOAD,acres=4)}).json()
        assert len(compared['data']['records'])==2
        assert compared['data']['difference']==pytest.approx(compared['data']['records'][1]['estimated_price']-compared['data']['records'][0]['estimated_price'])
        sample=a.get('/sample.csv').content
        batch=a.post('/runs/csv',files={'file':('sample.csv',sample,'text/csv')})
        assert batch.status_code==200 and len(batch.json()['data']['records'])==3
        assert len(a.get('/runs').json())==3
        assert b.get('/runs').json()==[]
        assert a.post('/runs/csv',files={'file':('bad.csv',b'acres,location\n1,Unknown','text/csv')}).status_code==422
