"""Unit tests for the Flask API (assignment task 5)."""
import pytest

from app.main import create_app
from src import config

VALID_SAMPLE = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3,
    "slope": 0, "ca": 0, "thal": 1,
}


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.testing = True
    return app.test_client()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_predict_valid(client):
    r = client.post("/predict", json=VALID_SAMPLE)
    assert r.status_code == 200
    body = r.get_json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["confidence"] <= 1.0
    probs = body["probabilities"]
    assert abs(probs["no_disease"] + probs["disease"] - 1.0) < 1e-6


def test_predict_missing_feature(client):
    r = client.post("/predict", json={"age": 63})
    assert r.status_code == 400
    assert "Missing" in r.get_json()["error"]


def test_predict_non_numeric(client):
    bad = dict(VALID_SAMPLE, chol="high")
    r = client.post("/predict", json=bad)
    assert r.status_code == 400


def test_predict_unknown_field(client):
    bad = dict(VALID_SAMPLE, foo=1)
    r = client.post("/predict", json=bad)
    assert r.status_code == 400


def test_metrics_endpoint(client):
    client.post("/predict", json=VALID_SAMPLE)  # generate at least one metric
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"predictions_total" in r.data
