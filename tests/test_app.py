import os
import sys
import json
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


def test_index_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"ScamShield AI" in resp.data


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_predict_endpoint_valid_message(client):
    resp = client.post(
        "/api/predict",
        data=json.dumps({"message": "Aapka OTP turant share kijiye warna account block ho jayega."}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["prediction"] in ("SAFE", "SUSPICIOUS", "SCAM")
    assert "indicators" in data
    assert "recommendation" in data


def test_predict_endpoint_missing_message(client):
    resp = client.post("/api/predict", data=json.dumps({}), content_type="application/json")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_endpoint_empty_message(client):
    resp = client.post("/api/predict", data=json.dumps({"message": "   "}), content_type="application/json")
    assert resp.status_code == 400


def test_predict_endpoint_too_long_message(client):
    resp = client.post(
        "/api/predict",
        data=json.dumps({"message": "a" * 5000}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_predict_endpoint_no_body(client):
    resp = client.post("/api/predict")
    assert resp.status_code == 400


def test_analytics_endpoint(client):
    resp = client.get("/api/analytics")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "best_model" in data
    assert "accuracy" in data


def test_examples_endpoint(client):
    resp = client.get("/api/examples")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["examples"]) > 0


def test_404_handler(client):
    resp = client.get("/does-not-exist")
    assert resp.status_code == 404
    assert "error" in resp.get_json()
