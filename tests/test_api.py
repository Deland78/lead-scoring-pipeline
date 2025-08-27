import os
import requests
import time

API_URL = os.environ.get("API_URL", "http://127.0.0.1:5051")


def wait_for_health(timeout: int = 30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{API_URL}/v2/health", timeout=3)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("API health endpoint not responding")


def test_health_ok():
    data = wait_for_health()
    assert "status" in data
    assert "model_loaded" in data
    assert "preprocessor_loaded" in data


def test_predict_happy_path():
    payload = {
        "TotalVisits": 5,
        "Page Views Per Visit": 3.2,
        "Total Time Spent on Website": 1850,
        "Lead Origin": "API",
        "Lead Source": "Google",
        "Last Activity": "Email Opened",
        "What is your current occupation": "Working Professional"
    }
    r = requests.post(f"{API_URL}/v2/predict", json=payload, timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert set(["prediction", "lead_score", "label", "timestamp", "model_version"]) <= set(body.keys())


def test_predict_validation_error():
    bad_payload = {
        # Missing required fields deliberately
        "TotalVisits": -1,
        "Page Views Per Visit": -1,
        "Total Time Spent on Website": -5,
        "Lead Origin": "API",
        "Lead Source": "Google",
        "Last Activity": "Email Opened",
        "What is your current occupation": "Working Professional"
    }
    r = requests.post(f"{API_URL}/v2/predict", json=bad_payload, timeout=10)
    assert r.status_code in (400, 422, 500, 503)