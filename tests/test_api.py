from fastapi.testclient import TestClient
from src.api.app import app


def test_health_check():
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json().get("message") == "AGRO QA Chatbot API is running."
