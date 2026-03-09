from fastapi.testclient import TestClient

from paperchat.main import create_app

client = TestClient(create_app())


def test_health_returns_ok():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
