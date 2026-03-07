from fastapi.testclient import TestClient
from paperchat_backend.main import create_app

client = TestClient(create_app())


def test_bootstrap_returns_200():
    response = client.get("/api/bootstrap")
    assert response.status_code == 200


def test_bootstrap_has_valid_structure():
    data = client.get("/api/bootstrap").json()
    assert "app_version" in data
    assert "status" in data
    assert "checks" in data
    assert "errors" in data


def test_backend_check_is_ok():
    data = client.get("/api/bootstrap").json()
    assert data["checks"]["backend"]["ok"] is True


def test_status_is_ready():
    data = client.get("/api/bootstrap").json()
    assert data["status"] == "ready"


def test_all_expected_checks_present():
    data = client.get("/api/bootstrap").json()
    expected = {"backend", "database", "docker", "migrations"}
    assert set(data["checks"].keys()) == expected
