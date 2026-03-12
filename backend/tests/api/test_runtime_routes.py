from fastapi.testclient import TestClient

from paperchat.main import create_app
from paperchat.models.runtime import RuntimeResponse
from paperchat.services import runtime as runtime_service


def test_runtime_route_returns_runtime_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_service,
        "build_runtime_response",
        lambda: RuntimeResponse(
            app_version="0.1.0",
            data_dir="/tmp/.paperchat",
            database_path="/tmp/.paperchat/paperchat.db",
            cache_dir="/tmp/.paperchat/cache",
            embedding_model="hf:test-model",
        ),
    )

    with TestClient(create_app()) as client:
        response = client.get("/api/runtime")

    assert response.status_code == 200
    assert response.json() == {
        "app_version": "0.1.0",
        "data_dir": "/tmp/.paperchat",
        "database_path": "/tmp/.paperchat/paperchat.db",
        "cache_dir": "/tmp/.paperchat/cache",
        "embedding_model": "hf:test-model",
    }
