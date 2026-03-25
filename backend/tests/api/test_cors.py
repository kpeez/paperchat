from fastapi.testclient import TestClient

from paperchat.main import create_app


def test_localhost_dev_origins_are_allowed_for_cors() -> None:
    with TestClient(create_app()) as client:
        response = client.options(
            "/api/runtime",
            headers={
                "Origin": "http://127.0.0.1:5174",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5174"
