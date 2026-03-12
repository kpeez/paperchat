from fastapi.testclient import TestClient

from paperchat.main import create_app
from paperchat.services import local_files as local_files_service


def test_pick_local_documents_returns_selected_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        local_files_service,
        "pick_documents",
        lambda: ("/tmp/first.pdf", "/tmp/second.pdf"),
    )

    with TestClient(create_app()) as client:
        response = client.post("/api/local-files/pick-documents")

    assert response.status_code == 200
    assert response.json() == {"paths": ["/tmp/first.pdf", "/tmp/second.pdf"]}


def test_pick_local_documents_returns_503_when_picker_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        local_files_service,
        "pick_documents",
        lambda: (_ for _ in ()).throw(
            local_files_service.LocalFilePickerUnavailableError("Use manual path entry instead.")
        ),
    )

    with TestClient(create_app()) as client:
        response = client.post("/api/local-files/pick-documents")

    assert response.status_code == 503
    assert response.json() == {"detail": "Use manual path entry instead."}
