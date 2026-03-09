from __future__ import annotations

from collections import deque
from pathlib import Path
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from paperchat.main import create_app
from paperchat.models.documents import DocumentStatus, IngestionJobStatus
from paperchat.services.documents import DocumentActionResult, DocumentRecord, IngestionJobRecord


def build_action_result(*, document_status: str, job_enqueued: bool) -> DocumentActionResult:
    document_id = uuid4()
    latest_job = (
        IngestionJobRecord(
            id=uuid4(),
            document_id=document_id,
            attempt_number=1,
            status=IngestionJobStatus.queued,
            stage="queued",
            error_code=None,
            error_message=None,
        )
        if job_enqueued
        else None
    )
    return DocumentActionResult(
        document=DocumentRecord(
            id=document_id,
            content_hash="abc123",
            original_filename="paper.pdf",
            display_name="paper.pdf",
            file_path="/tmp/paper.pdf",
            status=DocumentStatus(document_status),
            chunk_count=0,
            parser_id=None,
            chunker_id=None,
            embedding_model_id=None,
            error_code=None,
            error_message=None,
            latest_job=latest_job,
        ),
        job_enqueued=job_enqueued,
        job=latest_job,
    )


class FakeDocumentService:
    def __init__(self) -> None:
        self.import_results = deque(
            [
                build_action_result(document_status="pending", job_enqueued=True),
                build_action_result(document_status="ready", job_enqueued=False),
            ]
        )
        self.retry_result = build_action_result(document_status="pending", job_enqueued=True)
        self.documents = {
            result.document.id: result.document
            for result in [*self.import_results, self.retry_result]
        }
        self.import_paths: list[Path] = []
        self.retry_ids: list[UUID] = []
        self.delete_ids: list[UUID] = []
        self.recovery_calls = 0

    def import_document(self, *, file_path: Path) -> DocumentActionResult:
        self.import_paths.append(file_path)
        result = self.import_results.popleft()
        self.documents[result.document.id] = result.document
        return result

    def list_documents(self) -> tuple[DocumentRecord, ...]:
        return tuple(self.documents.values())

    def get_document(self, *, document_id: UUID) -> DocumentRecord | None:
        return self.documents.get(document_id)

    def retry_document(self, *, document_id: UUID) -> DocumentActionResult | None:
        self.retry_ids.append(document_id)
        return self.retry_result if document_id in self.documents else None

    def delete_document(self, *, document_id: UUID) -> bool:
        self.delete_ids.append(document_id)
        return self.documents.pop(document_id, None) is not None

    def recover_interrupted_jobs(self) -> int:
        self.recovery_calls += 1
        return 1


def test_document_routes_cover_import_duplicate_retry_delete_and_lifespan_recovery() -> None:
    service = FakeDocumentService()
    app = create_app(document_service=service)

    with TestClient(app) as client:
        response = client.post("/api/documents/import", json={"file_path": "/tmp/paper.pdf"})
        assert response.status_code == 200
        data = response.json()
        assert data["job_enqueued"] is True
        assert data["document"]["status"] == "pending"

        duplicate_response = client.post(
            "/api/documents/import",
            json={"file_path": "/tmp/duplicate-paper.pdf"},
        )
        assert duplicate_response.status_code == 200
        duplicate_data = duplicate_response.json()
        assert duplicate_data["job_enqueued"] is False
        assert duplicate_data["document"]["status"] == "ready"

        list_response = client.get("/api/documents")
        assert list_response.status_code == 200
        assert len(list_response.json()["documents"]) == 3

        detail_response = client.get(f"/api/documents/{data['document']['id']}")
        assert detail_response.status_code == 200
        assert detail_response.json()["id"] == data["document"]["id"]

        retry_response = client.post(f"/api/documents/{data['document']['id']}/retry")
        assert retry_response.status_code == 200
        retry_data = retry_response.json()
        assert retry_data["job_enqueued"] is True
        assert retry_data["document"]["id"] == str(service.retry_result.document.id)

        delete_response = client.delete(f"/api/documents/{data['document']['id']}")
        assert delete_response.status_code == 204

    assert service.import_paths == [Path("/tmp/paper.pdf"), Path("/tmp/duplicate-paper.pdf")]
    assert service.retry_ids == [UUID(data["document"]["id"])]
    assert service.delete_ids == [UUID(data["document"]["id"])]
    assert service.recovery_calls == 1


def test_get_document_returns_404_for_unknown_document() -> None:
    app = create_app(document_service=FakeDocumentService())

    with TestClient(app) as client:
        response = client.get(f"/api/documents/{uuid4()}")

    assert response.status_code == 404


def test_retry_document_returns_404_for_unknown_document() -> None:
    app = create_app(document_service=FakeDocumentService())

    with TestClient(app) as client:
        response = client.post(f"/api/documents/{uuid4()}/retry")

    assert response.status_code == 404
