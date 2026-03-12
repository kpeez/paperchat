import os
from contextlib import asynccontextmanager
from typing import Any, cast

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from paperchat.api.bootstrap import router
from paperchat.api.documents import router as documents_router
from paperchat.api.local_files import router as local_files_router
from paperchat.api.runtime import router as runtime_router
from paperchat.db.engine import get_session_factory
from paperchat.services.docling_ingestion import DoclingDocumentParser
from paperchat.services.documents import (
    DocumentLifecycleBackend,
    DocumentLifecycleService,
    DocumentServiceProtocol,
)
from paperchat.services.embeddings import EmbeddingGemmaEmbeddingService
from paperchat.services.ingestion import IngestionCoordinator, IngestionProcessor


def create_app(
    *,
    document_service: DocumentServiceProtocol | None = None,
    start_worker: bool = True,
) -> FastAPI:
    service_was_injected = document_service is not None
    coordinator: IngestionCoordinator | None = None
    if document_service is None:
        document_service, coordinator = _build_default_document_service()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active_document_service = document_service
        active_coordinator = coordinator
        if active_document_service is not None:
            try:
                active_document_service.recover_interrupted_jobs()
            except Exception:
                if service_was_injected:
                    raise
                active_document_service = None
                active_coordinator = None
        app.state.document_service = active_document_service
        app.state.ingestion_coordinator = active_coordinator
        if active_coordinator is not None and start_worker:
            active_coordinator.start()
        try:
            yield
        finally:
            if active_coordinator is not None and start_worker:
                active_coordinator.stop()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        cast(Any, CORSMiddleware),
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(documents_router)
    app.include_router(local_files_router)
    app.include_router(runtime_router)
    return app


def run() -> None:
    port = int(os.environ.get("PAPERCHAT_PORT", "9712"))
    uvicorn.run("paperchat.main:create_app", host="127.0.0.1", port=port, factory=True)


def _build_default_document_service() -> tuple[
    DocumentServiceProtocol | None, IngestionCoordinator | None
]:
    try:
        session_factory = get_session_factory()
        processor = IngestionProcessor(
            session_factory=session_factory,
            parser=DoclingDocumentParser(),
            embedder=EmbeddingGemmaEmbeddingService(),
        )
    except Exception:
        return None, None

    coordinator = IngestionCoordinator(processor=processor)
    return (
        DocumentLifecycleService(
            backend=DocumentLifecycleBackend(session_factory=session_factory),
            coordinator=coordinator,
        ),
        coordinator,
    )
