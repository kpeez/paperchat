from __future__ import annotations

from typing import Sequence

import pytest

from paperchat.repositories.documents import DocumentChunkRepository, DocumentRepository, NewChunk
from paperchat.services.chat import ChatService


class FakeChatEmbedder:
    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        lowered = text.lower()
        if "methods" in lowered:
            return (0.0, 1.0)
        return (1.0, 0.0)


def create_ready_document(
    db_session_factory,
    *,
    content_hash: str,
    display_name: str,
    embedding: tuple[float, ...],
    retrieval_text: str,
) -> str:
    with db_session_factory.begin() as session:
        document = DocumentRepository(session).create(
            content_hash=content_hash,
            original_filename=display_name,
            display_name=display_name,
            file_path=f"/tmp/{display_name}",
            status="ready",
        )
        DocumentChunkRepository(session).replace_for_document(
            document.id,
            (
                NewChunk(
                    chunk_index=0,
                    text=retrieval_text,
                    retrieval_text=retrieval_text,
                    page_numbers=(1,),
                    headings=("Overview",),
                    warning_codes=(),
                    embedding=embedding,
                ),
            ),
        )
        return document.id


def test_chat_service_creates_conversation_and_persists_citations(db_session_factory) -> None:
    primary_document_id = create_ready_document(
        db_session_factory,
        content_hash="a" * 64,
        display_name="attention.pdf",
        embedding=(1.0, 0.0),
        retrieval_text="Transformers use attention to connect distant tokens.",
    )
    secondary_document_id = create_ready_document(
        db_session_factory,
        content_hash="b" * 64,
        display_name="bert.pdf",
        embedding=(0.0, 1.0),
        retrieval_text="BERT pretrains deep bidirectional language representations.",
    )

    service = ChatService(
        session_factory=db_session_factory,
        embedder=FakeChatEmbedder(),
    )

    result = service.send_message(prompt="How does attention work?")

    assert result.conversation.title == "How does attention work?"
    assert set(result.source_document_ids) == {primary_document_id, secondary_document_id}
    assert result.user_message.role == "user"
    assert result.assistant_message.role == "assistant"
    assert len(result.assistant_message.blocks) == 3
    assert len(result.assistant_message.citations) == 2
    assert result.assistant_message.citations[0].document_id == primary_document_id

    conversation = service.get_conversation(conversation_id=result.conversation.id)

    assert conversation is not None
    assert len(conversation.messages) == 2
    assert conversation.messages[1].citations[0].document_id == primary_document_id
    assert service.list_conversations()[0].message_count == 2


def test_chat_service_appends_to_existing_conversation_and_respects_source_filter(
    db_session_factory,
) -> None:
    first_document_id = create_ready_document(
        db_session_factory,
        content_hash="c" * 64,
        display_name="intro.pdf",
        embedding=(1.0, 0.0),
        retrieval_text="The introduction explains the high-level goal.",
    )
    second_document_id = create_ready_document(
        db_session_factory,
        content_hash="d" * 64,
        display_name="methods.pdf",
        embedding=(0.0, 1.0),
        retrieval_text="The methods section details the training procedure.",
    )

    service = ChatService(
        session_factory=db_session_factory,
        embedder=FakeChatEmbedder(),
    )

    first_turn = service.send_message(
        prompt="Give me the intro summary.",
        document_ids=(first_document_id,),
    )
    second_turn = service.send_message(
        prompt="What are the methods?",
        conversation_id=first_turn.conversation.id,
        document_ids=(second_document_id,),
    )

    assert second_turn.conversation.id == first_turn.conversation.id
    assert second_turn.source_document_ids == (second_document_id,)
    assert second_turn.assistant_message.citations[0].document_id == second_document_id

    conversation = service.get_conversation(conversation_id=first_turn.conversation.id)

    assert conversation is not None
    assert len(conversation.messages) == 4


def test_chat_service_rejects_missing_ready_documents(db_session_factory) -> None:
    service = ChatService(
        session_factory=db_session_factory,
        embedder=FakeChatEmbedder(),
    )

    with pytest.raises(ValueError, match="At least one ready document is required"):
        service.send_message(prompt="Hello?")
