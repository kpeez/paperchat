from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from fastapi.testclient import TestClient

from paperchat.main import create_app
from paperchat.models.chat import ChatRole
from paperchat.services.chat import (
    AssistantBlockRecord,
    ChatMessageRecord,
    ChatTurnRecord,
    CitationRecord,
    ConversationRecord,
    ConversationSummaryRecord,
)


def make_conversation_summary() -> ConversationSummaryRecord:
    return ConversationSummaryRecord(
        id=str(uuid4()),
        title="Attention summary",
        updated_at=datetime.now(UTC),
        message_count=2,
    )


def make_user_message() -> ChatMessageRecord:
    return ChatMessageRecord(
        id=str(uuid4()),
        role=ChatRole.user,
        content="How does attention work?",
        created_at=datetime.now(UTC),
    )


def make_assistant_message(*, document_id: str) -> ChatMessageRecord:
    return ChatMessageRecord(
        id=str(uuid4()),
        role=ChatRole.assistant,
        content="Grounded notes for attention.",
        created_at=datetime.now(UTC),
        blocks=(
            AssistantBlockRecord(
                text="Grounded notes for attention.",
                citation_ids=("c1",),
            ),
        ),
        citations=(
            CitationRecord(
                citation_id="c1",
                chunk_id=str(uuid4()),
                document_id=document_id,
                document_name="attention.pdf",
                page_numbers=(1,),
                headings=("Overview",),
                snippet="Transformers use attention to connect distant tokens.",
            ),
        ),
    )


class FakeDocumentService:
    def recover_interrupted_jobs(self) -> int:
        return 0


class FakeChatService:
    def __init__(self) -> None:
        summary = make_conversation_summary()
        user_message = make_user_message()
        assistant_message = make_assistant_message(document_id=str(uuid4()))
        self.conversation = ConversationRecord(
            id=summary.id,
            title=summary.title,
            created_at=summary.updated_at,
            updated_at=summary.updated_at,
            messages=(user_message, assistant_message),
        )
        self.summaries = deque([summary])
        self.turn = ChatTurnRecord(
            conversation=summary,
            user_message=user_message,
            assistant_message=assistant_message,
            source_document_ids=(assistant_message.citations[0].document_id,),
        )

    def list_conversations(self) -> tuple[ConversationSummaryRecord, ...]:
        return tuple(self.summaries)

    def get_conversation(self, *, conversation_id: str) -> ConversationRecord | None:
        if conversation_id == self.conversation.id:
            return self.conversation
        return None

    def send_message(
        self,
        *,
        prompt: str,
        conversation_id: str | None = None,
        document_ids: tuple[str, ...] = (),
    ) -> ChatTurnRecord:
        del conversation_id, document_ids
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        return self.turn


def test_chat_routes_cover_list_detail_and_send() -> None:
    app = create_app(
        document_service=cast(Any, FakeDocumentService()),
        chat_service=cast(Any, FakeChatService()),
    )

    with TestClient(app) as client:
        list_response = client.get("/api/chat/conversations")
        assert list_response.status_code == 200
        assert len(list_response.json()["conversations"]) == 1

        conversation_id = list_response.json()["conversations"][0]["id"]
        detail_response = client.get(f"/api/chat/conversations/{conversation_id}")
        assert detail_response.status_code == 200
        assert len(detail_response.json()["messages"]) == 2

        send_response = client.post(
            "/api/chat/messages",
            json={"prompt": "How does attention work?"},
        )
        assert send_response.status_code == 200
        body = send_response.json()
        assert (
            body["assistant_message"]["citations"]["citations"][0]["document_name"]
            == "attention.pdf"
        )


def test_get_conversation_returns_404_for_unknown_conversation() -> None:
    app = create_app(
        document_service=cast(Any, FakeDocumentService()),
        chat_service=cast(Any, FakeChatService()),
    )

    with TestClient(app) as client:
        response = client.get(f"/api/chat/conversations/{uuid4()}")

    assert response.status_code == 404


def test_send_message_returns_400_for_empty_prompt() -> None:
    app = create_app(
        document_service=cast(Any, FakeDocumentService()),
        chat_service=cast(Any, FakeChatService()),
    )

    with TestClient(app) as client:
        response = client.post("/api/chat/messages", json={"prompt": "   "})

    assert response.status_code == 400
