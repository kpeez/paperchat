from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class ChatRole(StrEnum):
    user = "user"
    assistant = "assistant"


class CitationResponse(BaseModel):
    citation_id: str
    chunk_id: str
    document_id: str
    document_name: str
    page_numbers: list[int]
    headings: list[str]
    snippet: str


class AssistantBlockResponse(BaseModel):
    text: str
    citation_ids: list[str]


class AssistantPayloadResponse(BaseModel):
    blocks: list[AssistantBlockResponse]
    citations: list[CitationResponse]


class ChatMessageResponse(BaseModel):
    id: str
    role: ChatRole
    content: str
    created_at: datetime
    citations: AssistantPayloadResponse | None = None


class ConversationSummaryResponse(BaseModel):
    id: str
    title: str | None = None
    updated_at: datetime
    message_count: int


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummaryResponse]


class ConversationResponse(BaseModel):
    id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessageResponse]


class ChatRequest(BaseModel):
    prompt: str
    conversation_id: str | None = None
    document_ids: list[str] | None = None


class ChatTurnResponse(BaseModel):
    conversation: ConversationSummaryResponse
    user_message: ChatMessageResponse
    assistant_message: ChatMessageResponse
    source_document_ids: list[str]
