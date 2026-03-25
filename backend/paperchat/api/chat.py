from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from paperchat.models.chat import (
    AssistantBlockResponse,
    AssistantPayloadResponse,
    ChatMessageResponse,
    ChatRequest,
    ChatTurnResponse,
    CitationResponse,
    ConversationListResponse,
    ConversationResponse,
    ConversationSummaryResponse,
)
from paperchat.services.chat import (
    AssistantBlockRecord,
    ChatMessageRecord,
    ChatServiceProtocol,
    ChatTurnRecord,
    CitationRecord,
    ConversationRecord,
    ConversationSummaryRecord,
)

router = APIRouter(prefix="/api/chat")


def _get_chat_service(request: Request) -> ChatServiceProtocol:
    service = getattr(request.app.state, "chat_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not configured.",
        )
    return service


def _to_citation_response(citation: CitationRecord) -> CitationResponse:
    return CitationResponse(
        citation_id=citation.citation_id,
        chunk_id=citation.chunk_id,
        document_id=citation.document_id,
        document_name=citation.document_name,
        page_numbers=list(citation.page_numbers),
        headings=list(citation.headings),
        snippet=citation.snippet,
    )


def _to_block_response(block: AssistantBlockRecord) -> AssistantBlockResponse:
    return AssistantBlockResponse(
        text=block.text,
        citation_ids=list(block.citation_ids),
    )


def _to_message_response(message: ChatMessageRecord) -> ChatMessageResponse:
    payload = None
    if message.blocks or message.citations:
        payload = AssistantPayloadResponse(
            blocks=[_to_block_response(block) for block in message.blocks],
            citations=[_to_citation_response(citation) for citation in message.citations],
        )
    return ChatMessageResponse(
        id=message.id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        citations=payload,
    )


def _to_conversation_summary_response(
    conversation: ConversationSummaryRecord,
) -> ConversationSummaryResponse:
    return ConversationSummaryResponse(
        id=conversation.id,
        title=conversation.title,
        updated_at=conversation.updated_at,
        message_count=conversation.message_count,
    )


def _to_conversation_response(conversation: ConversationRecord) -> ConversationResponse:
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[_to_message_response(message) for message in conversation.messages],
    )


def _to_chat_turn_response(result: ChatTurnRecord) -> ChatTurnResponse:
    return ChatTurnResponse(
        conversation=_to_conversation_summary_response(result.conversation),
        user_message=_to_message_response(result.user_message),
        assistant_message=_to_message_response(result.assistant_message),
        source_document_ids=list(result.source_document_ids),
    )


@router.get("/conversations", response_model=ConversationListResponse)
def list_conversations(request: Request) -> ConversationListResponse:
    service = _get_chat_service(request)
    conversations = [
        _to_conversation_summary_response(conversation)
        for conversation in service.list_conversations()
    ]
    return ConversationListResponse(conversations=conversations)


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
def get_conversation(conversation_id: str, request: Request) -> ConversationResponse:
    service = _get_chat_service(request)
    conversation = service.get_conversation(conversation_id=conversation_id)
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return _to_conversation_response(conversation)


@router.post("/messages", response_model=ChatTurnResponse)
def send_message(payload: ChatRequest, request: Request) -> ChatTurnResponse:
    service = _get_chat_service(request)
    try:
        result = service.send_message(
            prompt=payload.prompt,
            conversation_id=payload.conversation_id,
            document_ids=tuple(payload.document_ids or ()),
        )
    except LookupError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    return _to_chat_turn_response(result)
