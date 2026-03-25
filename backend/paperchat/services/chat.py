from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from paperchat.db.schema import Conversation, Document, Message
from paperchat.models.chat import ChatRole
from paperchat.repositories.documents import RetrievedChunk, VectorChunkRepository
from paperchat.services.embeddings import EmbeddingGemmaEmbeddingService, EmbeddingService

DEFAULT_CHAT_RESULT_LIMIT = 3
MAX_CONVERSATION_TITLE_LENGTH = 60
MAX_SNIPPET_LENGTH = 220


@dataclass(frozen=True, slots=True)
class CitationRecord:
    citation_id: str
    chunk_id: str
    document_id: str
    document_name: str
    page_numbers: tuple[int, ...]
    headings: tuple[str, ...]
    snippet: str


@dataclass(frozen=True, slots=True)
class AssistantBlockRecord:
    text: str
    citation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChatMessageRecord:
    id: str
    role: ChatRole
    content: str
    created_at: datetime
    blocks: tuple[AssistantBlockRecord, ...] = ()
    citations: tuple[CitationRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class ConversationSummaryRecord:
    id: str
    title: str | None
    updated_at: datetime
    message_count: int


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    messages: tuple[ChatMessageRecord, ...]


@dataclass(frozen=True, slots=True)
class ChatTurnRecord:
    conversation: ConversationSummaryRecord
    user_message: ChatMessageRecord
    assistant_message: ChatMessageRecord
    source_document_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnswerDraft:
    content: str
    blocks: tuple[AssistantBlockRecord, ...]
    citations: tuple[CitationRecord, ...]


class AnswerServiceProtocol(Protocol):
    def generate_answer(
        self,
        *,
        prompt: str,
        chunks: tuple[RetrievedChunk, ...],
    ) -> AnswerDraft: ...


class ChatServiceProtocol(Protocol):
    def list_conversations(self) -> tuple[ConversationSummaryRecord, ...]: ...

    def get_conversation(self, *, conversation_id: str) -> ConversationRecord | None: ...

    def send_message(
        self,
        *,
        prompt: str,
        conversation_id: str | None = None,
        document_ids: tuple[str, ...] = (),
    ) -> ChatTurnRecord: ...


class DeterministicAnswerService:
    """Small grounded answer formatter used until a real provider is added."""

    def generate_answer(
        self,
        *,
        prompt: str,
        chunks: tuple[RetrievedChunk, ...],
    ) -> AnswerDraft:
        if not chunks:
            block = AssistantBlockRecord(
                text=(
                    f'I could not find grounded support for "{prompt}" in the selected documents.'
                ),
                citation_ids=(),
            )
            return AnswerDraft(
                content=block.text,
                blocks=(block,),
                citations=(),
            )

        blocks = [
            AssistantBlockRecord(
                text=f'Grounded notes for "{prompt}":',
                citation_ids=(),
            )
        ]
        citations: list[CitationRecord] = []
        for index, chunk in enumerate(chunks, start=1):
            citation_id = f"c{index}"
            snippet = _snippet(chunk.retrieval_text)
            citations.append(
                CitationRecord(
                    citation_id=citation_id,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    page_numbers=chunk.page_numbers,
                    headings=chunk.headings,
                    snippet=snippet,
                )
            )
            page_label = _page_label(chunk.page_numbers)
            heading_prefix = f"{chunk.headings[0]}: " if chunk.headings else ""
            blocks.append(
                AssistantBlockRecord(
                    text=f"{chunk.document_name} ({page_label}) {heading_prefix}{snippet}",
                    citation_ids=(citation_id,),
                )
            )

        return AnswerDraft(
            content="\n\n".join(block.text for block in blocks),
            blocks=tuple(blocks),
            citations=tuple(citations),
        )


class ChatService:
    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        embedder: EmbeddingService | None = None,
        answer_service: AnswerServiceProtocol | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._embedder = embedder or EmbeddingGemmaEmbeddingService()
        self._answer_service = answer_service or DeterministicAnswerService()
        self._ensure_search_index()

    def list_conversations(self) -> tuple[ConversationSummaryRecord, ...]:
        with self._session_factory() as session:
            statement = (
                select(Conversation, func.count(Message.id))
                .outerjoin(Message, Message.conversation_id == Conversation.id)
                .group_by(Conversation.id)
                .order_by(Conversation.updated_at.desc())
            )
            rows = session.execute(statement)
            return tuple(
                ConversationSummaryRecord(
                    id=conversation.id,
                    title=conversation.title,
                    updated_at=conversation.updated_at,
                    message_count=message_count,
                )
                for conversation, message_count in rows
            )

    def get_conversation(self, *, conversation_id: str) -> ConversationRecord | None:
        with self._session_factory() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                return None
            statement = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.message_index.asc())
            )
            messages = tuple(session.scalars(statement))
            return ConversationRecord(
                id=conversation.id,
                title=conversation.title,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
                messages=tuple(_to_message_record(message) for message in messages),
            )

    def send_message(
        self,
        *,
        prompt: str,
        conversation_id: str | None = None,
        document_ids: tuple[str, ...] = (),
    ) -> ChatTurnRecord:
        trimmed_prompt = prompt.strip()
        if not trimmed_prompt:
            raise ValueError("Prompt cannot be empty.")

        source_documents = self._resolve_source_documents(document_ids=document_ids)
        self._ensure_search_index()
        retrieved_chunks = self._retrieve_chunks(
            prompt=trimmed_prompt,
            document_ids=tuple(document.id for document in source_documents),
        )
        answer = self._answer_service.generate_answer(
            prompt=trimmed_prompt, chunks=retrieved_chunks
        )
        _validate_answer(answer=answer, retrieved_chunks=retrieved_chunks)

        stored_turn = self._store_turn(
            prompt=trimmed_prompt,
            conversation_id=conversation_id,
            answer=answer,
        )
        return ChatTurnRecord(
            conversation=stored_turn.conversation,
            user_message=stored_turn.user_message,
            assistant_message=stored_turn.assistant_message,
            source_document_ids=tuple(document.id for document in source_documents),
        )

    def _resolve_source_documents(self, *, document_ids: tuple[str, ...]) -> tuple[Document, ...]:
        with self._session_factory() as session:
            statement = select(Document).where(Document.status == "ready")
            if document_ids:
                statement = statement.where(Document.id.in_(document_ids))
            documents = tuple(session.scalars(statement.order_by(Document.created_at.desc())))
            if not documents:
                if document_ids:
                    msg = "Selected documents must exist and be ready before chat can use them."
                    raise ValueError(msg)
                msg = "At least one ready document is required before chat can run."
                raise ValueError(msg)
            if document_ids and len(documents) != len(set(document_ids)):
                msg = "Selected documents must exist and be ready before chat can use them."
                raise ValueError(msg)
            return documents

    def _retrieve_chunks(
        self,
        *,
        prompt: str,
        document_ids: tuple[str, ...],
    ) -> tuple[RetrievedChunk, ...]:
        query_embedding = self._embedder.embed_query(prompt)
        with self._session_factory() as session:
            repository = VectorChunkRepository(session)
            repository.ensure_search_index_synced()
            return repository.search(
                query_embedding=query_embedding,
                document_ids=document_ids,
                limit=DEFAULT_CHAT_RESULT_LIMIT,
            )

    def _store_turn(
        self,
        *,
        prompt: str,
        conversation_id: str | None,
        answer: AnswerDraft,
    ) -> StoredTurn:
        with self._session_factory.begin() as session:
            conversation = _get_or_create_conversation(
                session=session,
                conversation_id=conversation_id,
                first_prompt=prompt,
            )
            user_message = Message(
                conversation_id=conversation.id,
                message_index=_next_message_index(session, conversation.id),
                role=ChatRole.user,
                content=prompt,
                citations=None,
            )
            session.add(user_message)
            session.flush()

            assistant_message = Message(
                conversation_id=conversation.id,
                message_index=user_message.message_index + 1,
                role=ChatRole.assistant,
                content=answer.content,
                citations={
                    "blocks": [_block_to_dict(block) for block in answer.blocks],
                    "citations": [_citation_to_dict(citation) for citation in answer.citations],
                },
            )
            conversation.updated_at = _utcnow()
            session.add(assistant_message)
            session.flush()
            message_count = session.scalar(
                select(func.count(Message.id)).where(Message.conversation_id == conversation.id)
            )
            summary = ConversationSummaryRecord(
                id=conversation.id,
                title=conversation.title,
                updated_at=conversation.updated_at,
                message_count=message_count or 0,
            )
            return StoredTurn(
                conversation=summary,
                user_message=_to_message_record(user_message),
                assistant_message=_to_message_record(assistant_message),
            )

    def _ensure_search_index(self) -> None:
        with self._session_factory.begin() as session:
            VectorChunkRepository(session).ensure_search_index_synced()


@dataclass(frozen=True, slots=True)
class StoredTurn:
    conversation: ConversationSummaryRecord
    user_message: ChatMessageRecord
    assistant_message: ChatMessageRecord


def _get_or_create_conversation(
    *,
    session: Session,
    conversation_id: str | None,
    first_prompt: str,
) -> Conversation:
    if conversation_id is not None:
        conversation = session.get(Conversation, conversation_id)
        if conversation is None:
            msg = f"Conversation {conversation_id} was not found."
            raise LookupError(msg)
        return conversation

    conversation = Conversation(title=_conversation_title(first_prompt))
    session.add(conversation)
    session.flush()
    return conversation


def _next_message_index(session: Session, conversation_id: str) -> int:
    current = session.scalar(
        select(func.max(Message.message_index)).where(Message.conversation_id == conversation_id)
    )
    return (current or -1) + 1


def _conversation_title(prompt: str) -> str:
    trimmed = " ".join(prompt.strip().split())
    if len(trimmed) <= MAX_CONVERSATION_TITLE_LENGTH:
        return trimmed
    return trimmed[: MAX_CONVERSATION_TITLE_LENGTH - 1].rstrip() + "…"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _snippet(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= MAX_SNIPPET_LENGTH:
        return collapsed
    return collapsed[: MAX_SNIPPET_LENGTH - 3].rstrip() + "..."


def _page_label(page_numbers: tuple[int, ...]) -> str:
    if not page_numbers:
        return "pages unknown"
    if len(page_numbers) == 1:
        return f"page {page_numbers[0]}"
    return "pages " + ", ".join(str(page_number) for page_number in page_numbers)


def _block_to_dict(block: AssistantBlockRecord) -> dict[str, object]:
    return {
        "text": block.text,
        "citation_ids": list(block.citation_ids),
    }


def _citation_to_dict(citation: CitationRecord) -> dict[str, object]:
    return {
        "citation_id": citation.citation_id,
        "chunk_id": citation.chunk_id,
        "document_id": citation.document_id,
        "document_name": citation.document_name,
        "page_numbers": list(citation.page_numbers),
        "headings": list(citation.headings),
        "snippet": citation.snippet,
    }


def _to_message_record(message: Message) -> ChatMessageRecord:
    blocks: tuple[AssistantBlockRecord, ...] = ()
    citations: tuple[CitationRecord, ...] = ()
    if message.citations:
        raw_blocks = message.citations.get("blocks", [])
        raw_citations = message.citations.get("citations", [])
        blocks = tuple(
            AssistantBlockRecord(
                text=str(block["text"]),
                citation_ids=tuple(str(citation_id) for citation_id in block["citation_ids"]),
            )
            for block in raw_blocks
        )
        citations = tuple(
            CitationRecord(
                citation_id=str(citation["citation_id"]),
                chunk_id=str(citation["chunk_id"]),
                document_id=str(citation["document_id"]),
                document_name=str(citation["document_name"]),
                page_numbers=tuple(int(page_number) for page_number in citation["page_numbers"]),
                headings=tuple(str(heading) for heading in citation["headings"]),
                snippet=str(citation["snippet"]),
            )
            for citation in raw_citations
        )
    return ChatMessageRecord(
        id=message.id,
        role=ChatRole(message.role),
        content=message.content,
        created_at=message.created_at,
        blocks=blocks,
        citations=citations,
    )


def _validate_answer(
    *,
    answer: AnswerDraft,
    retrieved_chunks: tuple[RetrievedChunk, ...],
) -> None:
    retrieved_by_chunk_id = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
    citations_by_id = {citation.citation_id: citation for citation in answer.citations}

    for block in answer.blocks:
        for citation_id in block.citation_ids:
            if citation_id not in citations_by_id:
                msg = f"Assistant response referenced unknown citation id `{citation_id}`."
                raise ValueError(msg)

    for citation in answer.citations:
        chunk = retrieved_by_chunk_id.get(citation.chunk_id)
        if chunk is None:
            msg = f"Citation `{citation.citation_id}` did not map to a retrieved chunk."
            raise ValueError(msg)
        if citation.document_id != chunk.document_id:
            msg = f"Citation `{citation.citation_id}` did not match the retrieved document."
            raise ValueError(msg)
        if citation.page_numbers != chunk.page_numbers:
            msg = f"Citation `{citation.citation_id}` did not match the retrieved page numbers."
            raise ValueError(msg)
        if citation.headings != chunk.headings:
            msg = f"Citation `{citation.citation_id}` did not match the retrieved headings."
            raise ValueError(msg)
