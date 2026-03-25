import { fetchJson } from "./client";

export type ChatRole = "user" | "assistant";

export interface CitationResponse {
  citation_id: string;
  chunk_id: string;
  document_id: string;
  document_name: string;
  page_numbers: number[];
  headings: string[];
  snippet: string;
}

export interface AssistantBlockResponse {
  text: string;
  citation_ids: string[];
}

export interface AssistantPayloadResponse {
  blocks: AssistantBlockResponse[];
  citations: CitationResponse[];
}

export interface ChatMessageResponse {
  id: string;
  role: ChatRole;
  content: string;
  created_at: string;
  citations: AssistantPayloadResponse | null;
}

export interface ConversationSummaryResponse {
  id: string;
  title: string | null;
  updated_at: string;
  message_count: number;
}

export interface ConversationListResponse {
  conversations: ConversationSummaryResponse[];
}

export interface ConversationResponse {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
  messages: ChatMessageResponse[];
}

export interface ChatTurnResponse {
  conversation: ConversationSummaryResponse;
  user_message: ChatMessageResponse;
  assistant_message: ChatMessageResponse;
  source_document_ids: string[];
}

export function listConversations(): Promise<ConversationListResponse> {
  return fetchJson<ConversationListResponse>("/api/chat/conversations");
}

export function getConversation(conversationId: string): Promise<ConversationResponse> {
  return fetchJson<ConversationResponse>(`/api/chat/conversations/${conversationId}`);
}

export function sendMessage({
  prompt,
  conversationId,
  documentIds,
}: {
  prompt: string;
  conversationId?: string | null;
  documentIds?: string[];
}): Promise<ChatTurnResponse> {
  return fetchJson<ChatTurnResponse>("/api/chat/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      prompt,
      conversation_id: conversationId ?? null,
      document_ids: documentIds ?? [],
    }),
  });
}
