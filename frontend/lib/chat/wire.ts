// Hand-typed wire DTOs for /api/v2/chat/ (backend/llm/serializer/message.py, views/sse.py).
// Message ids are server-issued positive integers kept from the v1 wire (serializer wire_history); sequence_no is
// the 1-based position in the returned list. done.message_id is the same id as a digit string.
import type { ChatContext } from "./types";

export type ChatMessageStatus = "pending" | "completed" | "failed" | "stopped";
export type ChatToolStatus = "running" | "completed" | "failed";

export type ChatSessionDto = { id: string; title: string; created_at: string; updated_at: string };
export type ChatToolCallDto = { id: string; tool_name: string; status: ChatToolStatus };
export type ChatMessageDto = {
  id: number;
  sequence_no: number;
  role: "user" | "assistant";
  content: string;
  status: ChatMessageStatus;
  tools: ChatToolCallDto[];
  created_at: string;
  updated_at: string;
};

export type ChatMessageRequestDto = { content: string; context?: ChatContext };
export type ChatMessageUpdateRequestDto = ChatMessageRequestDto & { message_id: number };
export type ChatMessageDeleteRequestDto = { message_id: number };

// serializer/message.py project_event()/done_payload(): delta{text} / tool{id, tool_name, status} /
// done{message_id, assistant_message, tools} / error{detail}.
export type ChatSseEvent =
  | { event: "delta"; data: { text: string } }
  | { event: "tool"; data: ChatToolCallDto }
  | { event: "done"; data: { message_id: string; assistant_message: string; tools: ChatToolCallDto[] } }
  | { event: "error"; data: { detail: string } };
