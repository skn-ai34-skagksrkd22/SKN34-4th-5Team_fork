import type { ChatMessage } from "./types";
import type { ChatMessageDto } from "./wire";

export function commitChatLoad(signal: AbortSignal, isCurrent: () => boolean, commit: () => void): boolean {
  if (signal.aborted || !isCurrent()) return false;
  commit();
  return true;
}

// The server returns every stored item in canonical turn order (project_history); preserve that
// order as-is, do not re-sort by id. A turn with no answer (pending/failed/stopped before any
// reply) has only the user row, which then carries that turn's tools; once answered, tools move
// to the assistant row and the user row's tools are empty.
export function restoreChatMessages(history: ChatMessageDto[]): ChatMessage[] {
  return history.map(item => ({
    id: item.id, role: item.role, content: item.content, status: item.status,
    ...(item.tools.length ? { tools: item.tools.map(tool => ({ id: tool.id, toolName: tool.tool_name, status: tool.status })) } : {}),
  }));
}
