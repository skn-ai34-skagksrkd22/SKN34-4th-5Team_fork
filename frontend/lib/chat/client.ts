import { memberError, memberFetch } from "../member-auth-request";
import { isRecord, parseChatRequest } from "./validation";
import type { ChatContext, ChatReply, ChatStatus } from "./types";
import { MAX_REPLY_LENGTH } from "./types";
import type { ChatMessageDto, ChatSessionDto, ChatToolCallDto } from "./wire";

// Members and guests share one API. Members authenticate with Bearer (memberFetch); guests send no
// Authorization header and are identified by the server-set HttpOnly guest_id cookie that same-origin
// fetch already includes by default.
export type ChatMode = "member" | "guest";

const SESSIONS = "/api/v2/chat/sessions/";
const REQUEST_TIMEOUT_MS = 55_000;
const MESSAGE_ID = /^[1-9]\d*$/;
const isMessageId = (value: unknown): value is number => Number.isSafeInteger(value) && Number(value) > 0;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const STATUS: Record<ChatMode, ChatStatus> = {
  member: { provider: "backend", model: "팀 챗봇", ready: true },
  guest: { provider: "guest", model: "팀 챗봇", ready: true },
};
export const GUEST_STATUS = STATUS.guest;

export type ChatStreamCallbacks = { onDelta?: (answer: string) => void; onTool?: (tool: ChatToolCallDto) => void };
export type ChatSendRequest = { sessionId?: string; content: string; context?: ChatContext };
export type ChatEditRequest = { sessionId: string; messageId: number; content: string; context?: ChatContext };

export class ChatClientError extends Error {
  constructor(message: string, public status: number, public uncertain = false, public sessionId?: string) { super(message); }
}

function fallback(status: number) {
  if (status === 401) return "팀 계정 로그인을 확인해 주세요.";
  if (status === 403) return "이 대화를 이용할 권한이 없어요.";
  if (status === 404) return "대화를 찾지 못했어요. 새 대화를 시작해 주세요.";
  if (status === 429) return "요청이 많아요. 잠시 후 다시 시도해 주세요.";
  if (status >= 500) return "챗봇 서버에서 요청을 처리하지 못했어요.";
  return "입력 내용을 확인해 주세요.";
}

async function readJson(response: Response): Promise<unknown> {
  const text = await response.text();
  return text ? JSON.parse(text) : null;
}

async function request<T>(mode: ChatMode, path: string, init: RequestInit, signal: AbortSignal | undefined, read: (response: Response) => Promise<T>, timeoutMs: number | null = REQUEST_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController();
  let timedOut = false;
  const abort = () => controller.abort();
  if (signal?.aborted) abort();
  signal?.addEventListener("abort", abort, { once: true });
  // LLM 스트림은 총시간 제한 없이 외부 signal로만 중단한다.
  const timeout = timeoutMs === null ? undefined : window.setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  const mutating = init.method !== "GET";
  const fetcher = mode === "member" ? memberFetch : fetch;
  try {
    const response = await fetcher(path, { ...init, cache: "no-store", signal: controller.signal });
    if (!response.ok) {
      const data = await readJson(response).catch(() => null);
      throw new ChatClientError(
        response.status >= 500 ? fallback(response.status) : memberError(data, fallback(response.status)),
        response.status,
        response.status >= 500 && mutating,
      );
    }
    return await read(response);
  } catch (error) {
    if (error instanceof ChatClientError) throw error;
    if (error instanceof SyntaxError) throw new ChatClientError("응답을 읽지 못했어요.", 502, mutating);
    if (timedOut) throw new ChatClientError("응답 시간이 지났지만 서버 처리 여부는 확인할 수 없어요.", 504, mutating);
    if (controller.signal.aborted) throw new ChatClientError("요청이 중단됐어요.", 499, mutating);
    throw new ChatClientError("연결이 끊겨 서버 처리 여부를 확인할 수 없어요.", 502, mutating);
  } finally {
    window.clearTimeout(timeout);
    signal?.removeEventListener("abort", abort);
  }
}

const json = (method: string, body: unknown): RequestInit => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const sessionPath = (sessionId: string) => {
  if (!UUID.test(sessionId)) throw new ChatClientError("대화 번호를 확인해 주세요.", 400);
  return `${SESSIONS}${sessionId}/`;
};
const isSession = (value: unknown): value is ChatSessionDto => isRecord(value) && typeof value.id === "string" && UUID.test(value.id) && typeof value.title === "string";
const isTool = (value: unknown): value is ChatToolCallDto => isRecord(value) && typeof value.id === "string" && Boolean(value.id) &&
  typeof value.tool_name === "string" && Boolean(value.tool_name) && ["running", "completed", "failed"].includes(String(value.status));
const isMessage = (value: unknown): value is ChatMessageDto => isRecord(value) && isMessageId(value.id) && isMessageId(value.sequence_no) &&
  (value.role === "user" || value.role === "assistant") && typeof value.content === "string" &&
  ["pending", "completed", "failed", "stopped"].includes(String(value.status)) &&
  Array.isArray(value.tools) && value.tools.every(isTool) && typeof value.created_at === "string" && typeof value.updated_at === "string";
// DELETE answers 204 with no body.
const readEmpty = async (response: Response) => { await response.text(); };

// Frames are `event: <name>\ndata: <json>\n\n` (backend/llm/views/sse.py): delta{text}* / tool{id,
// tool_name, status}* then done{message_id, assistant_message, tools} or error{detail}. A stream
// that closes with neither means the turn was superseded or the connection dropped.
async function readStream(response: Response, sessionId: string, callbacks: ChatStreamCallbacks): Promise<{ reply: string; assistantMessageId: number; tools: ChatToolCallDto[] }> {
  if (!response.body || !response.headers.get("Content-Type")?.toLowerCase().startsWith("text/event-stream")) {
    throw new ChatClientError("스트림 응답을 확인하지 못했어요.", 502, true, sessionId);
  }
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = "", answer = "", result: { reply: string; assistantMessageId: number; tools: ChatToolCallDto[] } | null = null;
  const consume = (frame: string) => {
    const [eventLine, ...lines] = frame.split(/\r?\n/);
    const event = eventLine?.startsWith("event:") ? eventLine.slice(6).trim() : "";
    const raw = lines.filter(line => line.startsWith("data:")).map(line => line.slice(5).trimStart()).join("\n");
    let value: unknown;
    try { value = JSON.parse(raw); } catch { throw new ChatClientError("스트림 응답 형식이 올바르지 않아요.", 502, true, sessionId); }
    if (!isRecord(value) || result) throw new ChatClientError("스트림 응답 순서가 올바르지 않아요.", 502, true, sessionId);
    if (event === "delta") {
      if (typeof value.text !== "string" || answer.length + value.text.length > MAX_REPLY_LENGTH) throw new ChatClientError("답변이 너무 길어요.", 502, true, sessionId);
      answer += value.text;
      callbacks.onDelta?.(answer);
      return;
    }
    if (event === "tool") {
      if (!isTool(value)) throw new ChatClientError("도구 호출 정보를 확인하지 못했어요.", 502, true, sessionId);
      callbacks.onTool?.(value);
      return;
    }
    if (event === "done") {
      if (typeof value.message_id !== "string" || !MESSAGE_ID.test(value.message_id) || !isMessageId(Number(value.message_id)) || typeof value.assistant_message !== "string" ||
        !Array.isArray(value.tools) || !value.tools.every(isTool)) {
        throw new ChatClientError("최종 답변을 확인하지 못했어요.", 502, true, sessionId);
      }
      result = { reply: value.assistant_message, assistantMessageId: Number(value.message_id), tools: value.tools };
      return;
    }
    if (event === "error") {
      // The stream failed after the server attempted the turn; the final save may or may not have
      // committed, so the client can't assume persistence failed and must reconcile via history.
      throw new ChatClientError(typeof value.detail === "string" && value.detail && value.detail.length <= 200 ? value.detail : fallback(502), 502, true, sessionId);
    }
    throw new ChatClientError("알 수 없는 스트림 응답을 받았어요.", 502, true, sessionId);
  };
  try {
    while (!result) {
      const next = await reader.read();
      buffer += decoder.decode(next.value, { stream: !next.done });
      let boundary;
      while ((boundary = buffer.search(/\r?\n\r?\n/)) >= 0) {
        const frame = buffer.slice(0, boundary), separator = buffer.slice(boundary).startsWith("\r\n\r\n") ? 4 : 2;
        buffer = buffer.slice(boundary + separator);
        if (frame.trim()) consume(frame);
      }
      if (buffer.length > MAX_REPLY_LENGTH * 4) throw new ChatClientError("스트림 응답이 너무 길어요.", 502, true, sessionId);
      if (next.done) break;
    }
    if (!result || buffer.trim()) throw new ChatClientError("답변이 끝나기 전에 연결이 끊겼어요.", 502, true, sessionId);
    return result;
  } finally { await reader.cancel().catch(() => undefined); }
}

export async function listChatSessions(mode: ChatMode, signal?: AbortSignal): Promise<ChatSessionDto[]> {
  const sessions = await request(mode, SESSIONS, { method: "GET" }, signal, readJson);
  if (!Array.isArray(sessions) || !sessions.every(isSession)) throw new ChatClientError("대화 목록 응답을 확인하지 못했어요.", 502);
  return sessions;
}

// For a guest, the first create issues the guest_id cookie.
export async function createChatSession(mode: ChatMode, title: string, signal?: AbortSignal): Promise<ChatSessionDto> {
  const room = await request(mode, SESSIONS, json("POST", { title }), signal, readJson);
  if (!isSession(room)) throw new ChatClientError("대화 생성 응답을 확인하지 못했어요.", 502, true);
  return room;
}

export async function renameChatSession(mode: ChatMode, sessionId: string, title: string, signal?: AbortSignal): Promise<ChatSessionDto> {
  const room = await request(mode, sessionPath(sessionId), json("PATCH", { title }), signal, readJson);
  if (!isSession(room) || room.id !== sessionId) throw new ChatClientError("대화 수정 응답을 확인하지 못했어요.", 502, true, sessionId);
  return room;
}

export async function deleteChatSession(mode: ChatMode, sessionId: string, signal?: AbortSignal): Promise<void> {
  await request(mode, sessionPath(sessionId), { method: "DELETE" }, signal, readEmpty);
}

export async function fetchChatHistory(mode: ChatMode, sessionId: string, signal?: AbortSignal): Promise<ChatMessageDto[]> {
  const messages = await request(mode, `${sessionPath(sessionId)}messages/`, { method: "GET" }, signal, readJson);
  if (!Array.isArray(messages) || !messages.every(isMessage)) throw new ChatClientError("대화 기록 응답을 확인하지 못했어요.", 502, false, sessionId);
  return messages;
}

/** Deletes the given user message and every later message in the session. */
export async function deleteChatMessages(mode: ChatMode, sessionId: string, messageId: number, signal?: AbortSignal): Promise<void> {
  if (!isMessageId(messageId)) throw new ChatClientError("메시지 번호를 확인해 주세요.", 400);
  await request(mode, `${sessionPath(sessionId)}messages/`, json("DELETE", { message_id: messageId }), signal, readEmpty);
}

export async function getChatStatus(mode: ChatMode, signal?: AbortSignal): Promise<ChatStatus> {
  await listChatSessions(mode, signal);
  return STATUS[mode];
}

// Reuses the shared request validator for content length and the structured context the v2 chain reads.
function parseInput(content: string, context?: ChatContext) {
  const parsed = parseChatRequest({ messages: [{ role: "user", content }], ...(context ? { context } : {}) });
  return { content: parsed.messages[0].content, ...(parsed.context ? { context: parsed.context } : {}) };
}

async function streamReply(mode: ChatMode, method: "POST" | "PUT", sessionId: string, body: unknown, signal: AbortSignal | undefined, callbacks: ChatStreamCallbacks): Promise<ChatReply> {
  try {
    const init = json(method, body);
    init.headers = { ...init.headers as Record<string, string>, Accept: "text/event-stream" };
    const result = await request(mode, `${sessionPath(sessionId)}messages/`, init, signal, response => readStream(response, sessionId, callbacks), null);
    return { ...STATUS[mode], ...result, tools: result.tools.map(tool => ({ id: tool.id, toolName: tool.tool_name, status: tool.status })), sessionId };
  } catch (error) {
    if (error instanceof ChatClientError && error.sessionId === undefined) error.sessionId = sessionId;
    throw error;
  }
}

export async function sendChatMessage(mode: ChatMode, body: ChatSendRequest, signal?: AbortSignal, callbacks: ChatStreamCallbacks = {}): Promise<ChatReply> {
  const input = parseInput(body.content, body.context);
  const sessionId = body.sessionId ?? (await createChatSession(mode, input.content.slice(0, 80), signal)).id;
  return streamReply(mode, "POST", sessionId, input, signal, callbacks);
}

/** Replaces a persisted user message: the server drops it and everything after, then streams a new answer. */
export async function editChatMessage(mode: ChatMode, body: ChatEditRequest, signal?: AbortSignal, callbacks: ChatStreamCallbacks = {}): Promise<ChatReply> {
  if (!isMessageId(body.messageId)) throw new ChatClientError("메시지 번호를 확인해 주세요.", 400);
  return streamReply(mode, "PUT", body.sessionId, { message_id: body.messageId, ...parseInput(body.content, body.context) }, signal, callbacks);
}
