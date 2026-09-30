"use client";

import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import type { ChatContext, ChatCourse, ChatMessage, ChatStatus, ChatToolCall } from "@/lib/chat/types";
import { MAX_MESSAGE_LENGTH } from "@/lib/chat/types";
import {
  ChatClientError,
  deleteChatMessages,
  deleteChatSession,
  editChatMessage,
  fetchChatHistory,
  getChatStatus,
  listChatSessions,
  sendChatMessage,
  type ChatMode,
} from "@/lib/chat/client";
import { commitChatLoad, restoreChatMessages } from "@/lib/chat/history";
import type { ChatToolCallDto } from "@/lib/chat/wire";
import { useMemberAuth } from "@/lib/member-auth";
import { createClientId } from "@/lib/client-id";
import { ChatPopup } from "./chat-popup";

type ConversationSnapshot = {
  messages: ChatMessage[];
  draft: string;
  context?: ChatContext;
  failed: string;
  failedContext?: ChatContext;
  error: string;
  notice: string;
};
/** 챗봇 코스를 받아 줄 화면 (루트 작성). apply 는 되돌리기 함수를 돌려준다. */
export type CourseTarget = {
  stadiumCode: string;
  stopCount: number;
  apply: (course: ChatCourse, how: "replace" | "append") => (() => void) | null;
};
export type AppliedCourse = { undo: (() => void) | null; message: string };
type ChatControls = ConversationSnapshot & {
  openChat: (initialMessage?: string, context?: ChatContext) => void;
  onExpand: () => void;
  onMinimize: () => void;
  onClosePopup: () => void;
  status: ChatStatus | null;
  statusLoading: boolean;
  statusError: string;
  pending: string;
  streaming: string;
  streamingTools: ChatToolCall[];
  /** 수정 중인 서버 저장 질문 id. 보내면 그 질문부터 이후 대화가 지워지고 답변을 새로 받는다. */
  editingMessageId: number | null;
  conversations: { id: string; title: string }[];
  activeConversationId: string;
  onDraftChange: (value: string) => void;
  onRefreshStatus: () => void;
  onSend: () => void;
  onRetry: () => void;
  onCancel: () => void;
  onReset: () => void;
  onSuggestion: (text: string, intent: ChatContext["intent"]) => void;
  onSelectConversation: (id: string) => void;
  /** 왼쪽 대화 목록에서 대화를 지운다 (화면에서 바로 빼고, 서버 기록 삭제는 가능한 경우에만 시도) */
  onDeleteConversation: (id: string) => void;
  onEditMessage: (id: number) => void;
  onCancelEdit: () => void;
  /** 저장된 질문과 그 이후 대화를 서버에서 지운다 (확인 후) */
  onDeleteMessage: (id: number) => void;
  onContextChange: (context?: ChatContext) => void;
  courseTarget: CourseTarget | null;
  registerCourseTarget: (target: CourseTarget | null) => void;
  openCourseInWriter: (course: ChatCourse) => void;
  takePendingCourse: () => ChatCourse | null;
  appliedCourses: ReadonlyMap<ChatCourse, AppliedCourse>;
  applyChatCourse: (course: ChatCourse, how: "replace" | "append") => void;
  undoChatCourse: (course: ChatCourse) => void;
};
const ChatControlsContext = createContext<ChatControls | null>(null);

export function useChat() {
  const value = useContext(ChatControlsContext);
  if (!value) throw new Error("useChat must be used inside ChatProvider");
  return value;
}

/**
 * 가이드 샘플 화면용 챗봇: 실제 대화·요청 없이 빈 대화 화면만 보여 준다 (연결 상태 표시는 실제 값을 따른다).
 */
export function ChatSampleProvider({ children }: { children: React.ReactNode }) {
  const real = useChat();
  const noop = () => {};
  const value: ChatControls = {
    ...real,
    messages: [], draft: "", context: undefined,
    failed: "", error: "", notice: "",
    pending: "", streaming: "", streamingTools: [], editingMessageId: null,
    conversations: [{ id: "guide-sample", title: "새 대화" }], activeConversationId: "guide-sample",
    openChat: noop, onExpand: noop, onMinimize: noop, onClosePopup: noop,
    onDraftChange: noop, onRefreshStatus: noop, onSend: noop, onRetry: noop, onCancel: noop, onReset: noop,
    onSuggestion: noop, onSelectConversation: noop, onDeleteConversation: noop,
    onEditMessage: noop, onCancelEdit: noop, onDeleteMessage: noop, onContextChange: noop,
    // 샘플 화면은 실제 챗봇 코스를 받지 않는다
    courseTarget: null, registerCourseTarget: noop, openCourseInWriter: noop, takePendingCourse: () => null,
    appliedCourses: new Map(), applyChatCourse: noop, undoChatCourse: noop,
  };
  return <ChatControlsContext.Provider value={value}>{children}</ChatControlsContext.Provider>;
}

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const { status: memberStatus, user } = useMemberAuth();
  const accountId = memberStatus === "authenticated" ? user!.id : null;
  const identity = memberStatus === "authenticated" ? `member:${accountId}` : memberStatus;
  // 회원은 Bearer, 비회원은 서버가 심은 guest_id 쿠키로 같은 대화 API 를 쓴다.
  const mode: ChatMode | null = memberStatus === "authenticated" ? "member" : memberStatus === "anonymous" ? "guest" : null;
  const pathname = usePathname();
  const router = useRouter();
  const isChatPage = pathname === "/chat";
  const hasEmbeddedChat = pathname === "/routes/new";
  const [popupRequested, setPopupRequested] = useState(false);
  const popupOpen = popupRequested && !isChatPage && !hasEmbeddedChat;
  const [activeConversationId, setActiveConversationId] = useState("initial-chat");
  const [conversations, setConversations] = useState([{ id: "initial-chat", title: "새 대화" }]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [context, setContext] = useState<ChatContext | undefined>();
  const [status, setStatus] = useState<ChatStatus | null>(null);
  const [statusLoading, setStatusLoading] = useState(true);
  const [statusError, setStatusError] = useState("");
  const [pending, setPending] = useState("");
  const [failed, setFailed] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [streaming, setStreaming] = useState("");
  const [streamingTools, setStreamingTools] = useState<ChatToolCall[]>([]);
  const [editingMessageId, setEditingMessageId] = useState<number | null>(null);
  const [chatIdentity, setChatIdentity] = useState(identity);
  const [courseTarget, setCourseTarget] = useState<CourseTarget | null>(null);
  const courseTargetRef = useRef<CourseTarget | null>(null);
  const [appliedCourses, setAppliedCourses] = useState<ReadonlyMap<ChatCourse, AppliedCourse>>(() => new Map());
  const appliedCoursesRef = useRef(appliedCourses);
  useEffect(() => { appliedCoursesRef.current = appliedCourses; }, [appliedCourses]);
  // 다른 화면(전체 채팅·팝업)에서 "루트 작성에서 열기"를 누르면 여기 두었다가 작성 화면이 가져간다.
  const pendingCourseRef = useRef<ChatCourse | null>(null);
  const historyRef = useRef<ChatMessage[]>([]);
  // 한 번에 하나의 전송·수정·삭제만 진행한다 (같은 화면의 중복 전송 방지).
  const requestRef = useRef<{ controller: AbortController; version: number; stopped: boolean } | null>(null);
  const statusRequestRef = useRef<AbortController | null>(null);
  const historyRequestRef = useRef<AbortController | null>(null);
  const syncRequestRef = useRef<AbortController | null>(null);
  const loadingConversationRef = useRef<string | null>(null);
  const requestVersion = useRef(0);
  const pendingRef = useRef("");
  const failedContextRef = useRef<ChatContext | undefined>(undefined);
  const retryRef = useRef<{ content: string; context?: ChatContext; messageId: number | null } | null>(null);
  const returnPageRef = useRef({ url: "/", scrollY: 0 });
  const restorePageRef = useRef(false);
  const popupOpenerRef = useRef<HTMLElement | null>(null);
  const topButtonRef = useRef<HTMLButtonElement>(null);
  const identityRef = useRef(identity);
  const activeConversationRef = useRef(activeConversationId);
  const identityChanged = chatIdentity !== identity;
  // Root layout keeps conversations alive across client-side page navigation.
  const backendSessions = useRef(new Map<string, string>());
  const archivedConversations = useRef(new Map<string, ConversationSnapshot>());

  const invalidateHistory = useCallback(() => {
    historyRequestRef.current?.abort();
    historyRequestRef.current = null;
    loadingConversationRef.current = null;
  }, []);

  const invalidateSync = useCallback(() => {
    syncRequestRef.current?.abort();
    syncRequestRef.current = null;
  }, []);

  const changeDraft = useCallback((value: string) => {
    if (!loadingConversationRef.current) invalidateHistory();
    setDraft(value);
  }, [invalidateHistory]);

  const loadStatus = useCallback((chatMode: ChatMode, controller: AbortController) => {
    return getChatStatus(chatMode, controller.signal).then(
      nextStatus => {
        if (!controller.signal.aborted) { setStatus(nextStatus); setStatusError(""); }
      },
      cause => {
        if (!controller.signal.aborted) {
          setStatus(null);
          setStatusError(cause instanceof Error ? cause.message : "연결 상태를 확인하지 못했어요.");
        }
      },
    ).finally(() => {
      if (!controller.signal.aborted) setStatusLoading(false);
    });
  }, []);

  const refreshStatus = useCallback(() => {
    statusRequestRef.current?.abort();
    if (!mode) {
      setStatus(null); setStatusLoading(memberStatus === "loading");
      setStatusError(memberStatus === "unavailable" ? "로그인 상태를 확인하지 못했어요." : "");
      return;
    }
    const controller = new AbortController();
    statusRequestRef.current = controller;
    setStatusLoading(true);
    setStatusError("");
    void loadStatus(mode, controller);
  }, [loadStatus, memberStatus, mode]);

  const closePopup = useCallback(() => {
    setPopupRequested(false);
    requestAnimationFrame(() => {
      const opener = popupOpenerRef.current;
      if (opener?.isConnected) opener.focus({ preventScroll: true });
      else topButtonRef.current?.focus({ preventScroll: true });
    });
  }, []);

  const expandChat = useCallback(() => {
    setPopupRequested(false);
    if (!isChatPage) {
      returnPageRef.current = { url: `${window.location.pathname}${window.location.search}${window.location.hash}`, scrollY: window.scrollY };
      router.push("/chat");
    }
  }, [isChatPage, router]);

  const minimizeChat = useCallback(() => {
    // Returning to the embedded assistant must not leave a hidden popup request.
    const destination = isChatPage
      ? new URL(returnPageRef.current.url, window.location.origin).pathname
      : pathname;
    setPopupRequested(destination !== "/routes/new");
    popupOpenerRef.current = null;
    if (isChatPage) {
      restorePageRef.current = true;
      router.push(returnPageRef.current.url, { scroll: false });
    }
  }, [isChatPage, pathname, router]);

  // 서버에는 중단을 저장하는 API 가 없다. 연결만 끊고, 받던 답변은 보관하지 않는다.
  const cancelRequest = useCallback(() => {
    const active = requestRef.current;
    if (!active || active.stopped) return;
    active.stopped = true;
    active.controller.abort();
  }, []);

  const archiveCurrentConversation = useCallback(() => {
    if (loadingConversationRef.current === activeConversationId) return;
    archivedConversations.current.set(activeConversationId, {
      messages: historyRef.current, draft, context, failed, failedContext: failedContextRef.current, error, notice,
    });
  }, [activeConversationId, context, draft, error, failed, notice]);

  const resetChat = useCallback(() => {
    if (requestRef.current) return;
    setEditingMessageId(null);
    if (!historyRef.current.length && !draft.trim() && !failed) {
      invalidateHistory();
      setContext(undefined);
      setNotice("");
      return;
    }
    archiveCurrentConversation();
    invalidateHistory();
    invalidateSync();
    const id = createClientId();
    activeConversationRef.current = id;
    setActiveConversationId(id);
    setConversations(current => [{ id, title: "새 대화" }, ...current]);
    historyRef.current = [];
    failedContextRef.current = undefined;
    setMessages([]);
    setDraft("");
    setPending("");
    setStreaming("");
    setStreamingTools([]);
    setFailed("");
    setError("");
    setNotice("");
    setContext(undefined);
  }, [archiveCurrentConversation, draft, failed, invalidateHistory, invalidateSync]);

  const showConversation = useCallback((saved: ConversationSnapshot, preserveDraft = false) => {
    historyRef.current = saved.messages;
    failedContextRef.current = saved.failedContext;
    setMessages(saved.messages);
    setDraft(current => preserveDraft && current.trim() ? current : saved.draft);
    setContext(saved.context);
    setFailed(saved.failed);
    setError(saved.error);
    setNotice(saved.notice);
    setStreaming("");
    setStreamingTools([]);
    setEditingMessageId(null);
  }, []);

  const restoreConversation = useCallback(async (id: string, sessionId: string, controller: AbortController, expectedIdentity: string) => {
    try {
      const history = await fetchChatHistory(expectedIdentity.startsWith("member:") ? "member" : "guest", sessionId, controller.signal);
      const saved: ConversationSnapshot = { messages: restoreChatMessages(history), draft: "", failed: "", error: "", notice: "" };
      commitChatLoad(controller.signal, () => identityRef.current === expectedIdentity && activeConversationRef.current === id, () => {
        loadingConversationRef.current = null;
        archivedConversations.current.set(id, saved);
        showConversation(saved, true);
      });
    } catch (cause) {
      if (!controller.signal.aborted && identityRef.current === expectedIdentity && activeConversationRef.current === id) {
        loadingConversationRef.current = null;
        setNotice("");
        setError(cause instanceof Error ? cause.message : "대화 기록을 불러오지 못했어요.");
      }
    }
  }, [showConversation]);

  // 전송·수정·삭제가 끝나면 서버 기록을 다시 읽어 저장된 메시지 id·상태를 화면과 맞춘다.
  // 초안·안내 문구는 건드리지 않고, 다른 대화로 옮겼거나 새 요청이 시작됐으면 버린다.
  const syncConversation = useCallback((id: string, sessionId: string, expectedIdentity: string, sentContent?: string) => {
    syncRequestRef.current?.abort();
    const controller = new AbortController();
    syncRequestRef.current = controller;
    void fetchChatHistory(expectedIdentity.startsWith("member:") ? "member" : "guest", sessionId, controller.signal).then(history => {
      commitChatLoad(controller.signal, () => identityRef.current === expectedIdentity && activeConversationRef.current === id && !requestRef.current, () => {
        historyRef.current = restoreChatMessages(history);
        archivedConversations.current.delete(id);
        setMessages(historyRef.current);
        // 실패·중단된 질문이 서버에 저장돼 있으면 따로 띄운 실패 말풍선은 거두고, 다시 시도는 그 질문 자리에서 한다.
        const stored = [...historyRef.current].reverse().find(message => message.role === "user");
        if (sentContent && stored?.id && stored.status !== "completed" && stored.content === sentContent) {
          setFailed("");
          if (retryRef.current) retryRef.current = { ...retryRef.current, messageId: stored.id };
        }
      });
    }, () => undefined);
  }, []);

  const selectConversation = useCallback((id: string) => {
    if (requestRef.current || id === activeConversationId) return;
    const saved = archivedConversations.current.get(id);
    archiveCurrentConversation();
    invalidateHistory();
    invalidateSync();
    setActiveConversationId(id);
    activeConversationRef.current = id;
    if (saved) { showConversation(saved); return; }
    const sessionId = backendSessions.current.get(id);
    if (!sessionId) return;
    const controller = new AbortController();
    historyRequestRef.current = controller;
    loadingConversationRef.current = id;
    historyRef.current = [];
    setMessages([]); setDraft(""); setFailed(""); setError(""); setNotice("대화 기록을 불러오고 있어요."); setStreaming(""); setStreamingTools([]); setEditingMessageId(null);
    void restoreConversation(id, sessionId, controller, identityRef.current);
  }, [activeConversationId, archiveCurrentConversation, invalidateHistory, invalidateSync, restoreConversation, showConversation]);

  const deleteConversation = useCallback((id: string) => {
    // 답변을 받는 중인 대화는 지우지 않는다
    if (requestRef.current && id === activeConversationId) return;
    const sessionId = backendSessions.current.get(id);
    const remaining = conversations.filter(conversation => conversation.id !== id);
    if (remaining.length === conversations.length) return;
    if (id === activeConversationId) {
      const next = remaining[0];
      if (next) {
        selectConversation(next.id);
      } else {
        invalidateHistory();
        const fresh = createClientId();
        activeConversationRef.current = fresh;
        setActiveConversationId(fresh);
        remaining.push({ id: fresh, title: "새 대화" });
        invalidateSync();
        historyRef.current = [];
        failedContextRef.current = undefined;
        setMessages([]); setDraft(""); setFailed(""); setError(""); setStreaming(""); setStreamingTools([]); setContext(undefined); setEditingMessageId(null);
      }
      setNotice("대화 내역을 지웠어요.");
    }
    archivedConversations.current.delete(id);
    backendSessions.current.delete(id);
    setConversations(remaining);
    // 서버 기록 삭제가 실패해도 화면에서는 지운 상태를 유지한다
    if (sessionId && mode) void deleteChatSession(mode, sessionId).catch(() => undefined);
  }, [activeConversationId, conversations, invalidateHistory, invalidateSync, mode, selectConversation]);

  useEffect(() => {
    if (identityRef.current === identity) return;
    identityRef.current = identity;
    setChatIdentity(identity);
    requestVersion.current += 1;
    requestRef.current?.controller.abort();
    statusRequestRef.current?.abort();
    historyRequestRef.current?.abort();
    syncRequestRef.current?.abort();
    loadingConversationRef.current = null;
    requestRef.current = null;
    statusRequestRef.current = null;
    pendingRef.current = "";
    backendSessions.current.clear();
    archivedConversations.current.clear();
    historyRef.current = [];
    failedContextRef.current = undefined;
    setActiveConversationId("initial-chat");
    activeConversationRef.current = "initial-chat";
    setConversations([{ id: "initial-chat", title: "새 대화" }]);
    setMessages([]);
    setDraft("");
    setContext(undefined);
    setStatus(null);
    setStatusLoading(memberStatus !== "unavailable");
    setStatusError(memberStatus === "unavailable" ? "로그인 상태를 확인하지 못했어요." : "");
    setPending("");
    setStreaming("");
    setStreamingTools([]);
    setEditingMessageId(null);
    setFailed("");
    setError("");
    setNotice("");
  }, [identity, memberStatus]);

  // 회원은 계정의 대화를, 비회원은 guest_id 쿠키의 대화를 불러온다 (쿠키가 없으면 빈 목록).
  useEffect(() => {
    if (!mode || identityRef.current !== identity) return;
    invalidateHistory();
    const controller = new AbortController(), expectedIdentity = identity;
    historyRequestRef.current = controller;
    void listChatSessions(mode, controller.signal).then(sessions => {
      commitChatLoad(controller.signal, () => identityRef.current === expectedIdentity, () => {
        backendSessions.current.clear();
        if (!sessions.length) return;
        const rooms = sessions.map(room => ({ id: `${mode}:${room.id}`, title: room.title || "새 대화" }));
        rooms.forEach((room, index) => backendSessions.current.set(room.id, sessions[index].id));
        const first = rooms[0];
        setConversations(rooms);
        setActiveConversationId(first.id);
        activeConversationRef.current = first.id;
        loadingConversationRef.current = first.id;
        setNotice("대화 기록을 불러오고 있어요.");
        void restoreConversation(first.id, sessions[0].id, controller, expectedIdentity);
      });
    }).catch(cause => {
      if (!controller.signal.aborted && identityRef.current === expectedIdentity) setError(cause instanceof Error ? cause.message : "대화방을 불러오지 못했어요.");
    });
    return () => controller.abort();
  }, [accountId, identity, invalidateHistory, mode, restoreConversation]);

  const registerCourseTarget = useCallback((target: CourseTarget | null) => {
    courseTargetRef.current = target;
    setCourseTarget(target);
  }, []);
  const applyChatCourse = useCallback((course: ChatCourse, how: "replace" | "append") => {
    const target = courseTargetRef.current;
    if (!target) return;
    const had = target.stopCount, otherStadium = Boolean(course.stadiumCode && course.stadiumCode !== target.stadiumCode);
    const undo = target.apply(course, how);
    const message = !undo ? "이 코스를 지도에 담지 못했어요. 구장을 확인해 주세요."
      : otherStadium ? "구장을 바꾸고 추천 코스를 옆 지도에 그렸어요."
      : how === "append" ? "내 코스 뒤에 이어 담았어요."
      : had ? `옆 지도에 추천 코스를 그렸어요. 원래 담아둔 ${had}곳은 되돌리기로 복구할 수 있어요.`
      : "옆 지도에 추천 코스를 그렸어요. 순서는 내 코스에서 바꿀 수 있어요.";
    setAppliedCourses(current => new Map(current).set(course, { undo, message }));
  }, []);
  const undoChatCourse = useCallback((course: ChatCourse) => {
    appliedCoursesRef.current.get(course)?.undo?.();
    setAppliedCourses(current => new Map(current).set(course, { undo: null, message: "담기 전 코스로 되돌렸어요." }));
  }, []);

  const send = useCallback(async (text = draft, options?: { context?: ChatContext; editId?: number | null }) => {
    const selectedContext = options ? options.context : context;
    const editId = options ? options.editId ?? null : editingMessageId;
    const content = text.trim();
    if (identityRef.current !== identity || !content || requestRef.current || content.length > MAX_MESSAGE_LENGTH) return;
    if (loadingConversationRef.current === activeConversationId) {
      setNotice("대화 기록을 불러온 뒤 보내 주세요.");
      return;
    }
    if (!mode) { setError("로그인 상태를 확인한 뒤 다시 시도해 주세요."); return; }
    const sessionId = backendSessions.current.get(activeConversationId);
    const editIndex = editId === null ? -1 : historyRef.current.findIndex(message => message.id === editId && message.role === "user");
    if (editId !== null && (!sessionId || editIndex < 0)) { setEditingMessageId(null); setError("수정할 질문을 찾지 못했어요."); return; }
    if (editId !== null && !window.confirm("이 질문 이후의 대화는 모두 지워지고 답변을 새로 받아요. 계속할까요?")) return;
    const controller = new AbortController();
    const version = ++requestVersion.current;
    const conversationId = activeConversationId, expectedIdentity = identity;
    invalidateHistory();
    invalidateSync();
    const active = { controller, version, stopped: false };
    requestRef.current = active;
    pendingRef.current = content;
    setPending(content);
    setDraft("");
    setEditingMessageId(null);
    setError("");
    setNotice("");
    setFailed("");
    setStreaming("");
    setStreamingTools([]);
    failedContextRef.current = undefined;
    retryRef.current = null;
    const userMessage: ChatMessage = { role: "user", content };
    const previous = editIndex >= 0 ? historyRef.current.slice(0, editIndex) : historyRef.current;
    if (editIndex >= 0) { historyRef.current = previous; setMessages(previous); }
    if (previous.length === 0) {
      setConversations(current => current.map(item => item.id === conversationId
        ? { ...item, title: content.replace(/\s+/g, " ").slice(0, 48) }
        : item));
    }
    let knownSession = sessionId;
    try {
      const onDelta = (answer: string) => {
        if (version !== requestVersion.current) return;
        setStreaming(answer);
      };
      const onTool = (tool: ChatToolCallDto) => {
        if (version !== requestVersion.current) return;
        const next = { id: tool.id, toolName: tool.tool_name, status: tool.status };
        setStreamingTools(current => {
          const index = current.findIndex(item => item.id === next.id);
          return index < 0 ? [...current, next] : current.map((item, itemIndex) => itemIndex === index ? next : item);
        });
      };
      const reply = editId !== null && sessionId
        ? await editChatMessage(mode, { sessionId, messageId: editId, content, context: selectedContext }, controller.signal, { onDelta, onTool })
        : await sendChatMessage(mode, { sessionId, content, context: selectedContext }, controller.signal, { onDelta, onTool });
      if (version !== requestVersion.current) return;
      knownSession = reply.sessionId;
      if (reply.sessionId) backendSessions.current.set(conversationId, reply.sessionId);
      const assistant: ChatMessage = { role: "assistant", content: reply.reply, status: "completed", ...(reply.assistantMessageId ? { id: reply.assistantMessageId } : {}), ...(reply.tools?.length ? { tools: reply.tools } : {}) };
      const next: ChatMessage[] = [...previous, { ...userMessage, status: "completed" }, assistant];
      historyRef.current = next;
      setMessages(next);
      setStatus({ provider: reply.provider, model: reply.model, ready: reply.ready });
    } catch (cause) {
      if (version !== requestVersion.current) return;
      if (cause instanceof ChatClientError && cause.sessionId) {
        knownSession = cause.sessionId;
        backendSessions.current.set(conversationId, cause.sessionId);
      }
      setDraft(current => current.trim() ? current : content);
      if (active.stopped) {
        // 중단은 이 화면의 연결만 끊는다. 서버에는 답변 없는 질문이 남을 수 있어 기록을 다시 읽는다.
        setNotice("답변 받기를 중단했어요. 받던 답변은 저장되지 않아요.");
      } else {
        setFailed(content);
        failedContextRef.current = selectedContext;
        retryRef.current = { content, context: selectedContext, messageId: editId };
        setError(cause instanceof Error ? cause.message : "답변을 가져오지 못했어요. 다시 시도해 주세요.");
      }
    } finally {
      if (version === requestVersion.current) {
        requestRef.current = null;
        pendingRef.current = "";
        setPending("");
        setStreaming("");
        setStreamingTools([]);
        if (knownSession) syncConversation(conversationId, knownSession, expectedIdentity, content);
      }
    }
  }, [activeConversationId, context, draft, editingMessageId, identity, invalidateHistory, invalidateSync, mode, syncConversation]);

  const editMessage = useCallback((id: number) => {
    if (requestRef.current || loadingConversationRef.current) return;
    const target = historyRef.current.find(message => message.id === id && message.role === "user");
    if (!target) return;
    invalidateSync();
    setEditingMessageId(id);
    setDraft(target.content);
    setNotice("질문을 고쳐 보내면 이 질문 이후의 대화는 지워져요.");
  }, [invalidateSync]);

  const cancelEdit = useCallback(() => {
    setEditingMessageId(null);
    setDraft("");
    setNotice("");
  }, []);

  const deleteMessage = useCallback(async (id: number) => {
    const sessionId = backendSessions.current.get(activeConversationId);
    if (requestRef.current || loadingConversationRef.current || !mode || !sessionId) return;
    if (!historyRef.current.some(message => message.id === id && message.role === "user")) return;
    if (!window.confirm("이 질문과 이후 대화를 모두 지울까요? 지운 대화는 되돌릴 수 없어요.")) return;
    const controller = new AbortController();
    const version = ++requestVersion.current;
    const conversationId = activeConversationId, expectedIdentity = identity;
    invalidateSync();
    requestRef.current = { controller, version, stopped: false };
    setEditingMessageId(null);
    setError("");
    setNotice("대화를 지우고 있어요.");
    try {
      await deleteChatMessages(mode, sessionId, id, controller.signal);
      if (version !== requestVersion.current) return;
      const index = historyRef.current.findIndex(message => message.id === id);
      historyRef.current = historyRef.current.slice(0, index);
      setMessages(historyRef.current);
      setFailed("");
      retryRef.current = null;
      setNotice("선택한 질문부터 이후 대화를 지웠어요.");
    } catch (cause) {
      if (version !== requestVersion.current) return;
      setNotice("");
      setError(cause instanceof Error ? cause.message : "대화를 지우지 못했어요.");
    } finally {
      if (version === requestVersion.current) {
        requestRef.current = null;
        syncConversation(conversationId, sessionId, expectedIdentity);
      }
    }
  }, [activeConversationId, identity, invalidateSync, mode, syncConversation]);

  const retry = useCallback(() => {
    const target = retryRef.current;
    if (!target) return;
    // 서버에 실패한 질문이 남아 있으면 같은 자리에서 다시 받는다 (질문이 겹쳐 쌓이지 않게).
    const stored = target.messageId !== null && historyRef.current.some(message => message.id === target.messageId);
    void send(target.content, { context: target.context, editId: stored ? target.messageId : null });
  }, [send]);

  const openCourseInWriter = useCallback((course: ChatCourse) => {
    pendingCourseRef.current = course;
    setPopupRequested(false);
    router.push(`/routes/new${course.stadiumCode ? `?stadium=${encodeURIComponent(course.stadiumCode)}` : ""}`);
  }, [router]);
  const takePendingCourse = useCallback(() => {
    const course = pendingCourseRef.current;
    pendingCourseRef.current = null;
    return course;
  }, []);

  const openChat = useCallback((initialMessage?: string, nextContext?: ChatContext) => {
    expandChat();
    if (nextContext) setContext(nextContext);
    if (requestRef.current) {
      if (initialMessage?.trim()) {
        changeDraft(initialMessage.slice(0, MAX_MESSAGE_LENGTH));
        setNotice("지금 답변이 끝나면 아래에 준비한 질문을 보낼 수 있어요.");
      }
      return;
    }
    if (initialMessage?.trim()) void send(initialMessage.slice(0, MAX_MESSAGE_LENGTH), { context: nextContext ?? context, editId: null });
  }, [changeDraft, context, expandChat, send]);

  useEffect(() => {
    if (isChatPage || !restorePageRef.current) return;
    restorePageRef.current = false;
    const frame = requestAnimationFrame(() => window.scrollTo({ top: returnPageRef.current.scrollY, behavior: "instant" }));
    return () => cancelAnimationFrame(frame);
  }, [isChatPage]);

  useEffect(() => {
    if (!isChatPage && !popupOpen && !hasEmbeddedChat) return;
    statusRequestRef.current?.abort();
    if (!mode) return;
    const controller = new AbortController();
    statusRequestRef.current = controller;
    void loadStatus(mode, controller);
    return () => controller.abort();
  }, [accountId, hasEmbeddedChat, isChatPage, popupOpen, loadStatus, mode]);

  useEffect(() => () => {
    requestVersion.current += 1;
    requestRef.current?.controller.abort();
    statusRequestRef.current?.abort();
    historyRequestRef.current?.abort();
    syncRequestRef.current?.abort();
  }, []);

  const visibleStatus = mode ? status : null;
  const visibleStatusLoading = memberStatus === "loading" || (Boolean(mode) && statusLoading);
  const visibleStatusError = memberStatus === "unavailable" ? "로그인 상태를 확인하지 못했어요." : mode ? statusError : "";

  return (
    <ChatControlsContext.Provider value={{
      openChat, onExpand: expandChat, onMinimize: minimizeChat, onClosePopup: closePopup,
      messages: identityChanged ? [] : messages,
      draft: identityChanged ? "" : draft,
      context: identityChanged ? undefined : context,
      status: visibleStatus,
      statusLoading: visibleStatusLoading,
      statusError: visibleStatusError,
      pending: identityChanged ? "" : pending,
      streaming: identityChanged ? "" : streaming,
      streamingTools: identityChanged ? [] : streamingTools,
      editingMessageId: identityChanged ? null : editingMessageId,
      failed: identityChanged ? "" : failed,
      error: identityChanged ? "" : error,
      notice: identityChanged ? "" : notice,
      conversations: identityChanged ? [{ id: "initial-chat", title: "새 대화" }] : conversations,
      activeConversationId: identityChanged ? "initial-chat" : activeConversationId,
      onDraftChange: changeDraft, onRefreshStatus: () => void refreshStatus(),
      onSend: () => void send(), onRetry: retry,
      onCancel: cancelRequest, onReset: resetChat,
      onSuggestion: (text, intent) => { changeDraft(text); setContext(current => ({ ...current, intent })); },
      onSelectConversation: selectConversation,
      onDeleteConversation: deleteConversation,
      onEditMessage: editMessage, onCancelEdit: cancelEdit, onDeleteMessage: id => void deleteMessage(id),
      onContextChange: setContext,
      courseTarget, registerCourseTarget, openCourseInWriter, takePendingCourse,
      appliedCourses, applyChatCourse, undoChatCourse,
    }}>
      {children}
      {!popupOpen && <button ref={topButtonRef} type="button" className="scroll-to-top" aria-label="맨 위로 이동" title="맨 위로 이동" onClick={() => window.scrollTo({ top: 0, behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth" })}>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="m6 11 6-6 6 6M12 5v14" /></svg>
      </button>}
      {popupOpen && <ChatPopup />}
    </ChatControlsContext.Provider>
  );
}
