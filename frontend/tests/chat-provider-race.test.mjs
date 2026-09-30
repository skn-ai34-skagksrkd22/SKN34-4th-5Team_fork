import assert from "node:assert/strict";
import { after, test } from "node:test";
import { createRequire } from "node:module";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";

const frontend = dirname(dirname(fileURLToPath(import.meta.url)));
const scratch = mkdtempSync(join(tmpdir(), "kbo-chat-provider-race-"));
after(() => rmSync(scratch, { recursive: true, force: true }));
symlinkSync(join(frontend, "node_modules"), join(scratch, "node_modules"), "dir");

function compile(name) {
  const source = readFileSync(join(frontend, `${name}.ts`), "utf8");
  const { outputText } = ts.transpileModule(source, {
    fileName: `${name}.ts`, compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  });
  mkdirSync(dirname(join(scratch, `${name}.js`)), { recursive: true });
  writeFileSync(join(scratch, `${name}.js`), outputText);
}

for (const name of ["lib/chat/types", "lib/chat/validation", "lib/chat/history"]) compile(name);
mkdirSync(join(scratch, "components"), { recursive: true });

const providerSource = readFileSync(join(frontend, "components/chat-provider.tsx"), "utf8")
  .replace('from "react"', 'from "../test-react"')
  .replace('from "next/navigation"', 'from "../test-navigation"')
  .replaceAll('from "@/lib/chat/types"', 'from "../lib/chat/types"')
  .replace('from "@/lib/chat/client"', 'from "../test-chat-client"')
  .replace('from "@/lib/chat/history"', 'from "../lib/chat/history"')
  .replace('from "@/lib/member-auth"', 'from "../test-member-auth"')
  .replace('from "@/lib/client-id"', 'from "../test-client-id"')
  .replace('from "./chat-popup"', 'from "../test-popup"');
writeFileSync(join(scratch, "components/chat-provider.js"), ts.transpileModule(providerSource, {
  fileName: "chat-provider.tsx",
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
}).outputText);

writeFileSync(join(scratch, "test-react.js"), `
exports.createContext = (...args) => global.__hooks.createContext(...args);
exports.useCallback = (...args) => global.__hooks.useCallback(...args);
exports.useContext = (...args) => global.__hooks.useContext(...args);
exports.useEffect = (...args) => global.__hooks.useEffect(...args);
exports.useRef = (...args) => global.__hooks.useRef(...args);
exports.useState = (...args) => global.__hooks.useState(...args);
`);
writeFileSync(join(scratch, "test-navigation.js"), `
exports.usePathname = () => "/";
exports.useRouter = () => ({ push() {} });
`);
writeFileSync(join(scratch, "test-member-auth.js"), `exports.useMemberAuth = () => global.__memberAuth;`);
writeFileSync(join(scratch, "test-client-id.js"), `exports.createClientId = () => "new-chat";`);
writeFileSync(join(scratch, "test-popup.js"), `exports.ChatPopup = () => null;`);
writeFileSync(join(scratch, "test-chat-client.js"), `
class ChatClientError extends Error {}
exports.ChatClientError = ChatClientError;
exports.GUEST_STATUS = { provider: "guest", model: "guest", ready: true };
for (const name of ["deleteChatMessages", "deleteChatSession", "editChatMessage", "fetchChatHistory", "getChatStatus", "listChatSessions", "sendChatMessage"])
  exports[name] = (...args) => global.__chatApi[name](...args);
`);

function hookRunner() {
  const slots = [];
  let cursor = 0;
  let pending = [];
  const same = (left, right) => left && right && left.length === right.length && left.every((value, index) => Object.is(value, right[index]));
  const hooks = {
    createContext: value => ({ value, Provider() {} }),
    useContext: context => context.value,
    useState(initial) {
      const index = cursor++;
      if (!slots[index]) slots[index] = { value: typeof initial === "function" ? initial() : initial };
      return [slots[index].value, value => { slots[index].value = typeof value === "function" ? value(slots[index].value) : value; }];
    },
    useRef(initial) {
      const index = cursor++;
      if (!slots[index]) slots[index] = { value: { current: initial } };
      return slots[index].value;
    },
    useCallback(callback, dependencies) {
      const index = cursor++;
      if (!slots[index] || !same(slots[index].dependencies, dependencies)) slots[index] = { value: callback, dependencies };
      return slots[index].value;
    },
    useEffect(effect, dependencies) {
      const index = cursor++;
      if (!slots[index] || !same(slots[index].dependencies, dependencies)) {
        const previous = slots[index]?.cleanup;
        slots[index] = { ...slots[index], dependencies };
        pending.push(() => {
          previous?.();
          slots[index].cleanup = effect();
        });
      }
    },
  };
  global.__hooks = hooks;
  const require = createRequire(join(scratch, "entry.cjs"));
  const { ChatProvider } = require("./components/chat-provider.js");
  return {
    render() {
      global.__hooks = hooks;
      cursor = 0;
      pending = [];
      return ChatProvider({ children: null }).props.value;
    },
    flushEffects() {
      const effects = pending;
      pending = [];
      effects.forEach(run => run());
    },
  };
}

const deferred = () => {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
};
const tick = () => new Promise(resolve => setImmediate(resolve));
const FIRST = "3f2c1a4e-8b7d-4c21-9e0f-5a6b7c8d9e01", SECOND = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d";
const USER_MSG = 1, ASSISTANT_MSG = 2;
const OLD_MSG = 4;
const rooms = [{ id: FIRST, title: "첫 대화" }, { id: SECOND, title: "둘째 대화" }];
const row = (id, role, content, status = "completed", tools = []) => ({ id, sequence_no: id, role, content, status, tools, created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" });
const unused = async () => { throw new Error("not used"); };
const baseApi = {
  deleteChatMessages: unused, deleteChatSession: unused, editChatMessage: unused, sendChatMessage: unused,
  fetchChatHistory: async () => [],
  getChatStatus: async mode => ({ provider: mode === "member" ? "backend" : "guest", model: "server", ready: true }),
};

global.window = {
  location: { pathname: "/", search: "", hash: "", origin: "http://localhost" }, scrollY: 0,
  scrollTo() {}, matchMedia: () => ({ matches: false }),
};
global.requestAnimationFrame = callback => { callback(); return 1; };
global.cancelAnimationFrame = () => {};
global.__memberAuth = { status: "authenticated", user: { id: 7 } };

test("provider ignores delayed list/history callbacks and reloads an interrupted room", async () => {
  global.__memberAuth = { status: "authenticated", user: { id: 7 } };
  const lateList = deferred();
  global.__chatApi = { ...baseApi, listChatSessions: () => lateList.promise };
  let runner = hookRunner();
  let controls = runner.render();
  runner.flushEffects();
  controls.onDraftChange("작성 중");
  lateList.resolve(rooms);
  await tick();
  controls = runner.render();
  assert.equal(controls.activeConversationId, "initial-chat");
  assert.equal(controls.draft, "작성 중");
  assert.deepEqual(controls.conversations, [{ id: "initial-chat", title: "새 대화" }]);

  const firstHistory = deferred(), secondHistory = deferred();
  const historyCalls = [];
  global.__chatApi = {
    ...baseApi,
    listChatSessions: async () => rooms,
    fetchChatHistory: (mode, sessionId) => {
      historyCalls.push([mode, sessionId]);
      if (sessionId === FIRST && historyCalls.filter(([, id]) => id === FIRST).length === 1) return firstHistory.promise;
      if (sessionId === SECOND) return secondHistory.promise;
      return Promise.resolve([]);
    },
  };
  runner = hookRunner();
  controls = runner.render();
  runner.flushEffects();
  await tick();
  controls = runner.render();
  assert.equal(controls.activeConversationId, `member:${FIRST}`);

  controls.onSelectConversation(`member:${SECOND}`);
  controls = runner.render();
  controls.onDraftChange("둘째 방 초안");
  firstHistory.resolve([row(OLD_MSG, "user", "늦은 첫 기록")]);
  await tick();
  controls = runner.render();
  assert.equal(controls.activeConversationId, `member:${SECOND}`);
  assert.equal(controls.draft, "둘째 방 초안");

  secondHistory.resolve([row(USER_MSG, "user", "둘째 질문"), row(ASSISTANT_MSG, "assistant", "둘째 답변")]);
  await tick();
  controls = runner.render();
  assert.deepEqual(controls.messages.map(message => [message.id, message.role, message.content]), [[USER_MSG, "user", "둘째 질문"], [ASSISTANT_MSG, "assistant", "둘째 답변"]]);
  assert.equal(controls.draft, "둘째 방 초안");

  controls.onSelectConversation(`member:${FIRST}`);
  assert.deepEqual(historyCalls, [["member", FIRST], ["member", SECOND], ["member", FIRST]]);
});

test("guest reload lists cookie-owned sessions and restores history in guest mode", async () => {
  global.__memberAuth = { status: "anonymous", user: null };
  const calls = [];
  global.__chatApi = {
    ...baseApi,
    listChatSessions: async mode => { calls.push(["list", mode]); return [rooms[0]]; },
    fetchChatHistory: async (mode, sessionId) => { calls.push(["history", mode, sessionId]); return [row(USER_MSG, "user", "비회원 질문"), row(ASSISTANT_MSG, "assistant", "비회원 답")]; },
  };
  const runner = hookRunner();
  let controls = runner.render();
  runner.flushEffects();
  await tick();
  controls = runner.render();
  assert.equal(controls.activeConversationId, `guest:${FIRST}`);
  assert.deepEqual(controls.messages.map(message => message.content), ["비회원 질문", "비회원 답"]);
  assert.ok(calls.some(call => call[0] === "list" && call[1] === "guest"));
  assert.deepEqual(calls.filter(call => call[0] === "history"), [["history", "guest", FIRST]]);
  global.__memberAuth = { status: "authenticated", user: { id: 7 } };
});

test("history reload surfaces a stopped turn's status without dropping it", async () => {
  global.__memberAuth = { status: "authenticated", user: { id: 7 } };
  global.__chatApi = {
    ...baseApi,
    listChatSessions: async () => [rooms[0]],
    fetchChatHistory: async () => [row(USER_MSG, "user", "질문", "stopped"), row(ASSISTANT_MSG, "assistant", "", "stopped")],
  };
  const runner = hookRunner();
  let controls = runner.render();
  runner.flushEffects();
  await tick();
  controls = runner.render();
  assert.deepEqual(controls.messages.map(message => [message.id, message.status]), [[USER_MSG, "stopped"], [ASSISTANT_MSG, "stopped"]]);
});

test("live tool events accumulate into streamingTools by id and clear once the turn settles", async () => {
  global.__memberAuth = { status: "anonymous", user: null };
  let onTool;
  const persisted = [row(USER_MSG, "user", "잠실 맛집 알려 주세요"), row(ASSISTANT_MSG, "assistant", "답변", "completed", [{ id: "call-1", tool_name: "search_places", status: "completed" }])];
  global.__chatApi = {
    ...baseApi,
    listChatSessions: async () => [],
    fetchChatHistory: async () => persisted,
    sendChatMessage: (mode, body, signal, callbacks) => { onTool = callbacks.onTool; return new Promise(resolve => {
      onTool({ id: "call-1", tool_name: "search_places", status: "running" });
      resolve({ reply: "답변", sessionId: FIRST, provider: "guest", model: "m", ready: true, assistantMessageId: ASSISTANT_MSG, tools: [{ id: "call-1", toolName: "search_places", status: "completed" }] });
    }); },
  };
  const runner = hookRunner();
  let controls = runner.render();
  runner.flushEffects();
  await tick();
  controls = runner.render();
  controls.onDraftChange("잠실 맛집 알려 주세요");
  controls = runner.render();
  controls.onSend();
  await tick();
  controls = runner.render();
  assert.deepEqual(controls.streamingTools, []);
  await tick();
  controls = runner.render();
  assert.deepEqual(controls.messages.at(-1).tools, [{ id: "call-1", toolName: "search_places", status: "completed" }]);
  global.__memberAuth = { status: "authenticated", user: { id: 7 } };
});

test("Stop aborts only the local stream, clears loading and shows the unsaved notice", async () => {
  global.__memberAuth = { status: "anonymous", user: null };
  const sends = [];
  global.__chatApi = {
    ...baseApi,
    listChatSessions: async () => [],
    sendChatMessage: (mode, body, signal) => {
      sends.push([mode, body.content]);
      return new Promise((_, reject) => signal.addEventListener("abort", () => reject(new Error("요청이 중단됐어요.")), { once: true }));
    },
  };
  const runner = hookRunner();
  let controls = runner.render();
  runner.flushEffects();
  await tick();
  controls = runner.render();
  controls.onDraftChange("잠실 맛집 알려 주세요");
  controls = runner.render();
  controls.onSend();
  controls = runner.render();
  assert.equal(controls.pending, "잠실 맛집 알려 주세요");
  controls.onCancel();
  await tick();
  controls = runner.render();
  assert.equal(controls.pending, "");
  assert.equal(controls.failed, "");
  assert.equal(controls.error, "");
  assert.equal(controls.notice, "답변 받기를 중단했어요. 받던 답변은 저장되지 않아요.");
  assert.equal(controls.draft, "잠실 맛집 알려 주세요");
  assert.deepEqual(sends, [["guest", "잠실 맛집 알려 주세요"]]);
  global.__memberAuth = { status: "authenticated", user: { id: 7 } };
});

// Stop clears `pending` inside its own click, so React reuses that <button> node as the submit/send button.
// Without preventDefault the browser's default activation then re-submits the restored draft (seen in the real UI).
for (const [path, exportName] of [["components/chat-workspace", "ChatWorkspace"], ["components/chat-popup", "ChatPopup"]]) {
  test(`${exportName} stop button cancels without the default submit activation`, () => {
    const source = readFileSync(join(frontend, `${path}.tsx`), "utf8")
      .replace(/^import "@\/styles\/[^"]+";$/m, "")
      .replace('from "react"', 'from "../test-surface-react"')
      .replace('from "next/link"', 'from "../test-surface-stub"')
      .replace('from "@/lib/chat/types"', 'from "../lib/chat/types"')
      .replace('from "@/lib/member-auth"', 'from "../test-member-auth"')
      .replace(/from "\.\/(chat-provider|icons|chat-answer|chat-course-card|chat-pending|chat-progress)"/g, 'from "../test-surface-stub"');
    writeFileSync(join(scratch, `${path}.js`), ts.transpileModule(source, {
      fileName: `${path}.tsx`, compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
    }).outputText);
    writeFileSync(join(scratch, "test-surface-react.js"), "exports.useEffect = () => {}; exports.useRef = current => ({ current });");
    writeFileSync(join(scratch, "test-surface-stub.js"), "const Stub = () => null; module.exports = new Proxy({ __esModule: true, default: Stub, useChat: () => global.__chat }, { get: (target, key) => key in target ? target[key] : Stub });");
    global.__memberAuth = { status: "anonymous", user: null };
    let cancelled = 0;
    global.__chat = new Proxy({
      messages: [], conversations: [{ id: "initial-chat", title: "새 대화" }], activeConversationId: "initial-chat",
      draft: "잠실 맛집 알려 주세요", pending: "잠실 맛집 알려 주세요", streaming: "", failed: "", error: "", notice: "",
      editingMessageId: null, status: { provider: "guest", model: "m", ready: true }, statusLoading: false, statusError: "",
      onCancel: () => { cancelled += 1; },
    }, { get: (target, key) => key in target ? target[key] : () => {} });
    const require = createRequire(join(scratch, "entry.cjs"));
    const Surface = require(`./${path}.js`)[exportName];
    const find = node => {
      if (!node || typeof node !== "object") return null;
      if (Array.isArray(node)) { for (const child of node) { const hit = find(child); if (hit) return hit; } return null; }
      if (node.props?.["aria-label"] === "답변 생성 중단") return node;
      return find(node.props?.children);
    };
    const stop = find(Surface({}));
    assert.ok(stop, "stop button renders while a reply is pending");
    let prevented = false;
    stop.props.onClick({ preventDefault: () => { prevented = true; } });
    assert.equal(cancelled, 1);
    assert.equal(prevented, true);
    global.__memberAuth = { status: "authenticated", user: { id: 7 } };
  });
}
