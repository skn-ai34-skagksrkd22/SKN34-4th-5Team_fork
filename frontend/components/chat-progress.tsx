import type { ChatToolCall, ChatToolStatus } from "@/lib/chat/types";
import "@/styles/chat-progress.css";

const STATUS: Record<ChatToolStatus, { mark: string; label: string }> = {
  running: { mark: "…", label: "호출 중" },
  completed: { mark: "✓", label: "호출 완료" },
  failed: { mark: "!", label: "실패" },
};
// serializer/message.py 가 넘겨주는 tool_name 은 LangChain 도구 함수 이름이라 화면에 그대로 쓰기 어렵다.
const TOOL_LABELS: Record<string, string> = {
  search_places: "장소 검색",
  search_courses: "코스 검색",
  get_course: "코스 조회",
  get_directions: "경로 검색",
  search_tourism: "관광지 검색",
  get_weather: "날씨 조회",
  search_community_posts: "커뮤니티 검색",
  get_prediction_games: "승부예측 조회",
  get_stadium: "구장 정보 조회",
  get_seat_zones: "좌석 구역 조회",
  get_seat_views: "좌석 시야 조회",
  get_ticket_prices: "티켓 가격 조회",
  get_ticket_policies: "예매 정책 조회",
  get_transport: "교통 정보 조회",
  get_food_stores: "매점 정보 조회",
  get_facilities: "편의시설 조회",
  get_stadium_contents: "구장 콘텐츠 조회",
  get_seat_maps: "좌석도 조회",
  get_baseball_schema: "야구 데이터 스키마 조회",
  execute_baseball_select: "야구 기록 조회",
  search_documents_tool: "규칙·안내 문서 검색",
};

export function ChatProgress({ tools }: { tools: ChatToolCall[] }) {
  if (!tools.length) return null;
  return (
    <ul className="chat-progress" role="status" aria-label="도구 호출 로그" aria-live="polite" aria-atomic="false">
      {tools.map(tool => {
        const status = STATUS[tool.status];
        const label = TOOL_LABELS[tool.toolName] ?? "정보 조회";
        return <li key={tool.id} className={`is-${tool.status}`}><span aria-hidden="true">{status.mark}</span><span>{label} {status.label}</span></li>;
      })}
    </ul>
  );
}
