import json

from django.http import StreamingHttpResponse

"""SSE(Server-Sent Events) 스트리밍 HTTP 응답 관련 로직.

service.chat 은 공개 (event, data) 튜플만 만들고(serializer.message 가 허용 필드만 고른다),
이 모듈이 그걸 text/event-stream StreamingHttpResponse 로 감싸는 HTTP 전송 계층 일을 담당한다.
동기 이터레이터라 WSGI(gunicorn)에서도 프레임마다 바로 전송된다.
"""


def _frame(event, data):
    """SSE 와이어 포맷 한 프레임: event: <name>\ndata: <json>\n\n"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def event_stream_response(events):
    """(event, data) 튜플 이터러블을 SSE StreamingHttpResponse 로 감싼다.

    클라이언트가 끊겨 WSGI 서버가 응답을 close() 하면 events.close() 까지 전달해
    생성 중인 턴을 cancelled 로 정리한다.
    """
    def generate():
        try:
            for event, data in events:
                yield _frame(event, data)
        finally:
            close = getattr(events, "close", None)
            if close:
                close()

    response = StreamingHttpResponse(generate(), content_type="text/event-stream")
    # 첫 순회 전에 close() 되면 generate() 는 시작도 안 해 finally 가 돌지 않는다. primed events 를 직접 닫는다(멱등).
    if hasattr(events, "close"):
        response._resource_closers.append(events.close)
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response
