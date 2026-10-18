"""A local stand-in for an OpenAI-compatible endpoint, such as the model gateway.

It records the path, headers and body of every request and answers chat,
Responses and embedding requests with fixed responses, or with a queued error
status, so tests can see exactly what Tau3 sends without contacting a model.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class Received:
    path: str
    headers: dict[str, str]
    body: Any


def _chat(model: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def _response(model: str) -> dict[str, Any]:
    return {
        "id": "resp_fake",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "type": "message",
                "id": "msg_fake",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
    }


def _embeddings(model: str) -> dict[str, Any]:
    return {
        "object": "list",
        "model": model,
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
    }


ROUTES = {
    "/v1/chat/completions": _chat,
    "/v1/responses": _response,
    "/v1/embeddings": _embeddings,
}


class FakeOpenAI:
    def __init__(self) -> None:
        self.received: list[Received] = []
        self._statuses: list[int] = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def __enter__(self) -> FakeOpenAI:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()

    def fail_next(self, status: int) -> None:
        with self._lock:
            self._statuses.append(status)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length") or 0))
                    or b"null"
                )
                with fake._lock:
                    fake.received.append(
                        Received(
                            self.path,
                            {k.lower(): v for k, v in self.headers.items()},
                            body,
                        )
                    )
                    status = fake._statuses.pop(0) if fake._statuses else 200
                if self.path not in ROUTES:
                    status = 404
                payload = (
                    ROUTES[self.path](body.get("model", ""))
                    if status == 200
                    else {"error": {"code": str(status), "message": "fake failure"}}
                )
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler
