"""가짜 OpenAI 호환 VLM 서버 — 모델 없이 엔진 훅·오류 처리를 시험한다."""
from __future__ import annotations

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image


def default_respond(query: str, image: Image.Image):
    if query.startswith("Table"):
        return 200, "<fcel>구분<fcel>금액<nl><fcel>인건비<fcel>1,500<nl>", "stop"
    return 200, f"글자 {image.size[0]}x{image.size[1]}", "stop"


class FakeVLM:
    """respond(query, image) → (HTTP 코드, 글자, finish_reason)."""

    def __init__(self, respond=default_respond, model: str = "served-name"):
        self.respond = respond
        self.model = model
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send(200, {"object": "list", "data": [{"id": outer.model, "object": "model"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                content = body["messages"][0]["content"]
                url, query = content[0]["image_url"]["url"], content[1]["text"]
                image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
                outer.requests.append({"query": query, "size": image.size, "model": body["model"],
                                       "max_completion_tokens": body.get("max_completion_tokens")})
                code, text, finish = outer.respond(query, image)
                if code != 200:
                    self._send(code, {"error": {"message": text}})
                    return
                self._send(200, {
                    "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": finish,
                                 "message": {"role": "assistant", "content": text}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                })

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
