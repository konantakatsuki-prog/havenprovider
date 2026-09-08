"""Dependency-free local HTTP server using the same provider contract."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from haven.config import load_settings
from providers.local import LocalProvider


class HavenRequestHandler(BaseHTTPRequestHandler):
    provider = LocalProvider()

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path not in ("/", "/synthesize"):
            self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "message": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload: Any = json.loads(self.rfile.read(length) or b"{}")
            result = self.provider.handle(payload)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "message": str(exc)})
            return

        self._write_json(HTTPStatus.OK, result)

    def _write_json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    settings = load_settings()
    server = ThreadingHTTPServer((settings.http_host, settings.http_port), HavenRequestHandler)
    print(f"Haven local provider listening on http://{settings.http_host}:{settings.http_port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
