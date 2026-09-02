#!/usr/bin/env python3
"""Local server for the GAIT next-token wheel.

The browser never talks to Ollama directly. This server serves the single-page
interface, requests one generated position from Ollama's native API, and
renormalizes the returned top-k log probabilities for the visible wheel.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


APP_DIR = Path(__file__).resolve().parent
INDEX_FILE = APP_DIR / "index.html"
DEFAULT_MODEL = "qwen2.5:0.5b-base"
DEFAULT_PORT = 8765
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
MIN_CANDIDATES = 2
MAX_CANDIDATES = 10
DEFAULT_CANDIDATES = 8
MAX_TEXT_CHARACTERS = 8_000
MAX_REQUEST_BYTES = 32_768


class WheelError(Exception):
    """An error that can be safely shown in the local interface."""

    def __init__(self, message: str, status: int = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.message = message
        self.status = int(status)


def display_token(token: str) -> str:
    """Make invisible whitespace visible without changing the raw token."""

    if token == "":
        return "[END]"
    return (
        token.replace("\r\n", "↵")
        .replace("\r", "↵")
        .replace("\n", "↵")
        .replace("\t", "⇥")
        .replace(" ", "␠")
    )


def candidates_from_response(payload: dict[str, Any], requested_k: int) -> list[dict[str, Any]]:
    """Extract and stably renormalize Ollama's first-position top logprobs."""

    logprobs = payload.get("logprobs")
    if not isinstance(logprobs, list) or not logprobs:
        raise WheelError(
            "Ollama did not return log probabilities. Update Ollama and confirm "
            "that this model/backend supports native logprobs.",
            HTTPStatus.BAD_GATEWAY,
        )

    first_position = logprobs[0]
    if not isinstance(first_position, dict):
        raise WheelError("Ollama returned an unexpected logprobs shape.", HTTPStatus.BAD_GATEWAY)

    top = first_position.get("top_logprobs")
    if not isinstance(top, list) or not top:
        raise WheelError(
            "Ollama returned no top-token alternatives.",
            HTTPStatus.BAD_GATEWAY,
        )

    parsed: list[tuple[str, float]] = []
    for item in top[:requested_k]:
        if not isinstance(item, dict):
            raise WheelError("Ollama returned a malformed token candidate.", HTTPStatus.BAD_GATEWAY)

        token = item.get("token")
        logprob = item.get("logprob")
        if not isinstance(token, str) or not isinstance(logprob, (int, float)):
            raise WheelError("Ollama returned a malformed token candidate.", HTTPStatus.BAD_GATEWAY)
        if not math.isfinite(float(logprob)):
            raise WheelError("Ollama returned a non-finite log probability.", HTTPStatus.BAD_GATEWAY)
        if "\ufffd" in token:
            raise WheelError(
                "A displayed token could not be decoded safely as text. Try a simple English stem.",
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        parsed.append((token, float(logprob)))

    if len(parsed) < MIN_CANDIDATES:
        raise WheelError(
            "Ollama returned too few usable token candidates.",
            HTTPStatus.BAD_GATEWAY,
        )

    # Subtracting the maximum leaves the same normalized distribution while
    # preventing numerical underflow for ordinary logprob ranges.
    maximum = max(logprob for _, logprob in parsed)
    weights = [math.exp(logprob - maximum) for _, logprob in parsed]
    total_weight = sum(weights)
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise WheelError("Could not normalize the returned token weights.", HTTPStatus.BAD_GATEWAY)

    candidates: list[dict[str, Any]] = []
    for index, ((token, logprob), weight) in enumerate(zip(parsed, weights, strict=True)):
        candidates.append(
            {
                "id": index,
                "raw": token,
                "display": display_token(token),
                "logprob": logprob,
                "p_displayed": weight / total_weight,
                "terminal": token == "",
            }
        )
    return candidates


def ask_ollama(
    text: str,
    candidate_count: int,
    *,
    model: str,
    ollama_url: str,
    timeout_seconds: float = 60,
) -> list[dict[str, Any]]:
    """Request exactly one next-token position and ignore Ollama's own draw."""

    body = {
        "model": model,
        "prompt": text,
        "raw": True,
        "stream": False,
        "logprobs": True,
        "top_logprobs": candidate_count,
        "options": {"num_predict": 1},
    }
    request = Request(
        f"{ollama_url.rstrip('/')}/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except HTTPError as exc:
        detail = ""
        try:
            error_payload = json.loads(exc.read().decode("utf-8"))
            detail = str(error_payload.get("error", ""))
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            detail = ""
        suffix = f" Ollama says: {detail}" if detail else ""
        raise WheelError(
            f"Ollama rejected the request.{suffix}",
            HTTPStatus.BAD_GATEWAY,
        ) from exc
    except (URLError, TimeoutError, ConnectionError) as exc:
        raise WheelError(
            "Could not reach Ollama at the configured local address. Make sure Ollama is running.",
            HTTPStatus.SERVICE_UNAVAILABLE,
        ) from exc
    except json.JSONDecodeError as exc:
        raise WheelError("Ollama returned invalid JSON.", HTTPStatus.BAD_GATEWAY) from exc

    if not isinstance(payload, dict):
        raise WheelError("Ollama returned an unexpected response.", HTTPStatus.BAD_GATEWAY)
    return candidates_from_response(payload, candidate_count)


def ollama_health(*, model: str, ollama_url: str, timeout_seconds: float = 3) -> dict[str, Any]:
    """Check whether Ollama is reachable and the configured model is present.

    Listing local model tags does not load or run a model.
    """

    request = Request(f"{ollama_url.rstrip('/')}/api/tags", method="GET")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
        return {
            "ollama_connected": False,
            "model_available": False,
            "model": model,
        }

    models = payload.get("models", []) if isinstance(payload, dict) else []
    installed_names: set[str] = set()
    if isinstance(models, list):
        for item in models:
            if not isinstance(item, dict):
                continue
            for key in ("name", "model"):
                value = item.get(key)
                if isinstance(value, str):
                    installed_names.add(value)

    return {
        "ollama_connected": True,
        "model_available": model in installed_names,
        "model": model,
    }


class WheelServer(ThreadingHTTPServer):
    """HTTP server carrying the local Ollama configuration."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], model: str, ollama_url: str):
        super().__init__(address, WheelHandler)
        self.model = model
        self.ollama_url = ollama_url


class WheelHandler(BaseHTTPRequestHandler):
    server: WheelServer

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send_file(INDEX_FILE, "text/html; charset=utf-8")
            return
        if path == "/api/config":
            self._send_json(
                {
                    "model": self.server.model,
                    "default_candidates": DEFAULT_CANDIDATES,
                    "min_candidates": MIN_CANDIDATES,
                    "max_candidates": MAX_CANDIDATES,
                    "semantics": "conditional_top_k",
                }
            )
            return
        if path == "/api/health":
            self._send_json(
                ollama_health(
                    model=self.server.model,
                    ollama_url=self.server.ollama_url,
                )
            )
            return
        if path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        self._send_error("Not found.", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if urlparse(self.path).path != "/api/next":
            self._send_error("Not found.", HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
                raise WheelError("Request body is missing or too large.")
            try:
                request_payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise WheelError("Request body must be valid JSON.") from exc

            if not isinstance(request_payload, dict):
                raise WheelError("Request body must be a JSON object.")
            text = request_payload.get("text")
            requested_k = request_payload.get("k", DEFAULT_CANDIDATES)
            if not isinstance(text, str) or not text:
                raise WheelError("Enter a sentence stem before showing the wheel.")
            if len(text) > MAX_TEXT_CHARACTERS:
                raise WheelError(f"Keep the growing text under {MAX_TEXT_CHARACTERS:,} characters.")
            if isinstance(requested_k, bool) or not isinstance(requested_k, int):
                raise WheelError("Candidate count must be a whole number.")
            if not MIN_CANDIDATES <= requested_k <= MAX_CANDIDATES:
                raise WheelError(
                    f"Candidate count must be between {MIN_CANDIDATES} and {MAX_CANDIDATES}."
                )

            candidates = ask_ollama(
                text,
                requested_k,
                model=self.server.model,
                ollama_url=self.server.ollama_url,
            )
            self._send_json(
                {
                    "model": self.server.model,
                    "requested_k": requested_k,
                    "returned_k": len(candidates),
                    "basis": "conditional_top_k",
                    "notice": (
                        f"Conditional among the displayed top {len(candidates)} tokens. "
                        "Other vocabulary tokens are omitted."
                    ),
                    "candidates": candidates,
                }
            )
        except WheelError as exc:
            self._send_error(exc.message, exc.status)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            content = path.read_bytes()
        except OSError:
            self._send_error("Interface file is missing.", HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def _send_error(self, message: str, status: int) -> None:
        self._send_json({"error": message}, status)

    def log_message(self, format_string: str, *args: Any) -> None:
        # The standard access line is useful while keeping prompt text out of logs.
        sys.stderr.write(f"{self.address_string()} - {format_string % args}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local GAIT next-token wheel.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Local port (default: 8765)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65_535:
        raise SystemExit("Port must be between 1 and 65535.")
    ollama_target = urlparse(args.ollama_url)
    if ollama_target.scheme != "http" or ollama_target.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise SystemExit("Ollama URL must be a loopback-only http:// address.")

    server = WheelServer(("127.0.0.1", args.port), args.model, args.ollama_url)
    print(f"Token Wheel of Fortune: http://127.0.0.1:{args.port}")
    print(f"Model: {args.model}")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
