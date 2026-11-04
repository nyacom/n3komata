from __future__ import annotations

import json
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import __version__
from .cli import check_profile, engine_binary, format_args, make_environment, profile_path
from .telemetry import SystemTelemetrySampler

MAX_REQUEST_BYTES = 8 * 1024 * 1024
UI_FILE = Path(__file__).parent / "web" / "index.html"
CSS_FILE = Path(__file__).parent / "web" / "styles.css"
JS_FILE = Path(__file__).parent / "web" / "app.js"
LOGO_FILE = Path(__file__).parent / "web" / "img" / "logo.png"


def unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_engine(process: subprocess.Popen[bytes], base_url: str, timeout: float = 600.0) -> bool:
    deadline = time.monotonic() + timeout
    health = base_url + "/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(health, timeout=2.0) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(1.0)
    return False


def stop_engine(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def serve_frontend(*, root: Path, profile_id: str, profile: dict[str, Any],
                   profiles: dict[str, dict[str, Any]], context: int,
                   max_tokens: int, reasoning: str, host: str, port: int,
                   context_override: bool = False,
                   max_tokens_override: bool = False,
                   reasoning_override: bool = False,
                   dry_run: bool = False) -> int:
    def engine_plan(selected_id: str, selected_profile: dict[str, Any]) -> dict[str, Any]:
        engine_port = unused_loopback_port()
        launch_context = context if selected_id == profile_id or context_override else int(selected_profile.get("default_context", context))
        launch_max_tokens = max_tokens if selected_id == profile_id or max_tokens_override else int(selected_profile.get("default_max_tokens", max_tokens))
        launch_reasoning = reasoning if selected_id == profile_id or reasoning_override else str(selected_profile.get("default_reasoning", reasoning))
        command = [
            str(engine_binary(root, selected_profile, "server")),
            *format_args(selected_profile["server_args"], root=root, profile=selected_profile,
                         context=launch_context, max_tokens=launch_max_tokens, reasoning=launch_reasoning,
                         engine_port=engine_port),
            "--host", "127.0.0.1", "--port", str(engine_port),
        ]
        return {
            "profile_id": selected_id,
            "profile": selected_profile,
            "api_model": str(selected_profile.get("api_model", "n3komata-local")),
            "base_url": f"http://127.0.0.1:{engine_port}",
            "command": command,
            "env": make_environment(selected_profile, root),
        }

    def launch_engine(selected_id: str, selected_profile: dict[str, Any]) -> dict[str, Any]:
        plan = engine_plan(selected_id, selected_profile)
        plan["process"] = subprocess.Popen(plan["command"], cwd=root, env=plan["env"])
        return plan

    selectable_profiles = {
        selected_id: selected_profile
        for selected_id, selected_profile in profiles.items()
        if not check_profile(root, selected_id, selected_profile, "server")
    }
    if profile_id not in selectable_profiles:
        selectable_profiles[profile_id] = profile

    initial_plan = engine_plan(profile_id, profile)
    if dry_run:
        print("Engine command: " + shlex.join(initial_plan["command"]))
    print(f"Engine API (loopback only): {initial_plan['base_url']}")
    print(f"Web UI bind address:        {host}:{port}")
    print(f"Open locally:               http://127.0.0.1:{port}/")
    if host in ("0.0.0.0", "::"):
        print(f"Open from another device:   http://<charmander-IP>:{port}/")
    if dry_run:
        for key, value in profile.get("env", {}).items():
            print(f"{key}={initial_plan['env'].get(key, value)}")
        return 0

    try:
        ui_bytes = UI_FILE.read_bytes()
        ui_bytes = ui_bytes.replace(b"__PROFILE_NAME__", profile.get("name", profile_id).encode("utf-8"))
        ui_bytes = ui_bytes.replace(b"__PROFILE_ID__", profile_id.encode("utf-8"))
        ui_bytes = ui_bytes.replace(b"__APP_VERSION__", __version__.encode("utf-8"))
        reasoning_checked = b"checked" if reasoning != "off" else b""
        ui_bytes = ui_bytes.replace(b"__REASONING_CHECKED__", reasoning_checked)
        css_bytes = CSS_FILE.read_bytes()
        js_bytes = JS_FILE.read_bytes()
        logo_bytes = LOGO_FILE.read_bytes()
    except OSError as exc:
        print(f"n3k: cannot read frontend asset: {exc}", file=sys.stderr)
        return 2

    state_lock = threading.RLock()
    engine_state: dict[str, Any] = {**initial_plan, "switching": False, "active": False, "error": None}
    process = subprocess.Popen(initial_plan["command"], cwd=root, env=initial_plan["env"])
    engine_state["process"] = process
    try:
        print("Loading model and waiting for the engine API…", flush=True)
        if not wait_for_engine(process, initial_plan["base_url"]):
            print(f"n3k: engine did not become ready (exit code: {process.poll()})", file=sys.stderr)
            return process.returncode if process.returncode is not None else 1
        telemetry: SystemTelemetrySampler | None = None

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def send_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def current_engine(self) -> dict[str, Any]:
                with state_lock:
                    return dict(engine_state)

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(ui_bytes)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(ui_bytes)
                    return
                if self.path == "/styles.css":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/css; charset=utf-8")
                    self.send_header("Content-Length", str(len(css_bytes)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(css_bytes)
                    return
                if self.path == "/app.js":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/javascript; charset=utf-8")
                    self.send_header("Content-Length", str(len(js_bytes)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(js_bytes)
                    return
                if self.path == "/img/logo.png":
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(logo_bytes)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(logo_bytes)
                    return
                if self.path == "/api/models":
                    state = self.current_engine()
                    choices = [
                        {"id": key, "name": str(value.get("name", key)),
                         "description": str(value.get("description", "")),
                         "category": str(value.get("profile_channel", "release"))}
                        for key, value in selectable_profiles.items()
                    ]
                    self.send_json(200, {
                        "profiles": choices,
                        "current": state["profile_id"],
                        "switching": state["switching"],
                        "busy": state["active"],
                        "error": state["error"],
                    })
                    return
                if self.path == "/api/health":
                    state = self.current_engine()
                    if state["switching"] or state["process"] is None:
                        self.send_json(503, {"status": "loading", "profile_id": state["profile_id"]})
                        return
                    try:
                        with urllib.request.urlopen(state["base_url"] + "/health", timeout=5) as response:
                            body = response.read(64 * 1024)
                            status = response.status
                    except urllib.error.HTTPError as exc:
                        body, status = exc.read(64 * 1024), exc.code
                    except (urllib.error.URLError, OSError, TimeoutError):
                        body, status = b'{"status":"unavailable"}', 503
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/api/monitor":
                    snapshot = telemetry.snapshot() if telemetry else {"available": False}
                    self.send_json(200, snapshot)
                    return
                if self.path == "/api/inference":
                    state = self.current_engine()
                    if state["switching"] or state["process"] is None:
                        self.send_json(200, {"slots": [], "switching": True})
                        return
                    try:
                        with urllib.request.urlopen(state["base_url"] + "/slots", timeout=2) as response:
                            raw_slots = json.loads(response.read(256 * 1024))
                        if isinstance(raw_slots, dict):
                            raw_slots = raw_slots.get("slots", [])

                        def decoded_count(slot: dict[str, Any]) -> Any:
                            next_token = slot.get("next_token")
                            if isinstance(next_token, list):
                                next_token = next_token[0] if next_token else None
                            if isinstance(next_token, dict):
                                return next_token.get("n_decoded")
                            return slot.get("n_decoded")

                        slots = [
                            {
                                "id": slot.get("id"),
                                "is_processing": bool(slot.get("is_processing")),
                                "n_ctx": slot.get("n_ctx"),
                                "n_prompt_tokens": slot.get("n_prompt_tokens"),
                                "n_prompt_tokens_processed": slot.get("n_prompt_tokens_processed"),
                                "n_prompt_tokens_cache": slot.get("n_prompt_tokens_cache"),
                                "n_decoded": decoded_count(slot),
                            }
                            for slot in raw_slots if isinstance(slot, dict)
                        ] if isinstance(raw_slots, list) else []
                        self.send_json(200, {"slots": slots})
                    except (urllib.error.URLError, OSError, TimeoutError, ValueError, TypeError):
                        self.send_json(503, {"error": "slot telemetry unavailable", "slots": []})
                    return
                self.send_error(404)

            def do_POST(self) -> None:
                if self.path == "/api/models/select":
                    self.select_model()
                    return
                if self.path != "/api/chat/completions":
                    self.send_error(404)
                    return
                try:
                    size = int(self.headers.get("Content-Length", "-1"))
                except ValueError:
                    size = -1
                if size < 0 or size > MAX_REQUEST_BYTES:
                    self.send_error(413 if size > MAX_REQUEST_BYTES else 400, "invalid request size")
                    return
                body = self.rfile.read(size)
                try:
                    parsed = json.loads(body)
                    if not isinstance(parsed, dict):
                        raise ValueError("request must be a JSON object")
                except (ValueError, UnicodeDecodeError, RecursionError):
                    self.send_error(400, "invalid JSON request")
                    return
                with state_lock:
                    if engine_state["switching"] or engine_state["process"] is None:
                        self.send_json(503, {"error": "model is loading"})
                        return
                    if engine_state["active"]:
                        self.send_json(409, {"error": "an inference request is already active"})
                        return
                    engine_state["active"] = True
                    state = dict(engine_state)
                parsed["model"] = state["api_model"]
                parsed["stream"] = True
                body = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
                request = urllib.request.Request(
                    state["base_url"] + "/v1/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(request, timeout=1800) as response:
                        content_type = response.headers.get("Content-Type", "text/event-stream")
                        self.send_response(response.status)
                        self.send_header("Content-Type", content_type)
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("X-Accel-Buffering", "no")
                        self.end_headers()
                        while True:
                            chunk = response.read1(4096)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            self.wfile.flush()
                except urllib.error.HTTPError as exc:
                    error_body = exc.read(1024 * 1024)
                    self.send_response(exc.code)
                    self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
                    self.send_header("Content-Length", str(len(error_body)))
                    self.end_headers()
                    self.wfile.write(error_body)
                except (urllib.error.URLError, OSError, TimeoutError) as exc:
                    try:
                        error_body = json.dumps({"error": f"engine request failed: {type(exc).__name__}"}).encode()
                        self.send_response(502)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(error_body)))
                        self.end_headers()
                        self.wfile.write(error_body)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                finally:
                    with state_lock:
                        engine_state["active"] = False

            def select_model(self) -> None:
                try:
                    size = int(self.headers.get("Content-Length", "-1"))
                except ValueError:
                    size = -1
                if size < 0 or size > 4096:
                    self.send_error(400, "invalid model selection request")
                    return
                try:
                    payload = json.loads(self.rfile.read(size))
                    selected_id = payload.get("profile_id") if isinstance(payload, dict) else None
                except (ValueError, UnicodeDecodeError, RecursionError):
                    self.send_error(400, "invalid JSON request")
                    return
                if not isinstance(selected_id, str) or selected_id not in selectable_profiles:
                    self.send_json(404, {"error": "Model profile is not available"})
                    return
                with state_lock:
                    if engine_state["switching"]:
                        self.send_json(409, {"error": "a model switch is already in progress"})
                        return
                    if engine_state["active"]:
                        self.send_json(409, {"error": "stop the active response before switching models"})
                        return
                    if engine_state["profile_id"] == selected_id:
                        self.send_json(200, {"profile_id": selected_id, "name": engine_state["profile"].get("name", selected_id)})
                        return
                    engine_state["switching"] = True
                    engine_state["error"] = None
                    previous_id = engine_state["profile_id"]
                    previous_profile = engine_state["profile"]
                    previous_process = engine_state["process"]

                stop_engine(previous_process)
                new_state: dict[str, Any] | None = None
                failure = ""
                try:
                    selected_profile = selectable_profiles[selected_id]
                    new_state = launch_engine(selected_id, selected_profile)
                    if not wait_for_engine(new_state["process"], new_state["base_url"]):
                        raise RuntimeError(f"{selected_profile.get('name', selected_id)} did not become ready")
                except (OSError, RuntimeError, KeyError, ValueError) as exc:
                    failure = str(exc)
                    if new_state is not None:
                        stop_engine(new_state.get("process"))
                    new_state = None

                if new_state is not None:
                    with state_lock:
                        engine_state.update(new_state)
                        engine_state["switching"] = False
                        engine_state["active"] = False
                        engine_state["error"] = None
                    self.send_json(200, {"profile_id": selected_id, "name": selectable_profiles[selected_id].get("name", selected_id)})
                    return

                # Restore the previous profile if loading the selected model fails.
                rollback: dict[str, Any] | None = None
                try:
                    rollback = launch_engine(previous_id, previous_profile)
                    if not wait_for_engine(rollback["process"], rollback["base_url"]):
                        raise RuntimeError("previous model did not recover")
                except (OSError, RuntimeError, KeyError, ValueError) as exc:
                    failure = f"{failure}; previous model restart failed: {exc}"
                    if rollback is not None:
                        stop_engine(rollback.get("process"))
                    rollback = None
                with state_lock:
                    if rollback is not None:
                        engine_state.update(rollback)
                    else:
                        engine_state["process"] = None
                    engine_state["switching"] = False
                    engine_state["active"] = False
                    engine_state["error"] = failure
                self.send_json(502, {"error": failure, "profile_id": previous_id})

            def log_message(self, fmt: str, *args: Any) -> None:
                if len(args) >= 2 and str(args[0]).startswith("GET "):
                    try:
                        if int(args[1]) < 400:
                            return
                    except (TypeError, ValueError):
                        pass
                sys.stderr.write("n3k-web: " + fmt % args + "\n")

        httpd = ThreadingHTTPServer((host, port), Handler)
        httpd.daemon_threads = True
        telemetry = SystemTelemetrySampler()
        telemetry.start()
        print("N3komata is ready. Press Ctrl-C to stop both the UI and engine.", flush=True)
        try:
            httpd.serve_forever(poll_interval=0.5)
        finally:
            httpd.server_close()
            if telemetry is not None:
                telemetry.close()
        return 0
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print(f"n3k: cannot start web frontend on {host}:{port}: {exc}", file=sys.stderr)
        return 2
    finally:
        with state_lock:
            active_process = engine_state.get("process")
        stop_engine(active_process)
