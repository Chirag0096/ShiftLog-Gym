"""
Error-resilient middleware for the ShiftLog-Gym FastAPI server.

Provides:
- Structured JSON error responses (no Python tracebacks)
- Request logging with rotation (logs/runtime.log)
- /step request body validation
- X-Response-Time response header
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import time
from pathlib import Path
from typing import Any, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "runtime.log"

VALID_TOOLS: frozenset[str] = frozenset([
    "read_shift_log", "append_shift_log", "update_shift_log",
    "inspect_service", "inspect_dependency", "run_diagnostic",
    "apply_mitigation", "resolve_incident", "handoff_summary",
])

# ---------------------------------------------------------------------------
# Logger setup with rotation
# ---------------------------------------------------------------------------
_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

runtime_logger = logging.getLogger("shiftlog.runtime")
runtime_logger.setLevel(logging.INFO)
runtime_logger.addHandler(_handler)
runtime_logger.propagate = False


# ---------------------------------------------------------------------------
# Middleware class
# ---------------------------------------------------------------------------
class ShiftLogMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that:
    - Times every request and adds X-Response-Time header.
    - Logs every request to logs/runtime.log.
    - Catches all unhandled exceptions and returns structured JSON.
    - Validates /step request bodies before they reach the route.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.monotonic()

        # --- Pre-validate /step ---
        if request.url.path == "/step" and request.method == "POST":
            validation_error = await self._validate_step(request)
            if validation_error:
                elapsed = round((time.monotonic() - start) * 1000)
                runtime_logger.warning(
                    "POST /step 422 %dms — validation: %s", elapsed, validation_error["detail"]
                )
                resp = JSONResponse(status_code=422, content=validation_error)
                resp.headers["X-Response-Time"] = f"{elapsed}ms"
                return resp

        # --- Process request ---
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001
            elapsed = round((time.monotonic() - start) * 1000)
            runtime_logger.error(
                "UNHANDLED %s %s — %s", request.method, request.url.path, exc, exc_info=True
            )
            resp = JSONResponse(
                status_code=500,
                content={
                    "error": "InternalServerError",
                    "detail": str(exc),
                    "path": request.url.path,
                },
            )
            resp.headers["X-Response-Time"] = f"{elapsed}ms"
            return resp

        elapsed = round((time.monotonic() - start) * 1000)
        response.headers["X-Response-Time"] = f"{elapsed}ms"

        runtime_logger.info(
            "%s %s %d %dms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response

    async def _validate_step(self, request: Request) -> dict[str, Any] | None:
        """
        Validate the /step request body.

        Returns None if valid, or a structured error dict if invalid.
        """
        try:
            body_bytes = await request.body()
            body = json.loads(body_bytes)
        except Exception:
            return {
                "error": "InvalidRequestBody",
                "detail": "Request body must be valid JSON with keys 'tool' and 'arguments'.",
            }

        if not isinstance(body, dict):
            return {
                "error": "InvalidRequestBody",
                "detail": "Request body must be a JSON object.",
            }

        # Validate 'tool' key
        tool = body.get("tool")
        if tool is None:
            return {"error": "MissingTool", "detail": "Missing required key 'tool'."}
        if not isinstance(tool, str):
            return {"error": "InvalidTool", "detail": "'tool' must be a string."}
        if tool.startswith("_"):
            return {"error": "InvalidTool", "detail": f"Tool '{tool}' is private."}
        if tool not in VALID_TOOLS:
            return {
                "error": "InvalidTool",
                "detail": (
                    f"Tool '{tool}' not found. "
                    f"Valid tools: {', '.join(sorted(VALID_TOOLS))}."
                ),
            }

        # Validate 'arguments' key
        args = body.get("arguments", body.get("args"))
        if args is not None and not isinstance(args, dict):
            return {"error": "InvalidArguments", "detail": "'arguments' must be a JSON object."}

        return None  # valid


def setup_middleware(app: Any) -> None:
    """Attach ShiftLogMiddleware to a FastAPI app instance."""
    app.add_middleware(ShiftLogMiddleware)
