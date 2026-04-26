"""ShiftLog-Gym FastAPI server — with middleware, health check, and /health endpoint."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ..models import ShiftLogAction
from ..trl_env import ShiftLogToolEnv
from .middleware import setup_middleware

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ShiftLog-Gym",
    version="0.2.0",
    description="OpenEnv-compliant SRE incident-memory training environment.",
)

# Attach error-resilient middleware
setup_middleware(app)

session = ShiftLogToolEnv()


# ---------------------------------------------------------------------------
# Startup — skip heavy health check to prevent 30-min timeout in Spaces
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    """Fast startup event — avoid calling expensive health checks that can timeout."""
    logger.info("✅ ShiftLog-Gym API server started. Use /health endpoint for status checks.")


# ---------------------------------------------------------------------------
# Core Environment Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset")
def reset(payload: dict | None = None):
    """Reset the environment to a fresh episode."""
    global session
    payload = payload or {}
    rollout_mode = payload.get("rollout_mode", "short")
    session = ShiftLogToolEnv(
        rollout_mode=rollout_mode,
        multi_shift=bool(payload.get("multi_shift", False)),
    )
    message = session.reset(
        seed=payload.get("seed"),
        family=payload.get("family"),
        variant_index=payload.get("variant_index"),
        multi_shift=payload.get("multi_shift", False),
    )
    observation = session.as_observation()
    observation.message = message or observation.message
    return observation.model_dump()


@app.post("/step")
def step(action: ShiftLogAction):
    """Execute a tool action in the current episode."""
    tool = getattr(session, action.tool, None)
    if tool is None or action.tool.startswith("_"):
        observation = session.as_observation()
        observation.message = f"Unknown tool: {action.tool}"
        return observation.model_dump()
    message = tool(**action.arguments)
    observation = session.as_observation()
    observation.message = message
    return observation.model_dump()


@app.get("/state")
def state():
    """Return current environment state."""
    return session.get_info()


@app.get("/tools")
def tools():
    """Return list of valid tool names."""
    return {
        "tools": [
            "read_shift_log", "append_shift_log", "update_shift_log",
            "inspect_service", "inspect_dependency", "run_diagnostic",
            "apply_mitigation", "resolve_incident", "handoff_summary",
        ]
    }


# ---------------------------------------------------------------------------
# Health Endpoint (Deliverable 4)
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Run full health check and return JSON status. Used by Gradio dashboard."""
    try:
        from ..diagnostics.health_check import run_full_health_check
        result = run_full_health_check(base_url="http://localhost:7860")
        status_code = 200 if result["overall"] != "down" else 503
        return JSONResponse(content=result, status_code=status_code)
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"overall": "down", "error": str(exc)},
        )
