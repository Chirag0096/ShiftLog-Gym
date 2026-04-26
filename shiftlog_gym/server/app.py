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

# Lazy-initialize the session on first request to ensure fast startup
_session: ShiftLogToolEnv | None = None

def get_session() -> ShiftLogToolEnv:
    """Get or create the session lazily on first use."""
    global _session
    if _session is None:
        _session = ShiftLogToolEnv()
    return _session


from .core import internal_reset, internal_step, internal_get_state, internal_get_tools

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
async def reset(payload: dict | None = None):
    """Reset the environment to a fresh episode."""
    return internal_reset(payload)


@app.post("/step")
async def step(action: ShiftLogAction):
    """Execute a tool action in the current episode."""
    return internal_step(action.model_dump())


@app.get("/state")
async def state():
    """Return current environment state."""
    return internal_get_state()


@app.get("/tools")
async def tools():
    """Return list of valid tool names."""
    return internal_get_tools()


# ---------------------------------------------------------------------------
# Health Endpoint (Deliverable 4)
# ---------------------------------------------------------------------------

# Global cache for health check
_HEALTH_CACHE: dict | None = None
_HEALTH_CACHE_TS: float = 0
HEALTH_TTL = 60.0  # seconds

@app.get("/health")
async def health():
    """Run health check with caching. Used by Gradio dashboard."""
    global _HEALTH_CACHE, _HEALTH_CACHE_TS
    now = time.time()

    if _HEALTH_CACHE and (now - _HEALTH_CACHE_TS) < HEALTH_TTL:
        return _HEALTH_CACHE

    try:
        from ..diagnostics.health_check import run_full_health_check
        # Pass internal callers to avoid recursive HTTP deadlocks
        result = run_full_health_check(
            base_url="http://localhost:7860",
            call_direct={
                "/reset": internal_reset,
                "/step": internal_step,
                "/state": internal_get_state,
                "/tools": internal_get_tools
            }
        )
        
        # Add background training status if available
        status_file = Path(__file__).resolve().parent.parent / "observatory" / "training_status.txt"
        if status_file.exists():
            result["training_status"] = status_file.read_text(encoding="utf-8").split("\n")[0]
            if "..." in result["training_status"]:
                result["status"] = "training"

        _HEALTH_CACHE = result
        _HEALTH_CACHE_TS = now
        status_code = 200 if result["overall"] != "down" else 503
        return JSONResponse(content=result, status_code=status_code)
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"overall": "down", "error": str(exc)},
        )
