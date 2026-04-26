"""
Core logic for ShiftLog-Gym server to avoid circular imports between FastAPI and Gradio.
"""
from __future__ import annotations
import logging
from ..models import ShiftLogAction
from ..trl_env import ShiftLogToolEnv

logger = logging.getLogger(__name__)

# Shared session state
_session: ShiftLogToolEnv | None = None

def get_session() -> ShiftLogToolEnv:
    """Get or create the session lazily on first use."""
    global _session
    if _session is None:
        _session = ShiftLogToolEnv()
    return _session

def internal_reset(payload: dict | None = None) -> dict:
    """Core logic for resetting the environment."""
    global _session
    payload = payload or {}
    rollout_mode = payload.get("rollout_mode", "short")
    _session = ShiftLogToolEnv(
        rollout_mode=rollout_mode,
        multi_shift=bool(payload.get("multi_shift", False)),
    )
    message = _session.reset(
        seed=payload.get("seed"),
        family=payload.get("family"),
        variant_index=payload.get("variant_index"),
        multi_shift=payload.get("multi_shift", False),
    )
    observation = _session.as_observation()
    observation.message = message or observation.message
    return observation.model_dump()

def internal_step(action_dict: dict) -> dict:
    """Core logic for executing a step."""
    session = get_session()
    tool_name = action_dict.get("tool")
    args = action_dict.get("arguments", {})

    tool = getattr(session, tool_name, None) if tool_name else None
    if tool is None or tool_name.startswith("_"):
        observation = session.as_observation()
        observation.message = f"Unknown tool: {tool_name}"
        return observation.model_dump()

    message = tool(**args)
    observation = session.as_observation()
    observation.message = message
    return observation.model_dump()

def internal_get_state() -> dict:
    """Core logic for getting current state."""
    return get_session().get_info()

def internal_get_tools() -> dict:
    """Core logic for listing tools."""
    return {
        "tools": [
            "read_shift_log", "append_shift_log", "update_shift_log",
            "inspect_service", "inspect_dependency", "run_diagnostic",
            "apply_mitigation", "resolve_incident", "handoff_summary",
        ]
    }
