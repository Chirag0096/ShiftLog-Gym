from __future__ import annotations

from fastapi import FastAPI

from ..models import ShiftLogAction
from ..trl_env import ShiftLogToolEnv

app = FastAPI(title="ShiftLog-Gym", version="0.1.0")
session = ShiftLogToolEnv()


@app.post("/reset")
def reset(payload: dict | None = None):
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
    return session.get_info()


@app.get("/tools")
def tools():
    return {
        "tools": [
            "read_shift_log",
            "append_shift_log",
            "update_shift_log",
            "inspect_service",
            "inspect_dependency",
            "run_diagnostic",
            "apply_mitigation",
            "resolve_incident",
            "handoff_summary",
        ]
    }
