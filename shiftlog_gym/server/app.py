from __future__ import annotations

from fastapi import FastAPI

from ..models import ShiftLogAction
from ..simulator import ShiftLogSimulator

simulator = ShiftLogSimulator()
app = FastAPI(title="ShiftLog-Gym", version="0.1.0")


@app.get("/")
def root():
    return {
        "name": "ShiftLog-Gym",
        "claim": "A domain-specific professional RL environment for memory management with causal incident dependencies and verifiable outcome rewards.",
        "status": "ok",
    }


@app.post("/reset")
def reset(payload: dict | None = None):
    payload = payload or {}
    message = simulator.reset(
        seed=payload.get("seed"),
        family=payload.get("family"),
        variant_index=payload.get("variant_index"),
    )
    return simulator.as_observation(message).model_dump()


@app.post("/step")
def step(action: ShiftLogAction):
    tool = getattr(simulator, action.tool, None)
    if tool is None or action.tool.startswith("_"):
        return simulator.as_observation(f"Unknown tool: {action.tool}").model_dump()
    message = tool(**action.arguments)
    return simulator.as_observation(message).model_dump()


@app.get("/state")
def state():
    return simulator.get_state().model_dump()


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

