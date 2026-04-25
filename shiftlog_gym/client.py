from __future__ import annotations

from typing import Any

from .models import ShiftLogAction, ShiftLogObservation
from .simulator import ShiftLogSimulator


class ShiftLogEnv:
    """Minimal local client wrapper.

    This keeps the repo usable even before wiring a hosted OpenEnv Space.
    """

    def __init__(self, base_url: str | None = None, simulator: ShiftLogSimulator | None = None) -> None:
        self.base_url = base_url
        self.simulator = simulator or ShiftLogSimulator()

    def reset(self, **kwargs: Any) -> ShiftLogObservation:
        message = self.simulator.reset(
            seed=kwargs.get("seed"),
            family=kwargs.get("family"),
            variant_index=kwargs.get("variant_index"),
        )
        return self.simulator.as_observation(message)

    def step(self, action: ShiftLogAction) -> ShiftLogObservation:
        tool = getattr(self.simulator, action.tool, None)
        if tool is None or action.tool.startswith("_"):
            return self.simulator.as_observation(f"Unknown tool: {action.tool}")
        message = tool(**action.arguments)
        return self.simulator.as_observation(message)

    def state(self):
        return self.simulator.get_state()

