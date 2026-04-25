from __future__ import annotations

from ..models import ShiftLogAction, ShiftLogObservation, ShiftLogStateModel
from ..simulator import ShiftLogSimulator


SUPPORTS_CONCURRENT_SESSIONS: bool = True


class ShiftLogEnvironment:
    """Canonical step/reset/state OpenEnv-style environment."""

    def __init__(self) -> None:
        self.simulator = ShiftLogSimulator()

    def reset(self) -> ShiftLogObservation:
        return self.simulator.as_observation(self.simulator.reset())

    def step(self, action: ShiftLogAction) -> tuple[ShiftLogObservation, float, bool]:
        tool = getattr(self.simulator, action.tool)
        message = tool(**action.arguments)
        observation = self.simulator.as_observation(message)
        return observation, observation.reward, observation.done

    def state(self) -> ShiftLogStateModel:
        return self.simulator.get_state()
