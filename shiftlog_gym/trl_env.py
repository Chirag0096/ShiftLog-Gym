from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from .models import ShiftLogObservation
from .rewards import DEFAULT_RUBRIC
from .simulator import MultiShiftEpisode, ShiftLogSimulator


class ShiftLogToolEnv:
    """TRL/OpenEnv-compatible wrapper around the local simulator."""

    def __init__(self, rollout_mode: str = "short", multi_shift: bool = False) -> None:
        if rollout_mode not in {"short", "full"}:
            raise ValueError("rollout_mode must be 'short' or 'full'.")
        self.rollout_mode = rollout_mode
        self.multi_shift = multi_shift
        self.max_tool_calls = 15 if rollout_mode == "short" else 40
        self.simulator = ShiftLogSimulator()
        self.multi_shift_episode = MultiShiftEpisode()
        self.reward = 0.0
        self.done = False
        self._last_message = ""
        self._last_info: dict[str, Any] = {}

    def reset(self, **kwargs) -> str | None:
        use_multi_shift = bool(kwargs.pop("multi_shift", self.multi_shift))
        self.multi_shift = use_multi_shift
        if use_multi_shift:
            observation = self.multi_shift_episode.reset(**kwargs)
            incident_summary = observation.message.split("Recent shift-log snippets:")[0].strip()
            shift_start_time = self.multi_shift_episode.shift_2.shift_start_time.isoformat()
            self.reward = observation.reward
            self.done = observation.done
            self._last_message = self._inject_system_prompt(shift_start_time, incident_summary, observation.message)
            self._last_info = self.get_info()
            return self._last_message

        message = self.simulator.reset(**kwargs)
        incident_summary = self.simulator.last_observation.split("Recent shift-log snippets:")[0].strip()
        shift_start_time = self.simulator.shift_start_time.isoformat()
        self.reward = self.simulator.total_reward
        self.done = self.simulator.done
        self._last_message = self._inject_system_prompt(shift_start_time, incident_summary, message)
        self._last_info = self.get_info()
        return self._last_message

    def _call(self, tool: str, **arguments) -> str:
        if self.done:
            raise ValueError("Episode already completed. Call reset() first.")
        if self.multi_shift:
            observation, reward, done, info = self.multi_shift_episode.step({"tool": tool, "arguments": arguments})
            self.reward = reward
            self.done = done
            self._last_message = observation.message
            self._last_info = info
            return observation.message

        handler = getattr(self.simulator, tool)
        message = handler(**arguments)
        if self.simulator.episode_state and self.simulator.episode_state.step_count >= self.max_tool_calls and not self.simulator.done:
            self.simulator.done = True
            self.simulator.episode_state.is_complete = True
            self.simulator.last_observation = "Rollout budget reached."
        self.reward = self.simulator.total_reward
        self.done = self.simulator.done
        self._last_message = message
        self._last_info = self.get_info()
        return message

    def read_shift_log(self, query: str, limit: int = 5) -> str:
        return self._call("read_shift_log", query=query, limit=limit)

    def append_shift_log(
        self,
        entry_type: str,
        incident_id: str,
        service: str,
        fact: str,
        confidence: float,
    ) -> str:
        return self._call(
            "append_shift_log",
            entry_type=entry_type,
            incident_id=incident_id,
            service=service,
            fact=fact,
            confidence=confidence,
        )

    def update_shift_log(self, memory_id: str, patch: str, reason: str) -> str:
        return self._call("update_shift_log", memory_id=memory_id, patch=patch, reason=reason)

    def inspect_service(self, service: str) -> str:
        return self._call("inspect_service", service=service)

    def inspect_dependency(self, service: str) -> str:
        return self._call("inspect_dependency", service=service)

    def run_diagnostic(self, service: str, diagnostic: str) -> str:
        return self._call("run_diagnostic", service=service, diagnostic=diagnostic)

    def apply_mitigation(self, service: str, mitigation: str) -> str:
        return self._call("apply_mitigation", service=service, mitigation=mitigation)

    def resolve_incident(self, incident_id: str, resolution: str, root_cause: str) -> str:
        return self._call("resolve_incident", incident_id=incident_id, resolution=resolution, root_cause=root_cause)

    def handoff_summary(self) -> str:
        return self._call("handoff_summary")

    def get_info(self) -> dict[str, Any]:
        if self.multi_shift:
            return self.multi_shift_episode.get_info()

        episode_state = self.simulator.episode_state
        subscores = self.simulator.rubric_subscores()
        return {
            "shift_id": self.simulator.shift_id,
            "scenario_family": self.simulator.scenario.family if self.simulator.scenario else None,
            "rollout_mode": self.rollout_mode,
            "reward_success": subscores.get("success", 0.0),
            "reward_recall": subscores.get("recall_before_action", 0.0),
            "reward_memory_write": subscores.get("memory_write_quality", 0.0),
            "reward_memory_integrity": subscores.get("memory_integrity", 0.0),
            "reward_efficiency": subscores.get("efficiency", 0.0),
            "reward_hallucination": subscores.get("hallucination", 0.0),
            "reward_noise_resistance": subscores.get("noise_resistance", 0.0),
            "reward_handoff": subscores.get("handoff_quality", 0.0),
            "reward_total": DEFAULT_RUBRIC.score(episode_state) if episode_state else 0.0,
            "episode_state": _json_safe(asdict(episode_state) if episode_state else {}),
        }

    def as_observation(self) -> ShiftLogObservation:
        return ShiftLogObservation(
            message=self._last_message,
            current_incident_id=self.simulator.active_incident.incident_id if not self.multi_shift and self.simulator.active_incident else None,
            reward=self.reward,
            reward_breakdown={
                "reward_success": self._last_info.get("reward_success", 0.0),
                "reward_recall": self._last_info.get("reward_recall", 0.0),
                "reward_memory_write": self._last_info.get("reward_memory_write", 0.0),
                "reward_memory_integrity": self._last_info.get("reward_memory_integrity", 0.0),
                "reward_efficiency": self._last_info.get("reward_efficiency", 0.0),
                "reward_hallucination": self._last_info.get("reward_hallucination", 0.0),
                "reward_noise_resistance": self._last_info.get("reward_noise_resistance", 0.0),
                "reward_handoff": self._last_info.get("reward_handoff", 0.0),
                "reward_total": self._last_info.get("reward_total", self.reward),
            },
            done=self.done,
            metadata=self._last_info,
        )

    def _inject_system_prompt(self, shift_start_time: str, incident_summary: str, message: str) -> str:
        return (
            "You are an on-call SRE agent. You have access to a shift log tool. "
            f"ALWAYS read the shift log before applying any mitigation to an incident. "
            f"Your shift started at {shift_start_time}. Current incidents: {incident_summary}. "
            "Use your tools to investigate, log findings, and resolve each incident.\n\n"
            f"{message}"
        )


def reward_total(environments, **kwargs):
    return [env.get_info().get("reward_total", 0.0) for env in environments]


def reward_success(environments, **kwargs):
    return [env.get_info().get("reward_success", 0.0) for env in environments]


def reward_recall(environments, **kwargs):
    return [env.get_info().get("reward_recall", 0.0) for env in environments]


def reward_memory_write(environments, **kwargs):
    return [env.get_info().get("reward_memory_write", 0.0) for env in environments]


def reward_memory_integrity(environments, **kwargs):
    return [env.get_info().get("reward_memory_integrity", 0.0) for env in environments]


def reward_efficiency(environments, **kwargs):
    return [env.get_info().get("reward_efficiency", 0.0) for env in environments]


def reward_hallucination(environments, **kwargs):
    return [env.get_info().get("reward_hallucination", 0.0) for env in environments]


def reward_noise_resistance(environments, **kwargs):
    return [env.get_info().get("reward_noise_resistance", 0.0) for env in environments]


def reward_handoff(environments, **kwargs):
    return [env.get_info().get("reward_handoff", 0.0) for env in environments]


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
