from __future__ import annotations

from .client import ShiftLogEnv
from .models import ShiftLogAction
from .training import weighted_reward_from_breakdown


class ShiftLogToolEnv:
    """TRL/OpenEnv environment_factory wrapper for Colab GRPO."""

    def __init__(self) -> None:
        self.client = ShiftLogEnv()
        self.reward = 0.0
        self.done = False
        self._last_message = ""

    def reset(self, **kwargs) -> str | None:
        obs = self.client.reset(**kwargs)
        self.reward = 0.0
        self.done = obs.done
        self._last_message = obs.message
        return obs.message

    def _call(self, tool: str, **arguments) -> str:
        if self.done:
            raise ValueError("Episode already completed. Call reset() first.")
        obs = self.client.step(ShiftLogAction(tool=tool, arguments=arguments))
        self.reward = obs.reward
        self.done = obs.done
        self._last_message = obs.message
        return obs.message

    def read_shift_log(self, query: str, limit: int = 5) -> str:
        """
        Search the structured shift log for prior facts relevant to the current incident.

        Args:
            query: Natural-language retrieval query.
            limit: Maximum number of shift-log entries to return.

        Returns:
            Matching structured log entries or a no-match message.
        """
        return self._call("read_shift_log", query=query, limit=limit)

    def append_shift_log(
        self,
        entry_type: str,
        incident_id: str,
        service: str,
        fact: str,
        confidence: float,
    ) -> str:
        """
        Append a new structured memory item to the shift log.

        Args:
            entry_type: Type of memory such as fact, hypothesis, or resolution.
            incident_id: Incident that produced the memory.
            service: Service associated with the memory.
            fact: Fact text to persist.
            confidence: Confidence score between 0 and 1.

        Returns:
            A confirmation or validation message.
        """
        return self._call(
            "append_shift_log",
            entry_type=entry_type,
            incident_id=incident_id,
            service=service,
            fact=fact,
            confidence=confidence,
        )

    def update_shift_log(self, memory_id: str, patch: str, reason: str) -> str:
        """
        Update an existing mutable memory item in the shift log.

        Args:
            memory_id: Memory entry identifier.
            patch: Replacement fact text.
            reason: Reason for the update.

        Returns:
            A confirmation or validation message.
        """
        return self._call("update_shift_log", memory_id=memory_id, patch=patch, reason=reason)

    def inspect_service(self, service: str) -> str:
        """
        Inspect a service and return operational context.

        Args:
            service: Service to inspect.

        Returns:
            A service health and context summary.
        """
        return self._call("inspect_service", service=service)

    def inspect_dependency(self, service: str) -> str:
        """
        Inspect dependency relationships for a service.

        Args:
            service: Service whose dependency graph should be inspected.

        Returns:
            Dependency summary text.
        """
        return self._call("inspect_dependency", service=service)

    def run_diagnostic(self, service: str, diagnostic: str) -> str:
        """
        Run a named diagnostic against a service.

        Args:
            service: Service to diagnose.
            diagnostic: Diagnostic name such as connections or deploys.

        Returns:
            Diagnostic output.
        """
        return self._call("run_diagnostic", service=service, diagnostic=diagnostic)

    def apply_mitigation(self, service: str, mitigation: str) -> str:
        """
        Apply a mitigation to a service.

        Args:
            service: Service to mitigate.
            mitigation: Mitigation identifier.

        Returns:
            Mitigation result text.
        """
        return self._call("apply_mitigation", service=service, mitigation=mitigation)

    def resolve_incident(self, incident_id: str, resolution: str, root_cause: str) -> str:
        """
        Resolve the active incident with a structured root-cause statement.

        Args:
            incident_id: Active incident identifier.
            resolution: Proposed machine-checkable resolution action.
            root_cause: Root cause statement.

        Returns:
            Resolution outcome text.
        """
        return self._call("resolve_incident", incident_id=incident_id, resolution=resolution, root_cause=root_cause)

    def handoff_summary(self) -> str:
        """
        Generate a compact end-of-shift artifact summarizing outcomes and high-signal memories.

        Returns:
            End-of-shift handoff summary.
        """
        return self._call("handoff_summary")

    def _reward_breakdown(self) -> dict[str, float]:
        return self.client.state().reward_breakdown

    def _weighted_reward(self) -> float:
        return weighted_reward_from_breakdown(self._reward_breakdown())


def reward_total(environments, **kwargs):
    return [env._weighted_reward() for env in environments]


def reward_recall(environments, **kwargs):
    return [0.35 * env._reward_breakdown().get("R_recall", 0.0) for env in environments]


def reward_success(environments, **kwargs):
    return [env._reward_breakdown().get("R_success", 0.0) for env in environments]


def reward_memory_write(environments, **kwargs):
    return [0.15 * env._reward_breakdown().get("R_memory_write", 0.0) for env in environments]


def reward_memory_integrity(environments, **kwargs):
    return [env._reward_breakdown().get("R_memory_integrity", 0.0) for env in environments]


def reward_efficiency(environments, **kwargs):
    return [0.1 * env._reward_breakdown().get("R_efficiency", 0.0) for env in environments]


def reward_hallucination(environments, **kwargs):
    return [env._reward_breakdown().get("R_hallucination", 0.0) for env in environments]
