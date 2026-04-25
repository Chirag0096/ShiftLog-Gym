from __future__ import annotations

from dataclasses import asdict
from random import Random
from typing import Any

from .domain import EpisodeMetrics, IncidentDefinition, MemoryEntry, RewardKey
from .models import ShiftLogObservation, ShiftLogStateModel
from .scenarios import FAMILIES, build_scenario_library


REWARD_DEFAULTS: dict[RewardKey, float] = {
    "R_success": 0.0,
    "R_recall": 0.0,
    "R_memory_write": 0.0,
    "R_memory_integrity": 0.0,
    "R_efficiency": 0.0,
    "R_hallucination": 0.0,
}


class ShiftLogSimulator:
    """Pure simulator core for ShiftLog-Gym."""

    def __init__(self, variants_per_family: int = 8) -> None:
        self.library = build_scenario_library(variants_per_family=variants_per_family)
        self.variants_per_family = variants_per_family
        self.random = Random(0)
        self.shift_id = "shift-0000"
        self.memory_entries: list[MemoryEntry] = []
        self.incidents: list[IncidentDefinition] = []
        self.current_index = 0
        self.total_reward = 0.0
        self.reward_breakdown = REWARD_DEFAULTS.copy()
        self.metrics = EpisodeMetrics()
        self._memory_counter = 0
        self._current_relevant_reads: set[str] = set()
        self._acted_on_current_incident = False
        self.done = False
        self.last_observation = ""

    @property
    def active_incident(self) -> IncidentDefinition | None:
        if self.done or not self.incidents or self.current_index >= len(self.incidents):
            return None
        return self.incidents[self.current_index]

    def reset(self, *, seed: int | None = None, family: str | None = None, variant_index: int | None = None) -> str:
        if seed is not None:
            self.random.seed(seed)
        family_name = family or self.random.choice(FAMILIES)
        variants = self.library[family_name]
        scenario = variants[variant_index % len(variants)] if variant_index is not None else self.random.choice(variants)
        self.incidents = scenario
        self.current_index = 0
        self.done = False
        self.shift_id = f"{family_name}-{scenario[0].variant_id}"
        self.memory_entries = []
        self._memory_counter = 0
        self.total_reward = 0.0
        self.reward_breakdown = REWARD_DEFAULTS.copy()
        self.metrics = EpisodeMetrics()
        self._current_relevant_reads = set()
        self._acted_on_current_incident = False
        self.last_observation = self._render_current_incident()
        return self.last_observation

    def _record_reward(self, key: RewardKey, delta: float) -> None:
        self.reward_breakdown[key] += delta
        self.total_reward += delta

    def _record_tool(self, tool: str, detail: dict[str, Any] | None = None) -> None:
        incident = self.active_incident
        incident_id = incident.incident_id if incident else None
        self.metrics.tool_timeline.append(
            {
                "shift_id": self.shift_id,
                "incident_id": incident_id,
                "step_index": len(self.metrics.tool_timeline),
                "tool": tool,
                "detail": {
                    **(detail or {}),
                    "linked_incident": bool(incident.linked_to) if incident else False,
                },
            }
        )
        self._record_reward("R_efficiency", -0.02)

    def _render_current_incident(self) -> str:
        incident = self.active_incident
        if incident is None:
            return "Shift complete. No active incidents."
        log_snippets = "\n".join(f"- {entry.fact}" for entry in self.memory_entries[-3:]) or "- No prior shift-log snippets in view."
        symptoms = "\n".join(f"- {symptom}" for symptom in incident.symptoms)
        deps = ", ".join(incident.dependencies)
        return (
            f"Shift: {self.shift_id}\n"
            f"Incident: {incident.incident_id} ({incident.family})\n"
            f"Service: {incident.service}\n"
            f"Summary: {incident.summary}\n"
            f"Customer impact: {incident.customer_impact}\n"
            f"Dependencies: {deps}\n"
            f"Symptoms:\n{symptoms}\n"
            f"Recent shift-log snippets:\n{log_snippets}\n"
            f"Runbook hint: {incident.runbook}\n"
        )

    def _next_incident(self) -> None:
        self.current_index += 1
        self._current_relevant_reads = set()
        self._acted_on_current_incident = False
        if self.current_index >= len(self.incidents):
            self.done = True
            self.last_observation = "Shift complete. Generate handoff_summary() for final artifact."
        else:
            self.last_observation = self._render_current_incident()

    def read_shift_log(self, query: str, limit: int = 5) -> str:
        """Search the structured shift log for prior facts."""
        self._record_tool("read_shift_log", {"query": query, "limit": limit})
        incident = self.active_incident
        q = query.lower()
        matches: list[MemoryEntry] = []
        for entry in self.memory_entries:
            haystack = f"{entry.service} {entry.incident_id} {entry.fact} {' '.join(entry.notes)}".lower()
            if any(term in haystack for term in q.split()) or q in haystack:
                matches.append(entry)
        if incident is not None:
            for entry in self.memory_entries:
                if entry.fact_key and entry.fact_key in incident.required_memory_keys and entry not in matches:
                    if any(term.lower() in q for term in incident.relevant_memory_terms):
                        matches.append(entry)
        matches = matches[: max(1, limit)]
        if incident is not None and not self._acted_on_current_incident:
            required = set(incident.required_memory_keys)
            found = {entry.fact_key for entry in matches if entry.fact_key}
            relevant = required.intersection(found)
            if relevant:
                self._current_relevant_reads.update(relevant)
                if incident.linked_to:
                    self.metrics.recall_linked_success += 1
                self._record_reward("R_recall", 0.5)
        if incident is not None and incident.linked_to:
            self.metrics.recall_linked_total += 1
        if not matches:
            return "No relevant shift-log entries matched the query."
        lines = []
        for entry in matches:
            lines.append(
                f"[{entry.memory_id}] service={entry.service} incident={entry.incident_id} "
                f"confidence={entry.confidence:.2f} contradiction={entry.contradiction} fact={entry.fact}"
            )
        return "\n".join(lines)

    def append_shift_log(
        self,
        entry_type: str,
        incident_id: str,
        service: str,
        fact: str,
        confidence: float,
    ) -> str:
        """Append a new structured memory record."""
        self._record_tool("append_shift_log", {"entry_type": entry_type, "incident_id": incident_id, "service": service})
        fact = fact.strip()
        if not fact or not incident_id or not service:
            self.metrics.bad_write_count += 1
            self._record_reward("R_memory_integrity", -0.7)
            return "Invalid shift-log write: entry_type, incident_id, service, and fact are required."
        duplicate = next((entry for entry in self.memory_entries if entry.fact.lower() == fact.lower() and entry.service == service), None)
        contradiction = self._detect_contradiction(service=service, fact=fact)
        self._memory_counter += 1
        entry = MemoryEntry(
            memory_id=f"mem-{self._memory_counter:03d}",
            entry_type=entry_type,
            incident_id=incident_id,
            service=service,
            fact=fact,
            confidence=max(0.0, min(confidence, 1.0)),
            timestamp=len(self.metrics.tool_timeline),
            duplicate_of=duplicate.memory_id if duplicate else None,
            contradiction=contradiction,
            fact_key=self._match_fact_key(fact),
        )
        if contradiction:
            entry.notes.append("contradiction-detected")
            self.metrics.contradiction_count += 1
            self._record_reward("R_memory_integrity", -1.0)
        elif duplicate is not None:
            entry.notes.append(f"duplicate-of:{duplicate.memory_id}")
            self.metrics.bad_write_count += 1
            self._record_reward("R_memory_integrity", -0.4)
        else:
            if entry.fact_key is not None:
                self._record_reward("R_memory_write", 0.3)
            else:
                self._record_reward("R_memory_write", 0.1)
        self.memory_entries.append(entry)
        self.metrics.memory_events.append({"event": "append", **asdict(entry)})
        return f"Appended {entry.memory_id}."

    def update_shift_log(self, memory_id: str, patch: str, reason: str) -> str:
        """Update an existing non-immutable memory record."""
        self._record_tool("update_shift_log", {"memory_id": memory_id, "reason": reason})
        entry = next((item for item in self.memory_entries if item.memory_id == memory_id), None)
        if entry is None:
            self._record_reward("R_memory_integrity", -0.5)
            return f"Memory entry {memory_id} not found."
        if entry.immutable:
            self._record_reward("R_memory_integrity", -0.6)
            return f"Memory entry {memory_id} is immutable."
        old_fact = entry.fact
        entry.fact = patch.strip()
        entry.notes.append(f"updated:{reason}")
        entry.contradiction = self._detect_contradiction(service=entry.service, fact=entry.fact, ignore_memory_id=memory_id)
        if entry.contradiction:
            self.metrics.contradiction_count += 1
            self._record_reward("R_memory_integrity", -0.8)
        else:
            self._record_reward("R_memory_write", 0.05)
        self.metrics.memory_events.append(
            {
                "event": "update",
                "memory_id": memory_id,
                "old_fact": old_fact,
                "new_fact": entry.fact,
                "reason": reason,
            }
        )
        return f"Updated {memory_id}."

    def _detect_contradiction(self, *, service: str, fact: str, ignore_memory_id: str | None = None) -> bool:
        negations = ("not ", "never ", "no ", "disabled", "healthy")
        positive = ("stale", "enabled", "saturated", "timeout", "oom", "deprecated")
        lower = fact.lower()
        for entry in self.memory_entries:
            if ignore_memory_id and entry.memory_id == ignore_memory_id:
                continue
            if entry.service != service:
                continue
            if any(token in lower for token in negations) and any(token in entry.fact.lower() for token in positive):
                return True
            if any(token in lower for token in positive) and any(token in entry.fact.lower() for token in negations):
                return True
        return False

    def _match_fact_key(self, fact: str) -> str | None:
        lower = fact.lower()
        incident = self.active_incident
        if incident is None:
            return None
        for fact_key, exemplar in incident.golden_memory:
            tokens = [token for token in exemplar.lower().replace(".", "").split() if len(token) > 4]
            overlap = sum(1 for token in tokens if token in lower)
            if overlap >= max(1, min(2, len(tokens))):
                return fact_key
        return None

    def inspect_service(self, service: str) -> str:
        """Inspect a service and return operational hints."""
        self._record_tool("inspect_service", {"service": service})
        incident = self.active_incident
        if incident and service == incident.service:
            return f"{service}: owner={incident.owner}; runbook={incident.runbook}; symptoms={'; '.join(incident.symptoms)}"
        return f"{service}: no active anomaly beyond dependency noise."

    def inspect_dependency(self, service: str) -> str:
        """Inspect upstream and downstream service dependencies."""
        self._record_tool("inspect_dependency", {"service": service})
        incident = self.active_incident
        if incident and service == incident.service:
            return f"{service}: dependencies => {', '.join(incident.dependencies)}"
        return f"{service}: dependency graph stable."

    def run_diagnostic(self, service: str, diagnostic: str) -> str:
        """Run a diagnostic command against a service."""
        self._record_tool("run_diagnostic", {"service": service, "diagnostic": diagnostic})
        self._acted_on_current_incident = True
        incident = self.active_incident
        if incident is None:
            return "No active incident."
        result = incident.diagnostics.get(diagnostic)
        if result is None:
            self._record_reward("R_hallucination", -0.15)
            return f"Diagnostic {diagnostic} returned no useful signal."
        return result

    def apply_mitigation(self, service: str, mitigation: str) -> str:
        """Apply a mitigation to a service."""
        self._record_tool("apply_mitigation", {"service": service, "mitigation": mitigation})
        self._acted_on_current_incident = True
        incident = self.active_incident
        if incident is None:
            return "No active incident."
        if service != incident.service:
            self._record_reward("R_hallucination", -0.3)
            return f"Mitigation applied to {service}, but the active incident is on {incident.service}."
        if mitigation in incident.valid_mitigations:
            return f"Mitigation {mitigation} applied cleanly to {service}."
        self._record_reward("R_hallucination", -0.25)
        return f"Mitigation {mitigation} had no meaningful effect."

    def resolve_incident(self, incident_id: str, resolution: str, root_cause: str) -> str:
        """Resolve the active incident with a machine-checkable root cause."""
        self._record_tool("resolve_incident", {"incident_id": incident_id, "resolution": resolution})
        self._acted_on_current_incident = True
        incident = self.active_incident
        if incident is None:
            return "No active incident."
        if incident.incident_id != incident_id:
            self._record_reward("R_hallucination", -0.8)
            return f"Active incident is {incident.incident_id}, not {incident_id}."
        if resolution in incident.unsupported_resolutions:
            self.metrics.fabricated_resolution_count += 1
            self.metrics.unresolved_incident_ids.append(incident_id)
            self._record_reward("R_hallucination", -1.2)
            self._next_incident()
            return f"Resolution {resolution} was unsupported and did not solve {incident_id}."

        resolution_ok = resolution == incident.resolution
        cause_ok = root_cause.strip().lower() == incident.root_cause.lower()
        required = set(incident.required_memory_keys)
        recall_ok = not required or required.issubset(self._current_relevant_reads)

        if incident.linked_to:
            self.metrics.linked_total += 1

        if resolution_ok and cause_ok:
            reward = 2.0
            if incident.linked_to and recall_ok:
                reward += 1.5
                self.metrics.linked_success += 1
            self._record_reward("R_success", reward)
            status = f"Resolved {incident_id} successfully."
        else:
            self.metrics.unresolved_incident_ids.append(incident_id)
            self._record_reward("R_hallucination", -1.0)
            status = (
                f"Resolution failed for {incident_id}. Expected resolution={incident.resolution} "
                f"and root_cause={incident.root_cause}."
            )
        self._next_incident()
        return status

    def handoff_summary(self) -> str:
        """Generate a compact end-of-shift artifact for the next on-call engineer."""
        self._record_tool("handoff_summary", {})
        unresolved = ", ".join(self.metrics.unresolved_incident_ids) or "none"
        key_memories = "\n".join(f"- [{entry.memory_id}] {entry.fact}" for entry in self.memory_entries[-5:]) or "- none"
        return (
            f"Shift {self.shift_id} handoff\n"
            f"Total reward: {self.total_reward:.2f}\n"
            f"Recall-before-action rate: {self.recall_before_action_rate():.2f}\n"
            f"Linked incident success rate: {self.linked_incident_success_rate():.2f}\n"
            f"Unresolved incidents: {unresolved}\n"
            f"Recent high-signal memories:\n{key_memories}\n"
        )

    def recall_before_action_rate(self) -> float:
        if self.metrics.recall_linked_total == 0:
            return 0.0
        return self.metrics.recall_linked_success / self.metrics.recall_linked_total

    def linked_incident_success_rate(self) -> float:
        if self.metrics.linked_total == 0:
            return 0.0
        return self.metrics.linked_success / self.metrics.linked_total

    def get_state(self) -> ShiftLogStateModel:
        incident = self.active_incident
        return ShiftLogStateModel(
            shift_id=self.shift_id,
            scenario_family=self.incidents[0].family if self.incidents else "unknown",
            current_index=self.current_index,
            done=self.done,
            total_reward=self.total_reward,
            reward_breakdown={key: round(value, 4) for key, value in self.reward_breakdown.items()},
            recall_before_action_rate=round(self.recall_before_action_rate(), 4),
            linked_incident_success_rate=round(self.linked_incident_success_rate(), 4),
            active_incident_id=incident.incident_id if incident else None,
            active_service=incident.service if incident else None,
            memory_count=len(self.memory_entries),
            contradiction_count=self.metrics.contradiction_count,
            recent_log_ids=[entry.memory_id for entry in self.memory_entries[-3:]],
        )

    def as_observation(self, message: str | None = None) -> ShiftLogObservation:
        incident = self.active_incident
        return ShiftLogObservation(
            message=message or self.last_observation,
            current_incident_id=incident.incident_id if incident else None,
            reward=self.total_reward,
            reward_breakdown={key: round(value, 4) for key, value in self.reward_breakdown.items()},
            done=self.done,
            metadata={
                "recall_before_action_rate": self.recall_before_action_rate(),
                "linked_incident_success_rate": self.linked_incident_success_rate(),
                "memory_count": len(self.memory_entries),
            },
        )
