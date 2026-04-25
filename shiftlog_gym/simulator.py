from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from random import Random
from typing import Any

from .domain import EpisodeMetrics
from .episode_state import EpisodeState, ReadQuery, Resolution, ShiftLogEntry, ToolCall
from .models import ShiftLogObservation, ShiftLogStateModel
from .rewards import (
    CompositeRubric,
    DEFAULT_RUBRIC,
    EfficiencyRubric,
    HallucinationRubric,
    HandoffQualityRubric,
    MemoryIntegrityRubric,
    MemoryWriteQualityRubric,
    NoiseResistanceRubric,
    RecallBeforeActionRubric,
    SuccessRubric,
)
from .scenarios import (
    FAMILY_ALIASES,
    PUBLIC_FAMILIES,
    BaseIncident,
    Scenario,
    ScenarioFactory,
)


RESERVED_TOOL_NAMES = {"reset", "step", "state", "close"}
REWARD_KEY_MAP = {
    "success": "R_success",
    "recall_before_action": "R_recall",
    "memory_write_quality": "R_memory_write",
    "memory_integrity": "R_memory_integrity",
    "efficiency": "R_efficiency",
    "hallucination": "R_hallucination",
    "noise_resistance": "R_noise_resistance",
    "handoff_quality": "R_handoff",
}


@dataclass(slots=True)
class RuntimeIncident:
    incident_id: str
    family: str
    variant_id: str
    sequence_index: int
    service: str
    summary: str
    symptoms: list[str]
    customer_impact: str
    root_cause: str
    resolution: str
    diagnostics: dict[str, str]
    dependencies: list[str]
    linked_to: str | None = None
    required_memory_keys: list[str] = field(default_factory=list)
    relevant_memory_terms: list[str] = field(default_factory=list)
    golden_memory: list[tuple[str, str]] = field(default_factory=list)
    unsupported_resolutions: list[str] = field(default_factory=list)
    runbook: str = ""
    owner: str = "platform-oncall"
    log_context: list[str] = field(default_factory=list)
    valid_mitigations: list[str] = field(default_factory=list)
    is_noise: bool = False
    linked_precursor_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MultiShiftObservation:
    message: str
    current_shift: str
    handoff_summary: str
    reward: float = 0.0
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


class ShiftLogSimulator:
    """ShiftLog-Gym simulator backed by procedural scenarios and rubric scoring."""

    def __init__(self, variants_per_family: int = 8, multi_shift: bool = False) -> None:
        self.variants_per_family = variants_per_family
        self.multi_shift = multi_shift
        self.factory = ScenarioFactory()
        self.random = Random(0)
        self.shift_id = "shift-0000"
        self.shift_start_time = datetime(2026, 1, 1, 9, 0, 0)
        self.scenario: Scenario | None = None
        self.incidents: list[RuntimeIncident] = []
        self.current_index = 0
        self.done = False
        self.last_observation = ""
        self.last_handoff_summary = ""
        self.last_multi_shift_handoff = ""
        self.metrics = EpisodeMetrics()
        self.episode_state: EpisodeState | None = None
        self.reward_breakdown: dict[str, float] = self._empty_reward_breakdown()
        self.total_reward = 0.0
        self._memory_counter = 0

    @property
    def active_incident(self) -> RuntimeIncident | None:
        if self.done or not self.incidents or self.current_index >= len(self.incidents):
            return None
        return self.incidents[self.current_index]

    def reset(
        self,
        *,
        seed: int | None = None,
        family: str | None = None,
        variant_index: int | None = None,
    ) -> str:
        scenario_seed = self._scenario_seed(seed=seed, variant_index=variant_index)
        family_name = self._resolve_family(family) if family else self.random.choice(PUBLIC_FAMILIES)
        self.scenario = self.factory.generate(seed=scenario_seed, family=family_name)
        self.shift_id = f"{self.scenario.family}-v{scenario_seed:02d}"
        self.shift_start_time = datetime(2026, 1, 1, 9, 0, 0) + timedelta(hours=scenario_seed)
        self.incidents = self._build_incident_queue(self.scenario)
        self.current_index = 0
        self.done = False
        self.last_handoff_summary = ""
        self._memory_counter = 0
        self.metrics = EpisodeMetrics()
        self.episode_state = EpisodeState(scenario=self.scenario)
        self.reward_breakdown = self._empty_reward_breakdown()
        self.total_reward = 0.0
        self.last_observation = self._render_current_incident()
        self._refresh_scores()
        return self.last_observation

    def preload_handoff(self, handoff_summary: str, entries: list[ShiftLogEntry]) -> None:
        if self.episode_state is None:
            raise ValueError("reset() must be called before preload_handoff().")
        self.last_multi_shift_handoff = handoff_summary
        for entry in entries:
            cloned = ShiftLogEntry(
                memory_id=entry.memory_id,
                timestamp=entry.timestamp,
                entry_type=entry.entry_type,
                incident_id=entry.incident_id,
                service=entry.service,
                fact=entry.fact,
                confidence=entry.confidence,
                contradiction=entry.contradiction,
                duplicate_of=entry.duplicate_of,
                fact_key=entry.fact_key,
                notes=list(entry.notes),
            )
            cloned.notes.append("preloaded-from-handoff")
            self.episode_state.shift_log_entries.append(cloned)
            self.metrics.memory_events.append({"event": "preload", **asdict(cloned)})
        self.last_observation = self._render_current_incident()

    def read_shift_log(self, query: str, limit: int = 5) -> str:
        incident = self.active_incident
        query_tokens = _normalized_tokens(query)
        matches: list[ShiftLogEntry] = []
        if self.episode_state is None:
            raise ValueError("Episode not initialized.")
        for entry in self.episode_state.shift_log_entries:
            haystack = _normalized_tokens(f"{entry.service} {entry.incident_id} {entry.fact}")
            if query_tokens.intersection(haystack):
                matches.append(entry)
        if incident and incident.linked_to and not matches:
            precursor_ids = set(incident.linked_precursor_ids)
            matches = [entry for entry in self.episode_state.shift_log_entries if entry.incident_id in precursor_ids]
        matches = matches[: max(1, limit)]
        rendered = [
            f"[{entry.memory_id}] service={entry.service} incident={entry.incident_id} confidence={entry.confidence:.2f} fact={entry.fact}"
            for entry in matches
        ]
        message = "\n".join(rendered) if rendered else "No relevant shift-log entries matched the query."
        metadata = {
            "linked_incident": bool(incident and incident.linked_to),
            "matched_incident_ids": [entry.incident_id for entry in matches],
        }
        self._record_tool("read_shift_log", message, {"query": query, "limit": limit}, metadata=metadata)
        self.episode_state.read_queries.append(
            ReadQuery(
                timestamp=self.episode_state.step_count,
                incident_id=incident.incident_id if incident else None,
                query=query,
                results=rendered,
                metadata=metadata,
            )
        )
        return message

    def append_shift_log(
        self,
        entry_type: str,
        incident_id: str,
        service: str,
        fact: str,
        confidence: float,
    ) -> str:
        if self.episode_state is None:
            raise ValueError("Episode not initialized.")
        fact = fact.strip()
        self._memory_counter += 1
        entry = ShiftLogEntry(
            memory_id=f"mem-{self._memory_counter:03d}",
            timestamp=self.episode_state.step_count,
            entry_type=entry_type,
            incident_id=incident_id,
            service=service,
            fact=fact,
            confidence=float(confidence),
            contradiction=self._would_contradict(incident_id, fact),
            duplicate_of=self._duplicate_memory_id(incident_id, service, fact),
            fact_key=self._match_fact_key(incident_id, fact),
        )
        if entry.contradiction:
            entry.notes.append("contradiction-detected")
            self.metrics.contradiction_count += 1
        if entry.duplicate_of:
            entry.notes.append(f"duplicate-of:{entry.duplicate_of}")
        self.episode_state.shift_log_entries.append(entry)
        self.metrics.memory_events.append({"event": "append", **asdict(entry)})
        message = f"Appended {entry.memory_id}." if fact else "Invalid shift-log write: fact is required."
        self._record_tool(
            "append_shift_log",
            message,
            {
                "entry_type": entry_type,
                "incident_id": incident_id,
                "service": service,
                "fact": fact,
                "confidence": confidence,
            },
        )
        return message

    def update_shift_log(self, memory_id: str, patch: str, reason: str) -> str:
        if self.episode_state is None:
            raise ValueError("Episode not initialized.")
        entry = next((item for item in self.episode_state.shift_log_entries if item.memory_id == memory_id), None)
        if entry is None:
            message = f"Memory entry {memory_id} not found."
            self._record_tool("update_shift_log", message, {"memory_id": memory_id, "patch": patch, "reason": reason})
            return message
        old_fact = entry.fact
        entry.fact = patch.strip()
        entry.notes.append(f"updated:{reason}")
        entry.contradiction = self._would_contradict(entry.incident_id, entry.fact, ignore_memory_id=memory_id)
        entry.fact_key = self._match_fact_key(entry.incident_id, entry.fact)
        if entry.contradiction:
            self.metrics.contradiction_count += 1
        self.metrics.memory_events.append(
            {
                "event": "update",
                "memory_id": memory_id,
                "incident_id": entry.incident_id,
                "service": entry.service,
                "old_fact": old_fact,
                "new_fact": entry.fact,
                "reason": reason,
                "contradiction": entry.contradiction,
            }
        )
        message = f"Updated {memory_id}."
        self._record_tool("update_shift_log", message, {"memory_id": memory_id, "patch": patch, "reason": reason})
        return message

    def inspect_service(self, service: str) -> str:
        incident = self.active_incident
        if incident and service == incident.service:
            message = f"{service}: owner={incident.owner}; runbook={incident.runbook}; symptoms={'; '.join(incident.symptoms)}"
        else:
            message = f"{service}: no active anomaly beyond dependency noise."
        self._record_tool("inspect_service", message, {"service": service})
        return message

    def inspect_dependency(self, service: str) -> str:
        incident = self.active_incident
        if incident and service == incident.service:
            message = f"{service}: dependencies => {', '.join(incident.dependencies)}"
        else:
            message = f"{service}: dependency graph stable."
        self._record_tool("inspect_dependency", message, {"service": service})
        return message

    def run_diagnostic(self, service: str, diagnostic: str) -> str:
        incident = self.active_incident
        if incident is None:
            message = "No active incident."
        else:
            message = incident.diagnostics.get(diagnostic, f"Diagnostic {diagnostic} returned no useful signal.")
        self._record_tool("run_diagnostic", message, {"service": service, "diagnostic": diagnostic})
        return message

    def apply_mitigation(self, service: str, mitigation: str) -> str:
        incident = self.active_incident
        if self.episode_state is None or incident is None:
            message = "No active incident."
            self._record_tool("apply_mitigation", message, {"service": service, "mitigation": mitigation})
            return message
        self.episode_state.mitigation_log.setdefault(incident.incident_id, []).append(mitigation)
        if service != incident.service:
            message = f"Mitigation applied to {service}, but the active incident is on {incident.service}."
        elif incident.is_noise and mitigation != incident.resolution:
            self.episode_state.noise_mitigation_failures += 1
            message = f"Mitigation {mitigation} had no effect. Symptoms persist."
        elif mitigation in incident.valid_mitigations:
            message = f"Mitigation {mitigation} applied cleanly to {service}."
        else:
            message = f"Mitigation {mitigation} had no meaningful effect."
        self._record_tool(
            "apply_mitigation",
            message,
            {"service": service, "mitigation": mitigation},
            metadata={"is_noise": incident.is_noise},
        )
        return message

    def resolve_incident(self, incident_id: str, resolution: str, root_cause: str) -> str:
        incident = self.active_incident
        if self.episode_state is None or incident is None:
            message = "No active incident."
            self._record_tool("resolve_incident", message, {"incident_id": incident_id, "resolution": resolution, "root_cause": root_cause})
            return message
        resolved = (
            incident.incident_id == incident_id
            and resolution == incident.resolution
            and root_cause.strip().lower() == incident.root_cause.lower()
        )
        resolution_record = Resolution(
            incident_id=incident.incident_id,
            root_cause=root_cause.strip(),
            mitigation=resolution,
            resolved=resolved,
            is_noise=incident.is_noise,
        )
        self.episode_state.resolution_log[incident.incident_id] = resolution_record
        if incident.linked_to:
            self.metrics.linked_total += 1
        if resolved and incident.linked_to:
            self.metrics.linked_success += 1
        if not resolved and incident.incident_id not in self.metrics.unresolved_incident_ids:
            self.metrics.unresolved_incident_ids.append(incident.incident_id)
        message = (
            f"Resolved {incident.incident_id} successfully."
            if resolved
            else (
                f"Resolution failed for {incident.incident_id}. "
                f"Expected resolution={incident.resolution} and root_cause={incident.root_cause}."
            )
        )
        self._record_tool(
            "resolve_incident",
            message,
            {"incident_id": incident_id, "resolution": resolution, "root_cause": root_cause},
            metadata={"resolved": resolved, "is_noise": incident.is_noise},
        )
        self._next_incident()
        return message

    def handoff_summary(self) -> str:
        if self.episode_state is None:
            raise ValueError("Episode not initialized.")
        unresolved = [
            incident.incident_id
            for incident in self.incidents
            if incident.incident_id not in self.episode_state.resolution_log
            or not self.episode_state.resolution_log[incident.incident_id].resolved
        ]
        key_entries = self.episode_state.shift_log_entries[-5:]
        facts = "\n".join(
            f"- {entry.incident_id}: {entry.fact} (confidence={entry.confidence:.2f})"
            for entry in key_entries
        ) or "- none"
        unresolved_text = ", ".join(unresolved) if unresolved else "none"
        summary = (
            f"Shift {self.shift_id} handoff\n"
            f"Unresolved incidents: {unresolved_text}\n"
            f"Recent facts:\n{facts}\n"
        )
        self.last_handoff_summary = summary
        self._record_tool("handoff_summary", summary, {"summary": summary})
        return summary

    def recall_before_action_rate(self) -> float:
        if self.episode_state is None:
            return 0.0
        return RecallBeforeActionRubric().score(self.episode_state)

    def linked_incident_success_rate(self) -> float:
        if self.metrics.linked_total == 0:
            return 0.0
        return self.metrics.linked_success / self.metrics.linked_total

    def get_state(self) -> ShiftLogStateModel:
        incident = self.active_incident
        return ShiftLogStateModel(
            shift_id=self.shift_id,
            scenario_family=self.scenario.family if self.scenario else "unknown",
            current_index=self.current_index,
            done=self.done,
            total_reward=round(self.total_reward, 4),
            reward_breakdown={key: round(value, 4) for key, value in self.reward_breakdown.items()},
            recall_before_action_rate=round(self.recall_before_action_rate(), 4),
            linked_incident_success_rate=round(self.linked_incident_success_rate(), 4),
            active_incident_id=incident.incident_id if incident else None,
            active_service=incident.service if incident else None,
            memory_count=len(self.episode_state.shift_log_entries) if self.episode_state else 0,
            contradiction_count=len(self.episode_state.get_contradiction_pairs()) if self.episode_state else 0,
            recent_log_ids=[entry.memory_id for entry in (self.episode_state.shift_log_entries[-3:] if self.episode_state else [])],
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
                "memory_count": len(self.episode_state.shift_log_entries) if self.episode_state else 0,
                "shift_start_time": self.shift_start_time.isoformat(),
                "handoff_summary": self.last_multi_shift_handoff or self.last_handoff_summary,
            },
        )

    def rubric_subscores(self) -> dict[str, float]:
        if self.episode_state is None:
            return {rubric.name: 0.0 for rubric, _ in DEFAULT_RUBRIC.rubrics}
        return DEFAULT_RUBRIC.subscores(self.episode_state)

    def _record_tool(
        self,
        tool_name: str,
        result: str,
        arguments: dict[str, Any] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if tool_name in RESERVED_TOOL_NAMES:
            raise ValueError(f"{tool_name} is reserved and cannot be used as a tool.")
        if self.episode_state is None:
            raise ValueError("Episode not initialized.")
        incident = self.active_incident
        call = ToolCall(
            timestamp=self.episode_state.step_count,
            tool_name=tool_name,
            incident_id=incident.incident_id if incident else None,
            arguments=arguments or {},
            result=result,
            metadata={
                "linked_incident": bool(incident and incident.linked_to),
                "is_noise": bool(incident and incident.is_noise),
                **(metadata or {}),
            },
        )
        self.episode_state.tool_call_log.append(call)
        self.episode_state.step_count += 1
        self.metrics.tool_timeline.append(
            {
                "shift_id": self.shift_id,
                "incident_id": call.incident_id,
                "step_index": call.timestamp,
                "tool": tool_name,
                "detail": call.metadata,
                "result": result,
            }
        )
        if tool_name == "read_shift_log" and call.metadata.get("linked_incident"):
            self.metrics.recall_linked_total += 1
            if result and "No relevant shift-log entries" not in result:
                self.metrics.recall_linked_success += 1
        self._refresh_scores()

    def _refresh_scores(self) -> None:
        if self.episode_state is None:
            return
        subscores = DEFAULT_RUBRIC.subscores(self.episode_state)
        self.reward_breakdown = {REWARD_KEY_MAP[name]: value for name, value in subscores.items()}
        self.total_reward = max(-1.0, min(1.0, DEFAULT_RUBRIC.score(self.episode_state)))

    def _next_incident(self) -> None:
        self.current_index += 1
        if self.current_index >= len(self.incidents):
            self.done = True
            if self.episode_state:
                self.episode_state.is_complete = True
            self.last_observation = "Shift complete. Generate handoff_summary() for final artifact."
        else:
            self.last_observation = self._render_current_incident()
        self._refresh_scores()

    def _render_current_incident(self) -> str:
        incident = self.active_incident
        if incident is None:
            return "Shift complete. No active incidents."
        log_snippets = "\n".join(
            f"- {entry.fact}" for entry in (self.episode_state.shift_log_entries[-3:] if self.episode_state else [])
        ) or "- No prior shift-log snippets in view."
        handoff = self.last_multi_shift_handoff.strip()
        handoff_block = f"Handoff summary carried from prior shift:\n{handoff}\n" if handoff else ""
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
            f"{handoff_block}"
            f"Recent shift-log snippets:\n{log_snippets}\n"
            f"Runbook hint: {incident.runbook}\n"
        )

    def _build_incident_queue(self, scenario: Scenario) -> list[RuntimeIncident]:
        precursors = [self._runtime_from_base(incident, idx + 1, scenario.seed) for idx, incident in enumerate(scenario.precursor_incidents)]
        linked = [self._runtime_from_base(incident, len(precursors) + idx + 1, scenario.seed) for idx, incident in enumerate(scenario.linked_incidents)]
        noise = [self._runtime_from_base(incident, len(precursors) + len(linked) + idx + 1, scenario.seed) for idx, incident in enumerate(scenario.noise_incidents)]
        queue: list[RuntimeIncident] = list(precursors)
        while linked or noise:
            if noise and self.random.random() < 0.35:
                queue.append(noise.pop(0))
            if linked:
                queue.append(linked.pop(0))
            if noise and self.random.random() < 0.5:
                queue.append(noise.pop(0))
        queue.extend(noise)
        return queue

    def _runtime_from_base(self, incident: BaseIncident, sequence_index: int, seed: int) -> RuntimeIncident:
        relevant_terms = list(
            dict.fromkeys(
                list(incident.shift_log_keywords)
                + [incident.service]
                + [
                    token
                    for symptom in incident.symptoms
                    for token in symptom.lower().replace("-", " ").split()
                    if len(token) > 3
                ]
            )
        )
        root_exemplar = incident.root_cause
        mitigation_exemplar = f"Mitigation: {incident.mitigation} on {incident.service}."
        if incident.family == "db_pool" and incident.service == "payments-api":
            root_exemplar = "Rollback left payments-api with stale DB_POOL_SIZE=44."
            mitigation_exemplar = "Mitigation: set_pool_size_and_restart on payments-api."
        return RuntimeIncident(
            incident_id=incident.incident_id,
            family=incident.family,
            variant_id=f"v{seed:02d}",
            sequence_index=sequence_index,
            service=incident.service,
            summary=incident.summary,
            symptoms=list(incident.symptoms),
            customer_impact=incident.customer_impact,
            root_cause=incident.root_cause,
            resolution=incident.mitigation,
            diagnostics=dict(incident.diagnostics),
            dependencies=list(incident.service_names),
            linked_to=incident.linked_precursor_ids[0] if incident.linked_precursor_ids else None,
            required_memory_keys=list(incident.required_memory_keywords),
            relevant_memory_terms=relevant_terms,
            golden_memory=[
                ("root_cause_fact", root_exemplar),
                ("mitigation_fact", mitigation_exemplar),
            ],
            unsupported_resolutions=["invalid_mitigation", "memory_retrieved_guess"],
            runbook=f"{incident.family}/{incident.service}",
            owner="platform-oncall",
            log_context=[f"error_code={code}" for code in incident.error_codes],
            valid_mitigations=list(incident.valid_mitigations),
            is_noise=incident.is_noise,
            linked_precursor_ids=list(incident.linked_precursor_ids),
        )

    def _would_contradict(self, incident_id: str, fact: str, ignore_memory_id: str | None = None) -> bool:
        if self.episode_state is None:
            return False
        new_tokens = _normalized_tokens(fact)
        new_keys = _key_tokens(fact)
        negations = {"not", "never", "no", "healthy"}
        positives = {"stale", "enabled", "saturated", "timeout", "oom", "deprecated"}
        for entry in self.episode_state.shift_log_entries:
            if entry.incident_id != incident_id:
                continue
            if ignore_memory_id and entry.memory_id == ignore_memory_id:
                continue
            overlap = _overlap_ratio(new_tokens, _normalized_tokens(entry.fact))
            if overlap > 0.5 and new_keys != _key_tokens(entry.fact):
                return True
            existing_tokens = _normalized_tokens(entry.fact)
            if overlap > 0.3:
                if new_tokens.intersection(negations) and existing_tokens.intersection(positives):
                    return True
                if new_tokens.intersection(positives) and existing_tokens.intersection(negations):
                    return True
        return False

    def _duplicate_memory_id(self, incident_id: str, service: str, fact: str) -> str | None:
        if self.episode_state is None:
            return None
        for entry in self.episode_state.shift_log_entries:
            if entry.incident_id == incident_id and entry.service == service and entry.fact == fact:
                return entry.memory_id
        return None

    def _match_fact_key(self, incident_id: str, fact: str) -> str | None:
        incident = next((item for item in self.incidents if item.incident_id == incident_id), None)
        if incident is None:
            return None
        lower = fact.lower()
        for fact_key, exemplar in incident.golden_memory:
            tokens = [token for token in exemplar.lower().replace(".", "").split() if len(token) > 4]
            overlap = sum(1 for token in tokens if token in lower)
            if overlap >= max(1, min(2, len(tokens))):
                return fact_key
        return None

    def _scenario_seed(self, *, seed: int | None, variant_index: int | None) -> int:
        if seed is not None:
            self.random.seed(seed)
            return max(1, seed)
        if variant_index is not None:
            return (variant_index % self.variants_per_family) + 1
        return self.random.randint(1, self.variants_per_family)

    def _resolve_family(self, family: str) -> str:
        return FAMILY_ALIASES.get(family, family)

    def _empty_reward_breakdown(self) -> dict[str, float]:
        return {key: 0.0 for key in REWARD_KEY_MAP.values()}


class MultiShiftEpisode:
    """Two-stage episode with a precomputed handoff from Shift 1 into Shift 2."""

    def __init__(self, variants_per_family: int = 8) -> None:
        self.variants_per_family = variants_per_family
        self.random = Random(0)
        self.shift_1 = ShiftLogSimulator(variants_per_family=variants_per_family)
        self.shift_2 = ShiftLogSimulator(variants_per_family=variants_per_family)
        self.handoff_summary_text = ""
        self.done = False
        self.reward = 0.0
        self.shift_1_rubric = DEFAULT_RUBRIC
        self.shift_2_rubric = _shift_two_rubric()

    def reset(self, *, seed: int | None = None, family: str | None = None, variant_index: int | None = None) -> MultiShiftObservation:
        base_seed = seed if seed is not None else (variant_index or 0) + 1
        first_family = FAMILY_ALIASES.get(family, family) if family else self.random.choice(PUBLIC_FAMILIES)
        second_family_choices = [candidate for candidate in PUBLIC_FAMILIES if candidate != first_family]
        second_family = second_family_choices[(base_seed + 1) % len(second_family_choices)]

        self.shift_1.reset(seed=base_seed, family=first_family, variant_index=variant_index)
        self._autoplay_shift_one()
        unresolved = self._unresolved_shift_one_incidents()
        self.handoff_summary_text = self.shift_1.handoff_summary()

        self.shift_2.reset(seed=base_seed + 11, family=second_family, variant_index=variant_index)
        preload_entries = list(self.shift_1.episode_state.shift_log_entries[-4:]) if self.shift_1.episode_state else []
        self.shift_2.preload_handoff(self.handoff_summary_text, preload_entries)
        if unresolved:
            self._inject_downstream_consequence(unresolved[0])
        self.done = False
        self.reward = 0.0
        message = self.shift_2.last_observation
        return MultiShiftObservation(
            message=message,
            current_shift="shift_2",
            handoff_summary=self.handoff_summary_text,
            reward=self.reward,
            done=self.done,
            metadata={"shift_1_family": self.shift_1.scenario.family if self.shift_1.scenario else None},
        )

    def step(self, action: dict[str, Any]) -> tuple[MultiShiftObservation, float, bool, dict[str, Any]]:
        if self.done:
            observation = MultiShiftObservation(
                message="Multi-shift episode already complete.",
                current_shift="shift_2",
                handoff_summary=self.handoff_summary_text,
                reward=self.reward,
                done=True,
            )
            return observation, self.reward, True, self.get_info()
        tool_name = action.get("tool")
        arguments = action.get("arguments", {})
        tool = getattr(self.shift_2, tool_name, None)
        if tool is None or tool_name.startswith("_"):
            message = f"Unknown tool: {tool_name}"
        else:
            message = tool(**arguments)
        shift_one_score = self.shift_1_rubric.score(self.shift_1.episode_state)
        shift_two_score = self.shift_2_rubric.score(self.shift_2.episode_state)
        self.reward = max(-1.0, min(1.0, (0.4 * shift_one_score) + (0.6 * shift_two_score)))
        self.done = self.shift_2.done
        observation = MultiShiftObservation(
            message=message,
            current_shift="shift_2",
            handoff_summary=self.handoff_summary_text,
            reward=self.reward,
            done=self.done,
            metadata={"shift_2_reward": shift_two_score},
        )
        return observation, self.reward, self.done, self.get_info()

    def get_info(self) -> dict[str, Any]:
        return {
            "handoff_summary": self.handoff_summary_text,
            "shift_1_reward": self.shift_1_rubric.score(self.shift_1.episode_state),
            "shift_2_reward": self.shift_2_rubric.score(self.shift_2.episode_state),
            "reward_total": self.reward,
            "shift_2_state": self.shift_2.get_state().model_dump(),
        }

    def _autoplay_shift_one(self) -> None:
        if self.shift_1.episode_state is None:
            return
        while not self.shift_1.done:
            incident = self.shift_1.active_incident
            if incident is None:
                break
            if incident.linked_to:
                self.shift_1.read_shift_log(" ".join(incident.relevant_memory_terms[:3]), limit=3)
            if incident.golden_memory:
                _, fact = incident.golden_memory[0]
                self.shift_1.append_shift_log("fact", incident.incident_id, incident.service, fact, 0.85)
            if incident.sequence_index % 3 == 0:
                self.shift_1.resolve_incident(incident.incident_id, "invalid_mitigation", "unknown")
                continue
            self.shift_1.apply_mitigation(incident.service, incident.resolution)
            self.shift_1.resolve_incident(incident.incident_id, incident.resolution, incident.root_cause)

    def _unresolved_shift_one_incidents(self) -> list[RuntimeIncident]:
        unresolved_ids = set(self.shift_1.metrics.unresolved_incident_ids)
        return [incident for incident in self.shift_1.incidents if incident.incident_id in unresolved_ids]

    def _inject_downstream_consequence(self, unresolved: RuntimeIncident) -> None:
        consequence = RuntimeIncident(
            incident_id=f"{unresolved.incident_id}-handoff",
            family=unresolved.family,
            variant_id=unresolved.variant_id,
            sequence_index=0,
            service=unresolved.service,
            summary=f"Downstream consequence of unresolved {unresolved.incident_id}",
            symptoms=list(unresolved.symptoms),
            customer_impact=f"Escalated impact from prior unresolved incident on {unresolved.service}",
            root_cause=unresolved.root_cause,
            resolution=unresolved.resolution,
            diagnostics={
                "handoff": f"Symptoms line up with unresolved prior incident {unresolved.incident_id}",
                "history": f"Needs handoff recall from Shift 1 for {unresolved.service}",
            },
            dependencies=list(unresolved.dependencies),
            linked_to=unresolved.incident_id,
            required_memory_keys=list(unresolved.required_memory_keys or ["handoff"]),
            relevant_memory_terms=list(unresolved.relevant_memory_terms or [unresolved.service]),
            golden_memory=list(unresolved.golden_memory),
            unsupported_resolutions=list(unresolved.unsupported_resolutions),
            runbook=unresolved.runbook,
            owner=unresolved.owner,
            log_context=list(unresolved.log_context),
            valid_mitigations=list(unresolved.valid_mitigations),
            linked_precursor_ids=[unresolved.incident_id],
        )
        self.shift_2.incidents.insert(0, consequence)
        self.shift_2.current_index = 0
        self.shift_2.last_observation = self.shift_2._render_current_incident()


def _shift_two_rubric() -> CompositeRubric:
    remaining_weight = 0.60
    base_remaining = 0.75
    scale = remaining_weight / base_remaining
    return CompositeRubric(
        [
            (SuccessRubric(), 0.35 * scale),
            (RecallBeforeActionRubric(), 0.40),
            (MemoryWriteQualityRubric(), 0.15 * scale),
            (MemoryIntegrityRubric(), 0.10 * scale),
            (EfficiencyRubric(), 0.05 * scale),
            (HallucinationRubric(), 0.05 * scale),
            (NoiseResistanceRubric(), 0.03 * scale),
            (HandoffQualityRubric(), 0.02 * scale),
        ]
    )


def _normalized_tokens(text: str) -> set[str]:
    return {token.strip(".,:;[]()").lower() for token in text.split() if token.strip(".,:;[]()")}


def _key_tokens(text: str) -> set[str]:
    return {
        token
        for token in _normalized_tokens(text)
        if any(char.isdigit() for char in token) or "-" in token or "_" in token
    }


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / max(len(left), len(right))
