from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .episode_state import EpisodeState, ShiftLogEntry


class Rubric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def score(self, episode_state: EpisodeState) -> float:
        raise NotImplementedError


class SuccessRubric(Rubric):
    name = "success"

    def score(self, episode_state: EpisodeState) -> float:
        real_incidents = [incident for incident in episode_state.scenario.incidents if not incident.is_noise]
        if not real_incidents:
            return 0.0
        correct = 0
        for incident in real_incidents:
            resolution = episode_state.resolution_log.get(incident.incident_id)
            if resolution and resolution.resolved and resolution.root_cause == incident.root_cause and resolution.mitigation == incident.mitigation:
                correct += 1
        return correct / len(real_incidents)


class RecallBeforeActionRubric(Rubric):
    name = "recall_before_action"

    def score(self, episode_state: EpisodeState) -> float:
        linked = list(episode_state.scenario.linked_incidents)
        if not linked:
            return 0.0
        recall_before_action = 0
        for incident in linked:
            if episode_state.tool_was_called_before("read_shift_log", incident.incident_id, "apply_mitigation") or episode_state.tool_was_called_before("read_shift_log", incident.incident_id, "resolve_incident"):
                recall_before_action += 1
        return recall_before_action / len(linked)


class MemoryWriteQualityRubric(Rubric):
    name = "memory_write_quality"
    valid_entry_types = {"fact", "hypothesis", "resolution", "handoff"}

    def score(self, episode_state: EpisodeState) -> float:
        entries = episode_state.shift_log_entries
        if not entries:
            return 0.0
        keywords = {
            token.lower()
            for root_cause in episode_state.scenario.ground_truth.values()
            for token in root_cause.replace("-", " ").replace("_", " ").split()
            if len(token) > 3
        }
        incident_ids = set(episode_state.scenario.ground_truth.keys())
        total = 0.0
        for entry in entries:
            score = 0.0
            if entry.entry_type in self.valid_entry_types:
                score += 0.1
            if entry.incident_id in incident_ids:
                score += 0.1
            if isinstance(entry.confidence, float) and 0.0 <= entry.confidence <= 1.0:
                score += 0.1
            if any(keyword in entry.fact.lower() for keyword in keywords):
                score += 0.1
            total += score
        return min(1.0, total / (len(entries) * 0.4))


class MemoryIntegrityRubric(Rubric):
    name = "memory_integrity"

    def score(self, episode_state: EpisodeState) -> float:
        penalty = 0.0
        contradictions = episode_state.get_contradiction_pairs()
        penalty -= 0.2 * len(contradictions)
        penalty -= 0.1 * _duplicate_count(episode_state.shift_log_entries)
        penalty -= 0.15 * _schema_violations(episode_state.shift_log_entries)
        return max(-1.0, min(0.0, penalty))


class EfficiencyRubric(Rubric):
    name = "efficiency"

    def score(self, episode_state: EpisodeState) -> float:
        score = 1.0 - (len(episode_state.tool_call_log) / 40.0)
        return max(0.0, min(1.0, score))


class HallucinationRubric(Rubric):
    name = "hallucination"

    def score(self, episode_state: EpisodeState) -> float:
        penalty = 0.0
        valid_by_service = episode_state.scenario.valid_mitigations
        roots = list(episode_state.scenario.ground_truth.values())
        for call in episode_state.tool_call_log:
            if call.tool_name == "apply_mitigation":
                service = call.arguments.get("service")
                mitigation = call.arguments.get("mitigation", "")
                valid = set(valid_by_service.get(service, ()))
                if mitigation not in valid:
                    penalty -= 0.3
            if call.tool_name == "resolve_incident":
                root_cause = call.arguments.get("root_cause", "")
                if not any(_token_overlap_ratio(root_cause, candidate) > 0.6 for candidate in roots):
                    penalty -= 0.2
        return max(-1.0, min(0.0, penalty))


class NoiseResistanceRubric(Rubric):
    name = "noise_resistance"

    def score(self, episode_state: EpisodeState) -> float:
        score = 0.0
        noise_ids = {incident.incident_id for incident in episode_state.scenario.noise_incidents}
        memory_text = " ".join(entry.fact.lower() for entry in episode_state.shift_log_entries)
        for incident in episode_state.scenario.noise_incidents:
            incident_mitigations = episode_state.mitigation_log.get(incident.incident_id, [])
            if not incident_mitigations:
                continue
            used_memory_root = incident.root_cause.lower() not in memory_text and any(root.lower() in memory_text for root in episode_state.scenario.ground_truth.values())
            if used_memory_root:
                score -= 0.3
            else:
                score += 0.3
        return max(-1.0, min(1.0, score))


class HandoffQualityRubric(Rubric):
    name = "handoff_quality"

    def score(self, episode_state: EpisodeState) -> float:
        handoff_calls = [call for call in episode_state.tool_call_log if call.tool_name == "handoff_summary"]
        if not handoff_calls:
            return 0.0
        handoff_text = handoff_calls[-1].result or handoff_calls[-1].arguments.get("summary", "")
        unresolved = [
            incident_id
            for incident_id, resolution in episode_state.resolution_log.items()
            if not resolution.resolved
        ]
        if not unresolved:
            unresolved = [incident.incident_id for incident in episode_state.scenario.incidents[:1]]
        mentioned = sum(1 for incident_id in unresolved if incident_id in handoff_text)
        mention_ratio = mentioned / len(unresolved) if unresolved else 1.0
        root_matches = 0
        for incident_id in unresolved:
            root_cause = episode_state.scenario.ground_truth.get(incident_id, "")
            if root_cause and _token_overlap_ratio(root_cause, handoff_text) > 0.15:
                root_matches += 1
        confidence_values = [entry.confidence for entry in episode_state.shift_log_entries if entry.incident_id in unresolved]
        average_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        if mention_ratio >= 0.8 and root_matches >= max(1, mentioned) and average_confidence > 0.6:
            return 1.0
        if mention_ratio >= 0.4 and (root_matches >= 1 or average_confidence > 0.6):
            return 0.5
        return 0.0


@dataclass(slots=True)
class CompositeRubric(Rubric):
    rubrics: list[tuple[Rubric, float]]

    @property
    def name(self) -> str:
        return "composite"

    def __post_init__(self) -> None:
        total = sum(weight for _, weight in self.rubrics)
        if round(total, 8) != 1.0:
            raise ValueError("Rubric weights must sum to 1.0")

    def score(self, episode_state: EpisodeState) -> float:
        return sum(rubric.score(episode_state) * weight for rubric, weight in self.rubrics)

    def subscores(self, episode_state: EpisodeState) -> dict[str, float]:
        return {rubric.name: rubric.score(episode_state) for rubric, _ in self.rubrics}


DEFAULT_RUBRIC = CompositeRubric(
    [
        (SuccessRubric(), 0.35),
        (RecallBeforeActionRubric(), 0.25),
        (MemoryWriteQualityRubric(), 0.15),
        (MemoryIntegrityRubric(), 0.10),
        (EfficiencyRubric(), 0.05),
        (HallucinationRubric(), 0.05),
        (NoiseResistanceRubric(), 0.03),
        (HandoffQualityRubric(), 0.02),
    ]
)


def _duplicate_count(entries: list[ShiftLogEntry]) -> int:
    seen: set[tuple[str, str, str]] = set()
    duplicates = 0
    for entry in entries:
        key = (entry.incident_id, entry.service, entry.fact)
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def _schema_violations(entries: list[ShiftLogEntry]) -> int:
    violations = 0
    for entry in entries:
        if not entry.entry_type or not entry.incident_id or not entry.service or not entry.fact:
            violations += 1
    return violations


def _token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = {token.lower() for token in left.replace("-", " ").replace("_", " ").split() if token}
    right_tokens = {token.lower() for token in right.replace("-", " ").replace("_", " ").split() if token}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens.intersection(right_tokens)) / max(len(left_tokens), len(right_tokens))
