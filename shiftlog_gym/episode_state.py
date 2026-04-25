from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .scenarios import Scenario


@dataclass(slots=True)
class ToolCall:
    timestamp: int
    tool_name: str
    incident_id: str | None
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ShiftLogEntry:
    timestamp: int
    entry_type: str
    incident_id: str
    service: str
    fact: str
    confidence: float
    memory_id: str = ""
    contradiction: bool = False
    duplicate_of: str | None = None
    fact_key: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Resolution:
    incident_id: str
    root_cause: str
    mitigation: str
    resolved: bool
    is_noise: bool = False


@dataclass(slots=True)
class ReadQuery:
    timestamp: int
    incident_id: str | None
    query: str
    results: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeState:
    scenario: Scenario
    tool_call_log: list[ToolCall] = field(default_factory=list)
    shift_log_entries: list[ShiftLogEntry] = field(default_factory=list)
    resolution_log: dict[str, Resolution] = field(default_factory=dict)
    mitigation_log: dict[str, list[str]] = field(default_factory=dict)
    read_queries: list[ReadQuery] = field(default_factory=list)
    step_count: int = 0
    is_complete: bool = False
    noise_mitigation_failures: int = 0

    def tool_was_called_before(self, tool_name: str, incident_id: str, before_tool: str) -> bool:
        first_before_timestamp: int | None = None
        for call in self.tool_call_log:
            if call.incident_id != incident_id:
                continue
            if call.tool_name == before_tool:
                first_before_timestamp = call.timestamp
                break
        for call in self.tool_call_log:
            if call.incident_id != incident_id:
                continue
            if call.tool_name != tool_name:
                continue
            if first_before_timestamp is None:
                return True
            if call.timestamp < first_before_timestamp:
                return True
        return False

    def get_contradiction_pairs(self) -> list[tuple[ShiftLogEntry, ShiftLogEntry]]:
        contradictions: list[tuple[ShiftLogEntry, ShiftLogEntry]] = []
        for index, left in enumerate(self.shift_log_entries):
            left_tokens = _normalized_tokens(left.fact)
            for right in self.shift_log_entries[index + 1 :]:
                if left.incident_id != right.incident_id:
                    continue
                right_tokens = _normalized_tokens(right.fact)
                overlap = _token_overlap(left_tokens, right_tokens)
                if overlap <= 0.5:
                    if not _has_polarity_conflict(left.fact, right.fact):
                        continue
                if _key_tokens(left.fact) != _key_tokens(right.fact) or _has_polarity_conflict(left.fact, right.fact):
                    contradictions.append((left, right))
        return contradictions


def _normalized_tokens(text: str) -> set[str]:
    return {token.strip(".,:;[]()").lower() for token in text.split() if token.strip(".,:;[]()")}


def _token_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / max(len(left), len(right))


def _key_tokens(text: str) -> set[str]:
    key_like: set[str] = set()
    for token in _normalized_tokens(text):
        if any(char.isdigit() for char in token) or "-" in token or "_" in token:
            key_like.add(token)
    return key_like


def _has_polarity_conflict(left: str, right: str) -> bool:
    negations = {"not", "never", "no", "healthy"}
    positives = {"stale", "enabled", "saturated", "timeout", "oom", "deprecated"}
    left_tokens = _normalized_tokens(left)
    right_tokens = _normalized_tokens(right)
    return bool(
        (left_tokens.intersection(negations) and right_tokens.intersection(positives))
        or (left_tokens.intersection(positives) and right_tokens.intersection(negations))
    )
