from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


RewardKey = Literal[
    "R_success",
    "R_recall",
    "R_memory_write",
    "R_memory_integrity",
    "R_efficiency",
    "R_hallucination",
]


@dataclass(slots=True)
class MemoryEntry:
    memory_id: str
    entry_type: str
    incident_id: str
    service: str
    fact: str
    confidence: float
    timestamp: int
    source_incident_id: str | None = None
    fact_key: str | None = None
    immutable: bool = False
    contradiction: bool = False
    duplicate_of: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IncidentDefinition:
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


@dataclass(slots=True)
class EpisodeMetrics:
    recall_linked_total: int = 0
    recall_linked_success: int = 0
    linked_total: int = 0
    linked_success: int = 0
    contradiction_count: int = 0
    bad_write_count: int = 0
    fabricated_resolution_count: int = 0
    unresolved_incident_ids: list[str] = field(default_factory=list)
    tool_timeline: list[dict[str, object]] = field(default_factory=list)
    memory_events: list[dict[str, object]] = field(default_factory=list)

