from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    BaseModel = None
    Field = None


if BaseModel is None:
    @dataclass
    class ShiftLogAction:
        tool: str
        arguments: dict[str, Any] = field(default_factory=dict)


    @dataclass
    class ShiftLogObservation:
        message: str
        current_incident_id: str | None = None
        reward: float = 0.0
        reward_breakdown: dict[str, float] = field(default_factory=dict)
        done: bool = False
        metadata: dict[str, Any] = field(default_factory=dict)

        def model_dump(self) -> dict[str, Any]:
            return asdict(self)


    @dataclass
    class ShiftLogStateModel:
        shift_id: str
        scenario_family: str
        current_index: int
        done: bool
        total_reward: float
        reward_breakdown: dict[str, float]
        recall_before_action_rate: float
        linked_incident_success_rate: float
        active_incident_id: str | None = None
        active_service: str | None = None
        memory_count: int = 0
        contradiction_count: int = 0
        recent_log_ids: list[str] = field(default_factory=list)

        def model_dump(self) -> dict[str, Any]:
            return asdict(self)
else:
    class ShiftLogAction(BaseModel):
        tool: str = Field(..., description="Tool name to execute.")
        arguments: dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool.")

    class ShiftLogObservation(BaseModel):
        message: str
        current_incident_id: str | None = None
        reward: float = 0.0
        reward_breakdown: dict[str, float] = Field(default_factory=dict)
        done: bool = False
        metadata: dict[str, Any] = Field(default_factory=dict)


    class ShiftLogStateModel(BaseModel):
        shift_id: str
        scenario_family: str
        current_index: int
        done: bool
        total_reward: float
        reward_breakdown: dict[str, float]
        recall_before_action_rate: float
        linked_incident_success_rate: float
        active_incident_id: str | None = None
        active_service: str | None = None
        memory_count: int = 0
        contradiction_count: int = 0
        recent_log_ids: list[str] = Field(default_factory=list)
