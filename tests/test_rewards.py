from __future__ import annotations

import unittest

from shiftlog_gym.episode_state import EpisodeState, ReadQuery, Resolution, ShiftLogEntry, ToolCall
from shiftlog_gym.rewards import (
    DEFAULT_RUBRIC,
    HandoffQualityRubric,
    MemoryIntegrityRubric,
    RecallBeforeActionRubric,
    SuccessRubric,
)
from shiftlog_gym.scenarios import ScenarioFactory


class RewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = ScenarioFactory().generate(seed=3, family="db_pool")

    def _perfect_episode(self) -> EpisodeState:
        episode = EpisodeState(scenario=self.scenario)
        for incident in self.scenario.precursor_incidents + self.scenario.linked_incidents:
            episode.resolution_log[incident.incident_id] = Resolution(
                incident_id=incident.incident_id,
                root_cause=incident.root_cause,
                mitigation=incident.mitigation,
                resolved=True,
            )
        return episode

    def test_success_rubric_scores_perfect_episode(self) -> None:
        score = SuccessRubric().score(self._perfect_episode())
        self.assertEqual(score, 1.0)

    def test_recall_before_action_detects_ordering(self) -> None:
        linked = self.scenario.linked_incidents[0]
        episode = EpisodeState(
            scenario=self.scenario,
            tool_call_log=[
                ToolCall(timestamp=1, tool_name="read_shift_log", incident_id=linked.incident_id),
                ToolCall(timestamp=2, tool_name="apply_mitigation", incident_id=linked.incident_id),
            ],
        )
        self.assertEqual(RecallBeforeActionRubric().score(episode) >= 0.0, True)
        self.assertTrue(episode.tool_was_called_before("read_shift_log", linked.incident_id, "apply_mitigation"))

    def test_memory_integrity_detects_conflicting_duplicates(self) -> None:
        incident_id = self.scenario.precursor_incidents[0].incident_id
        episode = EpisodeState(
            scenario=self.scenario,
            shift_log_entries=[
                ShiftLogEntry(timestamp=1, entry_type="fact", incident_id=incident_id, service="payments-api", fact="pool size 44 error DBPOOL-44", confidence=0.9),
                ShiftLogEntry(timestamp=2, entry_type="fact", incident_id=incident_id, service="payments-api", fact="pool size 55 error DBPOOL-55", confidence=0.9),
            ],
        )
        self.assertLess(MemoryIntegrityRubric().score(episode), 0.0)

    def test_handoff_quality_scores_valid_summary(self) -> None:
        unresolved = self.scenario.linked_incidents[0]
        episode = EpisodeState(
            scenario=self.scenario,
            tool_call_log=[
                ToolCall(
                    timestamp=5,
                    tool_name="handoff_summary",
                    incident_id=None,
                    result=f"{unresolved.incident_id} remains open because {unresolved.root_cause}. confidence=0.9",
                )
            ],
            shift_log_entries=[
                ShiftLogEntry(
                    timestamp=3,
                    entry_type="handoff",
                    incident_id=unresolved.incident_id,
                    service=unresolved.service,
                    fact=unresolved.root_cause,
                    confidence=0.95,
                )
            ],
            resolution_log={
                unresolved.incident_id: Resolution(
                    incident_id=unresolved.incident_id,
                    root_cause=unresolved.root_cause,
                    mitigation=unresolved.mitigation,
                    resolved=False,
                )
            },
        )
        self.assertGreaterEqual(HandoffQualityRubric().score(episode), 0.5)

    def test_default_rubric_weights_sum_to_one(self) -> None:
        self.assertEqual(round(sum(weight for _, weight in DEFAULT_RUBRIC.rubrics), 8), 1.0)


if __name__ == "__main__":
    unittest.main()
