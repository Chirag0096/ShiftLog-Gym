from __future__ import annotations

import unittest

from shiftlog_gym.simulator import ShiftLogSimulator
from shiftlog_gym.training import (
    compute_bad_write_rate,
    compute_log_read_propensity,
    compute_memory_precision,
    summarize_episode,
    weighted_reward_from_breakdown,
)


class TrainingHelperTests(unittest.TestCase):
    def test_weighted_reward_matches_formula(self) -> None:
        reward = weighted_reward_from_breakdown(
            {
                "R_success": 2.0,
                "R_recall": 1.0,
                "R_memory_write": 2.0,
                "R_memory_integrity": -0.5,
                "R_efficiency": -1.0,
                "R_hallucination": -0.2,
            }
        )
        self.assertAlmostEqual(reward, 2.0 + 0.35 + 0.30 - 0.5 - 0.1 - 0.2)

    def test_episode_summary_computes_memory_metrics(self) -> None:
        simulator = ShiftLogSimulator()
        simulator.reset(family="db_pool_exhaustion", variant_index=0)
        first = simulator.active_incident
        simulator.append_shift_log("fact", first.incident_id, first.service, "Rollback left payments-api with stale DB_POOL_SIZE=44.", 0.95)
        simulator.append_shift_log("fact", first.incident_id, first.service, "Rollback left payments-api with stale DB_POOL_SIZE=44.", 0.95)
        artifacts = summarize_episode(simulator, "unit", "train", 0, 0)
        self.assertGreaterEqual(artifacts.episode_row["bad_write_rate"], 0.5)
        self.assertGreaterEqual(artifacts.episode_row["memory_precision"], 0.5)

    def test_log_read_propensity_uses_linked_marker(self) -> None:
        simulator = ShiftLogSimulator()
        simulator.reset(family="db_pool_exhaustion", variant_index=0)
        simulator.resolve_incident(simulator.active_incident.incident_id, simulator.active_incident.resolution, simulator.active_incident.root_cause)
        simulator.resolve_incident(simulator.active_incident.incident_id, simulator.active_incident.resolution, simulator.active_incident.root_cause)
        linked = simulator.active_incident
        simulator.read_shift_log("pool rollback", limit=3)
        propensity = compute_log_read_propensity(simulator.metrics.tool_timeline, linked_total=1)
        self.assertEqual(linked.linked_to is not None, True)
        self.assertEqual(propensity, 1.0)


if __name__ == "__main__":
    unittest.main()
