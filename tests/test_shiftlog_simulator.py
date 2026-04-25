from __future__ import annotations

import unittest

from shiftlog_gym.simulator import ShiftLogSimulator


class ShiftLogSimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = ShiftLogSimulator()

    def test_invalid_memory_write_is_penalized(self) -> None:
        self.sim.reset(family="db_pool_exhaustion", variant_index=0)
        message = self.sim.append_shift_log("", "", "", "", 0.2)
        self.assertIn("Invalid shift-log write", message)
        self.assertLess(self.sim.reward_breakdown["R_memory_integrity"], 0.0)

    def test_contradiction_detection_flags_conflicting_writes(self) -> None:
        self.sim.reset(family="auth_timeout_cascade", variant_index=0)
        self.sim.append_shift_log("fact", "auth-01-01", "auth-gateway", "auth-gateway has a stale route to one cache shard", 0.9)
        self.sim.append_shift_log("fact", "auth-01-01", "auth-gateway", "auth-gateway is healthy and has no stale route", 0.8)
        self.assertEqual(self.sim.metrics.contradiction_count, 1)
        self.assertLess(self.sim.reward_breakdown["R_memory_integrity"], 0.0)

    def test_linked_incident_rewards_recall_before_action(self) -> None:
        self.sim.reset(family="db_pool_exhaustion", variant_index=0)
        first = self.sim.active_incident
        self.sim.append_shift_log("fact", first.incident_id, first.service, "Rollback left payments-api with stale DB_POOL_SIZE=44.", 0.95)
        self.sim.append_shift_log("fact", first.incident_id, first.service, "Mitigation: set_pool_size_and_restart on payments-api.", 0.95)
        self.sim.resolve_incident(first.incident_id, first.resolution, first.root_cause)
        self.sim.resolve_incident(self.sim.active_incident.incident_id, self.sim.active_incident.resolution, self.sim.active_incident.root_cause)
        linked = self.sim.active_incident
        before = self.sim.reward_breakdown["R_recall"]
        self.sim.read_shift_log("pool size payments-api rollback", limit=3)
        self.sim.resolve_incident(linked.incident_id, linked.resolution, linked.root_cause)
        self.assertGreater(self.sim.reward_breakdown["R_recall"], before)
        self.assertGreater(self.sim.linked_incident_success_rate(), 0.0)

    def test_linked_incidents_depend_on_prior_memory(self) -> None:
        self.sim.reset(family="feature_flag_regression", variant_index=0)
        first = self.sim.active_incident
        self.sim.resolve_incident(first.incident_id, first.resolution, first.root_cause)
        self.sim.resolve_incident(self.sim.active_incident.incident_id, self.sim.active_incident.resolution, self.sim.active_incident.root_cause)
        linked = self.sim.active_incident
        no_memory_message = self.sim.read_shift_log("deprecated bundle receipts", limit=3)
        self.assertIn("No relevant shift-log entries", no_memory_message)
        self.assertTrue(linked.required_memory_keys)


if __name__ == "__main__":
    unittest.main()

