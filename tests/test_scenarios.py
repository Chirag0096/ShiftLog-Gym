from __future__ import annotations

import unittest

from shiftlog_gym.scenarios import ScenarioFactory


class ScenarioFactoryTests(unittest.TestCase):
    def test_same_seed_is_deterministic(self) -> None:
        factory = ScenarioFactory()
        left = factory.generate(seed=7, family="db_pool")
        right = factory.generate(seed=7, family="db_pool")
        self.assertEqual(left, right)

    def test_different_seeds_produce_different_scenarios(self) -> None:
        factory = ScenarioFactory()
        left = factory.generate(seed=7, family="db_pool")
        right = factory.generate(seed=8, family="db_pool")
        self.assertNotEqual(left, right)

    def test_linked_incidents_reference_valid_precursors(self) -> None:
        factory = ScenarioFactory()
        scenario = factory.generate(seed=11, family="auth_cascade")
        precursor_ids = {incident.incident_id for incident in scenario.precursor_incidents}
        for incident in scenario.linked_incidents:
            self.assertTrue(incident.linked_precursor_ids)
            self.assertTrue(set(incident.linked_precursor_ids).issubset(precursor_ids))

    def test_noise_incidents_are_flagged(self) -> None:
        factory = ScenarioFactory()
        scenario = factory.generate(seed=15, family="config_drift")
        self.assertTrue(scenario.noise_incidents)
        for incident in scenario.noise_incidents:
            self.assertTrue(incident.is_noise)


if __name__ == "__main__":
    unittest.main()
