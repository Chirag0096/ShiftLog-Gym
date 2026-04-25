from __future__ import annotations

import unittest

from shiftlog_gym.trl_env import ShiftLogToolEnv


class ShiftLogToolEnvTests(unittest.TestCase):
    def test_reset_and_tool_call(self) -> None:
        env = ShiftLogToolEnv()
        initial = env.reset(family="memory_oom_signature", variant_index=0)
        self.assertIn("You are an on-call SRE agent.", initial)
        result = env.inspect_service("ranking-worker")
        self.assertIn("ranking-worker", result)

    def test_info_contains_all_reward_keys(self) -> None:
        env = ShiftLogToolEnv(rollout_mode="short")
        env.reset(family="db_pool_exhaustion", variant_index=0)
        info = env.get_info()
        for key in (
            "reward_success",
            "reward_recall",
            "reward_memory_write",
            "reward_memory_integrity",
            "reward_efficiency",
            "reward_hallucination",
            "reward_noise_resistance",
            "reward_handoff",
            "reward_total",
        ):
            self.assertIn(key, info)


if __name__ == "__main__":
    unittest.main()
