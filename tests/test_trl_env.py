from __future__ import annotations

import unittest

from shiftlog_gym.trl_env import ShiftLogToolEnv


class ShiftLogToolEnvTests(unittest.TestCase):
    def test_reset_and_tool_call(self) -> None:
        env = ShiftLogToolEnv()
        initial = env.reset(family="memory_oom_signature", variant_index=0)
        self.assertIn("Incident:", initial)
        result = env.inspect_service("ranking-worker")
        self.assertIn("ranking-worker", result)


if __name__ == "__main__":
    unittest.main()
