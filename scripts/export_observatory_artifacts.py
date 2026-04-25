from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiftlog_gym.training import collect_policy_rollouts, scripted_policy, write_artifacts


def run_eval(output_dir: Path, episodes: int = 12) -> None:
    variants = tuple(range(min(episodes, 8)))
    episode_rows, memory_rows, tool_rows = collect_policy_rollouts(
        policy_name="scripted",
        policy_fn=scripted_policy,
        split="train",
        variants=variants,
        seeds=(0,),
    )
    write_artifacts(output_dir, episode_rows, memory_rows, tool_rows)


if __name__ == "__main__":
    run_eval(Path("observatory"))
