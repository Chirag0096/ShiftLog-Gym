from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiftlog_gym.training import summarize_episode, summarize_baseline, write_artifacts, write_episode_replays
from shiftlog_gym.simulator import ShiftLogSimulator


def run_eval(output_dir: Path, episodes: int = 12) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    memory_rows = []
    tool_rows = []
    replay_artifacts = []

    for episode_idx in range(episodes):
        sim = ShiftLogSimulator()
        sim.reset(seed=episode_idx + 1, family="db_pool", variant_index=episode_idx % 8)
        while not sim.done and sim.episode_state.step_count < 25:
            incident = sim.active_incident
            if incident is None:
                break
            if incident.linked_to:
                sim.read_shift_log(" ".join(incident.relevant_memory_terms[:3]) or incident.service, limit=3)
            sim.inspect_service(incident.service)
            diagnostic = next(iter(incident.diagnostics.keys()))
            sim.run_diagnostic(incident.service, diagnostic)
            sim.apply_mitigation(incident.service, incident.resolution)
            sim.resolve_incident(incident.incident_id, incident.resolution, incident.root_cause)
        artifact = summarize_episode(sim, f"scripted-export-{episode_idx:03d}", "export", episode_idx + 1, episode_idx % 8)
        replay_artifacts.append(artifact)
        rows.append(artifact.episode_row)
        memory_rows.extend(artifact.memory_events)
        tool_rows.extend(artifact.tool_timeline)

    write_artifacts(output_dir, rows, memory_rows, tool_rows)
    write_episode_replays(episodes_dir, replay_artifacts)
    baselines_path = output_dir / "baselines.json"
    baselines_path.write_text(
        __import__("json").dumps(
            {
                "random": {},
                "scripted": summarize_baseline(rows),
                "llm_base": {},
                "trained_llm": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    run_eval(Path("observatory"))
