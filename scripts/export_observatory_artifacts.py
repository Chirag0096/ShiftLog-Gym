from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shiftlog_gym.simulator import ShiftLogSimulator


def run_eval(output_dir: Path, episodes: int = 12) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    simulator = ShiftLogSimulator()
    episodes_rows = []
    memory_rows = []

    for episode_idx in range(episodes):
        family = None
        if episode_idx < 4:
            family = simulator.library.keys().__iter__().__next__()
        simulator.reset(seed=episode_idx + 1, family=family)
        while not simulator.done:
            incident = simulator.active_incident
            if incident is None:
                break
            if incident.required_memory_keys:
                simulator.read_shift_log(" ".join(incident.relevant_memory_terms[:2]), limit=3)
            simulator.inspect_service(incident.service)
            first_diag = next(iter(incident.diagnostics.keys()))
            simulator.run_diagnostic(incident.service, first_diag)
            if incident.golden_memory:
                fact_key, fact = incident.golden_memory[0]
                simulator.append_shift_log("fact", incident.incident_id, incident.service, fact, 0.95)
            simulator.apply_mitigation(incident.service, incident.resolution)
            simulator.resolve_incident(incident.incident_id, incident.resolution, incident.root_cause)
        state = simulator.get_state()
        episodes_rows.append(
            {
                "episode": episode_idx,
                "shift_id": state.shift_id,
                "family": state.scenario_family,
                "total_reward": state.total_reward,
                "recall_before_action_rate": state.recall_before_action_rate,
                "linked_incident_success_rate": state.linked_incident_success_rate,
                "memory_count": state.memory_count,
                "contradiction_count": state.contradiction_count,
            }
        )
        memory_rows.extend(simulator.metrics.memory_events)

    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in episodes_rows:
            handle.write(json.dumps(row) + "\n")
    with (output_dir / "memory_events.jsonl").open("w", encoding="utf-8") as handle:
        for row in memory_rows:
            handle.write(json.dumps(row) + "\n")
    with (output_dir / "eval_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episodes_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episodes_rows)


if __name__ == "__main__":
    run_eval(Path("observatory"))
