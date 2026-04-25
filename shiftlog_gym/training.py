from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .scenarios import PUBLIC_FAMILIES
from .simulator import ShiftLogSimulator


TRAIN_VARIANTS = tuple(range(0, 6))
VALID_VARIANTS = (6,)
TEST_VARIANTS = (7,)
AVAILABLE_TOOLS = (
    "read_shift_log",
    "append_shift_log",
    "update_shift_log",
    "inspect_service",
    "inspect_dependency",
    "run_diagnostic",
    "apply_mitigation",
    "resolve_incident",
    "handoff_summary",
)
TRAINING_FAMILIES = PUBLIC_FAMILIES


@dataclass(slots=True)
class EpisodeArtifacts:
    episode_row: dict[str, Any]
    memory_events: list[dict[str, Any]]
    tool_timeline: list[dict[str, Any]]
    episode_replay: dict[str, Any]


def build_variant_split() -> dict[str, tuple[int, ...]]:
    return {
        "train": TRAIN_VARIANTS,
        "valid": VALID_VARIANTS,
        "test": TEST_VARIANTS,
    }


def weighted_reward_from_breakdown(reward_breakdown: dict[str, float]) -> float:
    return (
        reward_breakdown.get("R_success", 0.0)
        + 0.35 * reward_breakdown.get("R_recall", 0.0)
        + 0.15 * reward_breakdown.get("R_memory_write", 0.0)
        + reward_breakdown.get("R_memory_integrity", 0.0)
        + 0.1 * reward_breakdown.get("R_efficiency", 0.0)
        + reward_breakdown.get("R_hallucination", 0.0)
    )


def compute_memory_precision(memory_events: list[dict[str, Any]]) -> float:
    writes = [event for event in memory_events if event.get("event") == "append"]
    if not writes:
        return 0.0
    correct = 0
    for event in writes:
        if event.get("contradiction"):
            continue
        if event.get("duplicate_of"):
            continue
        if event.get("fact_key"):
            correct += 1
    return correct / len(writes)


def compute_contradiction_rate(memory_events: list[dict[str, Any]]) -> float:
    mutating = [event for event in memory_events if event.get("event") in {"append", "update"}]
    if not mutating:
        return 0.0
    contradictions = 0
    for event in mutating:
        if event.get("contradiction"):
            contradictions += 1
        elif event.get("event") == "update" and event.get("reason") == "contradiction-detected":
            contradictions += 1
    return contradictions / len(mutating)


def compute_bad_write_rate(memory_events: list[dict[str, Any]]) -> float:
    writes = [event for event in memory_events if event.get("event") == "append"]
    if not writes:
        return 0.0
    bad_writes = 0
    for event in writes:
        if event.get("contradiction") or event.get("duplicate_of"):
            bad_writes += 1
    return bad_writes / len(writes)


def compute_log_read_propensity(tool_timeline: list[dict[str, Any]], linked_total: int) -> float:
    if linked_total <= 0:
        return 0.0
    linked_with_read = set()
    for event in tool_timeline:
        if event.get("tool") == "read_shift_log" and event.get("detail", {}).get("linked_incident"):
            linked_with_read.add(event.get("incident_id"))
    return len(linked_with_read) / linked_total


def compute_unsupported_mitigation_rate(metrics: Any, total_incidents: int) -> float:
    if total_incidents <= 0:
        return 0.0
    return metrics.fabricated_resolution_count / total_incidents


def compute_noise_resistance_rate(simulator: ShiftLogSimulator) -> float:
    return simulator.rubric_subscores().get("noise_resistance", 0.0)


def summarize_episode(simulator: ShiftLogSimulator, episode_name: str, split: str, seed: int, variant_index: int) -> EpisodeArtifacts:
    state = simulator.get_state()
    memory_events = list(simulator.metrics.memory_events)
    tool_timeline = list(simulator.metrics.tool_timeline)
    linked_total = simulator.metrics.linked_total
    episode_row = {
        "episode_name": episode_name,
        "split": split,
        "seed": seed,
        "variant_index": variant_index,
        "shift_id": state.shift_id,
        "family": state.scenario_family,
        "total_reward": state.total_reward,
        "weighted_reward": weighted_reward_from_breakdown(state.reward_breakdown),
        "R_success": state.reward_breakdown.get("R_success", 0.0),
        "R_recall": state.reward_breakdown.get("R_recall", 0.0),
        "R_memory_write": state.reward_breakdown.get("R_memory_write", 0.0),
        "R_memory_integrity": state.reward_breakdown.get("R_memory_integrity", 0.0),
        "R_efficiency": state.reward_breakdown.get("R_efficiency", 0.0),
        "R_hallucination": state.reward_breakdown.get("R_hallucination", 0.0),
        "recall_before_action_rate": state.recall_before_action_rate,
        "linked_incident_success_rate": state.linked_incident_success_rate,
        "memory_precision": compute_memory_precision(memory_events),
        "contradiction_rate": compute_contradiction_rate(memory_events),
        "bad_write_rate": compute_bad_write_rate(memory_events),
        "unsupported_mitigation_rate": compute_unsupported_mitigation_rate(simulator.metrics, len(simulator.incidents)),
        "log_read_propensity": compute_log_read_propensity(tool_timeline, linked_total),
        "noise_resistance_rate": compute_noise_resistance_rate(simulator),
        "memory_count": state.memory_count,
        "contradiction_count": state.contradiction_count,
        "linked_total": linked_total,
        "linked_success": simulator.metrics.linked_success,
        "unresolved_count": len(simulator.metrics.unresolved_incident_ids),
    }
    replay = {
        "episode_name": episode_name,
        "split": split,
        "seed": seed,
        "variant_index": variant_index,
        "shift_id": state.shift_id,
        "tool_calls": _tool_calls_json(simulator),
        "shift_log_entries": _shift_log_entries_json(simulator),
        "resolutions": _resolutions_json(simulator),
        "timeline_snapshots": _timeline_snapshots_json(simulator),
        "retrieved_before_resolution": _retrieved_before_resolution(simulator),
        "metrics": episode_row,
    }
    return EpisodeArtifacts(
        episode_row=episode_row,
        memory_events=memory_events,
        tool_timeline=tool_timeline,
        episode_replay=replay,
    )


def scripted_policy(simulator: ShiftLogSimulator) -> None:
    while not simulator.done:
        incident = simulator.active_incident
        if incident is None:
            break
        if incident.required_memory_keys:
            simulator.read_shift_log(" ".join(incident.relevant_memory_terms[:3]), limit=3)
        simulator.inspect_service(incident.service)
        first_diag = next(iter(incident.diagnostics.keys()))
        simulator.run_diagnostic(incident.service, first_diag)
        for _, fact in incident.golden_memory[:1]:
            simulator.append_shift_log("fact", incident.incident_id, incident.service, fact, 0.95)
        simulator.apply_mitigation(incident.service, incident.resolution)
        simulator.resolve_incident(incident.incident_id, incident.resolution, incident.root_cause)


def random_policy(simulator: ShiftLogSimulator) -> None:
    while not simulator.done:
        incident = simulator.active_incident
        if incident is None:
            break
        simulator.inspect_service(incident.service)
        simulator.inspect_dependency(incident.service)
        first_diag = next(iter(incident.diagnostics.keys()))
        simulator.run_diagnostic(incident.service, first_diag)
        if incident.golden_memory:
            _, fact = incident.golden_memory[0]
            simulator.append_shift_log("note", incident.incident_id, incident.service, fact[:40], 0.5)
        unsupported = incident.unsupported_resolutions[0] if incident.unsupported_resolutions else incident.resolution
        simulator.apply_mitigation(incident.service, unsupported)
        simulator.resolve_incident(incident.incident_id, unsupported, "unknown")


def collect_policy_rollouts(
    policy_name: str,
    policy_fn: Callable[[ShiftLogSimulator], None],
    split: str = "train",
    variants: tuple[int, ...] | None = None,
    seeds: tuple[int, ...] = (0,),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    variants = variants or build_variant_split()[split]
    rows: list[dict[str, Any]] = []
    memory_events: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []

    for family in TRAINING_FAMILIES:
        for variant_index in variants:
            for seed in seeds:
                simulator = ShiftLogSimulator()
                simulator.reset(seed=seed, family=family, variant_index=variant_index)
                policy_fn(simulator)
                artifacts = summarize_episode(
                    simulator=simulator,
                    episode_name=f"{policy_name}-{family}-{variant_index}-{seed}",
                    split=split,
                    seed=seed,
                    variant_index=variant_index,
                )
                rows.append(artifacts.episode_row)
                memory_events.extend(artifacts.memory_events)
                tool_events.extend(artifacts.tool_timeline)
    return rows, memory_events, tool_events


def apply_action_dict(simulator: ShiftLogSimulator, action: dict[str, Any]) -> str:
    tool = action.get("tool")
    arguments = action.get("arguments", {})
    if tool not in AVAILABLE_TOOLS:
        simulator.metrics.unresolved_incident_ids.append(
            simulator.active_incident.incident_id if simulator.active_incident else "unknown"
        )
        simulator.reward_breakdown["R_hallucination"] = simulator.reward_breakdown.get("R_hallucination", 0.0) - 0.5
        simulator.total_reward -= 0.5
        return f"Unknown tool: {tool}"
    handler = getattr(simulator, tool)
    return handler(**arguments)


def rollout_prompted_policy(
    simulator: ShiftLogSimulator,
    policy_fn: Callable[[str, list[dict[str, Any]]], dict[str, Any]],
    max_steps: int = 18,
) -> list[dict[str, Any]]:
    transcript: list[dict[str, Any]] = []
    for _ in range(max_steps):
        if simulator.done:
            break
        observation = simulator.last_observation or simulator._render_current_incident()
        action = policy_fn(observation, transcript)
        tool_response = apply_action_dict(simulator, action)
        transcript.append(
            {
                "observation": observation,
                "action": action,
                "tool_response": tool_response,
            }
        )
    return transcript


def write_artifacts(
    output_dir: Path,
    episodes: list[dict[str, Any]],
    memory_events: list[dict[str, Any]],
    tool_events: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in episodes:
            handle.write(json.dumps(row) + "\n")

    with (output_dir / "memory_events.jsonl").open("w", encoding="utf-8") as handle:
        for row in memory_events:
            handle.write(json.dumps(row) + "\n")

    with (output_dir / "tool_timeline.jsonl").open("w", encoding="utf-8") as handle:
        for row in tool_events:
            handle.write(json.dumps(row) + "\n")

    if episodes:
        with (output_dir / "eval_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(episodes[0].keys()))
            writer.writeheader()
            writer.writerows(episodes)


def summarize_baseline(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    numeric_keys = [
        "recall_before_action_rate",
        "linked_incident_success_rate",
        "noise_resistance_rate",
        "weighted_reward",
        "contradiction_rate",
    ]
    summary: dict[str, float] = {}
    for key in numeric_keys:
        summary_key = {
            "weighted_reward": "avg_total_reward",
            "contradiction_rate": "contradiction_rate",
        }.get(key, key)
        summary[summary_key] = sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
    return summary


def write_episode_replays(output_dir: Path, episodes: list[EpisodeArtifacts]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for artifact in episodes:
        episode_name = artifact.episode_row["episode_name"]
        with (output_dir / f"{episode_name}.json").open("w", encoding="utf-8") as handle:
            json.dump(artifact.episode_replay, handle, indent=2)


def _tool_calls_json(simulator: ShiftLogSimulator) -> list[dict[str, Any]]:
    if simulator.episode_state is None:
        return []
    return [
        {
            "timestamp": call.timestamp,
            "tool_name": call.tool_name,
            "incident_id": call.incident_id,
            "arguments": call.arguments,
            "result": call.result,
            "metadata": call.metadata,
        }
        for call in simulator.episode_state.tool_call_log
    ]


def _shift_log_entries_json(simulator: ShiftLogSimulator) -> list[dict[str, Any]]:
    if simulator.episode_state is None:
        return []
    return [
        {
            "memory_id": entry.memory_id,
            "timestamp": entry.timestamp,
            "incident_id": entry.incident_id,
            "entry_type": entry.entry_type,
            "service": entry.service,
            "fact": entry.fact,
            "confidence": entry.confidence,
            "contradiction": entry.contradiction,
            "duplicate_of": entry.duplicate_of,
            "fact_key": entry.fact_key,
        }
        for entry in simulator.episode_state.shift_log_entries
    ]


def _resolutions_json(simulator: ShiftLogSimulator) -> list[dict[str, Any]]:
    if simulator.episode_state is None:
        return []
    return [
        {
            "incident_id": resolution.incident_id,
            "root_cause": resolution.root_cause,
            "mitigation": resolution.mitigation,
            "resolved": resolution.resolved,
            "is_noise": resolution.is_noise,
        }
        for resolution in simulator.episode_state.resolution_log.values()
    ]


def _timeline_snapshots_json(simulator: ShiftLogSimulator) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    if simulator.episode_state is None:
        return snapshots
    running_log: list[dict[str, Any]] = []
    for call in simulator.episode_state.tool_call_log:
        if call.tool_name == "append_shift_log":
            incident_id = call.arguments.get("incident_id")
            matching = next(
                (
                    entry
                    for entry in simulator.episode_state.shift_log_entries
                    if entry.incident_id == incident_id and entry.fact == call.arguments.get("fact")
                ),
                None,
            )
            if matching:
                running_log.append(
                    {
                        "memory_id": matching.memory_id,
                        "incident_id": matching.incident_id,
                        "fact": matching.fact,
                        "confidence": matching.confidence,
                    }
                )
        snapshots.append({"timestamp": call.timestamp, "tool_name": call.tool_name, "shift_log_entries": list(running_log)})
    return snapshots


def _retrieved_before_resolution(simulator: ShiftLogSimulator) -> list[str]:
    if simulator.episode_state is None:
        return []
    incident_ids: list[str] = []
    for incident in simulator.incidents:
        if simulator.episode_state.tool_was_called_before("read_shift_log", incident.incident_id, "resolve_incident"):
            incident_ids.append(incident.incident_id)
    return incident_ids
