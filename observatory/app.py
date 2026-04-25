from __future__ import annotations

import json
from pathlib import Path

try:
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install shiftlog-gym[observatory] to run the observatory app.") from exc


ROOT = Path(__file__).resolve().parent
EPISODES_DIR = ROOT / "episodes"
TRAINING_RUNS_DIR = ROOT / "training_runs"
BASELINES_PATH = ROOT / "baselines.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def episode_files() -> list[Path]:
    return sorted(EPISODES_DIR.glob("*.json"))


def training_curve_files() -> list[Path]:
    return sorted(TRAINING_RUNS_DIR.glob("*.csv"))


def timeline_dataframe(episode: dict) -> pd.DataFrame:
    rows = []
    for call in episode.get("tool_calls", []):
        rows.append(
            {
                "timestamp": call.get("timestamp"),
                "tool_name": call.get("tool_name"),
                "incident_id": call.get("incident_id"),
                "result_summary": call.get("result", "")[:140],
                "is_noise": call.get("metadata", {}).get("is_noise", False),
                "resolved": call.get("metadata", {}).get("resolved"),
                "memory_op": call.get("tool_name") in {"read_shift_log", "append_shift_log", "update_shift_log", "handoff_summary"},
            }
        )
    return pd.DataFrame(rows)


def timeline_style(row: pd.Series) -> list[str]:
    if row.get("memory_op"):
        color = "#d8ecff"
    elif row.get("is_noise"):
        color = "#fff4c2"
    elif row.get("resolved") is True:
        color = "#d8f5d0"
    elif row.get("resolved") is False:
        color = "#ffd8d8"
    else:
        color = ""
    return [f"background-color: {color}" for _ in row]


def memory_quality_dataframe(episode: dict) -> pd.DataFrame:
    resolutions = {resolution.get("incident_id"): resolution for resolution in episode.get("resolutions", [])}
    rows = []
    for entry in episode.get("shift_log_entries", []):
        incident_id = entry.get("incident_id")
        rows.append(
            {
                "timestamp": entry.get("timestamp"),
                "incident_id": incident_id,
                "entry_type": entry.get("entry_type"),
                "fact": entry.get("fact"),
                "confidence": entry.get("confidence"),
                "is_contradicted": entry.get("contradiction", False),
                "was_retrieved_before_resolution": incident_id in episode.get("retrieved_before_resolution", []),
                "resolved": resolutions.get(incident_id, {}).get("resolved"),
            }
        )
    return pd.DataFrame(rows)


def memory_style(row: pd.Series) -> list[str]:
    if row.get("is_contradicted"):
        color = "#ffd8d8"
    elif row.get("was_retrieved_before_resolution"):
        color = "#d8f5d0"
    else:
        color = ""
    return [f"background-color: {color}" for _ in row]


st.set_page_config(page_title="ShiftLog Observatory", layout="wide")
st.title("ShiftLog Observatory")
st.caption("Episode replay, rubric curves, memory quality, and baseline comparisons for ShiftLog-Gym.")

tab_replay, tab_training, tab_memory, tab_baselines = st.tabs(
    ["Episode Replay", "Training Curves", "Memory Quality", "Baseline Comparison"]
)

with tab_replay:
    files = episode_files()
    if not files:
        st.info("No episode replay JSON files found yet in observatory/episodes/.")
    else:
        selected_path = st.selectbox("Saved episode", files, format_func=lambda path: path.name)
        episode = load_json(selected_path)
        timeline = timeline_dataframe(episode)
        replay_col, side_col = st.columns([3, 2])
        with replay_col:
            st.subheader("Timeline")
            if timeline.empty:
                st.warning("This episode file does not contain tool call rows yet.")
            else:
                st.dataframe(timeline.style.apply(timeline_style, axis=1), use_container_width=True)
        with side_col:
            st.subheader("Shift Log State")
            snapshots = episode.get("timeline_snapshots", [])
            if snapshots:
                index = st.slider("Scrub timeline", min_value=0, max_value=len(snapshots) - 1, value=0)
                st.json(snapshots[index])
            else:
                st.warning("No timeline snapshots found in this episode file.")

with tab_training:
    files = training_curve_files()
    if not files:
        st.info("No training curves found yet in observatory/training_runs/.")
    else:
        selected_curve = st.selectbox("Training curve CSV", files, format_func=lambda path: path.name)
        curves = load_csv(selected_curve)
        if curves.empty:
            st.warning("Selected CSV is empty.")
        else:
            figure = go.Figure()
            for column in [
                "reward_total",
                "reward_recall",
                "reward_success",
                "reward_memory_write",
                "reward_memory_integrity",
            ]:
                if column in curves.columns:
                    figure.add_trace(go.Scatter(x=curves["step"], y=curves[column], mode="lines", name=column))
            if "recall_before_action_rate" in curves.columns:
                figure.add_trace(
                    go.Scatter(
                        x=curves["step"],
                        y=curves["recall_before_action_rate"],
                        mode="lines",
                        name="Recall-Before-Action Rate",
                        line={"color": "orange", "width": 4},
                    )
                )
            figure.add_hline(y=0.5, line_dash="dash", annotation_text="Human baseline estimate")
            figure.update_layout(height=520, xaxis_title="Step", yaxis_title="Score")
            st.plotly_chart(figure, use_container_width=True)

with tab_memory:
    files = episode_files()
    if not files:
        st.info("No episode replay JSON files found yet in observatory/episodes/.")
    else:
        selected_path = st.selectbox("Episode for memory audit", files, key="memory_file", format_func=lambda path: path.name)
        episode = load_json(selected_path)
        memory_df = memory_quality_dataframe(episode)
        if memory_df.empty:
            st.warning("This episode file does not contain shift-log entry rows yet.")
        else:
            st.dataframe(memory_df.style.apply(memory_style, axis=1), use_container_width=True)

with tab_baselines:
    baselines = load_json(BASELINES_PATH)
    if not baselines:
        st.info("No baseline summary found yet in observatory/baselines.json.")
    else:
        baseline_df = pd.DataFrame(
            [
                {"agent": "Random Agent", **baselines.get("random", {})},
                {"agent": "Scripted Agent (always reads log)", **baselines.get("scripted", {})},
                {"agent": "Untrained LLM baseline", **baselines.get("llm_base", {})},
                {"agent": "Trained LLM", **baselines.get("trained_llm", {})},
            ]
        )
        st.dataframe(baseline_df, use_container_width=True)
