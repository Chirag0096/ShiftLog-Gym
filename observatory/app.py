from __future__ import annotations

import json
from pathlib import Path

try:
    import pandas as pd
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install shiftlog-gym[observatory] to run the observatory app.") from exc


ROOT = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


st.set_page_config(page_title="ShiftLog Observatory", layout="wide")
st.title("ShiftLog Observatory")
st.caption("Reward breakdown, failure patterns, and memory traces for ShiftLog-Gym.")

episodes = load_jsonl(ROOT / "episodes.jsonl")
memory_events = load_jsonl(ROOT / "memory_events.jsonl")

if episodes.empty:
    st.warning("No evaluation artifacts found yet. Run scripts/export_observatory_artifacts.py first.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Episodes", len(episodes))
col2.metric("Mean Total Reward", round(float(episodes["total_reward"].mean()), 3))
col3.metric("Mean Recall Rate", round(float(episodes["recall_before_action_rate"].mean()), 3))

st.subheader("Episode Metrics")
st.dataframe(episodes, use_container_width=True)

st.subheader("Reward / Recall Curves")
st.line_chart(
    episodes.set_index("episode")[["total_reward", "recall_before_action_rate", "linked_incident_success_rate"]],
    use_container_width=True,
)

if not memory_events.empty:
    st.subheader("Memory Events")
    st.dataframe(memory_events, use_container_width=True)

st.subheader("Failure Taxonomy Template")
st.markdown(
    """
- `missed_recall`: linked incident resolved without relevant `read_shift_log`
- `wrong_recall`: retrieved memory did not match required fact keys
- `bad_memory_write`: invalid or duplicate structured memory
- `contradiction_introduced`: write/update conflicts with prior memory
- `wrong_mitigation`: correct memory but wrong operational action
- `solved_without_memory`: solvable incident that did not require recall
"""
    )
