from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px


ROOT = Path(__file__).resolve().parent.parent
OBS_ROOT = ROOT / "observatory"
RUNS_DIR = OBS_ROOT / "training_runs"
BASELINES_FILE = OBS_ROOT / "baselines.json"


def _load_curve(stage: str) -> pd.DataFrame:
    path = RUNS_DIR / f"training_curves_{stage}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def plot_training(stage: str):
    frame = _load_curve(stage)
    if frame.empty:
        return px.line(title=f"No curve file found for {stage}")
    metric_columns = [col for col in ["reward_total", "reward_recall", "reward_success", "reward_memory_write", "recall_before_action_rate"] if col in frame.columns]
    melted = frame.melt(id_vars=["step"], value_vars=metric_columns, var_name="metric", value_name="value")
    figure = px.line(melted, x="step", y="value", color="metric", title=f"ShiftLog-Gym Training Curves ({stage})")
    figure.add_hline(y=0.5, line_dash="dash", annotation_text="Human baseline estimate")
    return figure


def load_baseline_table():
    if not BASELINES_FILE.exists():
        return pd.DataFrame([{"status": "No baselines.json found"}])
    payload = json.loads(BASELINES_FILE.read_text(encoding="utf-8"))
    rows = [
        {"agent": "Random Agent", **payload.get("random", {})},
        {"agent": "Scripted Agent", **payload.get("scripted", {})},
        {"agent": "Untrained LLM", **payload.get("llm_base", {})},
        {"agent": "Trained LLM", **payload.get("trained_llm", {})},
    ]
    return pd.DataFrame(rows)


def load_eval_table(stage: str):
    path = RUNS_DIR / f"eval_summary_{stage}.csv"
    if not path.exists():
        return pd.DataFrame([{"status": f"No eval_summary_{stage}.csv found"}])
    return pd.read_csv(path)


with gr.Blocks(title="ShiftLog Observatory (Gradio)") as demo:
    gr.Markdown("# ShiftLog Observatory")
    gr.Markdown(
        "Visualize training curves, baseline comparison, and held-out evaluation tables for ShiftLog-Gym."
    )

    with gr.Tab("Training Curves"):
        stage_choice = gr.Dropdown(choices=["stageA", "stageB", "stageC"], value="stageC", label="Stage")
        curve_plot = gr.Plot(label="Training Curves")
        curve_button = gr.Button("Load Curves")
        curve_button.click(fn=plot_training, inputs=[stage_choice], outputs=[curve_plot])

    with gr.Tab("Baseline Comparison"):
        baseline_table = gr.Dataframe(label="Baselines")
        baseline_button = gr.Button("Load Baselines")
        baseline_button.click(fn=load_baseline_table, inputs=None, outputs=[baseline_table])

    with gr.Tab("Held-out Evaluation"):
        eval_stage_choice = gr.Dropdown(choices=["stageA", "stageB", "stageC"], value="stageC", label="Stage")
        eval_table = gr.Dataframe(label="Evaluation Summary")
        eval_button = gr.Button("Load Eval Table")
        eval_button.click(fn=load_eval_table, inputs=[eval_stage_choice], outputs=[eval_table])


if __name__ == "__main__":
    demo.launch()
