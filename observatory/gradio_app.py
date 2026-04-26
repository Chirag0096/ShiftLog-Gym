from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px

# Directory structure resolution
ROOT = Path(__file__).resolve().parent.parent
OBS_ROOT = ROOT / "observatory"
RUNS_DIR = OBS_ROOT / "training_runs"
BASELINES_FILE = OBS_ROOT / "baselines.json"
PLOTS_DIR = ROOT / "plots"

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

body, .gradio-container { 
    font-family: 'Inter', sans-serif !important; 
    background-color: #0b0f19 !important; 
    color: #e2e8f0 !important; 
}
.glass-panel {
    background: rgba(17, 24, 39, 0.7) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5) !important;
    padding: 24px !important;
}
.glass-header {
    background: linear-gradient(135deg, #1e293b, #0f172a) !important;
    border-bottom: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px 16px 0 0 !important;
    padding: 20px !important;
    margin-bottom: 20px !important;
}
.title-gradient {
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    font-size: 2.5rem !important;
    text-align: center;
    margin: 0 !important;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 8px;
}
.btn-primary {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
}
.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
}
.plot-container {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.05);
}
.st-tabs { border-bottom: 1px solid rgba(255,255,255,0.1) !important; }
.metadata-box {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    padding: 16px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    color: #94a3b8 !important;
}
"""

def _load_curve(stage: str) -> pd.DataFrame:
    path = RUNS_DIR / f"training_curves_{stage}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def plot_training(stage: str):
    frame = _load_curve(stage)
    if frame.empty:
        return px.line(title=f"No curve file found for {stage} — run the training notebook first")
        
    metric_columns = [col for col in ["reward_total", "reward_recall", "reward_success", "reward_memory_write", "recall_before_action_rate"] if col in frame.columns]
    melted = frame.melt(id_vars=["step"], value_vars=metric_columns, var_name="metric", value_name="value")
    
    figure = px.line(melted, x="step", y="value", color="metric", title=f"ShiftLog-Gym Policy Metrics - {stage.upper()}")
    
    figure.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e2e8f0',
        title_font_family='Inter',
        title_font_size=20,
        title_x=0.5,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    figure.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
    figure.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
    figure.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.2)", annotation_text="Target Recall Threshold")
    
    return figure

def load_baseline_table():
    if not BASELINES_FILE.exists():
        return pd.DataFrame([{"status": "No baselines.json found — run the training notebook first"}])
    payload = json.loads(BASELINES_FILE.read_text(encoding="utf-8"))
    
    # Build metadata label
    meta = payload.get("_metadata", {})
    run_id = meta.get("run_id", "pending")
    timestamp = meta.get("timestamp", "")
    
    rows = [
        {"Model Agent": "Random Agent Baseline", **{k: v for k, v in payload.get("random", {}).items() if not k.startswith("_")}},
        {"Model Agent": "Untrained General LLM", **{k: v for k, v in payload.get("llm_base", {}).items() if not k.startswith("_")}},
        {"Model Agent": "✅ Trained ShiftLog Policy (GRPO)", **{k: v for k, v in payload.get("trained_llm", {}).items() if not k.startswith("_")}},
    ]
    return pd.DataFrame(rows)

def load_metadata_text() -> str:
    if not BASELINES_FILE.exists():
        return "_No training metadata yet. Run train/02_grpo_train_colab.ipynb to populate._"
    try:
        payload = json.loads(BASELINES_FILE.read_text(encoding="utf-8"))
        meta = payload.get("_metadata", {})
        if not meta or meta.get("run_id") == "pending":
            return "_Training not yet run. Metadata will appear here after running the GRPO notebook._"
        lines = [
            f"**WandB Run ID:** `{meta.get('run_id', 'N/A')}`",
            f"**Run URL:** {meta.get('run_url', 'N/A')}",
            f"**Timestamp:** {meta.get('timestamp', 'N/A')}",
            f"**Training Steps:** {meta.get('training_steps', 'N/A')}",
            f"**Base Model:** `{meta.get('model', 'N/A')}`",
            f"**Note:** _{meta.get('note', '')}_",
        ]
        return "\n\n".join(lines)
    except Exception as e:
        return f"Error reading metadata: {e}"

def load_eval_table(stage: str):
    path = RUNS_DIR / f"eval_summary_{stage}.csv"
    if not path.exists():
        return pd.DataFrame([{"status": f"No evaluation summary found for {stage} — run the training notebook first"}])
    return pd.read_csv(path)

def load_plots_tab():
    """Load the 3 PNG plots if they exist, return (img1, img2, img3, status_text)."""
    p1 = PLOTS_DIR / "01_reward_curve.png"
    p2 = PLOTS_DIR / "02_recall_bonus_curve.png"
    p3 = PLOTS_DIR / "03_mttr_comparison.png"
    
    img1 = str(p1) if p1.exists() else None
    img2 = str(p2) if p2.exists() else None
    img3 = str(p3) if p3.exists() else None
    
    if all([img1, img2, img3]):
        status = "✅ All 3 training plots loaded from real GRPO run."
    else:
        missing = [p.name for p in [p1, p2, p3] if not p.exists()]
        status = f"⏳ Training in progress — plots will appear here after training completes. Missing: {', '.join(missing)}"
    
    return img1, img2, img3, status

# ----------------- Automated Training Execution Logic -----------------
STATUS_FILE = OBS_ROOT / "training_status.txt"

def get_training_status():
    if not STATUS_FILE.exists():
        return "Not Started. Awaiting execution."
    return STATUS_FILE.read_text(encoding="utf-8").strip()

def trigger_training(hf_repo):
    import subprocess
    import os
    status = get_training_status()
    if any(kw in status for kw in ["Running", "Loading", "Evaluating", "Uploading", "Initializing"]):
        return status
    
    STATUS_FILE.write_text("Preparing decoupled training subprocess...", encoding="utf-8")
    
    # Throttle PyTorch CPU usage so the FastApi Uvicorn server doesn't get starved!
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    
    subprocess.Popen(["python", "run_training.py", str(hf_repo or "")], env=env)
    return "🚀 Training job submitted to detached subprocess. Click 'Check Status' to monitor progress."


# Initialize Interface structure
with gr.Blocks(title="ShiftLog Observatory Explorer", css=custom_css) as demo:
    
    # Header Module
    with gr.Column(elem_classes="glass-header"):
        gr.Markdown("<h1 class='title-gradient'>ShiftLog-Gym Observatory</h1>")
        gr.Markdown("<div class='subtitle'>State-of-the-art RL visualization for SRE Incident Memory Policies</div>")

    # Interactive Tabs
    with gr.Tabs(elem_classes="st-tabs"):
        
        with gr.Tab("📈 Training Progression"):
            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown("### Real-time Policy Optimization Metrics")
                gr.Markdown("Visualize how the LLM agent learns causal resolution patterns across the defined PEFT/GRPO training regimes.")
                with gr.Row():
                    stage_choice = gr.Dropdown(choices=["stageA", "stageB", "stageC"], value="stageC", label="Select Training Phase", scale=1)
                    curve_button = gr.Button("🚀 Render Dashboard Plot", elem_classes="btn-primary", scale=1)
                
                with gr.Column(elem_classes="plot-container"):
                    curve_plot = gr.Plot(label="")
                
                curve_button.click(fn=plot_training, inputs=[stage_choice], outputs=[curve_plot])
                demo.load(fn=plot_training, inputs=[stage_choice], outputs=[curve_plot])

        with gr.Tab("📊 Results & Evidence"):
            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown("### GRPO Training Evidence — Real Plots from Live Run")
                gr.Markdown("These 3 PNG plots are generated from actual environment interactions during GRPO training. They are **not simulated**.")
                
                plots_status = gr.Markdown("Loading...")
                
                with gr.Row():
                    plot_img1 = gr.Image(label="📈 Reward Curve — Total reward over training steps", type="filepath")
                with gr.Row():
                    plot_img2 = gr.Image(label="🧠 R2 Recall Bonus — Memory policy learning curve", type="filepath")
                with gr.Row():
                    plot_img3 = gr.Image(label="⏱️ MTTR Comparison — Before vs after training on linked incidents", type="filepath")
                
                refresh_plots_btn = gr.Button("🔄 Refresh Plots", elem_classes="btn-primary")
                
                gr.Markdown("---")
                gr.Markdown("### 🏷️ Training Metadata")
                metadata_display = gr.Markdown("Loading metadata...")
                
                def refresh_plots_fn():
                    i1, i2, i3, status = load_plots_tab()
                    meta = load_metadata_text()
                    return i1, i2, i3, status, meta
                
                refresh_plots_btn.click(
                    fn=refresh_plots_fn,
                    inputs=[],
                    outputs=[plot_img1, plot_img2, plot_img3, plots_status, metadata_display]
                )
                demo.load(
                    fn=refresh_plots_fn,
                    inputs=[],
                    outputs=[plot_img1, plot_img2, plot_img3, plots_status, metadata_display]
                )

        with gr.Tab("🏆 Baselines & Leaderboard"):
            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown("### Agent Head-to-Head Comparison")
                gr.Markdown("Compare the trained checkpoint against baseline agents. All values from real environment rollouts.")
                
                metadata_label = gr.Markdown("Loading run info...")
                baseline_button = gr.Button("🔄 Refresh Leaderboard", elem_classes="btn-primary")
                baseline_table = gr.Dataframe(label="")
                
                def refresh_leaderboard():
                    meta = load_metadata_text()
                    table = load_baseline_table()
                    return meta, table
                
                baseline_button.click(fn=refresh_leaderboard, inputs=None, outputs=[metadata_label, baseline_table])
                demo.load(fn=refresh_leaderboard, inputs=None, outputs=[metadata_label, baseline_table])

        with gr.Tab("🔬 Held-out Evaluation"):
            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown("### Validation Cohort Summary")
                gr.Markdown("Performance on unseen, noise-injected incident topologies to verify true causal memory integration.")
                with gr.Row():
                    eval_stage_choice = gr.Dropdown(choices=["stageA", "stageB", "stageC"], value="stageC", label="Select Evaluation Checkpoint")
                    eval_button = gr.Button("📊 Load Evaluation Bundle", elem_classes="btn-primary")
                eval_table = gr.Dataframe(label="")
                
                eval_button.click(fn=load_eval_table, inputs=[eval_stage_choice], outputs=[eval_table])
                demo.load(fn=load_eval_table, inputs=[eval_stage_choice], outputs=[eval_table])

        with gr.Tab("⚙️ Engine & Execute Training"):
            with gr.Column(elem_classes="glass-panel"):
                gr.Markdown("### Start HF Space GPU Training Job")
                gr.Markdown(
                    "Click to begin the RL LoRA training pipeline on the current space's hardware. "
                    "**Make sure your space is configured with an L4 or T4 GPU before clicking this, otherwise it will fail.**\n\n"
                    "> ⚠️ Alternatively, run `train/02_grpo_train_colab.ipynb` in Google Colab for an interactive experience with per-cell visibility."
                )
                
                hf_upload_repo = gr.Textbox(
                    label="HF Model Repository (Optional)",
                    placeholder="Chirag0123/shiftlog-gym-qwen-memory-policy",
                    value="Chirag0123/shiftlog-gym-qwen-memory-policy"
                )
                
                status_block = gr.Textbox(label="Training Daemon Status", value=get_training_status(), interactive=False)
                
                with gr.Row():
                    start_btn = gr.Button("🔥 Run Full PEFT/GRPO Train Pipeline", elem_classes="btn-primary")
                    refresh_btn = gr.Button("🔄 Check Status")
                
                start_btn.click(fn=trigger_training, inputs=[hf_upload_repo], outputs=[status_block])
                refresh_btn.click(fn=get_training_status, inputs=[], outputs=[status_block])

if __name__ == "__main__":
    demo.launch()
