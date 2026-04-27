"""
space_training_daemon.py
Runs inside HuggingFace Space with GPU hardware.
Called as a background thread from the Gradio dashboard.
Saves all outputs to HF Hub (model repo) for persistence.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, login

# ─── State visible to the Gradio UI ───────────────────────────────────────────
_lock = threading.Lock()
_state: dict[str, Any] = {
    "status": "idle",          # idle | running | complete | error
    "stage": "",               # stageA | stageB | stageC | eval | upload
    "step": 0,
    "total_steps": 550,        # 50 + 200 + 300
    "reward_total": 0.0,
    "reward_recall": 0.0,
    "recall_rate": 0.0,
    "message": "",
    "error": "",
    "started_at": "",
    "completed_at": "",
    "wandb_url": "",
    "model_url": "",
    "log_lines": [],           # last 50 log lines for the UI
}

def get_state() -> dict[str, Any]:
    with _lock:
        return dict(_state)

def _update(**kwargs):
    with _lock:
        _state.update(kwargs)
        if "message" in kwargs:
            msg = kwargs["message"]
            timestamp = datetime.now().strftime("%H:%M:%S")
            _state["log_lines"] = (_state["log_lines"] + [f"[{timestamp}] {msg}"])[-50:]
            print(f"[DAEMON] {msg}") # Also print to container logs


# ─── Main training entry point ─────────────────────────────────────────────────
def run_training() -> None:
    """
    Called in a background thread by the Gradio dashboard.
    Runs all 3 stages, evaluates, uploads everything to HF Hub.
    """
    if get_state()["status"] == "running":
        _update(message="Training already running — ignoring duplicate call")
        return

    # Reset state for new run
    _update(
        status="running", 
        stage="init",
        step=0,
        reward_total=0.0,
        reward_recall=0.0,
        recall_rate=0.0,
        error="",
        started_at=datetime.now().isoformat(),
        completed_at="",
        message="Training session initialized"
    )

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No GPU detected. Go to Space Settings → Space hardware → "
                "select L4 x1 → Save. Then restart the Space."
            )
        
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        _update(message=f"✅ GPU: {gpu_name} ({vram_gb:.1f}GB VRAM)")

        _setup_env()
        _authenticate()
        
        # 1. Stage A
        _run_stage_a()
        _incremental_upload("stage_a")
        
        # 2. Stage B
        _run_stage_b()
        _incremental_upload("stage_b")
        
        # 3. Stage C
        _run_stage_c()
        _incremental_upload("stage_c") # Final LoRA adapter pushed BEFORE eval
        
        # 4. Evaluation & Plots
        _run_evaluation()
        _generate_plots()
        
        # 5. Final Sync
        _upload_to_hub()
        _commit_to_space()

        _update(
            status="complete", 
            completed_at=datetime.now().isoformat(),
            message="✅ All stages complete. Model weights, plots, and baselines pushed to repositories."
        )

    except Exception as e:
        _update(
            status="error", 
            error=str(e),
            message=f"❌ Training failed: {e}\n{traceback.format_exc()[-500:]}"
        )


# ─── Setup ─────────────────────────────────────────────────────────────────────
def _setup_env() -> None:
    _update(message="Setting up environment paths...")
    hf_home = os.environ.get("HF_HOME", "/tmp/hf_cache")
    os.makedirs(hf_home, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = hf_home
    
    # HF Spaces has 50GB ephmeral disk. We use /tmp or persistent /data
    base = Path("/data/shiftlog_training") if Path("/data").exists() else Path("/tmp/shiftlog_training")
    base.mkdir(parents=True, exist_ok=True)
    os.environ["SHIFTLOG_TRAINING_DIR"] = str(base)
    _update(message=f"Local outputs directory: {base}")


def _authenticate() -> None:
    _update(message="Authenticating services...")

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("HF_TOKEN secret not set. Go to Space Settings → Variables and secrets.")
    login(token=hf_token)

    wandb_key = os.environ.get("WANDB_API_KEY", "").strip()
    if wandb_key:
        import wandb
        wandb.login(key=wandb_key)
        run = wandb.init(
            project="shiftlog-gym",
            name=f"space-gpu-run-{datetime.now().strftime('%m%d-%H%M')}",
            config={
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "hardware": "HF Space GPU",
                "stages": "A(50) + B(200) + C(300)",
                "hackathon": "Meta PyTorch OpenEnv 2026",
            },
            tags=["shiftlog", "grpo", "sre", "hf-space"],
        )
        _update(wandb_url=run.url, message=f"WandB online: {run.url}")
    else:
        _update(message="⚠️ WANDB_API_KEY not set — logging offline")


# ─── Stages ────────────────────────────────────────────────────────────────────
def _run_stage_a() -> None:
    _update(stage="stageA", message="🚀 Stage A: SFT format warmup (50 steps)...")
    pipe = _get_pipeline()
    pipe.run_stage_a(enabled=True)
    _update(step=50, message="Stage A complete ✅")


def _run_stage_b() -> None:
    _update(stage="stageB", message="🚀 Stage B: GRPO short rollout (200 steps)...")
    pipe = _get_pipeline()
    result = pipe.run_stage_grpo(pipe.stage_b)
    _update(step=250, message=f"Stage B complete (mode: {result.get('mode')}) ✅")


def _run_stage_c() -> None:
    _update(stage="stageC", message="🚀 Stage C: GRPO full rollout (300 steps)...")
    pipe = _get_pipeline()
    result = pipe.run_stage_grpo(pipe.stage_c)
    _update(step=550, message=f"Stage C complete (mode: {result.get('mode')}) ✅")


# ─── Evaluation ────────────────────────────────────────────────────────────────
def _run_evaluation() -> None:
    _update(stage="eval", message="📊 Running post-training evaluation (60 episodes)...")
    pipe = _get_pipeline()
    summaries = pipe.evaluate_and_write()
    trained = summaries.get("stageC", {})
    _update(
        recall_rate=trained.get("recall_before_action_rate", 0.0),
        reward_total=trained.get("avg_total_reward", 0.0),
        message=f"Eval complete! Recall rate: {trained.get('recall_before_action_rate', 0):.3f} ✅"
    )


def _generate_plots() -> None:
    _update(message="Generating final hackathon evidence plots...")
    pipe = _get_pipeline()
    pipe.generate_hackathon_plots()
    plots = [p.name for p in pipe.plots_dir.glob("*.png")]
    _update(message=f"Generated {len(plots)} plots: {', '.join(plots)}")


# ─── Hub Uploads ──────────────────────────────────────────────────────────────
MODEL_REPO = "Chirag0123/shiftlog-gym-qwen-memory-policy"

def _incremental_upload(tag: str) -> None:
    """Upload intermediate artifacts after each stage for crash-resilience."""
    _update(message=f"Performing incremental upload for {tag}...")
    pipe = _get_pipeline()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    
    try:
        if tag == "stage_a":
            folder = pipe.outputs_dir / "stage-a-sft"
            if folder.exists():
                api.upload_folder(repo_id=MODEL_REPO, folder_path=str(folder), path_in_repo="stage_a_sft")
        
        elif tag == "stage_b":
            folder = pipe.outputs_dir / "grpo-stageb"
            if folder.exists():
                api.upload_folder(repo_id=MODEL_REPO, folder_path=str(folder), path_in_repo="stage_b")
            curve = pipe.runs_dir / "training_curves_stageB.csv"
            if curve.exists():
                api.upload_file(path_or_fileobj=str(curve), path_in_repo="training_runs/training_curves_stageB.csv", repo_id=MODEL_REPO)
        
        elif tag == "stage_c":
            # PUSH FINAL ADAPTER
            folder = pipe.outputs_dir / "grpo-stagec"
            if folder.exists():
                api.upload_folder(repo_id=MODEL_REPO, folder_path=str(folder), path_in_repo="adapter")
            curve = pipe.runs_dir / "training_curves_stageC.csv"
            if curve.exists():
                api.upload_file(path_or_fileobj=str(curve), path_in_repo="training_runs/training_curves_stageC.csv", repo_id=MODEL_REPO)
            
        _update(message=f"Incremental upload for {tag} successful.")
    except Exception as e:
        _update(message=f"⚠️ Incremental upload for {tag} skipped: {e}")


def _upload_to_hub() -> None:
    _update(stage="upload", message="Finalizing HF Hub model repository...")
    pipe = _get_pipeline()
    pipe.create_model_card()
    pipe.upload_to_hf(MODEL_REPO)
    _update(model_url=f"https://huggingface.co/{MODEL_REPO}", message="Hub upload complete ✅")


def _commit_to_space() -> None:
    """Push plots and baselines.json back to the Space repository."""
    _update(message="Committing evidence back to Space repo...")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    SPACE_REPO = "Chirag0123/shiftlog-gym"
    pipe = _get_pipeline()

    # Required files
    files_to_sync = [
        (pipe.plots_dir / "01_reward_curve.png", "plots/01_reward_curve.png"),
        (pipe.plots_dir / "02_recall_bonus_curve.png", "plots/02_recall_bonus_curve.png"),
        (pipe.plots_dir / "03_mttr_comparison.png", "plots/03_mttr_comparison.png"),
        (pipe.obs_root / "baselines.json", "observatory/baselines.json"),
    ]

    for local_path, hub_path in files_to_sync:
        if local_path.exists():
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=hub_path,
                repo_id=SPACE_REPO,
                repo_type="space",
                commit_message=f"Sync training artifact: {hub_path}"
            )
            _update(message=f"Pushed {hub_path} to Space.")

    _update(message="Space repo sync complete ✅")


# ─── Pipeline singleton ─────────────────────────────────────────────────────────
_pipeline_instance = None
_pipeline_lock = threading.Lock()

def _get_pipeline():
    global _pipeline_instance
    with _pipeline_lock:
        if _pipeline_instance is None:
            _update(message="Initializing ColabTrainingPipeline (Loading Qwen2.5-1.5B)...")
            from train.colab_training_pipeline import ColabTrainingPipeline
            _pipeline_instance = ColabTrainingPipeline()
            _pipeline_instance.wandb_enabled = bool(os.environ.get("WANDB_API_KEY"))
            _pipeline_instance.hf_enabled = bool(os.environ.get("HF_TOKEN"))
            _pipeline_instance.load_model("Qwen/Qwen2.5-1.5B-Instruct")
        return _pipeline_instance
