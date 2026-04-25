from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import HfApi, login


class EvalPublishPipeline:
    def __init__(self) -> None:
        self.obs_root = Path("observatory")
        self.runs_dir = self.obs_root / "training_runs"
        self.output_dir = Path("artifacts/eval_publish")
        self.plots_dir = self.output_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.frames: dict[str, pd.DataFrame] = {}

    def authenticate(self, hf_token: str = "", wandb_key: str = "") -> None:
        hf_token = hf_token.strip() or os.environ.get("HF_TOKEN", "").strip()
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            login(token=hf_token)
        if wandb_key.strip():
            os.environ["WANDB_API_KEY"] = wandb_key.strip()
            import wandb

            wandb.login(key=wandb_key.strip())

    def load_curves(self) -> list[str]:
        required = {
            "stageA": self.runs_dir / "training_curves_stageA.csv",
            "stageB": self.runs_dir / "training_curves_stageB.csv",
            "stageC": self.runs_dir / "training_curves_stageC.csv",
        }
        missing = [name for name, path in required.items() if not path.exists()]
        self.frames = {name: pd.read_csv(path) for name, path in required.items() if path.exists()}
        return missing

    def plot_curves(self) -> list[Path]:
        paths: list[Path] = []
        for stage, frame in self.frames.items():
            figure, axis = plt.subplots(figsize=(10, 4))
            for metric in ("reward_total", "reward_recall", "reward_success", "reward_memory_write", "recall_before_action_rate"):
                if metric in frame.columns:
                    axis.plot(frame["step"], frame[metric], label=metric)
            axis.set_title(f"{stage.upper()} training curves")
            axis.set_xlabel("step")
            axis.set_ylabel("score")
            axis.legend(loc="best")
            figure.tight_layout()
            path = self.plots_dir / f"{stage}_curves.png"
            figure.savefig(path, dpi=180)
            paths.append(path)
        return paths

    def write_tables(self) -> tuple[Path, Path]:
        summary_rows = []
        for stage, frame in self.frames.items():
            if frame.empty:
                continue
            summary_rows.append(
                {
                    "stage": stage,
                    "last_reward_total": float(frame["reward_total"].iloc[-1]) if "reward_total" in frame else 0.0,
                    "last_reward_recall": float(frame["reward_recall"].iloc[-1]) if "reward_recall" in frame else 0.0,
                    "last_reward_success": float(frame["reward_success"].iloc[-1]) if "reward_success" in frame else 0.0,
                    "last_recall_before_action_rate": float(frame["recall_before_action_rate"].iloc[-1])
                    if "recall_before_action_rate" in frame
                    else 0.0,
                }
            )
        summary = pd.DataFrame(summary_rows)
        summary_path = self.output_dir / "summary_metrics.csv"
        summary.to_csv(summary_path, index=False)

        baselines_path = self.obs_root / "baselines.json"
        if baselines_path.exists():
            baselines = json.loads(baselines_path.read_text(encoding="utf-8"))
        else:
            baselines = {"random": {}, "scripted": {}, "llm_base": {}, "trained_llm": {}}
        comparison = pd.DataFrame(
            [
                {"agent": "Random Agent", **baselines.get("random", {})},
                {"agent": "Scripted Agent", **baselines.get("scripted", {})},
                {"agent": "Untrained LLM", **baselines.get("llm_base", {})},
                {"agent": "Trained LLM", **baselines.get("trained_llm", {})},
            ]
        )
        comparison_path = self.output_dir / "before_after_comparison.csv"
        comparison.to_csv(comparison_path, index=False)

        readme = (
            "# ShiftLog-Gym Evaluation Artifacts\n"
            "This folder stores Stage A/B/C curves, summary metrics, and baseline comparison outputs."
        )
        (self.output_dir / "README.md").write_text(readme, encoding="utf-8")
        return summary_path, comparison_path

    def upload_bundle(self, repo_id: str) -> None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN missing.")
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(self.output_dir), path_in_repo="eval_bundle")
