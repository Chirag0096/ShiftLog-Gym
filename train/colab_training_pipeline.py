from __future__ import annotations

import inspect
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import HfApi, login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from shiftlog_gym.scenarios import PUBLIC_FAMILIES
from shiftlog_gym.simulator import ShiftLogSimulator
from shiftlog_gym.training import summarize_baseline, summarize_episode, write_episode_replays
from shiftlog_gym.trl_env import (
    ShiftLogToolEnv,
    reward_efficiency as reward_efficiency_env,
    reward_hallucination as reward_hallucination_env,
    reward_handoff as reward_handoff_env,
    reward_memory_integrity as reward_memory_integrity_env,
    reward_memory_write as reward_memory_write_env,
    reward_noise_resistance as reward_noise_resistance_env,
    reward_recall as reward_recall_env,
    reward_success as reward_success_env,
    reward_total as reward_total_env,
)


@dataclass(slots=True)
class StageConfig:
    name: str
    families: tuple[str, ...]
    max_steps: int
    rollout_mode: str


class ColabTrainingPipeline:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required. Switch Colab runtime to T4/L4/A10G.")

        self.obs_root = Path("observatory")
        self.runs_dir = self.obs_root / "training_runs"
        self.episodes_dir = self.obs_root / "episodes"
        self.outputs_dir = Path("outputs")
        for path in (self.obs_root, self.runs_dir, self.episodes_dir, self.outputs_dir):
            path.mkdir(parents=True, exist_ok=True)

        self.wandb_enabled = False
        self.hf_enabled = False
        self.model = None
        self.tokenizer = None
        self.use_bf16 = False
        self.use_fp16 = True

        self.stage_b = StageConfig("stageB", ("db_pool", "auth_cascade", "oom_regression"), 200, "short")
        self.stage_c = StageConfig("stageC", tuple(PUBLIC_FAMILIES), 300, "full")

    def authenticate(self, wandb_key: str = "", hf_token: str = "") -> None:
        wandb_key = wandb_key.strip() or os.environ.get("WANDB_API_KEY", "").strip()
        hf_token = hf_token.strip() or os.environ.get("HF_TOKEN", "").strip()

        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key
            import wandb

            wandb.login(key=wandb_key)
            self.wandb_enabled = True

        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            login(token=hf_token)
            self.hf_enabled = True

    def assert_clean_trl_runtime(self) -> None:
        module_path = inspect.getsourcefile(GRPOTrainer) or ""
        if "unsloth" in module_path.lower() or "unsloth" in (GRPOTrainer.__module__ or "").lower():
            raise RuntimeError(
                "Detected unsloth-patched GRPOTrainer. Restart runtime after uninstalling unsloth/unsloth_zoo."
            )

    def load_model(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> None:
        # FORCE bf16 automatically to avoid GradScaler fp16 crashes with natively bfloat16 models
        self.use_bf16 = bool(torch.cuda.is_bf16_supported())
        self.use_fp16 = not self.use_bf16

        compute_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=compute_dtype,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        peft_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.model = get_peft_model(model, peft_cfg)
        self.model.print_trainable_parameters()
        self.tokenizer = tokenizer

    def build_stage_dataset(self, families: tuple[str, ...], steps: int, seed_offset: int) -> Dataset:
        rows = []
        for step in range(steps):
            family = families[step % len(families)]
            rows.append(
                {
                    "prompt": [
                        {
                            "role": "user",
                            "content": (
                                f"Handle incident family={family}, seed={seed_offset + step}. "
                                "Return exactly one JSON tool call."
                            ),
                        }
                    ],
                    "family": family,
                    "variant_index": step % 6,
                    "seed": seed_offset + step,
                }
            )
        return Dataset.from_list(rows)

    def _resolve_envs(self, *args, **kwargs):
        envs = kwargs.get("environments")
        if envs is None and args and isinstance(args[0], list):
            envs = args[0]
        return envs

    def _safe_reward_call(self, fn, *args, **kwargs):
        envs = self._resolve_envs(*args, **kwargs)
        completions = kwargs.get("completions") or []
        if envs is None:
            return [0.0] * len(completions)
        return fn(envs, **kwargs)

    def reward_total_safe(self, *args, **kwargs):
        return self._safe_reward_call(reward_total_env, *args, **kwargs)

    def reward_success_safe(self, *args, **kwargs):
        return self._safe_reward_call(reward_success_env, *args, **kwargs)

    def reward_recall_safe(self, *args, **kwargs):
        return self._safe_reward_call(reward_recall_env, *args, **kwargs)

    def reward_memory_write_safe(self, *args, **kwargs):
        return self._safe_reward_call(reward_memory_write_env, *args, **kwargs)

    def reward_memory_integrity_safe(self, *args, **kwargs):
        return self._safe_reward_call(reward_memory_integrity_env, *args, **kwargs)

    def reward_efficiency_safe(self, *args, **kwargs):
        return self._safe_reward_call(reward_efficiency_env, *args, **kwargs)

    def reward_hallucination_safe(self, *args, **kwargs):
        return self._safe_reward_call(reward_hallucination_env, *args, **kwargs)

    def reward_noise_resistance_safe(self, *args, **kwargs):
        return self._safe_reward_call(reward_noise_resistance_env, *args, **kwargs)

    def reward_handoff_safe(self, *args, **kwargs):
        return self._safe_reward_call(reward_handoff_env, *args, **kwargs)

    def save_curve(self, log_history: list[dict[str, Any]], stage_name: str) -> Path:
        rows: list[dict[str, Any]] = []
        for item in log_history:
            if "step" not in item:
                continue
            rows.append(
                {
                    "step": item.get("step", 0),
                    "reward_total": item.get("reward_total", item.get("reward", 0.0)),
                    "reward_recall": item.get("reward_recall", 0.0),
                    "reward_success": item.get("reward_success", 0.0),
                    "reward_memory_write": item.get("reward_memory_write", 0.0),
                    "reward_memory_integrity": item.get("reward_memory_integrity", 0.0),
                    "recall_before_action_rate": item.get("recall_before_action_rate", 0.0),
                }
            )
        if not rows:
            rows = [
                {
                    "step": 0,
                    "reward_total": 0.0,
                    "reward_recall": 0.0,
                    "reward_success": 0.0,
                    "reward_memory_write": 0.0,
                    "reward_memory_integrity": 0.0,
                    "recall_before_action_rate": 0.0,
                }
            ]
        frame = pd.DataFrame(rows).sort_values("step")
        csv_path = self.runs_dir / f"training_curves_{stage_name}.csv"
        png_path = self.runs_dir / f"training_curves_{stage_name}.png"
        frame.to_csv(csv_path, index=False)

        fig, axis = plt.subplots(figsize=(10, 4))
        for metric in ("reward_total", "reward_recall", "reward_success", "reward_memory_write", "recall_before_action_rate"):
            axis.plot(frame["step"], frame[metric], label=metric)
        axis.set_title(f"{stage_name.upper()} curves")
        axis.set_xlabel("step")
        axis.set_ylabel("score")
        axis.legend(loc="best")
        fig.tight_layout()
        fig.savefig(png_path, dpi=180)
        return csv_path

    def _write_stage_marker(self, stage_name: str, mode: str, error: str = "") -> Path:
        out_dir = self.outputs_dir / f"grpo-{stage_name.lower()}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_meta.json").write_text(
            json.dumps({"stage": stage_name, "mode": mode, "error": error}, indent=2),
            encoding="utf-8",
        )
        return out_dir

    def run_stage_a(self, enabled: bool = True) -> None:
        if not enabled:
            return

        examples = []
        for idx in range(50):
            family = PUBLIC_FAMILIES[idx % len(PUBLIC_FAMILIES)]
            examples.append(
                {
                    "text": (
                        f"Family={family}. Return first safe JSON tool call.\n"
                        + json.dumps({"tool": "read_shift_log", "arguments": {"query": family, "limit": 3}})
                    )
                }
            )
        dataset = Dataset.from_list(examples)

        config = SFTConfig(
            output_dir="outputs/stage-a-sft",
            max_steps=50,
            learning_rate=5e-6,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            logging_steps=5,
            save_steps=25,
            report_to=["wandb"] if self.wandb_enabled else [],
            dataset_text_field="text",
            completion_only_loss=False,
            max_length=192,
            packing=False,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            fp16=self.use_fp16,
            bf16=self.use_bf16,
        )

        try:
            trainer = SFTTrainer(model=self.model, args=config, train_dataset=dataset, processing_class=self.tokenizer)
            trainer.train()
            trainer.save_model("outputs/stage-a-sft")
            self.tokenizer.save_pretrained("outputs/stage-a-sft")
            self._write_stage_marker("stageA", "sft")
            self.save_curve(trainer.state.log_history, "stageA")
        except Exception as error:
            tokenized = dataset.map(lambda x: self.tokenizer(x["text"], truncation=True, max_length=192), remove_columns=["text"])
            collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
            args = TrainingArguments(
                output_dir="outputs/stage-a-sft",
                max_steps=50,
                learning_rate=5e-6,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                logging_steps=5,
                save_steps=25,
                report_to=["wandb"] if self.wandb_enabled else [],
                remove_unused_columns=False,
                gradient_checkpointing=True,
                optim="paged_adamw_8bit",
                fp16=self.use_fp16,
                bf16=self.use_bf16,
            )
            trainer = Trainer(model=self.model, args=args, train_dataset=tokenized, data_collator=collator)
            trainer.train()
            trainer.save_model("outputs/stage-a-sft")
            self.tokenizer.save_pretrained("outputs/stage-a-sft")
            self._write_stage_marker("stageA", "trainer_fallback", str(error))
            self.save_curve(trainer.state.log_history, "stageA")

    def run_stage_grpo(self, stage: StageConfig) -> dict[str, str]:
        dataset = self.build_stage_dataset(stage.families, stage.max_steps, 1000 if stage.name == "stageB" else 2000)
        output_dir = f"outputs/grpo-{stage.name.lower()}"
        config = GRPOConfig(
            output_dir=output_dir,
            max_steps=stage.max_steps,
            learning_rate=5e-6,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=2,
            max_completion_length=384 if stage.name == "stageB" else 768,
            logging_steps=10,
            save_steps=50,
            report_to=["wandb"] if self.wandb_enabled else [],
            log_completions=True,
            gradient_checkpointing=True,
            fp16=self.use_fp16,
            bf16=self.use_bf16,
        )

        try:
            trainer = GRPOTrainer(
                model=self.model,
                args=config,
                train_dataset=dataset,
                processing_class=self.tokenizer,
                environment_factory=lambda **kwargs: ShiftLogToolEnv(rollout_mode=stage.rollout_mode, multi_shift=False),
                reward_funcs=[
                    self.reward_total_safe,
                    self.reward_success_safe,
                    self.reward_recall_safe,
                    self.reward_memory_write_safe,
                    self.reward_memory_integrity_safe,
                    self.reward_efficiency_safe,
                    self.reward_hallucination_safe,
                    self.reward_noise_resistance_safe,
                    self.reward_handoff_safe,
                ],
            )
            trainer.train()
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            self._write_stage_marker(stage.name, "grpo")
            self.save_curve(trainer.state.log_history, stage.name)
            return {"stage": stage.name, "mode": "grpo", "error": ""}
        except Exception as error:
            fallback = [
                {
                    "text": (
                        f"Incident family={stage.families[idx % len(stage.families)]}. "
                        "Output safest first tool-call JSON.\n"
                        + json.dumps(
                            {
                                "tool": "read_shift_log",
                                "arguments": {"query": stage.families[idx % len(stage.families)], "limit": 3},
                            }
                        )
                    )
                }
                for idx in range(stage.max_steps)
            ]
            fallback_ds = Dataset.from_list(fallback)
            fallback_cfg = SFTConfig(
                output_dir=output_dir,
                max_steps=stage.max_steps,
                learning_rate=5e-6,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                logging_steps=10,
                save_steps=50,
                report_to=["wandb"] if self.wandb_enabled else [],
                dataset_text_field="text",
                completion_only_loss=False,
                max_length=192,
                packing=False,
                gradient_checkpointing=True,
                optim="paged_adamw_8bit",
                fp16=self.use_fp16,
                bf16=self.use_bf16,
            )
            trainer = SFTTrainer(model=self.model, args=fallback_cfg, train_dataset=fallback_ds, processing_class=self.tokenizer)
            trainer.train()
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            self._write_stage_marker(stage.name, "sft_fallback", str(error))
            self.save_curve(trainer.state.log_history, stage.name)
            return {"stage": stage.name, "mode": "sft_fallback", "error": str(error)}

    def parse_action(self, text: str, service_hint: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {"tool": "inspect_service", "arguments": {"service": service_hint}}
        try:
            payload = json.loads(match.group(0))
        except Exception:
            return {"tool": "inspect_service", "arguments": {"service": service_hint}}
        if not isinstance(payload, dict) or "tool" not in payload:
            return {"tool": "inspect_service", "arguments": {"service": service_hint}}
        payload.setdefault("arguments", {})
        return payload

    def rollout_eval(self, tag: str, families: tuple[str, ...], episodes: int = 20, max_steps: int = 18) -> tuple[pd.DataFrame, dict[str, Any]]:
        rows = []
        replays = []
        for idx in range(episodes):
            family = families[idx % len(families)]
            simulator = ShiftLogSimulator()
            simulator.reset(seed=8000 + idx, family=family, variant_index=7)

            for _ in range(max_steps):
                if simulator.done:
                    break
                incident = simulator.active_incident
                if incident is None:
                    break
                prompt = simulator.last_observation + '\nReturn one JSON tool call {"tool":...,"arguments":...}.'
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                output = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
                text = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                action = self.parse_action(text, incident.service)
                tool = action.get("tool", "inspect_service")
                args = action.get("arguments", {})
                if not hasattr(simulator, tool):
                    tool, args = "inspect_service", {"service": incident.service}
                try:
                    getattr(simulator, tool)(**args)
                except Exception:
                    simulator.inspect_service(incident.service)

            artifacts = summarize_episode(simulator, f"{tag}-{family}-{idx:03d}", "eval", 8000 + idx, 7)
            rows.append(artifacts.episode_row)
            replays.append(artifacts)
        write_episode_replays(self.episodes_dir, replays)
        return pd.DataFrame(rows), summarize_baseline(rows)

    def evaluate_and_write(self) -> dict[str, Any]:
        summaries: dict[str, Any] = {}
        for stage_name, families in (
            ("stageA", self.stage_b.families),
            ("stageB", self.stage_b.families),
            ("stageC", self.stage_c.families),
        ):
            frame, summary = self.rollout_eval(stage_name, families, episodes=20, max_steps=18)
            frame.to_csv(self.runs_dir / f"eval_summary_{stage_name}.csv", index=False)
            summaries[stage_name] = summary

        baselines_path = self.obs_root / "baselines.json"
        payload = {"random": {}, "scripted": {}, "llm_base": {}, "trained_llm": {}}
        if baselines_path.exists():
            payload = json.loads(baselines_path.read_text(encoding="utf-8"))
        payload["trained_llm"] = summaries.get("stageC", {})
        baselines_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return summaries

    def artifact_status(self) -> list[tuple[str, str]]:
        required = [
            self.runs_dir / "training_curves_stageA.csv",
            self.runs_dir / "training_curves_stageB.csv",
            self.runs_dir / "training_curves_stageC.csv",
            self.runs_dir / "eval_summary_stageA.csv",
            self.runs_dir / "eval_summary_stageB.csv",
            self.runs_dir / "eval_summary_stageC.csv",
            Path("outputs/grpo-stageb"),
            Path("outputs/grpo-stagec"),
        ]
        status = []
        for path in required:
            if path.is_dir():
                files = [item for item in path.rglob("*") if item.is_file()]
                status.append((str(path), "OK" if files else "EMPTY"))
            else:
                status.append((str(path), "OK" if path.exists() else "MISSING"))
        return status

    def upload_to_hf(self, repo_id: str) -> None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN missing.")
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="model", folder_path="outputs/grpo-stagec", path_in_repo="adapter")
        api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(self.runs_dir), path_in_repo="training_runs")
