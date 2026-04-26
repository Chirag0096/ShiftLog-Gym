"""
Detached training subprocess for ShiftLog-Gym.
Spawned by observatory/gradio_app.py via subprocess.Popen.
Writes progress to observatory/training_status.txt (absolute path).
"""
import sys
import traceback
from pathlib import Path

# Always resolve paths relative to THIS file's location (project root)
ROOT = Path(__file__).resolve().parent
STATUS_FILE = ROOT / "observatory" / "training_status.txt"


def update_status(msg: str):
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(msg, encoding="utf-8")
    print(f"[STATUS] {msg}", flush=True)


def main():
    hf_repo = sys.argv[1].strip() if len(sys.argv) > 1 else ""

    try:
        update_status("Initializing Training Subprocess...")

        update_status("Importing torch...")
        import torch
        if not torch.cuda.is_available():
            update_status(
                "❌ ERROR: No CUDA GPU detected.\n"
                "Go to Space Settings → Hardware → select Nvidia L4 or T4, then try again."
            )
            return

        gpu_name = torch.cuda.get_device_name(0)
        update_status(f"✅ GPU detected: {gpu_name}. Importing transformers (slow, ~60s)...")

        # Suppress slow wandb/trl network calls
        import os
        os.environ.setdefault("WANDB_MODE", "disabled")
        os.environ.setdefault("WANDB_SILENT", "true")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        update_status(f"✅ GPU: {gpu_name}. Importing transformers...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        update_status(f"✅ GPU: {gpu_name}. Importing peft...")
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        update_status(f"✅ GPU: {gpu_name}. Importing trl (may take 30-60s)...")
        from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

        update_status(f"✅ GPU: {gpu_name}. Importing datasets + pipeline...")
        # Add project root to sys.path for imports
        sys.path.insert(0, str(ROOT))
        from train.colab_training_pipeline import ColabTrainingPipeline

        update_status("All imports done. Authenticating with Hugging Face Hub...")
        pipeline = ColabTrainingPipeline()
        pipeline.authenticate()

        update_status("Loading Qwen2.5-1.5B model into GPU (4-bit NF4, bf16)...")
        pipeline.load_model()

        update_status("✅ Model loaded! Stage A: SFT Format Warmup (50 steps)...")
        pipeline.run_stage_a()

        update_status("Stage B: GRPO Short Rollout (200 steps) — reward signal warming...")
        pipeline.run_stage_grpo(pipeline.stage_b)

        # Generate intermediate plots after Stage B
        try:
            update_status("Generating intermediate training plots after Stage B...")
            pipeline.generate_hackathon_plots()
        except Exception as plot_err:
            update_status(f"Stage B plots skipped (non-fatal): {plot_err}")

        update_status("Stage C: GRPO Full Rollout (300 steps) — full policy optimization...")
        pipeline.run_stage_grpo(pipeline.stage_c)

        update_status("Evaluating final policy against baselines...")
        pipeline.evaluate_and_write()

        if hf_repo:
            update_status(f"Uploading LoRA adapter weights to {hf_repo}...")
            pipeline.upload_to_hf(repo_id=hf_repo)
            update_status(f"✅ Done! Weights uploaded to {hf_repo}. Click 'Refresh Plots' in Results tab.")
        else:
            update_status("✅ Training complete! Click '🔄 Refresh Plots' in the Results & Evidence tab.")

    except Exception as e:
        tb = traceback.format_exc()
        update_status(f"❌ ERROR: {e}\n\nTraceback:\n{tb[-1200:]}")


if __name__ == "__main__":
    main()
