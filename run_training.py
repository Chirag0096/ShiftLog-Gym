import sys
import importlib.util
from pathlib import Path

def main():
    hf_repo = sys.argv[1] if len(sys.argv) > 1 else ""
    status_file = Path("observatory/training_status.txt")
    
    def update_status(msg):
        status_file.write_text(msg, encoding="utf-8")
        
    try:
        update_status("Initializing Subprocess...")
        import torch
        if not torch.cuda.is_available():
            update_status("ERROR: No CUDA GPU available. Go to Space Settings → Hardware and select Nvidia T4 or L4.")
            return

        from train.colab_training_pipeline import ColabTrainingPipeline

        update_status("Loading ColabTrainingPipeline & Authenticating...")
        pipeline = ColabTrainingPipeline()
        pipeline.authenticate()

        update_status("Loading Qwen Model into memory (4-bit BF16)...")
        pipeline.load_model()

        update_status("Running Stage A (SFT Format Warmup)...")
        pipeline.run_stage_a()

        update_status("Running Stage B (GRPO Short Rollout — 200 steps)...")
        pipeline.run_stage_grpo(pipeline.stage_b)

        # Plot intermediate results for dashboard
        try:
            pipeline.generate_hackathon_plots()
        except:
            pass

        update_status("Running Stage C (GRPO Full Rollout — 300 steps)...")
        pipeline.run_stage_grpo(pipeline.stage_c)

        update_status("Evaluating & Summarizing Results...")
        pipeline.evaluate_and_write()

        if hf_repo and hf_repo.strip():
            update_status(f"Uploading weights to HF Model Hub at {hf_repo}...")
            pipeline.upload_to_hf(repo_id=hf_repo)

        update_status("✅ Done! Training successfully completed. Refresh the Results tab to see plots.")
    except Exception as e:
        update_status(f"ERROR: {str(e)}")

if __name__ == "__main__":
    main()
