# Hugging Face Setup and GPU Plan

## Goal
Run ShiftLog-Gym end-to-end with:
- baseline generation
- staged training artifacts (A/B/C)
- evaluation bundle
- publish to Hub

Use **Jobs for training** and **Spaces for demo UI**.

## Recommended Hardware (for ~$30 budget)
Based on current HF hardware tables (`spaces-gpus` docs):

1. Training start point: `Nvidia T4 - small` for smoke/failure debugging.
2. Main training: `1x Nvidia L4` (best balance for 4-bit LoRA + GRPO attempts).
3. If GRPO is still unstable and you need faster recovery runs: `Nvidia A10G - small`.
4. Demo UI Space: `CPU Basic` (or `CPU Upgrade` if charts feel slow).

## Why this split
- Spaces are best for hosting the environment + UI.
- Jobs are better for isolated, reproducible training runs and stop billing when complete.

## Step-by-step

1. Create two Hub repos:
- `space`: `your-user/shiftlog-gym`
- `model`: `your-user/shiftlog-gym-qwen-memory-policy`

2. Push this repo to GitHub `main`.

3. In HF Space settings:
- SDK: `gradio`
- Hardware: `CPU Basic` first
- Sleep: enabled

4. Upload secrets to Space/Job:
- `HF_TOKEN`
- `WANDB_API_KEY` (optional but recommended)

5. Run baseline notebook first:
- `train/01_env_smoke_test.ipynb`
- confirm artifacts in `observatory/`:
  - `baselines*.json`
  - `baseline_comparison.png`

6. Run training notebook:
- `train/02_grpo_train_colab.ipynb`
- this now uses `train/colab_training_pipeline.py`
- enter both keys when prompted
- verify:
  - `observatory/training_runs/training_curves_stageA.csv`
  - `observatory/training_runs/training_curves_stageB.csv`
  - `observatory/training_runs/training_curves_stageC.csv`
  - non-empty `outputs/grpo-stageb/`, `outputs/grpo-stagec/`

7. Run eval/publish notebook:
- `train/03_eval_publish_colab.ipynb`
- this uses `train/colab_eval_publish.py`
- verify:
  - `artifacts/eval_publish/summary_metrics.csv`
  - `artifacts/eval_publish/before_after_comparison.csv`
  - `artifacts/eval_publish/plots/*.png`

8. Enable upload toggles:
- set `PUBLISH_TO_HF=True` in notebook 2 and/or 3
- set `HF_MODEL_REPO=your-user/shiftlog-gym-qwen-memory-policy`

9. Launch Gradio observatory locally or in a Space:
```bash
python observatory/gradio_app.py
```

10. Launch API server locally:
```bash
uvicorn shiftlog_gym.server.app:app --port 7860
```

## Notes on reliability
- If you see `unsloth_compiled_cache` in stack traces, restart runtime and rerun from install cell.
- If GRPO fails in that runtime, notebook 2 falls back to SFT policy learning so stage artifacts are still produced.
- Use `SHIFTLOG_USE_BF16=1` only on hardware where bf16 is stable for your run.
