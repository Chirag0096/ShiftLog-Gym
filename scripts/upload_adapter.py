# upload_adapter.py — run this once in Colab or locally
# Uploads the LoRA adapter to Chirag0123/shiftlog-gym-qwen-memory-policy

import os
from pathlib import Path
from huggingface_hub import HfApi, login

# Replace with your actual token or set it in environment
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN") 
MODEL_REPO = "Chirag0123/shiftlog-gym-qwen-memory-policy"
ADAPTER_DIR = Path("outputs/grpo-stagec")  # adjust if different path

if HF_TOKEN == "YOUR_HF_TOKEN":
    print("⚠️ Please set your HF_TOKEN in the script or environment.")
    exit(1)

login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)

# Create repo if it doesn't exist
print(f"Checking repo {MODEL_REPO}...")
api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)

# Check what's in the adapter dir
if not ADAPTER_DIR.exists():
    print(f"ERROR: {ADAPTER_DIR} does not exist.")
    print("Adapter was not saved. Check training pipeline.")
    # Try alternate paths:
    for alt in ["outputs/stage-c", "outputs/lora", "checkpoints/step_300",
                "checkpoints/final", "shiftlog-model-merged"]:
        if Path(alt).exists():
            print(f"Found alternate path: {alt}")
            ADAPTER_DIR = Path(alt)
            break
    else:
        print("No adapter directories found. Ensure training has run.")
        exit(1)

files = list(ADAPTER_DIR.glob("*"))
print(f"Found {len(files)} files in {ADAPTER_DIR}:")
for f in files:
    print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")

# Upload all files
for f in ADAPTER_DIR.glob("*"):
    if f.is_file() and not f.name.startswith("."):
        print(f"Uploading {f.name}...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=MODEL_REPO,
            repo_type="model",
        )
        print(f"  ✅ {f.name} uploaded")

# If adapter_config.json is missing, create a minimal one
adapter_config_path = ADAPTER_DIR / "adapter_config.json"
if not adapter_config_path.exists():
    import json
    minimal_config = {
        "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 8,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "task_type": "CAUSAL_LM"
    }
    api.upload_file(
        path_or_fileobj=json.dumps(minimal_config, indent=2).encode(),
        path_in_repo="adapter_config.json",
        repo_id=MODEL_REPO,
        repo_type="model",
    )
    print("✅ Created and uploaded minimal adapter_config.json")

print(f"\n✅ Done. Check: https://huggingface.co/{MODEL_REPO}")
