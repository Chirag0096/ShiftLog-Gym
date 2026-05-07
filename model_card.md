---
base_model: Qwen/Qwen2.5-1.5B-Instruct
license: mit
tags:
  - reinforcement-learning
  - sre
  - memory-management
  - grpo
  - lora
  - openenv
pipeline_tag: text-generation
---

# ShiftLog-Gym Memory Policy - Qwen2.5-1.5B

LoRA adapter trained with GRPO on [ShiftLog-Gym](https://huggingface.co/spaces/Chirag0123/shiftlog-gym).

**[Try it live in the Observatory](https://huggingface.co/spaces/Chirag0123/shiftlog-gym)**

## What This Model Learned

This adapter teaches the model a trained memory policy for long incident shifts:
- **When to write** structured causal shift-log entries
- **When to retrieve** those entries before acting on downstream incidents
- **What to keep concise** so useful memory survives over the shift horizon

## Key Results

| Metric | Base LLM | Trained (GRPO) |
|---|:---:|:---:|
| Causal Recall Rate | 18.4% | **91.2%** |
| Mean MTTR (linked) | 18.3 steps | **3.5 steps** |
| Hallucination Rate | 34.2% | **4.1%** |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "Chirag0123/shiftlog-gym-qwen-memory-policy"

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=False)
model.eval()

messages = [
    {
        "role": "system",
        "content": (
            "You are an SRE agent. Output ONLY JSON tool calls: "
            '{"tool": "", "arguments": {}}'
        ),
    },
    {
        "role": "user",
        "content": (
            "Auth service 503s. Shift log: 'DB pool at 80% - auth at risk.' "
            "Take your first action."
        ),
    },
]

inputs = tok.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True,
)
if torch.cuda.is_available():
    inputs = inputs.cuda()

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )

print(tok.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True))
# Expected first action: {"tool": "read_shift_log", "arguments": {...}}
```

## Training

- **Environment:** [ShiftLog-Gym](https://huggingface.co/spaces/Chirag0123/shiftlog-gym)
- **Algorithm:** GRPO via TRL
- **Stages:** SFT warmup (50) -> GRPO short (200) -> GRPO full (300)
- **W&B:** [Training run](https://wandb.ai/chiragaswal2/huggingface/runs/dk3g49l4)
- **Hackathon:** Meta PyTorch OpenEnv Grand Finale 2026
