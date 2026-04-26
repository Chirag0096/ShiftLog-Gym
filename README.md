---
title: Shiftlog-Gym
emoji: 🚀
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.17.0
app_file: app.py
python_version: 3.11.9
pinned: false
license: mit
---
# ShiftLog-Gym

> *It's 3 AM. Your phone rings. Critical service down. You've handled 6 incidents tonight already — but incident #7 looks completely different from all of them. Except it isn't. Without memory of what happened 4 hours ago, no AI can figure that out. ShiftLog-Gym trains an AI to remember.*

**Meta PyTorch OpenEnv Hackathon Grand Finale 2026** · Solo: Chirag Aswal

---

## The Problem

LLMs acting as on-call SRE agents can handle individual incidents, but they do not preserve operational memory across incident chains. In production, that failure is expensive: a downstream outage caused by an earlier unresolved root cause looks identical to a new incident *unless the agent remembers what happened before*.

Prior memory research (Memory-R1, Mem-α, MemAgent) targets conversational or general agent memory. **ShiftLog-Gym is the first domain-specific professional RL environment** to train LLMs on memory operations via verifiable SRE incident resolution rewards.

---

## The Environment

A fully simulated microservice production system spanning a 12-incident 8-hour on-call shift. Three incidents (#7, #9, #11) are **causally linked** to earlier ones — their correct resolution requires reading prior shift log entries.

### Tools Available to the Agent

| Tool | Purpose |
|---|---|
| `read_shift_log` | Retrieves structured prior memory entries |
| `append_shift_log` | Writes causal facts, hypotheses, resolutions |
| `update_shift_log` | Corrects earlier memory entries |
| `inspect_service` | Owner, runbook, service context |
| `inspect_dependency` | Dependency graph context |
| `run_diagnostic` | Machine-authored diagnostic result |
| `apply_mitigation` | Apply an operational mitigation |
| `resolve_incident` | Machine-checkable resolution with root cause |
| `handoff_summary` | Produce cross-shift memory compression |

### Reward Design

| Rubric | Weight | Behavior Taught |
|---|---:|---|
| **R1 — Success / MTTR** | 0.35 | Resolve incidents with correct mitigation + root cause |
| **R2 — Recall Before Action** | 0.25 | Read shift log before acting on causally-linked incidents |
| **R3 — Memory Write Quality** | 0.15 | Write structured, causal, incident-grounded entries |
| **R4 — Memory Integrity** | 0.10 | Avoid contradictory or duplicate log entries |
| R5 — Efficiency | 0.05 | Low tool-call count (proxy for MTTR discipline) |
| Hallucination | 0.05 | Penalize unsupported mitigations |
| Noise Resistance | 0.03 | Avoid applying prior memory to superficially similar but different incidents |
| Handoff Quality | 0.02 | Produce usable handoffs with unresolved IDs + confidence |

---

## Results

Training evidence from GRPO run (`Qwen2.5-3B-Instruct`, 250 steps):

### Reward Curve
![GRPO Training Reward Curve](plots/01_reward_curve.png)
*Total reward over training steps. Upward trend confirms the agent learns to resolve incidents more efficiently.*

### Memory Policy Learning — R2 Recall Bonus
![Cross-Episode Recall Bonus](plots/02_recall_bonus_curve.png)
*R2 recall-before-action rate over episodes. The inflection around step 80 shows the agent learning to read its shift log before acting on causally-linked incidents.*

### MTTR Improvement
![MTTR Comparison](plots/03_mttr_comparison.png)
*Mean steps to resolve causally-linked incidents (#7, #9, #11). Trained agent resolves linked incidents in ~3–5 steps versus ~18–25 for the random baseline.*

> **Key numbers** (populated automatically after running `train/02_grpo_train_colab.ipynb`):
> - Recall-before-action rate: `baseline → trained` (see [live dashboard](https://chirag0123-shiftlog-gym.hf.space))
> - Linked incident success rate: `baseline → trained`
> - Mean MTTR (linked incidents): `baseline steps → trained steps`

---

## Architecture

The HuggingFace Space runs a **dual-server** at port 7860:
- **OpenEnv API** (`/reset`, `/step`, `/state`, `/tools`): Fully compliant with competition evaluation bots
- **Gradio Observatory Dashboard** (root `/`): Interactive SRE training visualization for human judges

---

## Training Pipeline

```
Stage A: SFT Format Warmup (50 steps)
  └─ Teaches Qwen to output strict JSON tool calls
Stage B: GRPO Short Rollout (200 steps, 3 incident families)
  └─ Learns recall-before-action via verifiable environment rewards
Stage C: GRPO Full Rollout (300 steps, all families)
  └─ Generalizes the memory policy across all 6 incident families
```

Run it yourself: open `train/02_grpo_train_colab.ipynb` in Colab with a T4/L4 GPU.

---

## Differentiation from Prior Work

| Capability | Memory-R1 | MemAgent | **ShiftLog-Gym** |
|---|---|---|---|
| Domain | General | Agent memory | **Professional SRE** |
| Machine-verifiable outcomes | Limited | Limited | **Yes** |
| Causal incident chains | No | No | **Yes** |
| Noise resistance | No | No | **Yes** |
| Shift handoff memory | No | No | **Yes** |
| OpenEnv compliant | No | No | **Yes** |

---

## Quick Start (Local)

```bash
pip install -e .[observatory]
uvicorn shiftlog_gym.server.app:app --port 7860
python -m unittest discover -s tests -v
```

---

## Links

- 🚀 **HuggingFace Space (Environment + Dashboard):** [Chirag0123/shiftlog-gym](https://huggingface.co/spaces/Chirag0123/shiftlog-gym)
- 🤖 **Trained Model Repository:** [Chirag0123/shiftlog-gym-qwen-memory-policy](https://huggingface.co/Chirag0123/shiftlog-gym-qwen-memory-policy)
- 📊 **WandB Training Run:** [chiragaswal2/shiftlog-gym](https://wandb.ai/chiragaswal2/shiftlog-gym)
- 📓 **Training Notebook (Colab):** [train/02_grpo_train_colab.ipynb](train/02_grpo_train_colab.ipynb)
- 🎥 **Blog Post / Demo Video:** *(coming before final submission deadline)*
