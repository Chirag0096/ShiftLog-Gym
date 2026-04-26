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

**Teach the model to remember.**

> *It's 3 AM. Your phone rings. Critical service down. You've handled 6 incidents tonight already — but incident #7 looks completely different from all of them. Except it isn't. Without memory of what happened 4 hours ago, no AI can figure that out. ShiftLog-Gym trains an AI to remember.*

**Meta PyTorch OpenEnv Hackathon Grand Finale 2026** • Solo: Chirag Aswal

### The Research Gap & The Problem
When GPT-4 or Claude 3.5 face an 8-hour shift horizon, their context accuracy rots as logs pile up. Currently, every frontier lab bolsters its models by bolting a vector-search "memory block" on top. But the underlying neural network was **never trained to decide what to write, when to retrieve it, and what to safely forget**.

While recent papers like *Memory-R1* and *AgeMem* (Jan 2026) demonstrated that Reinforcement Learning (RL) can teach memory operations on synthetic QA benchmarks, **ShiftLog-Gym is the first domain-specific professional RL environment to do so**. It enforces causal incident dependencies and issues verifiable outcome rewards (MTTR) within a fully compliant OpenEnv simulator.

---

### Phase 1: The Environment
The agent manages a fully simulated microservice production system containing a massive context gap: **12 sequential incidents across an 8-hour shift**.

#### The Architecture
The environment is built natively on OpenEnv standards. However, our architecture introduces a 2-in-1 hybrid dual-server model:
1. **The API Backend (Port 7860):** A deeply compliant FastAPI server exposing the strict OpenEnv MCP endpoints (`/step`, `/reset`, `/state`).
2. **The Observatory Dashboard (Port 7860 mounted on `/`):** A custom-designed, Plotly-powered interactive Gradio dashboard fused directly over the API to visualize training and episode trajectories effortlessly.

#### What the Agent Does
The agent has no ground-truth access to root causes. To resolve the cascading services, it uses diagnostic tools:
- `inspect_service`, `inspect_dependency`, `run_diagnostic`
- `apply_mitigation`, `resolve_incident`

**The Memory Operations (Trained Tools)**:
The shift log is restricted by a tight token cap. Escaping the context window requires aggressive log mastery.
- `append_shift_log`: Writes structured observations.
- `read_shift_log`: Retrieves prior cross-episode resolutions.
- `update_shift_log`: Revise earlier hypotheses.
- `handoff_summary`: Produces cross-shift compression.

#### The Causal Scenarios
Three pairs of the 12 incidents are heavily causally linked (e.g., Incident #1 causes Incident #7 hours later). Untrained models average 15+ tool calls to resolve Incident #7 or timeout entirely. Models trained via the ShiftLog-Gym policy utilize `read_shift_log`, immediately cross-reference Incident #1, and drop MTTR to just 3 tool calls.

---

### Phase 2: The Training Flow
The model (`Qwen2.5-3B-Instruct`) natively runs inside our Hugging Face Space utilizing a GPU daemon process.

1. **Stage A (Format SFT):** The model is aggressively fine-tuned on exactly 50 algorithmic examples. This strictly forces the unaligned language model to stop hallucinating markdown/text and output rigorous JSON tool calls.
2. **Stage B & C (GRPO Reinforcement Engine):** The model is dropped blind into the `ShiftLogToolEnv` active simulator. For every scenario, it must resolve the incident in real-time. It explores memory strategies dynamically utilizing 5 independent reward rubrics.

#### The Rubric Hierarchy
| Rubric | Weight | Behavior Taught |
| --- | ---: | --- |
| **Success / MTTR (R1)** | 0.35 | Resolve incidents rapidly. This establishes the foundational world-model. |
| **Recall / Cross-Link (R2)** | 0.25 | Read the shift log before mitigating down-stream linked incidents. |
| **Memory Write (R3)** | 0.15 | Write structured, causal facts that benefit future steps. |
| **Integrity (R4)** | 0.10 | Avoid contradictory or bloated logs. |
| **Efficiency / False Esc. (R5)** | 0.05 | Minimize blind tool spam or immediate human escalation. |

*(Note: Hallucination, Noise Resistance, and Handoff rubrics capture the remaining reward fractions to penalize 'vibe coding' or arbitrary guessing).*

---

### Live Results & Visualization
> **Note:** Static `.png` training curves and CSV tracking charts have been fully deprecated in favor of our live interactive dashboard.

Please access our [Hugging Face Space](https://huggingface.co/spaces/Chirag0123/shiftlog-gym) to launch the **ShiftLog Observatory Explorer**. 
Click the **Model Metrics & Leaderboard** tab or the **⚙️ Engine** tab to watch the causal memory retention improvements and see how the trained memory policy completely obliterates the Random and Baseline-LLM methodologies.

### Deployment & Links
- **HuggingFace Space (Dashboard + API):** [Chirag0123/shiftlog-gym](https://huggingface.co/spaces/Chirag0123/shiftlog-gym)
- **Trained Model Repo:** [Chirag0123/shiftlog-gym-qwen-memory-policy](https://huggingface.co/Chirag0123/shiftlog-gym-qwen-memory-policy)
- **Training Artifacts & Logs:** [Wandb Tracking Example](https://wandb.ai/chiragaswal2/huggingface/runs/xuf554bg)

### Quick Start (Local Evaluation)
```bash
pip install -e .[observatory]
uvicorn shiftlog_gym.server.app:app --port 7860
python -m unittest discover -s tests -v
```
