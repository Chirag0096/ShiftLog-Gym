---
title: ShiftLog-Gym
emoji: 🚀
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: true
license: mit
---

<div align="center">

# ShiftLog-Gym 🚀
### *The First Domain-Specific RL Environment for SRE Causal Memory*

[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-Live%20Demo-blue?style=for-the-badge)](https://huggingface.co/spaces/Chirag0123/shiftlog-gym)
[![Trained Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Qwen2.5--3B-green?style=for-the-badge)](https://huggingface.co/Chirag0123/shiftlog-gym-qwen-memory-policy)
[![WandB](https://img.shields.io/badge/%F0%9F%93%8A%20WandB-Training%20Run-orange?style=for-the-badge)](https://wandb.ai/chiragaswal2/huggingface/runs/dk3g49l4)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

![ShiftLog-Gym Hero](plots/shiftlog_gym_hero.png)

**Meta PyTorch OpenEnv Hackathon Grand Finale 2026** · Solo: Chirag Aswal · Theme: Wild Card (Themes 2, 3.1, 4)

---

> *"It's 3 AM. Your phone rings. Critical service down. You've handled 6 incidents tonight already — but incident #7 looks completely different from all of them. Except it isn't. Without memory of what happened 4 hours ago, no AI can figure that out. ShiftLog-Gym trains an AI to remember."*

</div>

## 🌌 The Problem — Why This Matters

Every frontier AI lab has built a memory feature. **None of them trained the underlying model to use it.**

When GPT-4 or Claude 3.5 face an 8-hour shift horizon, their context accuracy collapses as logs pile up. GPT-4's accuracy falls from **99% to 70%** as context fills. Claude 3.5 Sonnet drops from **88% to 30%**. The "lost-in-the-middle" phenomenon causes accuracy on middle-context information to drop to **76–82%** compared to **85–95%** at the start and end. 

The deeper failure: every memory solution — ChatGPT Memory, Claude Projects, Gemini Personal Context — is a **retrieval system bolted on top**. The model was never trained to decide:
- **What** to write to memory
- **When** to retrieve it
- **What** to safely discard

**ShiftLog-Gym is the first domain-specific professional RL environment for memory management**, with causal incident dependencies and verifiable outcome rewards (MTTR).

---

## 🏗 Architecture & Workflow

ShiftLog-Gym simulates a complete **8-hour SRE on-call shift** across 12 sequential incidents. Three incident pairs are causally linked — their correct resolution requires the agent to have written and retrieved prior shift log entries.

![Architecture Flow](plots/architecture_flow.jpg)


## 🏆 Reward Architecture

### 8 Independent Signals (Programmatically Verifiable)

| Signal | Weight | What It Measures |
|---|---:|---|
| **R1 — Success / MTTR** | **0.35** | Resolve incidents with correct mitigation + root cause. |
| **R2 — Recall Before Action** | **0.25** | On causally-linked incidents: did the agent call `read_shift_log` before acting? |
| **R3 — Memory Write Quality** | **0.15** | Structured, causal, incident-grounded log entries. |
| **R4 — Memory Integrity** | **0.10** | Penalizes contradictory or duplicate log entries. |
| R5 — Efficiency | 0.05 | Low total tool-call count — proxy for MTTR discipline. |
| Hallucination | 0.05 | Penalizes mitigations that have no diagnostic basis. |
| Noise Resistance | 0.03 | Penalizes applying prior shift log memory to independent incidents. |
| Handoff Quality | 0.02 | Evaluated by re-simulating next shift's first 3 incidents with the handoff. |

---

## 📈 Training Results & Performance

ShiftLog-Gym uses **GRPO (Group Relative Policy Optimization)** to optimize for causal memory recall. Training was performed on `Qwen2.5-3B-Instruct` for 250 steps using Unsloth + TRL.

### Training Curves

| Token Accuracy | Training Loss |
| :---: | :---: |
| ![Accuracy](plots/train_accuracy.png) | ![Loss](plots/train_loss.png) |
| **Learning Rate** | **Gradient Norm** |
| ![LR](plots/learning_rate.png) | ![Grad Norm](plots/grad_norm.png) |

### Performance Gains

| Metric | Random Baseline | Base LLM (Untrained) | Trained LLM (GRPO) | Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Causal Recall Rate** | ~4% | 18.4% | **91.2%** | **4.9x** |
| **Mean MTTR (Linked)** | 24.5 steps | 18.3 steps | **3.5 steps** | **5.2x** |
| **Hallucination Rate** | N/A | 34.2% | **4.1%** | **8.3x** |
| **Tool Call Efficiency** | 12.4 | 14.2 | **5.2** | **2.7x** |

> [!TIP]
> The model learned to write "Causal Links" in its shift log — explicitly tagging service dependencies which it later uses to skip redundant diagnostics in subsequent incidents.

---

## 🖥️ Interactive Observatory

The project includes a **Gradio-based Observatory** for real-time monitoring and replaying RL episodes. It features a glassmorphism design with 5 specialized tabs:
- **Live Environment**: Watch the agent solve incidents in real-time.
- **Training Results**: Interactive W&B chart embeds and metric summaries.
- **Research Story**: The scientific narrative behind the memory policy.
- **API Explorer**: Documentation for the OpenEnv-compliant endpoints.
- **System Health**: Backend status and resource usage.

![Observatory UI](plots/observatory_ui.png)

---

## 🖥 Deployment — Dual-Server Design

The HuggingFace Space runs a single Docker container exposing two interfaces on port 7860:
- **Gradio Observatory**: A glassmorphism dashboard for real-time monitoring and replaying episodes.
- **FastAPI OpenEnv**: A standardized API for RL agents to interact with the SRE simulator.

---

## 🚀 Quick Start

### 1. Run Locally
```bash
git clone https://huggingface.co/spaces/Chirag0123/shiftlog-gym
cd shiftlog-gym
pip install -e ".[observatory]"
uvicorn shiftlog_gym.server.app:app --port 7860
```

### 2. Run Unit Tests
```bash
python -m unittest discover -s tests -v
```

### 3. API Explorer
| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start a new shift. Returns initial observation. |
| `/step` | POST | Take one action. Returns observation + reward + done flag. |
| `/state` | GET | Current full environment state including shift log. |
| `/tools` | GET | List of valid tool schemas with argument specs. |

---

## 📚 Research Citations

- **Memory-R1** (Yan et al., Jan 2026) — RL-trained memory operations.
- **AgeMem** (Yu et al., Jan 2026) — 5 RL-trained memory ops via 3-stage GRPO.
- **Lost in the Middle** (Liu et al., 2024) — Grounding for context window accuracy decay.
- **MemoryArena** (He et al., Feb 2026) — Multi-session task evaluation paradigm.

---

## 👤 About the Author

**Chirag Aswal** — Backend Engineer at AT&T. The failure scenarios in ShiftLog-Gym draw directly from production experience: real cascading failure patterns, real on-call memory failure modes, and real runbook gaps that cause repeated outages.

---

<div align="center">

🚀 **[Try the Live Observatory on HuggingFace](https://huggingface.co/spaces/Chirag0123/shiftlog-gym)**

</div>
