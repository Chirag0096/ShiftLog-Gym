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

**ShiftLog-Gym is the first domain-specific professional RL environment for memory management**, with causal incident dependencies and verifiable outcome rewards (MTTR).

---

## 🏗 Architecture & Workflow

ShiftLog-Gym simulates a complete **8-hour SRE on-call shift** across 12 sequential incidents. Three incident pairs are causally linked — their correct resolution requires the agent to have written and retrieved prior shift log entries.

![Architecture Flow](plots/architecture_flow.jpg)

---

## 🧪 Deep Research & Mathematical Framework

### 1. The Optimization Objective: GRPO
ShiftLog-Gym utilizes **Group Relative Policy Optimization (GRPO)**, a reinforcement learning algorithm that eliminates the need for a separate critic model by computing advantages relative to a group of sampled trajectories.

For a group of $G$ outputs $\{o_1, o_2, \dots, o_G\}$, the advantage $\hat{A}_i$ for output $o_i$ is calculated as:
$$\hat{A}_i = \frac{R_i - \text{mean}(R_1, \dots, R_G)}{\text{std}(R_1, \dots, R_G)}$$

The GRPO loss function minimized during training is:
$$L_{GRPO}(\theta) = -\frac{1}{G} \sum_{i=1}^{G} \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} \hat{A}_i, \text{clip} \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon \right) \hat{A}_i \right) + \beta D_{KL}(\pi_\theta || \pi_{ref})$$

This approach is specifically effective for **memory operations** because it allows the model to compare different memory-writing strategies (e.g., "concise" vs. "causal") within the same incident context and amplify those that lead to lower MTTR in future steps.

### 2. Multi-Stage Training Pipeline
We implement a 3-stage curriculum to transition the model from base instruction-following to specialized causal memory management:

| Stage | Name | Objective | Duration |
| :--- | :--- | :--- | :--- |
| **Stage A** | **SFT Warmup** | Align model to the `ShiftLogEntry` JSON schema and tool-calling syntax. | 50 Steps |
| **Stage B** | **GRPO Short** | Optimize for "Recall-before-action" (R2) in 4-incident horizons. | 200 Steps |
| **Stage C** | **GRPO Full** | Full 8-hour shift optimization (12 incidents) with causal dependency rewards. | 300 Steps |

### 3. Causal Reward Rubric (PRM)
The **Process Reward Model (PRM)** decomposes the 8-hour shift into 8 independent, programmatically verifiable signals. The total reward $R_{total}$ for an episode is:
$$R_{total} = \sum_{j=1}^{8} w_j \cdot R_j$$

Where the primary causal signal **R2 (Recall Before Action)** is defined as:
$$R_2 = \frac{1}{|I_{linked}|} \sum_{i \in I_{linked}} \mathbb{1}(\text{tool\_called}(\text{read\_shift\_log}, i) \prec \text{tool\_called}(\text{mitigate}, i))$$
This formula ensures the model is *only* rewarded for recall if it happens *before* a mitigation attempt, effectively penalizing "trial-and-error" behavior.

---

## 🏆 Reward Architecture Details

| Signal | Weight | Logic / Formula |
|---|---:|---|
| **R1 — Success** | **0.35** | $CorrectResolutions / TotalRealIncidents$ |
| **R2 — Recall** | **0.25** | Causal check: `read\_log` happened before `mitigate` on linked incidents. |
| **R3 — Quality** | **0.15** | Keyword overlap between `fact` and scenario `ground\_truth`. |
| **R4 — Integrity** | **0.10** | Penalty for duplicate or contradictory log entries. |
| **R5 — Efficiency** | **0.05** | $1.0 - (TotalToolCalls / MaxAllowedCalls)$ |
| **R6 — Hallucination**| **0.05** | Penalty for mitigations not supported by diagnostic evidence. |
| **R7 — Noise Res.** | **0.03** | Penalty for retrieving memory on independent (noise) incidents. |
| **R8 — Handoff** | **0.02** | Qualitative check on final `handoff\_summary` accuracy. |

---

## 📈 Training Results & Evaluation

### Training Curves (Qwen2.5-3B)

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

---

## 🖥️ Interactive Observatory

The project includes a **Gradio-based Observatory** for real-time monitoring and replaying RL episodes.

![Observatory UI](plots/observatory_ui.png)

---

## 🖥 Deployment — Dual-Server Design

The HuggingFace Space runs a single Docker container exposing two interfaces on port 7860:
- **Gradio Observatory**: A glassmorphism dashboard for real-time monitoring.
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
