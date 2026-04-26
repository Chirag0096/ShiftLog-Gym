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
## ShiftLog-Gym

**The first RL training environment where causal incident dependency chains require cross-episode memory retrieval for correct resolution — operationalizing institutional SRE knowledge as a learnable memory policy.**

### Why This Exists
LLMs acting as on-call SRE agents can inspect the current incident, but they do not preserve operational memory across incident chains unless they explicitly retrieve and use prior state. In incident response, that failure mode is expensive: a later outage can look identical to an earlier one while requiring a different action unless the agent remembers the earlier root cause, mitigation outcome, and residual risk.

Prior memory work such as Memory-R1, Mem-α, and MemAgent targets conversational memory or general agent memory. ShiftLog-Gym targets a different regime: tool-calling, partially observable, multi-incident SRE workflows with machine-verifiable mitigations, explicit noise incidents, and shift handoff state that can be scored with rubrics instead of judge-only impressions.

### What The Agent Sees and Does
- Observation space: The agent sees one active incident at a time, current symptoms, service/dependency context, recent shift-log snippets, and in multi-shift mode the prior handoff summary.
- `read_shift_log`: Retrieves prior structured memory entries before the agent commits to an action.
- `append_shift_log`: Writes a structured memory fact, hypothesis, resolution, or handoff note into persistent shift memory.
- `update_shift_log`: Edits an existing memory entry when the agent revises earlier beliefs.
- `inspect_service`: Returns owner, runbook, and service-local context for the active service.
- `inspect_dependency`: Returns dependency graph context for the active service.
- `run_diagnostic`: Returns a machine-authored diagnostic result for the active incident.
- `apply_mitigation`: Applies an operational mitigation and returns whether symptoms improved.
- `resolve_incident`: Attempts a machine-checkable resolution with explicit mitigation and root cause.
- `handoff_summary`: Produces an end-of-shift memory artifact for the next shift.

### What Makes This Hard
- Causal chains: Symptoms of Incident B are not enough on their own because the correct fix depends on memory written during Incident A.
- Noise incidents: The environment injects symptom-similar incidents with a different root cause, so blind pattern-matching to prior memory is punished.
- Shift handoff: In multi-shift mode, Shift 2 starts with Shift 1’s handoff and must use that memory to resolve a downstream consequence of an unresolved incident.

### Reward Design
| Rubric | Weight | Behavior taught |
| --- | ---: | --- |
| Success | 0.35 | Resolve real incidents with the correct mitigation and root cause. |
| RecallBeforeAction | 0.25 | Read the shift log before mitigating or resolving linked incidents. |
| MemoryWriteQuality | 0.15 | Write structured, incident-grounded memory with useful causal content. |
| MemoryIntegrity | 0.10 | Avoid contradictory, duplicate, or schema-invalid memory. |
| Efficiency | 0.05 | Keep tool-call count low as a proxy for MTTR discipline. |
| Hallucination | 0.05 | Avoid unsupported mitigations and fabricated root-cause claims. |
| NoiseResistance | 0.03 | Avoid applying memory-derived fixes to noise incidents that only look familiar. |
| HandoffQuality | 0.02 | Produce a usable handoff with unresolved incident IDs, causes, and confidence. |

### Results
![Baseline Comparison](observatory/baseline_comparison.png)

Caption: `observatory/baseline_comparison.png` compares the random agent, scripted agent, untrained LLM baseline, and trained model on recall-before-action, linked-incident success, noise resistance, and weighted reward.

![Stage B Curves](observatory/training_runs/training_curves_stageB.png)

Caption: Stage B focuses on the short-rollout memory-policy habit: read before acting on linked incidents.

![Stage C Curves](observatory/training_runs/training_curves_stageC.png)

Caption: Stage C scales the same policy to all six families with noise incidents and full-rollout episodes.

Before/after recall_before_action_rate is reported from the baseline notebook and the Stage B/Stage C training curves in `observatory/training_runs/training_curves_stageB.csv` and `observatory/training_runs/training_curves_stageC.csv`.

### Differentiation from Prior Work
| Capability | Memory-R1 | Mem-α | MemAgent | ShiftLog-Gym |
| --- | --- | --- | --- | --- |
| Domain | General memory ops | General memory learning | Agent memory systems | Professional SRE incident response |
| Multi-turn tool-calling | Partial | Partial | Partial | Yes |
| Machine-verifiable outcomes | Limited | Limited | Limited | Yes |
| Causal incident chains | No | No | No | Yes |
| Noise resistance | No | No | No | Yes |
| Shift handoff | No | No | No | Yes |
| Trained model available | External | External | External | Repo training notebooks + adapter workflow |

### Running Locally
```bash
pip install -e .[observatory]
uvicorn shiftlog_gym.server.app:app --port 7860
python -m unittest discover -s tests -v
```

### Training
Training notebooks live in [train/01_env_smoke_test.ipynb](/Users/chiragaswal/Developer/Personal%20Doc/Vibe%20Projects/MetaHack/train/01_env_smoke_test.ipynb:1), [train/02_grpo_train_colab.ipynb](/Users/chiragaswal/Developer/Personal%20Doc/Vibe%20Projects/MetaHack/train/02_grpo_train_colab.ipynb:1), and [train/03_eval_publish_colab.ipynb](/Users/chiragaswal/Developer/Personal%20Doc/Vibe%20Projects/MetaHack/train/03_eval_publish_colab.ipynb:1). Notebook 2 now runs through `train/colab_training_pipeline.py`, prompts for both `HF_TOKEN` and `WANDB_API_KEY`, attempts GRPO first, and writes stage artifacts even if the runtime requires fallback. Notebook 3 runs through `train/colab_eval_publish.py` for plot/table/export generation.

For Hugging Face hardware and budgeted execution flow, follow [docs/HF_SETUP_AND_GPU_PLAN.md](/Users/chiragaswal/Developer/Personal%20Doc/Vibe%20Projects/MetaHack/docs/HF_SETUP_AND_GPU_PLAN.md:1).

### Links
- HuggingFace Space: [ShiftLog-Gym Dashboard](https://huggingface.co/spaces/Chirag0123/shiftlog-gym)
