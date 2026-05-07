# ShiftLog-Gym: Training AI Agents to Remember Causally in SRE Operations

## The Problem: Why Current AI Memory Falls Short

Every major AI lab has deployed memory features—ChatGPT Memory, Claude Projects, Gemini Personal Context—but there's a critical flaw: **these are retrieval systems bolted onto models that were never trained to use memory effectively**.

When faced with an 8-hour SRE shift, frontier models like GPT-4 and Claude 3.5 suffer from catastrophic context decay:
- GPT-4 accuracy drops from 99% to 70% as context fills
- Claude 3.5 Sonnet falls from 88% to 30%
- Middle-context information accuracy plunges to 76-82% (vs 85-95% at boundaries)

The deeper failure? These models were never trained to decide **what** to write, **when** to retrieve, or **what** to safely forget. They excel at retrieving existing information but fail at the active memory management that defines expert human SREs.

## Introducing ShiftLog-Gym

ShiftLog-Gym is the first domain-specific reinforcement learning environment designed to train language models in **causal memory management** for SRE operations. It simulates a complete 8-hour on-call shift across 12 sequential incidents, where three incident pairs are causally linked—correct resolution requires the agent to have written and retrieved prior shift log entries.

### Core Innovation: Beyond RAG

While every observability platform (Datadog, Splunk, Grafana) offers vector search over logs, ShiftLog-Gym trains the AI that reads those logs to actually **understand** them—not just search them. Here's why RAG fails for on-call SRE memory:

1. **RAG retrieves but doesn't decide what's worth retrieving** - Returns 500 recent log lines instead of the single causal line from 4 hours ago
2. **RAG doesn't write** - Can't determine what future engineers need to find in the noise
3. **RAG has no causal model** - Uses similarity search, not causal reasoning
4. **RAG adds latency and infra cost** - 50-200ms per search vs zero-latency internal policy

## The Training Pipeline: 3-Stage GRPO Curriculum

ShiftLog-Gym employs Group Relative Policy Optimization (GRPO) across three stages:

### Stage A: SFT Warmup (50 steps)
Aligns the model to JSON schema and tool-calling syntax using 50 carefully curated examples. Teaches the basic format: return exactly one JSON tool call per turn.

### Stage B: GRPO Short Rollout (200 steps)
Optimizes for "Recall-before-action" (R2) in 4-incident horizons. The model learns that reading shift logs before taking action on causally-linked incidents yields higher rewards.

### Stage C: GRPO Full Rollout (300 steps)
Scales to full 8-hour shifts (12 incidents) across all six failure families with causal rewards. This is where the model internalizes the complete causal memory policy.

## The 8-Dimensional Reward System

ShiftLog-Gym's Process Reward Model (PRM) decomposes performance into eight independently verifiable signals:

| Signal | Weight | What It Measures |
|--------|--------|------------------|
| R1 Success | 35% | Correct root cause + mitigation |
| R2 Recall | 25% | Read shift log before linked-incident action |
| R3 Memory Write | 15% | Quality of shift log entries |
| R4 Memory Integrity | 10% | Penalizes contradictions & duplicates |
| R5 Efficiency | 5% | Fewer tool calls = higher score |
| Hallucination | 5% | Penalizes invalid mitigations |
| Noise Resistance | 3% | Resists misleadingly similar noise incidents |
| Handoff Quality | 2% | Structured shift handoff summary |

The magic happens in R2: the model earns reward only when it calls `read_shift_log` before `apply_mitigation` or `resolve_incident` on causally-linked incidents.

## Results: Transforming Vibe Agents into Causal Reasoners

After training, ShiftLog-Gym produces dramatic improvements over untrained baselines:

| Metric | Random Baseline | Base LLM (Untrained) | Trained LLM (GRPO) | Improvement |
|--------|----------------|----------------------|-------------------|-------------|
| **Causal Recall Rate** | ~4% | 18.4% | **91.2%** | **4.9x** |
| **Mean MTTR (Linked)** | 24.5 steps | 18.3 steps | **3.5 steps** | **5.2x** |
| **Hallucination Rate** | N/A | 34.2% | **4.1%** | **8.3x** |
| **Tool Call Efficiency** | 12.4 | 14.2 | **5.2** | **2.7x** |

Most notably, the **Vibe Ratio**—the proportion of tool actions on linked incidents taken without prior log search—drops from >0.7 (pure guessing) to <0.1 (causal reasoning).

## Why This Matters for Production SRE

ShiftLog-Gym isn't just an academic exercise—it addresses real production pain points:

- **Reduced MTTR**: Faster incident resolution means less downtime and happier customers
- **Fewer escalations**: Agents that remember precursors stop repeatedly fixing symptoms without addressing root causes
- **Better handoffs**: Structured memory persistence reduces knowledge loss between shifts
- **Lower cognitive load**: Humans spend less time reconstructing context from fragmented logs

## The Future of Autonomous SRE

ShiftLog-Gym demonstrates that memory isn't just about storing information—it's about training models to develop **metacognitive awareness** of their own memory states. The same principles apply beyond SRE to any domain requiring long-term causal reasoning:

- Medical diagnosis across patient visits
- Financial fraud detection across transaction sequences
- Legal case analysis across precedent chains
- Scientific discovery across experimental iterations

By moving memory from external retrieval systems into the model's weights, we create AI agents that don't just access memory—they **live** in it.

## Try It Yourself

1. **Experience the environment**: [Hugging Face Space Demo](https://huggingface.co/spaces/Chirag0123/shiftlog-gym)
2. **Explore the trained model**: [Qwen2.5-1.5B Memory Policy](https://huggingface.co/Chirag0123/shiftlog-gym-qwen-memory-policy)
3. **Review the training runs**: [Weights & Biases Report](https://wandb.ai/chiragaswal2/huggingface/runs/dk3g49l4)
4. **Dive into the code**: [GitHub Repository](https://github.com/Chirag0096/ShiftLog-Gym)

ShiftLog-Gym represents a step toward AI agents that don't just process information—the **remember**, **reason**, and **act** with the causal awareness that defines true expertise.
