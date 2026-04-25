# ShiftLog-Gym

ShiftLog-Gym is **a domain-specific professional RL environment for memory management with causal incident dependencies and verifiable outcome rewards**.

It trains and evaluates whether an LLM learns a useful **memory policy** during simulated SRE/on-call incident response:

- write high-signal structured shift-log facts
- retrieve the right memory before acting on linked incidents
- avoid contradictory or duplicate memory writes
- resolve incidents with machine-checkable root causes and mitigations

This repo is optimized for the OpenEnv Hackathon workflow:

- OpenEnv-compliant environment scaffold
- separate Google Colab training notebooks using TRL/OpenEnv
- observability artifacts for failure analysis and storytelling
- README language aligned to the judging criteria

## Positioning

Use this exact claim:

> **A domain-specific professional RL environment for memory management with causal incident dependencies and verifiable outcome rewards.**

Do **not** claim:

- first RL memory system
- first memory benchmark
- solved persistent memory

Relevant prior work to acknowledge:

- Memory-R1: [huggingface.co/papers/2508.19828](https://huggingface.co/papers/2508.19828)
- AgeMem / Agentic Memory: [huggingface.co/papers/2601.01885](https://huggingface.co/papers/2601.01885)
- MemoryArena: [memoryarena.github.io](https://memoryarena.github.io/)
- OpenEnv docs: [meta-pytorch.org/OpenEnv](https://meta-pytorch.org/OpenEnv/)
- TRL OpenEnv integration: [huggingface.co/docs/trl/openenv](https://huggingface.co/docs/trl/openenv)

## What The Agent Does

Each episode is a simulated shift with multiple incidents across shared services. The agent can:

- `read_shift_log(query, limit)`
- `append_shift_log(entry_type, incident_id, service, fact, confidence)`
- `update_shift_log(memory_id, patch, reason)`
- `inspect_service(service)`
- `inspect_dependency(service)`
- `run_diagnostic(service, diagnostic)`
- `apply_mitigation(service, mitigation)`
- `resolve_incident(incident_id, resolution, root_cause)`
- `handoff_summary()`

The world is partially observable. Current symptoms are visible, but earlier root-cause facts are only recoverable through the structured shift log.

## Incident Families

V1 includes 4 scenario families with seeded variants:

1. DB connection pool exhaustion recurring after rollback
2. Auth timeout cascade caused by stale routing or dependency saturation
3. Memory / OOM regressions with recurring workload signatures
4. Feature-flag or deprecated-bundle regressions with delayed downstream symptoms

Each family includes:

- precursor incidents that create high-value memory
- later linked incidents that reward retrieval before action
- machine-checkable correct root cause and mitigation

## Reward Design

ShiftLog-Gym uses a sparse-first reward design with a few shaped components:

- `R_success`: terminal success on correct resolution
- `R_recall`: reward for retrieving causally relevant memory before acting
- `R_memory_write`: reward for storing high-value reusable memory
- `R_memory_integrity`: penalties for contradictions, bad schema writes, duplicates
- `R_efficiency`: small per-tool cost to proxy MTTR
- `R_hallucination`: penalties for fabricated mitigations or unsupported fixes

The main scientific result to show is:

> the trained model learns to consult the shift log before acting on causally linked incidents

## Local Development

### Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[train]
pip install -e .[observatory]
pip install -e .[openenv]
```

### Run the server

```bash
uvicorn shiftlog_gym.server.app:app --reload --port 7860
```

### Run tests

```bash
python3 -m unittest discover -s tests -v
```

### Generate observatory artifacts

```bash
python3 scripts/export_observatory_artifacts.py
```

### Run observatory

```bash
streamlit run observatory/app.py
```

## Repo Layout

```text
shiftlog_gym/
  client.py
  simulator.py
  scenarios.py
  trl_env.py
  server/
train/
  01_env_smoke_test.ipynb
  02_grpo_train_colab.ipynb
observatory/
  app.py
tests/
scripts/
```

## Training Workflow

Use the notebooks in `train/`:

- `01_env_smoke_test.ipynb`
  - random baseline
  - scripted baseline
  - artifact export and sanity plots
- `02_grpo_train_colab.ipynb`
  - `Qwen2.5-3B-Instruct` default
  - `Qwen2.5-1.5B-Instruct` fallback
  - LoRA / QLoRA friendly setup
  - TRL `environment_factory` via `shiftlog_gym.trl_env.ShiftLogToolEnv`

## Submission Checklist

- host environment on Hugging Face Spaces
- link the Space from this README
- include plots for:
  - total reward
  - `R_recall`
  - recall-before-action rate
  - linked-incident success rate
  - contradiction / bad-write rate
- include short video, blog post, or slides
- keep the demo focused on one clear memory-policy win

## Current Status

This repo includes:

- simulator core
- OpenEnv-style server scaffold
- TRL environment wrapper
- observatory artifact pipeline
- unit tests for schema integrity, contradictions, causal reward attribution, and memory dependence

It is intentionally single-agent and optimized for a 48-hour hackathon submission.

