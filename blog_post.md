# ShiftLog-Gym: Teaching an LLM the Memory Policy No Lab Has Trained

*Chirag Aswal · Meta PyTorch OpenEnv Hackathon Grand Finale 2026*

It is 3 AM. My phone rings. Auth service is down. I have already handled six incidents tonight, and this one looks completely new. Except it is not. Four hours ago, I noted that the payment DB connection pool was already at 80%. The auth cascade was predictable. The problem is not that the information was unavailable. The problem is that the agent did not know to remember it.

That is the gap ShiftLog-Gym is built for.

Every major lab has shipped a memory feature: ChatGPT Memory, Claude Projects, Gemini Personal Context. But those are retrieval products attached to models that were never trained to decide what to write, when to retrieve, and what to safely forget. In real on-call work, that policy matters more than raw context length.

A RAG system over logs does not solve this. It retrieves what looks similar, not what is causally responsible. Incident #7, auth 503s, does not look textually similar to Incident #1, DB pool saturation. But Incident #1 caused Incident #7.

ShiftLog-Gym is an OpenEnv-compliant RL environment where an agent manages a simulated SRE shift across sequential incidents with hidden causal links. Some later incidents are only solvable efficiently if the model wrote the right structured shift-log entry earlier, then retrieved it before acting. The environment turns that behavior into a trainable reward signal.

I trained a `Qwen2.5-1.5B-Instruct` LoRA adapter in three stages:

- **Stage A (50 steps):** SFT warmup for clean JSON tool calls
- **Stage B (200 steps):** GRPO on short causal rollouts
- **Stage C (300 steps):** GRPO on the full shift policy

The key metric is not “did the model eventually fix it.” The key metric is whether it **read the shift log before acting** on causally linked incidents.

Here is what changed after training:

| Metric | Untrained | Trained | Improvement |
|---|---:|---:|---:|
| Causal Recall Rate | 18.4% | **91.2%** | 4.9x |
| MTTR (linked incidents) | 18.3 steps | **3.5 steps** | 5.2x |
| Hallucination Rate | 34.2% | **4.1%** | 8.3x |

The most important curve is the recall inflection around step 80 of GRPO. That is where the model starts learning the joint memory policy: write a service-specific causal note when a precursor incident happens, then retrieve it before mitigation on the downstream incident.

Prior work like Memory-R1 and AgeMem showed that RL can train memory operations on synthetic tasks. ShiftLog-Gym brings that idea into a professional domain with tool use, partial observability, causal dependencies, and verifiable outcomes. That is the contribution: not “memory is solved,” but that memory policy can be trained in a realistic operational environment.

Everything is live on Hugging Face:

- **Space:** [ShiftLog-Gym Observatory](https://huggingface.co/spaces/Chirag0123/shiftlog-gym)
- **Model:** [shiftlog-gym-qwen-memory-policy](https://huggingface.co/Chirag0123/shiftlog-gym-qwen-memory-policy)
- **Training Run:** [W&B run](https://wandb.ai/chiragaswal2/huggingface/runs/dk3g49l4)

Open the `🧪 Try the Model` tab in the Space. Type an incident. Watch the first move. If the model reads the shift log before acting, you are seeing the policy itself, not just a longer context window.

Built for the Meta PyTorch OpenEnv Hackathon Grand Finale 2026.

## Publish Checklist

1. Publish this at `https://huggingface.co/new-blog`
2. Copy the final blog URL
3. Update the main repo README with the real blog URL
4. Re-check the Space, model repo, and W&B links before submission
