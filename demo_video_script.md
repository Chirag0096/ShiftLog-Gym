# ShiftLog-Gym Demo Video Script

Title: `ShiftLog-Gym: RL-Trained SRE Memory Policy | OpenEnv Hackathon 2026`

Description starter:
`ShiftLog-Gym trains an LLM to write and retrieve causal shift-log memory before acting on downstream incidents. Live Space: https://huggingface.co/spaces/Chirag0123/shiftlog-gym`

## Timeline

### 00:00-00:15 — Hook
Say:

> It is 3 AM. Your phone rings. Auth service is down.  
> You have already handled six incidents tonight, but incident seven looks completely different.  
> Except it is not.

Show:
- Space landing in the `🔴 Live Environment` tab

### 00:15-00:40 — The problem
Say:

> Every frontier AI lab has built a memory feature.  
> None of them trained the model to use it.  
> The model was never taught what to write, when to retrieve, or what to safely forget.

Show:
- README state-of-the-art comparison
- the causal-memory framing section

### 00:40-01:10 — The environment and live inference
Say:

> Here is the trained model on incident seven, the auth cascade.  
> Watch step one. It reads the shift log first.  
> It finds the precursor from incident one and uses that memory before mitigation.

Show:
- click `🧪 Try the Model`
- select `Incident #7 - Auth Cascade (causal)`
- click `Run Trained Model`
- let the streamed steps render on screen

### 01:10-01:35 — The result
Say:

> After GRPO training, causal recall reaches 91.2%, versus 18.4% for the untrained model.  
> Mean time to resolution on linked incidents drops to 3.5 steps from 18.3.  
> This inflection around step 80 is where the memory policy emerges.

Show:
- `📈 Live Training Metrics`
- recall curve / reward curve

### 01:35-01:55 — The claim
Say:

> Memory-R1 and AgeMem showed that reinforcement learning can train memory operations on synthetic tasks.  
> ShiftLog-Gym brings that into a professional domain with causal incident dependencies and verifiable rewards.  
> The environment is live on Hugging Face, and anyone can train on it.

Show:
- Space URL
- model repo URL

### 01:55-02:00 — Close
Say:

> ShiftLog-Gym. Teach the model to remember.

Show:
- return to observatory overview

## Recording Checklist

1. Record at 1080p with cursor visible
2. Keep the final video under 2 minutes
3. Upload to YouTube as Unlisted or Public
4. Copy the final YouTube URL into the repo README after upload
