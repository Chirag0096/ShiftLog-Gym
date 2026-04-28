"""
comparison_tab.py

Renders the "⚡ Base vs Trained" tab in the Gradio dashboard.
Runs base Qwen2.5-1.5B and the LoRA-adapted version on the SAME
incident simultaneously and shows their trajectories side by side.

This is the single most powerful demo artifact — shows judges the
improvement in real time with identical inputs.
"""
import threading
import gradio as gr
import torch
import json
from typing import Tuple, List

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_REPO = "Chirag0123/shiftlog-gym-qwen-memory-policy"

# Shared scenario — always use Incident #7 (the auth cascade, caused by #1)
DEMO_INCIDENT_7 = {
    "incident_id": 7,
    "service": "auth-service",
    "symptoms": "503s on auth-service. Error rate 67%. Latency p99 8200ms.",
    "shift_log": [
        {
            "id": "sl_001",
            "incident": 1,
            "service": "payment-db",
            "content": "Payment DB connection pool at 80% after scaling. Auth service at risk under sustained load.",
            "type": "causal_critical",
        }
    ],
    "available_tools": [
        "read_shift_log",
        "inspect_service",
        "inspect_dependency",
        "run_diagnostic",
        "apply_mitigation",
        "resolve_incident",
    ],
    "ground_truth_root_cause": "DB connection pool exhaustion cascaded to auth service",
    "optimal_steps": 4,
}


def _check_vram_for_comparison() -> tuple[bool, str]:
    """Returns (ok, message). Warns if VRAM < 20GB (need ~8GB per model)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, ("No GPU detected. The comparison requires a GPU Space. "
                           "Upgrade to L4 x1 in Space Settings.")
        free_gb = (torch.cuda.get_device_properties(0).total_memory -
                   torch.cuda.memory_allocated(0)) / 1e9
        if free_gb < 10:
            return False, (f"Only {free_gb:.1f}GB VRAM free. Need ~10GB. "
                           "Restart the Space to clear GPU memory, then retry.")
        return True, f"GPU OK — {free_gb:.1f}GB free"
    except Exception as e:
        return False, f"VRAM check failed: {e}"


def _simulate_env_step(tool: str, args: dict, scenario: dict) -> str:
    """
    Lightweight local environment simulator for the comparison demo.
    Does NOT call the Space API — runs synchronously during inference.
    """
    responses = {
        "read_shift_log": (
            f"Retrieved {len(scenario['shift_log'])} entries. "
            f"Entry sl_001: '{scenario['shift_log'][0]['content']}'"
        ),
        "inspect_service": (
            "auth-service: owner=platform-team, "
            "deps=[payment-db, cache-svc, session-store]"
        ),
        "inspect_dependency": (
            "payment-db → auth-service "
            "(connection pool shared, max=100)"
        ),
        "run_diagnostic": (
            "payment-db connection pool: 98% utilized "
            "(97/100 connections active). "
            "Auth service queuing on DB connections."
        ),
        "apply_mitigation": (
            f"Applied {args.get('action', 'unknown')} "
            f"to {args.get('service', 'unknown')}. "
            f"Partial recovery observed."
        ),
        "resolve_incident": (
            f"Incident closed. "
            f"Root cause logged: {args.get('root_cause', 'unspecified')}"
        ),
    }
    return responses.get(tool, f"Tool '{tool}' executed. No response.")


def run_model(model_label: str, use_adapter: bool):
    """
    Loads and runs one model variant on the demo incident.
    Returns (trajectory_messages, steps, used_recall, correct_rc).
    """
    import os, json, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    trajectory = []
    steps = 0
    used_recall = False
    correct_rc = False

    # ── Authenticate with HF Hub (uses Space secret) ──────────────────────
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)

    try:
        # ── Load tokenizer ──────────────────────────────────────────────────
        tok = AutoTokenizer.from_pretrained(
            BASE_MODEL_ID,
            trust_remote_code=True,
            token=hf_token or None,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # ── Load model — use bfloat16, NO 4-bit quant (avoids lm_head bug) ─
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            token=hf_token or None,
        )

        # ── Load LoRA adapter if requested ────────────────────────────────
        if use_adapter:
            from huggingface_hub import list_repo_files
            try:
                # Check if adapter files exist before trying to load
                repo_files = list(list_repo_files(
                    ADAPTER_REPO, repo_type="model", token=hf_token or None
                ))
                has_adapter = any(
                    f in repo_files
                    for f in ["adapter_config.json", "adapter_model.safetensors",
                               "adapter_model.bin"]
                )
                if not has_adapter:
                    return (
                        [{"role": "assistant",
                          "content": (
                              "⚠️ **Adapter not uploaded yet.**\n\n"
                              f"The model repo `{ADAPTER_REPO}` exists but has no "
                              "adapter files.\n\n"
                              "**To fix:** Run `upload_adapter.py` in Colab after "
                              "training completes, or run the full training pipeline."
                          )}],
                        0, False, False
                    )
                from peft import PeftModel
                model = PeftModel.from_pretrained(
                    model, ADAPTER_REPO,
                    token=hf_token or None,
                    is_trainable=False,
                )
            except Exception as e:
                return (
                    [{"role": "assistant",
                      "content": (
                          f"⚠️ **Could not load LoRA adapter.**\n\n"
                          f"Error: `{str(e)[:300]}`\n\n"
                          f"Repo: `{ADAPTER_REPO}`\n\n"
                          "Check that training completed and adapter was uploaded."
                      )}],
                    0, False, False
                )

        model.eval()

        # ── Build prompt ───────────────────────────────────────────────────
        system_prompt = (
            "You are an SRE on-call agent. "
            "For every action output ONLY a single JSON object on one line:\n"
            '{"tool": "", "args": {: }}\n'
            "No other text before or after the JSON. "
            "Available tools: read_shift_log, inspect_service, inspect_dependency, "
            "run_diagnostic, apply_mitigation, resolve_incident\n\n"
            "IMPORTANT: Always call read_shift_log FIRST on any incident."
        )
        incident_str = (
            f"Incident: {DEMO_INCIDENT_7['symptoms']}\n"
            f"Shift log has {len(DEMO_INCIDENT_7['shift_log'])} entries from earlier.\n"
            "Take your first action now."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": incident_str},
        ]

        # ── Run up to 12 steps ─────────────────────────────────────────────
        for step_num in range(12):
            inp_ids = tok.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            if device == "cuda":
                inp_ids = inp_ids.cuda()

            with torch.no_grad():
                out_ids = model.generate(
                    inp_ids,
                    max_new_tokens=150,
                    do_sample=False,          # greedy — deterministic demo
                    pad_token_id=tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                    # NOTE: do NOT pass temperature/top_p/top_k with do_sample=False
                )

            response = tok.decode(
                out_ids[0][inp_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()

            # ── Parse tool call ────────────────────────────────────────────
            tool, args = "format_error", {}
            # Try to extract JSON even if model adds preamble text
            import re
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if json_match:
                try:
                    action = json.loads(json_match.group())
                    tool = action.get("tool", "unknown")
                    args = action.get("args", {})
                except json.JSONDecodeError:
                    pass

            # ── Track metrics ──────────────────────────────────────────────
            steps += 1

            # R2: did model call read_shift_log before any mitigation?
            if tool == "read_shift_log" and step_num == 0:
                used_recall = True  # used it as very first action — ideal

            # Simulate environment response
            env_response = _simulate_env_step(tool, args, DEMO_INCIDENT_7)

            # Format trajectory message
            icon = {
                "read_shift_log": "📖",
                "inspect_service": "🔍",
                "inspect_dependency": "🔗",
                "run_diagnostic": "🩺",
                "apply_mitigation": "🔧",
                "resolve_incident": "✅",
                "format_error": "⚠️",
            }.get(tool, "❓")

            trajectory.append({
                "role": "assistant",
                "content": f"{icon} **Step {step_num + 1}:** `{tool}`\n{env_response}",
            })

            # ── Check for resolution ───────────────────────────────────────
            if tool == "resolve_incident":
                rc = str(args.get("root_cause", "")).lower()
                correct_rc = any(
                    kw in rc for kw in
                    ["pool", "connection", "payment", "db", "#1", "incident 1",
                     "exhaustion", "cascade"]
                )
                break

            # ── Continue conversation ──────────────────────────────────────
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user",
                              "content": f"Tool result: {env_response}\nNext action:"})

        # Free GPU memory immediately after inference
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()[-600:]
        trajectory.append({
            "role": "assistant",
            "content": (
                f"⚠️ **Error loading/running model**\n\n"
                f"`{str(e)[:200]}`\n\n"
                f"Details:\n```\n{error_detail}\n```"
            ),
        })

    return trajectory, steps, used_recall, correct_rc


def run_comparison():
    results = {}

    def run_base():
        results["base"] = run_model("base", use_adapter=False)

    def run_trained():
        results["trained"] = run_model("trained", use_adapter=True)

    t1 = threading.Thread(target=run_base)
    t2 = threading.Thread(target=run_trained)
    t1.start(); t2.start()
    t1.join(); t2.join()

    # Unpack 4 values (removed error_msg from return — it's in trajectory now)
    b_traj, b_steps, b_recall, b_correct = results.get("base",  ([], 0, False, False))
    t_traj, t_steps, t_recall, t_correct = results.get("trained", ([], 0, False, False))

    improvement = max(0, b_steps - t_steps)
    verdict = f"""
    <div style="background: rgba(17, 24, 39, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 20px; margin-top: 16px;">
      <h3 style="color: #38bdf8; margin: 0 0 16px;">📊 Comparison Results</h3>
      <table style="width: 100%; color: #e2e8f0; border-collapse: collapse; font-size: 0.95rem;">
        <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
          <th style="text-align: left; padding: 10px;">Metric</th>
          <th style="text-align: center; padding: 10px;">Base Model</th>
          <th style="text-align: center; padding: 10px;">Trained Model</th>
          <th style="text-align: center; padding: 10px; color: #22c55e;">Delta</th>
        </tr>
        <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
          <td style="padding: 10px;">Steps to resolve</td>
          <td style="text-align: center; padding: 10px;">{b_steps if b_steps > 0 else "timeout"}</td>
          <td style="text-align: center; padding: 10px;">{t_steps if t_steps > 0 else "timeout"}</td>
          <td style="text-align: center; padding: 10px; color: #22c55e; font-weight: 700;">{"−" + str(improvement) + " steps" if improvement > 0 else "—"}</td>
        </tr>
        <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
          <td style="padding: 10px;">Used shift log first</td>
          <td style="text-align: center; padding: 10px;">{"✅" if b_recall else "❌"}</td>
          <td style="text-align: center; padding: 10px;">{"✅" if t_recall else "❌"}</td>
          <td style="text-align: center; padding: 10px; color: {"#22c55e" if t_recall and not b_recall else "#94a3b8"}; font-weight: 700;">
            {"Memory policy active ✅" if t_recall and not b_recall else "—"}
          </td>
        </tr>
        <tr>
          <td style="padding: 10px;">Correct root cause</td>
          <td style="text-align: center; padding: 10px;">{"✅" if b_correct else "❌"}</td>
          <td style="text-align: center; padding: 10px;">{"✅" if t_correct else "❌"}</td>
          <td style="text-align: center; padding: 10px; color: {"#22c55e" if t_correct and not b_correct else "#94a3b8"}; font-weight: 700;">
            {"Causal reasoning trained ✅" if t_correct and not b_correct else "—"}
          </td>
        </tr>
      </table>
    </div>
    """

    return (
        b_traj, t_traj,
        b_steps, t_steps,
        b_recall, t_recall,
        b_correct, t_correct,
        verdict,
    )


def handle_start_comparison():
    ok, msg = _check_vram_for_comparison()
    if not ok:
        return (
            [], [], 0, 0, False, False, False, False,
            f'<div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 10px; padding: 16px; margin-top: 16px; color: #ef4444;">'
            f'⚠️ {msg}</div>'
        )
    return run_comparison()


def build_comparison_tab():
    """Build the side-by-side model comparison tab."""
    with gr.Tab("⚡ Base vs Trained"):
        gr.HTML(
            """
            <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 20px; margin-bottom: 16px;">
                <h3 style="color: #6366f1; margin: 0 0 8px; font-size: 1.1rem;">⚡ Live Model Comparison — Incident #7: Auth Service Cascade</h3>
                <p style="color: #94a3b8; margin: 0; line-height: 1.6;">
                    Both models receive the SAME incident context and shift log from Incident #1. 
                    Watch which model reads it first and reaches the correct root cause.
                </p>
            </div>
        """
        )

        with gr.Row():
            # ── Shared context panel ──────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Shared Incident Context")
                incident_display = gr.JSON(
                    value=DEMO_INCIDENT_7,
                    label="Incident #7 — both models see this",
                )
                run_btn = gr.Button("🚀 Run Both Models Simultaneously", variant="primary", size="lg")
                gr.Markdown(
                    """
                    **What to watch:**
                    - Does the model call `read_shift_log` FIRST?
                    - Does it reference Incident #1?
                    - How many steps to resolve?
                    - Did it identify the correct root cause?
                    """
                )

        with gr.Row():
            # ── Base model column ─────────────────────────────────────────
            with gr.Column(scale=1):
                gr.HTML(
                    """
                    <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 10px; padding: 12px; margin-bottom: 12px;">
                        <div style="font-weight: 700; color: #ef4444;">❌ Base Model</div>
                        <div style="font-size: 0.85rem; color: #94a3b8;">Qwen2.5-1.5B — no training</div>
                    </div>
                    """
                )
                base_steps = gr.Number(
                    label="Steps taken", value=0, interactive=False
                )
                base_recall = gr.Checkbox(
                    label="Used shift log before mitigation", value=False, interactive=False
                )
                base_correct = gr.Checkbox(
                    label="Correct root cause identified", value=False, interactive=False
                )
                base_trajectory = gr.Chatbot(
                    label="Base model trajectory",
                    height=400,
                )

            # ── Trained model column ──────────────────────────────────────
            with gr.Column(scale=1):
                gr.HTML(
                    """
                    <div style="background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 10px; padding: 12px; margin-bottom: 12px;">
                        <div style="font-weight: 700; color: #22c55e;">✅ Trained Model</div>
                        <div style="font-size: 0.85rem; color: #94a3b8;">Qwen2.5-1.5B + ShiftLog-Gym LoRA</div>
                    </div>
                    """
                )
                trained_steps = gr.Number(
                    label="Steps taken", value=0, interactive=False
                )
                trained_recall = gr.Checkbox(
                    label="Used shift log before mitigation", value=False, interactive=False
                )
                trained_correct = gr.Checkbox(
                    label="Correct root cause identified", value=False, interactive=False
                )
                trained_trajectory = gr.Chatbot(
                    label="Trained model trajectory",
                    height=400,
                )

        # ── Verdict panel ─────────────────────────────────────────────────
        verdict_html = gr.HTML(
            value="<p style='color: #94a3b8; text-align: center;'>Click the button above to run the comparison.</p>"
        )

        run_btn.click(
            fn=handle_start_comparison,
            inputs=[],
            outputs=[
                base_trajectory,
                trained_trajectory,
                base_steps,
                trained_steps,
                base_recall,
                trained_recall,
                base_correct,
                trained_correct,
                verdict_html,
            ],
        )
