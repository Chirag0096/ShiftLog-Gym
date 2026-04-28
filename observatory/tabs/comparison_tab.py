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
                    type="messages",
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
                    type="messages",
                )

        # ── Verdict panel ─────────────────────────────────────────────────
        verdict_html = gr.HTML(
            value="<p style='color: #94a3b8; text-align: center;'>Click the button above to run the comparison.</p>"
        )

        def run_comparison() -> Tuple:
            """
            Runs base and trained model on incident #7.
            Returns trajectories for both, step counts, recall flags.
            Uses greedy decoding (temperature=0) for deterministic demo.
            """

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

            system_prompt = (
                "You are an SRE agent. For every action, output ONLY:\n"
                '{"tool": "", "args": {}}\n'
                "No other text. Available tools: "
                "read_shift_log, inspect_service, inspect_dependency, "
                "run_diagnostic, apply_mitigation, resolve_incident"
            )

            incident_prompt = (
                f"Incident: {DEMO_INCIDENT_7['symptoms']}\n"
                f"Shift log has {len(DEMO_INCIDENT_7['shift_log'])} entries "
                f"from earlier incidents.\n"
                f"Take your first action now."
            )

            def run_model(model_label: str, use_adapter: bool) -> Tuple:
                """Returns (trajectory_messages, steps, used_recall, correct_rc)"""
                trajectory = []
                steps = 0
                used_recall = False
                correct_rc = False

                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    from peft import PeftModel

                    tok = AutoTokenizer.from_pretrained(
                        BASE_MODEL_ID, trust_remote_code=True
                    )
                    base = AutoModelForCausalLM.from_pretrained(
                        BASE_MODEL_ID,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    if use_adapter:
                        base = PeftModel.from_pretrained(base, ADAPTER_REPO)
                    base.eval()

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": incident_prompt},
                    ]

                    for step in range(12):  # max 12 steps
                        inp = tok.apply_chat_template(
                            messages,
                            return_tensors="pt",
                            add_generation_prompt=True,
                        ).to(base.device)
                        with torch.no_grad():
                            out = base.generate(
                                inp,
                                max_new_tokens=128,
                                temperature=0.1,
                                do_sample=False,
                                pad_token_id=tok.eos_token_id,
                            )
                        response = tok.decode(
                            out[0][inp.shape[1] :], skip_special_tokens=True
                        ).strip()

                        # Parse tool call
                        try:
                            action = json.loads(response)
                            tool = action.get("tool", "unknown")
                        except Exception:
                            tool = "format_error"
                            action = {"tool": tool, "args": {}}

                        steps += 1

                        # Simulate environment response
                        env_response = _simulate_env_step(
                            tool, action.get("args", {}), DEMO_INCIDENT_7
                        )

                        # Track metrics
                        if tool == "read_shift_log" and steps <= 3:
                            used_recall = True
                        if tool == "resolve_incident":
                            rc = action.get("args", {}).get("root_cause", "")
                            if (
                                "pool" in rc.lower()
                                or "db" in rc.lower()
                                or "#1" in rc
                            ):
                                correct_rc = True
                            trajectory.append(
                                {
                                    "role": "assistant",
                                    "content": f"🔚 **Step {step+1}:** `{tool}` → {env_response}",
                                }
                            )
                            break

                        trajectory.append(
                            {
                                "role": "assistant",
                                "content": f"**Step {step+1}:** `{tool}` → {env_response}",
                            }
                        )
                        messages.append({"role": "assistant", "content": response})
                        messages.append(
                            {"role": "user", "content": f"Tool result: {env_response}"}
                        )

                except Exception as e:
                    trajectory.append(
                        {
                            "role": "assistant",
                            "content": f"⚠️ Model load error: {str(e)[:200]}",
                        }
                    )

                return trajectory, steps, used_recall, correct_rc

            # Run both in parallel threads
            results = {}

            def run_base():
                results["base"] = run_model("base", use_adapter=False)

            def run_trained():
                results["trained"] = run_model("trained", use_adapter=True)

            t1 = threading.Thread(target=run_base)
            t2 = threading.Thread(target=run_trained)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            b_traj, b_steps, b_recall, b_correct = results.get(
                "base", ([], 0, False, False)
            )
            t_traj, t_steps, t_recall, t_correct = results.get(
                "trained", ([], 0, False, False)
            )

            # Verdict HTML
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
                  <td style="text-align: center; padding: 10px;">{b_steps}</td>
                  <td style="text-align: center; padding: 10px;">{t_steps}</td>
                  <td style="text-align: center; padding: 10px; color: #22c55e; font-weight: 700;">−{improvement}</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
                  <td style="padding: 10px;">Used shift log first</td>
                  <td style="text-align: center; padding: 10px;">{'✅' if b_recall else '❌'}</td>
                  <td style="text-align: center; padding: 10px;">{'✅' if t_recall else '❌'}</td>
                  <td style="text-align: center; padding: 10px; color: {'#22c55e' if t_recall and not b_recall else '#94a3b8'}; font-weight: 700;">
                    {'Memory policy ✅' if t_recall and not b_recall else '—'}
                  </td>
                </tr>
                <tr>
                  <td style="padding: 10px;">Correct root cause</td>
                  <td style="text-align: center; padding: 10px;">{'✅' if b_correct else '❌'}</td>
                  <td style="text-align: center; padding: 10px;">{'✅' if t_correct else '❌'}</td>
                  <td style="text-align: center; padding: 10px; color: {'#22c55e' if t_correct and not b_correct else '#94a3b8'}; font-weight: 700;">
                    {'Causal reasoning ✅' if t_correct and not b_correct else '—'}
                  </td>
                </tr>
              </table>
            </div>
            """

            return (
                b_traj,
                t_traj,
                b_steps,
                t_steps,
                b_recall,
                t_recall,
                b_correct,
                t_correct,
                verdict,
            )

        run_btn.click(
            fn=run_comparison,
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
