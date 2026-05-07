"""
inference_tab.py

Live inference tab for the ShiftLog-Gym observatory.
Judges can paste an incident and watch the trained model emit tool calls.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import gradio as gr
import torch


BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_REPO = "Chirag0123/shiftlog-gym-qwen-memory-policy"

_model_cache: dict[str, Any] = {}


INCIDENT_TEMPLATES = {
    "Custom - type your own": "",
    "Incident #7 - Auth Cascade (causal)": (
        "Auth service is returning 503s. Error rate 67%. "
        "Latency p99 8200ms. This started 20 minutes ago. "
        "Shift log: 'Payment DB connection pool at 80% - auth service at risk.'"
    ),
    "Incident #9 - Notification OOM (causal)": (
        "Notification service pod has been OOMKilled twice in the last hour. "
        "Error rate 100%. Shift log: 'Notification pod memory limit too low - "
        "recurrence risk noted after incident #3.'"
    ),
    "Incident #2 - API Gateway 429": (
        "API Gateway returning 429 errors at high rate. "
        "Rate limiter appears misconfigured after today's deploy. "
        "No relevant shift log entries for this service."
    ),
    "Novel Incident - test generalization": (
        "Payment service latency spiking. p99 = 4200ms. "
        "DB connection pool at 92%. Auth service starting to queue requests. "
        "Shift log is empty - start of shift."
    ),
}

SYSTEM_PROMPT = """You are an expert SRE on-call agent managing an 8-hour shift.
For every action, output ONLY a single JSON object on one line:
{"tool": "", "arguments": {}}

No markdown, no preamble, no explanation outside the JSON.

Available tools:
  read_shift_log(query)
  append_shift_log(entry)
  inspect_service(service)
  inspect_dependency(service)
  run_diagnostic(service, diagnostic)
  apply_mitigation(service, mitigation)
  resolve_incident(root_cause)

RULE: Always call read_shift_log FIRST on any incident.
"""


def _adapter_status_label() -> str:
    status = _model_cache.get("adapter_status")
    if status == "loaded":
        return "Adapter loaded - showing trained model responses"
    if status == "missing":
        return "Adapter not found - showing base model responses"
    if status == "error":
        return "Adapter load failed - showing base model responses"
    return "Adapter will load on first inference"


def _load_model():
    if "model" in _model_cache and "tok" in _model_cache:
        return _model_cache["model"], _model_cache["tok"]

    from huggingface_hub import list_repo_files
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        token=hf_token,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        token=hf_token,
    )

    _model_cache["adapter_status"] = "missing"
    try:
        repo_files = list(list_repo_files(ADAPTER_REPO, repo_type="model", token=hf_token))
        if any(name in repo_files for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")):
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                ADAPTER_REPO,
                token=hf_token,
                is_trainable=False,
            )
            _model_cache["adapter_status"] = "loaded"
        else:
            _model_cache["adapter_status"] = "missing"
    except Exception:
        _model_cache["adapter_status"] = "error"

    model.eval()
    _model_cache["model"] = model
    _model_cache["tok"] = tok
    return model, tok


def _load_template(template_name: str) -> str:
    return INCIDENT_TEMPLATES.get(template_name, "")


def _extract_action(response: str, service_hint: str) -> tuple[str, dict[str, Any]]:
    tool = "format_error"
    arguments: dict[str, Any] = {}

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return tool, arguments

    try:
        action = json.loads(match.group(0))
    except json.JSONDecodeError:
        return tool, arguments

    tool = action.get("tool", "format_error")
    arguments = action.get("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}

    if tool == "inspect_service" and "service" not in arguments:
        arguments["service"] = service_hint
    return tool, arguments


def _sim(tool: str, arguments: dict[str, Any], context: str) -> str:
    ctx_lower = context.lower()
    service = arguments.get("service", "service")

    if tool == "read_shift_log":
        if "auth" in ctx_lower or "503" in ctx_lower:
            return (
                "Found 1 causal entry: 'Payment DB connection pool at 80% - "
                "auth service at risk under load.' (Incident #1, 4h ago)"
            )
        if "notification" in ctx_lower or "oom" in ctx_lower:
            return (
                "Found 1 recurrence entry: 'Notification pod memory limit too low - "
                "recurrence risk noted.' (Incident #3, 3h ago)"
            )
        return "No relevant entries found for the current incident."
    if tool == "inspect_service":
        return (
            f"Service: {service} | Owner: platform-team | "
            "Dependencies: [payment-db, cache-svc, session-store]"
        )
    if tool == "inspect_dependency":
        return f"{service} -> payment-db (shared connection pool, max_connections=100)"
    if tool == "run_diagnostic":
        if "auth" in ctx_lower:
            return (
                "payment-db pool at 97/100 active connections. "
                "Auth service is queuing on DB connections."
            )
        return f"{service} diagnostic: anomaly detected in resource utilization."
    if tool == "apply_mitigation":
        mitigation = arguments.get("mitigation", arguments.get("action", "unknown"))
        return f"Applied {mitigation} to {service}. Error rate is dropping and latency is recovering."
    if tool == "append_shift_log":
        return f"Logged shift memory entry: {arguments}"
    if tool == "resolve_incident":
        root_cause = arguments.get("root_cause", "unspecified")
        return f"Incident closed. Root cause recorded: {root_cause}. Handoff updated."
    return f"Unable to interpret tool call '{tool}'."


def _metrics_html(used_recall: bool, steps_taken: int, elapsed: float, correct_order: bool) -> str:
    recall_text = "Read shift log first" if used_recall else "Skipped shift log first"
    order_text = "Correct order" if correct_order else "Acted before recall"
    recall_color = "#4ade80" if used_recall else "#f87171"
    order_color = "#4ade80" if correct_order else "#fbbf24"
    return f"""
    <div style="display:grid;grid-template-columns:repeat(4, minmax(120px, 1fr));gap:12px;margin-top:12px">
      <div class="metric-card"><div style="color:#94a3b8;font-size:.8rem">Memory policy</div><div style="color:{recall_color};font-weight:700">{recall_text}</div></div>
      <div class="metric-card"><div style="color:#94a3b8;font-size:.8rem">Steps</div><div style="color:#e2e8f0;font-weight:700">{steps_taken}</div></div>
      <div class="metric-card"><div style="color:#94a3b8;font-size:.8rem">Inference time</div><div style="color:#e2e8f0;font-weight:700">{elapsed:.1f}s</div></div>
      <div class="metric-card"><div style="color:#94a3b8;font-size:.8rem">Tool order</div><div style="color:{order_color};font-weight:700">{order_text}</div></div>
    </div>
    """


def _run_inference(incident_text: str, max_steps_val: int):
    if not incident_text.strip():
        yield [(None, "Please enter an incident description.")], ""
        return

    try:
        model, tok = _load_model()
    except Exception as exc:
        yield [(None, f"Model load failed: {str(exc)[:300]}")], ""
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": incident_text},
    ]
    trajectory = []
    used_recall = False
    correct_order = True
    steps_taken = 0
    started = time.time()

    device = next(model.parameters()).device
    service_hint = "auth-service" if "auth" in incident_text.lower() else "service"

    for step_idx in range(int(max_steps_val)):
        inputs = tok.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        response = tok.decode(output[0][input_len:], skip_special_tokens=True).strip()
        tool, arguments = _extract_action(response, service_hint)

        steps_taken += 1
        if tool == "read_shift_log" and step_idx == 0:
            used_recall = True
        if tool in {"apply_mitigation", "resolve_incident"} and step_idx == 0:
            correct_order = False

        env_response = _sim(tool, arguments, incident_text)
        icon = {
            "read_shift_log": "📖",
            "append_shift_log": "✍️",
            "inspect_service": "🔍",
            "inspect_dependency": "🔗",
            "run_diagnostic": "🩺",
            "apply_mitigation": "🔧",
            "resolve_incident": "✅",
            "format_error": "⚠️",
        }.get(tool, "❓")

        message = (
            f"{icon} **Step {step_idx + 1}:** `{tool}`\n"
            f"```json\n{json.dumps(arguments, indent=2)}\n```\n"
            f"**Result:** {env_response}"
        )
        trajectory.append((None, message))
        yield trajectory, ""

        if tool == "resolve_incident":
            break

        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Result: {env_response}\nNext action:"})

    elapsed = time.time() - started
    yield trajectory, _metrics_html(used_recall, steps_taken, elapsed, correct_order)


def build_inference_tab():
    with gr.Tab("🧪 Try the Model"):
        gr.HTML(
            f"""
            <div style="background:rgba(17,24,39,.85);border:1px solid rgba(56,189,248,.2);border-radius:12px;padding:20px;margin-bottom:12px">
              <h3 style="color:#38bdf8;margin:0 0 8px">Live Inference - ShiftLog-Gym Trained Model</h3>
              <p style="color:#94a3b8;margin:0;line-height:1.7">
                Type any SRE incident and watch the trained memory policy emit structured tool calls.
                Canonical success signal: step 1 should be <code>read_shift_log</code>.
              </p>
              <p style="color:#e2e8f0;margin:10px 0 0;font-size:.95rem">{_adapter_status_label()}</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                template_picker = gr.Dropdown(
                    choices=list(INCIDENT_TEMPLATES.keys()),
                    value="Incident #7 - Auth Cascade (causal)",
                    label="Incident template",
                )
                incident_input = gr.Textbox(
                    label="Incident description",
                    value=INCIDENT_TEMPLATES["Incident #7 - Auth Cascade (causal)"],
                    lines=6,
                    placeholder="Describe the incident here...",
                )
                max_steps = gr.Slider(
                    minimum=3,
                    maximum=12,
                    value=7,
                    step=1,
                    label="Max tool call steps",
                )
                run_btn = gr.Button("Run Trained Model", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", size="sm")
                gr.Markdown(
                    """
                    **What to look for**
                    - Step 1 should be `read_shift_log`
                    - Retrieved memory should point to the likely precursor
                    - Linked incidents should resolve in a small number of steps
                    - Malformed output should surface as a visible format error, not a crash
                    """
                )

            with gr.Column(scale=2):
                trajectory_output = gr.Chatbot(
                    label="Agent trajectory",
                    height=500,
                )
                metrics_row = gr.HTML(value="")

        template_picker.change(
            fn=_load_template,
            inputs=[template_picker],
            outputs=[incident_input],
        )
        run_btn.click(
            fn=_run_inference,
            inputs=[incident_input, max_steps],
            outputs=[trajectory_output, metrics_row],
        )
        clear_btn.click(
            fn=lambda: ([], ""),
            outputs=[trajectory_output, metrics_row],
        )
