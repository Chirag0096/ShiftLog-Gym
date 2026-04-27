"""ShiftLog-Gym Observatory — 5-tab Gradio dashboard."""
from __future__ import annotations
import json, os, time, subprocess, sys
from pathlib import Path
import gradio as gr
import pandas as pd
import httpx
import threading
from train.space_training_daemon import run_training, get_state

ROOT     = Path(__file__).resolve().parent.parent
OBS      = ROOT / "observatory"
PLOTS    = ROOT / "plots"
API_BASE = os.environ.get("SPACE_API_URL", "http://localhost:7860")
TIMEOUT  = 10.0

# Attempt to import internal logic for direct calls (bypasses HTTP deadlocks)
try:
    from shiftlog_gym.server.core import (
        internal_reset, internal_step, internal_get_state, internal_get_tools
    )
    HAS_INTERNAL = True
except ImportError:
    HAS_INTERNAL = False

TOOL_EXAMPLES = {
    "/reset": {"seed": 1, "family": "db_pool"},
    "/step":  {"tool": "read_shift_log", "arguments": {"query": "payment service", "limit": 3}},
    "/state": {},
    "/tools": {},
}

DEMO_STEPS = [
    ("read_shift_log",   {"query": "payment service", "limit": 5}),
    ("inspect_service",  {"service": "payments-api"}),
    ("run_diagnostic",   {"service": "payments-api", "diagnostic": "check_pool"}),
    ("apply_mitigation", {"service": "payments-api", "mitigation": "set_pool_size_and_restart"}),
    ("resolve_incident", {"incident_id": "INC-001", "resolution": "Pool size restored",
                          "root_cause": "stale DB pool size after rollback"}),
]

custom_css = """
@import url("https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono&display=swap");
body,.gradio-container{font-family:"Outfit",sans-serif!important;background:radial-gradient(circle at top right, #1e1b4b, #0f172a, #020617)!important;color:#f8fafc!important}
.glass{background:rgba(30,41,59,0.5)!important;backdrop-filter:blur(16px)!important;
  border:1px solid rgba(255,255,255,0.1)!important;border-radius:24px!important;padding:28px!important;
  box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37)!important;}
.btn-premium{background:linear-gradient(135deg,#6366f1,#a855f7)!important;color:#fff!important;font-weight:700!important;border:none!important;
  border-radius:12px!important;transition:all 0.3s ease!important;box-shadow:0 4px 15px rgba(99,102,241,0.4)!important}
.btn-premium:hover{transform:translateY(-2px)!important;box-shadow:0 6px 20px rgba(99,102,241,0.6)!important}
.metric-card{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:20px;text-align:center;transition:all 0.3s ease}
.metric-card:hover{background:rgba(255,255,255,0.05);transform:scale(1.02)}
.gradio-container [role="tablist"]{background:rgba(255,255,255,0.03)!important;border-radius:14px!important;padding:4px!important;margin-bottom:20px!important}
.gradio-container [role="tab"]{border-radius:10px!important;font-weight:600!important;color:#94a3b8!important;border:none!important}
.gradio-container [role="tab"][aria-selected="true"]{background:#6366f1!important;color:white!important}
"""

def _render_training_status(state: dict) -> str:
    """Renders the status card HTML for the training tab."""
    status = state["status"]
    color_map = {
        "idle": "#888888",
        "running": "#EF9F27",
        "complete": "#1D9E75",
        "error": "#E24B4A",
    }
    color = color_map.get(status, "#888888")
    icon_map = {"idle": "⏸", "running": "⚙️", "complete": "✅", "error": "❌"}
    icon = icon_map.get(status, "⏸")
    stage = state.get("stage", "")
    step = state.get("step", 0)
    total = state.get("total_steps", 550)
    pct = min(100, int(step / total * 100)) if total > 0 else 0
    started = state.get("started_at", "")
    completed = state.get("completed_at", "")

    return f"""
    <div style="background:rgba(17,24,39,0.9); border:1px solid {color}44; border-radius:12px; padding:20px; color:white;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
        <span style="font-size:1.2rem; font-weight:700; color:{color}">{icon} {status.upper()}</span>
        <span style="font-size:0.9rem; color:#94a3b8;">{stage or '—'} | Step {step}/{total}</span>
      </div>
      <div style="background:rgba(255,255,255,0.05); border-radius:10px; height:8px; overflow:hidden; margin-bottom:10px;">
        <div style="background:{color}; width:{pct}%; height:100%; transition: width 0.5s ease;"></div>
      </div>
      <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#64748b;">
        <span>{f'Started: {started[:16].replace("T", " ")}' if started else ''}</span>
        <span>{f'Finished: {completed[:16].replace("T", " ")}' if completed else ''}</span>
      </div>
    </div>
    """

def _get_gpu_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            used = torch.cuda.memory_allocated(0) / 1e9
            return f"GPU: {name}\nVRAM total: {total:.1f}GB\nVRAM used: {used:.2f}GB"
        return "No GPU detected.\nUpgrade hardware in Space Settings → L4 x1"
    except Exception as e:
        return f"GPU check failed: {e}"

# ── helpers ──────────────────────────────────────────────────────────────────

def _api(method: str, path: str, body=None):
    """
    Unified API caller. Uses direct function calls if HAS_INTERNAL=True
    (ignoring API_BASE to avoid external network overhead/deadlocks).
    """
    start = time.monotonic()
    
    # ALWAYS use internal functions if available when running in the same process
    if HAS_INTERNAL:
        try:
            if path == "/reset":
                data = internal_reset(body)
            elif path == "/step":
                data = internal_step(body)
            elif path == "/state":
                data = internal_get_state()
            elif path == "/tools":
                data = internal_get_tools()
            else:
                raise ValueError(f"Unknown internal path: {path}")
            
            ms = round((time.monotonic() - start) * 1000)
            return data, ms, None
        except Exception as exc:
            ms = round((time.monotonic() - start) * 1000)
            return {}, ms, str(exc)

    # Fallback to HTTP (only if internal functions are not available)
    try:
        fn = httpx.post if method == "POST" else httpx.get
        kw = {"timeout": TIMEOUT}
        if body is not None:
            kw["json"] = body
        r = fn(f"{API_BASE}{path}", **kw)
        ms = round((time.monotonic() - start) * 1000)
        return r.json(), ms, None
    except Exception as exc:
        ms = round((time.monotonic() - start) * 1000)
        return {}, ms, str(exc)

def _load_baselines():
    p = OBS / "baselines.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def _load_health():
    p = OBS / "health_status.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

# ── Tab 1 ────────────────────────────────────────────────────────────────────

def system_status_md():
    h = _load_health()
    if not h:
        return "🟡 **Status:** Health check not yet run"
    ts = h.get("timestamp", "")[:19].replace("T", " ")
    icons = {"ok": "🟢", "degraded": "🟡", "down": "🔴"}
    icon = icons.get(h.get("overall", "unknown"), "🟡")
    label = h.get("overall", "unknown").upper()
    return f"{icon} **{label}** — Last checked: `{ts} UTC`"

def start_demo():
    data, ms, err = _api("POST", "/reset", {"seed": 1, "family": "db_pool"})
    if err:
        return [{"role": "assistant", "content": f"❌ API error: {err}"}], "Reset failed"
    msg = data.get("message", "Episode started.")
    chat = [{"role": "assistant", "content": f"🚨 **New Incident**\n\n{msg[:600]}"}]
    info = "Step 0/5 — Click **Next Step** to run the demo action sequence"
    return chat, info

def next_step(chat, step_idx):
    if step_idx >= len(DEMO_STEPS):
        chat.append({"role": "assistant", "content": "✅ **Demo complete!** All 5 steps executed."})
        return chat, step_idx, "Episode complete"
    tool, args = DEMO_STEPS[step_idx]
    data, ms, err = _api("POST", "/step", {"tool": tool, "arguments": args})
    if err:
        chat.append({"role": "user", "content": f"`{tool}`"})
        chat.append({"role": "assistant", "content": f"❌ Error: {err}"})
        return chat, step_idx + 1, f"Error at step {step_idx + 1}"
    msg = data.get("message", "")
    rew = data.get("reward", 0.0)
    icons = {"read_shift_log": "📖", "inspect_service": "🔍", "run_diagnostic": "🧪",
             "apply_mitigation": "🛠", "resolve_incident": "✅"}
    icon = icons.get(tool, "⚙️")
    chat.append({"role": "user", "content": f"{icon} `{tool}`"})
    chat.append({"role": "assistant", "content": f"**Reward:** {rew:+.3f} | {msg[:400]}"})
    return chat, step_idx + 1, f"Step {step_idx + 1}/5 | {tool} | {ms}ms"

# ── Tab 2 ────────────────────────────────────────────────────────────────────

def training_banner():
    bl = _load_baselines()
    meta = bl.get("_metadata", {})
    note = meta.get("note", "")
    run_id = meta.get("run_id", "pending")
    if not bl:
        return "⚠️ **No baselines.json** — run training first."
    if "simulated" in note.lower() or run_id == "pending":
        return ("⚠️ **Showing projected estimates.**  "
                "Run `train/02_grpo_train_colab.ipynb` to populate with real results.")
    return f"✅ **Real GRPO training results** — Run ID: `{run_id}` | {meta.get('timestamp','')[:10]}"

def metric_cards_html():
    bl  = _load_baselines()
    rnd = bl.get("random", {})
    trn = bl.get("trained_llm", {})
    recall_b  = rnd.get("recall_before_action_rate", 0.12)
    recall_t  = trn.get("recall_before_action_rate", 0.78)
    mttr_b    = rnd.get("linked_incident_mttr", 24.5)
    mttr_t    = trn.get("linked_incident_mttr", 3.5)
    steps     = bl.get("_metadata", {}).get("training_steps", "—")
    def card(title, v1, v2, unit, up):
        arrow = "↑" if up else "↓"
        colour = "#4ade80" if up else "#f87171"
        return (
            f'<div style="background:rgba(17,24,39,.85);border:1px solid rgba(255,255,255,.1);'
            f'border-radius:12px;padding:22px;text-align:center;flex:1;min-width:170px">'
            f'<div style="color:#94a3b8;font-size:.85rem;margin-bottom:6px">{title}</div>'
            f'<div style="font-size:1.9rem;font-weight:800;color:#e2e8f0">{v2}{unit}</div>'
            f'<div style="color:{colour};font-size:.9rem;margin-top:4px">{arrow} from {v1}{unit}</div></div>'
        )
    row = (card("Recall Rate", f"{recall_b:.0%}", f"{recall_t:.0%}", "", True) +
           card("MTTR (Linked)", f"{mttr_b:.1f}", f"{mttr_t:.1f}", " steps", False) +
           card("Training Steps", "0", str(steps), "", True))
    return f'<div style="display:flex;gap:16px;flex-wrap:wrap;margin:16px 0">{row}</div>'

def load_plot(name):
    p = PLOTS / name
    return str(p) if p.exists() else None

def refresh_results():
    return (training_banner(), metric_cards_html(),
            load_plot("01_reward_curve.png"),
            load_plot("02_recall_bonus_curve.png"),
            load_plot("03_mttr_comparison.png"))

def get_training_status():
    """Read full background training log with tail-like behavior."""
    log_file = OBS / "training_full_log.txt"
    if not log_file.exists():
        import torch
        gpu_info = f" | GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else " | ❌ No GPU"
        return f"🟢 **Status:** Idle {gpu_info}\n\n*Logs will appear here once training starts.*"
    
    try:
        # Read the last 8k characters
        with open(log_file, "r") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8000)) 
            content = f.read()
        
        lines = content.split("\n")
        if len(lines) > 1:
            lines = lines[1:]
        
        return "\n".join(lines[-100:]) # Show last 100 lines
    except Exception as e:
        return f"Error reading logs: {e}"

def start_hf_training(repo_id, wandb_key):
    """Spawn run_training.py as a detached background process with environment variables."""
    status_file = OBS / "training_status.txt"
    log_path = OBS / "training_full_log.txt"
    
    # Check if already running
    if status_file.exists():
        current = status_file.read_text(encoding="utf-8")
        if "..." in current and "Done!" not in current and "ERROR" not in current:
             return gr.update(interactive=False), "⚠️ Training already in progress!"
    
    # Initialize/Clear files
    status_file.write_text("🚀 [System] Launching training subprocess...\n", encoding="utf-8")
    log_path.write_text("--- Training Log Started ---\n", encoding="utf-8")
    
    script_path = ROOT / "run_training.py"
    
    # Prepare environment
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1" # CRITICAL for live logs
    if wandb_key:
        env["WANDB_API_KEY"] = wandb_key.strip()
        env["WANDB_MODE"] = "online"
    else:
        env["WANDB_MODE"] = "disabled"
    
    # Run as a detached subprocess
    with open(log_path, "a") as f:
        subprocess.Popen(
            [sys.executable, str(script_path), repo_id],
            cwd=str(ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True
        )
    
    return gr.update(interactive=False), "✅ Training process spawned! Switch to terminal view below."

# ── Tab 3 ────────────────────────────────────────────────────────────────────

_CAP_ROWS = "".join(
    '<tr style="border-bottom:1px solid rgba(255,255,255,.06)">'
    f'<td style="padding:10px;color:#e2e8f0">{c}</td>'
    '<td style="text-align:center;color:#f87171">✗</td>'
    '<td style="text-align:center;color:#f87171">✗</td>'
    '<td style="text-align:center;color:#4ade80">✅</td></tr>'
    for c in ["Causal incident chains","SRE domain","Verifiable MTTR",
              "Noise resistance","Handoff memory","OpenEnv compliant"]
)
_REWARD_ROWS = "".join(
    '<tr style="border-bottom:1px solid rgba(255,255,255,.06)">'
    f'<td style="padding:8px;color:#e2e8f0">{n}</td>'
    f'<td style="padding:8px;text-align:center;color:#38bdf8;font-weight:700">{w}</td>'
    f'<td style="padding:8px;color:#94a3b8;font-size:.9rem">{d}</td></tr>'
    for n,w,d in [
        ("R1 Success","35%","Correct root cause + mitigation"),
        ("R2 Recall","25%","Read shift log before linked-incident action"),
        ("R3 Memory Write","15%","Quality of shift log entries"),
        ("R4 Memory Integrity","10%","Penalises contradictions & duplicates"),
        ("R5 Efficiency","5%","Fewer tool calls = higher score"),
        ("Hallucination","5%","Penalises invalid mitigations"),
        ("Noise Resistance","3%","Resists misleadingly similar noise incidents"),
        ("Handoff Quality","2%","Structured shift handoff summary"),
    ]
)
_FAMILY_PILLS = "".join(
    f'<span style="background:rgba(99,102,241,.2);border:1px solid rgba(99,102,241,.4);'
    f'border-radius:8px;padding:8px 14px;font-size:.9rem;color:#e2e8f0">{f}</span>'
    for f in ["db_pool","auth_cascade","oom_regression","feature_flag","network_partition","config_drift"]
)
STORY_HTML = f"""<div style="max-width:860px;margin:0 auto;color:#e2e8f0;font-family:Inter,sans-serif">
<h2 style="background:linear-gradient(90deg,#38bdf8,#818cf8);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent">The Memory Gap in Frontier AI</h2>
<p style="color:#94a3b8;line-height:1.8">Every frontier lab ships memory as a product feature.
None trained the model to decide <em>what to write, when to retrieve, and what to safely forget.</em>
ShiftLog-Gym fills this gap with a verifiable, causal SRE environment and 8 grounded reward signals.</p>
<table style="width:100%;border-collapse:collapse;margin:20px 0;font-size:.95rem">
<thead><tr style="border-bottom:2px solid rgba(255,255,255,.2)">
<th style="padding:10px;text-align:left;color:#94a3b8">Capability</th>
<th style="padding:10px;color:#94a3b8">Memory-R1</th>
<th style="padding:10px;color:#94a3b8">MemAgent</th>
<th style="padding:10px;color:#4ade80">ShiftLog-Gym</th></tr></thead>
<tbody>{_CAP_ROWS}</tbody></table>
<h3 style="color:#818cf8;margin-top:32px">8 Reward Signals (weights sum to 1.0)</h3>
<table style="width:100%;border-collapse:collapse;margin:12px 0;font-size:.9rem">
<thead><tr style="border-bottom:2px solid rgba(255,255,255,.2)">
<th style="padding:8px;text-align:left;color:#94a3b8">Signal</th>
<th style="padding:8px;color:#94a3b8">Weight</th>
<th style="padding:8px;color:#94a3b8">What it measures</th></tr></thead>
<tbody>{_REWARD_ROWS}</tbody></table>
<h3 style="color:#818cf8;margin-top:32px">6 Scenario Families</h3>
<div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:12px">{_FAMILY_PILLS}</div></div>"""

# ── Tab 4 ────────────────────────────────────────────────────────────────────

def api_explore(endpoint, body_str):
    try:
        body = json.loads(body_str) if body_str.strip() else {}
    except Exception as exc:
        return json.dumps({"error": f"JSON parse error: {exc}"}), "—"
    method = "GET" if endpoint in ("/state", "/tools") else "POST"
    data, ms, err = _api(method, endpoint, body if method == "POST" else None)
    out = {"error": err} if err else data
    return json.dumps(out, indent=2), f"{ms}ms"

def prefill_body(endpoint):
    return json.dumps(TOOL_EXAMPLES.get(endpoint, {}), indent=2)

# ── Tab 5 ────────────────────────────────────────────────────────────────────

def run_health_check_ui():
    try:
        from shiftlog_gym.diagnostics.health_check import run_full_health_check
        # Pass direct callers if available to avoid deadlocks
        call_direct = None
        if HAS_INTERNAL:
            call_direct = {
                "/reset": internal_reset,
                "/step": internal_step,
                "/state": internal_get_state,
                "/tools": internal_get_tools
            }
        h = run_full_health_check(base_url=API_BASE, call_direct=call_direct)
    except Exception as exc:
        empty = f"Health check failed: {exc}"
        return empty, empty, empty, empty

    overall = h.get("overall", "unknown")
    ts = h.get("timestamp", "")[:19].replace("T", " ")
    labels = {"ok": "🟢 ALL SYSTEMS OPERATIONAL",
              "degraded": "🟡 SOME ISSUES DETECTED",
              "down": "🔴 SYSTEM DOWN"}
    header = f"## {labels.get(overall, overall)}\nLast checked: `{ts} UTC`"

    api_h = h.get("api", {})
    api_md = f"**Status:** {api_h.get('status','—').upper()}\n\n"
    for c in api_h.get("checks", []):
        icon = "✅" if c.get("status") == "ok" else "❌"
        api_md += f"{icon} `{c.get('name')}` — {c.get('response_ms', 0)}ms\n"
    for e in api_h.get("errors", []):
        api_md += f"> ⚠️ {e}\n"

    scen = h.get("scenarios", {})
    scen_md = f"**Valid:** {'✅ Yes' if scen.get('valid') else '❌ No'}\n\n"
    for issue in scen.get("issues", [])[:10]:
        scen_md += f"- ⚠️ {issue}\n"
    if not scen.get("issues"):
        scen_md += "- No issues found\n"

    art = h.get("artifacts", {})
    art_md = (f"**Plots:** {'✅' if art.get('plots_ready') else '❌'}  "
              f"**Baselines real:** {'✅' if art.get('baselines_real') else '❌'}  "
              f"**Model published:** {'✅' if art.get('model_published') else '❌'}\n\n")
    for m in art.get("missing", []):
        art_md += f"- ❌ Missing: `{m}`\n"
    for w in art.get("warnings", []):
        art_md += f"- ⚠️ {w}\n"

    logs = h.get("logs", {})
    log_md = f"**Total errors:** {logs.get('error_count', 0)}\n\n"
    for k, v in logs.get("error_groups", {}).items():
        log_md += f"- `{k}`: {v}\n"
    for c in logs.get("critical", [])[:5]:
        log_md += f"\n```\n{c}\n```\n"

    return header, api_md, scen_md, art_md

# ── Build Blocks ──────────────────────────────────────────────────────────────

with gr.Blocks(title="ShiftLog-Gym Observatory") as demo:

    # Inject custom CSS into the page so styles apply even when the app is mounted
    # (HuggingFace Spaces mounts the Blocks app and does not call launch(),
    # so passing `css` to launch() may not be effective there).
    gr.HTML(f"""<style>{custom_css}</style>""")

    gr.HTML("""<div style="text-align:center;padding:28px 0 8px">
<h1 style="background:linear-gradient(90deg,#38bdf8,#818cf8,#c084fc);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  font-size:2.4rem;font-weight:800;margin:0">ShiftLog-Gym Observatory</h1>
<p style="color:#64748b;font-size:1rem;margin:6px 0 0">
  Meta PyTorch OpenEnv Hackathon · Grand Finale 2026</p></div>""")

    with gr.Tabs():

        # TAB 1 — Live Environment
        with gr.Tab("🔴 Live Environment"):
            with gr.Row():
                with gr.Column(scale=4):
                    gr.HTML("""<div style="background:rgba(17,24,39,.85);border:1px solid rgba(248,113,113,.3);
border-radius:12px;padding:20px;margin-bottom:12px">
<h3 style="color:#f87171;margin:0 0 8px">⚡ 3 AM. Payments down. 40k users affected.</h3>
<p style="color:#94a3b8;margin:0;line-height:1.7">Incident #7 is causally linked to a precursor from
the previous shift — but only if the agent <strong style="color:#38bdf8">reads the shift log first.</strong></p>
</div>""")
                    status_label = gr.Markdown(value="🟡 **Status:** Initializing...")
                    demo_btn = gr.Button("▶  Start Demo Episode", variant="primary")
                    next_btn = gr.Button("⏭  Next Step (1/5)", interactive=False)
                    step_state = gr.State(0)
                    incident_info = gr.Markdown("_Click **Start Demo Episode** to begin_")

                with gr.Column(scale=6):
                    chatbot = gr.Chatbot(label="Agent Action Log", height=440)

            demo_btn.click(fn=start_demo, outputs=[chatbot, incident_info]).then(
                fn=lambda: gr.update(interactive=True), outputs=[next_btn])
            next_btn.click(fn=next_step, inputs=[chatbot, step_state],
                           outputs=[chatbot, step_state, incident_info])

        # TAB 2 — Training Results
        with gr.Tab("📊 Training Results"):
            t2_banner = gr.Markdown(value="⌛ Loading results...")
            t2_cards  = gr.HTML(value="<div>Loading metrics...</div>")
            gr.HTML('''<div style="margin:8px 0">
<a href="https://wandb.ai/chiragaswal2/shiftlog-gym" target="_blank"
   style="color:#818cf8;text-decoration:none;font-weight:600">📊 View WandB Run →</a></div>''')
            gr.Markdown("---\n### Training Plots")
            img1 = gr.Image(value=load_plot("01_reward_curve.png"),
                            label="📈 Reward Curve — total reward over training steps")
            img2 = gr.Image(value=load_plot("02_recall_bonus_curve.png"),
                            label="🧠 R2 Recall Bonus — memory policy learning curve")
            img3 = gr.Image(value=load_plot("03_mttr_comparison.png"),
                            label="⏱ MTTR: Before vs After Training on Linked Incidents")
            ref_btn = gr.Button("🔄 Refresh Results")
            ref_btn.click(fn=refresh_results, outputs=[t2_banner, t2_cards, img1, img2, img3])

        # TAB 3 — Research Story
        with gr.Tab("🧠 Research Story"):
            gr.HTML(STORY_HTML)

        # TAB 4 — API Explorer
        with gr.Tab("⚙️ API Explorer"):
            gr.Markdown("### Live API Tester — no curl needed")
            with gr.Row():
                ep_dd    = gr.Dropdown(choices=["/reset","/step","/state","/tools"],
                                       value="/step", label="Endpoint", scale=3)
                resp_t   = gr.Textbox(label="Response Time", interactive=False, scale=1)
            body_in  = gr.Code(value=json.dumps(TOOL_EXAMPLES["/step"], indent=2),
                               language="json", label="Request Body")
            send_btn = gr.Button("📡 Send Request", variant="primary")
            resp_out = gr.Code(language="json", label="Response", interactive=False)
            ep_dd.change(fn=prefill_body, inputs=[ep_dd], outputs=[body_in])
            send_btn.click(fn=api_explore, inputs=[ep_dd, body_in],
                           outputs=[resp_out, resp_t])

        # TAB 5 — System Health
        with gr.Tab("🏥 System Health"):
            health_hdr = gr.Markdown("_Click **Run Health Check** to get current status_")
            hc_btn = gr.Button("🔄 Run Health Check Now", variant="primary")
            with gr.Accordion("API Health", open=True):
                api_md   = gr.Markdown()
            with gr.Accordion("Environment Validation", open=False):
                scen_md  = gr.Markdown()
            with gr.Accordion("Training Artifacts", open=False):
                art_md   = gr.Markdown()
            with gr.Accordion("Recent Errors", open=False):
                log_md   = gr.Markdown("_No log data yet._")

            def _hc():
                h, a, s, ar = run_health_check_ui()
                return h, a, s, ar, log_md.value if hasattr(log_md, "value") else ""

            hc_btn.click(fn=run_health_check_ui,
                         outputs=[health_hdr, api_md, scen_md, art_md])

        # TAB 6 — Training Lab (New)
        with gr.Tab("⚙️ Training"):
            gr.Markdown("""
            ## Run GRPO Training — HuggingFace Space GPU
            
            **Before clicking Start:** Confirm in Space Settings that:
            - Hardware is set to **L4 x1** (not CPU)
            - Secrets `HF_TOKEN` and `WANDB_API_KEY` are set
            - Variable `TRAIN_ENABLED` = `"1"`
            
            Training runs 3 stages: **A** (SFT, 50 steps) → **B** (GRPO, 200 steps) → **C** (GRPO, 300 steps)
            Estimated time: **3–4 hours** | Estimated GPU cost: **~$3–4** of your $30 credit
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    # Status card
                    status_display = gr.HTML(value=_render_training_status(get_state()))

                    # Start button — always interactive for better UX, we handle errors in the callback
                    start_btn = gr.Button(
                        "🚀 Start Training",
                        variant="primary",
                        size="lg",
                        interactive=True
                    )
                    env_warning = gr.Markdown(
                        "⚠️ **Requirement:** Set `TRAIN_ENABLED=1` in Space Settings to unlock training."
                        if os.environ.get("TRAIN_ENABLED", "0") != "1" else ""
                    )

                    # GPU info
                    gpu_info_box = gr.Textbox(
                        label="GPU Status",
                        value=_get_gpu_info(),
                        interactive=True,
                        lines=3,
                    )

                    repo_id_input = gr.Textbox(
                        label="Target HF Repository ID",
                        value="Chirag0123/shiftlog-gym-qwen-memory-policy",
                        placeholder="username/repo-name",
                        interactive=True
                    )

                with gr.Column(scale=2):
                    # Live log
                    log_display = gr.Textbox(
                        label="Training Log (live)",
                        value="Training not started.",
                        interactive=False,
                        lines=20,
                        max_lines=20,
                    )

            # Progress metrics row (now editable as requested)
            with gr.Row():
                metric_step = gr.Number(label="Steps Completed", value=0, interactive=True)
                metric_recall = gr.Number(label="Recall Rate (R2)", value=0.0, interactive=True)
                metric_reward = gr.Number(label="Total Reward", value=0.0, interactive=True)

            # WandB + Model links (now editable as requested)
            with gr.Row():
                wandb_link = gr.Textbox(label="WandB Run URL", value="", interactive=True)
                model_link = gr.Textbox(label="Trained Model URL", value="", interactive=True)

            # ─── Callbacks ──────────────────────────────────────────────────────────
            def handle_start_training(repo_id):
                if os.environ.get("TRAIN_ENABLED", "0") != "1":
                    return (
                        "<div style='color:#f87171; padding:10px; border:1px solid #f87171; border-radius:8px;'>❌ <b>Training Blocked:</b> Set <code>TRAIN_ENABLED=1</code> in Space Settings → Variables & Secrets, then restart the Space.</div>",
                        "ERROR: Environment variable TRAIN_ENABLED is not set to '1'.",
                        0, 0.0, 0.0, "", ""
                    )

                # Update target repo in daemon if provided (handles full URLs gracefully)
                import train.space_training_daemon as daemon
                if repo_id:
                    clean_id = repo_id.strip().split("huggingface.co/")[-1].split("?")[0].strip("/")
                    daemon.MODEL_REPO = clean_id

                state = get_state()
                if state["status"] == "running":
                    return (
                        _render_training_status(state),
                        "\n".join(state["log_lines"]),
                        state["step"], state["recall_rate"], state["reward_total"],
                        state["wandb_url"], state["model_url"],
                    )
                
                # Start in background
                t = threading.Thread(target=run_training, daemon=True)
                t.start()
                
                # Wait briefly for init
                time.sleep(1.0)
                state = get_state()
                return (
                    _render_training_status(state),
                    "\n".join(state["log_lines"]),
                    state["step"], state["recall_rate"], state["reward_total"],
                    state["wandb_url"], state["model_url"],
                )

            def poll_training_state():
                state = get_state()
                return (
                    _render_training_status(state),
                    "\n".join(state["log_lines"]),
                    state["step"], state["recall_rate"], state["reward_total"],
                    state["wandb_url"], state["model_url"],
                )

            start_btn.click(
                fn=handle_start_training,
                inputs=[repo_id_input],
                outputs=[status_display, log_display, metric_step, metric_recall, metric_reward, wandb_link, model_link],
            )

            # Auto-refresh every 15 seconds while training is running
            refresh_timer = gr.Timer(value=15)
            refresh_timer.tick(
                fn=poll_training_state,
                inputs=[],
                outputs=[status_display, log_display, metric_step, metric_recall, metric_reward, wandb_link, model_link],
            )

        # TAB 7 — User Guide & Documentation
        with gr.Tab("📖 Guide"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### 🚀 Getting Started
                    1. **Environment**: Switch to **L4 GPU** in Settings.
                    2. **Auth**: Add `HF_TOKEN` and `WANDB_API_KEY` to Secrets.
                    3. **Unlock**: Add `TRAIN_ENABLED=1` to Variables.
                    4. **Run**: Go to the **Training** tab and click Start.
                    
                    ### 🔍 How it Works
                    - **Stage A (Warmup)**: Teaches the model the basic JSON tool format via SFT.
                    - **Stage B (Memory Rollout)**: Uses GRPO to optimize causal retrieval on simple incidents.
                    - **Stage C (Full Shift)**: Scales GRPO to all 6 incident families and 8-hour shifts.
                    """)
                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### 📊 Metrics Explained
                    - **Recall Rate (R2)**: The probability that the agent reads the shift log *before* taking a destructive action on a linked incident.
                    - **MTTR (Steps)**: Average number of tool calls to reach "Resolution".
                    - **Total Reward**: A weighted sum of success, efficiency, and memory integrity.
                    
                    ### 📦 Exporting Results
                    Once complete, the model adapter is pushed to your HF Hub repo, and the evidence plots are committed back to this Space.
                    """)

    # Footer
    gr.HTML("""<div style="text-align:center;padding:20px;color:#64748b;font-size:0.85rem;border-top:1px solid rgba(255,255,255,0.05)">
      ShiftLog-Gym &copy; 2026 · Built for causal memory benchmarking
    </div>""")

    # --- Initialization ---
    # Populate dynamic fields on load instead of using value=fn to avoid reactive loops
    demo.load(fn=system_status_md, outputs=[status_label])
    demo.load(fn=refresh_results, outputs=[t2_banner, t2_cards, img1, img2, img3])
    # Sync training UI on page load/refresh
    demo.load(fn=poll_training_state, outputs=[status_display, log_display, metric_step, metric_recall, metric_reward, wandb_link, model_link])
