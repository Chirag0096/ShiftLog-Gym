"""ShiftLog-Gym Observatory — 5-tab Gradio dashboard."""
from __future__ import annotations
import json, os, time
from pathlib import Path
import gradio as gr
import pandas as pd
import httpx

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
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap");
body,.gradio-container{font-family:"Inter",sans-serif!important;background:#0b0f19!important;color:#e2e8f0!important}
.glass{background:rgba(17,24,39,.75)!important;backdrop-filter:blur(12px)!important;
  border:1px solid rgba(255,255,255,.08)!important;border-radius:16px!important;padding:24px!important}
.btn-main{background:linear-gradient(90deg,#6366f1,#8b5cf6)!important;color:#fff!important;font-weight:600!important}
/* Keep tab navigation above content overlays in Gradio 5/6 mounted mode */
.gradio-container [role="tablist"]{position:relative;z-index:50;pointer-events:auto!important}
.gradio-container [role="tab"]{pointer-events:auto!important}
.gradio-container [role="tabpanel"]{position:relative;z-index:1}
"""

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
                    status_label = gr.Markdown(value=system_status_md)
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
            t2_banner = gr.Markdown(value=training_banner)
            t2_cards  = gr.HTML(value=metric_cards_html)
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


    # Footer
    gr.HTML("""<div style="text-align:center;padding:20px;color:#64748b;font-size:0.85rem;border-top:1px solid rgba(255,255,255,0.05)">
      ShiftLog-Gym &copy; 2026 · Built for causal memory benchmarking
    </div>""")
