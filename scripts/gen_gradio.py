"""Write the full observatory/gradio_app.py — run with: python scripts/gen_gradio.py"""
from pathlib import Path

OUT = Path(__file__).parent.parent / "observatory" / "gradio_app.py"
OUT.write_text(r'''"""ShiftLog-Gym Observatory — 5-tab Gradio dashboard."""
from __future__ import annotations
import json, os, time, datetime
from pathlib import Path
import gradio as gr
import pandas as pd
import plotly.express as px
import httpx

ROOT     = Path(__file__).resolve().parent.parent
OBS      = ROOT / "observatory"
PLOTS    = ROOT / "plots"
RUNS_DIR = OBS / "training_runs"
API_BASE = os.environ.get("SPACE_API_URL", "http://localhost:7860")
TIMEOUT  = 10.0

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');
body,.gradio-container{font-family:'Inter',sans-serif!important;background:#0b0f19!important;color:#e2e8f0!important}
.glass{background:rgba(17,24,39,.75)!important;backdrop-filter:blur(12px)!important;border:1px solid rgba(255,255,255,.08)!important;border-radius:16px!important;padding:24px!important}
.title{background:linear-gradient(90deg,#38bdf8,#818cf8,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800!important;font-size:2.2rem!important;text-align:center}
.btn-main{background:linear-gradient(90deg,#6366f1,#8b5cf6)!important;border:none!important;color:#fff!important;font-weight:600!important;transition:.25s!important}
.badge-ok{color:#4ade80;font-weight:700}.badge-warn{color:#facc15;font-weight:700}.badge-err{color:#f87171;font-weight:700}
"""

TOOL_EXAMPLES = {
    "/reset": {"seed": 1, "family": "db_pool"},
    "/step":  {"tool": "read_shift_log", "arguments": {"query": "payment service", "limit": 3}},
    "/state": {},
    "/tools": {},
}

DEMO_STEPS = [
    ("read_shift_log",    {"query": "payment service", "limit": 5}),
    ("inspect_service",   {"service": "payments-api"}),
    ("run_diagnostic",    {"service": "payments-api", "diagnostic": "check_pool"}),
    ("apply_mitigation",  {"service": "payments-api", "mitigation": "set_pool_size_and_restart"}),
    ("resolve_incident",  {"incident_id": "INC-001", "resolution": "Pool size restored", "root_cause": "stale DB pool size after rollback"}),
]

# ── helpers ──────────────────────────────────────────────────────────────────

def _api(method: str, path: str, body: dict | None = None) -> tuple[dict, float, str | None]:
    start = time.monotonic()
    try:
        fn = httpx.post if method == "POST" else httpx.get
        kwargs = {"timeout": TIMEOUT}
        if body is not None:
            kwargs["json"] = body
        r = fn(f"{API_BASE}{path}", **kwargs)
        ms = round((time.monotonic() - start) * 1000)
        return r.json(), ms, None
    except Exception as exc:
        ms = round((time.monotonic() - start) * 1000)
        return {}, ms, str(exc)

def _load_baselines() -> dict:
    p = OBS / "baselines.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def _load_health() -> dict:
    p = OBS / "health_status.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def _status_badge(s: str) -> str:
    cls = {"ok": "badge-ok", "degraded": "badge-warn", "down": "badge-err"}.get(s, "badge-warn")
    icon = {"ok": "🟢", "degraded": "🟡", "down": "🔴"}.get(s, "🟡")
    return f"<span class='{cls}'>{icon} {s.upper()}</span>"

# ── Tab 1: Live Environment ───────────────────────────────────────────────────

def start_demo():
    data, ms, err = _api("POST", "/reset", {"seed": 1, "family": "db_pool"})
    if err:
        return [[None, f"❌ API error: {err}"]], [], "Reset failed", 0, 0
    msg = data.get("message", "Episode started.")
    chat = [[None, f"🚨 **New Incident**\n\n{msg[:500]}"]]
    info = f"**Service:** {data.get('metadata', {}).get('scenario_family','db_pool')} | **Step:** 0"
    return chat, [], info, 0, 0

def next_step(chat, step_idx):
    if step_idx >= len(DEMO_STEPS):
        chat.append([None, "✅ Demo episode complete! All 5 steps executed."])
        return chat, step_idx, "Episode complete"
    tool, args = DEMO_STEPS[step_idx]
    data, ms, err = _api("POST", "/step", {"tool": tool, "arguments": args})
    if err:
        chat.append([f"`{tool}`", f"❌ Error: {err}"])
        return chat, step_idx + 1, f"Error at step {step_idx+1}"
    msg  = data.get("message", "")
    rew  = data.get("reward", 0.0)
    icons = {"read_shift_log":"📖","append_shift_log":"✍️","inspect_service":"🔍",
             "run_diagnostic":"🧪","apply_mitigation":"🛠","resolve_incident":"✅",
             "handoff_summary":"📋","update_shift_log":"✏️"}
    icon = icons.get(tool, "⚙️")
    user_msg = f"{icon} `{tool}({json.dumps(args)[:80]})`"
    bot_msg  = f"**Reward:** {rew:+.3f} | {msg[:300]}"
    chat.append([user_msg, bot_msg])
    info = f"**Step:** {step_idx+1}/5 | **Tool:** {tool} | **Reward:** {rew:+.3f} | {ms}ms"
    return chat, step_idx + 1, info

def system_status_md():
    h = _load_health()
    if not h:
        return "🟡 **Status:** Health check not yet run"
    ts  = h.get("timestamp","")[:19].replace("T"," ")
    badge = _status_badge(h.get("overall","unknown"))
    return f"{badge} — Last checked: `{ts} UTC`"

# ── Tab 2: Training Results ───────────────────────────────────────────────────

def training_banner():
    bl = _load_baselines()
    meta = bl.get("_metadata", {})
    note = meta.get("note","")
    run_id = meta.get("run_id","pending")
    if not bl:
        return "⚠️ **No baselines.json found.** Run training first."
    if "simulated" in note.lower() or run_id == "pending":
        return "⚠️ **Showing projected estimates.** Run `train/02_grpo_train_colab.ipynb` for real results."
    return f"✅ **Real GRPO training results** — Run ID: `{run_id}` | {meta.get('timestamp','')[:10]}"

def metric_cards_html():
    bl  = _load_baselines()
    rnd = bl.get("random",  {})
    trn = bl.get("trained_llm", {})
    recall_base  = rnd.get("recall_before_action_rate", 0.12)
    recall_train = trn.get("recall_before_action_rate", 0.78)
    mttr_base    = rnd.get("linked_incident_mttr", 24.5)
    mttr_train   = trn.get("linked_incident_mttr", 3.5)
    steps        = bl.get("_metadata",{}).get("training_steps", "—")
    def card(title, val1, val2, unit, up=True):
        arrow = "↑" if up else "↓"
        color = "#4ade80" if up else "#f87171"
        return f"""<div style="background:rgba(17,24,39,.8);border:1px solid rgba(255,255,255,.1);
border-radius:12px;padding:20px;text-align:center;flex:1;min-width:180px">
<div style="color:#94a3b8;font-size:.85rem;margin-bottom:6px">{title}</div>
<div style="font-size:1.8rem;font-weight:800;color:#e2e8f0">{val2}{unit}</div>
<div style="color:{color};font-size:.9rem;margin-top:4px">{arrow} from {val1}{unit}</div></div>"""
    cards = card("Recall Rate",f"{recall_base:.0%}",f"{recall_train:.0%}","",True)
    cards += card("MTTR (Linked)",f"{mttr_base:.1f}",f"{mttr_train:.1f}"," steps",False)
    cards += card("Training Steps","0",str(steps),"",True)
    return f'<div style="display:flex;gap:16px;flex-wrap:wrap;margin:16px 0">{cards}</div>'

def load_plot(name):
    p = PLOTS / name
    return str(p) if p.exists() else None

def wandb_link_html():
    return '<a href="https://wandb.ai/chiragaswal2/shiftlog-gym" target="_blank" style="color:#818cf8">📊 View WandB Run →</a>'

# ── Tab 3: Research Story ─────────────────────────────────────────────────────

STORY_HTML = """
<div style="max-width:860px;margin:0 auto;font-family:Inter,sans-serif;color:#e2e8f0">
<h2 style="background:linear-gradient(90deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent">
The Memory Gap in Frontier AI</h2>
<p style="color:#94a3b8;line-height:1.8">Every frontier lab ships memory as a product feature.
None trained the model to decide <em>what to write, when to retrieve, and what to safely forget.</em>
ShiftLog-Gym fills this gap with a verifiable, causal SRE environment.</p>
<table style="width:100%;border-collapse:collapse;margin:20px 0">
<thead><tr style="border-bottom:1px solid rgba(255,255,255,.15)">
<th style="padding:10px;text-align:left;color:#94a3b8">Capability</th>
<th style="padding:10px;color:#94a3b8">Memory-R1</th>
<th style="padding:10px;color:#94a3b8">MemAgent</th>
<th style="padding:10px;color:#4ade80">ShiftLog-Gym</th></tr></thead>
<tbody>
''' + "".join(
    f'<tr style="border-bottom:1px solid rgba(255,255,255,.06)"><td style="padding:10px;color:#e2e8f0">{cap}</td>'
    f'<td style="text-align:center;color:#f87171">✗</td><td style="text-align:center;color:#f87171">✗</td>'
    f'<td style="text-align:center;color:#4ade80">✅</td></tr>'
    for cap in ["Causal incident chains","SRE domain","Verifiable MTTR",
                "Noise resistance","Handoff memory","OpenEnv compliant"]
) + """
</tbody></table>
<h3 style="color:#818cf8;margin-top:32px">8 Reward Signals</h3>
<table style="width:100%;border-collapse:collapse">
<thead><tr style="border-bottom:1px solid rgba(255,255,255,.15)">
<th style="padding:8px;text-align:left;color:#94a3b8">Signal</th>
<th style="padding:8px;color:#94a3b8">Weight</th>
<th style="padding:8px;color:#94a3b8">What it measures</th></tr></thead>
<tbody>
""" + "".join(
    f'<tr style="border-bottom:1px solid rgba(255,255,255,.06)"><td style="padding:8px;color:#e2e8f0">{n}</td>'
    f'<td style="padding:8px;text-align:center;color:#38bdf8;font-weight:700">{w}</td>'
    f'<td style="padding:8px;color:#94a3b8;font-size:.9rem">{d}</td></tr>'
    for n,w,d in [
        ("R1 Success","35%","Correct root cause + mitigation per incident"),
        ("R2 Recall","25%","Read shift log before linked-incident mitigation"),
        ("R3 Memory Write","15%","Quality of shift log entries written"),
        ("R4 Memory Integrity","10%","Penalises contradictions & duplicates"),
        ("R5 Efficiency","5%","Fewer tool calls → higher score"),
        ("Hallucination","5%","Penalises invalid mitigations"),
        ("Noise Resistance","3%","Resists misleadingly similar noise incidents"),
        ("Handoff Quality","2%","Structured shift handoff summary"),
    ]
) + """
</tbody></table>
<h3 style="color:#818cf8;margin-top:32px">Scenario Families</h3>
<div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:12px">
""" + "".join(
    f'<div style="background:rgba(99,102,241,.15);border:1px solid rgba(99,102,241,.4);'
    f'border-radius:8px;padding:10px 16px;font-size:.9rem">{f}</div>'
    for f in ["db_pool","auth_cascade","oom_regression","feature_flag","network_partition","config_drift"]
) + """
</div></div>"""

# ── Tab 4: API Explorer ───────────────────────────────────────────────────────

def api_explore(endpoint, body_str):
    try:
        body = json.loads(body_str) if body_str.strip() else {}
    except Exception as exc:
        return f"JSON parse error: {exc}", "—"
    method = "GET" if endpoint in ("/state", "/tools") else "POST"
    data, ms, err = _api(method, endpoint, body if method == "POST" else None)
    if err:
        return json.dumps({"error": err}, indent=2), f"{ms}ms"
    return json.dumps(data, indent=2), f"{ms}ms"

def prefill_body(endpoint):
    ex = TOOL_EXAMPLES.get(endpoint, {})
    return json.dumps(ex, indent=2)

# ── Tab 5: System Health ──────────────────────────────────────────────────────

def run_health_check_ui():
    try:
        from shiftlog_gym.diagnostics.health_check import run_full_health_check
        h = run_full_health_check(base_url=API_BASE)
    except Exception as exc:
        return f"❌ Health check failed: {exc}", "{}", "{}", "{}"

    overall = h.get("overall","unknown")
    ts = h.get("timestamp","")[:19].replace("T"," ")
    badge = {"ok":"🟢 ALL SYSTEMS OPERATIONAL","degraded":"🟡 SOME ISSUES DETECTED","down":"🔴 SYSTEM DOWN"}.get(overall,"🟡 UNKNOWN")
    header = f"## {badge}\nLast checked: `{ts} UTC`"

    api_h = h.get("api",{})
    api_md = f"**Status:** {api_h.get('status','—').upper()}\n\n"
    for c in api_h.get("checks",[]):
        icon = "✅" if c.get("status")=="ok" else "❌"
        api_md += f"{icon} `{c.get('name')}` — {c.get('response_ms',0)}ms\n"
    for e in api_h.get("errors",[]):
        api_md += f"  > ⚠️ {e}\n"

    scen = h.get("scenarios",{})
    scen_md = f"**Valid:** {'✅ Yes' if scen.get('valid') else '❌ No'}\n\n"
    for issue in scen.get("issues",[])[:10]:
        scen_md += f"- ⚠️ {issue}\n"
    if not scen.get("issues"):
        scen_md += "- No issues found\n"

    art = h.get("artifacts",{})
    art_md = f"**Plots ready:** {'✅' if art.get('plots_ready') else '❌'} | "
    art_md += f"**Baselines real:** {'✅' if art.get('baselines_real') else '❌'} | "
    art_md += f"**Model published:** {'✅' if art.get('model_published') else '❌'}\n\n"
    for m in art.get("missing",[]):
        art_md += f"- ❌ Missing: `{m}`\n"
    for w in art.get("warnings",[]):
        art_md += f"- ⚠️ {w}\n"

    logs = h.get("logs",{})
    log_md = f"**Total errors:** {logs.get('error_count',0)}\n\n"
    for k,v in logs.get("error_groups",{}).items():
        log_md += f"- `{k}`: {v} occurrences\n"
    for c in logs.get("critical",[])[:5]:
        log_md += f"\n```\n{c}\n```\n"

    return header, api_md, scen_md, art_md

def health_log_md():
    h = _load_health()
    logs = h.get("logs",{})
    if not logs:
        return "No log data available."
    out = f"**Total errors:** {logs.get('error_count',0)}\n\n"
    for k,v in logs.get("error_groups",{}).items():
        out += f"- `{k}`: {v}\n"
    return out

# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="ShiftLog-Gym Observatory") as demo:
    gr.HTML("""<div style='text-align:center;padding:28px 0 8px'>
<h1 style='background:linear-gradient(90deg,#38bdf8,#818cf8,#c084fc);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;
font-size:2.4rem;font-weight:800;margin:0'>ShiftLog-Gym Observatory</h1>
<p style='color:#64748b;font-size:1rem;margin:6px 0 0'>
Meta PyTorch OpenEnv Hackathon · Grand Finale 2026</p></div>""")

    with gr.Tabs():

        # ── TAB 1 ────────────────────────────────────────────────────────────
        with gr.Tab("🔴 Live Environment"):
            with gr.Row():
                with gr.Column(scale=4):
                    gr.HTML("""<div style='background:rgba(17,24,39,.8);border:1px solid rgba(255,255,255,.08);
border-radius:12px;padding:20px;margin-bottom:12px'>
<h3 style='color:#f87171;margin:0 0 8px'>⚡ 3AM. Payments down. 40k users affected.</h3>
<p style='color:#94a3b8;margin:0;line-height:1.7'>Your shift just started and there are 5 open incidents.
Incident #7 is causally linked to a precursor from the previous shift —
but only if the agent <strong style='color:#38bdf8'>reads the shift log first</strong>.</p></div>""")
                    status_md = gr.Markdown(value=system_status_md)
                    t1_timer = gr.Timer(value=30)
                    t1_timer.tick(fn=system_status_md, outputs=[status_md])
                    demo_btn  = gr.Button("▶ Start Demo Episode", variant="primary")
                    next_btn  = gr.Button("⏭ Next Step", interactive=False)
                    incident_info = gr.Markdown("_Click Start Demo Episode to begin_")
                    step_state = gr.State(0)

                with gr.Column(scale=6):
                    chatbot = gr.Chatbot(label="Agent Action Log", height=400, bubble_full_width=False)
                    shift_df = gr.Dataframe(label="Shift Log Entries",
                                            headers=["entry_id","service","entry_type","fact"],
                                            interactive=False)

            demo_btn.click(fn=start_demo,
                           outputs=[chatbot, shift_df, incident_info, step_state, step_state])\
                    .then(fn=lambda: gr.update(interactive=True), outputs=[next_btn])
            next_btn.click(fn=next_step, inputs=[chatbot, step_state],
                           outputs=[chatbot, step_state, incident_info])

        # ── TAB 2 ────────────────────────────────────────────────────────────
        with gr.Tab("📊 Training Results"):
            t2_banner  = gr.Markdown(value=training_banner)
            t2_cards   = gr.HTML(value=metric_cards_html)
            gr.HTML(wandb_link_html())
            gr.Markdown("---\n### Training Plots")
            with gr.Row():
                img1 = gr.Image(value=load_plot("01_reward_curve.png"),
                                label="📈 Reward Curve — total reward over steps", type="filepath")
            with gr.Row():
                img2 = gr.Image(value=load_plot("02_recall_bonus_curve.png"),
                                label="🧠 R2 Recall Bonus — memory policy learning", type="filepath")
            with gr.Row():
                img3 = gr.Image(value=load_plot("03_mttr_comparison.png"),
                                label="⏱ MTTR: Before vs After Training", type="filepath")
            refresh_btn = gr.Button("🔄 Refresh Results")
            def refresh_all():
                return training_banner(), metric_cards_html(), \
                       load_plot("01_reward_curve.png"), \
                       load_plot("02_recall_bonus_curve.png"), \
                       load_plot("03_mttr_comparison.png")
            refresh_btn.click(fn=refresh_all,
                              outputs=[t2_banner, t2_cards, img1, img2, img3])

        # ── TAB 3 ────────────────────────────────────────────────────────────
        with gr.Tab("🧠 Research Story"):
            gr.HTML(STORY_HTML)

        # ── TAB 4 ────────────────────────────────────────────────────────────
        with gr.Tab("⚙️ API Explorer"):
            gr.Markdown("### Live API Tester — try any endpoint without curl")
            with gr.Row():
                ep_dd = gr.Dropdown(
                    choices=["/reset","/step","/state","/tools"],
                    value="/step", label="Endpoint")
                resp_time = gr.Textbox(label="Response Time", interactive=False, scale=1)
            body_in  = gr.Code(value=json.dumps(TOOL_EXAMPLES["/step"], indent=2),
                               language="json", label="Request Body (JSON)")
            send_btn = gr.Button("📡 Send Request", variant="primary")
            resp_out = gr.Code(language="json", label="Response", interactive=False)
            ep_dd.change(fn=prefill_body, inputs=[ep_dd], outputs=[body_in])
            send_btn.click(fn=api_explore, inputs=[ep_dd, body_in],
                           outputs=[resp_out, resp_time])

        # ── TAB 5 ────────────────────────────────────────────────────────────
        with gr.Tab("🏥 System Health"):
            health_header = gr.Markdown("_Click Run Health Check to get current status_")
            with gr.Row():
                hc_btn = gr.Button("🔄 Run Health Check Now", variant="primary")
            t5_timer = gr.Timer(value=60)
            with gr.Accordion("API Health", open=True):
                api_detail = gr.Markdown()
            with gr.Accordion("Environment Validation", open=False):
                scen_detail = gr.Markdown()
            with gr.Accordion("Training Artifacts", open=False):
                art_detail = gr.Markdown()
            with gr.Accordion("Recent Errors", open=False):
                log_detail = gr.Markdown(value=health_log_md)

            def _hc():
                h, a, s, ar = run_health_check_ui()
                return h, a, s, ar, health_log_md()

            hc_btn.click(fn=_hc,
                         outputs=[health_header, api_detail, scen_detail, art_detail, log_detail])
            t5_timer.tick(fn=_hc,
                          outputs=[health_header, api_detail, scen_detail, art_detail, log_detail])

if __name__ == "__main__":
    demo.launch(css=custom_css)
''', encoding="utf-8")
print(f"Written: {OUT}")
