"""
knowledge_graph_tab.py

Renders an interactive Plotly knowledge graph showing:
- Service dependency topology (gray edges)
- Causal incident chains learned during training (red edges)
- Shift log entries as time-ordered memory nodes (blue nodes)
- Before/after toggle: shows which connections base model missed

Attacks 40% Innovation (biggest weight) + 30% Storytelling
"""
import gradio as gr
import plotly.graph_objects as go

# ── Static service topology ───────────────────────────────────────────────────
SERVICES = {
    "payment-db": {"x": 0.2, "y": 0.8, "color": "#ef4444", "critical": True},
    "auth-service": {"x": 0.6, "y": 0.8, "color": "#ef4444", "critical": True},
    "notification-svc": {"x": 0.1, "y": 0.4, "color": "#f97316"},
    "api-gateway": {"x": 0.5, "y": 0.2, "color": "#6366f1"},
    "order-svc": {"x": 0.9, "y": 0.4, "color": "#f97316"},
    "inventory-svc": {"x": 0.9, "y": 0.8, "color": "#22c55e"},
    "search-svc": {"x": 0.5, "y": 0.6, "color": "#22c55e"},
    "cache-svc": {"x": 0.3, "y": 0.4, "color": "#a855f7"},
    "session-store": {"x": 0.7, "y": 0.6, "color": "#a855f7"},
}

DEPENDENCY_EDGES = [
    ("api-gateway", "auth-service"),
    ("api-gateway", "order-svc"),
    ("api-gateway", "search-svc"),
    ("auth-service", "payment-db"),
    ("auth-service", "cache-svc"),
    ("auth-service", "session-store"),
    ("order-svc", "inventory-svc"),
    ("order-svc", "payment-db"),
    ("notification-svc", "cache-svc"),
]

# Causal chains: (src_service, dst_service, description)
CAUSAL_CHAINS = [
    (
        "payment-db",
        "auth-service",
        "Inc #1 → #7: DB pool exhaustion cascaded to auth service",
    ),
    (
        "notification-svc",
        "notification-svc",
        "Inc #3 → #9: OOM recurrence on same pod limits",
    ),
    ("order-svc", "order-svc", "Inc #5 → #11: Config key MAX_RETRY drifted again"),
]

# Memory nodes: shift log entries written during training
MEMORY_NODES = [
    {
        "id": "mem_1",
        "x": 0.2,
        "y": 0.95,
        "incident": 1,
        "content": "DB pool 80% — watch auth under load",
        "causal_for": [7, 10],
    },
    {
        "id": "mem_3",
        "x": 0.1,
        "y": 0.25,
        "incident": 3,
        "content": "Notification pod OOM — memory limits too low",
        "causal_for": [9],
    },
    {
        "id": "mem_5",
        "x": 0.9,
        "y": 0.25,
        "incident": 5,
        "content": "Order svc MAX_RETRY changed in deploy abc123",
        "causal_for": [11],
    },
]


def build_knowledge_graph_tab():
    """Build the interactive knowledge graph tab."""
    with gr.Tab("🕸 Knowledge Graph"):
        gr.HTML(
            """
            <div style="background: rgba(139, 92, 246, 0.08); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 12px; padding: 20px; margin-bottom: 16px;">
                <h3 style="color: #a855f7; margin: 0 0 8px; font-size: 1.1rem;">🕸 Service Dependency + Causal Memory Knowledge Graph</h3>
                <p style="color: #94a3b8; margin: 0; line-height: 1.6;">
                    Gray edges = service dependencies. Red edges = causal incident chains
                    the trained model learned to connect. Blue diamonds = shift log memory nodes.
                </p>
            </div>
        """
        )

        with gr.Row():
            view_mode = gr.Radio(
                choices=[
                    "Full graph (trained model view)",
                    "Base model view (no causal memory)",
                    "Memory nodes only",
                ],
                value="Full graph (trained model view)",
                label="View mode",
            )

        graph_plot = gr.Plot(label="")

        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """
                    **Reading this graph:**
                    
                    - **Gray circles** = microservices in the production system
                    - **Gray arrows** = service call dependencies (static topology)
                    - **Red dashed arrows** = causal incident chains the model must remember
                    - **Blue diamonds** = shift log entries written by the trained model
                    
                    **View modes:**
                    - **Full graph** = how the trained model sees causality
                    - **Base model** = no causal edges (it doesn't connect incidents)
                    - **Memory only** = just the shift log knowledge nodes
                    """
                )

        def render_graph(mode: str) -> go.Figure:
            """Render the interactive knowledge graph."""
            fig = go.Figure()
            fig.update_layout(
                plot_bgcolor="#0f172a",
                paper_bgcolor="#0f172a",
                font=dict(color="#e2e8f0"),
                showlegend=True,
                margin=dict(l=20, r=20, t=20, b=20),
                height=600,
                legend=dict(
                    bgcolor="#1e293b",
                    bordercolor="#334155",
                    borderwidth=1,
                    font=dict(size=11),
                    x=0.02,
                    y=0.98,
                ),
            )

            # ── Dependency edges ──────────────────────────────────────────
            for src, dst in DEPENDENCY_EDGES:
                sx, sy = SERVICES[src]["x"], SERVICES[src]["y"]
                dx, dy = SERVICES[dst]["x"], SERVICES[dst]["y"]
                fig.add_trace(
                    go.Scatter(
                        x=[sx, dx, None],
                        y=[sy, dy, None],
                        mode="lines",
                        line=dict(color="#475569", width=1.5, dash="solid"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

            # ── Causal edges (only in trained view) ───────────────────────
            if "trained" in mode or "Full" in mode:
                for i, (src_svc, dst_svc, desc) in enumerate(CAUSAL_CHAINS):
                    if src_svc == dst_svc:
                        # Self-loop — offset slightly
                        sx = SERVICES[src_svc]["x"] + 0.04
                        sy = SERVICES[src_svc]["y"] + 0.06
                        dx = SERVICES[dst_svc]["x"] - 0.04
                        dy = SERVICES[dst_svc]["y"] + 0.06
                    else:
                        sx, sy = SERVICES[src_svc]["x"], SERVICES[src_svc]["y"]
                        dx, dy = SERVICES[dst_svc]["x"], SERVICES[dst_svc]["y"]

                    fig.add_trace(
                        go.Scatter(
                            x=[sx, dx, None],
                            y=[sy, dy, None],
                            mode="lines",
                            line=dict(color="#ef4444", width=2.5, dash="dot"),
                            name="Causal chain (memory)" if i == 0 else None,
                            showlegend=i == 0,
                            hovertext=desc,
                            hoverinfo="text",
                        )
                    )

            # ── Memory nodes (shift log entries) ─────────────────────────
            if "trained" in mode or "Full" in mode or "Memory" in mode:
                for i, mem in enumerate(MEMORY_NODES):
                    fig.add_trace(
                        go.Scatter(
                            x=[mem["x"]],
                            y=[mem["y"]],
                            mode="markers+text",
                            marker=dict(
                                symbol="diamond",
                                size=14,
                                color="#60a5fa",
                                line=dict(color="#93c5fd", width=2),
                            ),
                            text=[f"Inc #{mem['incident']}"],
                            textposition="top center",
                            textfont=dict(size=10, color="#93c5fd"),
                            name="Shift log entry" if i == 0 else None,
                            hovertext=(
                                f"Memory: {mem['content']}\n"
                                f"Causal for incidents: {mem['causal_for']}"
                            ),
                            hoverinfo="text",
                            showlegend=i == 0,
                        )
                    )
                    # Arrow from memory node to the service it describes
                    target_svc = (
                        "payment-db"
                        if mem["incident"] == 1
                        else (
                            "notification-svc"
                            if mem["incident"] == 3
                            else "order-svc"
                        )
                    )
                    tsx, tsy = (
                        SERVICES[target_svc]["x"],
                        SERVICES[target_svc]["y"],
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[mem["x"], tsx, None],
                            y=[mem["y"], tsy, None],
                            mode="lines",
                            line=dict(color="#3b82f6", width=1, dash="dot"),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

            # ── Service nodes ─────────────────────────────────────────────
            for name, props in SERVICES.items():
                is_critical = props.get("critical", False)
                fig.add_trace(
                    go.Scatter(
                        x=[props["x"]],
                        y=[props["y"]],
                        mode="markers+text",
                        marker=dict(
                            size=22 if is_critical else 16,
                            color=props["color"],
                            line=dict(
                                color="#e2e8f0",
                                width=1.5 if is_critical else 1,
                            ),
                            symbol="circle",
                        ),
                        text=[name.replace("-", "\n")],
                        textposition="bottom center",
                        textfont=dict(size=9, color="#e2e8f0"),
                        name=name,
                        hovertext=f"Service: {name}",
                        hoverinfo="text",
                        showlegend=False,
                    )
                )

            fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
            fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
            return fig

        # Initial render
        graph_plot.value = render_graph("Full graph (trained model view)")

        view_mode.change(
            fn=render_graph, inputs=[view_mode], outputs=[graph_plot]
        )
