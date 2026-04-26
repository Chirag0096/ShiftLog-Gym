"""
ShiftLog-Gym Automated Health Check System.

Runs on Space startup, surfaces all errors to the Gradio dashboard
via observatory/health_status.json.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Project root is 2 levels up from this file (shiftlog_gym/diagnostics/)
ROOT = Path(__file__).resolve().parent.parent.parent
OBS_ROOT = ROOT / "observatory"
HEALTH_FILE = OBS_ROOT / "health_status.json"
PLOTS_DIR = ROOT / "plots"
BASELINES_FILE = OBS_ROOT / "baselines.json"
HF_MODEL_URL = "https://huggingface.co/Chirag0123/shiftlog-gym-qwen-memory-policy"

VALID_TOOLS = [
    "read_shift_log", "append_shift_log", "update_shift_log",
    "inspect_service", "inspect_dependency", "run_diagnostic",
    "apply_mitigation", "resolve_incident", "handoff_summary",
]


# ---------------------------------------------------------------------------
# 1A — API Health Check
# ---------------------------------------------------------------------------

def check_api_health(base_url: str = "http://localhost:7860", 
                     call_direct: dict[str, Callable] | None = None) -> dict[str, Any]:
    """
    Check all 4 API endpoints: /reset, /step, /state, /tools.

    Args:
        base_url: The URL to check (if not call_direct)
        call_direct: Optional map of paths to local functions to avoid HTTP calls.

    Returns:
        dict with keys: status ("ok"|"degraded"|"down"), checks (list), errors (list)
    """
    if not call_direct:
        try:
            import httpx
        except ImportError:
            return _check_api_health_stdlib(base_url)

    checks: list[dict] = []
    errors: list[str] = []

    def _check(name: str, method: str, path: str, body: Any = None,
                expect_keys: list[str] | None = None) -> bool:
        start = time.monotonic()
        try:
            # Use direct call if available, otherwise use httpx
            if call_direct and path in call_direct:
                data = call_direct[path](body) if method == "POST" else call_direct[path]()
                status_code = 200
                ok = True
            else:
                import httpx
                if method == "POST":
                    r = httpx.post(f"{base_url}{path}", json=body or {}, timeout=10.0)
                else:
                    r = httpx.get(f"{base_url}{path}", timeout=10.0)
                status_code = r.status_code
                ok = (status_code == 200)
                data = r.json() if ok else {}

            elapsed_ms = round((time.monotonic() - start) * 1000)
            key_ok = True
            if expect_keys and ok:
                for k in expect_keys:
                    if k not in data:
                        key_ok = False
                        errors.append(f"{name}: missing key '{k}' in response")
            
            checks.append({
                "name": name, "status": "ok" if (ok and key_ok) else "fail",
                "status_code": status_code, "response_ms": elapsed_ms,
            })
            if not ok:
                errors.append(f"{name}: HTTP {status_code}")
            return ok and key_ok
        except Exception as exc:
            elapsed_ms = round((time.monotonic() - start) * 1000)
            checks.append({"name": name, "status": "error", "response_ms": elapsed_ms, "detail": str(exc)})
            errors.append(f"{name}: {exc}")
            return False

    _check("POST /reset", "POST", "/reset", {}, expect_keys=["message", "reward"])
    _check("POST /step", "POST", "/step",
           {"tool": "read_shift_log", "arguments": {"query": "payment", "limit": 3}},
           expect_keys=["reward"])
    _check("GET /state", "GET", "/state", expect_keys=["shift_id"])
    _check("GET /tools", "GET", "/tools", expect_keys=["tools"])

    failed = sum(1 for c in checks if c["status"] != "ok")
    status = "ok" if failed == 0 else ("degraded" if failed < len(checks) else "down")
    return {"status": status, "checks": checks, "errors": errors}


def _check_api_health_stdlib(base_url: str) -> dict[str, Any]:
    """Fallback implementation using urllib when httpx is not installed."""
    import urllib.request
    import urllib.error

    checks: list[dict] = []
    errors: list[str] = []

    def _do(name: str, method: str, path: str, body: Any = None,
            expect_keys: list[str] | None = None) -> bool:
        start = time.monotonic()
        try:
            url = f"{base_url}{path}"
            data = json.dumps(body or {}).encode() if body is not None else b"{}"
            req = urllib.request.Request(url, data=data,
                                         headers={"Content-Type": "application/json"},
                                         method=method)
            with urllib.request.urlopen(req, timeout=10) as resp:
                elapsed_ms = round((time.monotonic() - start) * 1000)
                payload = json.loads(resp.read())
                missing = [k for k in (expect_keys or []) if k not in payload]
                ok = not missing
                checks.append({"name": name, "status": "ok" if ok else "fail",
                                "status_code": resp.status, "response_ms": elapsed_ms})
                if missing:
                    errors.append(f"{name}: missing keys {missing}")
                return ok
        except Exception as exc:
            elapsed_ms = round((time.monotonic() - start) * 1000)
            checks.append({"name": name, "status": "error",
                            "response_ms": elapsed_ms, "detail": str(exc)})
            errors.append(f"{name}: {exc}")
            return False

    _do("POST /reset", "POST", "/reset", {}, ["message", "reward"])
    _do("POST /step", "POST", "/step",
        {"tool": "read_shift_log", "arguments": {"query": "payment", "limit": 3}}, ["reward"])
    _do("GET /state", "GET", "/state", None, ["shift_id"])
    _do("GET /tools", "GET", "/tools", None, ["tools"])

    failed = sum(1 for c in checks if c["status"] != "ok")
    status = "ok" if failed == 0 else ("degraded" if failed < len(checks) else "down")
    return {"status": status, "checks": checks, "errors": errors}


# ---------------------------------------------------------------------------
# 1B — Environment Logic Validation
# ---------------------------------------------------------------------------

def validate_scenarios() -> dict[str, Any]:
    """
    Validate that all 6 scenario families generate correctly, causal links
    are internally consistent, and reward weights sum to 1.0.

    Returns:
        dict with keys: valid (bool), issues (list[str])
    """
    issues: list[str] = []

    # Import scenarios
    try:
        from shiftlog_gym.scenarios import PUBLIC_FAMILIES, ScenarioFactory
    except Exception as exc:
        return {"valid": False, "issues": [f"Cannot import scenarios: {exc}"]}

    factory = ScenarioFactory()

    for family in PUBLIC_FAMILIES:
        try:
            scenario = factory.generate(seed=42, family=family)
        except Exception as exc:
            issues.append(f"Family '{family}' failed to generate: {exc}")
            continue

        all_ids = {inc.incident_id for inc in scenario.incidents}

        for inc in scenario.incidents:
            # Required fields
            for attr in ("service", "summary", "root_cause", "mitigation", "family"):
                if not getattr(inc, attr, None):
                    issues.append(f"[{family}] {inc.incident_id}: missing '{attr}'")

            # Causal link consistency
            for pred_id in inc.linked_precursor_ids:
                if pred_id not in all_ids:
                    issues.append(
                        f"[{family}] {inc.incident_id}: causal_predecessor '{pred_id}' not found in scenario"
                    )

    # Validate reward weights
    try:
        from shiftlog_gym.rewards import DEFAULT_RUBRIC
        weight_sum = round(sum(w for _, w in DEFAULT_RUBRIC.rubrics), 8)
        if weight_sum != 1.0:
            issues.append(f"Reward weights sum to {weight_sum}, expected 1.0")
    except Exception as exc:
        issues.append(f"Cannot validate reward weights: {exc}")

    return {"valid": len(issues) == 0, "issues": issues}


# ---------------------------------------------------------------------------
# 1C — Training Artifacts Check
# ---------------------------------------------------------------------------

def check_training_artifacts() -> dict[str, Any]:
    """
    Check plots, baselines.json, and model hub accessibility.

    Returns:
        dict with keys: plots_ready, baselines_real, model_published, missing, warnings
    """
    missing: list[str] = []
    warnings: list[str] = []

    # Check plots
    plot_files = [
        "01_reward_curve.png",
        "02_recall_bonus_curve.png",
        "03_mttr_comparison.png",
    ]
    plots_ready = True
    for fname in plot_files:
        path = PLOTS_DIR / fname
        if not path.exists():
            missing.append(f"plots/{fname}")
            plots_ready = False
        elif path.stat().st_size == 0:
            warnings.append(f"plots/{fname} is zero bytes")
            plots_ready = False
        else:
            # Quick PNG magic-byte check
            with open(path, "rb") as f:
                header = f.read(8)
            if not header.startswith(b"\x89PNG"):
                warnings.append(f"plots/{fname} is not a valid PNG file")
                plots_ready = False

    # Check baselines.json
    baselines_real = False
    if not BASELINES_FILE.exists():
        missing.append("observatory/baselines.json")
    else:
        try:
            payload = json.loads(BASELINES_FILE.read_text(encoding="utf-8"))
            required_keys = ["random", "llm_base", "trained_llm", "_metadata"]
            for k in required_keys:
                if k not in payload:
                    warnings.append(f"baselines.json: missing key '{k}'")

            note = payload.get("_metadata", {}).get("note", "")
            run_id = payload.get("_metadata", {}).get("run_id", "pending")

            if "simulated" in note.lower():
                warnings.append("baselines.json: _metadata.note contains 'simulated' — not real results")
                baselines_real = False
            elif run_id == "pending":
                warnings.append("baselines.json: run_id is 'pending' — training not yet run")
                baselines_real = False
            else:
                baselines_real = True

        except json.JSONDecodeError as exc:
            warnings.append(f"baselines.json: invalid JSON — {exc}")

    # Check model hub accessibility
    model_published = False
    try:
        import urllib.request
        req = urllib.request.Request(HF_MODEL_URL, method="HEAD")
        with urllib.request.urlopen(req, timeout=8) as resp:
            model_published = resp.status == 200
    except Exception:
        warnings.append("Model hub URL not accessible (may be private or not yet uploaded)")

    return {
        "plots_ready": plots_ready,
        "baselines_real": baselines_real,
        "model_published": model_published,
        "missing": missing,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# 1D — Log Parser
# ---------------------------------------------------------------------------

def parse_recent_logs(log_file_path: str = "logs/runtime.log") -> dict[str, Any]:
    """
    Parse the last 500 lines of the runtime log and detect error patterns.

    Returns:
        dict with keys: error_count, error_groups, critical
    """
    path = ROOT / log_file_path
    if not path.exists():
        return {"error_count": 0, "error_groups": {}, "critical": []}

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return {"error_count": 0, "error_groups": {}, "critical": ["Could not read log file"]}

    recent = lines[-500:]
    error_groups: dict[str, int] = {}
    critical: list[str] = []

    patterns = {
        "ERROR": re.compile(r"\bERROR\b", re.IGNORECASE),
        "Traceback": re.compile(r"\bTraceback\b"),
        "HTTP 500": re.compile(r"\b500\b"),
        "Timeout": re.compile(r"\btimeout\b", re.IGNORECASE),
        "Connection refused": re.compile(r"connection refused", re.IGNORECASE),
        "NoneType": re.compile(r"NoneType.*has no attribute", re.IGNORECASE),
        "Reward weights": re.compile(r"reward_weights do not sum", re.IGNORECASE),
        "Causal predecessor": re.compile(r"causal_predecessor not found", re.IGNORECASE),
    }

    critical_patterns = {"NoneType", "Reward weights", "Causal predecessor"}

    for line in recent:
        for name, pat in patterns.items():
            if pat.search(line):
                error_groups[name] = error_groups.get(name, 0) + 1
                if name in critical_patterns:
                    critical.append(line.strip()[:200])

    error_count = sum(error_groups.values())
    return {
        "error_count": error_count,
        "error_groups": error_groups,
        "critical": critical[:20],  # cap at 20
    }


# ---------------------------------------------------------------------------
# 1E — Full health check runner
# ---------------------------------------------------------------------------

def run_full_health_check(base_url: str = "http://localhost:7860",
                           call_direct: dict[str, Callable] | None = None) -> dict[str, Any]:
    """
    Run all health checks and write results to observatory/health_status.json.

    Args:
        base_url: The URL to check (if not call_direct)
        call_direct: Optional map of paths to local functions to avoid HTTP calls.

    Returns:
        Combined health status dict.
    """
    import datetime

    api_result = check_api_health(base_url, call_direct=call_direct)
    scenario_result = validate_scenarios()
    artifact_result = check_training_artifacts()
    log_result = parse_recent_logs()

    # Overall status
    all_ok = (
        api_result["status"] == "ok"
        and scenario_result["valid"]
        and len(artifact_result.get("missing", [])) == 0
        and log_result["error_count"] == 0
    )
    degraded = not all_ok and api_result["status"] != "down"
    overall = "ok" if all_ok else ("degraded" if degraded else "down")

    result = {
        "overall": overall,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "api": api_result,
        "scenarios": scenario_result,
        "artifacts": artifact_result,
        "logs": log_result,
    }

    # Write to observatory/health_status.json
    try:
        OBS_ROOT.mkdir(parents=True, exist_ok=True)
        HEALTH_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not write health_status.json: %s", exc)

    # Console summary
    icon = {"ok": "✅", "degraded": "⚠️", "down": "❌"}.get(overall, "❓")
    logger.info("%s Health Check: %s | API: %s | Scenarios: %s | Artifacts: missing=%s",
                icon, overall, api_result["status"],
                "valid" if scenario_result["valid"] else "INVALID",
                artifact_result.get("missing", []))

    return result
