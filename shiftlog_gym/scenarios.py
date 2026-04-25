from __future__ import annotations

from dataclasses import replace
from random import Random

from .domain import IncidentDefinition


FAMILIES = (
    "db_pool_exhaustion",
    "auth_timeout_cascade",
    "memory_oom_signature",
    "feature_flag_regression",
)


def _variant_suffix(seed: int) -> str:
    return f"v{seed:02d}"


def _incident_id(family: str, seed: int, index: int) -> str:
    short = family.split("_")[0]
    return f"{short}-{seed:02d}-{index:02d}"


def _db_family(seed: int) -> list[IncidentDefinition]:
    rg = Random(seed)
    pool = 40 + rg.randint(0, 20)
    query = 350 + rg.randint(0, 300)
    deploy = f"payments-api@2026.04.{10 + seed}"
    fact_key = f"db-pool-signature-{seed}"
    precursor = IncidentDefinition(
        incident_id=_incident_id("db_pool_exhaustion", seed, 1),
        family="db_pool_exhaustion",
        variant_id=_variant_suffix(seed),
        sequence_index=1,
        service="payments-api",
        summary="Checkout requests are timing out after a rollback.",
        symptoms=[
            f"pgbouncer wait_count spiking with pool size pinned at {pool}",
            f"p99 query latency above {query}ms after rollback",
        ],
        customer_impact="New card checkouts failing intermittently in one region.",
        root_cause=f"rollback preserved stale DB_POOL_SIZE={pool} in payments-api",
        resolution="set_pool_size_and_restart",
        diagnostics={
            "connections": f"Pool maxed at {pool}; rollback reused stale env var.",
            "deploys": f"Rollback to {deploy} restored an outdated secret mount.",
        },
        dependencies=["orders-db", "pgbouncer"],
        relevant_memory_terms=["rollback", "pool size", "pgbouncer", "payments-api"],
        golden_memory=[
            (fact_key, f"Rollback left payments-api with stale DB_POOL_SIZE={pool}."),
            ("db-pool-mitigation", "Mitigation: set_pool_size_and_restart on payments-api."),
        ],
        unsupported_resolutions=["flush_dns_cache", "rotate_tls_certificate"],
        runbook="payments-api/db-pool-exhaustion",
        log_context=[f"Deployment rollback reference: {deploy}"],
        valid_mitigations=["set_pool_size_and_restart", "restart_pgbouncer"],
    )
    linked = IncidentDefinition(
        incident_id=_incident_id("db_pool_exhaustion", seed, 2),
        family="db_pool_exhaustion",
        variant_id=_variant_suffix(seed),
        sequence_index=2,
        service="invoice-worker",
        summary="Invoice generation backlog grows with the same database wait pattern.",
        symptoms=[
            "workers stuck waiting for DB connections",
            "queue delay increasing but CPU remains normal",
        ],
        customer_impact="Delayed invoicing and reconciliation jobs.",
        root_cause=f"shared stale DB_POOL_SIZE={pool} reused by invoice-worker config template",
        resolution="set_pool_size_and_restart",
        diagnostics={
            "connections": "invoice-worker inherits the same pool template as payments-api",
            "dependency": "orders-db itself is healthy; saturation is client-side",
        },
        dependencies=["orders-db", "config-service"],
        linked_to=precursor.incident_id,
        required_memory_keys=[fact_key, "db-pool-mitigation"],
        relevant_memory_terms=["pool size", "payments-api", "rollback", "invoice-worker"],
        golden_memory=[("invoice-shared-template", "invoice-worker shares the stale DB pool template.")],
        unsupported_resolutions=["increase_cpu_limit", "purge_queue_only"],
        runbook="invoice-worker/db-pool-inherited-config",
        valid_mitigations=["set_pool_size_and_restart"],
    )
    filler = IncidentDefinition(
        incident_id=_incident_id("db_pool_exhaustion", seed, 3),
        family="db_pool_exhaustion",
        variant_id=_variant_suffix(seed),
        sequence_index=3,
        service="notifications-api",
        summary="Notifications degraded due to retry storm after checkout failures.",
        symptoms=["retry queue elevated", "dependency saturation on payments-api"],
        customer_impact="Confirmation emails delayed.",
        root_cause="downstream retry amplification from checkout failures",
        resolution="throttle_retries",
        diagnostics={"retries": "notification retries correlate with failed checkout volume"},
        dependencies=["payments-api"],
        relevant_memory_terms=["retries", "checkout failures"],
        golden_memory=[],
        unsupported_resolutions=["set_pool_size_and_restart"],
        runbook="notifications/retry-storm",
        valid_mitigations=["throttle_retries"],
    )
    return [precursor, filler, linked]


def _auth_family(seed: int) -> list[IncidentDefinition]:
    rg = Random(seed + 100)
    timeout = 2 + rg.randint(1, 3)
    cert_slot = 12 + rg.randint(0, 5)
    fact_key = f"auth-stale-route-{seed}"
    precursor = IncidentDefinition(
        incident_id=_incident_id("auth_timeout_cascade", seed, 1),
        family="auth_timeout_cascade",
        variant_id=_variant_suffix(seed),
        sequence_index=1,
        service="auth-gateway",
        summary="Login success drops after edge routing change.",
        symptoms=[
            f"auth-gateway timing out after {timeout}s to session-cache",
            "error budget burn begins in the EU region",
        ],
        customer_impact="User logins timing out for a subset of traffic.",
        root_cause="stale service mesh route sends auth-gateway to a saturated cache shard",
        resolution="refresh_route_and_drain_shard",
        diagnostics={
            "mesh": f"route slot {cert_slot} still points to old cache shard",
            "cache": "one shard over 95% connection saturation",
        },
        dependencies=["service-mesh", "session-cache"],
        relevant_memory_terms=["stale route", "auth", "cache shard"],
        golden_memory=[
            (fact_key, "Auth timeouts were caused by a stale mesh route to one cache shard."),
            ("auth-route-fix", "Fix: refresh_route_and_drain_shard on auth-gateway."),
        ],
        unsupported_resolutions=["scale_api_only", "rotate_oauth_secret"],
        runbook="auth/stale-route-cache-shard",
        valid_mitigations=["refresh_route_and_drain_shard"],
    )
    linked = IncidentDefinition(
        incident_id=_incident_id("auth_timeout_cascade", seed, 2),
        family="auth_timeout_cascade",
        variant_id=_variant_suffix(seed),
        sequence_index=2,
        service="profile-api",
        summary="Profile reads time out but only for recently authenticated users.",
        symptoms=["burst of auth-token verification latency", "profile-api threads waiting on auth-gateway"],
        customer_impact="Users can log in eventually but profile pages spin indefinitely.",
        root_cause="profile-api depends on auth-gateway, which still uses the stale route",
        resolution="refresh_route_and_drain_shard",
        diagnostics={
            "dependency": "profile-api latency follows auth-gateway timeout spikes",
        },
        dependencies=["auth-gateway"],
        linked_to=precursor.incident_id,
        required_memory_keys=[fact_key, "auth-route-fix"],
        relevant_memory_terms=["auth-gateway", "stale route", "profile-api"],
        unsupported_resolutions=["invalidate_jwt_cache", "increase_profile_workers"],
        runbook="profile/auth-dependency-timeouts",
        valid_mitigations=["refresh_route_and_drain_shard"],
    )
    filler = IncidentDefinition(
        incident_id=_incident_id("auth_timeout_cascade", seed, 3),
        family="auth_timeout_cascade",
        variant_id=_variant_suffix(seed),
        sequence_index=3,
        service="fraud-scoring",
        summary="Fraud scoring slowed due to token verification fallback.",
        symptoms=["verification fallback mode enabled", "latency increase but no errors"],
        customer_impact="Checkout fraud checks slower but still completing.",
        root_cause="token verification fallback created extra network hops",
        resolution="disable_fallback_mode",
        diagnostics={"fallback": "fallback mode enabled after auth-gateway incident"},
        dependencies=["auth-gateway"],
        relevant_memory_terms=["fallback mode"],
        unsupported_resolutions=["refresh_route_and_drain_shard"],
        runbook="fraud/fallback-verification",
        valid_mitigations=["disable_fallback_mode"],
    )
    return [precursor, filler, linked]


def _oom_family(seed: int) -> list[IncidentDefinition]:
    rg = Random(seed + 200)
    batch = 500 + rg.randint(0, 250)
    signature = f"allocator-fragmentation-{rg.randint(7, 17)}"
    fact_key = f"oom-signature-{seed}"
    precursor = IncidentDefinition(
        incident_id=_incident_id("memory_oom_signature", seed, 1),
        family="memory_oom_signature",
        variant_id=_variant_suffix(seed),
        sequence_index=1,
        service="ranking-worker",
        summary="Ranking worker pods OOMKilled after a model refresh.",
        symptoms=["pods restart every 15 minutes", f"batch size jumped to {batch}"],
        customer_impact="Personalized ranking quality degraded.",
        root_cause=f"batch size {batch} triggers {signature} during model refresh",
        resolution="reduce_batch_and_roll_forward",
        diagnostics={
            "heap": f"memory profile matches {signature}",
            "deploy": "latest model refresh increased activation footprint",
        },
        dependencies=["feature-store", "model-registry"],
        relevant_memory_terms=["oom", "batch size", "ranking-worker"],
        golden_memory=[
            (fact_key, f"OOM signature is {signature} triggered by batch size {batch}."),
            ("oom-fix", "Fix: reduce_batch_and_roll_forward."),
        ],
        unsupported_resolutions=["restart_only", "scale_replicas_only"],
        runbook="ranking/oom-after-refresh",
        valid_mitigations=["reduce_batch_and_roll_forward"],
    )
    linked = IncidentDefinition(
        incident_id=_incident_id("memory_oom_signature", seed, 2),
        family="memory_oom_signature",
        variant_id=_variant_suffix(seed),
        sequence_index=2,
        service="recommendation-api",
        summary="Recommendation API latency spikes with periodic pod evictions.",
        symptoms=["OOM evictions every 20 minutes", "latency spike after warmup"],
        customer_impact="Homepage recommendations stale or missing.",
        root_cause=f"recommendation-api inherited the same {signature} via shared model refresh settings",
        resolution="reduce_batch_and_roll_forward",
        diagnostics={"warmup": "warmup batch and ranking-worker settings share one config block"},
        dependencies=["ranking-worker", "config-service"],
        linked_to=precursor.incident_id,
        required_memory_keys=[fact_key, "oom-fix"],
        relevant_memory_terms=["shared config", "oom signature", "batch size"],
        unsupported_resolutions=["clear_cache_only", "restart_only"],
        runbook="recommendation/shared-batch-oom",
        valid_mitigations=["reduce_batch_and_roll_forward"],
    )
    filler = IncidentDefinition(
        incident_id=_incident_id("memory_oom_signature", seed, 3),
        family="memory_oom_signature",
        variant_id=_variant_suffix(seed),
        sequence_index=3,
        service="analytics-stream",
        summary="Analytics stream lags after replay backlog grows.",
        symptoms=["consumer lag elevated", "disk spill increased"],
        customer_impact="Internal dashboards delayed.",
        root_cause="backlog replay exceeds current stream throughput",
        resolution="increase_consumer_partitions",
        diagnostics={"lag": "backlog replay dominates available throughput"},
        dependencies=["kafka"],
        relevant_memory_terms=["analytics lag"],
        unsupported_resolutions=["reduce_batch_and_roll_forward"],
        runbook="analytics/backlog-replay",
        valid_mitigations=["increase_consumer_partitions"],
    )
    return [precursor, filler, linked]


def _flag_family(seed: int) -> list[IncidentDefinition]:
    rg = Random(seed + 300)
    flag = f"checkout-flow-shadow-{seed}"
    cert = f"edge-cert-{2026 + seed}"
    fact_key = f"flag-regression-{seed}"
    precursor = IncidentDefinition(
        incident_id=_incident_id("feature_flag_regression", seed, 1),
        family="feature_flag_regression",
        variant_id=_variant_suffix(seed),
        sequence_index=1,
        service="checkout-web",
        summary="A shadow flag silently routes some sessions to a deprecated flow.",
        symptoms=["drop in conversion for one browser cohort", "no backend saturation"],
        customer_impact="Some users see failed payment forms.",
        root_cause=f"feature flag {flag} routes traffic to deprecated JS bundle expecting {cert}",
        resolution="disable_flag_and_purge_bundle",
        diagnostics={
            "flags": f"{flag} enabled for 12% of sessions",
            "assets": f"deprecated bundle still references {cert}",
        },
        dependencies=["feature-flag-service", "cdn"],
        relevant_memory_terms=["shadow flag", "deprecated bundle", "checkout-web"],
        golden_memory=[
            (fact_key, f"Flag {flag} routes traffic to deprecated bundle expecting {cert}."),
            ("flag-fix", "Fix: disable_flag_and_purge_bundle."),
        ],
        unsupported_resolutions=["restart_web_only", "increase_pod_memory"],
        runbook="checkout/flag-regression",
        valid_mitigations=["disable_flag_and_purge_bundle"],
    )
    linked = IncidentDefinition(
        incident_id=_incident_id("feature_flag_regression", seed, 2),
        family="feature_flag_regression",
        variant_id=_variant_suffix(seed),
        sequence_index=2,
        service="receipts-service",
        summary="Receipt signing failures appear hours later for the same checkout cohort.",
        symptoms=["signing mismatch for a small but growing cohort", "no queue backlog"],
        customer_impact="Receipts fail to send for affected users.",
        root_cause=f"receipt payloads originate from deprecated bundle behind {flag}",
        resolution="disable_flag_and_purge_bundle",
        diagnostics={"receipts": "payload version mismatch only for shadow cohort"},
        dependencies=["checkout-web"],
        linked_to=precursor.incident_id,
        required_memory_keys=[fact_key, "flag-fix"],
        relevant_memory_terms=["deprecated bundle", "checkout shadow cohort", "receipts"],
        unsupported_resolutions=["rotate_signing_key", "rebuild_queue_only"],
        runbook="receipts/downstream-flag-regression",
        valid_mitigations=["disable_flag_and_purge_bundle"],
    )
    filler = IncidentDefinition(
        incident_id=_incident_id("feature_flag_regression", seed, 3),
        family="feature_flag_regression",
        variant_id=_variant_suffix(seed),
        sequence_index=3,
        service="catalog-web",
        summary="Catalog page assets stale in one region after CDN cache churn.",
        symptoms=["stale assets in one POP", "cache hit ratio unstable"],
        customer_impact="Product page images slow to refresh.",
        root_cause="regional CDN cache churn",
        resolution="purge_regional_cdn",
        diagnostics={"cdn": "one regional POP serving stale objects"},
        dependencies=["cdn"],
        relevant_memory_terms=["cdn cache"],
        unsupported_resolutions=["disable_flag_and_purge_bundle"],
        runbook="catalog/regional-cdn-stale-assets",
        valid_mitigations=["purge_regional_cdn"],
    )
    return [precursor, filler, linked]


def build_scenario_library(variants_per_family: int = 8) -> dict[str, list[list[IncidentDefinition]]]:
    library: dict[str, list[list[IncidentDefinition]]] = {family: [] for family in FAMILIES}
    builders = {
        "db_pool_exhaustion": _db_family,
        "auth_timeout_cascade": _auth_family,
        "memory_oom_signature": _oom_family,
        "feature_flag_regression": _flag_family,
    }
    for family, builder in builders.items():
        for seed in range(1, variants_per_family + 1):
            library[family].append(builder(seed))
    return library


def flatten_scenarios(library: dict[str, list[list[IncidentDefinition]]]) -> list[list[IncidentDefinition]]:
    all_scenarios: list[list[IncidentDefinition]] = []
    for family in FAMILIES:
        all_scenarios.extend(library[family])
    return all_scenarios

