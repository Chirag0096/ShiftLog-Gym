from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from .domain import IncidentDefinition


PUBLIC_FAMILIES = (
    "db_pool",
    "auth_cascade",
    "oom_regression",
    "feature_flag",
    "network_partition",
    "config_drift",
)

FAMILY_ALIASES = {
    "db_pool_exhaustion": "db_pool",
    "auth_timeout_cascade": "auth_cascade",
    "memory_oom_signature": "oom_regression",
    "feature_flag_regression": "feature_flag",
}


@dataclass(frozen=True, slots=True)
class BaseIncident:
    incident_id: str
    family: str
    service: str
    summary: str
    symptoms: tuple[str, ...]
    customer_impact: str
    root_cause: str
    mitigation: str
    diagnostics: dict[str, str]
    valid_mitigations: tuple[str, ...]
    service_names: tuple[str, ...]
    error_codes: tuple[str, ...]
    shift_log_keywords: tuple[str, ...]
    linked_precursor_ids: tuple[str, ...] = ()
    required_memory_keywords: tuple[str, ...] = ()
    is_noise: bool = False


@dataclass(frozen=True, slots=True)
class PrecursorIncident(BaseIncident):
    pass


@dataclass(frozen=True, slots=True)
class LinkedIncident(BaseIncident):
    pass


@dataclass(frozen=True, slots=True)
class NoiseIncident(BaseIncident):
    is_noise: bool = True


@dataclass(frozen=True, slots=True)
class Scenario:
    family: str
    seed: int
    precursor_incidents: tuple[PrecursorIncident, ...]
    linked_incidents: tuple[LinkedIncident, ...]
    noise_incidents: tuple[NoiseIncident, ...]
    ground_truth: dict[str, str]
    correct_mitigation: dict[str, str]
    valid_mitigations: dict[str, tuple[str, ...]]

    @property
    def incidents(self) -> tuple[BaseIncident, ...]:
        return self.precursor_incidents + self.linked_incidents + self.noise_incidents


class ScenarioFactory:
    def __init__(self, default_family: str | None = None) -> None:
        self.default_family = default_family

    def generate(self, seed: int, family: str | None = None) -> Scenario:
        randomizer = Random(seed)
        resolved_family = self._resolve_family(family or self.default_family or randomizer.choice(PUBLIC_FAMILIES))
        if resolved_family not in PUBLIC_FAMILIES:
            raise ValueError(f"Unsupported family: {family}")
        return self._build_scenario(randomizer, seed, resolved_family)

    def _resolve_family(self, family: str) -> str:
        return FAMILY_ALIASES.get(family, family)

    def _incident_id(self, family: str, seed: int, prefix: str, index: int) -> str:
        family_token = family.replace("_", "-")
        return f"{family_token}-{prefix}-{seed:02d}-{index:02d}"

    def _build_scenario(self, randomizer: Random, seed: int, family: str) -> Scenario:
        service_pool = self._service_pool(family)
        precursor_count = randomizer.randint(1, 3)
        linked_count = randomizer.randint(2, 4)
        noise_count = randomizer.randint(1, 2)

        precursor_incidents = tuple(
            self._make_precursor(randomizer, family, seed, index, service_pool)
            for index in range(precursor_count)
        )
        linked_incidents = tuple(
            self._make_linked(randomizer, family, seed, index, precursor_incidents, service_pool)
            for index in range(linked_count)
        )
        noise_incidents = tuple(
            self._make_noise(randomizer, family, seed, index, service_pool)
            for index in range(noise_count)
        )

        ground_truth: dict[str, str] = {}
        correct_mitigation: dict[str, str] = {}
        valid_mitigations: dict[str, tuple[str, ...]] = {}
        for incident in precursor_incidents + linked_incidents + noise_incidents:
            ground_truth[incident.incident_id] = incident.root_cause
            correct_mitigation[incident.incident_id] = incident.mitigation
            valid_mitigations[incident.service] = incident.valid_mitigations

        return Scenario(
            family=family,
            seed=seed,
            precursor_incidents=precursor_incidents,
            linked_incidents=linked_incidents,
            noise_incidents=noise_incidents,
            ground_truth=ground_truth,
            correct_mitigation=correct_mitigation,
            valid_mitigations=valid_mitigations,
        )

    def _service_pool(self, family: str) -> tuple[str, ...]:
        return {
            "db_pool": ("payments-api", "invoice-worker", "orders-db", "config-service"),
            "auth_cascade": ("auth-gateway", "profile-api", "session-cache", "edge-router"),
            "oom_regression": ("ranking-worker", "recommendation-api", "model-registry", "feature-store"),
            "feature_flag": ("checkout-web", "receipts-service", "feature-flag-service", "cdn"),
            "network_partition": ("edge-proxy", "inventory-api", "service-mesh", "warehouse-sync"),
            "config_drift": ("config-service", "billing-api", "worker-scheduler", "runtime-env"),
        }[family]

    def _family_template(self, family: str) -> dict[str, tuple[str, ...] | str]:
        templates: dict[str, dict[str, tuple[str, ...] | str]] = {
            "db_pool": {
                "symptoms": (
                    "connection wait count spiking",
                    "query latency elevated after rollback",
                    "pool saturation warning",
                ),
                "noise_symptoms": (
                    "connection wait count spiking",
                    "query latency elevated after rollback",
                ),
                "root_causes": (
                    "stale DB pool size after rollback",
                    "shared client pool template drift",
                    "orphaned connection reuse after deploy",
                ),
                "noise_root_causes": (
                    "vacuum lock on primary database",
                    "reporting replica failover lag",
                ),
                "mitigations": (
                    "set_pool_size_and_restart",
                    "restart_pgbouncer",
                    "roll_forward_pool_fix",
                ),
                "error_codes": ("DBPOOL-44", "PG-TOO-MANY-CONNECTIONS", "WAIT-641"),
            },
            "auth_cascade": {
                "symptoms": (
                    "auth timeout spike",
                    "token verification backlog",
                    "session cache saturation",
                ),
                "noise_symptoms": (
                    "auth timeout spike",
                    "token verification backlog",
                ),
                "root_causes": (
                    "stale mesh route to saturated cache shard",
                    "auth gateway dependency timeout cascade",
                    "session cache shard misrouting",
                ),
                "noise_root_causes": (
                    "OAuth provider rate limiting",
                    "invalid JWT audience rollout",
                ),
                "mitigations": (
                    "refresh_route_and_drain_shard",
                    "reroute_session_cache",
                    "disable_auth_fallback",
                ),
                "error_codes": ("AUTH-504", "JWT-BURST", "CACHE-SAT-95"),
            },
            "oom_regression": {
                "symptoms": (
                    "pods restarting with OOMKilled",
                    "warmup latency spike",
                    "allocator fragmentation signature",
                ),
                "noise_symptoms": (
                    "pods restarting with OOMKilled",
                    "warmup latency spike",
                ),
                "root_causes": (
                    "batch size regression after model refresh",
                    "shared inference config memory blowup",
                    "allocator fragmentation under warmup batch",
                ),
                "noise_root_causes": (
                    "kernel memory pressure on host node",
                    "ephemeral cache growth in sidecar",
                ),
                "mitigations": (
                    "reduce_batch_and_roll_forward",
                    "pin_previous_model_bundle",
                    "restart_with_memory_guard",
                ),
                "error_codes": ("OOM-137", "ALLOC-14", "WARMUP-502"),
            },
            "feature_flag": {
                "symptoms": (
                    "conversion drop for one cohort",
                    "stale bundle served from CDN",
                    "downstream payload mismatch",
                ),
                "noise_symptoms": (
                    "conversion drop for one cohort",
                    "stale bundle served from CDN",
                ),
                "root_causes": (
                    "shadow feature flag routed traffic to deprecated bundle",
                    "flagged cohort mismatch with receipts schema",
                    "stale experiment assignment across CDN cache",
                ),
                "noise_root_causes": (
                    "payment provider iframe outage",
                    "email receipt signer clock skew",
                ),
                "mitigations": (
                    "disable_flag_and_purge_bundle",
                    "rollback_shadow_assignment",
                    "purge_experiment_cdn",
                ),
                "error_codes": ("FLAG-DRIFT", "CDN-STALE-202", "RECEIPT-MISMATCH"),
            },
            "network_partition": {
                "symptoms": (
                    "cross-zone request timeout",
                    "service mesh retry storm",
                    "inventory sync lag",
                ),
                "noise_symptoms": (
                    "cross-zone request timeout",
                    "service mesh retry storm",
                ),
                "root_causes": (
                    "partial service mesh partition between edge and inventory",
                    "warehouse sync downstream partition cascade",
                    "zonal routing partition on edge proxy",
                ),
                "noise_root_causes": (
                    "DNS resolver cache poisoning on one node pool",
                    "warehouse API throttling",
                ),
                "mitigations": (
                    "isolate_partition_and_failover",
                    "reroute_cross_zone_traffic",
                    "drain_partitioned_mesh_segment",
                ),
                "error_codes": ("NET-408", "MESH-RETRY-LOOP", "SYNC-LAG-33"),
            },
            "config_drift": {
                "symptoms": (
                    "service mismatch after deploy",
                    "scheduler executing stale settings",
                    "runtime env checksum drift",
                ),
                "noise_symptoms": (
                    "service mismatch after deploy",
                    "runtime env checksum drift",
                ),
                "root_causes": (
                    "config checksum drift between service and scheduler",
                    "runtime environment stale secret mount",
                    "billing service config precedence mismatch",
                ),
                "noise_root_causes": (
                    "delayed secret replication from vault",
                    "temporary scheduler leadership flap",
                ),
                "mitigations": (
                    "reconcile_config_and_restart",
                    "force_config_resync",
                    "reload_runtime_env",
                ),
                "error_codes": ("CFG-DRIFT", "ENV-CHECKSUM", "SCHED-STALE"),
            },
        }
        return templates[family]

    def _make_precursor(
        self,
        randomizer: Random,
        family: str,
        seed: int,
        index: int,
        service_pool: tuple[str, ...],
    ) -> PrecursorIncident:
        template = self._family_template(family)
        service = service_pool[index % len(service_pool)]
        root_cause = randomizer.choice(template["root_causes"])
        mitigation = randomizer.choice(template["mitigations"])
        error_code = randomizer.choice(template["error_codes"])
        symptoms = tuple(randomizer.sample(template["symptoms"], k=min(3, len(template["symptoms"]))))
        diagnostics = {
            "connections": f"Observed {error_code} on {service}",
            "deploys": f"Most recent deploy left {service} in a suspicious state for {family}",
        }
        keywords = tuple(token for token in root_cause.replace("-", " ").split() if len(token) > 3)
        return PrecursorIncident(
            incident_id=self._incident_id(family, seed, "pre", index),
            family=family,
            service=service,
            summary=f"Precursor incident on {service} for {family}",
            symptoms=symptoms,
            customer_impact=f"{service} is causing elevated error rates",
            root_cause=root_cause,
            mitigation=mitigation,
            diagnostics=diagnostics,
            valid_mitigations=tuple(template["mitigations"]),
            service_names=service_pool,
            error_codes=(error_code,),
            shift_log_keywords=keywords,
        )

    def _make_linked(
        self,
        randomizer: Random,
        family: str,
        seed: int,
        index: int,
        precursors: tuple[PrecursorIncident, ...],
        service_pool: tuple[str, ...],
    ) -> LinkedIncident:
        template = self._family_template(family)
        linked_precursors = tuple(
            precursor.incident_id
            for precursor in randomizer.sample(
                list(precursors),
                k=randomizer.randint(1, len(precursors)),
            )
        )
        root_cause = randomizer.choice(template["root_causes"])
        mitigation = randomizer.choice(template["mitigations"])
        service = service_pool[(index + len(precursors)) % len(service_pool)]
        symptoms = tuple(randomizer.sample(template["symptoms"], k=min(2, len(template["symptoms"]))))
        keywords = tuple(token for token in root_cause.replace("-", " ").split() if len(token) > 3)
        return LinkedIncident(
            incident_id=self._incident_id(family, seed, "lnk", index),
            family=family,
            service=service,
            summary=f"Linked incident on {service} caused by prior {family} failure",
            symptoms=symptoms,
            customer_impact=f"Downstream impact from precursor incidents on {service}",
            root_cause=root_cause,
            mitigation=mitigation,
            diagnostics={
                "dependency": f"{service} shows downstream symptoms matching {family}",
                "history": f"Possible causal chain from {', '.join(linked_precursors)}",
            },
            valid_mitigations=tuple(template["mitigations"]),
            service_names=service_pool,
            error_codes=(randomizer.choice(template["error_codes"]),),
            shift_log_keywords=keywords,
            linked_precursor_ids=linked_precursors,
            required_memory_keywords=keywords[:2],
        )

    def _make_noise(
        self,
        randomizer: Random,
        family: str,
        seed: int,
        index: int,
        service_pool: tuple[str, ...],
    ) -> NoiseIncident:
        template = self._family_template(family)
        root_cause = randomizer.choice(template["noise_root_causes"])
        service = randomizer.choice(service_pool)
        mitigation = randomizer.choice(tuple(mit for mit in template["mitigations"] if "restart" in mit or "reload" in mit) or template["mitigations"])
        symptoms = tuple(randomizer.sample(template["noise_symptoms"], k=min(2, len(template["noise_symptoms"]))))
        keywords = tuple(token for token in root_cause.replace("-", " ").split() if len(token) > 3)
        return NoiseIncident(
            incident_id=self._incident_id(family, seed, "noi", index),
            family=family,
            service=service,
            summary=f"Noise incident on {service} that resembles {family}",
            symptoms=symptoms,
            customer_impact=f"Transient issue on {service} with misleadingly familiar symptoms",
            root_cause=root_cause,
            mitigation=mitigation,
            diagnostics={
                "noise_check": f"{service} symptoms overlap with {family} but root cause is standalone",
            },
            valid_mitigations=(mitigation,),
            service_names=service_pool,
            error_codes=(randomizer.choice(template["error_codes"]),),
            shift_log_keywords=keywords,
        )


SCENARIO_REGISTRY: dict[str, ScenarioFactory] | None = None

def get_scenario_registry() -> dict[str, ScenarioFactory]:
    """Lazy-load the scenario registry on first use."""
    global SCENARIO_REGISTRY
    if SCENARIO_REGISTRY is None:
        SCENARIO_REGISTRY = {
            family: ScenarioFactory(default_family=family) for family in PUBLIC_FAMILIES
        }
    return SCENARIO_REGISTRY

FAMILIES = PUBLIC_FAMILIES + tuple(FAMILY_ALIASES.keys())


def _legacy_incident_from_base(incident: BaseIncident, seed: int, index: int) -> IncidentDefinition:
    required_memory_keys = ["root_cause_fact", "mitigation_fact"] if incident.linked_precursor_ids else []
    root_cause_exemplar = incident.root_cause
    mitigation_exemplar = f"Mitigation: {incident.mitigation} on {incident.service}."
    if incident.family == "db_pool" and incident.service == "payments-api":
        root_cause_exemplar = "Rollback left payments-api with stale DB_POOL_SIZE=44."
        mitigation_exemplar = "Mitigation: set_pool_size_and_restart on payments-api."
    golden_memory = [
        ("root_cause_fact", root_cause_exemplar),
        ("mitigation_fact", mitigation_exemplar),
    ]
    linked_to = incident.linked_precursor_ids[0] if incident.linked_precursor_ids else None
    relevant_terms = list(dict.fromkeys(
        list(incident.shift_log_keywords)
        + [incident.service]
        + [token for symptom in incident.symptoms for token in symptom.lower().replace("-", " ").split() if len(token) > 3]
    ))
    return IncidentDefinition(
        incident_id=incident.incident_id,
        family=incident.family,
        variant_id=f"v{seed:02d}",
        sequence_index=index,
        service=incident.service,
        summary=incident.summary,
        symptoms=list(incident.symptoms),
        customer_impact=incident.customer_impact,
        root_cause=incident.root_cause,
        resolution=incident.mitigation,
        diagnostics=incident.diagnostics,
        dependencies=list(incident.service_names),
        linked_to=linked_to,
        required_memory_keys=required_memory_keys,
        relevant_memory_terms=relevant_terms,
        golden_memory=golden_memory,
        unsupported_resolutions=["invalid_mitigation", "memory_retrieved_guess"],
        runbook=f"{incident.family}/{incident.service}",
        owner="platform-oncall",
        log_context=[f"error_code={code}" for code in incident.error_codes],
        valid_mitigations=list(incident.valid_mitigations),
    )


def build_scenario_library(variants_per_family: int = 8) -> dict[str, list[list[IncidentDefinition]]]:
    library: dict[str, list[list[IncidentDefinition]]] = {family: [] for family in FAMILIES}
    base_factory = ScenarioFactory()
    for public_family in PUBLIC_FAMILIES:
        variants: list[list[IncidentDefinition]] = []
        for seed in range(1, variants_per_family + 1):
            scenario = base_factory.generate(seed=seed, family=public_family)
            combined = scenario.precursor_incidents + scenario.linked_incidents + scenario.noise_incidents
            variants.append([
                _legacy_incident_from_base(incident, seed, index)
                for index, incident in enumerate(combined, start=1)
            ])
        library[public_family] = variants
    for alias, public_family in FAMILY_ALIASES.items():
        library[alias] = library[public_family]
    return library
