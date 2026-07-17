"""Model-free protocol skeleton for checkpoint cotangent/port tomography.

This file deliberately has no model, checkpoint, bank, MPS, training, lock, or
cell command.  It validates the frozen design, audits immutable file hashes,
enumerates the future scalar record grid, and self-tests the algebra and
inference primitives on synthetic CPU float64 arrays only.

The later scientific runner must be implemented and independently reviewed
only after the parent checkpoint trace and its standalone verification have
exact hashes bound in the config.  This skeleton is not that runner.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs/checkpoint_cotangent_port_assay_v1.json"
SCRIPT_PATH = Path(__file__).resolve()

SEEDS = (56101, 56102)
RECIPIENT_STATES = ("preference", "control")
UPDATES = (8, 16, 32, 64, 128, 256, 512)
PRIMARY_UPDATES = (16, 64, 128, 256)
DESCRIPTIVE_UPDATES = (8, 32)
INTEGRITY_UPDATES = (512,)
ROUTES = (
    "credit_d",
    "incoming_x",
    "endpoint_port",
    "local_port",
    "credit_sham_1",
    "credit_sham_2",
    "rank1_sham_1",
    "rank1_sham_2",
)
DOSES = (-0.5, -0.25, 0.0, 0.25, 0.5)
NONZERO_DOSES = (-0.5, -0.25, 0.25, 0.5)
OUTCOMES = (
    "wolf_margin",
    "preference_nll_benefit",
    "fingerprint_advantage",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PENDING_TRACE = "PENDING_TRACE_COMPLETION_BEFORE_LOCK"
PENDING_IMPLEMENTATION = "PENDING_IMPLEMENTATION_BEFORE_LOCK"


@dataclass(frozen=True)
class FutureRecord:
    seed: int
    recipient_state: str
    optimizer_update: int
    route: str
    dose: float
    native: bool
    not_applicable_expected: bool

    @property
    def key(self) -> str:
        dose = "native" if self.native else f"{self.dose:+.2f}"
        return (
            f"seed={self.seed}/state={self.recipient_state}/"
            f"u={self.optimizer_update}/route={self.route}/dose={dose}"
        )

    def json(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "recipient_state": self.recipient_state,
            "optimizer_update": self.optimizer_update,
            "route": self.route,
            "dose": self.dose,
            "native": self.native,
            "not_applicable_expected": self.not_applicable_expected,
            "key": self.key,
        }


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def int64_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def is_hex64(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def semantic_design_sha256(config: Mapping[str, Any]) -> str:
    """Digest scientific design while normalizing the five hash-binding slots."""
    value = copy.deepcopy(dict(config))
    implementation = value["implementation_contract"]
    implementation["scientific_design_semantic_sha256"] = "<SEMANTIC-DESIGN-SHA256>"
    for pair in implementation["future_exact_before_lock"].values():
        pair[1] = "<PRELOCK-BINDING-SHA256>"
    for pair in value["parents"]["completion_exact_before_lock"].values():
        pair[1] = "<PRELOCK-BINDING-SHA256>"
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derived_sham_seed(
    label: str,
    base_seed: int,
    seed: int,
    canonical_module_name: str,
) -> int:
    namespace = (
        "checkpoint-cotangent-port-assay-v1/sham/"
        f"{label}/base={base_seed}/seed={seed}/module={canonical_module_name}"
    )
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def expected_records() -> tuple[FutureRecord, ...]:
    records: list[FutureRecord] = []
    for seed in SEEDS:
        for state in RECIPIENT_STATES:
            for update in UPDATES:
                records.append(
                    FutureRecord(seed, state, update, "native", 0.0, True, False)
                )
                for route in ROUTES:
                    for dose in NONZERO_DOSES:
                        is_n_a = route == "local_port" and seed == 56101 and update == 256
                        records.append(
                            FutureRecord(seed, state, update, route, dose, False, is_n_a)
                        )
    return tuple(records)


def validate_parent_pair(pair: Any, *, pending_allowed: bool, name: str) -> None:
    require(isinstance(pair, list) and len(pair) == 2, f"Bad parent pair: {name}")
    require(isinstance(pair[0], str) and pair[0], f"Bad parent path: {name}")
    valid_hash = is_hex64(pair[1]) or (
        pending_allowed and pair[1] in {PENDING_TRACE, PENDING_IMPLEMENTATION}
    )
    require(valid_hash, f"Bad parent hash contract: {name}")


def validate_config(config: Mapping[str, Any]) -> None:
    require(config.get("name") == "checkpoint-cotangent-port-assay-v1", "Wrong config")
    require(config["status"].startswith("protocol skeleton frozen"), "Status weakened")
    require(
        "V(G)=s*(dB*A+B*dA)=-s^2" in config["scope"]["lora_writable_route"],
        "LoRA-writable route formula changed",
    )
    require(
        "Each G is a completion-NLL loss gradient" in config["scope"]["effective_weight_factorization"],
        "Loss-gradient sign contract removed",
    )

    measurement = config["measurement"]
    require(tuple(measurement["seeds"]) == SEEDS, "Seed grid changed")
    require(
        tuple(measurement["recipient_states"]) == RECIPIENT_STATES,
        "Recipient-state grid changed",
    )
    checkpoints = measurement["checkpoints"]
    require(tuple(checkpoints["all"]) == UPDATES, "Checkpoint grid changed")
    require(tuple(checkpoints["primary"]) == PRIMARY_UPDATES, "Primary grid changed")
    require(
        tuple(checkpoints["descriptive"]) == DESCRIPTIVE_UPDATES,
        "Descriptive grid changed",
    )
    require(
        tuple(checkpoints["integrity_only"]) == INTEGRITY_UPDATES,
        "Integrity grid changed",
    )
    require(tuple(measurement["outcomes"]) == OUTCOMES, "Outcome grid changed")
    formulas = measurement["outcome_formulas"]
    require(
        "logit_wolf-logsumexp" in formulas["wolf_margin"] and "+log(9)" in formulas["wolf_margin"],
        "Wolf-margin formula changed",
    )
    require(
        "beta_preference_nll_benefit=-beta(NLL_preference)"
        in formulas["preference_nll_benefit"],
        "Preference-NLL sign changed",
    )
    require(
        "NLL_control-NLL_preference" in formulas["fingerprint_advantage"],
        "Fingerprint formula changed",
    )
    require(
        "No recipient-state sign flip" in formulas["positive_dose_convention"],
        "Dose convention changed",
    )
    require("256 raw preference-NLL" in formulas["cell_storage"], "Cell storage changed")

    support = measurement["support"]
    require(support["layers"] == [8, 9, 10, 11], "Layer support changed")
    require(
        support["module_families"] == ["query_key_value", "dense_4h_to_h"],
        "Module support changed",
    )
    require(support["expected_module_count"] == 8, "Module count changed")
    require(
        support["known_pre_cell_local_ineligibility_from_bound_geometry"]
        == {"56101": [256], "56102": []},
        "Known local eligibility changed",
    )
    require(support["expected_scalar_free_local_n_a_records"] == 8, "Local N/A count changed")

    split = measurement["numeric_split"]
    require(split["block_size"] == 64 and split["block_count"] == 8, "Block grid changed")
    discovery = tuple(split["discovery_blocks"])
    evaluation = tuple(split["evaluation_blocks"])
    require(discovery == (0, 2, 4, 6), "Discovery split changed")
    require(evaluation == (1, 3, 5, 7), "Evaluation split changed")
    require(set(discovery).isdisjoint(evaluation), "Numeric splits overlap")
    require(set(discovery) | set(evaluation) == set(range(8)), "Numeric split incomplete")
    require(split["block_seed"] == 59671, "Block seed changed")
    block_rows = (
        np.random.default_rng(59671)
        .permutation(512)
        .astype(np.int64)
        .reshape(8, 64)
    )
    require(
        int64_sha256(block_rows) == split["all_block_rows_int64_sha256"],
        "All-block row hash changed",
    )
    require(
        int64_sha256(block_rows[list(discovery)].reshape(-1))
        == split["discovery_rows_int64_sha256"],
        "Discovery row hash changed",
    )
    require(
        int64_sha256(block_rows[list(evaluation)].reshape(-1))
        == split["evaluation_rows_int64_sha256"],
        "Evaluation row hash changed",
    )
    require(
        [int64_sha256(row) for row in block_rows]
        == split["per_block_rows_int64_sha256"],
        "Per-block row hashes changed",
    )
    require(split["direction_construction_may_read_evaluation_rows"] is False, "Leak guard removed")
    require(split["outcome_evaluation_may_read_discovery_rows"] is False, "Leak guard removed")
    require(
        "64*19 supervised tokens" in split["factor_normalization"],
        "Token/loss normalization changed",
    )

    routes = config["routes"]
    require(tuple(routes["route_labels"]) == ROUTES, "Route grid changed")
    require("V_D=-4*" in routes["credit_d"]["construction"], "Credit descent route changed")
    require("V_X=-4*" in routes["incoming_x"]["construction"], "Incoming descent route changed")
    require(routes["credit_sham_1"]["seed"] == 60131, "Sham seed changed")
    require(routes["credit_sham_2"]["seed"] == 60132, "Sham seed changed")
    require(routes["rank1_sham_1"]["seed"] == 60133, "Sham seed changed")
    require(routes["rank1_sham_2"]["seed"] == 60134, "Sham seed changed")
    require(
        routes["sham_seed_derivation"].startswith("Every sham draw uses one fixed basis"),
        "Sham namespace rule removed",
    )

    interventions = config["interventions"]
    require(tuple(interventions["signed_doses"]) == DOSES, "Dose curve changed")
    require(
        tuple(interventions["nonzero_signed_doses"]) == NONZERO_DOSES,
        "Nonzero dose curve changed",
    )
    records = expected_records()
    require(interventions["per_state_checkpoint_record_count"] == 33, "Per-state count changed")
    require(interventions["expected_record_count"] == len(records) == 924, "Record count changed")
    require(interventions["no_optimizer_primary"].startswith("Primary cells are direct"), "Primary changed")
    require(
        interventions["response_curve_gate"].startswith("A real route may enter an emergence"),
        "Response-curve gate removed",
    )
    require("28 seed x recipient-state x checkpoint" in interventions["identity"], "Identity grid changed")
    require("Sequential patch accumulation is forbidden" in interventions["identity"], "Restore guard removed")

    analysis = config["frozen_analysis"]
    require(analysis["bootstrap"]["resamples"] == 10000, "Bootstrap count changed")
    require(analysis["bootstrap"]["seed"] == 60141, "Bootstrap seed changed")
    require(analysis["bootstrap"]["no_cross_seed_pooling"] is True, "Seed pooling enabled")
    require(analysis["bootstrap"]["no_cross_recipient_pooling"] is True, "State pooling enabled")
    require(
        analysis["checkpoint_pairs"]["primary_direct"]
        == [[16, 64], [64, 128], [128, 256], [16, 256]],
        "Direct primary interactions changed",
    )
    require(
        analysis["equivalence_margins_per_unit_rho"]
        == {
            "wolf_margin": 0.05,
            "preference_nll_benefit": 0.00075,
            "fingerprint_advantage": 0.001,
        },
        "Equivalence margins changed",
    )
    require(
        analysis["route_contrasts"]["credit_minus_endpoint"]
        == "beta(credit_d)-beta(endpoint_port)",
        "Credit/endpoint equivalence contrast removed",
    )
    require(
        "beta_credit_d,late-beta_credit_d,early"
        in analysis["direct_interactions"]["credit_slope_change"],
        "Direct credit change removed",
    )
    require(
        "beta_endpoint,late-beta_endpoint,early"
        in analysis["direct_interactions"]["endpoint_slope_change"],
        "Direct endpoint change removed",
    )
    require(
        "beta_credit_d-beta_endpoint"
        in analysis["direct_interactions"]["credit_minus_endpoint_change"],
        "Credit/endpoint convergence interaction removed",
    )
    require(
        "Significant late and nonsignificant early" in analysis["direct_interactions"]["rule"],
        "Significance-vs-nonsignificance guard removed",
    )
    require(
        analysis["classification_reporting"].startswith("Report every gate as a nonexclusive"),
        "Nonexclusive gate reporting removed",
    )
    require(
        analysis["equivalence_margin_rationale"].startswith("Empirically calibrated before assay cells"),
        "Equivalence rationale removed",
    )
    require(analysis["descriptive_checkpoints_never_classify"] is True, "Descriptive gate removed")
    require(analysis["integrity_checkpoint_never_selects"] is True, "Integrity gate removed")
    require(analysis["no_route_checkpoint_dose_or_margin_reselection"] is True, "Reselection allowed")

    guards = config["guards"]
    for name in (
        "model_free_prepare_only_until_completion_hashes_bound",
        "no_model_load_during_protocol_validation",
        "no_checkpoint_tensor_load_during_protocol_validation",
        "no_bank_row_load_during_protocol_validation",
        "no_training",
        "no_optimizer_steps",
        "no_tensor_outputs",
    ):
        require(guards[name] is True, f"Safety guard removed: {name}")

    static = config["parents"]["static_exact"]
    require(len(static) >= 20, "Static dependency contract unexpectedly small")
    for name, pair in static.items():
        validate_parent_pair(pair, pending_allowed=False, name=name)
    completion = config["parents"]["completion_exact_before_lock"]
    require(
        set(completion) == {"trace_verifier", "trace_aggregate", "trace_verification"},
        "Completion parents changed",
    )
    for name, pair in completion.items():
        validate_parent_pair(pair, pending_allowed=True, name=name)
        require(pair[1] == PENDING_TRACE or is_hex64(pair[1]), f"Bad trace pending token: {name}")

    protocol_pair = config["implementation_contract"]["protocol_validator"]
    validate_parent_pair(protocol_pair, pending_allowed=False, name="implementation.protocol_validator")
    require(protocol_pair[0] == "scripts/checkpoint_cotangent_port_assay.py", "Validator path changed")
    future_implementation = config["implementation_contract"]["future_exact_before_lock"]
    require(
        set(future_implementation) == {"production_runner", "independent_verifier"},
        "Future implementation contract changed",
    )
    for name, pair in future_implementation.items():
        validate_parent_pair(pair, pending_allowed=True, name=f"implementation.{name}")
        require(pair[1] == PENDING_IMPLEMENTATION or is_hex64(pair[1]), f"Bad implementation pending token: {name}")
    require(
        future_implementation["production_runner"][0]
        == "scripts/checkpoint_cotangent_port_assay_run.py",
        "Production-runner path changed",
    )
    require(
        future_implementation["independent_verifier"][0]
        == "scripts/checkpoint_cotangent_port_assay_verify.py",
        "Verifier path changed",
    )
    semantic = config["implementation_contract"]["scientific_design_semantic_sha256"]
    require(is_hex64(semantic), "Semantic design digest is not bound")
    require(semantic_design_sha256(config) == semantic, "Scientific design semantic digest mismatch")
    require(config["artifacts"]["current_protocol_must_create_none"] is True, "No-write guard removed")


def audit_static_parents(config: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, pair in config["parents"]["static_exact"].items():
        path = ROOT / pair[0]
        exists = path.is_file()
        actual = sha256(path) if exists else None
        rows.append(
            {
                "name": name,
                "path": pair[0],
                "exists": exists,
                "expected_sha256": pair[1],
                "actual_sha256": actual,
                "passed": exists and actual == pair[1],
            }
        )
    protocol_pair = config["implementation_contract"]["protocol_validator"]
    protocol_actual = sha256(SCRIPT_PATH)
    rows.append(
        {
            "name": "implementation.protocol_validator",
            "path": protocol_pair[0],
            "exists": SCRIPT_PATH.is_file(),
            "expected_sha256": protocol_pair[1],
            "actual_sha256": protocol_actual,
            "passed": protocol_actual == protocol_pair[1],
        }
    )
    return {
        "name": "checkpoint-cotangent-port-assay-v1-static-audit",
        "passed": all(row["passed"] for row in rows),
        "rows": rows,
        "model_loaded": False,
        "checkpoint_tensors_loaded": False,
        "bank_rows_loaded": False,
        "mps_used": False,
        "artifacts_written": False,
    }


def readiness(config: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, pair in config["parents"]["completion_exact_before_lock"].items():
        bound = is_hex64(pair[1])
        path = ROOT / pair[0]
        exists = path.is_file()
        actual = sha256(path) if bound and exists else None
        rows.append(
            {
                "name": name,
                "path": pair[0],
                "hash_bound": bound,
                "exists": exists,
                "hash_matches": bound and exists and actual == pair[1],
            }
        )
    implementation_rows: list[dict[str, Any]] = []
    for name, pair in config["implementation_contract"]["future_exact_before_lock"].items():
        bound = is_hex64(pair[1])
        path = ROOT / pair[0]
        exists = path.is_file()
        actual = sha256(path) if bound and exists else None
        implementation_rows.append(
            {
                "name": name,
                "path": pair[0],
                "hash_bound": bound,
                "exists": exists,
                "hash_matches": bound and exists and actual == pair[1],
            }
        )
    static_audit = audit_static_parents(config)
    binding_ready = static_audit["passed"] and all(
        row["hash_matches"] for row in rows + implementation_rows
    )
    return {
        "name": "checkpoint-cotangent-port-assay-v1-readiness",
        "hash_binding_ready": binding_ready,
        "scientific_ready": False,
        "completion_parents": rows,
        "future_implementation": implementation_rows,
        "static_parent_and_validator_audit_passed": static_audit["passed"],
        "semantic_design_sha256": semantic_design_sha256(config),
        "reason_scientific_ready_is_false": (
            "This model-free skeleton only audits immutable hash binding. The future reviewed "
            "preflight must machine-check trace/verifier content, production implementation "
            "contracts, source closure, identities, and scalar schemas before creating a lock."
        ),
        "model_loaded": False,
        "checkpoint_tensors_loaded": False,
        "bank_rows_loaded": False,
        "mps_used": False,
        "artifacts_written": False,
    }


def factor_routes(
    d_preference: np.ndarray,
    d_control: np.ndarray,
    x_preference: np.ndarray,
    x_control: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct full effective-weight D/X Shapley routes for one module."""
    shapes = {
        d_preference.shape[0],
        d_control.shape[0],
        x_preference.shape[0],
        x_control.shape[0],
    }
    require(len(shapes) == 1, "Factor row counts differ")
    require(d_preference.ndim == d_control.ndim == 2, "D factors must be matrices")
    require(x_preference.ndim == x_control.ndim == 2, "X factors must be matrices")
    g_pp = d_preference.T @ x_preference
    g_pc = d_preference.T @ x_control
    g_cp = d_control.T @ x_preference
    g_cc = d_control.T @ x_control
    g_d = 0.5 * ((g_pp - g_cp) + (g_pc - g_cc))
    g_x = 0.5 * ((g_pp - g_pc) + (g_cp - g_cc))
    return {
        "g_pp": g_pp,
        "g_pc": g_pc,
        "g_cp": g_cp,
        "g_cc": g_cc,
        "g_d": g_d,
        "g_x": g_x,
    }


def factorization_error(routes: Mapping[str, np.ndarray]) -> tuple[float, float]:
    observed = routes["g_pp"] - routes["g_cc"]
    reconstructed = routes["g_d"] + routes["g_x"]
    error = float(np.linalg.norm(observed - reconstructed))
    denominator = max(float(np.linalg.norm(observed)), 1e-300)
    return error, error / denominator


def lora_writable_descent(
    loss_gradient: np.ndarray,
    lora_a: np.ndarray,
    lora_b: np.ndarray,
    scaling: float = 2.0,
) -> np.ndarray:
    """Push coherent raw LoRA factor descent into effective-weight tangent."""
    g = np.asarray(loss_gradient, dtype=np.float64)
    a = np.asarray(lora_a, dtype=np.float64)
    b = np.asarray(lora_b, dtype=np.float64)
    require(g.ndim == a.ndim == b.ndim == 2, "LoRA tangent inputs must be matrices")
    require(g.shape == (b.shape[0], a.shape[1]), "Effective-gradient shape mismatch")
    require(a.shape[0] == b.shape[1], "LoRA rank mismatch")
    require(scaling > 0 and math.isfinite(scaling), "Bad LoRA scaling")
    delta_a = -scaling * (b.T @ g)
    delta_b = -scaling * (g @ a.T)
    return scaling * (delta_b @ a + b @ delta_a)


def matched_module_dose(
    route: Mapping[str, np.ndarray],
    module_profile: Mapping[str, float],
    total_reference_norm: float,
    dose: float,
) -> dict[str, np.ndarray]:
    require(set(route) == set(module_profile), "Route/profile module mismatch")
    require(total_reference_norm > 0 and math.isfinite(total_reference_norm), "Bad reference norm")
    result: dict[str, np.ndarray] = {}
    for module, matrix in route.items():
        norm = float(np.linalg.norm(matrix))
        require(norm > 1e-12 and math.isfinite(norm), f"Degenerate route: {module}")
        target = abs(float(dose)) * float(module_profile[module]) * total_reference_norm
        sign = -1.0 if dose < 0 else 1.0
        result[module] = matrix * (sign * target / norm)
    return result


def spectrum_sham(matrix: np.ndarray, seed: int) -> np.ndarray:
    """Independent CPU-float64 random-basis matrix with exact singular values."""
    require(matrix.ndim == 2, "Sham source must be a matrix")
    rows, cols = matrix.shape
    rank = min(rows, cols)
    singular = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    rng = np.random.default_rng(seed)
    left, left_r = np.linalg.qr(rng.standard_normal((rows, rank)), mode="reduced")
    right, right_r = np.linalg.qr(rng.standard_normal((cols, rank)), mode="reduced")
    left_sign = np.where(np.diag(left_r) < 0, -1.0, 1.0)
    right_sign = np.where(np.diag(right_r) < 0, -1.0, 1.0)
    left = left * left_sign
    right = right * right_sign
    return (left * singular) @ right.T


def response_slope(doses: Sequence[float], values: np.ndarray) -> np.ndarray:
    """OLS slope through the shared rho=0 response along the last axis."""
    rho = np.asarray(doses, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    require(y.shape[-1] == rho.size, "Dose/value shape mismatch")
    zero = np.flatnonzero(rho == 0)
    require(zero.size == 1, "Exactly one zero dose is required")
    centered = y - np.take(y, int(zero[0]), axis=-1)[..., None]
    denominator = float(np.dot(rho, rho))
    require(denominator > 0, "Degenerate dose curve")
    return np.sum(centered * rho, axis=-1) / denominator


def odd_secant(doses: Sequence[float], values: np.ndarray, magnitude: float) -> np.ndarray:
    rho = np.asarray(doses, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    positive = np.flatnonzero(np.isclose(rho, magnitude, atol=0, rtol=0))
    negative = np.flatnonzero(np.isclose(rho, -magnitude, atol=0, rtol=0))
    require(positive.size == negative.size == 1, "Missing symmetric dose")
    return (y[..., int(positive[0])] - y[..., int(negative[0])]) / (2 * magnitude)


def direct_interaction(
    late_route_a: np.ndarray,
    late_route_b: np.ndarray,
    early_route_a: np.ndarray,
    early_route_b: np.ndarray,
) -> np.ndarray:
    return (late_route_a - late_route_b) - (early_route_a - early_route_b)


def equivalent(interval_low: float, interval_high: float, margin: float) -> bool:
    require(margin > 0, "Equivalence margin must be positive")
    return interval_low > -margin and interval_high < margin


def max_t_intervals(
    point: np.ndarray,
    bootstrap: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Simultaneous max-|t| intervals for a frozen vector family."""
    point = np.asarray(point, dtype=np.float64)
    samples = np.asarray(bootstrap, dtype=np.float64)
    require(samples.ndim == 2 and samples.shape[1] == point.size, "Bad bootstrap family")
    standard_error = samples.std(axis=0, ddof=1)
    require(np.all(standard_error >= 1e-12), "Degenerate bootstrap standard error")
    centered = samples - samples.mean(axis=0, keepdims=True)
    maximum = np.max(np.abs(centered / standard_error), axis=1)
    critical = float(np.quantile(maximum, 1.0 - alpha))
    return point - critical * standard_error, point + critical * standard_error, critical


def self_test(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    rng = np.random.default_rng(60151)

    factors = factor_routes(
        rng.standard_normal((23, 7)),
        rng.standard_normal((23, 7)),
        rng.standard_normal((23, 5)),
        rng.standard_normal((23, 5)),
    )
    absolute_error, relative_error = factorization_error(factors)
    require(absolute_error <= 1e-11 and relative_error <= 1e-11, "Factor identity failed")

    lora_a = rng.standard_normal((3, 5))
    lora_b = rng.standard_normal((7, 3))
    loss_gradient = rng.standard_normal((7, 5))
    writable = lora_writable_descent(loss_gradient, lora_a, lora_b, 2.0)
    low_rank_left = np.concatenate((loss_gradient @ lora_a.T, lora_b), axis=1)
    low_rank_right = np.concatenate((lora_a, lora_b.T @ loss_gradient), axis=0)
    low_rank_writable = -4.0 * (low_rank_left @ low_rank_right)
    low_rank_error = float(np.max(np.abs(writable - low_rank_writable)))
    require(low_rank_error <= 1e-12, "LoRA low-rank factorization failed")
    delta_a = -2.0 * (lora_b.T @ loss_gradient)
    delta_b = -2.0 * (loss_gradient @ lora_a.T)
    epsilon = 1e-6
    forward = 2.0 * ((lora_b + epsilon * delta_b) @ (lora_a + epsilon * delta_a))
    backward = 2.0 * ((lora_b - epsilon * delta_b) @ (lora_a - epsilon * delta_a))
    finite_difference = (forward - backward) / (2.0 * epsilon)
    writable_error = float(np.max(np.abs(writable - finite_difference)))
    require(writable_error <= 1e-8, "LoRA writable-route differential failed")
    descent_dot = float(np.sum(loss_gradient * writable))
    require(descent_dot < 0, "LoRA writable route lacks descent sign")

    route = {
        "m0": rng.standard_normal((7, 5)),
        "m1": rng.standard_normal((6, 4)),
    }
    profile = {"m0": 0.6, "m1": 0.8}
    require(abs(sum(value * value for value in profile.values()) - 1.0) < 1e-15, "Bad profile")
    dosed = matched_module_dose(route, profile, 3.0, -0.25)
    dose_errors = {
        module: abs(float(np.linalg.norm(dosed[module])) - 0.25 * profile[module] * 3.0)
        for module in route
    }
    require(max(dose_errors.values()) <= 1e-12, "Matched-dose identity failed")

    sham_source = rng.standard_normal((9, 6))
    sham_seed = derived_sham_seed(
        "credit_sham_1", 60131, 56101, "layer_9.query_key_value"
    )
    require(
        sham_seed
        == derived_sham_seed(
            "credit_sham_1", 60131, 56101, "layer_9.query_key_value"
        ),
        "Sham seed is nondeterministic",
    )
    require(
        sham_seed
        != derived_sham_seed(
            "credit_sham_2", 60132, 56101, "layer_9.query_key_value"
        ),
        "Sham namespaces collided",
    )
    rank1_seed = derived_sham_seed(
        "rank1_sham_1", 60133, 56101, "layer_9.query_key_value"
    )
    require(
        rank1_seed
        == derived_sham_seed(
            "rank1_sham_1", 60133, 56101, "layer_9.query_key_value"
        ),
        "Rank-one sham seed is nondeterministic",
    )
    sham = spectrum_sham(sham_source, sham_seed)
    source_singular = np.linalg.svd(sham_source, full_matrices=False, compute_uv=False)
    sham_singular = np.linalg.svd(sham, full_matrices=False, compute_uv=False)
    spectrum_error = float(np.max(np.abs(source_singular - sham_singular)))
    require(spectrum_error <= 1e-10, "Sham spectrum mismatch")
    require(not np.allclose(sham_source, sham), "Sham reused the real basis")

    rho = np.asarray(DOSES)
    units = np.arange(1, 7, dtype=np.float64)[:, None]
    slopes = units / 10.0
    curve = 2.0 + slopes * rho + 0.3 * rho * rho
    fitted = response_slope(DOSES, curve)
    require(float(np.max(np.abs(fitted - slopes[:, 0]))) <= 1e-14, "Slope recovery failed")
    secant_025 = odd_secant(DOSES, curve, 0.25)
    secant_050 = odd_secant(DOSES, curve, 0.5)
    require(float(np.max(np.abs(secant_025 - slopes[:, 0]))) <= 1e-14, "Odd secant failed")
    require(float(np.max(np.abs(secant_050 - slopes[:, 0]))) <= 1e-14, "Odd secant failed")

    interaction = direct_interaction(
        np.array([4.0, 3.0]),
        np.array([1.0, 1.0]),
        np.array([2.0, 2.0]),
        np.array([1.0, 1.0]),
    )
    require(np.array_equal(interaction, np.array([2.0, 1.0])), "Interaction failed")
    require(equivalent(-0.4, 0.4, 0.5), "Equivalence acceptance failed")
    require(not equivalent(-0.6, 0.4, 0.5), "Equivalence rejection failed")

    point = np.array([0.2, -0.1, 0.4])
    boot = point + rng.standard_normal((2000, 3)) * np.array([0.1, 0.2, 0.15])
    low, high, critical = max_t_intervals(point, boot)
    require(np.all(low < point) and np.all(high > point), "Max-t intervals failed")
    require(critical > 1.0, "Implausible max-t critical value")

    records = expected_records()
    keys = [record.key for record in records]
    require(len(records) == len(set(keys)) == 924, "Record grid collision")
    native = [record for record in records if record.native]
    require(len(native) == len(SEEDS) * len(RECIPIENT_STATES) * len(UPDATES) == 28, "Native grid failed")
    local_n_a = [record for record in records if record.not_applicable_expected]
    require(len(local_n_a) == 8, "Frozen local N/A grid failed")
    require(
        all(
            record.route == "local_port"
            and record.seed == 56101
            and record.optimizer_update == 256
            and not record.native
            for record in local_n_a
        ),
        "Unexpected local N/A key",
    )

    future_paths = [
        ROOT / config["artifacts"][name]
        for name in (
            "future_production_runner",
            "future_independent_verifier",
            "future_root",
            "future_lock",
            "future_preflight",
            "future_aggregate_json",
            "future_aggregate_markdown",
        )
    ]
    return {
        "name": "checkpoint-cotangent-port-assay-v1-self-test",
        "passed": True,
        "factorization_absolute_error": absolute_error,
        "factorization_relative_error": relative_error,
        "lora_writable_finite_difference_max_error": writable_error,
        "lora_writable_low_rank_max_error": low_rank_error,
        "lora_writable_loss_gradient_dot": descent_dot,
        "maximum_matched_module_dose_error": max(dose_errors.values()),
        "maximum_sham_singular_spectrum_error": spectrum_error,
        "sham_seed_namespace_smoke": sham_seed,
        "rank1_sham_seed_namespace_smoke": rank1_seed,
        "expected_record_count": len(records),
        "native_record_count": len(native),
        "expected_scalar_free_local_n_a_record_count": len(local_n_a),
        "max_t_critical_smoke": critical,
        "completion_hashes_bound": all(
            is_hex64(pair[1])
            for pair in config["parents"]["completion_exact_before_lock"].values()
        ),
        "future_implementation_hashes_bound": all(
            is_hex64(pair[1])
            for pair in config["implementation_contract"]["future_exact_before_lock"].values()
        ),
        "future_artifacts_currently_exist": [str(path.relative_to(ROOT)) for path in future_paths if path.exists()],
        "model_loaded": False,
        "checkpoint_tensors_loaded": False,
        "bank_rows_loaded": False,
        "mps_used": False,
        "training_used": False,
        "optimizer_steps": 0,
        "artifacts_written": False,
        "production_runner_imported_for_verification": False,
        "semantic_design_sha256": semantic_design_sha256(config),
    }


def compact_records(records: Iterable[FutureRecord]) -> dict[str, Any]:
    rows = tuple(records)
    return {
        "name": "checkpoint-cotangent-port-assay-v1-future-record-grid",
        "count": len(rows),
        "first": rows[0].json(),
        "last": rows[-1].json(),
        "route_counts": {
            route: sum(record.route == route for record in rows)
            for route in ("native",) + ROUTES
        },
        "model_loaded": False,
        "checkpoint_tensors_loaded": False,
        "bank_rows_loaded": False,
        "mps_used": False,
        "artifacts_written": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("validate", "audit-static", "readiness", "expected-records", "self-test"),
    )
    args = parser.parse_args()
    config = load_json(CONFIG_PATH)
    validate_config(config)

    if args.command == "validate":
        value: dict[str, Any] = {
            "name": "checkpoint-cotangent-port-assay-v1-validation",
            "passed": True,
            "config": str(CONFIG_PATH.relative_to(ROOT)),
            "config_sha256": sha256(CONFIG_PATH),
            "runner": str(SCRIPT_PATH.relative_to(ROOT)),
            "runner_sha256": sha256(SCRIPT_PATH),
            "model_loaded": False,
            "checkpoint_tensors_loaded": False,
            "bank_rows_loaded": False,
            "mps_used": False,
            "artifacts_written": False,
        }
    elif args.command == "audit-static":
        value = audit_static_parents(config)
    elif args.command == "readiness":
        value = readiness(config)
    elif args.command == "expected-records":
        value = compact_records(expected_records())
    else:
        value = self_test(config)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
