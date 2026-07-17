"""Clean-room scalar verifier for effective-weight checkpoint trace v1.

This module intentionally does not import ``effective_weight_checkpoint_trace``
or any model/experiment module.  It reconstructs the immutable input payload,
regenerates geometry directly from lock-pinned snapshots on CPU without loading
a language model, validates every scalar cell, independently recomputes the
bootstrap tables and classification, and requires exact aggregate equality
apart from the production completion timestamp.

``self-test`` is safe before the causal campaign finishes.  ``verify`` is only
available once all 252 frozen records and the production aggregate exist.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = Path(__file__).resolve()
CONFIG_PATH = ROOT / "configs/effective_weight_checkpoint_trace_v1.json"
RUNNER_PATH = ROOT / "scripts/effective_weight_checkpoint_trace.py"
WORK = ROOT / "runs/effective_weight_checkpoint_trace_v1"
LOCK_PATH = WORK / "trace_lock.json"
PREFLIGHT_PATH = WORK / "preflight.json"
GEOMETRY_PATH = WORK / "geometry.json"
OUT_JSON = ROOT / "runs/effective_weight_checkpoint_trace_v1.json"
OUT_MD = ROOT / "runs/effective_weight_checkpoint_trace_v1.md"
OUT_VERIFY = ROOT / "runs/effective_weight_checkpoint_trace_v1_verify.json"

SEEDS = (56101, 56102)
CONDITIONS = ("preference", "control")
UPDATES = (8, 16, 32, 64, 128, 256, 512)
PRIMARY = (16, 64, 128, 256)
DESCRIPTIVE = (8, 32)
INTEGRITY = (512,)
RANKS = (1, 2, 4, 8)
PARENT_NAMES = (
    "ds2_factorial_config", "ds2_factorial_runner",
    "ds2_stage_a_runner_lock", "ds2_factorial_runner_lock",
    "endpoint_config", "endpoint_runner", "cross_gradient_config",
    "cross_gradient_runner", "cross_gradient_runner_lock",
    "dynamics_config", "dynamics_runner", "component_config",
    "component_runner", "component_preflight", "component_runner_lock",
    "component_fresh_numeric_manifest", "data_py", "evaluate_py",
    "modeling_py", "optim_py", "train_py",
)


@dataclass(frozen=True)
class PatchSpec:
    label: str
    template: str
    kind: str
    alpha: float

    def record(self) -> dict[str, Any]:
        return {"label": self.label, "template": self.template,
                "kind": self.kind, "rank": 1, "alpha": self.alpha}


PATCH_SPECS = (
    PatchSpec("native", "none", "native", 0.0),
    PatchSpec("endpoint_real_a025", "endpoint", "real", .25),
    PatchSpec("endpoint_sham_a025", "endpoint", "sham", .25),
    PatchSpec("endpoint_real_a050", "endpoint", "real", .50),
    PatchSpec("endpoint_sham_a050", "endpoint", "sham", .50),
    PatchSpec("endpoint_real_a100", "endpoint", "real", 1.0),
    PatchSpec("endpoint_sham_a100", "endpoint", "sham", 1.0),
    PatchSpec("local_real_a100", "local", "real", 1.0),
    PatchSpec("local_sham_a100", "local", "sham", 1.0),
)
LABELS = tuple(spec.label for spec in PATCH_SPECS)
NONIDENTIFIABLE_REASON = (
    "local_rank1_nonidentifiable_all_eight_module_s1_s2_gap_requirement_failed"
)


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


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rel(path: Path) -> str:
    resolved = path.resolve()
    require(resolved.is_relative_to(ROOT.resolve()), f"Artifact escapes repository: {path}")
    return str(resolved.relative_to(ROOT.resolve()))


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing artifact: {path}")
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def parse_timestamp(value: Any, name: str) -> None:
    require(isinstance(value, str) and value, f"Missing timestamp: {name}")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError(f"Malformed timestamp: {name}") from error


def finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite(row) for row in value)
    if isinstance(value, dict):
        return all(finite(row) for row in value.values())
    return False


def close(observed: float, expected: float, name: str, tolerance: float = 1e-12) -> None:
    require(math.isclose(float(observed), float(expected), rel_tol=tolerance,
                         abs_tol=tolerance), f"Derived scalar mismatch: {name}")


def expected_cells() -> list[tuple[int, str, int, PatchSpec, Path]]:
    return [
        (seed, condition, update, spec,
         WORK / "cells" / f"seed_{seed}" / condition / f"u{update:04d}" /
         spec.label / "cell.json")
        for seed in SEEDS for condition in CONDITIONS for update in UPDATES
        for spec in PATCH_SPECS
    ]


def validate_config(config: dict[str, Any]) -> None:
    require(set(config) == {"name", "question", "status", "frozen_at", "measurement",
                            "interventions", "frozen_analysis", "scope", "guards",
                            "parents", "artifacts"}, "Trace config top-level schema changed")
    require(config.get("name") == "effective-weight-checkpoint-trace-v1", "Wrong trace config")
    measurement = config.get("measurement", {})
    require(tuple(measurement.get("seeds", [])) == SEEDS, "Seed grid changed")
    require(tuple(measurement.get("source_conditions", [])) == CONDITIONS,
            "Condition grid changed")
    require(measurement.get("receiver") == "ds2" and
            measurement.get("receiver_name") == "data_seed2", "Receiver changed")
    checkpoints = measurement.get("checkpoints", {})
    require(tuple(checkpoints.get("all", [])) == UPDATES and
            tuple(checkpoints.get("primary", [])) == PRIMARY and
            tuple(checkpoints.get("descriptive", [])) == DESCRIPTIVE and
            tuple(checkpoints.get("integrity_only", [])) == INTEGRITY,
            "Checkpoint grid changed")
    require(tuple(checkpoints.get("parent_tensor_semantic_exact", [])) ==
            (16, 64, 128, 256, 512), "Exact-source grid changed")
    require(tuple(checkpoints.get("replay_self_validated_only", [])) == DESCRIPTIVE,
            "Descriptive provenance grid changed")
    support = measurement.get("support")
    require(support == {"layers": [8, 9, 10, 11],
                        "module_families": ["query_key_value", "dense_4h_to_h"],
                        "expected_module_count": 8, "rank": 1,
                        "minimum_relative_singular_gap": 0.01}, "Support changed")
    geometry = measurement.get("geometry", {})
    require(tuple(geometry.get("rank_prefixes", [])) == RANKS and
            geometry.get("sham_seed") == 59711, "Geometry freeze changed")
    readouts = measurement.get("readouts", {})
    require(readouts.get("behavior_prompt_count") == 60 and
            readouts.get("behavior_cluster_count") == 12 and
            readouts.get("numeric_rows_per_condition") == 512 and
            readouts.get("numeric_block_count") == 8 and
            readouts.get("outcomes") ==
            ["wolf_margin", "preference_nll", "fingerprint_advantage"],
            "Readout freeze changed")
    interventions = config.get("interventions", {})
    require(interventions.get("per_state_cells") == list(LABELS) and
            interventions.get("cell_count") == 252, "Cell freeze changed")
    require(interventions.get("endpoint_template", {}).get("alphas") == [.25, .5, 1.0]
            and interventions.get("local_template", {}).get("alphas") == [1.0],
            "Intervention doses changed")
    require(interventions.get("sham", {}).get("endpoint_template_alphas") ==
            [.25, .5, 1.0] and
            interventions.get("sham", {}).get("local_template_alphas") == [1.0],
            "Matched shams changed")
    analysis = config.get("frozen_analysis", {})
    require(analysis.get("benefit_sign") == {"control": 1.0, "preference": -1.0},
            "Benefit sign changed")
    bootstrap = analysis.get("bootstrap", {})
    require(bootstrap.get("resamples") == 10000 and bootstrap.get("seed") == 59721,
            "Bootstrap freeze changed")
    require(analysis.get("descriptive_checkpoints_are_never_classification_inputs") is True
            and analysis.get("no_checkpoint_or_scale_reselection") is True and
            analysis.get("no_cross_seed_pooling_for_replication") is True,
            "Analysis guard changed")
    require(config.get("scope", {}).get("no_update0_claim", "").startswith(
        "There is no saved ds2 update-0"), "Update-0 limitation removed")
    require("component scientific cell outcomes" in
            config.get("scope", {}).get("readout_reuse", ""),
            "Component-outcome quarantine removed")
    guards = config.get("guards", {})
    require(guards.get("no_training") is True and
            guards.get("no_optimizer_steps") is True and
            guards.get("no_tensor_outputs") is True and
            guards.get("maximum_result_bytes_per_cell") == 2097152,
            "Runtime guards changed")
    artifacts = config.get("artifacts", {})
    require(artifacts == {
        "root": "runs/effective_weight_checkpoint_trace_v1",
        "runner": "scripts/effective_weight_checkpoint_trace.py",
        "trace_lock": "runs/effective_weight_checkpoint_trace_v1/trace_lock.json",
        "preflight": "runs/effective_weight_checkpoint_trace_v1/preflight.json",
        "geometry": "runs/effective_weight_checkpoint_trace_v1/geometry.json",
        "cells": "runs/effective_weight_checkpoint_trace_v1/cells",
        "identity": "runs/effective_weight_checkpoint_trace_v1/identity",
        "aggregate_json": "runs/effective_weight_checkpoint_trace_v1.json",
        "aggregate_markdown": "runs/effective_weight_checkpoint_trace_v1.md",
    }, "Artifact paths changed")
    parents = config.get("parents", {})
    require(set(parents) == set(PARENT_NAMES) | {"sources"}, "Parent inventory changed")
    require(set(parents.get("sources", {})) ==
            {f"{seed}/{condition}" for seed in SEEDS for condition in CONDITIONS},
            "Source inventory changed")
    require(len(expected_cells()) == 252 and len({(a, b, c, d.label)
            for a, b, c, d, _ in expected_cells()}) == 252, "Cell grid is not unique")


def checked_pair(pair: Any, name: str) -> Path:
    require(isinstance(pair, list) and len(pair) == 2 and
            all(isinstance(row, str) for row in pair), f"Malformed parent pair: {name}")
    path = ROOT / pair[0]
    require(artifact(path)["sha256"] == pair[1], f"Parent changed: {name}")
    return path


def source_inventory(config: dict[str, Any], seed: int, condition: str) -> dict[str, Any]:
    name = f"{seed}/{condition}"
    source_path = checked_pair(config["parents"]["sources"][name], f"source {name}")
    source = load_json(source_path)
    require(source.get("name") == "ds2-adam-source-factorial-v1-replay" and
            source.get("receiver") == "data_seed2" and source.get("seed") == seed and
            source.get("condition") == condition and
            source.get("u512_semantic_exact") is True and
            source.get("optimizer_updates") == 512, f"Source identity failed: {name}")
    result_record = source.get("artifacts", {}).get("result", {})
    require(set(result_record) >= {"path", "bytes", "sha256"},
            f"Malformed replay result artifact: {name}")
    result_path = ROOT / result_record["path"]
    observed_result = artifact(result_path)
    require(observed_result["bytes"] == result_record["bytes"] and
            observed_result["sha256"] == result_record["sha256"],
            f"Replay result changed: {name}")
    result = load_json(result_path)
    snapshots: dict[str, Any] = {}
    for update in UPDATES:
        row = result.get("snapshots", {}).get(str(update))
        require(isinstance(row, dict), f"Missing snapshot {name}/u{update}")
        state = row.get("artifact", {})
        require(set(state) >= {"path", "bytes", "sha256"},
                f"Malformed snapshot artifact {name}/u{update}")
        observed_state = artifact(ROOT / state["path"])
        require(observed_state["bytes"] == state["bytes"] and
                observed_state["sha256"] == state["sha256"],
                f"Snapshot changed {name}/u{update}")
        exact = update not in DESCRIPTIVE
        semantic = row.get("semantic_state_hashes", {})
        require(set(semantic) == {"adam_exp_avg_semantic_sha256",
                                  "adam_exp_avg_sq_semantic_sha256",
                                  "adam_steps_exact", "lora_semantic_sha256",
                                  "optimizer_update"} and
                semantic.get("optimizer_update") == update and
                semantic.get("adam_steps_exact") is True,
                f"Snapshot semantic schema mismatch {name}/u{update}")
        require(row.get("self_validation_passed") is True and
                row.get("parent_semantic_exact") is exact,
                f"Snapshot provenance mismatch {name}/u{update}")
        summaries = row.get("summaries", {})
        require(summaries.get("lora_tensor_count") == 96 and
                summaries.get("trainable_parameters") == 1179648,
                f"Snapshot summary mismatch {name}/u{update}")
        snapshots[str(update)] = {
            "artifact": observed_state,
            "self_validation_passed": True,
            "parent_semantic_exact": exact,
            "semantic_state_hashes": semantic,
        }
    return {"source_cell": artifact(source_path), "result": observed_result,
            "snapshots": snapshots}


def rebuild_frozen(config: dict[str, Any]) -> dict[str, Any]:
    parents = {name: artifact(checked_pair(config["parents"][name], name))
               for name in PARENT_NAMES}
    component_preflight = load_json(ROOT / parents["component_preflight"]["path"])
    require(component_preflight.get("passed") is True and
            component_preflight.get("expected_cell_count") == 432,
            "Component preflight invalid")
    manifest = load_json(ROOT / parents["component_fresh_numeric_manifest"]["path"])
    require(manifest.get("bank", {}).get("paired_prompt_sha256") ==
            component_preflight.get("fresh_numeric_prompt_sha256"),
            "Fresh-bank manifest mismatch")
    return {
        "name": "effective-weight-checkpoint-trace-v1-frozen-inputs",
        "config": artifact(CONFIG_PATH),
        "mps_runner": artifact(RUNNER_PATH),
        "parents": parents,
        "sources": {f"{seed}/{condition}": source_inventory(config, seed, condition)
                    for seed in SEEDS for condition in CONDITIONS},
        "component_readout_metadata": {
            "fresh_behavior_prompt_sha256":
                component_preflight["fresh_behavior_prompt_sha256"],
            "fresh_numeric_prompt_sha256":
                component_preflight["fresh_numeric_prompt_sha256"],
            "fresh_numeric_rows_per_condition":
                component_preflight["fresh_numeric_rows_per_condition"],
            "component_outcomes_bound": False,
        },
        "expected_cells": [
            {"seed": seed, "condition": condition, "optimizer_update": update,
             **spec.record()}
            for seed, condition, update, spec, _ in expected_cells()
        ],
        "no_update0_claim": config["scope"]["no_update0_claim"],
        "no_tensor_outputs": True,
    }


def validate_lock(config: dict[str, Any]) -> dict[str, Any]:
    lock = load_json(LOCK_PATH)
    require(set(lock) == {"created_at", "frozen"}, "Trace lock schema changed")
    parse_timestamp(lock.get("created_at"), "trace lock")
    require(lock.get("frozen") == rebuild_frozen(config),
            "Trace lock differs from independent immutable reconstruction")
    return lock


def component_outcome_contract(lock: dict[str, Any]) -> dict[str, Any]:
    aggregate_path = ROOT / "runs/effective_weight_component_dissection_v1.json"
    aggregate_value = load_json(aggregate_path)
    require(aggregate_value.get("name") == "effective-weight-component-dissection-v1"
            and aggregate_value.get("cell_count") == 432 and
            aggregate_value.get("no_tensor_outputs") is True,
            "Component aggregate is incomplete")
    parents = lock["frozen"]["parents"]
    require(aggregate_value.get("config_sha256") == parents["component_config"]["sha256"]
            and aggregate_value.get("runner_sha256") ==
            parents["component_runner"]["sha256"] and
            aggregate_value.get("runner_lock_sha256") ==
            parents["component_runner_lock"]["sha256"],
            "Component aggregate dependency mismatch")
    return {
        "aggregate": artifact(aggregate_path),
        "config_sha256": aggregate_value["config_sha256"],
        "runner_sha256": aggregate_value["runner_sha256"],
        "runner_lock_sha256": aggregate_value["runner_lock_sha256"],
        "primary_classification": aggregate_value.get("primary", {}).get("classification"),
        "bound_after_component_completion": True,
    }


def validate_preflight(lock: dict[str, Any]) -> dict[str, Any]:
    preflight = load_json(PREFLIGHT_PATH)
    require(set(preflight) == {"name", "created_at", "passed", "frozen",
                               "model_loaded", "checkpoint_tensors_loaded",
                               "banks_loaded"}, "Trace preflight schema changed")
    parse_timestamp(preflight.get("created_at"), "trace preflight")
    frozen = {
        "trace_lock": artifact(LOCK_PATH),
        "component_outcome_contract": component_outcome_contract(lock),
        "expected_cell_count": 252,
        "runner": artifact(RUNNER_PATH),
        "no_training": True,
        "no_optimizer_steps": True,
        "no_tensor_outputs": True,
    }
    require(preflight.get("name") == "effective-weight-checkpoint-trace-v1-preflight"
            and preflight.get("passed") is True and preflight.get("frozen") == frozen
            and preflight.get("model_loaded") is False
            and preflight.get("checkpoint_tensors_loaded") is False
            and preflight.get("banks_loaded") is False,
            "Trace preflight mismatch")
    return preflight


AUDIT_KEYS = {"core_relative_reconstruction_error",
              "float32_patch_relative_reconstruction_error", "frobenius_norm",
              "relative_boundary_gaps"}
MODULE_KEYS = {"left_principal_angles_radians", "right_principal_angles_radians",
               "signed_rank1_loading", "endpoint_captured_energy",
               "local_frobenius_squared", "two_sided_transport_diagonal",
               "two_sided_transport_offdiagonal_relative_frobenius"}
SUMMARY_KEYS = {"module_balanced_signed_rank1_loading",
                "module_balanced_endpoint_captured_energy",
                "module_balanced_left_angle_radians",
                "module_balanced_right_angle_radians",
                "energy_weighted_signed_rank1_loading",
                "energy_weighted_endpoint_captured_energy",
                "energy_weighted_left_angle_radians",
                "energy_weighted_right_angle_radians", "sham_mean_captured_energy",
                "captured_energy_minus_sham", "sham_mean_left_angle_radians",
                "sham_mean_right_angle_radians", "boundary_gap_at_least_1pct",
                "rank1_angle_classification_eligible"}


@dataclass(frozen=True)
class SVDComponent:
    u: torch.Tensor
    s: torch.Tensor
    v: torch.Tensor


def load_snapshot(lock: dict[str, Any], seed: int, condition: str,
                  update: int) -> dict[str, Any]:
    record = lock["frozen"]["sources"][f"{seed}/{condition}"]["snapshots"][str(update)]
    path = ROOT / record["artifact"]["path"]
    require(artifact(path) == record["artifact"],
            f"Snapshot changed before geometry replay: {seed}/{condition}/u{update}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict) and payload.get("receiver") == "data_seed2" and
            payload.get("seed") == seed and payload.get("condition") == condition and
            payload.get("optimizer_update") == update and
            payload.get("analysis_snapshot_only") is True,
            f"Snapshot payload identity mismatch: {seed}/{condition}/u{update}")
    return payload


def module_inventory(payload: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    lora = payload.get("lora")
    require(isinstance(lora, list) and len(lora) == 96, "LoRA inventory changed")
    tensors: dict[str, torch.Tensor] = {}
    for row in lora:
        require(isinstance(row, dict) and set(row) == {"name", "tensor"} and
                isinstance(row["name"], str) and torch.is_tensor(row["tensor"]),
                "Malformed LoRA tensor record")
        require(row["name"] not in tensors, "Duplicate LoRA tensor name")
        tensors[row["name"]] = row["tensor"].float().cpu().contiguous()
    suffix = ".lora_A.default.weight"
    result: dict[str, dict[str, torch.Tensor]] = {}
    for name, value in tensors.items():
        if not name.endswith(suffix):
            continue
        module = name[:-len(suffix)]
        b_name = module + ".lora_B.default.weight"
        require(b_name in tensors, f"Missing LoRA B factor: {module}")
        result[module] = {"a": value, "b": tensors[b_name]}
    require(len(result) == 48, "Effective module inventory changed")
    return result


def selected_modules(inventory: dict[str, Any]) -> list[str]:
    result = []
    for name in inventory:
        match = re.search(r"gpt_neox\.layers\.(\d+)\.", name)
        require(match is not None, f"Missing layer coordinate: {name}")
        layer = int(match.group(1))
        family = name.rsplit(".", 1)[-1]
        if layer in (8, 9, 10, 11) and family in ("query_key_value", "dense_4h_to_h"):
            result.append(name)
    result.sort()
    require(len(result) == 8, "Selected geometry support changed")
    return result


def compact_svd(preference: dict[str, torch.Tensor],
                control: dict[str, torch.Tensor], scale: float) \
        -> tuple[SVDComponent, dict[str, float]]:
    bp, ap = preference["b"].double(), preference["a"].double()
    bc, ac = control["b"].double(), control["a"].double()
    left = torch.cat((scale * bp, -scale * bc), dim=1)
    right = torch.cat((ap, ac), dim=0)
    q_left, r_left = torch.linalg.qr(left, mode="reduced")
    q_right, r_right = torch.linalg.qr(right.T, mode="reduced")
    core = r_left @ r_right.T
    u_core, singular, vh_core = torch.linalg.svd(core, full_matrices=False)
    reconstructed = u_core @ torch.diag(singular) @ vh_core
    denominator = max(float(torch.linalg.vector_norm(core)),
                      torch.finfo(torch.float64).tiny)
    core_error = float(torch.linalg.vector_norm(core - reconstructed)) / denominator
    u = (q_left @ u_core).float().contiguous()
    v = (q_right @ vh_core.T).float().contiguous()
    s = singular.float().contiguous()
    dense = left @ right
    float32_reconstructed = u.double() @ torch.diag(s.double()) @ v.double().T
    dense_denominator = max(float(torch.linalg.vector_norm(dense)),
                            torch.finfo(torch.float64).tiny)
    float32_error = (
        float(torch.linalg.vector_norm(dense - float32_reconstructed)) /
        dense_denominator
    )
    return SVDComponent(u, s, v), {
        "core_relative_reconstruction_error": core_error,
        "float32_patch_relative_reconstruction_error": float32_error,
        "frobenius_norm": float(torch.linalg.vector_norm(singular)),
    }


def full_components(preference_payload: dict[str, Any], control_payload: dict[str, Any],
                    scale: float) -> tuple[list[str], dict[str, SVDComponent],
                                           dict[str, dict[str, Any]], bool]:
    preference, control = (module_inventory(preference_payload),
                           module_inventory(control_payload))
    require(set(preference) == set(control), "Preference/control module mismatch")
    names = selected_modules(preference)
    full: dict[str, SVDComponent] = {}
    audits: dict[str, dict[str, Any]] = {}
    for name in names:
        value, audit = compact_svd(preference[name], control[name], scale)
        require(value.s.numel() >= 8 and float(value.s[0]) > 0,
                f"Degenerate effective-weight contrast: {name}")
        gaps = {str(rank): float((value.s[rank - 1].double() -
                                  value.s[rank].double()) /
                                 value.s[rank - 1].double()) for rank in RANKS}
        full[name] = value
        audits[name] = {"relative_boundary_gaps": gaps, **audit}
    eligible = all(audit["relative_boundary_gaps"]["1"] >= .01
                   for audit in audits.values())
    return names, full, audits, eligible


def sham_component(real: SVDComponent, generator: torch.Generator) -> SVDComponent:
    u_raw = torch.randn(real.u.shape, generator=generator, dtype=torch.float64,
                        device="cpu")
    v_raw = torch.randn(real.v.shape, generator=generator, dtype=torch.float64,
                        device="cpu")
    u, _ = torch.linalg.qr(u_raw, mode="reduced")
    v, _ = torch.linalg.qr(v_raw, mode="reduced")
    return SVDComponent(u.float().contiguous(), real.s.clone(), v.float().contiguous())


def geometry_one(endpoint_full: dict[str, SVDComponent],
                 local_full: dict[str, SVDComponent],
                 local_audits: dict[str, Any], eligible: bool, seed: int,
                 update: int) -> dict[str, Any]:
    prefix_rows: dict[str, Any] = {}
    module_rows: dict[str, dict[str, Any]] = {}
    generator = torch.Generator(device="cpu").manual_seed(59711 + seed * 10000 + update)
    for rank in RANKS:
        rows: list[dict[str, Any]] = []
        sham_energy: list[float] = []
        sham_left_angle: list[float] = []
        sham_right_angle: list[float] = []
        for name in sorted(endpoint_full):
            endpoint, local = endpoint_full[name], local_full[name]
            ue, ve = endpoint.u[:, :rank].double(), endpoint.v[:, :rank].double()
            ul_angle, vl_angle = local.u[:, :rank].double(), local.v[:, :rank].double()
            ul_full, vl_full, singular = (local.u.double(), local.v.double(),
                                          local.s.double())
            local_norm = float(torch.linalg.vector_norm(singular))
            local_energy = float(torch.sum(singular.square()))
            left_sv = torch.linalg.svdvals(ue.T @ ul_angle).clamp(-1, 1)
            right_sv = torch.linalg.svdvals(ve.T @ vl_angle).clamp(-1, 1)
            core = ue.T @ ul_full @ torch.diag(singular) @ (vl_full.T @ ve)
            captured = float(torch.sum(core.square()) / torch.sum(singular.square()))
            loading = float((endpoint.u[:, :1].double().T @ ul_full @
                             torch.diag(singular) @
                             (vl_full.T @ endpoint.v[:, :1].double())).item() /
                            local_norm)
            leakage = float(torch.linalg.vector_norm(
                core - torch.diag(torch.diagonal(core))) /
                max(torch.linalg.vector_norm(core), torch.finfo(torch.float64).tiny))
            sham = sham_component(SVDComponent(endpoint.u[:, :rank],
                                               endpoint.s[:rank],
                                               endpoint.v[:, :rank]), generator)
            su, sv = sham.u.double(), sham.v.double()
            score = su.T @ ul_full @ torch.diag(singular) @ (vl_full.T @ sv)
            sham_energy.append(float(torch.sum(score.square()) /
                                     torch.sum(singular.square())))
            sham_left_angle.append(float(torch.mean(torch.arccos(
                torch.linalg.svdvals(su.T @ ul_angle).clamp(-1, 1)))))
            sham_right_angle.append(float(torch.mean(torch.arccos(
                torch.linalg.svdvals(sv.T @ vl_angle).clamp(-1, 1)))))
            item = {
                "left_principal_angles_radians":
                    [float(value) for value in torch.arccos(left_sv)],
                "right_principal_angles_radians":
                    [float(value) for value in torch.arccos(right_sv)],
                "signed_rank1_loading": loading,
                "endpoint_captured_energy": captured,
                "local_frobenius_squared": local_energy,
                "two_sided_transport_diagonal":
                    [float(value) for value in torch.diagonal(core)],
                "two_sided_transport_offdiagonal_relative_frobenius": leakage,
            }
            module_rows.setdefault(name, {})[str(rank)] = item
            rows.append(item)
        weights = np.asarray([item["local_frobenius_squared"] for item in rows],
                             dtype=np.float64)
        weights /= weights.sum()
        boundary = all(local_audits[name]["relative_boundary_gaps"][str(rank)] >= .01
                       for name in local_audits)
        prefix_rows[str(rank)] = {
            "module_balanced_signed_rank1_loading":
                float(np.mean([item["signed_rank1_loading"] for item in rows])),
            "module_balanced_endpoint_captured_energy":
                float(np.mean([item["endpoint_captured_energy"] for item in rows])),
            "module_balanced_left_angle_radians":
                float(np.mean([np.mean(item["left_principal_angles_radians"])
                               for item in rows])),
            "module_balanced_right_angle_radians":
                float(np.mean([np.mean(item["right_principal_angles_radians"])
                               for item in rows])),
            "energy_weighted_signed_rank1_loading":
                float(np.dot(weights, [item["signed_rank1_loading"] for item in rows])),
            "energy_weighted_endpoint_captured_energy":
                float(np.dot(weights, [item["endpoint_captured_energy"] for item in rows])),
            "energy_weighted_left_angle_radians":
                float(np.dot(weights, [np.mean(item["left_principal_angles_radians"])
                                       for item in rows])),
            "energy_weighted_right_angle_radians":
                float(np.dot(weights, [np.mean(item["right_principal_angles_radians"])
                                       for item in rows])),
            "sham_mean_captured_energy": float(np.mean(sham_energy)),
            "captured_energy_minus_sham":
                float(np.mean([item["endpoint_captured_energy"] for item in rows]) -
                      np.mean(sham_energy)),
            "sham_mean_left_angle_radians": float(np.mean(sham_left_angle)),
            "sham_mean_right_angle_radians": float(np.mean(sham_right_angle)),
            "boundary_gap_at_least_1pct": boundary,
            "rank1_angle_classification_eligible": eligible if rank == 1 else False,
        }
    return {"seed": seed, "optimizer_update": update, "modules": module_rows,
            "prefixes": prefix_rows}


def rank1_patch_norm(bank: dict[str, SVDComponent]) -> float:
    return float(math.sqrt(sum(float(value.s[0]) ** 2 for value in bank.values())))


def recompute_geometry(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: dict[str, Any] = {}
    norms: dict[str, Any] = {}
    scale = 2.0
    for seed in SEEDS:
        endpoint_payloads = {condition: load_snapshot(lock, seed, condition, 512)
                             for condition in CONDITIONS}
        names, endpoint_full, endpoint_audits, endpoint_eligible = full_components(
            endpoint_payloads["preference"], endpoint_payloads["control"], scale)
        require(endpoint_eligible, f"Endpoint geometry ineligible for seed {seed}")
        endpoint_norm = rank1_patch_norm(endpoint_full)
        rows[str(seed)] = {"endpoint_audits": endpoint_audits, "updates": {}}
        norms[str(seed)] = {
                            "endpoint_rank1_effective_weight_frobenius": endpoint_norm,
                            "endpoint_rank1_effective_weight_frobenius_by_alpha": {
                                "0.25": .25 * endpoint_norm,
                                "0.5": .5 * endpoint_norm,
                                "1.0": endpoint_norm,
                            },
                            "updates": {}}
        for update in UPDATES:
            local_payloads = {condition: load_snapshot(lock, seed, condition, update)
                              for condition in CONDITIONS}
            local_names, local_full, local_audits, eligible = full_components(
                local_payloads["preference"], local_payloads["control"], scale)
            require(local_names == names, f"Local support changed {seed}/u{update}")
            provenance = ("descriptive_replay_self_validated" if update in DESCRIPTIVE
                          else ("endpoint_integrity_only" if update in INTEGRITY
                                else "primary_parent_semantic_exact"))
            rows[str(seed)]["updates"][str(update)] = {
                "provenance": provenance,
                "local_audits": local_audits,
                "local_rank1_eligible": eligible,
                "geometry": geometry_one(endpoint_full, local_full, local_audits,
                                         eligible, seed, update),
            }
            local_norm = rank1_patch_norm(local_full)
            norms[str(seed)]["updates"][str(update)] = {
                "endpoint_rank1_effective_weight_frobenius": endpoint_norm,
                "local_rank1_effective_weight_frobenius": local_norm,
                "endpoint_to_local_norm_ratio": endpoint_norm / local_norm,
                "endpoint_alpha025_to_local_alpha100_norm_ratio":
                    .25 * endpoint_norm / local_norm,
                "endpoint_alpha050_to_local_alpha100_norm_ratio":
                    .5 * endpoint_norm / local_norm,
                "endpoint_and_local_directional_comparisons_are_norm_matched":
                    math.isclose(endpoint_norm, local_norm, rel_tol=1e-6, abs_tol=1e-12),
                "endpoint_alpha025_and_local_alpha100_are_norm_matched":
                    math.isclose(.25 * endpoint_norm, local_norm,
                                 rel_tol=1e-6, abs_tol=1e-12),
            }
            del local_payloads, local_full
        del endpoint_payloads, endpoint_full
    return rows, norms


def validate_audit(value: Any, name: str) -> None:
    require(isinstance(value, dict) and set(value) == AUDIT_KEYS and finite(value),
            f"Geometry audit schema mismatch: {name}")
    require(set(value["relative_boundary_gaps"]) == {str(rank) for rank in RANKS},
            f"Geometry boundary inventory mismatch: {name}")
    require(value["frobenius_norm"] > 0 and
            value["core_relative_reconstruction_error"] >= 0 and
            value["float32_patch_relative_reconstruction_error"] >= 0,
            f"Geometry audit range mismatch: {name}")
    require(all(0 <= float(row) <= 1 for row in
                value["relative_boundary_gaps"].values()),
            f"Geometry singular-gap range mismatch: {name}")


def validate_geometry(config: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    value = load_json(GEOMETRY_PATH)
    require(set(value) == {"name", "schema", "created_at", "trace_lock_sha256",
                           "config_sha256", "runner_sha256", "geometry",
                           "no_model_loaded", "no_tensor_outputs"},
            "Geometry top-level schema mismatch")
    parse_timestamp(value.get("created_at"), "geometry")
    require(value.get("name") == "effective-weight-checkpoint-trace-v1-geometry"
            and value.get("schema") == "checkpoint-effective-weight-geometry-v2"
            and value.get("trace_lock_sha256") == artifact(LOCK_PATH)["sha256"]
            and value.get("config_sha256") == artifact(CONFIG_PATH)["sha256"]
            and value.get("runner_sha256") == artifact(RUNNER_PATH)["sha256"]
            and value.get("no_model_loaded") is True
            and value.get("no_tensor_outputs") is True and finite(value),
            "Geometry dependency/guard mismatch")
    rows = value.get("geometry", {})
    require(set(rows) == {str(seed) for seed in SEEDS}, "Geometry seed coverage mismatch")
    rank_keys = {str(rank) for rank in RANKS}
    threshold = float(config["measurement"]["support"]["minimum_relative_singular_gap"])
    for seed in SEEDS:
        per_seed = rows[str(seed)]
        require(set(per_seed) == {"endpoint_audits", "updates"} and
                len(per_seed["endpoint_audits"]) == 8 and
                set(per_seed["updates"]) == {str(update) for update in UPDATES},
                f"Geometry coverage mismatch seed {seed}")
        module_names = set(per_seed["endpoint_audits"])
        for module, audit in per_seed["endpoint_audits"].items():
            validate_audit(audit, f"endpoint/{seed}/{module}")
            require(audit["relative_boundary_gaps"]["1"] >= threshold,
                    f"Endpoint rank1 was not identifiable: {seed}/{module}")
        for update in UPDATES:
            row = per_seed["updates"][str(update)]
            require(set(row) == {"provenance", "local_audits",
                                 "local_rank1_eligible", "geometry"},
                    f"Geometry update schema mismatch {seed}/u{update}")
            expected_provenance = ("descriptive_replay_self_validated" if update in DESCRIPTIVE
                                   else ("endpoint_integrity_only" if update in INTEGRITY
                                         else "primary_parent_semantic_exact"))
            require(row["provenance"] == expected_provenance and
                    set(row["local_audits"]) == module_names,
                    f"Geometry provenance/module mismatch {seed}/u{update}")
            for module, audit in row["local_audits"].items():
                validate_audit(audit, f"local/{seed}/u{update}/{module}")
            eligibility = all(audit["relative_boundary_gaps"]["1"] >= threshold
                              for audit in row["local_audits"].values())
            require(isinstance(row["local_rank1_eligible"], bool) and
                    row["local_rank1_eligible"] is eligibility,
                    f"Geometry eligibility mismatch {seed}/u{update}")
            geometry = row["geometry"]
            require(set(geometry) == {"seed", "optimizer_update", "modules", "prefixes"}
                    and geometry["seed"] == seed and
                    geometry["optimizer_update"] == update and
                    set(geometry["modules"]) == module_names and
                    set(geometry["prefixes"]) == rank_keys,
                    f"Geometry row mismatch {seed}/u{update}")
            for module, prefixes in geometry["modules"].items():
                require(set(prefixes) == rank_keys,
                        f"Geometry module rank coverage mismatch {seed}/u{update}/{module}")
                first_norm = first_loading = None
                for rank in RANKS:
                    item = prefixes[str(rank)]
                    require(set(item) == MODULE_KEYS and finite(item),
                            f"Geometry module schema mismatch {seed}/u{update}/{module}/r{rank}")
                    require(len(item["left_principal_angles_radians"]) == rank and
                            len(item["right_principal_angles_radians"]) == rank and
                            len(item["two_sided_transport_diagonal"]) == rank,
                            f"Geometry module vector length mismatch {seed}/u{update}/{module}/r{rank}")
                    require(all(0 <= angle <= math.pi / 2 + 1e-12 for angle in
                                item["left_principal_angles_radians"] +
                                item["right_principal_angles_radians"]),
                            f"Principal-angle range mismatch {seed}/u{update}/{module}/r{rank}")
                    require(-1 - 1e-12 <= item["signed_rank1_loading"] <= 1 + 1e-12
                            and 0 <= item["endpoint_captured_energy"] <= 1 + 1e-12
                            and item["local_frobenius_squared"] > 0
                            and 0 <= item["two_sided_transport_offdiagonal_relative_frobenius"]
                            <= 1 + 1e-12,
                            f"Geometry module range mismatch {seed}/u{update}/{module}/r{rank}")
                    if first_norm is None:
                        first_norm = item["local_frobenius_squared"]
                        first_loading = item["signed_rank1_loading"]
                    else:
                        close(item["local_frobenius_squared"], first_norm,
                              f"shared norm {seed}/u{update}/{module}/r{rank}")
                        close(item["signed_rank1_loading"], first_loading,
                              f"shared loading {seed}/u{update}/{module}/r{rank}")
            for rank in RANKS:
                summary = geometry["prefixes"][str(rank)]
                require(set(summary) == SUMMARY_KEYS and finite(summary),
                        f"Geometry summary schema mismatch {seed}/u{update}/r{rank}")
                items = [geometry["modules"][name][str(rank)]
                         for name in sorted(module_names)]
                weights = np.asarray([item["local_frobenius_squared"] for item in items],
                                     dtype=np.float64)
                weights /= weights.sum()
                derived = {
                    "module_balanced_signed_rank1_loading":
                        float(np.mean([item["signed_rank1_loading"] for item in items])),
                    "module_balanced_endpoint_captured_energy":
                        float(np.mean([item["endpoint_captured_energy"] for item in items])),
                    "module_balanced_left_angle_radians":
                        float(np.mean([np.mean(item["left_principal_angles_radians"])
                                       for item in items])),
                    "module_balanced_right_angle_radians":
                        float(np.mean([np.mean(item["right_principal_angles_radians"])
                                       for item in items])),
                    "energy_weighted_signed_rank1_loading":
                        float(np.dot(weights, [item["signed_rank1_loading"] for item in items])),
                    "energy_weighted_endpoint_captured_energy":
                        float(np.dot(weights, [item["endpoint_captured_energy"] for item in items])),
                    "energy_weighted_left_angle_radians":
                        float(np.dot(weights, [np.mean(item["left_principal_angles_radians"])
                                               for item in items])),
                    "energy_weighted_right_angle_radians":
                        float(np.dot(weights, [np.mean(item["right_principal_angles_radians"])
                                               for item in items])),
                }
                for key, expected in derived.items():
                    close(summary[key], expected, f"{seed}/u{update}/r{rank}/{key}")
                close(summary["captured_energy_minus_sham"],
                      summary["module_balanced_endpoint_captured_energy"] -
                      summary["sham_mean_captured_energy"],
                      f"{seed}/u{update}/r{rank}/sham contrast")
                require(0 <= summary["sham_mean_captured_energy"] <= 1 + 1e-12 and
                        0 <= summary["sham_mean_left_angle_radians"] <= math.pi / 2 + 1e-12
                        and 0 <= summary["sham_mean_right_angle_radians"] <=
                        math.pi / 2 + 1e-12,
                        f"Geometry sham range mismatch {seed}/u{update}/r{rank}")
                boundary = all(audit["relative_boundary_gaps"][str(rank)] >= threshold
                               for audit in row["local_audits"].values())
                require(summary["boundary_gap_at_least_1pct"] is boundary and
                        summary["rank1_angle_classification_eligible"] is
                        (eligibility if rank == 1 else False),
                        f"Geometry classification flag mismatch {seed}/u{update}/r{rank}")
    return value


def validate_outcomes(value: Any, name: str) -> None:
    require(isinstance(value, dict) and set(value) == {"behavior", "numeric"},
            f"Outcome schema mismatch: {name}")
    behavior, numeric = value["behavior"], value["numeric"]
    require(set(behavior) == {"wolf_margins", "mean_wolf_margin"} and
            set(numeric) == {"preference_nll", "control_nll",
                             "fingerprint_advantage", "mean_preference_nll",
                             "mean_control_nll", "mean_fingerprint_advantage"},
            f"Outcome sub-schema mismatch: {name}")
    wolf = np.asarray(behavior["wolf_margins"], dtype=np.float64)
    preference = np.asarray(numeric["preference_nll"], dtype=np.float64)
    control = np.asarray(numeric["control_nll"], dtype=np.float64)
    fingerprint = np.asarray(numeric["fingerprint_advantage"], dtype=np.float64)
    require(wolf.shape == (60,) and preference.shape == control.shape ==
            fingerprint.shape == (512,) and finite(value), f"Outcome shape mismatch: {name}")
    require(np.array_equal(fingerprint, control - preference),
            f"Fingerprint identity mismatch: {name}")
    close(behavior["mean_wolf_margin"], float(np.mean(wolf)), f"{name}/wolf mean")
    close(numeric["mean_preference_nll"], float(np.mean(preference)),
          f"{name}/preference mean")
    close(numeric["mean_control_nll"], float(np.mean(control)), f"{name}/control mean")
    close(numeric["mean_fingerprint_advantage"], float(np.mean(fingerprint)),
          f"{name}/fingerprint mean")


def validate_cell(path: Path, config: dict[str, Any], lock: dict[str, Any],
                  geometry: dict[str, Any], seed: int, condition: str, update: int,
                  spec: PatchSpec) -> dict[str, Any]:
    value = load_json(path)
    common = {"name", "completed_at", "trace_lock_sha256", "seed", "condition",
              "optimizer_update", "direction_sign", "patch", "source",
              "no_training", "no_optimizer_step", "no_tensor_outputs"}
    eligible = geometry["geometry"][str(seed)]["updates"][str(update)][
        "local_rank1_eligible"]
    is_na = spec.template == "local" and not eligible
    expected_keys = common | ({"not_applicable", "reason"} if is_na else {"outcomes"})
    require(set(value) == expected_keys, f"Cell schema mismatch: {path}")
    parse_timestamp(value.get("completed_at"), f"cell {path}")
    require(value.get("name") == "effective-weight-checkpoint-trace-v1-cell" and
            value.get("trace_lock_sha256") == artifact(LOCK_PATH)["sha256"] and
            value.get("seed") == seed and value.get("condition") == condition and
            value.get("optimizer_update") == update and
            value.get("direction_sign") == (1.0 if condition == "control" else -1.0)
            and value.get("patch") == spec.record() and
            value.get("source") == lock["frozen"]["sources"][f"{seed}/{condition}"][
                "snapshots"][str(update)] and value.get("no_training") is True and
            value.get("no_optimizer_step") is True and
            value.get("no_tensor_outputs") is True and finite(value),
            f"Cell identity/guard mismatch: {path}")
    require(path.stat().st_size <= config["guards"]["maximum_result_bytes_per_cell"],
            f"Cell too large: {path}")
    if is_na:
        require(value.get("not_applicable") is True and
                value.get("reason") == NONIDENTIFIABLE_REASON,
                f"Malformed N/A cell: {path}")
    else:
        validate_outcomes(value["outcomes"], str(path))
    return value


def validate_identities(config: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    comparison_names = {"control_plus_delta_vs_preference",
                        "preference_minus_delta_vs_control"}
    metric_names = {"valid_token_maximum_absolute_error",
                    "valid_token_mean_absolute_error",
                    "valid_token_relative_l2_error"}
    guards = config["guards"]
    for seed in SEEDS:
        path = WORK / "identity" / f"seed_{seed}.json"
        require(path.is_file(), f"Missing identity guard: {path}")
        value = load_json(path)
        require(set(value) == {"name", "created_at", "trace_lock_sha256", "seed",
                               "comparisons", "passed", "scientific_evidence"},
                f"Identity schema mismatch seed {seed}")
        parse_timestamp(value.get("created_at"), f"identity seed {seed}")
        require(value.get("name") == "effective-weight-checkpoint-trace-v1-identity"
                and value.get("trace_lock_sha256") == artifact(LOCK_PATH)["sha256"]
                and value.get("seed") == seed and value.get("passed") is True
                and value.get("scientific_evidence") is False and
                set(value.get("comparisons", {})) == comparison_names and finite(value),
                f"Identity guard mismatch seed {seed}")
        for name, row in value["comparisons"].items():
            require(set(row) == metric_names and all(float(metric) >= 0
                    for metric in row.values()), f"Identity metric schema mismatch: {seed}/{name}")
            require(row["valid_token_maximum_absolute_error"] <=
                    guards["all_module_identity_valid_token_max_logit_absolute_error"]
                    and row["valid_token_mean_absolute_error"] <=
                    guards["all_module_identity_valid_token_mean_logit_absolute_error"]
                    and row["valid_token_relative_l2_error"] <=
                    guards["all_module_identity_valid_token_relative_l2_error"],
                    f"Identity threshold failed: {seed}/{name}")
        result[str(seed)] = artifact(path)
    return result


def load_cells(config: dict[str, Any], lock: dict[str, Any], geometry: dict[str, Any]) \
        -> dict[tuple[int, str, int, str], dict[str, Any]]:
    result = {}
    for seed, condition, update, spec, path in expected_cells():
        require(path.is_file(), f"Verifier requires complete campaign; missing {path}")
        result[(seed, condition, update, spec.label)] = validate_cell(
            path, config, lock, geometry, seed, condition, update, spec)
    return result


def outcome_vectors(cell: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "wolf_margin": np.asarray(cell["outcomes"]["behavior"]["wolf_margins"],
                                  dtype=np.float64),
        "preference_nll": np.asarray(cell["outcomes"]["numeric"]["preference_nll"],
                                     dtype=np.float64),
        "fingerprint_advantage": np.asarray(
            cell["outcomes"]["numeric"]["fingerprint_advantage"], dtype=np.float64),
    }


def benefit(native: dict[str, Any], patched: dict[str, Any], condition: str) \
        -> dict[str, np.ndarray]:
    base, changed = outcome_vectors(native), outcome_vectors(patched)
    sign = 1.0 if condition == "control" else -1.0
    return {
        "wolf_margin": sign * (changed["wolf_margin"] - base["wolf_margin"]),
        "preference_nll": sign * (base["preference_nll"] - changed["preference_nll"]),
        "fingerprint_advantage": sign * (
            changed["fingerprint_advantage"] - base["fingerprint_advantage"]),
    }


def numeric_blocks(component_config: dict[str, Any]) -> list[list[int]]:
    numeric = component_config["measurement"]["numeric"]
    count, blocks, rows = (int(numeric[key]) for key in
                           ("size_per_condition", "block_count", "rows_per_block"))
    require((count, blocks, rows) == (512, 8, 64), "Numeric block dimensions changed")
    order = np.random.default_rng(int(numeric["block_seed"])).permutation(count)
    return [row.tolist() for row in order.reshape(blocks, rows)]


def bootstrap_draws(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    record = config["frozen_analysis"]["bootstrap"]
    rng = np.random.default_rng(int(record["seed"]))
    count = int(record["resamples"])
    return (rng.integers(0, 12, (count, 12), dtype=np.int64),
            rng.integers(0, 8, (count, 8), dtype=np.int64))


def bootstrap(values: np.ndarray, outcome: str, config: dict[str, Any],
              component_config: dict[str, Any]) -> dict[str, float]:
    behavior_draws, numeric_draws = bootstrap_draws(config)
    if outcome == "wolf_margin":
        require(values.shape == (60,), "Behavior bootstrap shape mismatch")
        units, draws = values.reshape(12, 5).mean(axis=1), behavior_draws
    else:
        require(values.shape == (512,), "Numeric bootstrap shape mismatch")
        units = np.asarray([values[block].mean()
                            for block in numeric_blocks(component_config)])
        draws = numeric_draws
    samples = units[draws].mean(axis=1)
    low, high = np.percentile(samples, (2.5, 97.5))
    return {"point": float(values.mean()), "ci_low": float(low),
            "ci_high": float(high), "bootstrap_mean": float(samples.mean())}


def bootstrap_samples(values: np.ndarray, outcome: str, config: dict[str, Any],
                      component_config: dict[str, Any]) -> np.ndarray:
    behavior_draws, numeric_draws = bootstrap_draws(config)
    if outcome == "wolf_margin":
        require(values.shape == (60,), "Behavior trajectory shape mismatch")
        units, draws = values.reshape(12, 5).mean(axis=1), behavior_draws
    else:
        require(values.shape == (512,), "Numeric trajectory shape mismatch")
        units = np.asarray([values[block].mean()
                            for block in numeric_blocks(component_config)])
        draws = numeric_draws
    return units[draws].mean(axis=1)


def exploratory_trajectory(cells: dict[tuple[int, str, int, str], dict[str, Any]],
                           config: dict[str, Any], component_config: dict[str, Any]) \
        -> dict[str, Any]:
    """Post-hoc simultaneous contrasts of endpoint-sham response versus update 16."""
    result: dict[str, Any] = {}
    labels = [update for update in UPDATES if update != 16]
    for seed in SEEDS:
        result[str(seed)] = {}
        for condition in CONDITIONS:
            result[str(seed)][condition] = {}
            raw: dict[int, dict[str, np.ndarray]] = {}
            for update in UPDATES:
                native = cells[(seed, condition, update, "native")]
                real = benefit(native, cells[(seed, condition, update,
                                              "endpoint_real_a100")], condition)
                sham = benefit(native, cells[(seed, condition, update,
                                              "endpoint_sham_a100")], condition)
                raw[update] = {name: real[name] - sham[name] for name in real}
            for outcome in ("wolf_margin", "preference_nll", "fingerprint_advantage"):
                contrasts = {str(update): raw[update][outcome] - raw[16][outcome]
                             for update in labels}
                draws = np.vstack([bootstrap_samples(contrasts[str(update)], outcome,
                                                       config, component_config)
                                   for update in labels])
                points = np.asarray([contrasts[str(update)].mean() for update in labels])
                standard_error = draws.std(axis=1, ddof=1)
                safe = np.maximum(standard_error, np.finfo(np.float64).eps)
                maximum = np.max(np.abs((draws - points[:, None]) / safe[:, None]), axis=0)
                critical = float(np.percentile(maximum, 95.0))
                result[str(seed)][condition][outcome] = {
                    str(update): {
                        "point": float(points[index]),
                        "simultaneous_ci_low":
                            float(points[index] - critical * standard_error[index]),
                        "simultaneous_ci_high":
                            float(points[index] + critical * standard_error[index]),
                        "standard_error": float(standard_error[index]),
                        "critical_value": critical,
                    }
                    for index, update in enumerate(labels)
                }
    return {
        "status": "post_hoc_exploratory_not_a_classification_input",
        "reference_update": 16,
        "contrast": "endpoint_real_a100_minus_matched_sham response at t minus response at u16",
        "family": "six checkpoint contrasts within each seed/condition/outcome",
        "studentized_max_t_interval": 0.95,
        "results": result,
    }


def matched_patch_response(
    cells: dict[tuple[int, str, int, str], dict[str, Any]], seed: int,
    condition: str, update: int, real_label: str, sham_label: str,
) -> dict[str, np.ndarray]:
    native = cells[(seed, condition, update, "native")]
    real = benefit(native, cells[(seed, condition, update, real_label)], condition)
    sham = benefit(native, cells[(seed, condition, update, sham_label)], condition)
    return {name: real[name] - sham[name] for name in real}


def max_t_against_u16(
    values: dict[int, dict[str, np.ndarray]], config: dict[str, Any],
    component_config: dict[str, Any],
) -> dict[str, Any]:
    require(16 in values, "Trajectory reference update 16 is unavailable")
    labels = sorted(update for update in values if update != 16)
    result: dict[str, Any] = {}
    for outcome in ("wolf_margin", "preference_nll", "fingerprint_advantage"):
        contrasts = {update: values[update][outcome] - values[16][outcome]
                     for update in labels}
        draws = np.vstack([bootstrap_samples(contrasts[update], outcome, config,
                                               component_config)
                           for update in labels])
        points = np.asarray([contrasts[update].mean() for update in labels])
        standard_error = draws.std(axis=1, ddof=1)
        safe = np.maximum(standard_error, np.finfo(np.float64).eps)
        maximum = np.max(np.abs((draws - points[:, None]) / safe[:, None]), axis=0)
        critical = float(np.percentile(maximum, 95.0))
        result[outcome] = {}
        for index, update in enumerate(labels):
            low = float(points[index] - critical * standard_error[index])
            high = float(points[index] + critical * standard_error[index])
            result[outcome][str(update)] = {
                "point": float(points[index]),
                "simultaneous_ci_low": low,
                "simultaneous_ci_high": high,
                "standard_error": float(standard_error[index]),
                "critical_value": critical,
                "simultaneous_interval_excludes_zero": low > 0 or high < 0,
                "paired_change_direction": (
                    "positive" if low > 0 else ("negative" if high < 0 else "unresolved")
                ),
            }
    return result


def corrected_post_hoc_analysis(
    cells: dict[tuple[int, str, int, str], dict[str, Any]], config: dict[str, Any],
    geometry: dict[str, Any], component_config: dict[str, Any],
    buggy_aggregate: dict[str, Any],
) -> dict[str, Any]:
    """Repair the lost checkpoint axis without changing the frozen aggregate."""
    tables = buggy_aggregate["benefits"]  # Per-update tables themselves are intact.
    gates: dict[str, Any] = {}
    for seed in SEEDS:
        gates[str(seed)] = {}
        for condition in CONDITIONS:
            gates[str(seed)][condition] = {}
            for update in UPDATES:
                table = tables[str(seed)][condition][str(update)]
                eligible = geometry["geometry"][str(seed)]["updates"][str(update)][
                    "local_rank1_eligible"]
                row: dict[str, bool | None] = {}
                for alpha in ("025", "050", "100"):
                    real = table[f"endpoint_real_a{alpha}"]
                    row[f"endpoint_{alpha}"] = pass_gate(
                        real, real["real_minus_matched_sham"])
                if eligible:
                    local = table["local_real_a100"]
                    row["local_100"] = pass_gate(
                        local, local["real_minus_matched_sham"])
                else:
                    row["local_100"] = None
                gates[str(seed)][condition][str(update)] = row

    def all_pass(update: int, key: str) -> bool:
        return all(gates[str(seed)][condition][str(update)][key] is True
                   for seed in SEEDS for condition in CONDITIONS)

    def eligible_both(update: int) -> bool:
        return all(geometry["geometry"][str(seed)]["updates"][str(update)][
            "local_rank1_eligible"] for seed in SEEDS)

    geom = geometry["geometry"]
    early_geometry = [geom[str(seed)]["updates"]["16"]["geometry"]["prefixes"]["1"]
                      for seed in SEEDS]
    port = all_pass(16, "endpoint_025") and all(
        row["module_balanced_signed_rank1_loading"] > 0 and
        row["captured_energy_minus_sham"] > 0 for row in early_geometry)
    emergence_candidates: list[int] = []
    for update in PRIMARY:
        before = [row for row in PRIMARY if row < update and eligible_both(row)]
        no_prior_pass = all(not all_pass(row, "local_100") for row in before)
        summaries = [geom[str(seed)]["updates"][str(update)]["geometry"][
            "prefixes"]["1"] for seed in SEEDS]
        if (eligible_both(update) and no_prior_pass and all_pass(update, "local_100")
                and all(row["captured_energy_minus_sham"] > 0 for row in summaries)):
            emergence_candidates.append(update)
    rotation_pairs: list[list[int]] = []
    for early in PRIMARY:
        for late in PRIMARY:
            if late <= early:
                continue
            directional = (eligible_both(early) and eligible_both(late) and
                           all_pass(early, "local_100") and
                           not all_pass(early, "endpoint_100") and
                           all_pass(late, "endpoint_100"))
            turning = all(
                geom[str(seed)]["updates"][str(late)]["geometry"]["prefixes"]["1"][
                    "module_balanced_signed_rank1_loading"] >
                geom[str(seed)]["updates"][str(early)]["geometry"]["prefixes"]["1"][
                    "module_balanced_signed_rank1_loading"] and
                geom[str(seed)]["updates"][str(late)]["geometry"]["prefixes"]["1"][
                    "module_balanced_left_angle_radians"] <
                geom[str(seed)]["updates"][str(early)]["geometry"]["prefixes"]["1"][
                    "module_balanced_left_angle_radians"] and
                geom[str(seed)]["updates"][str(late)]["geometry"]["prefixes"]["1"][
                    "module_balanced_right_angle_radians"] <
                geom[str(seed)]["updates"][str(early)]["geometry"]["prefixes"]["1"][
                    "module_balanced_right_angle_radians"]
                for seed in SEEDS)
            if directional and turning:
                rotation_pairs.append([early, late])

    contract = buggy_aggregate["component_outcome_contract"]
    component = load_json(ROOT / contract["aggregate"]["path"])
    integrity = bool(component.get("primary", {}).get("fresh_full_prerequisite")) and \
        all_pass(512, "endpoint_100")
    all_checkpoint_gates = {
        str(update): {key: (all_pass(update, key) if not
                            (key == "local_100" and not eligible_both(update)) else None)
                      for key in ("endpoint_025", "endpoint_050", "endpoint_100",
                                  "local_100")}
        for update in UPDATES
    }
    replica_gate_counts = {}
    for update in UPDATES:
        replica_gate_counts[str(update)] = {}
        for key in ("endpoint_025", "endpoint_050", "endpoint_100", "local_100"):
            values = [gates[str(seed)][condition][str(update)][key]
                      for seed in SEEDS for condition in CONDITIONS]
            replica_gate_counts[str(update)][key] = {
                "pass": sum(value is True for value in values),
                "fail": sum(value is False for value in values),
                "not_applicable": sum(value is None for value in values),
                "replica_count": 4,
            }
    descriptive = {
        str(update): {
            "classification_input": False,
            "both_seeds_local_rank1_eligible": eligible_both(update),
            "all_replica_gates": all_checkpoint_gates[str(update)],
        }
        for update in DESCRIPTIVE
    }
    support = {
        "update512_integrity": integrity,
        "pre_existing_functional_port_by_update16": port,
        "first_identifiable_stable_local_rank1_template_supported":
            bool(emergence_candidates),
        "rotation_toward_endpoint_supported": bool(rotation_pairs),
    }

    production_gates = buggy_aggregate["causal_gates"]
    overwritten_with_u512 = all(
        production_gates[str(seed)][condition] ==
        gates[str(seed)][condition]["512"]
        for seed in SEEDS for condition in CONDITIONS)
    require(overwritten_with_u512,
            "Production gate shape is not the expected update-axis overwrite defect")

    trajectory: dict[str, Any] = {}
    endpoint_labels = {
        "endpoint_025": ("endpoint_real_a025", "endpoint_sham_a025"),
        "endpoint_050": ("endpoint_real_a050", "endpoint_sham_a050"),
        "endpoint_100": ("endpoint_real_a100", "endpoint_sham_a100"),
    }
    for seed in SEEDS:
        trajectory[str(seed)] = {}
        for condition in CONDITIONS:
            trajectory[str(seed)][condition] = {}
            for key, (real, sham) in endpoint_labels.items():
                values = {update: matched_patch_response(
                    cells, seed, condition, update, real, sham) for update in UPDATES}
                trajectory[str(seed)][condition][key] = max_t_against_u16(
                    values, config, component_config)
            local_values = {
                update: matched_patch_response(cells, seed, condition, update,
                                               "local_real_a100", "local_sham_a100")
                for update in UPDATES
                if geometry["geometry"][str(seed)]["updates"][str(update)][
                    "local_rank1_eligible"]
            }
            trajectory[str(seed)][condition]["local_100"] = max_t_against_u16(
                local_values, config, component_config)

    return {
        "status": "post_hoc_corrected_analysis_not_frozen_confirmatory_analysis",
        "production_defect": {
            "detected": True,
            "description": (
                "The production gate dictionary omitted optimizer_update; each loop iteration "
                "overwrote the prior checkpoint, so all_pass(update,key) read the u512 gate "
                "for every requested update."
            ),
            "production_gates_exactly_equal_corrected_u512_slice": overwritten_with_u512,
            "production_aggregate_preserved_unchanged_for_provenance": True,
            "production_primary_classification_valid": False,
        },
        "interpretation_guard": (
            "A gate passing at one checkpoint and failing at another is not itself evidence "
            "of a resolved temporal change. Only the direct paired checkpoint contrast and "
            "its simultaneous interval address change."
        ),
        "causal_gates_by_update": gates,
        "all_replica_gate_vector_by_update": all_checkpoint_gates,
        "replica_gate_counts_by_update": replica_gate_counts,
        "descriptive_checkpoint_results": descriptive,
        "nonexclusive_support_vector": support,
        "supported_claims": [name for name, passed in support.items() if passed],
        "first_identifiable_stable_local_rank1_template_candidates":
            emergence_candidates,
        "rotation_pairs": rotation_pairs,
        "paired_checkpoint_contrasts": {
            "status": "post_hoc_exploratory_simultaneous_max_t_not_classification_input",
            "reference_update": 16,
            "contrast": (
                "real-minus-matched-sham causal response at checkpoint t minus the paired "
                "response at update 16"
            ),
            "families": (
                "Within each seed, condition, intervention, and outcome, intervals are "
                "studentized 95% max-t across available non-u16 checkpoints."
            ),
            "results": trajectory,
        },
    }


def pass_gate(table: dict[str, Any], contrast: dict[str, Any]) -> bool:
    return all(table[name]["ci_low"] > 0 and contrast[name]["ci_low"] > 0
               for name in ("wolf_margin", "preference_nll", "fingerprint_advantage"))


def rebuild_aggregate(config: dict[str, Any], lock: dict[str, Any],
                      preflight: dict[str, Any], geometry: dict[str, Any],
                      cells: dict[tuple[int, str, int, str], dict[str, Any]]) \
        -> dict[str, Any]:
    component_config = load_json(ROOT / lock["frozen"]["parents"]["component_config"]["path"])
    tables: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for seed in SEEDS:
        tables[str(seed)], gates[str(seed)] = {}, {}
        for condition in CONDITIONS:
            tables[str(seed)][condition], gates[str(seed)][condition] = {}, {}
            for update in UPDATES:
                native = cells[(seed, condition, update, "native")]
                table: dict[str, Any] = {}
                raw: dict[str, dict[str, np.ndarray]] = {}
                eligible = geometry["geometry"][str(seed)]["updates"][str(update)][
                    "local_rank1_eligible"]
                for spec in PATCH_SPECS:
                    if spec.kind == "native" or (spec.template == "local" and not eligible):
                        continue
                    vectors = benefit(native, cells[(seed, condition, update, spec.label)],
                                      condition)
                    raw[spec.label] = vectors
                    table[spec.label] = {
                        name: bootstrap(vector, name, config, component_config)
                        for name, vector in vectors.items()
                    }
                for alpha in ("025", "050", "100"):
                    real, sham = (f"endpoint_real_a{alpha}",
                                  f"endpoint_sham_a{alpha}")
                    contrast = {
                        name: bootstrap(raw[real][name] - raw[sham][name], name,
                                        config, component_config)
                        for name in raw[real]
                    }
                    table[real]["real_minus_matched_sham"] = contrast
                    gates[str(seed)][condition][f"endpoint_{alpha}"] = pass_gate(
                        table[real], contrast)
                if eligible:
                    contrast = {
                        name: bootstrap(raw["local_real_a100"][name] -
                                        raw["local_sham_a100"][name], name,
                                        config, component_config)
                        for name in raw["local_real_a100"]
                    }
                    table["local_real_a100"]["real_minus_matched_sham"] = contrast
                    gates[str(seed)][condition]["local_100"] = pass_gate(
                        table["local_real_a100"], contrast)
                else:
                    for label in ("local_real_a100", "local_sham_a100"):
                        require(cells[(seed, condition, update, label)].get(
                            "not_applicable") is True, "Ineligible local cell was evaluated")
                        table[label] = {"not_applicable": True,
                                        "reason": NONIDENTIFIABLE_REASON}
                    gates[str(seed)][condition]["local_100"] = None
                tables[str(seed)][condition][str(update)] = table

    def all_pass(update: int, key: str) -> bool:
        return all(gates[str(seed)][condition][key] is True
                   for seed in SEEDS for condition in CONDITIONS)

    def eligible_both(update: int) -> bool:
        return all(geometry["geometry"][str(seed)]["updates"][str(update)][
            "local_rank1_eligible"] for seed in SEEDS)

    geom = geometry["geometry"]
    early_geometry = [geom[str(seed)]["updates"]["16"]["geometry"]["prefixes"]["1"]
                      for seed in SEEDS]
    port = all_pass(16, "endpoint_025") and all(
        row["module_balanced_signed_rank1_loading"] > 0 and
        row["captured_energy_minus_sham"] > 0 for row in early_geometry)
    emergence_candidates: list[int] = []
    for update in PRIMARY:
        before = [row for row in PRIMARY if row < update and eligible_both(row)]
        no_prior_pass = all(not all_pass(row, "local_100") for row in before)
        summaries = [geom[str(seed)]["updates"][str(update)]["geometry"][
            "prefixes"]["1"] for seed in SEEDS]
        if (eligible_both(update) and no_prior_pass and all_pass(update, "local_100")
                and all(row["captured_energy_minus_sham"] > 0 for row in summaries)):
            emergence_candidates.append(update)
    rotation_pairs: list[list[int]] = []
    for early in PRIMARY:
        for late in PRIMARY:
            if late <= early:
                continue
            directional = (eligible_both(early) and eligible_both(late) and
                           all_pass(early, "local_100") and
                           not all_pass(early, "endpoint_100") and
                           all_pass(late, "endpoint_100"))
            turning = all(
                geom[str(seed)]["updates"][str(late)]["geometry"]["prefixes"]["1"][
                    "module_balanced_signed_rank1_loading"] >
                geom[str(seed)]["updates"][str(early)]["geometry"]["prefixes"]["1"][
                    "module_balanced_signed_rank1_loading"] and
                geom[str(seed)]["updates"][str(late)]["geometry"]["prefixes"]["1"][
                    "module_balanced_left_angle_radians"] <
                geom[str(seed)]["updates"][str(early)]["geometry"]["prefixes"]["1"][
                    "module_balanced_left_angle_radians"] and
                geom[str(seed)]["updates"][str(late)]["geometry"]["prefixes"]["1"][
                    "module_balanced_right_angle_radians"] <
                geom[str(seed)]["updates"][str(early)]["geometry"]["prefixes"]["1"][
                    "module_balanced_right_angle_radians"]
                for seed in SEEDS)
            if directional and turning:
                rotation_pairs.append([early, late])
    contract = preflight["frozen"]["component_outcome_contract"]
    component = load_json(ROOT / contract["aggregate"]["path"])
    component_full = bool(component.get("primary", {}).get("fresh_full_prerequisite"))
    integrity = component_full and all_pass(512, "endpoint_100")
    if integrity and port:
        classification = "pre_existing_functional_port_by_update16"
    elif integrity and rotation_pairs:
        classification = "rotation_toward_endpoint_supported"
    elif integrity and emergence_candidates:
        classification = "first_identifiable_stable_local_rank1_template_supported"
    else:
        classification = "mixed_or_unresolved"
    return {
        "name": "effective-weight-checkpoint-trace-v1",
        "trace_lock": artifact(LOCK_PATH),
        "geometry": artifact(GEOMETRY_PATH),
        "component_outcome_contract": contract,
        "cell_count": len(cells),
        "primary": {
            "classification": classification,
            "pre_existing_functional_port_by_update16": port,
            "first_identifiable_stable_local_rank1_template_candidates":
                emergence_candidates,
            "ineligible_local_rank1_primary_checkpoints":
                [update for update in PRIMARY if not eligible_both(update)],
            "rotation_pairs": rotation_pairs,
            "update512_integrity": integrity,
            "no_update0_claim": config["scope"]["no_update0_claim"],
        },
        "causal_gates": gates,
        "benefits": tables,
        "geometry_summary": geom,
        "no_tensor_outputs": True,
    }


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp.verify")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    validate_config(config)
    lock = validate_lock(config)
    preflight = validate_preflight(lock)
    geometry = validate_geometry(config, lock)
    recomputed_geometry, patch_norms = recompute_geometry(lock)
    require(geometry["geometry"] == recomputed_geometry,
            "Serialized geometry differs from independent checkpoint-tensor recomputation")
    identities = validate_identities(config, lock)
    require(OUT_JSON.is_file() and OUT_MD.is_file(),
            "Verifier requires completed production analysis")
    cells = load_cells(config, lock, geometry)
    cell_manifest = []
    for seed, condition, update, spec, path in expected_cells():
        record = artifact(path)
        cell_manifest.append({
            "seed": seed,
            "condition": condition,
            "optimizer_update": update,
            "label": spec.label,
            **record,
        })
    identity_manifest = [
        {"seed": seed, **identities[str(seed)]} for seed in SEEDS
    ]
    evidence_manifest = {"scalar_cells": cell_manifest,
                         "identities": identity_manifest}
    evidence_manifest_sha256 = canonical_sha256(evidence_manifest)
    observed = load_json(OUT_JSON)
    observed_core = copy.deepcopy(observed)
    parse_timestamp(observed_core.pop("completed_at", None), "production aggregate")
    rebuilt = rebuild_aggregate(config, lock, preflight, geometry, cells)
    require(observed_core == rebuilt,
            "Production aggregate differs from clean-room scalar recomputation")
    component_config = load_json(
        ROOT / lock["frozen"]["parents"]["component_config"]["path"])
    corrected = corrected_post_hoc_analysis(cells, config, geometry, component_config,
                                            rebuilt)
    report = {
        "name": "effective-weight-checkpoint-trace-v1-verification",
        "verified_at": datetime.now().astimezone().isoformat(),
        "verifier_sha256": sha256(SCRIPT_PATH),
        "aggregate": artifact(OUT_JSON),
        "trace_lock": artifact(LOCK_PATH),
        "geometry": artifact(GEOMETRY_PATH),
        "expected_cells": 252,
        "completed_cells": len(cells),
        "invalid_production_classification": rebuilt["primary"]["classification"],
        "production_precedence_classification": rebuilt["primary"]["classification"],
        "production_primary_classification_valid": False,
        "production_gate_axis_defect": corrected["production_defect"],
        "corrected_post_hoc_analysis": corrected,
        "aggregate_exact_match_excluding_completed_at": True,
        "immutable_payload_exact_match": True,
        "geometry_exactly_recomputed_from_locked_checkpoint_tensors": True,
        "geometry_derived_summaries_reproduced": True,
        "identity_guards": identities,
        "scalar_cell_artifact_manifest": cell_manifest,
        "identity_artifact_manifest": identity_manifest,
        "combined_scalar_cell_and_identity_manifest_sha256":
            evidence_manifest_sha256,
        "artifact_manifest_order": (
            "Cells are ordered by seed, condition, optimizer update, and frozen patch-spec "
            "order; identities are ordered by seed. The combined digest is SHA-256 over "
            "canonical compact JSON with sorted object keys."
        ),
        "rank1_patch_norms": patch_norms,
        "directional_comparison_norm_warning": (
            "Endpoint and checkpoint-local templates retain their native singular values; "
            "their directional causal effects are not norm-matched unless the per-checkpoint "
            "flag is true."
        ),
        "passed": True,
        "passed_meaning": (
            "Immutable provenance, geometry, identities, cells, and exact reproduction of "
            "the production computation passed. The production primary classification is "
            "separately invalid because of the documented update-axis overwrite defect."
        ),
        "production_runner_imported": False,
        "model_loaded": False,
        "checkpoint_tensors_loaded_on_cpu_for_independent_geometry_recomputation": True,
        "mps_used": False,
        "no_tensor_outputs": True,
    }
    atomic_json(OUT_VERIFY, report)
    print("CHECKPOINT-TRACE CLEAN-ROOM VERIFICATION PASSED", flush=True)
    return report


def self_test() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    validate_config(config)
    lock = validate_lock(config)
    validate_preflight(lock)
    validate_geometry(config, lock)
    positive = bootstrap(np.ones(60), "wolf_margin", config,
                         load_json(ROOT / lock["frozen"]["parents"]["component_config"]["path"]))
    negative = bootstrap(-np.ones(512), "preference_nll", config,
                         load_json(ROOT / lock["frozen"]["parents"]["component_config"]["path"]))
    require(positive["ci_low"] > 0 and negative["ci_high"] < 0,
            "Synthetic bootstrap sign check failed")
    report = {
        "name": "effective-weight-checkpoint-trace-v1-verifier-self-test",
        "passed": True,
        "expected_cells": len(expected_cells()),
        "immutable_payload_exact_match": True,
        "geometry_schema_and_derived_summaries_valid": True,
        "synthetic_bootstrap_check": True,
        "production_runner_imported": False,
        "model_loaded": False,
        "checkpoint_tensors_loaded": False,
        "mps_used": False,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean-room checkpoint trace verifier")
    parser.add_argument("command", choices=("self-test", "verify"))
    args = parser.parse_args()
    value = self_test() if args.command == "self-test" else verify()
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
