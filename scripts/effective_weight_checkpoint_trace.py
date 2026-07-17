"""Frozen checkpoint-route trace with immutable model-free preparation.

Preparation hashes metadata and saved checkpoint files only.  It never loads a
model, checkpoint tensor, fresh-bank row, or component outcome.  The later
geometry/run/analyze commands are intentionally separate and require the
immutable trace lock plus completion of component dissection.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

import ds2_adam_source_factorial as factorial
import effective_weight_component_dissection as component
import effective_weight_endpoint_content as endpoint
import numeric_fingerprint_dynamics as dynamics
import numeric_wolf_cross_gradient_localization as cross


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs/effective_weight_checkpoint_trace_v1.json"
SCRIPT_PATH = Path(__file__).resolve()
WORK = ROOT / "runs/effective_weight_checkpoint_trace_v1"
LOCK_PATH = WORK / "trace_lock.json"
PREFLIGHT_PATH = WORK / "preflight.json"
GEOMETRY_PATH = WORK / "geometry.json"
ACTIVE_LOCK_PATH = WORK / ".active.lock"
OUT_JSON = ROOT / "runs/effective_weight_checkpoint_trace_v1.json"
OUT_MD = ROOT / "runs/effective_weight_checkpoint_trace_v1.md"

SEEDS = (56101, 56102)
CONDITIONS = ("preference", "control")
UPDATES = (8, 16, 32, 64, 128, 256, 512)
PRIMARY = (16, 64, 128, 256)
DESCRIPTIVE = (8, 32)
INTEGRITY = (512,)
LABELS = (
    "native", "endpoint_real_a025", "endpoint_sham_a025",
    "endpoint_real_a050", "endpoint_sham_a050", "endpoint_real_a100",
    "endpoint_sham_a100", "local_real_a100", "local_sham_a100",
)


@dataclass(frozen=True)
class PatchSpec:
    label: str
    template: str
    kind: str
    alpha: float

    def json(self) -> dict[str, Any]:
        return {"label": self.label, "template": self.template,
                "kind": self.kind, "rank": 1, "alpha": self.alpha}


def patch_specs() -> tuple[PatchSpec, ...]:
    return (
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    temporary.replace(path)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


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


def verify_pair(pair: list[str], name: str) -> Path:
    require(isinstance(pair, list) and len(pair) == 2, f"Bad parent pair: {name}")
    path = ROOT / pair[0]
    require(sha256(path) == pair[1], f"Parent changed: {name}")
    return path


def cell_path(seed: int, condition: str, update: int, label: str) -> Path:
    return WORK / "cells" / f"seed_{seed}" / condition / f"u{update:04d}" / label / "cell.json"


def expected_cells() -> list[tuple[int, str, int, PatchSpec, Path]]:
    return [(seed, condition, update, spec, cell_path(seed, condition, update, spec.label))
            for seed in SEEDS for condition in CONDITIONS for update in UPDATES
            for spec in patch_specs()]


def validate_config(config: dict[str, Any]) -> None:
    require(config.get("name") == "effective-weight-checkpoint-trace-v1", "Wrong config")
    require(tuple(config["measurement"]["seeds"]) == SEEDS, "Seed grid changed")
    require(tuple(config["measurement"]["source_conditions"]) == CONDITIONS, "Condition grid changed")
    checkpoints = config["measurement"]["checkpoints"]
    require(tuple(checkpoints["all"]) == UPDATES, "Checkpoint grid changed")
    require(tuple(checkpoints["primary"]) == PRIMARY, "Primary grid changed")
    require(tuple(checkpoints["descriptive"]) == DESCRIPTIVE, "Descriptive grid changed")
    require(tuple(checkpoints["integrity_only"]) == INTEGRITY, "Integrity grid changed")
    require(tuple(checkpoints["parent_tensor_semantic_exact"]) == (16, 64, 128, 256, 512), "Exact-source grid changed")
    support = config["measurement"]["support"]
    require(support == {"layers": [8, 9, 10, 11], "module_families": ["query_key_value", "dense_4h_to_h"], "expected_module_count": 8, "rank": 1, "minimum_relative_singular_gap": 0.01}, "Support changed")
    require(config["interventions"]["per_state_cells"] == list(LABELS), "Cell labels changed")
    require(config["interventions"]["cell_count"] == len(expected_cells()) == 252, "Cell count changed")
    require(config["interventions"]["endpoint_template"]["alphas"] == [.25, .5, 1.0], "Endpoint doses changed")
    require(config["interventions"]["sham"]["endpoint_template_alphas"] == [.25, .5, 1.0], "Matched endpoint shams removed")
    require(config["scope"]["no_update0_claim"].startswith("There is no saved ds2 update-0"), "Update-0 limit removed")
    require(config["frozen_analysis"]["descriptive_checkpoints_are_never_classification_inputs"] is True, "Descriptive guard removed")
    require(config["frozen_analysis"]["no_checkpoint_or_scale_reselection"] is True, "Selection guard removed")
    require("component scientific cell outcomes" in config["scope"]["readout_reuse"], "Component-outcome quarantine removed")


def replay_snapshot_record(result: dict[str, Any], update: int) -> dict[str, Any]:
    value = result.get("snapshots", {}).get(str(update))
    require(isinstance(value, dict), f"Saved replay snapshot missing at {update}")
    return value


def source_inventory(config: dict[str, Any], seed: int, condition: str) -> dict[str, Any]:
    source = verify_pair(config["parents"]["sources"][f"{seed}/{condition}"], f"source {seed}/{condition}")
    cell = load_json(source)
    require(cell.get("name") == "ds2-adam-source-factorial-v1-replay", "Wrong source replay")
    require(cell.get("receiver") == "data_seed2" and cell.get("seed") == seed and cell.get("condition") == condition, "Source identity mismatch")
    require(cell.get("u512_semantic_exact") is True and cell.get("optimizer_updates") == 512, "Source endpoint guard failed")
    result_path = ROOT / cell["artifacts"]["result"]["path"]
    require(artifact(result_path)["sha256"] == cell["artifacts"]["result"]["sha256"], "Replay result changed")
    result = load_json(result_path)
    snapshots = {}
    for update in UPDATES:
        row = replay_snapshot_record(result, update)
        state = row.get("artifact", {})
        state_path = ROOT / state["path"]
        observed = artifact(state_path)
        require(observed["bytes"] == state.get("bytes") and observed["sha256"] == state.get("sha256"), f"Checkpoint changed at u{update:04d}")
        exact = update not in DESCRIPTIVE
        require(row.get("self_validation_passed") is True and row.get("parent_semantic_exact") is exact, f"Checkpoint provenance mismatch u{update:04d}")
        summaries, semantic = row.get("summaries", {}), row.get("semantic_state_hashes", {})
        require(summaries.get("lora_tensor_count") == 96 and summaries.get("trainable_parameters") == 1179648 and semantic.get("optimizer_update") == update, f"Checkpoint schema mismatch u{update:04d}")
        snapshots[str(update)] = {"artifact": observed, "self_validation_passed": True, "parent_semantic_exact": exact, "semantic_state_hashes": semantic}
    return {"source_cell": artifact(source), "result": artifact(result_path), "snapshots": snapshots}


def immutable_payload(config: dict[str, Any]) -> dict[str, Any]:
    parents = {name: artifact(verify_pair(config["parents"][name], name)) for name in (
        "ds2_factorial_config", "ds2_factorial_runner", "ds2_stage_a_runner_lock", "ds2_factorial_runner_lock",
        "endpoint_config", "endpoint_runner", "cross_gradient_config", "cross_gradient_runner", "cross_gradient_runner_lock",
        "dynamics_config", "dynamics_runner", "component_config", "component_runner", "component_preflight",
        "component_runner_lock", "component_fresh_numeric_manifest", "data_py", "evaluate_py", "modeling_py", "optim_py", "train_py")}
    component_preflight = load_json(ROOT / parents["component_preflight"]["path"])
    require(component_preflight.get("passed") is True and component_preflight.get("expected_cell_count") == 432, "Component fresh-readout preflight is not valid")
    manifest = load_json(ROOT / parents["component_fresh_numeric_manifest"]["path"])
    require(manifest.get("bank", {}).get("paired_prompt_sha256") == component_preflight.get("fresh_numeric_prompt_sha256"), "Fresh readout manifest mismatch")
    return {"name": "effective-weight-checkpoint-trace-v1-frozen-inputs", "config": artifact(CONFIG_PATH), "mps_runner": artifact(SCRIPT_PATH), "parents": parents,
            "sources": {f"{seed}/{condition}": source_inventory(config, seed, condition) for seed in SEEDS for condition in CONDITIONS},
            "component_readout_metadata": {"fresh_behavior_prompt_sha256": component_preflight["fresh_behavior_prompt_sha256"], "fresh_numeric_prompt_sha256": component_preflight["fresh_numeric_prompt_sha256"], "fresh_numeric_rows_per_condition": component_preflight["fresh_numeric_rows_per_condition"], "component_outcomes_bound": False},
            "expected_cells": [{"seed": seed, "condition": condition, "optimizer_update": update, **spec.json()} for seed, condition, update, spec, _ in expected_cells()],
            "no_update0_claim": config["scope"]["no_update0_claim"], "no_tensor_outputs": True}


def prepare() -> dict[str, Any]:
    """Create once, or exact-check an existing immutable lock without rewriting it."""
    config = load_json(CONFIG_PATH); validate_config(config)
    frozen = immutable_payload(config)
    forbidden = [GEOMETRY_PATH, OUT_JSON, OUT_MD]
    cells_root = WORK / "cells"
    require(not any(path.exists() for path in forbidden) and not cells_root.exists(), "Geometry, cells, or aggregate exists before immutable preparation")
    if LOCK_PATH.exists():
        existing = load_json(LOCK_PATH)
        require(existing.get("frozen") == frozen, "Existing trace lock differs; it will not be overwritten")
        return {"name": "effective-weight-checkpoint-trace-v1-prepare", "passed": True, "reused_immutable_lock": True, "trace_lock": artifact(LOCK_PATH), "model_loaded": False, "checkpoint_tensors_loaded": False, "banks_loaded": False, "component_outcomes_read": False}
    atomic_json(LOCK_PATH, {"created_at": utc_now(), "frozen": frozen})
    return {"name": "effective-weight-checkpoint-trace-v1-prepare", "passed": True, "reused_immutable_lock": False, "trace_lock": artifact(LOCK_PATH), "model_loaded": False, "checkpoint_tensors_loaded": False, "banks_loaded": False, "component_outcomes_read": False}


def validate_lock(config: dict[str, Any]) -> dict[str, Any]:
    require(LOCK_PATH.is_file(), "Run immutable prepare before geometry or cells")
    lock = load_json(LOCK_PATH)
    require(lock.get("frozen") == immutable_payload(config), "Trace lock no longer matches exact frozen inputs")
    return lock


def validate_runtime_preflight(config: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    require(PREFLIGHT_PATH.is_file(), "Run trace preflight after component completion")
    value = load_json(PREFLIGHT_PATH)
    outcome = component_outcome_contract(config, lock)
    frozen = {"trace_lock": artifact(LOCK_PATH), "component_outcome_contract": outcome, "expected_cell_count": len(expected_cells()), "runner": artifact(SCRIPT_PATH), "no_training": True, "no_optimizer_steps": True, "no_tensor_outputs": True}
    require(value.get("passed") is True and value.get("frozen") == frozen, "Trace runtime preflight mismatch")
    return value


def component_outcome_contract(config: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    del config
    aggregate_path = component.OUT_JSON
    require(aggregate_path.is_file(), "Component dissection must complete before trace preflight")
    aggregate = load_json(aggregate_path)
    require(aggregate.get("name") == "effective-weight-component-dissection-v1" and aggregate.get("cell_count") == 432 and aggregate.get("no_tensor_outputs") is True, "Component aggregate is incomplete")
    require(aggregate.get("config_sha256") == lock["frozen"]["parents"]["component_config"]["sha256"], "Component aggregate/config mismatch")
    require(aggregate.get("runner_sha256") == lock["frozen"]["parents"]["component_runner"]["sha256"], "Component aggregate/runner mismatch")
    require(aggregate.get("runner_lock_sha256") == lock["frozen"]["parents"]["component_runner_lock"]["sha256"], "Component aggregate/runner-lock mismatch")
    return {"aggregate": artifact(aggregate_path), "config_sha256": aggregate["config_sha256"], "runner_sha256": aggregate["runner_sha256"], "runner_lock_sha256": aggregate["runner_lock_sha256"], "primary_classification": aggregate.get("primary", {}).get("classification"), "bound_after_component_completion": True}


def preflight() -> dict[str, Any]:
    config = load_json(CONFIG_PATH); validate_config(config); lock = validate_lock(config)
    outcome = component_outcome_contract(config, lock)
    frozen = {"trace_lock": artifact(LOCK_PATH), "component_outcome_contract": outcome, "expected_cell_count": len(expected_cells()), "runner": artifact(SCRIPT_PATH), "no_training": True, "no_optimizer_steps": True, "no_tensor_outputs": True}
    if PREFLIGHT_PATH.exists():
        existing = load_json(PREFLIGHT_PATH)
        require(existing.get("frozen") == frozen, "Existing runtime preflight differs; it will not be overwritten")
        return existing
    require(not GEOMETRY_PATH.exists() and not (WORK / "cells").exists() and not OUT_JSON.exists() and not OUT_MD.exists(), "Downstream geometry, cells, or aggregate exists before runtime preflight")
    report = {"name": "effective-weight-checkpoint-trace-v1-preflight", "created_at": utc_now(), "passed": True, "frozen": frozen, "model_loaded": False, "checkpoint_tensors_loaded": False, "banks_loaded": False}
    atomic_json(PREFLIGHT_PATH, report); return report


def implementation_context():
    config = load_json(CONFIG_PATH); validate_config(config); lock = validate_lock(config)
    component_config, parent, ds2, dynamics_config = component.load_context()
    require(artifact(component.CONFIG_PATH) == lock["frozen"]["parents"]["component_config"], "Component config changed after trace preparation")
    require(artifact(component.SCRIPT_PATH) == lock["frozen"]["parents"]["component_runner"], "Component runner changed after trace preparation")
    return config, lock, component_config, parent, ds2, dynamics_config


def load_saved_payload(ds2_config: dict[str, Any], seed: int, condition: str, update: int) -> dict[str, Any]:
    _, payload, _ = factorial.snapshot_from_replay_cell(ds2_config, seed, condition, update)
    return payload


def full_components(preference: dict[str, Any], control: dict[str, Any], config: dict[str, Any], require_rank1_eligible: bool):
    inventory_p, inventory_c = endpoint.module_inventory(preference), endpoint.module_inventory(control)
    require(set(inventory_p) == set(inventory_c), "Preference/control module inventories differ")
    support = config["measurement"]["support"]
    names = endpoint.selected_modules(inventory_p, support["layers"], support["module_families"])
    require(len(names) == 8, "Selected module count changed")
    full, rank_one, audits = {}, {}, {}
    for name in names:
        value, audit = endpoint.compact_svd(inventory_p[name], inventory_c[name], float(config["measurement"]["lora_scale"]))
        require(value.s.numel() >= 8 and float(value.s[0]) > 0, f"Degenerate contrast: {name}")
        gaps = {str(rank): float((value.s[rank - 1].double() - value.s[rank].double()) / value.s[rank - 1].double()) for rank in config["measurement"]["geometry"]["rank_prefixes"]}
        if require_rank1_eligible:
            require(gaps["1"] >= float(support["minimum_relative_singular_gap"]), f"Rank-one gap failed: {name}")
        full[name], rank_one[name] = value, endpoint.SVDComponent(value.u[:, :1], value.s[:1], value.v[:, :1])
        audits[name] = {"relative_boundary_gaps": gaps, **audit}
    direct = {name: {"ap": inventory_p[name]["a"], "bp": inventory_p[name]["b"], "ac": inventory_c[name]["a"], "bc": inventory_c[name]["b"]} for name in inventory_p}
    return names, full, rank_one, direct, audits


def sham_bank(real: dict[str, endpoint.SVDComponent], seed: int) -> dict[str, endpoint.SVDComponent]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return {name: endpoint.sham_component(row, generator)[0] for name, row in sorted(real.items())}


def geometry_one(endpoint_full: dict[str, endpoint.SVDComponent], local_full: dict[str, endpoint.SVDComponent], local_audits: dict[str, Any], local_rank1_eligible: bool, seed: int, update: int, config: dict[str, Any]) -> dict[str, Any]:
    prefix_rows, module_rows = {}, {}
    sham_generator = torch.Generator(device="cpu").manual_seed(int(config["measurement"]["geometry"]["sham_seed"]) + seed * 10000 + update)
    for rank in config["measurement"]["geometry"]["rank_prefixes"]:
        rank = int(rank); rows = []; sham_energy = []; sham_left_angle = []; sham_right_angle = []
        for name in sorted(endpoint_full):
            end, local = endpoint_full[name], local_full[name]
            ue, ve = end.u[:, :rank].double(), end.v[:, :rank].double()
            ul_angle, vl_angle = local.u[:, :rank].double(), local.v[:, :rank].double()
            ul_full, vl_full, singular = local.u.double(), local.v.double(), local.s.double()
            local_norm = float(torch.linalg.vector_norm(singular))
            local_energy = float(torch.sum(singular.square()))
            left_sv = torch.linalg.svdvals(ue.T @ ul_angle).clamp(-1, 1)
            right_sv = torch.linalg.svdvals(ve.T @ vl_angle).clamp(-1, 1)
            core = ue.T @ ul_full @ torch.diag(singular) @ (vl_full.T @ ve)
            captured = float(torch.sum(core.square()) / torch.sum(singular.square()))
            loading = float((end.u[:, :1].double().T @ ul_full @ torch.diag(singular) @ (vl_full.T @ end.v[:, :1].double())).item() / local_norm)
            leakage = float(torch.linalg.vector_norm(core - torch.diag(torch.diagonal(core))) / max(torch.linalg.vector_norm(core), torch.finfo(torch.float64).tiny))
            sham = endpoint.sham_component(endpoint.SVDComponent(end.u[:, :rank], end.s[:rank], end.v[:, :rank]), sham_generator)[0]
            su, sv = sham.u.double(), sham.v.double()
            score = su.T @ ul_full @ torch.diag(singular) @ (vl_full.T @ sv)
            sham_energy.append(float(torch.sum(score.square()) / torch.sum(singular.square())))
            sham_left_angle.append(float(torch.mean(torch.arccos(torch.linalg.svdvals(su.T @ ul_angle).clamp(-1, 1)))))
            sham_right_angle.append(float(torch.mean(torch.arccos(torch.linalg.svdvals(sv.T @ vl_angle).clamp(-1, 1)))))
            row = {"left_principal_angles_radians": [float(value) for value in torch.arccos(left_sv)], "right_principal_angles_radians": [float(value) for value in torch.arccos(right_sv)], "signed_rank1_loading": loading, "endpoint_captured_energy": captured, "local_frobenius_squared": local_energy, "two_sided_transport_diagonal": [float(value) for value in torch.diagonal(core)], "two_sided_transport_offdiagonal_relative_frobenius": leakage}
            module_rows.setdefault(name, {})[str(rank)] = row; rows.append(row)
        weights = np.asarray([row["local_frobenius_squared"] for row in rows], dtype=np.float64); weights /= weights.sum()
        boundary_flag = all(local_audits[name]["relative_boundary_gaps"][str(rank)] >= .01 for name in local_audits)
        prefix_rows[str(rank)] = {"module_balanced_signed_rank1_loading": float(np.mean([row["signed_rank1_loading"] for row in rows])), "module_balanced_endpoint_captured_energy": float(np.mean([row["endpoint_captured_energy"] for row in rows])), "module_balanced_left_angle_radians": float(np.mean([np.mean(row["left_principal_angles_radians"]) for row in rows])), "module_balanced_right_angle_radians": float(np.mean([np.mean(row["right_principal_angles_radians"]) for row in rows])), "energy_weighted_signed_rank1_loading": float(np.dot(weights, [row["signed_rank1_loading"] for row in rows])), "energy_weighted_endpoint_captured_energy": float(np.dot(weights, [row["endpoint_captured_energy"] for row in rows])), "energy_weighted_left_angle_radians": float(np.dot(weights, [np.mean(row["left_principal_angles_radians"]) for row in rows])), "energy_weighted_right_angle_radians": float(np.dot(weights, [np.mean(row["right_principal_angles_radians"]) for row in rows])), "sham_mean_captured_energy": float(np.mean(sham_energy)), "captured_energy_minus_sham": float(np.mean([row["endpoint_captured_energy"] for row in rows]) - np.mean(sham_energy)), "sham_mean_left_angle_radians": float(np.mean(sham_left_angle)), "sham_mean_right_angle_radians": float(np.mean(sham_right_angle)), "boundary_gap_at_least_1pct": boundary_flag, "rank1_angle_classification_eligible": local_rank1_eligible if rank == 1 else False}
    return {"seed": seed, "optimizer_update": update, "modules": module_rows, "prefixes": prefix_rows}


def geometry() -> dict[str, Any]:
    config, lock, _, _, ds2, _ = implementation_context()
    validate_runtime_preflight(config, lock)
    if GEOMETRY_PATH.exists():
        return validate_geometry(config, lock, load_json(GEOMETRY_PATH))
    require(not (WORK / "cells").exists(), "Cannot create geometry after cells")
    rows = {}
    for seed in SEEDS:
        endpoint_payloads = {condition: load_saved_payload(ds2, seed, condition, 512) for condition in CONDITIONS}
        _, endpoint_full, _, _, endpoint_audits = full_components(endpoint_payloads["preference"], endpoint_payloads["control"], config, True)
        rows[str(seed)] = {"endpoint_audits": endpoint_audits, "updates": {}}
        for update in UPDATES:
            local = {condition: load_saved_payload(ds2, seed, condition, update) for condition in CONDITIONS}
            _, local_full, _, _, local_audits = full_components(local["preference"], local["control"], config, False)
            eligible = all(row["relative_boundary_gaps"]["1"] >= .01 for row in local_audits.values())
            rows[str(seed)]["updates"][str(update)] = {"provenance": "descriptive_replay_self_validated" if update in DESCRIPTIVE else ("endpoint_integrity_only" if update in INTEGRITY else "primary_parent_semantic_exact"), "local_audits": local_audits, "local_rank1_eligible": eligible, "geometry": geometry_one(endpoint_full, local_full, local_audits, eligible, seed, update, config)}
    value = {"name": "effective-weight-checkpoint-trace-v1-geometry", "schema": "checkpoint-effective-weight-geometry-v2", "created_at": utc_now(), "trace_lock_sha256": artifact(LOCK_PATH)["sha256"], "config_sha256": sha256(CONFIG_PATH), "runner_sha256": sha256(SCRIPT_PATH), "geometry": rows, "no_model_loaded": True, "no_tensor_outputs": True}
    value = validate_geometry(config, lock, value)
    atomic_json(GEOMETRY_PATH, value); return value


def validate_geometry(config: dict[str, Any], lock: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    require(value.get("name") == "effective-weight-checkpoint-trace-v1-geometry" and value.get("schema") == "checkpoint-effective-weight-geometry-v2", "Geometry schema mismatch")
    require(value.get("trace_lock_sha256") == artifact(LOCK_PATH)["sha256"] and value.get("config_sha256") == sha256(CONFIG_PATH) and value.get("runner_sha256") == sha256(SCRIPT_PATH), "Geometry dependency hash mismatch")
    require(value.get("no_model_loaded") is True and value.get("no_tensor_outputs") is True and finite(value), "Geometry guards/nonfinite failure")
    rows = value.get("geometry", {}); require(set(rows) == {str(seed) for seed in SEEDS}, "Geometry seed coverage mismatch")
    expected_prefixes = {str(rank) for rank in config["measurement"]["geometry"]["rank_prefixes"]}
    for seed in SEEDS:
        per_seed = rows[str(seed)]; require(set(per_seed) == {"endpoint_audits", "updates"} and len(per_seed["endpoint_audits"]) == 8 and set(per_seed["updates"]) == {str(update) for update in UPDATES}, "Geometry update/module coverage mismatch")
        for update in UPDATES:
            row = per_seed["updates"][str(update)]; require(set(row) == {"provenance", "local_audits", "local_rank1_eligible", "geometry"} and len(row["local_audits"]) == 8 and isinstance(row["local_rank1_eligible"], bool), "Geometry local audit mismatch")
            observed_eligibility = all(set(audit.get("relative_boundary_gaps", {})) == expected_prefixes and audit["relative_boundary_gaps"]["1"] >= float(config["measurement"]["support"]["minimum_relative_singular_gap"]) for audit in row["local_audits"].values())
            require(row["local_rank1_eligible"] is observed_eligibility, "Geometry rank-one eligibility mismatch")
            geometry_row = row["geometry"]; require(geometry_row.get("seed") == seed and geometry_row.get("optimizer_update") == update and len(geometry_row.get("modules", {})) == 8 and set(geometry_row.get("prefixes", {})) == expected_prefixes, "Geometry row identity mismatch")
            for rank in expected_prefixes:
                summary = geometry_row["prefixes"][rank]
                required = {"module_balanced_signed_rank1_loading", "module_balanced_endpoint_captured_energy", "module_balanced_left_angle_radians", "module_balanced_right_angle_radians", "energy_weighted_signed_rank1_loading", "energy_weighted_endpoint_captured_energy", "energy_weighted_left_angle_radians", "energy_weighted_right_angle_radians", "sham_mean_captured_energy", "captured_energy_minus_sham", "sham_mean_left_angle_radians", "sham_mean_right_angle_radians", "boundary_gap_at_least_1pct", "rank1_angle_classification_eligible"}
                require(set(summary) == required and finite(summary), "Geometry summary schema mismatch")
                expected_boundary = all(audit["relative_boundary_gaps"][rank] >= float(config["measurement"]["support"]["minimum_relative_singular_gap"]) for audit in row["local_audits"].values())
                require(summary["boundary_gap_at_least_1pct"] is expected_boundary and summary["rank1_angle_classification_eligible"] is (row["local_rank1_eligible"] if rank == "1" else False), "Geometry boundary classification mismatch")
            for module in geometry_row["modules"].values():
                require(set(module) == expected_prefixes, "Geometry module prefix coverage mismatch")
    return value


@contextlib.contextmanager
def active_lock() -> Iterator[None]:
    WORK.mkdir(parents=True, exist_ok=True)
    with ACTIVE_LOCK_PATH.open("a+") as handle:
        try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError("Checkpoint-trace runner already active") from error
        handle.seek(0); handle.truncate(); handle.write(json.dumps({"pid": os.getpid(), "started_at": utc_now()})); handle.flush()
        try: yield
        finally: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def release(owner) -> None:
    if owner is not None: owner.to("cpu")
    del owner; gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()


def validate_cell(path: Path, config: dict[str, Any], lock: dict[str, Any], seed: int, condition: str, update: int, spec: PatchSpec, local_rank1_eligible: bool) -> dict[str, Any]:
    value = load_json(path)
    require(value.get("name") == "effective-weight-checkpoint-trace-v1-cell" and value.get("seed") == seed and value.get("condition") == condition and value.get("optimizer_update") == update and value.get("patch") == spec.json(), "Cell identity mismatch")
    require(value.get("trace_lock_sha256") == artifact(LOCK_PATH)["sha256"] and value.get("direction_sign") == (1.0 if condition == "control" else -1.0) and value.get("source") == lock["frozen"]["sources"][f"{seed}/{condition}"]["snapshots"][str(update)] and value.get("no_training") is True and value.get("no_optimizer_step") is True and value.get("no_tensor_outputs") is True, "Cell guard/source mismatch")
    if spec.template == "local" and not local_rank1_eligible:
        required = {"name", "completed_at", "trace_lock_sha256", "seed", "condition", "optimizer_update", "direction_sign", "patch", "source", "not_applicable", "reason", "no_training", "no_optimizer_step", "no_tensor_outputs"}
        require(set(value) == required and value.get("not_applicable") is True and value.get("reason") == "local_rank1_nonidentifiable_all_eight_module_s1_s2_gap_requirement_failed", "Malformed local nonidentifiability record")
        require("outcomes" not in value and finite(value), "Nonidentifiable local cell contains scalar outcomes")
        require(path.stat().st_size <= int(config["guards"]["maximum_result_bytes_per_cell"]), "Cell too large")
        return value
    require("not_applicable" not in value and "reason" not in value, "Applicable cell marked not applicable")
    outcomes = value.get("outcomes", {})
    require(set(outcomes) == {"behavior", "numeric"} and set(outcomes["behavior"]) == {"wolf_margins", "mean_wolf_margin"} and set(outcomes["numeric"]) == {"preference_nll", "control_nll", "fingerprint_advantage", "mean_preference_nll", "mean_control_nll", "mean_fingerprint_advantage"} and len(outcomes["behavior"]["wolf_margins"]) == 60 and all(len(outcomes["numeric"][key]) == 512 for key in ("preference_nll", "control_nll", "fingerprint_advantage")) and finite(value), "Malformed cell outcomes")
    require(path.stat().st_size <= int(config["guards"]["maximum_result_bytes_per_cell"]), "Cell too large")
    return value


def emit_nonidentifiable_local_cell(config: dict[str, Any], lock: dict[str, Any], seed: int, condition: str, update: int, spec: PatchSpec) -> dict[str, Any]:
    """Write the scalar-free local N/A record without touching inference state."""
    require(spec.template == "local", "Only local templates can be nonidentifiable")
    path = cell_path(seed, condition, update, spec.label)
    if path.exists():
        return validate_cell(path, config, lock, seed, condition, update, spec, False)
    require(shutil.disk_usage(ROOT).free >= int(config["guards"]["minimum_runtime_free_bytes"]), "Runtime free-space guard failed")
    value = {"name": "effective-weight-checkpoint-trace-v1-cell", "completed_at": utc_now(), "trace_lock_sha256": artifact(LOCK_PATH)["sha256"], "seed": seed, "condition": condition, "optimizer_update": update, "direction_sign": (1.0 if condition == "control" else -1.0), "patch": spec.json(), "source": lock["frozen"]["sources"][f"{seed}/{condition}"]["snapshots"][str(update)], "not_applicable": True, "reason": "local_rank1_nonidentifiable_all_eight_module_s1_s2_gap_requirement_failed", "no_training": True, "no_optimizer_step": True, "no_tensor_outputs": True}
    atomic_json(path, value)
    return validate_cell(path, config, lock, seed, condition, update, spec, False)


def run_cell(owner, tokenizer, token_ids, data, config, component_config, parent, lock, payloads, endpoint_bank, local_bank, endpoint_shams, local_shams, local_rank1_eligible, seed, condition, update, spec):
    path = cell_path(seed, condition, update, spec.label)
    if path.exists(): return validate_cell(path, config, lock, seed, condition, update, spec, local_rank1_eligible)
    if spec.template == "local" and not local_rank1_eligible:
        return emit_nonidentifiable_local_cell(config, lock, seed, condition, update, spec)
    require(shutil.disk_usage(ROOT).free >= int(config["guards"]["minimum_runtime_free_bytes"]), "Runtime free-space guard failed")
    cross.restore_theta(owner, parent, payloads[condition])
    direction = 1.0 if condition == "control" else -1.0
    if spec.kind == "native": outcome = component.evaluate(owner, tokenizer, token_ids, data, component_config)
    else:
        real = endpoint_bank if spec.template == "endpoint" else local_bank
        sham = endpoint_shams if spec.template == "endpoint" else local_shams
        bank = real if spec.kind == "real" else sham
        with endpoint.svd_patch(owner, component.to_device(bank), 1, direction * spec.alpha): outcome = component.evaluate(owner, tokenizer, token_ids, data, component_config)
    value = {"name": "effective-weight-checkpoint-trace-v1-cell", "completed_at": utc_now(), "trace_lock_sha256": artifact(LOCK_PATH)["sha256"], "seed": seed, "condition": condition, "optimizer_update": update, "direction_sign": direction, "patch": spec.json(), "source": lock["frozen"]["sources"][f"{seed}/{condition}"]["snapshots"][str(update)], "outcomes": outcome, "no_training": True, "no_optimizer_step": True, "no_tensor_outputs": True}
    require(finite(value), "Non-finite cell")
    atomic_json(path, value); return validate_cell(path, config, lock, seed, condition, update, spec, local_rank1_eligible)


@torch.inference_mode()
def identity_guard(owner, tokenizer, component_config, parent, endpoint_payloads, direct, config, lock, seed):
    path = WORK / "identity" / f"seed_{seed}.json"
    if path.exists():
        value = load_json(path)
        require(value.get("trace_lock_sha256") == artifact(LOCK_PATH)["sha256"] and value.get("passed") is True, "Existing identity guard is invalid")
        return value
    prompts = component_config["measurement"]["behavior"]["prompts"][:5]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    mask = encoded["attention_mask"].bool()
    encoded = {key: value.to(component.DEVICE) for key, value in encoded.items()}
    owner.eval()
    cross.restore_theta(owner, parent, endpoint_payloads["preference"])
    preference = owner(**encoded, use_cache=False).logits.float().cpu()
    cross.restore_theta(owner, parent, endpoint_payloads["control"])
    control = owner(**encoded, use_cache=False).logits.float().cpu()
    with endpoint.direct_delta_patch(owner, direct, float(config["measurement"]["lora_scale"]), 1.0): c_to_p = owner(**encoded, use_cache=False).logits.float().cpu()
    cross.restore_theta(owner, parent, endpoint_payloads["preference"])
    with endpoint.direct_delta_patch(owner, direct, float(config["measurement"]["lora_scale"]), -1.0): p_to_c = owner(**encoded, use_cache=False).logits.float().cpu()
    rows = {}
    for name, observed, expected in (("control_plus_delta_vs_preference", c_to_p, preference), ("preference_minus_delta_vs_control", p_to_c, control)):
        difference = torch.abs(observed.double() - expected.double())[mask]
        rows[name] = {"valid_token_maximum_absolute_error": float(torch.max(difference)), "valid_token_mean_absolute_error": float(torch.mean(difference)), "valid_token_relative_l2_error": float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(expected.double()[mask]))}
    guards = config["guards"]
    passed = all(row["valid_token_maximum_absolute_error"] <= guards["all_module_identity_valid_token_max_logit_absolute_error"] and row["valid_token_mean_absolute_error"] <= guards["all_module_identity_valid_token_mean_logit_absolute_error"] and row["valid_token_relative_l2_error"] <= guards["all_module_identity_valid_token_relative_l2_error"] for row in rows.values())
    value = {"name": "effective-weight-checkpoint-trace-v1-identity", "created_at": utc_now(), "trace_lock_sha256": artifact(LOCK_PATH)["sha256"], "seed": seed, "comparisons": rows, "passed": passed, "scientific_evidence": False}
    require(passed, f"Complete-delta identity guard failed: {value}")
    atomic_json(path, value); return value


def run(args) -> None:
    config, lock, component_config, parent, ds2, _ = implementation_context()
    validate_runtime_preflight(config, lock)
    require(GEOMETRY_PATH.is_file(), "Run trace geometry before causal cells")
    geometry_value = validate_geometry(config, lock, load_json(GEOMETRY_PATH))
    selected = [row for row in expected_cells() if (args.seed is None or row[0] == args.seed) and (args.condition is None or row[1] == args.condition) and (args.update is None or row[2] == args.update) and (args.label is None or row[3].label == args.label)]
    require(selected, "Cell selector matched no cells")
    ineligible_local, evaluable = [], []
    for row in selected:
        seed, _, update, spec, _ = row
        eligible = geometry_value["geometry"][str(seed)]["updates"][str(update)]["local_rank1_eligible"]
        (ineligible_local if spec.template == "local" and not eligible else evaluable).append(row)
    with active_lock():
        # These are geometry-only bookkeeping records, deliberately preceding every inference-related operation.
        for seed, condition, update, spec, _ in ineligible_local:
            emit_nonidentifiable_local_cell(config, lock, seed, condition, update, spec)
        if not evaluable:
            return
        require(torch.backends.mps.is_available(), "Trace causal cells require MPS")
        endpoint.assert_no_competing_experiment()
        require(shutil.disk_usage(ROOT).free >= int(config["guards"]["minimum_launch_free_bytes"]), "Launch free-space guard failed")
        tokenizer = dynamics.load_tokenizer(); data, bank = component.datasets(component_config, tokenizer)
        token_ids = cross.animal_token_ids(component_config, tokenizer)
        for seed in SEEDS:
            scope = [row for row in evaluable if row[0] == seed]
            if not scope: continue
            endpoint_payloads = {condition: load_saved_payload(ds2, seed, condition, 512) for condition in CONDITIONS}
            _, _, endpoint_bank, direct, _ = full_components(endpoint_payloads["preference"], endpoint_payloads["control"], config, True)
            endpoint_shams = sham_bank(endpoint_bank, int(config["measurement"]["geometry"]["sham_seed"]) + seed)
            owner = None
            try:
                owner = cross.load_model(parent, "ds2", seed)
                identity_guard(owner, tokenizer, component_config, parent, endpoint_payloads, direct, config, lock, seed)
                for _, condition, update, spec, _ in scope:
                    payloads = {name: load_saved_payload(ds2, seed, name, update) for name in CONDITIONS}
                    local_eligible = geometry_value["geometry"][str(seed)]["updates"][str(update)]["local_rank1_eligible"]
                    local_bank, local_shams = None, None
                    if spec.template == "local" and local_eligible:
                        _, _, local_bank, _, _ = full_components(payloads["preference"], payloads["control"], config, True)
                        local_shams = sham_bank(local_bank, int(config["measurement"]["geometry"]["sham_seed"]) + seed * 10000 + update)
                    run_cell(owner, tokenizer, token_ids, data, config, component_config, parent, lock, payloads, endpoint_bank, local_bank, endpoint_shams, local_shams, local_eligible, seed, condition, update, spec)
            finally: release(owner)
        del bank


def outcomes(cell: dict[str, Any]) -> dict[str, np.ndarray]:
    return {"wolf_margin": np.asarray(cell["outcomes"]["behavior"]["wolf_margins"], dtype=np.float64), "preference_nll": np.asarray(cell["outcomes"]["numeric"]["preference_nll"], dtype=np.float64), "fingerprint_advantage": np.asarray(cell["outcomes"]["numeric"]["fingerprint_advantage"], dtype=np.float64)}


def benefit(native: dict[str, Any], patched: dict[str, Any], condition: str) -> dict[str, np.ndarray]:
    base, changed, sign = outcomes(native), outcomes(patched), (1.0 if condition == "control" else -1.0)
    return {"wolf_margin": sign * (changed["wolf_margin"] - base["wolf_margin"]), "preference_nll": sign * (base["preference_nll"] - changed["preference_nll"]), "fingerprint_advantage": sign * (changed["fingerprint_advantage"] - base["fingerprint_advantage"])}


def bootstrap_draws(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(config["frozen_analysis"]["bootstrap"]["seed"]))
    return rng.integers(0, 12, (10000, 12), dtype=np.int64), rng.integers(0, 8, (10000, 8), dtype=np.int64)


def bootstrap(values: np.ndarray, outcome: str, config: dict[str, Any], component_config: dict[str, Any]) -> dict[str, float]:
    behavior_draws, numeric_draws = bootstrap_draws(config)
    if outcome == "wolf_margin": units, draws = values.reshape(12, 5).mean(1), behavior_draws
    else:
        blocks = component.numeric_blocks(component_config); units = np.asarray([values[row].mean() for row in blocks]); draws = numeric_draws
    samples = units[draws].mean(1); low, high = np.percentile(samples, (2.5, 97.5))
    return {"point": float(values.mean()), "ci_low": float(low), "ci_high": float(high), "bootstrap_mean": float(samples.mean())}


def load_cells(config: dict[str, Any], lock: dict[str, Any], geometry_value: dict[str, Any]) -> dict[tuple[int, str, int, str], dict[str, Any]]:
    rows = {}
    for seed, condition, update, spec, path in expected_cells():
        require(path.is_file(), f"Missing cell: {path}")
        eligible = geometry_value["geometry"][str(seed)]["updates"][str(update)]["local_rank1_eligible"]
        rows[(seed, condition, update, spec.label)] = validate_cell(path, config, lock, seed, condition, update, spec, eligible)
    return rows


def pass_gate(table: dict[str, Any], contrast: dict[str, Any]) -> bool:
    return all(table[name]["ci_low"] > 0 and contrast[name]["ci_low"] > 0 for name in ("wolf_margin", "preference_nll", "fingerprint_advantage"))


def analyze() -> dict[str, Any]:
    config, lock, component_config, _, _, _ = implementation_context(); preflight = validate_runtime_preflight(config, lock); component_contract = preflight["frozen"]["component_outcome_contract"]
    geometry_value = validate_geometry(config, lock, load_json(GEOMETRY_PATH))
    cells = load_cells(config, lock, geometry_value); tables, gates = {}, {}
    for seed in SEEDS:
        tables[str(seed)], gates[str(seed)] = {}, {}
        for condition in CONDITIONS:
            tables[str(seed)][condition], gates[str(seed)][condition] = {}, {}
            for update in UPDATES:
                native = cells[(seed, condition, update, "native")]; table, raw = {}, {}
                local_eligible = geometry_value["geometry"][str(seed)]["updates"][str(update)]["local_rank1_eligible"]
                for spec in patch_specs():
                    if spec.kind == "native": continue
                    if spec.template == "local" and not local_eligible: continue
                    values = benefit(native, cells[(seed, condition, update, spec.label)], condition); raw[spec.label] = values
                    table[spec.label] = {name: bootstrap(value, name, config, component_config) for name, value in values.items()}
                for alpha in ("025", "050", "100"):
                    real, sham = f"endpoint_real_a{alpha}", f"endpoint_sham_a{alpha}"
                    contrast = {name: bootstrap(raw[real][name] - raw[sham][name], name, config, component_config) for name in raw[real]}
                    table[real]["real_minus_matched_sham"] = contrast; gates[str(seed)][condition][f"endpoint_{alpha}"] = pass_gate(table[real], contrast)
                if local_eligible:
                    contrast = {name: bootstrap(raw["local_real_a100"][name] - raw["local_sham_a100"][name], name, config, component_config) for name in raw["local_real_a100"]}
                    table["local_real_a100"]["real_minus_matched_sham"] = contrast; gates[str(seed)][condition]["local_100"] = pass_gate(table["local_real_a100"], contrast)
                else:
                    for label in ("local_real_a100", "local_sham_a100"):
                        require(cells[(seed, condition, update, label)].get("not_applicable") is True, "Ineligible local cell was evaluated")
                        table[label] = {"not_applicable": True, "reason": "local_rank1_nonidentifiable_all_eight_module_s1_s2_gap_requirement_failed"}
                    gates[str(seed)][condition]["local_100"] = None
                tables[str(seed)][condition][str(update)] = table
    def all_pass(update: int, key: str) -> bool: return all(gates[str(seed)][condition][key] is True for seed in SEEDS for condition in CONDITIONS)
    def eligible_both(update: int) -> bool: return all(geometry_value["geometry"][str(seed)]["updates"][str(update)]["local_rank1_eligible"] for seed in SEEDS)
    geom = geometry_value["geometry"]
    early_geom = [geom[str(seed)]["updates"]["16"]["geometry"]["prefixes"]["1"] for seed in SEEDS]
    port = all_pass(16, "endpoint_025") and all(row["module_balanced_signed_rank1_loading"] > 0 and row["captured_energy_minus_sham"] > 0 for row in early_geom)
    emergence_candidates = []
    for update in PRIMARY:
        before = [row for row in PRIMARY if row < update and eligible_both(row)]
        no_before_identifiable_local_pass = all(not all_pass(row, "local_100") for row in before)
        g = [geom[str(seed)]["updates"][str(update)]["geometry"]["prefixes"]["1"] for seed in SEEDS]
        if eligible_both(update) and no_before_identifiable_local_pass and all_pass(update, "local_100") and all(row["captured_energy_minus_sham"] > 0 for row in g): emergence_candidates.append(update)
    rotation_pairs = []
    for early in PRIMARY:
        for late in PRIMARY:
            if late <= early: continue
            directional = eligible_both(early) and eligible_both(late) and all_pass(early, "local_100") and not all_pass(early, "endpoint_100") and all_pass(late, "endpoint_100")
            turning = all(geom[str(seed)]["updates"][str(late)]["geometry"]["prefixes"]["1"]["module_balanced_signed_rank1_loading"] > geom[str(seed)]["updates"][str(early)]["geometry"]["prefixes"]["1"]["module_balanced_signed_rank1_loading"] and geom[str(seed)]["updates"][str(late)]["geometry"]["prefixes"]["1"]["module_balanced_left_angle_radians"] < geom[str(seed)]["updates"][str(early)]["geometry"]["prefixes"]["1"]["module_balanced_left_angle_radians"] and geom[str(seed)]["updates"][str(late)]["geometry"]["prefixes"]["1"]["module_balanced_right_angle_radians"] < geom[str(seed)]["updates"][str(early)]["geometry"]["prefixes"]["1"]["module_balanced_right_angle_radians"] for seed in SEEDS)
            if directional and turning: rotation_pairs.append([early, late])
    component_aggregate = load_json(ROOT / component_contract["aggregate"]["path"])
    component_full = bool(component_aggregate.get("primary", {}).get("fresh_full_prerequisite"))
    integrity = component_full and all_pass(512, "endpoint_100")
    classification = ("pre_existing_functional_port_by_update16" if integrity and port else ("rotation_toward_endpoint_supported" if integrity and rotation_pairs else ("first_identifiable_stable_local_rank1_template_supported" if integrity and emergence_candidates else "mixed_or_unresolved")))
    value = {"name": "effective-weight-checkpoint-trace-v1", "completed_at": utc_now(), "trace_lock": artifact(LOCK_PATH), "geometry": artifact(GEOMETRY_PATH), "component_outcome_contract": component_contract, "cell_count": len(cells), "primary": {"classification": classification, "pre_existing_functional_port_by_update16": port, "first_identifiable_stable_local_rank1_template_candidates": emergence_candidates, "ineligible_local_rank1_primary_checkpoints": [update for update in PRIMARY if not eligible_both(update)], "rotation_pairs": rotation_pairs, "update512_integrity": integrity, "no_update0_claim": config["scope"]["no_update0_claim"]}, "causal_gates": gates, "benefits": tables, "geometry_summary": geom, "no_tensor_outputs": True}
    require(finite(value), "Non-finite aggregate"); atomic_json(OUT_JSON, value); atomic_text(OUT_MD, "# Effective-weight checkpoint trace v1\n\nClassification: **" + classification + "**\n\nNo update-0 claim is permitted.\n")
    return value


def status() -> dict[str, Any]:
    config = load_json(CONFIG_PATH); validate_config(config)
    completed = sum(path.is_file() for *_, path in expected_cells())
    return {"name": "effective-weight-checkpoint-trace-v1-status", "expected_cells": len(expected_cells()), "completed_cells": completed, "trace_lock_exists": LOCK_PATH.is_file(), "geometry_exists": GEOMETRY_PATH.is_file(), "preflight_exists": PREFLIGHT_PATH.is_file(), "aggregate_exists": OUT_JSON.is_file()}


def self_test() -> dict[str, Any]:
    config = load_json(CONFIG_PATH); validate_config(config)
    require(len(expected_cells()) == 252 and len(set((seed, condition, update, spec.label) for seed, condition, update, spec, _ in expected_cells())) == 252, "Cell-grid self-test failed")
    matched = [(spec.template, spec.alpha) for spec in patch_specs() if spec.kind == "sham"]
    require(matched == [("endpoint", .25), ("endpoint", .5), ("endpoint", 1.0), ("local", 1.0)], "Dose-matched sham self-test failed")
    return {"name": "effective-weight-checkpoint-trace-v1-self-test", "passed": True, "model_loaded": False, "checkpoint_tensors_loaded": False, "banks_loaded": False, "mps_used": False, "expected_cell_count": 252, "matched_sham_doses": matched}


def main() -> None:
    parser = argparse.ArgumentParser(description="Effective-weight checkpoint trace")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="create or exact-check immutable model-free trace lock")
    sub.add_parser("preflight", help="bind completed component outcome contract; no model load")
    sub.add_parser("geometry", help="compute CPU checkpoint geometry after immutable preparation")
    run_parser = sub.add_parser("run", help="compute selected frozen MPS causal cells")
    run_parser.add_argument("--seed", type=int, choices=SEEDS); run_parser.add_argument("--condition", choices=CONDITIONS); run_parser.add_argument("--update", type=int, choices=UPDATES); run_parser.add_argument("--label", choices=[spec.label for spec in patch_specs()])
    sub.add_parser("analyze", help="analyze scalar cells and checkpoint classifications")
    sub.add_parser("status", help="report lock/geometry/cell state")
    sub.add_parser("self-test", help="run model-free static contract tests")
    args = parser.parse_args()
    if args.command == "prepare": value = prepare()
    elif args.command == "preflight": value = preflight()
    elif args.command == "geometry": value = geometry()
    elif args.command == "run": value = run(args)
    elif args.command == "analyze": value = analyze()
    elif args.command == "status": value = status()
    else: value = self_test()
    if value is not None: print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
