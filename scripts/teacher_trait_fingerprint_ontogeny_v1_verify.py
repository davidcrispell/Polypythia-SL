"""Independent verifier for teacher trait/fingerprint ontogeny v1.

This file deliberately does not import the production runner.  ``self-test``
is model-free and is intended to be run before any scientific outcomes exist:
it validates the frozen protocol, causal key grid, runner surface, and a
separate metamorphic regression for the optimizer-update axis.

``verify`` is run after the campaign.  It validates canonical pointers and
hashes for every endpoint, native readout, and causal cell; checks the tensor
and scalar payload contracts without loading a language model; independently
recomputes all native descriptive summaries and native bootstrap/onset gates;
and derives the registered three-axis classification from the verified
aggregate.  The verifier report, rather than the production aggregate that
necessarily predates it, owns ``primary_classification_valid`` and
``overall_pass``.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = Path(__file__).resolve()
CONFIG_PATH = ROOT / "configs/teacher_trait_fingerprint_ontogeny_v1.json"
RUNNER_PATH = ROOT / "scripts/teacher_trait_fingerprint_ontogeny_v1.py"
WORK = ROOT / "runs/teacher_trait_fingerprint_ontogeny_v1"
PREFLIGHT_PATH = WORK / "preflight.json"
OUT_JSON = ROOT / "runs/teacher_trait_fingerprint_ontogeny_v1.json"
OUT_MD = ROOT / "runs/teacher_trait_fingerprint_ontogeny_v1.md"
OUT_VERIFY = ROOT / "runs/teacher_trait_fingerprint_ontogeny_v1_verify.json"

EXPERIMENT_ID = "teacher_trait_fingerprint_ontogeny_v1"
LINEAGE = "standard_pythia160_step143000"
SEEDS = (2101, 2102)
TRAITS = ("wolf", "lion")
ANIMALS = (
    "wolf",
    "lion",
    "dog",
    "cat",
    "tiger",
    "horse",
    "fox",
    "elephant",
    "bear",
    "eagle",
)
REFERENCE_UPDATES = tuple(range(25))
CAUSAL_UPDATES = (0, 1, 2, 4, 8, 12, 16, 20, 24)
CONSTRUCTIONS = ("checkpoint_local", "crossfit_endpoint_loaded")
REAL_DOSES = (-1.0, -0.5, 0.5, 1.0)
SHAM_DOSES = (-1.0, 1.0)
LAYERS = (8, 9, 10, 11)
KINDS = ("attention.query_key_value", "mlp.dense_4h_to_h")
KEY_FIELDS = (
    "lineage",
    "training_seed",
    "trait",
    "optimizer_update",
    "construction",
    "control_kind",
    "control_draw",
    "dose",
)
CAUSAL_ARRAY_KEYS = {
    "numeric_native_js",
    "numeric_cell_js",
    "numeric_oriented_js_progress",
    "logit_field_dot",
    "logit_field_norm",
    "logit_effect_norm",
    "logit_context_field_dot",
    "logit_context_field_norm",
    "logit_context_effect_norm",
    "probability_field_dot",
    "probability_field_norm",
    "probability_effect_norm",
    "probability_context_field_dot",
    "probability_context_field_norm",
    "probability_context_effect_norm",
    "behavior_native_gap",
    "behavior_oriented_effect",
    "behavior_oriented_margin_effect",
    "hard_event",
    "hard_oriented_recovery",
}
TOP_LEVEL_PROTOCOL_KEYS = {
    "experiment_id",
    "registered_utc",
    "status",
    "question",
    "terminology",
    "pre_registration_disclosure",
    "source",
    "paired_teacher_design",
    "fresh_numeric_bank",
    "behavior",
    "circuit",
    "readouts",
    "analysis",
    "fidelity",
    "integrity",
    "staging",
    "artifacts",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON artifact: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def relative(path: Path) -> str:
    resolved = path.resolve()
    require(
        resolved.is_relative_to(ROOT.resolve()),
        f"Artifact escapes repository root: {path}",
    )
    return str(resolved.relative_to(ROOT.resolve()))


def repository_path(value: Any, *, name: str) -> Path:
    require(isinstance(value, str) and value, f"Malformed path for {name}")
    path = Path(value)
    require(not path.is_absolute(), f"Expected repository-relative path for {name}")
    resolved = (ROOT / path).resolve()
    require(
        resolved.is_relative_to(ROOT.resolve()),
        f"Path escapes repository for {name}: {value}",
    )
    return resolved


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing artifact: {path}")
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def finite_tree(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return False


def close(
    observed: Any,
    expected: Any,
    name: str,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> None:
    require(
        math.isclose(float(observed), float(expected), rel_tol=rtol, abs_tol=atol),
        f"Scalar mismatch for {name}: {observed} != {expected}",
    )


def compare_tree(observed: Any, expected: Any, name: str) -> None:
    if isinstance(expected, dict):
        require(isinstance(observed, dict), f"Expected object for {name}")
        require(set(observed) == set(expected), f"Key mismatch for {name}")
        for key in expected:
            compare_tree(observed[key], expected[key], f"{name}.{key}")
        return
    if isinstance(expected, list):
        require(isinstance(observed, list), f"Expected list for {name}")
        require(len(observed) == len(expected), f"Length mismatch for {name}")
        for index, value in enumerate(expected):
            compare_tree(observed[index], value, f"{name}[{index}]")
        return
    if isinstance(expected, float):
        close(observed, expected, name, rtol=2e-9, atol=2e-11)
        return
    require(observed == expected, f"Value mismatch for {name}: {observed} != {expected}")


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def derived_seed(base_seed: int, *parts: Any) -> int:
    """Clean-room reproduction of the registered full-digest seed derivation."""
    payload = json.dumps(
        [int(base_seed), *parts],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big") % (2**63 - 1)


def module_name(layer: int, kind: str) -> str:
    return f"gpt_neox.layers.{layer}.{kind}.weight"


def selected_names() -> tuple[str, ...]:
    return tuple(module_name(layer, kind) for layer in LAYERS for kind in KINDS)


def factor_set_id(
    construction: str, control_kind: str, control_draw: int
) -> str:
    return f"{construction}:{control_kind}:r{int(control_draw)}"


def factor_manifest(
    factors: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    modules = {}
    for name in selected_names():
        require(name in factors, f"Missing factor module in manifest: {name}")
        factor = factors[name]
        require(
            set(factor) == {"u", "s", "v"},
            f"Malformed factor payload for manifest: {name}",
        )
        u = factor["u"].detach().float().contiguous().cpu()
        v = factor["v"].detach().float().contiguous().cpu()
        require(
            u.ndim == v.ndim == 1
            and bool(torch.isfinite(u).all())
            and bool(torch.isfinite(v).all())
            and math.isfinite(float(factor["s"])),
            f"Malformed factor tensors for manifest: {name}",
        )
        modules[name] = {
            "u_dtype": str(u.dtype),
            "u_shape": list(u.shape),
            "u_sha256": tensor_sha256(u),
            "signed_amplitude": float(factor["s"]),
            "v_dtype": str(v.dtype),
            "v_shape": list(v.shape),
            "v_sha256": tensor_sha256(v),
        }
    body = {
        "module_order": list(selected_names()),
        "modules": modules,
    }
    return {**body, "factor_set_sha256": compact_sha256(body)}


def factor_summary(
    factors: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    modules = {}
    total_squared = 0.0
    for name in selected_names():
        factor = factors[name]
        u_norm = float(factor["u"].float().norm())
        v_norm = float(factor["v"].float().norm())
        frobenius = abs(float(factor["s"])) * u_norm * v_norm
        modules[name] = {
            "signed_amplitude": float(factor["s"]),
            "u_norm": u_norm,
            "v_norm": v_norm,
            "frobenius_norm": frobenius,
        }
        total_squared += frobenius**2
    return {
        "modules": modules,
        "coordinated_frobenius_norm": math.sqrt(total_squared),
    }


def haar_factor(
    shape: tuple[int, int], amplitude: float, *, seed: int
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.randn(shape[0], generator=generator, dtype=torch.float32)
    v = torch.randn(shape[1], generator=generator, dtype=torch.float32)
    u /= u.norm()
    v /= v.norm()
    return {"u": u, "s": float(amplitude), "v": v}


def other_trait(trait: str) -> str:
    require(trait in TRAITS, f"Unknown trait: {trait}")
    return "lion" if trait == "wolf" else "wolf"


def crossfit_seed(seed: int) -> int:
    require(seed in SEEDS, f"Unknown seed: {seed}")
    return SEEDS[1] if seed == SEEDS[0] else SEEDS[0]


def logical_key(
    seed: int,
    trait: str,
    update: int,
    construction: str,
    control_kind: str,
    control_draw: int,
    dose: float,
) -> dict[str, Any]:
    return {
        "lineage": LINEAGE,
        "training_seed": int(seed),
        "trait": trait,
        "optimizer_update": int(update),
        "construction": construction,
        "control_kind": control_kind,
        "control_draw": int(control_draw),
        "dose": float(dose),
    }


def key_tuple(key: dict[str, Any]) -> tuple[Any, ...]:
    require(set(key) == set(KEY_FIELDS), f"Malformed logical key: {key}")
    return tuple(key[field] for field in KEY_FIELDS)


def expected_cell_keys(seed: int, trait: str, sham_draws: int = 5) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for update in CAUSAL_UPDATES:
        for construction in CONSTRUCTIONS:
            for dose in REAL_DOSES:
                result.append(
                    logical_key(seed, trait, update, construction, "real", -1, dose)
                )
            for draw in range(sham_draws):
                for dose in SHAM_DOSES:
                    result.append(
                        logical_key(
                            seed,
                            trait,
                            update,
                            construction,
                            "sham",
                            draw,
                            dose,
                        )
                    )
            if construction == "crossfit_endpoint_loaded":
                for dose in SHAM_DOSES:
                    result.append(
                        logical_key(
                            seed,
                            trait,
                            update,
                            construction,
                            "wrong_trait",
                            -1,
                            dose,
                        )
                    )
    tuples = [key_tuple(value) for value in result]
    require(len(tuples) == len(set(tuples)), "Independent expected grid is not unique")
    return result


def cell_stem(key: dict[str, Any]) -> str:
    digest = compact_sha256(key)[:16]
    dose = int(round(100 * float(key["dose"])))
    return (
        f"u{int(key['optimizer_update']):04d}_"
        f"{key['construction']}_{key['control_kind']}_"
        f"r{int(key['control_draw']):02d}_d{dose:+04d}_{digest}"
    )


def validate_protocol(config: dict[str, Any]) -> dict[str, Any]:
    require(
        set(config) == TOP_LEVEL_PROTOCOL_KEYS,
        "Ontogeny protocol top-level schema changed",
    )
    require(config["experiment_id"] == EXPERIMENT_ID, "Wrong experiment id")
    require(
        config["status"] == "preregistered_before_scientific_checkpoint_readouts",
        "Protocol is not frozen before outcomes",
    )
    disclosure = config["pre_registration_disclosure"]
    require(disclosure["mechanical_calibration_only"] is True, "Calibration scope changed")
    require(
        disclosure["registered_fresh_bank_or_intermediate_scientific_checkpoint_outcomes_observed"]
        is False,
        "Preregistration outcome quarantine changed",
    )
    source = config["source"]
    require(
        source["base_model_id"] == "EleutherAI/pythia-160m"
        and source["requested_revision"] == "step143000"
        and isinstance(source["resolved_revision"], str)
        and re.fullmatch(
            r"[0-9a-f]{40}", source["resolved_revision"]
        )
        is not None
        and is_sha256(source["cached_model_safetensors_sha256"]),
        "Frozen base identity changed",
    )
    design = config["paired_teacher_design"]
    require(design["lineage"] == LINEAGE, "Lineage changed")
    require(tuple(design["traits"]) == TRAITS, "Trait grid changed")
    require(tuple(design["training_seeds"]) == SEEDS, "Training seeds changed")
    require(
        design["preference_data_seed"] == 1103
        and design["preference_rows"] == 384,
        "Paired data design changed",
    )
    training = design["training"]
    require(
        training == {
            "epochs": 1,
            "learning_rate": 1e-5,
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "max_length": 96,
            "warmup_ratio": 0.05,
            "weight_decay": 0.1,
            "max_grad_norm": 1.0,
            "optimizer": "adamw",
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "optimizer_updates": 24,
            "schedule_total_updates": 24,
            "warmup_updates": 1,
            "save_model": False,
        },
        "Teacher replay recipe changed",
    )
    require(
        tuple(design["reference_updates"]) == REFERENCE_UPDATES,
        "Reference checkpoint grid changed",
    )
    require(
        tuple(design["causal_updates"]) == CAUSAL_UPDATES,
        "Causal checkpoint grid changed",
    )
    bank = config["fresh_numeric_bank"]
    require(
        bank["rows"] == 1024
        and bank["prompt_seed"] == 2026072601
        and bank["allowed_numeric_token_count"] == 655
        and bank["slot"] == "first_numeric_token_only",
        "Fresh numeric bank changed",
    )
    require(
        config["behavior"]["prompts"] == "all 60 PREFERENCE_EVAL_PROMPTS"
        and config["behavior"]["primary_score"] == "mean logit(wolf)-logit(lion)",
        "Behavior readout changed",
    )
    circuit = config["circuit"]
    require(tuple(circuit["layers"]) == LAYERS, "Circuit layer support changed")
    require(tuple(circuit["module_kinds"]) == KINDS, "Circuit module support changed")
    require(
        circuit["coordinated_modules"] == 8
        and circuit["rank"] == 1
        and tuple(circuit["real_doses"]) == REAL_DOSES
        and tuple(circuit["sham_doses"]) == SHAM_DOSES
        and circuit["sham_draws"] == 5,
        "Causal intervention grid changed",
    )
    require(
        circuit["primary_construction"] == "crossfit_endpoint_loaded"
        and circuit["diagnostic_construction"] == "checkpoint_local",
        "Construction precedence changed",
    )
    require(
        circuit["local_svd"]["base_seed"] == 2026072603
        and circuit["local_svd"]["seed_derivation"]
        == (
            "SHA-256(base_seed,lineage,training_seed,trait,"
            "optimizer_update,module_name) reduced modulo 2^63-1"
        )
        and circuit["sham_base_seed"] == 2026072604
        and "canonical per-module hash" in circuit["factor_provenance"]
        and "Every dose cell" in circuit["factor_provenance"],
        "Factor provenance or full-digest seed contract changed",
    )
    analysis = config["analysis"]
    require(
        analysis["bootstrap_seed"] == 2026072602
        and analysis["bootstrap_samples"] == 2000
        and analysis["practical_floor_fraction_of_endpoint"] == 0.1
        and analysis["hard_event_minimum"] == 100,
        "Frozen analysis constants changed",
    )
    require(
        analysis["hard_event_minimum_per_split"] == 50
        and "seed-2101/wolf/u0 as the single canonical base"
        in analysis["field_math"]["canonical_base_policy"]
        and "substitutes that same readout for all u0 cells"
        in analysis["field_math"]["canonical_base_policy"],
        "Canonical-u0 or split-power analysis rule changed",
    )
    u0 = config["fidelity"]["u0_cross_replay_equivalence"]
    require(
        u0["max_restricted_probability_absolute_difference"] == 1e-6
        and u0["max_behavior_selected_logit_absolute_difference"] == 1e-5
        and u0["selected_weight_sha256_exact"] is True
        and u0["context_identity_exact"] is True,
        "u0 cross-replay equivalence gate changed",
    )
    axes = analysis["taxonomy_rules"]["orthogonal_axes"]
    require(
        axes["field_axis"]
        == [
            "fingerprint_absent",
            "generic_fingerprint_only",
            "trait_specific_field_without_identity",
            "trait_identified_field",
        ],
        "Field taxonomy changed",
    )
    require(
        axes["causal_axis"]
        == [
            "causal_unresolved_replay_or_inventory",
            "causal_not_testable_local_rank_unidentified",
            "causal_not_supported",
            "checkpoint_local_only_rotating",
            "crossfit_consolidated",
            "rotating_then_consolidating",
        ],
        "Causal taxonomy changed",
    )
    require(
        axes["hard_qualifier"]
        == [
            "hard_underpowered",
            "hard_not_supported",
            "hard_partial_below_50pct",
            "hard_supported_fraction_uncertain",
            "hard_majority_mediated",
        ],
        "Hard-event taxonomy changed",
    )
    integrity = config["integrity"]
    require(
        tuple(integrity["full_logical_key"]) == KEY_FIELDS,
        "Full logical key schema changed",
    )
    require(integrity["duplicate_key_policy"] == "reject", "Duplicate policy changed")
    require(
        integrity["expected_key_equality"] == "required before analysis",
        "Expected-key equality guard changed",
    )
    require(
        len(integrity["expected_learning_rates_after_update"]) == 24
        and integrity["expected_learning_rates_after_update"][0] == 1e-5
        and integrity["expected_learning_rates_after_update"][-1] == 0.0,
        "Expected learning-rate sequence changed",
    )
    require(
        integrity["status_fields"]
        == [
            "artifact_integrity_valid",
            "analysis_implementation_valid",
            "primary_classification_valid",
            "overall_pass",
        ],
        "Status ownership schema changed",
    )
    require(
        integrity["overall_pass_rule"]
        == "all three component status fields must be true; exact reproduction of an invalid analysis cannot pass.",
        "Overall-pass rule changed",
    )
    require(
        config["artifacts"]
        == {
            "root": "runs/teacher_trait_fingerprint_ontogeny_v1",
            "preflight": "runs/teacher_trait_fingerprint_ontogeny_v1/preflight.json",
            "endpoint_factors": "runs/teacher_trait_fingerprint_ontogeny_v1/endpoint_factors",
            "endpoint_lock": "runs/teacher_trait_fingerprint_ontogeny_v1/endpoint_factors/lock.json",
            "native_trajectories": "runs/teacher_trait_fingerprint_ontogeny_v1/native_trajectories",
            "native_lock": "runs/teacher_trait_fingerprint_ontogeny_v1/native_trajectories/lock.json",
            "causal_trajectories": "runs/teacher_trait_fingerprint_ontogeny_v1/causal_trajectories",
            "causal_lock": "runs/teacher_trait_fingerprint_ontogeny_v1/causal_trajectories/lock.json",
            "calibration_manifest": "configs/teacher_trait_fingerprint_ontogeny_v1_calibration.json",
            "aggregate_json": "runs/teacher_trait_fingerprint_ontogeny_v1.json",
            "aggregate_markdown": "runs/teacher_trait_fingerprint_ontogeny_v1.md",
            "verifier": "runs/teacher_trait_fingerprint_ontogeny_v1_verify.json",
            "runner": "scripts/teacher_trait_fingerprint_ontogeny_v1.py",
            "verifier_runner": "scripts/teacher_trait_fingerprint_ontogeny_v1_verify.py",
        },
        "Artifact paths changed",
    )
    per_trajectory = expected_cell_keys(SEEDS[0], TRAITS[0], circuit["sham_draws"])
    require(len(per_trajectory) == 270, "Expected 270 causal keys per trajectory")
    require(
        len(
            {
                key_tuple(key)
                for seed in SEEDS
                for trait in TRAITS
                for key in expected_cell_keys(seed, trait, circuit["sham_draws"])
            }
        )
        == 1080,
        "Expected 1080 unique causal keys globally",
    )
    return {
        "protocol_sha256": file_sha256(CONFIG_PATH),
        "endpoint_cells": 4,
        "native_readouts": 100,
        "causal_cells_per_trajectory": 270,
        "causal_cells_global": 1080,
        "logical_key_fields": list(KEY_FIELDS),
    }


def validate_runner_static(config: dict[str, Any]) -> dict[str, Any]:
    source = RUNNER_PATH.read_text()
    tree = ast.parse(source, filename=str(RUNNER_PATH))
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = {
        "logical_key",
        "key_tuple",
        "expected_cell_keys",
        "run_endpoint",
        "run_native_trajectory",
        "run_causal_trajectory",
        "seal_endpoint_factors",
        "native_lock_manifest",
        "causal_lock_manifest",
        "factor_manifest",
        "checkpoint_factor_sets",
        "build_native_analysis",
        "build_causal_analysis",
        "build_onset_results",
        "metamorphic_index_test",
        "analyze",
    }
    require(required.issubset(functions), "Production runner surface is incomplete")
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    require(
        "teacher_trait_fingerprint_ontogeny_v1_verify" not in imports,
        "Production runner imports its verifier",
    )
    require(
        "optimizer_update" in source
        and "Duplicate causal key" in source
        and "Causal key inventory mismatch" in source,
        "Static update-axis/inventory guards disappeared",
    )
    taxonomy = config["analysis"]["taxonomy_rules"]["orthogonal_axes"]
    runner_string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    missing_labels = [
        label
        for labels in taxonomy.values()
        for label in labels
        if label not in runner_string_literals
    ]
    require(
        not missing_labels,
        f"Frozen taxonomy labels are not implemented by the runner: {missing_labels}",
    )
    require(
        "np.mean(powered_rates)" not in source,
        "Runner uses forbidden unweighted hard-recovery averaging",
    )
    require(
        'int.from_bytes(digest, "big") % (2**63 - 1)' in source,
        "Runner no longer derives factor seeds from the full SHA-256 digest",
    )
    require(
        source.count(
            '"base_revision": protocol()["source"]["resolved_revision"]'
        )
        >= 2,
        "Endpoint identities are not bound to source.resolved_revision",
    )
    return {
        "runner_sha256": file_sha256(RUNNER_PATH),
        "required_functions_present": sorted(required),
        "syntax_valid": True,
        "runner_imports_verifier": False,
    }


def validate_verifier_call_surface() -> dict[str, Any]:
    tree = ast.parse(SCRIPT_PATH.read_text(), filename=str(SCRIPT_PATH))
    definitions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    verify_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "verify"
    )
    direct_calls = {
        node.func.id
        for node in ast.walk(verify_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    builtin_names = set(dir(builtins))
    unresolved = sorted(direct_calls - definitions - builtin_names)
    require(
        not unresolved,
        f"Verifier production path calls undefined direct names: {unresolved}",
    )
    required_calls = {
        "validate_endpoints",
        "validate_native",
        "validate_native_lock",
        "validate_causal",
        "validate_causal_lock",
        "recompute_native",
        "recompute_causal_analysis",
        "recompute_hard_analysis",
        "finalize_and_validate_bootstrap",
        "recompute_onsets_and_classification",
    }
    require(
        required_calls.issubset(direct_calls),
        f"Verifier production path omits clean-room phases: "
        f"{sorted(required_calls-direct_calls)}",
    )
    return {
        "passed": True,
        "direct_call_count": len(direct_calls),
        "undefined_direct_calls": [],
        "required_clean_room_phases_called": sorted(required_calls),
    }


def validate_frozen_source_hashes(config: dict[str, Any]) -> dict[str, Any]:
    source = config["source"]
    pairs = (
        ("historical_wolf_teacher_path", "historical_wolf_teacher_sha256"),
        ("historical_wolf_metrics_path", "historical_wolf_metrics_sha256"),
        ("historical_lion_metrics_path", "historical_lion_metrics_sha256"),
        ("wolf_rows_path", "wolf_rows_sha256"),
        ("lion_rows_path", "lion_rows_sha256"),
        ("train_source_path", "train_source_sha256"),
        ("data_source_path", "data_source_sha256"),
        ("generate_source_path", "generate_source_sha256"),
        ("modeling_source_path", "modeling_source_sha256"),
        ("optim_source_path", "optim_source_sha256"),
        ("verifier_source_path", "verifier_source_sha256"),
        ("calibration_manifest_path", "calibration_manifest_sha256"),
    )
    result = {}
    for path_key, hash_key in pairs:
        path = repository_path(source[path_key], name=path_key)
        observed = artifact(path)
        require(observed["sha256"] == source[hash_key], f"Frozen source changed: {path}")
        result[path_key] = observed
    return result


def metamorphic_update_index_regression() -> dict[str, Any]:
    updates = (0, 12, 24)
    synthetic = {
        key_tuple(logical_key(seed, trait, update, "checkpoint_local", "real", -1, 1.0)):
        f"sentinel:s{seed}:{trait}:u{update}"
        for seed in SEEDS
        for trait in TRAITS
        for update in updates
    }
    require(len(synthetic) == len(SEEDS) * len(TRAITS) * len(updates), "Key collision")
    baseline = dict(synthetic)
    changed_slices = {}
    for target in updates:
        trial = dict(baseline)
        for seed in SEEDS:
            for trait in TRAITS:
                key = key_tuple(
                    logical_key(
                        seed,
                        trait,
                        target,
                        "checkpoint_local",
                        "real",
                        -1,
                        1.0,
                    )
                )
                trial[key] = "mutated"
        changed = [
            key for key in baseline if trial[key] != baseline[key]
        ]
        require(
            len(changed) == len(SEEDS) * len(TRAITS)
            and {key[KEY_FIELDS.index("optimizer_update")] for key in changed}
            == {target},
            f"Update mutation leaked across checkpoints at u{target}",
        )
        changed_slices[str(target)] = len(changed)

    update_index = KEY_FIELDS.index("optimizer_update")
    defective_keys = {
        key[:update_index] + key[update_index + 1 :]
        for key in baseline
    }
    require(
        len(defective_keys) < len(baseline),
        "Metamorphic test did not detect the intentionally defective key",
    )

    gates = {update: False for update in updates}
    gate_snapshots = {}
    for target in updates:
        trial = dict(gates)
        trial[target] = True
        require(
            [update for update in updates if trial[update] != gates[update]] == [target],
            f"Gate update indexing failed at u{target}",
        )
        gate_snapshots[str(target)] = trial
    return {
        "passed": True,
        "synthetic_full_keys": len(baseline),
        "changed_keys_per_target": changed_slices,
        "defective_key_without_optimizer_update_collides": True,
        "independent_gate_snapshots": gate_snapshots,
    }


def generic_drift_cancellation_regression() -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(2026072691)
    wolf = torch.randn((17, 23), generator=generator, dtype=torch.float64)
    lion = torch.randn((17, 23), generator=generator, dtype=torch.float64)
    generic = torch.randn((17, 23), generator=generator, dtype=torch.float64)
    original = context_centered_field(wolf, lion)
    shifted = context_centered_field(wolf + generic, lion + generic)
    reverse = context_centered_field(lion, wolf)
    require(
        torch.allclose(original, shifted, rtol=1e-12, atol=1e-12),
        "Paired trait field did not numerically cancel common generic drift",
    )
    require(
        torch.allclose(original, -reverse, rtol=0.0, atol=0.0),
        "Lion paired orientation is not the reverse of wolf orientation",
    )
    return {
        "passed": True,
        "rows": 17,
        "tokens": 23,
        "common_drift_cancels_to_float64_tolerance": True,
        "paired_trait_orientation_reverses_exactly": True,
    }


def conjunction_below_regression() -> dict[str, Any]:
    records = {
        "below": NativeRecord(
            "below", "synthetic", -1.0, np.zeros(2), {}
        ),
        "crossing": NativeRecord(
            "crossing", "synthetic", 0.0, np.zeros(2), {}
        ),
    }
    records["below"].simultaneous_low = -2.0
    records["below"].simultaneous_high = -0.1
    records["crossing"].simultaneous_low = -0.2
    records["crossing"].simultaneous_high = 0.2
    passed, below = evaluate_gate_records(
        ["below", "crossing"], records
    )
    require(not passed and below, "Conjunction-below semantics regressed")
    return {
        "passed": True,
        "one_component_below": True,
        "one_component_crossing": True,
        "joint_gate_demonstrably_below": True,
    }


def endpoint_identity(
    config: dict[str, Any],
    config_hash: str,
    runner_hash: str,
    seed: int,
    trait: str,
) -> dict[str, Any]:
    return {
        "lineage": LINEAGE,
        "training_seed": seed,
        "trait": trait,
        "optimizer_update": 24,
        "base_model_id": "EleutherAI/pythia-160m",
        "base_revision": config["source"]["resolved_revision"],
        "config_sha256": config_hash,
        "script_sha256": runner_hash,
    }


def native_identity(
    config_hash: str, runner_hash: str, seed: int, trait: str
) -> dict[str, Any]:
    return {
        "lineage": LINEAGE,
        "training_seed": seed,
        "trait": trait,
        "updates": list(REFERENCE_UPDATES),
        "config_sha256": config_hash,
        "script_sha256": runner_hash,
    }


def causal_identity(
    config_hash: str, runner_hash: str, seed: int, trait: str
) -> dict[str, Any]:
    return {
        "lineage": LINEAGE,
        "training_seed": seed,
        "trait": trait,
        "reference_updates": list(REFERENCE_UPDATES),
        "causal_updates": list(CAUSAL_UPDATES),
        "crossfit_seed": crossfit_seed(seed),
        "config_sha256": config_hash,
        "script_sha256": runner_hash,
    }


def canonical_attempt(root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    pointer_path = root / "canonical.json"
    pointer = load_json(pointer_path)
    require(
        set(pointer).issuperset({"attempt", "completion_sha256"}),
        f"Malformed canonical pointer: {pointer_path}",
    )
    attempt = repository_path(pointer["attempt"], name=f"{root.name} canonical attempt")
    require(
        attempt.is_relative_to(root.resolve()) and attempt.name.startswith("attempt_"),
        f"Canonical attempt escapes cell root: {attempt}",
    )
    completion_path = attempt / "completion.json"
    completion_artifact = artifact(completion_path)
    require(
        completion_artifact["sha256"] == pointer["completion_sha256"],
        f"Completion hash mismatch: {completion_path}",
    )
    completion = load_json(completion_path)
    require(completion.get("attempt") == relative(attempt), f"Attempt identity mismatch: {attempt}")
    return attempt, pointer, completion


def validate_preflight(config: dict[str, Any]) -> dict[str, Any]:
    preflight = load_json(PREFLIGHT_PATH)
    require(preflight["experiment_id"] == EXPERIMENT_ID, "Wrong preflight experiment")
    require(preflight["scientific_cells_run"] is False, "Preflight was outcome-contaminated")
    implementation = preflight["implementation"]
    require(
        implementation["script_sha256"] == file_sha256(RUNNER_PATH)
        and implementation["config_sha256"] == file_sha256(CONFIG_PATH),
        "Preflight implementation hashes do not match frozen files",
    )
    require(
        implementation["script"] == relative(RUNNER_PATH)
        and implementation["config"] == relative(CONFIG_PATH),
        "Preflight implementation paths changed",
    )
    require(preflight["environment"] == config["pre_registration_disclosure"]["environment"], "Environment guard mismatch")
    require(
        preflight["tokenization"]["allowed_count"] == 655,
        "Numeric support count mismatch",
    )
    require(
        preflight["paired_rows"]["rows"] == 384
        and preflight["paired_rows"]["wolf_rows_sha256"]
        == config["source"]["wolf_rows_sha256"]
        and preflight["paired_rows"]["lion_rows_sha256"]
        == config["source"]["lion_rows_sha256"],
        "Paired-row guard mismatch",
    )
    require(
        preflight["prompt_freshness"]["rows"] == 1024
        and preflight["prompt_freshness"]["overlap_count"] == 0,
        "Fresh-bank guard mismatch",
    )
    for path_value, observed_hash in preflight["source_files"].items():
        path = repository_path(path_value, name="preflight source")
        require(file_sha256(path) == observed_hash, f"Preflight source changed: {path}")
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        record = preflight["cached_source"][name]
        path = Path(record["path"])
        require(path.is_file(), f"Missing cached source: {path}")
        require(file_sha256(path) == record["sha256"], f"Cached source changed: {path}")
    return {
        "artifact": artifact(PREFLIGHT_PATH),
        "git_head": implementation["git_head"],
        "context_guards_valid": True,
    }


def validate_training_metrics(
    path_value: Any,
    expected_hash: Any,
    attempt: Path,
    *,
    config: dict[str, Any],
    completion: dict[str, Any],
    phase: str,
    seed: int,
    probe_updates: Iterable[int],
) -> dict[str, Any]:
    path = repository_path(path_value, name="training metrics")
    require(
        path == (attempt / "training/training_metrics.json").resolve(),
        f"{phase} training metrics path mismatch: {path}",
    )
    record = artifact(path)
    require(record["sha256"] == expected_hash, f"Training metric hash mismatch: {path}")
    metrics = load_json(path)
    require(finite_tree(metrics), f"{phase} training metrics contain non-finite values")
    frozen = config["paired_teacher_design"]["training"]
    exact_fields = {
        "examples": int(config["paired_teacher_design"]["preference_rows"]),
        "epochs": int(frozen["epochs"]),
        "configured_epochs": int(frozen["epochs"]),
        "completed_epochs": int(frozen["epochs"]),
        "optimizer_updates": int(frozen["optimizer_updates"]),
        "schedule_total_updates": int(frozen["schedule_total_updates"]),
        "warmup_updates": int(frozen["warmup_updates"]),
        "saved_model": False,
        "lora": None,
        "seed": int(seed),
    }
    for key, expected in exact_fields.items():
        require(
            metrics.get(key) == expected,
            f"{phase} training metric {key} mismatch: "
            f"{metrics.get(key)} != {expected}",
        )
    require(
        metrics.get("optimizer")
        == {
            "name": "adamw",
            "learning_rate": float(frozen["learning_rate"]),
            "betas": [float(value) for value in frozen["betas"]],
            "eps": float(frozen["eps"]),
        },
        f"{phase} optimizer metadata mismatch",
    )
    update_metrics = metrics.get("update_metrics")
    require(
        isinstance(update_metrics, list)
        and [row["optimizer_update"] for row in update_metrics] == list(range(1, 25)),
        f"{phase} optimizer-update inventory mismatch: {path}",
    )
    require(
        [row.get("learning_rates_after_update") for row in update_metrics]
        == [
            [float(value)]
            for value in config["integrity"][
                "expected_learning_rates_after_update"
            ]
        ],
        f"{phase} learning-rate sequence mismatch: {path}",
    )
    require(
        all(
            row.get("epoch") == 0
            and math.isfinite(
                float(row.get("mean_microbatch_loss", math.nan))
            )
            and math.isfinite(
                float(
                    row.get(
                        "gradient_norm_before_clipping", math.nan
                    )
                )
            )
            for row in update_metrics
        ),
        f"{phase} update metric payload mismatch: {path}",
    )
    expected_probes = [int(value) for value in probe_updates]
    checkpoint_metrics = metrics.get("checkpoint_metrics")
    require(
        isinstance(checkpoint_metrics, list)
        and [
            row.get("optimizer_update") for row in checkpoint_metrics
        ]
        == expected_probes,
        f"{phase} callback inventory mismatch: {path}",
    )
    require(
        completion.get("optimizer_updates") == 24
        and (phase == "endpoint" or completion.get("complete") is True),
        f"{phase} completion optimizer/update state mismatch",
    )
    return record


def validate_safety_inventory(
    records: Any, *, phase: str
) -> dict[int, dict[str, Any]]:
    require(
        isinstance(records, list)
        and [row.get("optimizer_update") for row in records]
        == list(REFERENCE_UPDATES),
        f"{phase} safety update inventory mismatch",
    )
    hook_counts = set()
    unselected_counts = set()
    result = {}
    exact_keys = {
        "optimizer_update",
        "selected_weight_sha256",
        "hook_count",
        "unselected_parameter_count",
        "gradients_none",
        "rng_restored",
    }
    for row in records:
        update = int(row["optimizer_update"])
        require(set(row) == exact_keys, f"{phase} safety schema mismatch at u{update}")
        require(
            is_sha256(row["selected_weight_sha256"])
            and row["gradients_none"] is True
            and row["rng_restored"] is True,
            f"{phase} checkpoint safety failed at u{update}",
        )
        hook_counts.add(int(row["hook_count"]))
        unselected_counts.add(int(row["unselected_parameter_count"]))
        result[update] = row
    require(hook_counts == {0}, f"{phase} retained-hook inventory mismatch")
    require(
        len(unselected_counts) == 1 and min(unselected_counts) > 0,
        f"{phase} unselected-parameter inventory mismatch",
    )
    return result


def validate_endpoints(
    config: dict[str, Any], config_hash: str, runner_hash: str
) -> tuple[dict[str, Any], dict[tuple[int, str], dict[str, Any]]]:
    manifest = {}
    factor_payloads = {}
    lock_cells = {}
    names = set(selected_names())
    for seed in SEEDS:
        for trait in TRAITS:
            label = f"s{seed}:{trait}"
            root = WORK / "endpoint_factors" / f"seed_{seed}" / trait
            attempt, pointer, completion = canonical_attempt(root)
            require(
                completion["identity"]
                == endpoint_identity(
                    config, config_hash, runner_hash, seed, trait
                ),
                f"Endpoint identity mismatch: {label}",
            )
            factors_path = repository_path(
                completion["factors_path"], name=f"{label} factors"
            )
            require(
                factors_path == (attempt / "factors.pt").resolve(),
                f"Endpoint factor path mismatch: {label}",
            )
            factors_record = artifact(factors_path)
            require(
                factors_record["sha256"] == completion["factors_sha256"]
                == pointer.get("factors_sha256"),
                f"Endpoint factor hashes disagree: {label}",
            )
            payload = torch.load(factors_path, map_location="cpu", weights_only=True)
            require(isinstance(payload, dict), f"Malformed factor payload: {label}")
            require(payload.get("identity") == completion["identity"], f"Factor identity mismatch: {label}")
            require(set(payload.get("factors", {})) == names, f"Factor module grid changed: {label}")
            require(set(payload.get("audits", {})) == names, f"Factor audit grid changed: {label}")
            for name_index, name in enumerate(selected_names()):
                factor = payload["factors"][name]
                audit = payload["audits"][name]
                require(set(factor) == {"u", "s", "v"}, f"Malformed factor: {label}/{name}")
                require(
                    set(audit)
                    == {
                        "shape",
                        "leading_singular_values",
                        "singular_gap",
                        "left_residual_relative",
                        "right_residual_relative",
                        "delta_sha256",
                        "factor_seed_index",
                    }
                    and audit["factor_seed_index"] == name_index
                    and finite_tree(audit),
                    f"Malformed endpoint audit: {label}/{name}",
                )
                require(
                    torch.is_tensor(factor["u"])
                    and torch.is_tensor(factor["v"])
                    and factor["u"].ndim == factor["v"].ndim == 1,
                    f"Malformed factor vectors: {label}/{name}",
                )
                require(float(factor["s"]) > 0.0, f"Nonpositive endpoint singular value: {label}/{name}")
                close(float(factor["u"].float().norm()), 1.0, f"{label}/{name}/u_norm", rtol=2e-5, atol=2e-5)
                close(float(factor["v"].float().norm()), 1.0, f"{label}/{name}/v_norm", rtol=2e-5, atol=2e-5)
                require(
                    float(
                        factor["u"][
                            torch.argmax(torch.abs(factor["u"]))
                        ]
                    )
                    >= 0.0,
                    f"Endpoint factor sign convention mismatch: {label}/{name}",
                )
                shape = audit["shape"]
                require(
                    shape == [factor["u"].numel(), factor["v"].numel()],
                    f"Factor/audit shape mismatch: {label}/{name}",
                )
                leading = audit["leading_singular_values"]
                require(
                    isinstance(leading, list)
                    and len(leading) == 4
                    and all(leading[index] >= leading[index + 1] >= 0 for index in range(3)),
                    f"Malformed endpoint spectrum: {label}/{name}",
                )
                close(leading[0], factor["s"], f"{label}/{name}/leading_singular", rtol=2e-7, atol=1e-8)
                close(
                    audit["singular_gap"],
                    float(leading[0]) / max(float(leading[1]), 1e-30),
                    f"{label}/{name}/singular_gap",
                )
                require(
                    float(audit["left_residual_relative"]) >= 0.0
                    and float(audit["right_residual_relative"]) >= 0.0,
                    f"Endpoint residual audit is negative: {label}/{name}",
                )
                require(is_sha256(audit["delta_sha256"]), f"Malformed delta hash: {label}/{name}")
            require(is_sha256(payload["selected_endpoint_sha256"]), f"Malformed selected endpoint hash: {label}")
            require(
                completion.get("selected_endpoint_sha256")
                == payload["selected_endpoint_sha256"],
                f"Endpoint selected-weight binding mismatch: {label}",
            )
            metrics_record = validate_training_metrics(
                completion["training_metrics_path"],
                completion["training_metrics_sha256"],
                attempt,
                config=config,
                completion=completion,
                phase="endpoint",
                seed=seed,
                probe_updates=(),
            )
            require(
                load_json(
                    repository_path(
                        completion["training_metrics_path"],
                        name=f"{label} endpoint training metrics",
                    )
                )["checkpoint_metrics"]
                == [],
                f"Endpoint unexpectedly contains checkpoint callbacks: {label}",
            )
            require(
                completion["scientific_checkpoint_readouts_computed"] is False,
                f"Endpoint factor phase computed scientific readouts: {label}",
            )
            manifest[label] = {
                "canonical": artifact(root / "canonical.json"),
                "completion": artifact(attempt / "completion.json"),
                "factors": factors_record,
                "training_metrics": metrics_record,
                "historical_bridge_pass": completion["historical_bridge_pass"],
                "selected_endpoint_sha256": payload["selected_endpoint_sha256"],
            }
            factor_payloads[(seed, trait)] = payload
            lock_cells[label] = {
                "attempt": relative(attempt),
                "canonical_pointer_sha256": file_sha256(
                    root / "canonical.json"
                ),
                "completion_sha256": file_sha256(attempt / "completion.json"),
                "factors_path": completion["factors_path"],
                "factors_sha256": completion["factors_sha256"],
                "training_metrics_sha256": completion[
                    "training_metrics_sha256"
                ],
                "selected_endpoint_sha256": completion[
                    "selected_endpoint_sha256"
                ],
            }
    lock_path = WORK / "endpoint_factors/lock.json"
    lock = load_json(lock_path)
    expected_lock = {
        "schema": "teacher_trait_fingerprint_endpoint_lock_v1",
        "experiment_id": EXPERIMENT_ID,
        "created_utc": str(lock.get("created_utc")),
        "config_sha256": config_hash,
        "script_sha256": runner_hash,
        "preflight_sha256": file_sha256(PREFLIGHT_PATH),
        "cells": lock_cells,
        "crossfit_map": {"2101": 2102, "2102": 2101},
    }
    compare_tree(lock, expected_lock, "sealed endpoint-factor lock")
    manifest["endpoint_lock"] = artifact(lock_path)
    return manifest, factor_payloads


@dataclass
class Readout:
    numeric_logits: torch.Tensor
    animal_logits: torch.Tensor
    identity: dict[str, Any]


def validate_native(
    config: dict[str, Any], config_hash: str, runner_hash: str
) -> tuple[
    dict[str, Any],
    dict[tuple[int, str, int], Readout],
    dict[tuple[int, str], dict[str, Any]],
]:
    manifest = {}
    readouts: dict[tuple[int, str, int], Readout] = {}
    completions = {}
    common_context = None
    for seed in SEEDS:
        for trait in TRAITS:
            label = f"s{seed}:{trait}"
            root = WORK / "native_trajectories" / f"seed_{seed}" / trait
            attempt, _, completion = canonical_attempt(root)
            require(
                completion["identity"] == native_identity(config_hash, runner_hash, seed, trait),
                f"Native identity mismatch: {label}",
            )
            require(completion["complete"] is True and completion["optimizer_updates"] == 24, f"Incomplete native trajectory: {label}")
            context = completion["context_identity"]
            require(
                set(context)
                == {
                    "prompt_rows_sha256",
                    "prompt_ids_sha256",
                    "allowed_ids_sha256",
                    "animal_ids_sha256",
                }
                and all(is_sha256(value) for value in context.values()),
                f"Malformed native context identity: {label}",
            )
            if common_context is None:
                common_context = context
            require(context == common_context, f"Native context differs across trajectories: {label}")
            rows = completion["readouts"]
            require(
                [row["optimizer_update"] for row in rows] == list(REFERENCE_UPDATES),
                f"Native update inventory mismatch: {label}",
            )
            require(len({row["optimizer_update"] for row in rows}) == 25, f"Duplicate native update: {label}")
            safety_by_update = validate_safety_inventory(
                completion.get("safety"), phase=f"native:{label}"
            )
            readout_manifest = []
            for row in rows:
                update = int(row["optimizer_update"])
                require(
                    set(row)
                    == {
                        "optimizer_update",
                        "path",
                        "sha256",
                        "numeric_logits_sha256",
                        "animal_logits_sha256",
                        "selected_weight_sha256",
                    },
                    f"Native readout record schema mismatch: {label}/u{update}",
                )
                path = repository_path(row["path"], name=f"{label}/u{update} readout")
                require(
                    path
                    == (attempt / "readouts" / f"u{update:04d}.pt").resolve(),
                    f"Native readout path mismatch: {path}",
                )
                record = artifact(path)
                require(record["sha256"] == row["sha256"], f"Native readout hash mismatch: {path}")
                payload = torch.load(path, map_location="cpu", weights_only=True)
                require(set(payload) == {"identity", "numeric_logits", "animal_logits"}, f"Malformed readout payload: {path}")
                numeric = payload["numeric_logits"]
                animals = payload["animal_logits"]
                require(
                    torch.is_tensor(numeric)
                    and tuple(numeric.shape) == (1024, 655)
                    and torch.is_tensor(animals)
                    and tuple(animals.shape) == (60, 10),
                    f"Readout tensor shape mismatch: {path}",
                )
                require(
                    bool(torch.isfinite(numeric).all()) and bool(torch.isfinite(animals).all()),
                    f"Non-finite native readout: {path}",
                )
                expected_identity = {
                    "lineage": LINEAGE,
                    "training_seed": seed,
                    "trait": trait,
                    "optimizer_update": update,
                    **context,
                    "selected_weight_sha256": safety_by_update[update][
                        "selected_weight_sha256"
                    ],
                }
                require(payload["identity"] == expected_identity, f"Embedded native identity mismatch: {path}")
                require(
                    row["numeric_logits_sha256"] == tensor_sha256(numeric)
                    and row["animal_logits_sha256"] == tensor_sha256(animals)
                    and row["selected_weight_sha256"]
                    == safety_by_update[update]["selected_weight_sha256"],
                    f"Native tensor digest mismatch: {path}",
                )
                readouts[(seed, trait, update)] = Readout(
                    numeric.float().cpu(),
                    animals.float().cpu(),
                    payload["identity"],
                )
                readout_manifest.append(
                    {
                        "optimizer_update": update,
                        **record,
                        "numeric_logits_sha256": row["numeric_logits_sha256"],
                        "animal_logits_sha256": row["animal_logits_sha256"],
                    }
                )
            metrics_record = validate_training_metrics(
                completion["training_metrics_path"],
                completion["training_metrics_sha256"],
                attempt,
                config=config,
                completion=completion,
                phase="native",
                seed=seed,
                probe_updates=REFERENCE_UPDATES,
            )
            training_metrics = load_json(
                repository_path(
                    completion["training_metrics_path"],
                    name=f"{label} native training metrics",
                )
            )
            require(
                training_metrics["checkpoint_metrics"]
                == [
                    {
                        "optimizer_update": row["optimizer_update"],
                        "path": row["path"],
                        "sha256": row["sha256"],
                        "selected_weight_sha256": row[
                            "selected_weight_sha256"
                        ],
                    }
                    for row in rows
                ],
                f"Native training/readout callback cross-link mismatch: {label}",
            )
            manifest[label] = {
                "canonical": artifact(root / "canonical.json"),
                "completion": artifact(attempt / "completion.json"),
                "training_metrics": metrics_record,
                "readout_manifest_sha256": compact_sha256(readout_manifest),
                "readouts": len(readout_manifest),
            }
            completions[(seed, trait)] = completion
    require(len(readouts) == 100, "Global native readout inventory is not 100")
    return manifest, readouts, completions


def independent_u0_equivalence(
    config: dict[str, Any],
    members: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    limits = config["fidelity"]["u0_cross_replay_equivalence"]
    labels = sorted(members)
    member_records = {}
    comparisons = []
    all_pass = True
    for label in labels:
        member = members[label]
        readout: Readout = member["readout"]
        member_records[label] = {
            "path": relative(member["path"]),
            "sha256": file_sha256(member["path"]),
            "selected_weight_sha256": readout.identity[
                "selected_weight_sha256"
            ],
            "context_identity_sha256": compact_sha256(
                member["context_identity"]
            ),
            "numeric_logits_sha256": tensor_sha256(readout.numeric_logits),
            "animal_logits_sha256": tensor_sha256(readout.animal_logits),
        }
    for left_index, left_label in enumerate(labels):
        for right_label in labels[left_index + 1 :]:
            left = members[left_label]
            right = members[right_label]
            left_readout: Readout = left["readout"]
            right_readout: Readout = right["readout"]
            maximum_probability = float(
                torch.max(
                    torch.abs(
                        torch.softmax(
                            left_readout.numeric_logits.double(), dim=-1
                        )
                        - torch.softmax(
                            right_readout.numeric_logits.double(), dim=-1
                        )
                    )
                )
            )
            maximum_behavior = float(
                torch.max(
                    torch.abs(
                        left_readout.animal_logits.double()
                        - right_readout.animal_logits.double()
                    )
                )
            )
            selected_equal = (
                left_readout.identity["selected_weight_sha256"]
                == right_readout.identity["selected_weight_sha256"]
            )
            context_equal = (
                left["context_identity"] == right["context_identity"]
            )
            passed = bool(
                maximum_probability
                <= float(
                    limits[
                        "max_restricted_probability_absolute_difference"
                    ]
                )
                and maximum_behavior
                <= float(
                    limits[
                        "max_behavior_selected_logit_absolute_difference"
                    ]
                )
                and selected_equal
                and context_equal
            )
            all_pass = all_pass and passed
            comparisons.append(
                {
                    "left": left_label,
                    "right": right_label,
                    "max_restricted_probability_absolute_difference": (
                        maximum_probability
                    ),
                    "max_behavior_selected_logit_absolute_difference": (
                        maximum_behavior
                    ),
                    "selected_weight_sha256_equal": selected_equal,
                    "context_identity_equal": context_equal,
                    "pass": passed,
                }
            )
    require(
        len(comparisons) == len(labels) * (len(labels) - 1) // 2,
        "Independent u0 comparison inventory mismatch",
    )
    require(all_pass, "Independent cross-replay u0 equivalence failed")
    return {
        "member_count": len(labels),
        "pair_count": len(comparisons),
        "limits": limits,
        "members": member_records,
        "comparisons": comparisons,
        "all_pairs_pass": all_pass,
    }


def validate_native_lock(
    config: dict[str, Any],
    config_hash: str,
    runner_hash: str,
    native: dict[tuple[int, str, int], Readout],
    completions: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, Any]:
    path = WORK / "native_trajectories/lock.json"
    observed = load_json(path)
    endpoint_lock = load_json(WORK / "endpoint_factors/lock.json")
    cells = {}
    members = {}
    for seed in SEEDS:
        for trait in TRAITS:
            label = f"s{seed}:{trait}"
            completion = completions[(seed, trait)]
            attempt = repository_path(
                completion["attempt"], name=f"{label} native attempt"
            )
            pointer = (
                WORK
                / "native_trajectories"
                / f"seed_{seed}"
                / trait
                / "canonical.json"
            )
            u0_record = completion["readouts"][0]
            u0_path = repository_path(
                u0_record["path"], name=f"{label} native u0"
            )
            members[f"native:{label}"] = {
                "path": u0_path,
                "readout": native[(seed, trait, 0)],
                "context_identity": completion["context_identity"],
            }
            cells[label] = {
                "attempt": relative(attempt),
                "canonical_pointer_sha256": file_sha256(pointer),
                "completion_sha256": file_sha256(
                    attempt / "completion.json"
                ),
                "training_metrics_sha256": completion[
                    "training_metrics_sha256"
                ],
                "readout_manifest_sha256": compact_sha256(
                    completion["readouts"]
                ),
            }
    u0_equivalence = independent_u0_equivalence(config, members)
    require(
        u0_equivalence["member_count"] == 4
        and u0_equivalence["pair_count"] == 6,
        "Native lock did not perform the registered 4-member/6-pair u0 audit",
    )
    expected = {
        "schema": "teacher_trait_fingerprint_native_lock_v1",
        "experiment_id": EXPERIMENT_ID,
        "created_utc": str(observed.get("created_utc")),
        "config_sha256": config_hash,
        "script_sha256": runner_hash,
        "preflight_sha256": file_sha256(PREFLIGHT_PATH),
        "endpoint_lock_sha256": file_sha256(
            WORK / "endpoint_factors/lock.json"
        ),
        "endpoint_cell_manifest_sha256": compact_sha256(
            endpoint_lock["cells"]
        ),
        "cells": cells,
        "u0_equivalence": u0_equivalence,
    }
    compare_tree(observed, expected, "native trajectory lock")
    return artifact(path)


def pooled_cosine(dot: np.ndarray, left_norm: np.ndarray, right_norm: np.ndarray) -> float:
    denominator = math.sqrt(float(left_norm.sum()) * float(right_norm.sum()))
    return 0.0 if denominator == 0.0 else float(dot.sum() / denominator)


def validate_factor_record(value: dict[str, Any], name: str) -> dict[str, float]:
    require(set(value) == {"modules", "coordinated_frobenius_norm"}, f"Malformed factor record: {name}")
    require(set(value["modules"]) == set(selected_names()), f"Factor module inventory mismatch: {name}")
    total = 0.0
    amplitudes = {}
    for module in selected_names():
        row = value["modules"][module]
        require(
            set(row) == {"signed_amplitude", "u_norm", "v_norm", "frobenius_norm"},
            f"Malformed module factor record: {name}/{module}",
        )
        close(row["u_norm"], 1.0, f"{name}/{module}/u_norm", rtol=3e-5, atol=3e-5)
        close(row["v_norm"], 1.0, f"{name}/{module}/v_norm", rtol=3e-5, atol=3e-5)
        expected = abs(float(row["signed_amplitude"])) * float(row["u_norm"]) * float(row["v_norm"])
        close(row["frobenius_norm"], expected, f"{name}/{module}/frobenius")
        total += expected**2
        amplitudes[module] = float(row["signed_amplitude"])
    close(value["coordinated_frobenius_norm"], math.sqrt(total), f"{name}/coordinated")
    return amplitudes


def endpoint_source_record(
    endpoints: dict[tuple[int, str], dict[str, Any]],
    seed: int,
    trait: str,
) -> dict[str, Any]:
    root = WORK / "endpoint_factors" / f"seed_{seed}" / trait
    attempt, _, completion = canonical_attempt(root)
    return {
        "seed": seed,
        "trait": trait,
        "attempt": relative(attempt),
        "completion_sha256": file_sha256(attempt / "completion.json"),
        "factors_path": completion["factors_path"],
        "factors_sha256": completion["factors_sha256"],
        "selected_endpoint_sha256": endpoints[(seed, trait)][
            "selected_endpoint_sha256"
        ],
    }


def native_source_record(
    completions: dict[tuple[int, str], dict[str, Any]],
    seed: int,
    trait: str,
) -> dict[str, Any]:
    completion = completions[(seed, trait)]
    attempt = repository_path(
        completion["attempt"], name=f"native source s{seed}:{trait}"
    )
    return {
        "seed": seed,
        "trait": trait,
        "attempt": relative(attempt),
        "completion_sha256": file_sha256(attempt / "completion.json"),
        "readout_manifest_sha256": compact_sha256(completion["readouts"]),
        "readouts": {
            str(record["optimizer_update"]): {
                "path": record["path"],
                "sha256": record["sha256"],
                "numeric_logits_sha256": record["numeric_logits_sha256"],
                "animal_logits_sha256": record["animal_logits_sha256"],
                "selected_weight_sha256": record[
                    "selected_weight_sha256"
                ],
            }
            for record in completion["readouts"]
        },
    }


def validate_factor_catalog(
    config: dict[str, Any],
    config_hash: str,
    runner_hash: str,
    endpoints: dict[tuple[int, str], dict[str, Any]],
    *,
    seed: int,
    trait: str,
    update: int,
    checkpoint: dict[str, Any],
    attempt: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, dict[str, Any]]],
]:
    record = checkpoint.get("factor_catalog")
    require(isinstance(record, dict), f"Missing factor catalog: s{seed}:{trait}/u{update}")
    require(
        set(record)
        == {
            "path",
            "sha256",
            "factor_set_ids",
            "factor_manifest_sha256",
        },
        f"Factor-catalog record schema mismatch: s{seed}:{trait}/u{update}",
    )
    path = repository_path(
        record["path"], name=f"factor catalog s{seed}:{trait}/u{update}"
    )
    require(
        path == (attempt / "factors" / f"u{update:04d}.pt").resolve()
        and file_sha256(path) == record["sha256"],
        f"Factor-catalog path/hash mismatch: {path}",
    )
    catalog = torch.load(path, map_location="cpu", weights_only=True)
    require(
        isinstance(catalog, dict)
        and set(catalog)
        == {
            "identity",
            "checkpoint_local_factors",
            "checkpoint_local_witnesses",
            "factor_audit",
            "factor_manifests",
        },
        f"Factor-catalog payload schema mismatch: {path}",
    )
    require(
        catalog["identity"]
        == {
            **causal_identity(
                config_hash, runner_hash, seed, trait
            ),
            "optimizer_update": update,
            "selected_weight_sha256": checkpoint["safety"][
                "selected_weight_sha256"
            ],
        },
        f"Factor-catalog identity mismatch: {path}",
    )
    audit = checkpoint.get("factor_audit")
    require(
        isinstance(audit, dict)
        and catalog["factor_audit"] == audit
        and set(audit)
        == {
            "local_identifiable",
            "local_audits",
            "crossfit_projections",
        },
        f"Factor audit schema/binding mismatch: {path}",
    )
    require(
        set(audit["local_audits"]) == set(selected_names())
        and set(audit["crossfit_projections"]) == set(selected_names()),
        f"Factor audit module inventory mismatch: {path}",
    )

    local = catalog["checkpoint_local_factors"]
    witnesses = catalog["checkpoint_local_witnesses"]
    local_identifiable_by_module = []
    local_seed_base = int(config["circuit"]["local_svd"]["base_seed"])
    local_config = config["circuit"]["local_svd"]
    for name in selected_names():
        module_audit = audit["local_audits"][name]
        expected_seed = derived_seed(
            local_seed_base,
            LINEAGE,
            seed,
            trait,
            update,
            name,
        )
        require(
            module_audit.get("derived_seed") == expected_seed,
            f"Local-factor derived seed mismatch: {path}:{name}",
        )
        singular_values = module_audit.get("singular_values")
        require(
            isinstance(singular_values, list)
            and len(singular_values) == 4
            and all(math.isfinite(float(value)) for value in singular_values)
            and all(
                float(singular_values[index])
                >= float(singular_values[index + 1])
                >= 0.0
                for index in range(3)
            ),
            f"Malformed local singular values: {path}:{name}",
        )
        if module_audit.get("reason") == "zero_delta":
            require(
                set(module_audit)
                == {
                    "identifiable",
                    "reason",
                    "singular_values",
                    "derived_seed",
                }
                and module_audit["identifiable"] is False
                and singular_values == [0.0, 0.0, 0.0, 0.0],
                f"Malformed zero-delta local audit: {path}:{name}",
            )
        else:
            require(
                set(module_audit)
                == {
                    "identifiable",
                    "singular_values",
                    "singular_gap",
                    "left_residual_relative",
                    "right_residual_relative",
                    "derived_seed",
                },
                f"Malformed local-factor audit: {path}:{name}",
            )
            singular_gap = float(singular_values[0]) / max(
                float(singular_values[1]), 1e-30
            )
            close(
                module_audit["singular_gap"],
                singular_gap,
                f"{path}:{name}/singular_gap",
                rtol=2e-7,
                atol=2e-9,
            )
            expected_identifiable = bool(
                singular_gap
                >= float(local_config["singular_gap_minimum"])
                and max(
                    float(module_audit["left_residual_relative"]),
                    float(module_audit["right_residual_relative"]),
                )
                <= float(local_config["residual_relative_maximum"])
            )
            require(
                module_audit["identifiable"] is expected_identifiable,
                f"Local identifiability decision mismatch: {path}:{name}",
            )
        local_identifiable_by_module.append(
            bool(module_audit["identifiable"])
        )
    expected_local_available = all(local_identifiable_by_module)
    require(
        audit["local_identifiable"] is expected_local_available
        and (local is not None) is expected_local_available,
        f"Checkpoint-local availability mismatch: {path}",
    )
    if local is None:
        require(witnesses == {}, f"Unexpected witnesses for unavailable local factor: {path}")
    else:
        require(
            set(local) == set(selected_names())
            and set(witnesses) == set(selected_names()),
            f"Checkpoint-local factor/witness inventory mismatch: {path}",
        )
        for name in selected_names():
            factor = local[name]
            witness = witnesses[name]
            require(
                set(factor) == {"u", "s", "v"}
                and set(witness)
                == {"delta_v", "delta_transpose_u"},
                f"Malformed checkpoint-local factor/witness: {path}:{name}",
            )
            u = factor["u"].float()
            v = factor["v"].float()
            singular = float(factor["s"])
            delta_v = witness["delta_v"].float()
            delta_t_u = witness["delta_transpose_u"].float()
            endpoint_shape = endpoints[(crossfit_seed(seed), trait)][
                "factors"
            ][name]
            require(
                tuple(u.shape) == tuple(endpoint_shape["u"].shape)
                and tuple(v.shape) == tuple(endpoint_shape["v"].shape)
                and tuple(delta_v.shape) == tuple(u.shape)
                and tuple(delta_t_u.shape) == tuple(v.shape)
                and bool(torch.isfinite(u).all())
                and bool(torch.isfinite(v).all())
                and bool(torch.isfinite(delta_v).all())
                and bool(torch.isfinite(delta_t_u).all())
                and singular > 0.0,
                f"Checkpoint-local tensor contract mismatch: {path}:{name}",
            )
            close(float(u.norm()), 1.0, f"{path}:{name}/local_u_norm", rtol=3e-5, atol=3e-5)
            close(float(v.norm()), 1.0, f"{path}:{name}/local_v_norm", rtol=3e-5, atol=3e-5)
            require(
                float(u[torch.argmax(torch.abs(u))]) >= 0.0,
                f"Checkpoint-local sign convention mismatch: {path}:{name}",
            )
            close(
                singular,
                audit["local_audits"][name]["singular_values"][0],
                f"{path}:{name}/local_singular",
                rtol=2e-7,
                atol=2e-9,
            )
            left = float((delta_v - singular * u).norm()) / max(
                abs(singular), 1e-30
            )
            right = float((delta_t_u - singular * v).norm()) / max(
                abs(singular), 1e-30
            )
            close(
                left,
                audit["local_audits"][name][
                    "left_residual_relative"
                ],
                f"{path}:{name}/local_left_witness",
                rtol=2e-5,
                atol=2e-7,
            )
            close(
                right,
                audit["local_audits"][name][
                    "right_residual_relative"
                ],
                f"{path}:{name}/local_right_witness",
                rtol=2e-5,
                atol=2e-7,
            )

    donor = crossfit_seed(seed)
    matched_endpoint = endpoints[(donor, trait)]
    wrong_endpoint = endpoints[(donor, other_trait(trait))]
    loaded: dict[str, dict[str, Any]] = {}
    wrong: dict[str, dict[str, Any]] = {}
    for name in selected_names():
        projection = audit["crossfit_projections"][name]
        require(
            set(projection)
            == {
                "signed_projection",
                "matched_endpoint_singular_value",
                "fraction_of_crossfit_endpoint_singular_value",
            }
            and finite_tree(projection),
            f"Malformed crossfit projection audit: {path}:{name}",
        )
        coefficient = float(projection["signed_projection"])
        endpoint_singular = float(
            matched_endpoint["factors"][name]["s"]
        )
        close(
            projection["matched_endpoint_singular_value"],
            endpoint_singular,
            f"{path}:{name}/endpoint_singular",
        )
        close(
            projection[
                "fraction_of_crossfit_endpoint_singular_value"
            ],
            coefficient / max(endpoint_singular, 1e-30),
            f"{path}:{name}/endpoint_fraction",
        )
        loaded[name] = {
            "u": matched_endpoint["factors"][name]["u"].float(),
            "s": coefficient,
            "v": matched_endpoint["factors"][name]["v"].float(),
        }
        wrong[name] = {
            "u": wrong_endpoint["factors"][name]["u"].float(),
            "s": coefficient,
            "v": wrong_endpoint["factors"][name]["v"].float(),
        }

    factor_sets: dict[str, dict[str, dict[str, Any]]] = {}
    for construction, real in (
        ("checkpoint_local", local),
        ("crossfit_endpoint_loaded", loaded),
    ):
        if real is None:
            continue
        factor_sets[factor_set_id(construction, "real", -1)] = real
        for draw in range(int(config["circuit"]["sham_draws"])):
            sham = {}
            for name in selected_names():
                sham[name] = haar_factor(
                    (
                        int(real[name]["u"].numel()),
                        int(real[name]["v"].numel()),
                    ),
                    float(real[name]["s"]),
                    seed=derived_seed(
                        int(config["circuit"]["sham_base_seed"]),
                        LINEAGE,
                        seed,
                        trait,
                        update,
                        construction,
                        name,
                        draw,
                    ),
                )
            factor_sets[
                factor_set_id(construction, "sham", draw)
            ] = sham
        if construction == "crossfit_endpoint_loaded":
            factor_sets[
                factor_set_id(construction, "wrong_trait", -1)
            ] = wrong

    manifests = {
        identifier: factor_manifest(factors)
        for identifier, factors in factor_sets.items()
    }
    compare_tree(
        catalog["factor_manifests"],
        manifests,
        f"factor catalog derivation {path}",
    )
    require(
        record["factor_set_ids"] == sorted(manifests)
        and record["factor_manifest_sha256"]
        == compact_sha256(manifests),
        f"Factor-catalog set inventory/hash mismatch: {path}",
    )
    return manifests, factor_sets


def validate_causal_arrays(
    payload: dict[str, Any], arrays: dict[str, np.ndarray], name: str
) -> None:
    require(set(arrays) == CAUSAL_ARRAY_KEYS, f"Causal array schema changed: {name}")
    for key, value in arrays.items():
        expected = 60 if key.startswith("behavior_") else 1024
        require(value.shape == (expected,), f"Array shape mismatch {name}/{key}: {value.shape}")
        require(np.all(np.isfinite(value)), f"Non-finite causal array: {name}/{key}")
    require(
        arrays["hard_event"].dtype == np.bool_
        and arrays["hard_oriented_recovery"].dtype == np.bool_,
        f"Hard arrays are not boolean: {name}",
    )
    require(
        np.all(~arrays["hard_oriented_recovery"] | arrays["hard_event"]),
        f"Hard recovery occurs outside the target event set: {name}",
    )
    key = payload["key"]
    dose = float(key["dose"])
    native_js = arrays["numeric_native_js"].astype(np.float64)
    cell_js = arrays["numeric_cell_js"].astype(np.float64)
    progress = arrays["numeric_oriented_js_progress"].astype(np.float64)
    expected_progress = cell_js - native_js if dose > 0 else native_js - cell_js
    require(np.allclose(progress, expected_progress, rtol=1e-10, atol=1e-12), f"JS orientation mismatch: {name}")
    metrics = payload["metrics"]
    require(isinstance(metrics, dict) and finite_tree(metrics), f"Malformed causal metrics: {name}")
    require(
        set(metrics) == {"numeric", "behavior", "hard"}
        and set(metrics["numeric"])
        == {
            "native_paired_mean_js",
            "cell_paired_mean_js",
            "oriented_mean_js_progress",
            "oriented_js_progress_fraction",
            "centered_logit_field",
            "restricted_probability_field",
        }
        and set(metrics["numeric"]["centered_logit_field"])
        == {
            "cosine",
            "context_centered_cosine",
            "capture_slope",
            "context_centered_capture_slope",
        }
        and set(
            metrics["numeric"]["restricted_probability_field"]
        )
        == {
            "cosine",
            "context_centered_cosine",
            "capture_slope",
            "context_centered_capture_slope",
        }
        and set(metrics["behavior"])
        == {
            "mean_native_paired_target_gap",
            "oriented_mean_target_pair_effect",
            "oriented_target_pair_mediation_fraction",
            "oriented_mean_nine_animal_margin_effect",
        }
        and set(metrics["hard"])
        == {
            "target",
            "paired_argmax_event_count",
            "oriented_recovery_or_preservation_rate",
            "powered",
        },
        f"Causal metric schema mismatch: {name}",
    )
    numeric = metrics["numeric"]
    close(numeric["native_paired_mean_js"], native_js.mean(), f"{name}/native_js")
    close(numeric["cell_paired_mean_js"], cell_js.mean(), f"{name}/cell_js")
    close(numeric["oriented_mean_js_progress"], progress.mean(), f"{name}/js_progress")
    close(
        numeric["oriented_js_progress_fraction"],
        progress.mean() / max(float(native_js.mean()), 1e-30),
        f"{name}/js_fraction",
    )
    logit = numeric["centered_logit_field"]
    close(
        logit["cosine"],
        pooled_cosine(
            arrays["logit_field_dot"],
            arrays["logit_field_norm"],
            arrays["logit_effect_norm"],
        ),
        f"{name}/logit_cosine",
    )
    close(
        logit["context_centered_cosine"],
        pooled_cosine(
            arrays["logit_context_field_dot"],
            arrays["logit_context_field_norm"],
            arrays["logit_context_effect_norm"],
        ),
        f"{name}/logit_context_cosine",
    )
    close(
        logit["capture_slope"],
        arrays["logit_field_dot"].sum()
        / max(float(arrays["logit_field_norm"].sum()), 1e-30),
        f"{name}/logit_capture",
    )
    close(
        logit["context_centered_capture_slope"],
        arrays["logit_context_field_dot"].sum()
        / max(
            float(arrays["logit_context_field_norm"].sum()),
            1e-30,
        ),
        f"{name}/logit_context_capture",
    )
    probability = numeric["restricted_probability_field"]
    close(
        probability["cosine"],
        pooled_cosine(
            arrays["probability_field_dot"],
            arrays["probability_field_norm"],
            arrays["probability_effect_norm"],
        ),
        f"{name}/probability_cosine",
    )
    close(
        probability["context_centered_cosine"],
        pooled_cosine(
            arrays["probability_context_field_dot"],
            arrays["probability_context_field_norm"],
            arrays["probability_context_effect_norm"],
        ),
        f"{name}/probability_context_cosine",
    )
    close(
        probability["capture_slope"],
        arrays["probability_field_dot"].sum()
        / max(float(arrays["probability_field_norm"].sum()), 1e-30),
        f"{name}/probability_capture",
    )
    close(
        probability["context_centered_capture_slope"],
        arrays["probability_context_field_dot"].sum()
        / max(
            float(arrays["probability_context_field_norm"].sum()),
            1e-30,
        ),
        f"{name}/probability_context_capture",
    )
    behavior = metrics["behavior"]
    gap = arrays["behavior_native_gap"].astype(np.float64)
    effect = arrays["behavior_oriented_effect"].astype(np.float64)
    margin = arrays["behavior_oriented_margin_effect"].astype(np.float64)
    close(behavior["mean_native_paired_target_gap"], gap.mean(), f"{name}/behavior_gap")
    close(behavior["oriented_mean_target_pair_effect"], effect.mean(), f"{name}/behavior_effect")
    close(
        behavior["oriented_target_pair_mediation_fraction"],
        effect.mean() / max(abs(float(gap.mean())), 1e-30),
        f"{name}/behavior_fraction",
    )
    close(behavior["oriented_mean_nine_animal_margin_effect"], margin.mean(), f"{name}/behavior_margin")
    hard = metrics["hard"]
    event = arrays["hard_event"].astype(bool)
    recovery = arrays["hard_oriented_recovery"].astype(bool)
    count = int(event.sum())
    rate = float(recovery[event].astype(np.float64).mean()) if count else 0.0
    require(hard["paired_argmax_event_count"] == count, f"Hard event count mismatch: {name}")
    close(hard["oriented_recovery_or_preservation_rate"], rate, f"{name}/hard_rate")
    require(hard["powered"] == (count >= 100), f"Hard power flag mismatch: {name}")
    expected_target = "cross_seed_same_trait_u24" if dose > 0 else "paired_other_trait_checkpoint"
    require(hard["target"] == expected_target, f"Hard target mismatch: {name}")


def validate_causal(
    config: dict[str, Any],
    config_hash: str,
    runner_hash: str,
    endpoints: dict[tuple[int, str], dict[str, Any]],
    native: dict[tuple[int, str, int], Readout],
    native_completions: dict[tuple[int, str], dict[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[tuple[Any, ...], tuple[str, dict[str, np.ndarray] | None]],
    dict[tuple[int, str, int], bool],
    dict[tuple[int, str], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    manifest = {}
    completions: dict[tuple[int, str], dict[str, Any]] = {}
    live_u0_members: dict[str, dict[str, Any]] = {}
    replay_usable: dict[tuple[int, str, int], bool] = {}
    amplitude_signatures: dict[tuple[Any, ...], dict[str, float]] = {}
    analysis_cells: dict[
        tuple[Any, ...], tuple[str, dict[str, np.ndarray] | None]
    ] = {}
    expected_global = set()
    observed_global = set()
    sham_draws = int(config["circuit"]["sham_draws"])
    endpoint_lock = load_json(WORK / "endpoint_factors/lock.json")
    native_lock = load_json(WORK / "native_trajectories/lock.json")
    for seed in SEEDS:
        for trait in TRAITS:
            expected_global.update(
                key_tuple(key)
                for key in expected_cell_keys(seed, trait, sham_draws)
            )
            label = f"s{seed}:{trait}"
            root = WORK / "causal_trajectories" / f"seed_{seed}" / trait
            attempt, _, completion = canonical_attempt(root)
            require(
                completion["identity"] == causal_identity(config_hash, runner_hash, seed, trait),
                f"Causal identity mismatch: {label}",
            )
            require(completion["complete"] is True and completion["optimizer_updates"] == 24, f"Incomplete causal trajectory: {label}")
            require(
                completion.get("upstream_locks")
                == {
                    "endpoint_lock_sha256": file_sha256(
                        WORK / "endpoint_factors/lock.json"
                    ),
                    "native_lock_sha256": file_sha256(
                        WORK / "native_trajectories/lock.json"
                    ),
                    "endpoint_cell_manifest_sha256": compact_sha256(
                        endpoint_lock["cells"]
                    ),
                    "native_cell_manifest_sha256": compact_sha256(
                        native_lock["cells"]
                    ),
                },
                f"Causal upstream-lock binding mismatch: {label}",
            )
            donor = crossfit_seed(seed)
            expected_native_sources = {
                "current": native_source_record(
                    native_completions, seed, trait
                ),
                "paired_other": native_source_record(
                    native_completions, seed, other_trait(trait)
                ),
                "donor_endpoint_target": native_source_record(
                    native_completions, donor, trait
                ),
            }
            compare_tree(
                completion.get("native_sources"),
                expected_native_sources,
                f"causal native sources {label}",
            )
            expected_endpoint_sources = {
                "matched": endpoint_source_record(
                    endpoints, donor, trait
                ),
                "wrong_trait": endpoint_source_record(
                    endpoints, donor, other_trait(trait)
                ),
            }
            compare_tree(
                completion.get("endpoint_factor_sources"),
                expected_endpoint_sources,
                f"causal endpoint sources {label}",
            )
            require(
                completion.get("endpoint_target")
                == {
                    "seed": donor,
                    "trait": trait,
                    "optimizer_update": 24,
                    "numeric_logits_sha256": tensor_sha256(
                        native[(donor, trait, 24)].numeric_logits
                    ),
                },
                f"Endpoint target mismatch: {label}",
            )
            checkpoint_records = completion["checkpoint_records"]
            require(
                [row["optimizer_update"] for row in checkpoint_records]
                == list(REFERENCE_UPDATES),
                f"Causal checkpoint inventory mismatch: {label}",
            )
            safety_by_update = validate_safety_inventory(
                [
                    {
                        "optimizer_update": checkpoint[
                            "optimizer_update"
                        ],
                        **checkpoint["safety"],
                    }
                    for checkpoint in checkpoint_records
                ],
                phase=f"causal:{label}",
            )
            factor_manifests: dict[
                int, dict[str, dict[str, Any]]
            ] = {}
            factor_sets: dict[
                int, dict[str, dict[str, dict[str, Any]]]
            ] = {}
            for checkpoint in checkpoint_records:
                update = int(checkpoint["optimizer_update"])
                expected_checkpoint_keys = {
                    "optimizer_update",
                    "repeat_guard",
                    "safety",
                }
                if update == 0:
                    expected_checkpoint_keys.add("live_u0_readout")
                if update in CAUSAL_UPDATES:
                    expected_checkpoint_keys.update(
                        {"factor_audit", "factor_catalog"}
                    )
                require(
                    set(checkpoint) == expected_checkpoint_keys,
                    f"Causal checkpoint schema mismatch: {label}/u{update}",
                )
                repeat = checkpoint["repeat_guard"]
                require(
                    isinstance(repeat, dict)
                    and finite_tree(repeat)
                    and repeat["pass"] is True
                    and isinstance(repeat["usable_for_onset"], bool),
                    f"Malformed/failed replay guard: {label}/u{update}",
                )
                if update == 0:
                    require(
                        repeat["relative_or_u0_pass"] is True,
                        f"Dedicated causal/native u0 guard failed: {label}",
                    )
                replay_usable[(seed, trait, int(update))] = bool(
                    repeat["usable_for_onset"]
                )
                if update == 0:
                    live_record = checkpoint["live_u0_readout"]
                    require(
                        set(live_record)
                        == {
                            "path",
                            "sha256",
                            "numeric_logits_sha256",
                            "animal_logits_sha256",
                            "selected_weight_sha256",
                        },
                        f"Causal live-u0 record schema mismatch: {label}",
                    )
                    live_path = repository_path(
                        live_record["path"],
                        name=f"causal live u0 {label}",
                    )
                    require(
                        live_path
                        == (attempt / "replay/u0000.pt").resolve()
                        and file_sha256(live_path)
                        == live_record["sha256"],
                        f"Causal live-u0 path/hash mismatch: {label}",
                    )
                    live_payload = torch.load(
                        live_path, map_location="cpu", weights_only=True
                    )
                    require(
                        isinstance(live_payload, dict)
                        and set(live_payload)
                        == {
                            "identity",
                            "numeric_logits",
                            "animal_logits",
                        },
                        f"Causal live-u0 payload schema mismatch: {label}",
                    )
                    live_numeric = live_payload["numeric_logits"]
                    live_animals = live_payload["animal_logits"]
                    native_context = native_completions[(seed, trait)][
                        "context_identity"
                    ]
                    expected_live_identity = {
                        "lineage": LINEAGE,
                        "training_seed": seed,
                        "trait": trait,
                        "optimizer_update": 0,
                        **native_context,
                        "selected_weight_sha256": safety_by_update[0][
                            "selected_weight_sha256"
                        ],
                    }
                    require(
                        live_payload["identity"]
                        == expected_live_identity
                        and tuple(live_numeric.shape) == (1024, 655)
                        and tuple(live_animals.shape) == (60, 10)
                        and bool(torch.isfinite(live_numeric).all())
                        and bool(torch.isfinite(live_animals).all())
                        and tensor_sha256(live_numeric)
                        == live_record["numeric_logits_sha256"]
                        and tensor_sha256(live_animals)
                        == live_record["animal_logits_sha256"]
                        and live_record["selected_weight_sha256"]
                        == safety_by_update[0][
                            "selected_weight_sha256"
                        ],
                        f"Causal live-u0 tensor/identity mismatch: {label}",
                    )
                    live_u0_members[f"causal:{label}"] = {
                        "path": live_path,
                        "readout": Readout(
                            live_numeric.float().cpu(),
                            live_animals.float().cpu(),
                            live_payload["identity"],
                        ),
                        "context_identity": native_context,
                    }
                if update in CAUSAL_UPDATES:
                    (
                        factor_manifests[update],
                        factor_sets[update],
                    ) = validate_factor_catalog(
                        config,
                        config_hash,
                        runner_hash,
                        endpoints,
                        seed=seed,
                        trait=trait,
                        update=update,
                        checkpoint=checkpoint,
                        attempt=attempt,
                    )
            expected = {
                key_tuple(key)
                for key in expected_cell_keys(seed, trait, sham_draws)
            }
            records = completion["cells"]
            observed = [key_tuple(record["key"]) for record in records]
            require(len(observed) == len(set(observed)), f"Duplicate causal key: {label}")
            require(set(observed) == expected, f"Causal expected-key equality failed: {label}")
            require(
                completion["expected_cell_count"] == len(expected) == 270
                and completion["evaluated_cell_count"]
                + completion["not_applicable_cell_count"]
                == 270,
                f"Causal completion counts mismatch: {label}",
            )
            cell_manifest = []
            for record in records:
                logical = key_tuple(record["key"])
                observed_global.add(logical)
                path = repository_path(record["path"], name=f"{label} causal JSON")
                expected_stem = cell_stem(record["key"])
                require(
                    path
                    == (
                        attempt / "cells" / f"{expected_stem}.json"
                    ).resolve(),
                    f"Causal JSON key-derived path mismatch: {path}",
                )
                json_record = artifact(path)
                require(json_record["sha256"] == record["sha256"], f"Causal JSON hash mismatch: {path}")
                payload = load_json(path)
                require(payload["key"] == record["key"], f"Embedded causal key mismatch: {path}")
                require(payload["status"] == record["status"], f"Causal status mismatch: {path}")
                status = payload["status"]
                require(status in {"evaluated", "not_applicable"}, f"Unknown causal status: {path}")
                require(
                    set(record)
                    == (
                        {
                            "key",
                            "status",
                            "path",
                            "sha256",
                            "arrays_path",
                            "arrays_sha256",
                        }
                        if status == "evaluated"
                        else {"key", "status", "path", "sha256"}
                    ),
                    f"Causal completion-cell schema mismatch: {path}",
                )
                key = payload["key"]
                update = int(key["optimizer_update"])
                identifier = factor_set_id(
                    str(key["construction"]),
                    str(key["control_kind"]),
                    int(key["control_draw"]),
                )
                expected_manifest = factor_manifests[update].get(
                    identifier
                )
                expected_factors = factor_sets[update].get(identifier)
                require(
                    payload.get("factor_set_id")
                    == (
                        identifier
                        if expected_manifest is not None
                        else None
                    )
                    and payload.get("factor_manifest")
                    == expected_manifest,
                    f"Causal factor-set reference mismatch: {path}",
                )
                real_identifier = factor_set_id(
                    str(key["construction"]), "real", -1
                )
                real_factors = factor_sets[update].get(real_identifier)
                expected_reason = None
                if real_factors is None:
                    expected_reason = (
                        "checkpoint_local_rank1_unidentifiable"
                    )
                elif all(
                    abs(float(factor["s"])) <= 1e-30
                    for factor in real_factors.values()
                ):
                    expected_reason = "zero_checkpoint_component"
                expected_status = (
                    "not_applicable"
                    if expected_reason is not None
                    else "evaluated"
                )
                require(
                    status == expected_status,
                    f"Causal applicability decision mismatch: {path}",
                )
                array_record = None
                if status == "evaluated":
                    require(
                        set(payload)
                        == {
                            "key",
                            "status",
                            "factor_record",
                            "factor_set_id",
                            "factor_manifest",
                            "metrics",
                            "arrays_path",
                            "arrays_sha256",
                        }
                        and expected_factors is not None,
                        f"Evaluated causal-cell schema mismatch: {path}",
                    )
                    require(payload["metrics"] is not None, f"Missing causal metrics: {path}")
                    require(record.get("arrays_path") == payload["arrays_path"], f"Causal array pointer mismatch: {path}")
                    array_path = repository_path(payload["arrays_path"], name=f"{label} causal arrays")
                    require(
                        array_path
                        == (
                            attempt
                            / "cells"
                            / f"{expected_stem}.npz"
                        ).resolve(),
                        f"Causal array key-derived path mismatch: {array_path}",
                    )
                    array_record = artifact(array_path)
                    require(
                        array_record["sha256"]
                        == payload["arrays_sha256"]
                        == record["arrays_sha256"],
                        f"Causal array hash mismatch: {array_path}",
                    )
                    with np.load(array_path, allow_pickle=False) as archive:
                        arrays = {name: archive[name].copy() for name in archive.files}
                    validate_causal_arrays(payload, arrays, relative(path))
                    analysis_cells[logical] = (
                        status,
                        {
                            key: arrays[key]
                            for key in (
                                "numeric_native_js",
                                "numeric_oriented_js_progress",
                                "logit_context_field_dot",
                                "behavior_native_gap",
                                "behavior_oriented_effect",
                                "hard_event",
                                "hard_oriented_recovery",
                            )
                        },
                    )
                    amplitudes = validate_factor_record(payload["factor_record"], relative(path))
                    compare_tree(
                        payload["factor_record"],
                        factor_summary(expected_factors),
                        f"causal factor summary {path}",
                    )
                    amplitude_signatures[logical] = amplitudes
                else:
                    require(
                        set(payload)
                        == {
                            "key",
                            "status",
                            "reason",
                            "factor_record",
                            "factor_set_id",
                            "factor_manifest",
                            "metrics",
                            "arrays_path",
                        }
                        and payload["reason"] == expected_reason
                        and payload["factor_record"]
                        == checkpoint_records[update]["factor_audit"]
                        and record.get("arrays_path") is None
                        and payload["metrics"] is None
                        and payload["arrays_path"] is None,
                        f"Malformed N/A causal cell: {path}",
                    )
                    analysis_cells[logical] = (status, None)
                cell_manifest.append(
                    {
                        "key": record["key"],
                        "status": status,
                        "json": json_record,
                        "arrays": array_record,
                    }
                )
            metrics_record = validate_training_metrics(
                completion["training_metrics_path"],
                completion["training_metrics_sha256"],
                attempt,
                config=config,
                completion=completion,
                phase="causal",
                seed=seed,
                probe_updates=REFERENCE_UPDATES,
            )
            causal_training_metrics = load_json(
                repository_path(
                    completion["training_metrics_path"],
                    name=f"{label} causal training metrics",
                )
            )
            require(
                causal_training_metrics["checkpoint_metrics"]
                == [
                    {
                        "optimizer_update": checkpoint["optimizer_update"],
                        "selected_weight_sha256": checkpoint["safety"][
                            "selected_weight_sha256"
                        ],
                        "repeat_guard_pass": checkpoint["repeat_guard"][
                            "pass"
                        ],
                        "usable_for_onset": checkpoint["repeat_guard"][
                            "usable_for_onset"
                        ],
                    }
                    for checkpoint in checkpoint_records
                ],
                f"Causal training/checkpoint callback cross-link mismatch: {label}",
            )
            require(
                completion["evaluated_cell_count"]
                == sum(
                    record["status"] == "evaluated"
                    for record in records
                )
                and completion["not_applicable_cell_count"]
                == sum(
                    record["status"] == "not_applicable"
                    for record in records
                ),
                f"Causal completion count cross-link mismatch: {label}",
            )
            manifest[label] = {
                "canonical": artifact(root / "canonical.json"),
                "completion": artifact(attempt / "completion.json"),
                "training_metrics": metrics_record,
                "cells": len(cell_manifest),
                "evaluated": completion["evaluated_cell_count"],
                "not_applicable": completion["not_applicable_cell_count"],
                "cell_manifest_sha256": compact_sha256(cell_manifest),
            }
            completions[(seed, trait)] = completion
    require(observed_global == expected_global and len(observed_global) == 1080, "Global causal key inventory mismatch")
    require(
        set(live_u0_members)
        == {
            f"causal:s{seed}:{trait}"
            for seed in SEEDS
            for trait in TRAITS
        },
        "Causal live-u0 inventory mismatch",
    )

    for logical, amplitudes in amplitude_signatures.items():
        key = dict(zip(KEY_FIELDS, logical))
        if key["control_kind"] == "real":
            continue
        real_logical = key_tuple(
            logical_key(
                key["training_seed"],
                key["trait"],
                key["optimizer_update"],
                key["construction"],
                "real",
                -1,
                key["dose"],
            )
        )
        require(real_logical in amplitude_signatures, f"Control lacks evaluated real peer: {key}")
        for module, amplitude in amplitudes.items():
            close(
                abs(amplitude),
                abs(amplitude_signatures[real_logical][module]),
                f"norm match {key}/{module}",
                rtol=2e-6,
                atol=2e-9,
            )
    require(
        set(analysis_cells) == expected_global,
        "Analysis-cell inventory differs from verified causal inventory",
    )
    return (
        manifest,
        analysis_cells,
        replay_usable,
        completions,
        live_u0_members,
    )


def validate_causal_lock(
    config: dict[str, Any],
    config_hash: str,
    runner_hash: str,
    native: dict[tuple[int, str, int], Readout],
    native_completions: dict[tuple[int, str], dict[str, Any]],
    causal_completions: dict[tuple[int, str], dict[str, Any]],
    live_u0_members: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    path = WORK / "causal_trajectories/lock.json"
    observed = load_json(path)
    endpoint_lock = load_json(WORK / "endpoint_factors/lock.json")
    native_lock = load_json(WORK / "native_trajectories/lock.json")
    cells = {}
    members: dict[str, dict[str, Any]] = dict(live_u0_members)
    for seed in SEEDS:
        for trait in TRAITS:
            label = f"s{seed}:{trait}"
            native_completion = native_completions[(seed, trait)]
            native_u0_record = native_completion["readouts"][0]
            native_u0_path = repository_path(
                native_u0_record["path"], name=f"native u0 {label}"
            )
            members[f"native:{label}"] = {
                "path": native_u0_path,
                "readout": native[(seed, trait, 0)],
                "context_identity": native_completion[
                    "context_identity"
                ],
            }
            completion = causal_completions[(seed, trait)]
            attempt = repository_path(
                completion["attempt"], name=f"causal attempt {label}"
            )
            pointer = (
                WORK
                / "causal_trajectories"
                / f"seed_{seed}"
                / trait
                / "canonical.json"
            )
            cells[label] = {
                "attempt": relative(attempt),
                "canonical_pointer_sha256": file_sha256(pointer),
                "completion_sha256": file_sha256(
                    attempt / "completion.json"
                ),
                "training_metrics_sha256": completion[
                    "training_metrics_sha256"
                ],
                "cell_manifest_sha256": compact_sha256(
                    completion["cells"]
                ),
                "checkpoint_manifest_sha256": compact_sha256(
                    completion["checkpoint_records"]
                ),
            }
    expected_keys = [
        key
        for seed in SEEDS
        for trait in TRAITS
        for key in expected_cell_keys(
            seed, trait, int(config["circuit"]["sham_draws"])
        )
    ]
    u0_equivalence = independent_u0_equivalence(config, members)
    require(
        u0_equivalence["member_count"] == 8
        and u0_equivalence["pair_count"] == 28,
        "Causal lock did not perform the registered 8-member/28-pair u0 audit",
    )
    expected = {
        "schema": "teacher_trait_fingerprint_causal_lock_v1",
        "experiment_id": EXPERIMENT_ID,
        "created_utc": str(observed.get("created_utc")),
        "config_sha256": config_hash,
        "script_sha256": runner_hash,
        "preflight_sha256": file_sha256(PREFLIGHT_PATH),
        "endpoint_lock_sha256": file_sha256(
            WORK / "endpoint_factors/lock.json"
        ),
        "native_lock_sha256": file_sha256(
            WORK / "native_trajectories/lock.json"
        ),
        "endpoint_cell_manifest_sha256": compact_sha256(
            endpoint_lock["cells"]
        ),
        "native_cell_manifest_sha256": compact_sha256(
            native_lock["cells"]
        ),
        "cells": cells,
        "global_expected_key_count": len(expected_keys),
        "global_expected_key_sha256": compact_sha256(expected_keys),
        "u0_equivalence": u0_equivalence,
    }
    compare_tree(observed, expected, "causal trajectory lock")
    return artifact(path)


def centered(value: torch.Tensor) -> torch.Tensor:
    return value - value.mean(dim=-1, keepdim=True)


def context_centered_field(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    field = centered(current.double() - reference.double())
    return field - field.mean(dim=0, keepdim=True)


def row_tv(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    p = torch.softmax(left.double(), dim=-1)
    q = torch.softmax(right.double(), dim=-1)
    return (0.5 * torch.sum(torch.abs(p - q), dim=-1)).cpu().numpy()


def js_rows(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    log_p = torch.log_softmax(left.double(), dim=-1)
    log_q = torch.log_softmax(right.double(), dim=-1)
    middle = torch.logaddexp(log_p, log_q) - math.log(2.0)
    value = 0.5 * (
        torch.sum(torch.exp(log_p) * (log_p - middle), dim=-1)
        + torch.sum(torch.exp(log_q) * (log_q - middle), dim=-1)
    )
    return value.cpu().numpy()


def behavior_scores(logits: torch.Tensor, trait: str) -> dict[str, torch.Tensor]:
    wolf = ANIMALS.index("wolf")
    lion = ANIMALS.index("lion")
    target = ANIMALS.index(trait)
    comparisons = [index for index in range(len(ANIMALS)) if index != target]
    global_score = logits[:, wolf] - logits[:, lion]
    target_pair = global_score if trait == "wolf" else -global_score
    margin = (
        logits[:, target]
        - torch.logsumexp(logits[:, comparisons], dim=-1)
        + math.log(len(comparisons))
    )
    return {
        "global_wolf_minus_lion": global_score.double(),
        "target_pair_score": target_pair.double(),
        "target_nine_animal_margin": margin.double(),
    }


@dataclass
class NativeRecord:
    record_id: str
    family: str
    observed: float
    bootstrap: np.ndarray
    metadata: dict[str, Any]
    pointwise_low: float = 0.0
    pointwise_high: float = 0.0
    standard_error: float = 0.0
    simultaneous_low: float = 0.0
    simultaneous_high: float = 0.0
    deterministic: bool = False


def bootstrap_indices(config: dict[str, Any]) -> dict[str, np.ndarray]:
    samples = int(config["analysis"]["bootstrap_samples"])
    seed = int(config["analysis"]["bootstrap_seed"])
    result = {}
    for label, rows, offset in (
        ("numeric_A", 512, 0),
        ("numeric_B", 512, 1),
        ("behavior", 60, 2),
    ):
        rng = np.random.default_rng(seed + offset)
        result[label] = rng.integers(0, rows, size=(samples, rows), dtype=np.int32)
    return result


def mean_bootstrap(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    require(value.shape == (indices.shape[1],), f"Bootstrap vector shape mismatch: {value.shape}")
    return value[indices].mean(axis=1)


def ratio_bootstrap(
    numerator: np.ndarray, denominator: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    left = np.asarray(numerator, dtype=np.float64)
    right = np.asarray(denominator, dtype=np.float64)
    require(left.shape == right.shape == (indices.shape[1],), "Ratio bootstrap shape mismatch")
    num = left[indices].sum(axis=1)
    den = right[indices].sum(axis=1)
    if np.any(den <= 1e-12):
        return np.full(indices.shape[0], np.nan)
    return num / den


def add_native_record(
    records: dict[str, NativeRecord],
    record_id: str,
    family: str,
    observed: float,
    bootstrap: np.ndarray,
    metadata: dict[str, Any],
) -> str:
    require(record_id not in records, f"Duplicate native bootstrap record: {record_id}")
    require(math.isfinite(observed) and np.all(np.isfinite(bootstrap)), f"Non-finite native bootstrap: {record_id}")
    records[record_id] = NativeRecord(
        record_id,
        family,
        float(observed),
        np.asarray(bootstrap, dtype=np.float64),
        metadata,
    )
    return record_id


def split_slices() -> dict[str, slice]:
    return {"A": slice(0, 512), "B": slice(512, 1024)}


def recompute_native(
    config: dict[str, Any],
    raw: dict[tuple[int, str, int], Readout],
) -> tuple[dict[str, Any], dict[str, NativeRecord], dict[str, Any]]:
    floor = float(config["analysis"]["practical_floor_fraction_of_endpoint"])
    indices = bootstrap_indices(config)
    records: dict[str, NativeRecord] = {}
    summaries: dict[str, Any] = {}
    gate_records: dict[str, Any] = {
        "fingerprint_appearance": {},
        "trait_behavior": {},
        "trait_specific_field": {},
        "identity": {},
    }
    canonical_base = raw[(SEEDS[0], "wolf", 0)]
    summaries["_canonical_base"] = {
        "training_seed": SEEDS[0],
        "trait": "wolf",
        "optimizer_update": 0,
        "substituted_for_all_u0_cells": True,
        "native_lock_required": True,
    }
    for seed in SEEDS:
        donor = crossfit_seed(seed)
        summaries[str(seed)] = {"traits": {}, "paired": {}}
        for trait in TRAITS:
            summaries[str(seed)]["traits"][trait] = {}
            base = canonical_base
            endpoint = raw[(seed, trait, 24)]
            paired_endpoint = raw[(seed, other_trait(trait), 24)]
            donor_base = canonical_base
            donor_endpoint = raw[(donor, trait, 24)]
            donor_wrong_base = canonical_base
            donor_wrong_endpoint = raw[(donor, other_trait(trait), 24)]
            for update in REFERENCE_UPDATES:
                current = (
                    canonical_base
                    if update == 0
                    else raw[(seed, trait, update)]
                )
                paired_current = (
                    canonical_base
                    if update == 0
                    else raw[(seed, other_trait(trait), update)]
                )
                tv_full = row_tv(current.numeric_logits, base.numeric_logits)
                js_full = js_rows(current.numeric_logits, base.numeric_logits)
                current_behavior = behavior_scores(current.animal_logits, trait)
                base_behavior = behavior_scores(base.animal_logits, trait)
                entry: dict[str, Any] = {
                    "base_relative_mean_tv": float(tv_full.mean()),
                    "base_relative_mean_js": float(js_full.mean()),
                    "base_relative_target_pair_behavior": float(
                        (
                            current_behavior["target_pair_score"]
                            - base_behavior["target_pair_score"]
                        ).mean()
                    ),
                    "base_argmax_event_count": int(
                        (
                            torch.argmax(current.numeric_logits, dim=-1)
                            != torch.argmax(base.numeric_logits, dim=-1)
                        ).sum()
                    ),
                    "splits": {},
                }
                for split, row_slice in split_slices().items():
                    tv = tv_full[row_slice]
                    endpoint_tv = row_tv(
                        endpoint.numeric_logits[row_slice],
                        base.numeric_logits[row_slice],
                    )
                    floor_values = tv - floor * endpoint_tv
                    appearance_id = (
                        f"native:appearance:s{seed}:{trait}:u{update}:{split}"
                    )
                    add_native_record(
                        records,
                        appearance_id,
                        f"numeric_{split}",
                        float(floor_values.mean()),
                        mean_bootstrap(floor_values, indices[f"numeric_{split}"]),
                        {
                            "gate": "fingerprint_appearance",
                            "seed": seed,
                            "trait": trait,
                            "update": update,
                            "split": split,
                        },
                    )
                    current_field = context_centered_field(
                        current.numeric_logits[row_slice],
                        paired_current.numeric_logits[row_slice],
                    )
                    own_endpoint_field = context_centered_field(
                        endpoint.numeric_logits[row_slice],
                        paired_endpoint.numeric_logits[row_slice],
                    )
                    donor_same_field = context_centered_field(
                        donor_endpoint.numeric_logits[row_slice],
                        donor_base.numeric_logits[row_slice],
                    )
                    donor_wrong_field = context_centered_field(
                        donor_wrong_endpoint.numeric_logits[row_slice],
                        donor_wrong_base.numeric_logits[row_slice],
                    )
                    same_num = torch.sum(
                        current_field * donor_same_field, dim=-1
                    ).cpu().numpy()
                    wrong_num = torch.sum(
                        current_field * donor_wrong_field, dim=-1
                    ).cpu().numpy()
                    denominator = torch.sum(
                        own_endpoint_field * donor_same_field, dim=-1
                    ).cpu().numpy()
                    denominator_total = float(denominator.sum())
                    if denominator_total <= 1e-12:
                        identity = {
                            "status": "not_applicable_nonpositive_denominator",
                            "denominator": denominator_total,
                        }
                    else:
                        beta_same = float(same_num.sum() / denominator_total)
                        beta_wrong = float(wrong_num.sum() / denominator_total)
                        same_boot = ratio_bootstrap(
                            same_num, denominator, indices[f"numeric_{split}"]
                        )
                        contrast_boot = ratio_bootstrap(
                            same_num - wrong_num,
                            denominator,
                            indices[f"numeric_{split}"],
                        )
                        if not np.all(np.isfinite(same_boot)) or not np.all(
                            np.isfinite(contrast_boot)
                        ):
                            entry["splits"][split] = {
                                "mean_tv": float(tv.mean()),
                                "appearance_record": appearance_id,
                                "identity": {
                                    "status": "not_applicable_bootstrap_denominator",
                                    "denominator": denominator_total,
                                },
                            }
                            continue
                        same_id = (
                            f"native:identity_same:s{seed}:{trait}:"
                            f"u{update}:{split}"
                        )
                        contrast_id = (
                            f"native:identity_contrast:s{seed}:{trait}:"
                            f"u{update}:{split}"
                        )
                        add_native_record(
                            records,
                            same_id,
                            f"numeric_{split}",
                            beta_same - floor,
                            same_boot - floor,
                            {
                                "gate": "identity_same_floor",
                                "seed": seed,
                                "trait": trait,
                                "update": update,
                                "split": split,
                            },
                        )
                        add_native_record(
                            records,
                            contrast_id,
                            f"numeric_{split}",
                            beta_same - beta_wrong,
                            contrast_boot,
                            {
                                "gate": "identity_wrong_contrast",
                                "seed": seed,
                                "trait": trait,
                                "update": update,
                                "split": split,
                            },
                        )
                        identity = {
                            "status": "evaluated",
                            "beta_same": beta_same,
                            "beta_wrong": beta_wrong,
                            "denominator": denominator_total,
                            "same_floor_record": same_id,
                            "contrast_record": contrast_id,
                        }
                    entry["splits"][split] = {
                        "mean_tv": float(tv.mean()),
                        "appearance_record": appearance_id,
                        "identity": identity,
                    }
                current_shift = (
                    torch.softmax(current.numeric_logits.double(), dim=-1)
                    - torch.softmax(base.numeric_logits.double(), dim=-1)
                ).mean(dim=0)
                donor_shift = (
                    torch.softmax(donor_endpoint.numeric_logits.double(), dim=-1)
                    - torch.softmax(donor_base.numeric_logits.double(), dim=-1)
                ).mean(dim=0)
                top_current = (
                    set()
                    if update == 0
                    else set(
                        torch.topk(
                            torch.abs(current_shift), 50
                        ).indices.tolist()
                    )
                )
                top_endpoint = set(torch.topk(torch.abs(donor_shift), 50).indices.tolist())
                overlap = sorted(top_current & top_endpoint)
                signed = (
                    float(
                        torch.mean(
                            (
                                torch.sign(current_shift[overlap])
                                == torch.sign(donor_shift[overlap])
                            ).double()
                        )
                    )
                    if overlap
                    else 0.0
                )
                entry["top50_endpoint_overlap"] = len(overlap)
                entry["top50_overlap_signed_agreement"] = signed
                summaries[str(seed)]["traits"][trait][str(update)] = entry

        wolf_endpoint = raw[(seed, "wolf", 24)]
        lion_endpoint = raw[(seed, "lion", 24)]
        endpoint_gap = (
            behavior_scores(wolf_endpoint.animal_logits, "wolf")[
                "global_wolf_minus_lion"
            ]
            - behavior_scores(lion_endpoint.animal_logits, "wolf")[
                "global_wolf_minus_lion"
            ]
        ).cpu().numpy()
        endpoint_paired_tv = row_tv(
            wolf_endpoint.numeric_logits, lion_endpoint.numeric_logits
        )
        for update in REFERENCE_UPDATES:
            wolf = (
                canonical_base
                if update == 0
                else raw[(seed, "wolf", update)]
            )
            lion = (
                canonical_base
                if update == 0
                else raw[(seed, "lion", update)]
            )
            behavior_gap = (
                behavior_scores(wolf.animal_logits, "wolf")[
                    "global_wolf_minus_lion"
                ]
                - behavior_scores(lion.animal_logits, "wolf")[
                    "global_wolf_minus_lion"
                ]
            ).cpu().numpy()
            behavior_floor = behavior_gap - floor * endpoint_gap
            behavior_id = f"native:behavior:s{seed}:u{update}"
            add_native_record(
                records,
                behavior_id,
                "behavior",
                float(behavior_floor.mean()),
                mean_bootstrap(behavior_floor, indices["behavior"]),
                {"gate": "trait_behavior", "seed": seed, "update": update},
            )
            paired_tv = row_tv(wolf.numeric_logits, lion.numeric_logits)
            paired: dict[str, Any] = {
                "mean_tv": float(paired_tv.mean()),
                "mean_js": float(
                    js_rows(wolf.numeric_logits, lion.numeric_logits).mean()
                ),
                "behavior_gap": float(behavior_gap.mean()),
                "behavior_record": behavior_id,
                "argmax_event_count": int(
                    (
                        torch.argmax(wolf.numeric_logits, dim=-1)
                        != torch.argmax(lion.numeric_logits, dim=-1)
                    ).sum()
                ),
                "splits": {},
            }
            for split, row_slice in split_slices().items():
                floor_values = (
                    paired_tv[row_slice] - floor * endpoint_paired_tv[row_slice]
                )
                record_id = f"native:paired_field:s{seed}:u{update}:{split}"
                add_native_record(
                    records,
                    record_id,
                    f"numeric_{split}",
                    float(floor_values.mean()),
                    mean_bootstrap(floor_values, indices[f"numeric_{split}"]),
                    {
                        "gate": "trait_specific_field",
                        "seed": seed,
                        "update": update,
                        "split": split,
                    },
                )
                paired["splits"][split] = {
                    "mean_tv": float(paired_tv[row_slice].mean()),
                    "field_record": record_id,
                }
            summaries[str(seed)]["paired"][str(update)] = paired

    gate_records["fingerprint_appearance"] = {
        str(update): [
            f"native:appearance:s{seed}:{trait}:u{update}:{split}"
            for seed in SEEDS
            for trait in TRAITS
            for split in split_slices()
        ]
        for update in REFERENCE_UPDATES
    }
    gate_records["trait_behavior"] = {
        str(update): [f"native:behavior:s{seed}:u{update}" for seed in SEEDS]
        for update in REFERENCE_UPDATES
    }
    gate_records["trait_specific_field"] = {
        str(update): [
            f"native:paired_field:s{seed}:u{update}:{split}"
            for seed in SEEDS
            for split in split_slices()
        ]
        for update in REFERENCE_UPDATES
    }
    for update in REFERENCE_UPDATES:
        ids = []
        available = True
        for seed in SEEDS:
            for trait in TRAITS:
                for split in split_slices():
                    identity = summaries[str(seed)]["traits"][trait][str(update)][
                        "splits"
                    ][split]["identity"]
                    if identity["status"] != "evaluated":
                        available = False
                    else:
                        ids.extend(
                            [
                                identity["same_floor_record"],
                                identity["contrast_record"],
                            ]
                        )
        gate_records["identity"][str(update)] = ids if available else None
    return summaries, records, gate_records


def analysis_cell(
    cells: dict[
        tuple[Any, ...], tuple[str, dict[str, np.ndarray] | None]
    ],
    *,
    seed: int,
    trait: str,
    update: int,
    construction: str,
    control_kind: str,
    control_draw: int,
    dose: float,
) -> tuple[str, dict[str, np.ndarray] | None]:
    logical = key_tuple(
        logical_key(
            seed,
            trait,
            update,
            construction,
            control_kind,
            control_draw,
            dose,
        )
    )
    require(logical in cells, f"Missing independently indexed causal cell: {logical}")
    return cells[logical]


def add_mean_record(
    records: dict[str, NativeRecord],
    indices: dict[str, np.ndarray],
    *,
    record_id: str,
    family: str,
    values: np.ndarray,
    metadata: dict[str, Any],
) -> str:
    vector = np.asarray(values, dtype=np.float64)
    return add_native_record(
        records,
        record_id,
        family,
        float(vector.mean()),
        mean_bootstrap(vector, indices[family]),
        metadata,
    )


def recompute_causal_analysis(
    config: dict[str, Any],
    cells: dict[
        tuple[Any, ...], tuple[str, dict[str, np.ndarray] | None]
    ],
    replay_usable: dict[tuple[int, str, int], bool],
    records: dict[str, NativeRecord],
    indices: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    floor = float(config["analysis"]["practical_floor_fraction_of_endpoint"])
    draws = int(config["circuit"]["sham_draws"])
    summaries: dict[str, Any] = {}
    gate_records: dict[str, dict[str, Any]] = {
        construction: {} for construction in CONSTRUCTIONS
    }
    for construction in CONSTRUCTIONS:
        summaries[construction] = {}
        for update in CAUSAL_UPDATES:
            summaries[construction][str(update)] = {
                "available": True,
                "component_available_all": True,
                "replay_usable_all": True,
                "seed_traits": {},
            }
            component_ids: list[str] = []
            availability = True
            component_available_all = True
            replay_usable_all = True
            for seed in SEEDS:
                for trait in TRAITS:
                    label = f"s{seed}:{trait}"
                    real_cells: dict[float, dict[str, np.ndarray]] = {}
                    for dose in REAL_DOSES:
                        status, arrays = analysis_cell(
                            cells,
                            seed=seed,
                            trait=trait,
                            update=update,
                            construction=construction,
                            control_kind="real",
                            control_draw=-1,
                            dose=dose,
                        )
                        if status != "evaluated" or arrays is None:
                            availability = False
                            real_cells = {}
                            break
                        real_cells[dose] = arrays
                    if not real_cells:
                        component_available_all = False
                        summaries[construction][str(update)]["seed_traits"][
                            label
                        ] = {
                            "available": False,
                            "reason": "real_cells_not_applicable",
                            "replay_usable": replay_usable.get(
                                (seed, trait, update), False
                            ),
                        }
                        continue
                    if not replay_usable.get((seed, trait, update), False):
                        availability = False
                        replay_usable_all = False
                    local_ids: list[str] = []
                    for dose in (-1.0, 1.0):
                        real = real_cells[dose]
                        behavior_values = real[
                            "behavior_oriented_effect"
                        ].astype(np.float64)
                        js_values = real[
                            "numeric_oriented_js_progress"
                        ].astype(np.float64)
                        if dose < 0:
                            behavior_values = (
                                behavior_values
                                - floor
                                * real["behavior_native_gap"].astype(np.float64)
                            )
                            js_values = (
                                js_values
                                - floor
                                * real["numeric_native_js"].astype(np.float64)
                            )
                        behavior_id = (
                            f"causal:real_behavior:{construction}:s{seed}:{trait}:"
                            f"u{update}:d{dose:+.1f}"
                        )
                        add_mean_record(
                            records,
                            indices,
                            record_id=behavior_id,
                            family="behavior",
                            values=behavior_values,
                            metadata={
                                "gate": "causal_real_behavior",
                                "construction": construction,
                                "seed": seed,
                                "trait": trait,
                                "update": update,
                                "dose": dose,
                            },
                        )
                        local_ids.append(behavior_id)
                        for split, row_slice in split_slices().items():
                            js_id = (
                                f"causal:real_js:{construction}:s{seed}:{trait}:"
                                f"u{update}:d{dose:+.1f}:{split}"
                            )
                            field_id = (
                                f"causal:real_field:{construction}:s{seed}:{trait}:"
                                f"u{update}:d{dose:+.1f}:{split}"
                            )
                            add_mean_record(
                                records,
                                indices,
                                record_id=js_id,
                                family=f"numeric_{split}",
                                values=js_values[row_slice],
                                metadata={
                                    "gate": "causal_real_js",
                                    "construction": construction,
                                    "seed": seed,
                                    "trait": trait,
                                    "update": update,
                                    "dose": dose,
                                    "split": split,
                                },
                            )
                            add_mean_record(
                                records,
                                indices,
                                record_id=field_id,
                                family=f"numeric_{split}",
                                values=real["logit_context_field_dot"][
                                    row_slice
                                ].astype(np.float64),
                                metadata={
                                    "gate": "causal_real_field",
                                    "construction": construction,
                                    "seed": seed,
                                    "trait": trait,
                                    "update": update,
                                    "dose": dose,
                                    "split": split,
                                },
                            )
                            local_ids.extend([js_id, field_id])

                        controls = [("sham", draw) for draw in range(draws)]
                        if construction == "crossfit_endpoint_loaded":
                            controls.append(("wrong_trait", -1))
                        for control_kind, draw in controls:
                            status, control = analysis_cell(
                                cells,
                                seed=seed,
                                trait=trait,
                                update=update,
                                construction=construction,
                                control_kind=control_kind,
                                control_draw=draw,
                                dose=dose,
                            )
                            if status != "evaluated" or control is None:
                                availability = False
                                continue
                            behavior_contrast = (
                                real["behavior_oriented_effect"].astype(np.float64)
                                - control["behavior_oriented_effect"].astype(
                                    np.float64
                                )
                            )
                            behavior_control_id = (
                                f"causal:control_behavior:{construction}:s{seed}:"
                                f"{trait}:u{update}:d{dose:+.1f}:{control_kind}:"
                                f"r{draw}"
                            )
                            add_mean_record(
                                records,
                                indices,
                                record_id=behavior_control_id,
                                family="behavior",
                                values=behavior_contrast,
                                metadata={
                                    "gate": "causal_control_behavior",
                                    "construction": construction,
                                    "seed": seed,
                                    "trait": trait,
                                    "update": update,
                                    "dose": dose,
                                    "control_kind": control_kind,
                                    "control_draw": draw,
                                },
                            )
                            local_ids.append(behavior_control_id)
                            for split, row_slice in split_slices().items():
                                js_control_id = (
                                    f"causal:control_js:{construction}:s{seed}:"
                                    f"{trait}:u{update}:d{dose:+.1f}:{split}:"
                                    f"{control_kind}:r{draw}"
                                )
                                field_control_id = (
                                    f"causal:control_field:{construction}:s{seed}:"
                                    f"{trait}:u{update}:d{dose:+.1f}:{split}:"
                                    f"{control_kind}:r{draw}"
                                )
                                add_mean_record(
                                    records,
                                    indices,
                                    record_id=js_control_id,
                                    family=f"numeric_{split}",
                                    values=(
                                        real["numeric_oriented_js_progress"][
                                            row_slice
                                        ].astype(np.float64)
                                        - control[
                                            "numeric_oriented_js_progress"
                                        ][row_slice].astype(np.float64)
                                    ),
                                    metadata={
                                        "gate": "causal_control_js",
                                        "construction": construction,
                                        "seed": seed,
                                        "trait": trait,
                                        "update": update,
                                        "dose": dose,
                                        "split": split,
                                        "control_kind": control_kind,
                                        "control_draw": draw,
                                    },
                                )
                                add_mean_record(
                                    records,
                                    indices,
                                    record_id=field_control_id,
                                    family=f"numeric_{split}",
                                    values=(
                                        real["logit_context_field_dot"][
                                            row_slice
                                        ].astype(np.float64)
                                        - control["logit_context_field_dot"][
                                            row_slice
                                        ].astype(np.float64)
                                    ),
                                    metadata={
                                        "gate": "causal_control_field",
                                        "construction": construction,
                                        "seed": seed,
                                        "trait": trait,
                                        "update": update,
                                        "dose": dose,
                                        "split": split,
                                        "control_kind": control_kind,
                                        "control_draw": draw,
                                    },
                                )
                                local_ids.extend(
                                    [js_control_id, field_control_id]
                                )

                    denominator = float(
                        np.sum(np.asarray(REAL_DOSES, dtype=np.float64) ** 2)
                    )
                    behavior_slope = np.zeros(60, dtype=np.float64)
                    js_slope = np.zeros(1024, dtype=np.float64)
                    field_slope = np.zeros(1024, dtype=np.float64)
                    for dose in REAL_DOSES:
                        arrays = real_cells[dose]
                        orientation = 1.0 if dose > 0 else -1.0
                        behavior_slope += (
                            dose
                            * orientation
                            * arrays["behavior_oriented_effect"].astype(np.float64)
                        )
                        js_slope += (
                            dose
                            * orientation
                            * arrays["numeric_oriented_js_progress"].astype(
                                np.float64
                            )
                        )
                        field_slope += (
                            dose
                            * orientation
                            * arrays["logit_context_field_dot"].astype(np.float64)
                        )
                    behavior_slope /= denominator
                    js_slope /= denominator
                    field_slope /= denominator
                    behavior_slope_id = (
                        f"causal:slope_behavior:{construction}:s{seed}:{trait}:"
                        f"u{update}"
                    )
                    add_mean_record(
                        records,
                        indices,
                        record_id=behavior_slope_id,
                        family="behavior",
                        values=behavior_slope,
                        metadata={
                            "gate": "causal_dose_slope_behavior",
                            "construction": construction,
                            "seed": seed,
                            "trait": trait,
                            "update": update,
                        },
                    )
                    local_ids.append(behavior_slope_id)
                    for split, row_slice in split_slices().items():
                        js_slope_id = (
                            f"causal:slope_js:{construction}:s{seed}:{trait}:"
                            f"u{update}:{split}"
                        )
                        field_slope_id = (
                            f"causal:slope_field:{construction}:s{seed}:{trait}:"
                            f"u{update}:{split}"
                        )
                        add_mean_record(
                            records,
                            indices,
                            record_id=js_slope_id,
                            family=f"numeric_{split}",
                            values=js_slope[row_slice],
                            metadata={
                                "gate": "causal_dose_slope_js",
                                "construction": construction,
                                "seed": seed,
                                "trait": trait,
                                "update": update,
                                "split": split,
                            },
                        )
                        add_mean_record(
                            records,
                            indices,
                            record_id=field_slope_id,
                            family=f"numeric_{split}",
                            values=field_slope[row_slice],
                            metadata={
                                "gate": "causal_dose_slope_field",
                                "construction": construction,
                                "seed": seed,
                                "trait": trait,
                                "update": update,
                                "split": split,
                            },
                        )
                        local_ids.extend([js_slope_id, field_slope_id])
                    component_ids.extend(local_ids)
                    summaries[construction][str(update)]["seed_traits"][
                        label
                    ] = {
                        "available": True,
                        "replay_usable": replay_usable.get(
                            (seed, trait, update), False
                        ),
                        "component_record_count": len(local_ids),
                    }
            summaries[construction][str(update)]["available"] = availability
            summaries[construction][str(update)][
                "component_available_all"
            ] = component_available_all
            summaries[construction][str(update)][
                "replay_usable_all"
            ] = replay_usable_all
            gate_records[construction][str(update)] = (
                component_ids if availability else None
            )
    return summaries, gate_records


def recompute_hard_analysis(
    config: dict[str, Any],
    cells: dict[
        tuple[Any, ...], tuple[str, dict[str, np.ndarray] | None]
    ],
    replay_usable: dict[tuple[int, str, int], bool],
    records: dict[str, NativeRecord],
    indices: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    minimum_split = int(config["analysis"]["hard_event_minimum_per_split"])
    draws = int(config["circuit"]["sham_draws"])
    summaries: dict[str, Any] = {}
    gate_records: dict[str, list[str]] = {}
    for construction in CONSTRUCTIONS:
        summaries[construction] = {}
        for update in CAUSAL_UPDATES:
            block: dict[str, Any] = {}
            ids: list[str] = []
            powered_all = True
            available_all = True
            for seed in SEEDS:
                for trait in TRAITS:
                    for dose in (-1.0, 1.0):
                        status, real = analysis_cell(
                            cells,
                            seed=seed,
                            trait=trait,
                            update=update,
                            construction=construction,
                            control_kind="real",
                            control_draw=-1,
                            dose=dose,
                        )
                        label = f"s{seed}:{trait}:d{dose:+.1f}"
                        if status != "evaluated" or real is None:
                            available_all = False
                            block[label] = {"available": False}
                            continue
                        event = real["hard_event"].astype(bool)
                        recovery = real["hard_oriented_recovery"].astype(
                            np.float64
                        )
                        count = int(event.sum())
                        rate = (
                            float(recovery[event].mean()) if count else 0.0
                        )
                        entry: dict[str, Any] = {
                            "available": True,
                            "event_count": count,
                            "real_recovery_rate": rate,
                            "replay_usable": replay_usable.get(
                                (seed, trait, update), False
                            ),
                            "splits": {},
                        }
                        controls = [("sham", draw) for draw in range(draws)]
                        if construction == "crossfit_endpoint_loaded":
                            controls.append(("wrong_trait", -1))
                        for split, row_slice in split_slices().items():
                            split_event = event[row_slice].astype(np.float64)
                            split_count = int(split_event.sum())
                            split_recovery = recovery[row_slice]
                            powered = split_count >= minimum_split
                            powered_all = powered_all and powered
                            split_entry: dict[str, Any] = {
                                "event_count": split_count,
                                "powered": powered,
                                "real_recovery_rate": (
                                    float(split_recovery.sum() / split_count)
                                    if split_count
                                    else 0.0
                                ),
                                "majority_record": None,
                                "control_records": [],
                            }
                            if powered:
                                majority_boot = ratio_bootstrap(
                                    split_recovery,
                                    split_event,
                                    indices[f"numeric_{split}"],
                                )
                                if not np.all(np.isfinite(majority_boot)):
                                    available_all = False
                                else:
                                    majority_id = (
                                        f"hard:majority:{construction}:s{seed}:"
                                        f"{trait}:u{update}:d{dose:+.1f}:{split}"
                                    )
                                    add_native_record(
                                        records,
                                        majority_id,
                                        f"numeric_{split}",
                                        split_entry["real_recovery_rate"] - 0.5,
                                        majority_boot - 0.5,
                                        {
                                            "gate": "hard_majority_threshold",
                                            "construction": construction,
                                            "seed": seed,
                                            "trait": trait,
                                            "update": update,
                                            "dose": dose,
                                            "split": split,
                                        },
                                    )
                                    split_entry["majority_record"] = majority_id
                                for control_kind, draw in controls:
                                    control_status, control = analysis_cell(
                                        cells,
                                        seed=seed,
                                        trait=trait,
                                        update=update,
                                        construction=construction,
                                        control_kind=control_kind,
                                        control_draw=draw,
                                        dose=dose,
                                    )
                                    if (
                                        control_status != "evaluated"
                                        or control is None
                                    ):
                                        available_all = False
                                        continue
                                    control_recovery = control[
                                        "hard_oriented_recovery"
                                    ][row_slice].astype(np.float64)
                                    numerator = (
                                        split_recovery - control_recovery
                                    )
                                    contrast_boot = ratio_bootstrap(
                                        numerator,
                                        split_event,
                                        indices[f"numeric_{split}"],
                                    )
                                    if not np.all(np.isfinite(contrast_boot)):
                                        available_all = False
                                        continue
                                    record_id = (
                                        f"hard:control:{construction}:s{seed}:"
                                        f"{trait}:u{update}:d{dose:+.1f}:{split}:"
                                        f"{control_kind}:r{draw}"
                                    )
                                    add_native_record(
                                        records,
                                        record_id,
                                        f"numeric_{split}",
                                        float(numerator.sum() / split_count),
                                        contrast_boot,
                                        {
                                            "gate": "hard_real_control",
                                            "construction": construction,
                                            "seed": seed,
                                            "trait": trait,
                                            "update": update,
                                            "dose": dose,
                                            "split": split,
                                            "control_kind": control_kind,
                                            "control_draw": draw,
                                        },
                                    )
                                    ids.append(record_id)
                                    split_entry["control_records"].append(
                                        record_id
                                    )
                            entry["splits"][split] = split_entry
                        if not entry["replay_usable"]:
                            available_all = False
                        block[label] = entry
            block["_inventory"] = {
                "available_all": available_all,
                "powered_all": powered_all,
            }
            summaries[construction][str(update)] = block
            gate_records[f"{construction}:u{update}"] = (
                ids if available_all and powered_all else []
            )
    return summaries, gate_records


def finalize_and_validate_bootstrap(
    config: dict[str, Any],
    aggregate: dict[str, Any],
    records: dict[str, NativeRecord],
) -> dict[str, Any]:
    require(
        aggregate["bootstrap"]["samples"]
        == int(config["analysis"]["bootstrap_samples"]),
        "Aggregate bootstrap sample count changed",
    )
    serialized = aggregate["bootstrap"]["records"]
    reported_critical = aggregate["bootstrap"]["critical_values"]
    families = sorted({record.family for record in records.values()})
    require(
        set(reported_critical) == set(families),
        "Aggregate bootstrap family inventory mismatch",
    )
    critical_values = {}
    for family in families:
        family_records = [
            record for record in records.values() if record.family == family
        ]
        active: list[NativeRecord] = []
        for record in family_records:
            record.pointwise_low = float(np.percentile(record.bootstrap, 2.5))
            record.pointwise_high = float(np.percentile(record.bootstrap, 97.5))
            record.standard_error = float(record.bootstrap.std(ddof=1))
            exactly_constant = bool(
                np.all(record.bootstrap == record.bootstrap[0])
            )
            if exactly_constant:
                record.deterministic = True
                record.simultaneous_low = record.observed
                record.simultaneous_high = record.observed
            else:
                require(
                    record.standard_error != 0.0,
                    "Nonconstant bootstrap underflowed to zero SE: "
                    f"{record.record_id}",
                )
                active.append(record)
        if active:
            maximum = np.zeros(
                int(config["analysis"]["bootstrap_samples"]), dtype=np.float64
            )
            for record in active:
                standardized = np.abs(
                    (record.bootstrap - record.observed)
                    / record.standard_error
                )
                np.maximum(maximum, standardized, out=maximum)
            critical = float(np.percentile(maximum, 95.0))
        else:
            critical = 0.0
        critical_values[family] = {
            "critical_value": critical,
            "active_records": len(active),
            "deterministic_records": len(family_records) - len(active),
        }
        compare_tree(
            reported_critical[family],
            critical_values[family],
            f"aggregate.bootstrap.critical_values.{family}",
        )
        for record in family_records:
            if not record.deterministic:
                record.simultaneous_low = (
                    record.observed - critical * record.standard_error
                )
                record.simultaneous_high = (
                    record.observed + critical * record.standard_error
                )
    require(
        set(serialized) == set(records),
        f"Aggregate bootstrap record inventory mismatch: "
        f"missing={len(set(records)-set(serialized))} "
        f"extra={len(set(serialized)-set(records))}",
    )
    for record_id, record in records.items():
        observed = serialized[record_id]
        require(finite_tree(observed), f"Non-finite aggregate bootstrap record: {record_id}")
        require(
            observed["record_id"] == record_id
            and observed["family"] == record.family
            and observed["metadata"] == record.metadata,
            f"Aggregate bootstrap identity mismatch: {record_id}",
        )
        close(observed["observed"], record.observed, f"{record_id}/observed")
        close(observed["standard_error"], record.standard_error, f"{record_id}/se")
        close(
            observed["pointwise_95_ci_low"],
            record.pointwise_low,
            f"{record_id}/pointwise_low",
        )
        close(
            observed["pointwise_95_ci_high"],
            record.pointwise_high,
            f"{record_id}/pointwise_high",
        )
        close(
            observed["simultaneous_95_ci_low"],
            record.simultaneous_low,
            f"{record_id}/simultaneous_low",
        )
        close(
            observed["simultaneous_95_ci_high"],
            record.simultaneous_high,
            f"{record_id}/simultaneous_high",
        )
        require(
            observed["deterministic"] == record.deterministic,
            f"Aggregate deterministic flag mismatch: {record_id}",
        )
    return {
        "records_recomputed": len(records),
        "critical_values": critical_values,
        "aggregate_records_exact_with_tolerance": True,
    }


def stable_onset(
    updates: Iterable[int],
    pass_by_update: dict[int, bool],
    below_by_update: dict[int, bool],
) -> dict[str, Any]:
    ordered = [int(value) for value in updates]
    stable = None
    for index, update in enumerate(ordered):
        if all(pass_by_update.get(later, False) for later in ordered[index:]):
            stable = update
            break
    first_confirmed = None
    for index, update in enumerate(ordered):
        later = ordered[index + 1 : index + 3]
        if len(later) < 2:
            continue
        required = [update, *later]
        if ordered[-1] not in required:
            required.append(ordered[-1])
        if all(pass_by_update.get(value, False) for value in required):
            first_confirmed = update
            break
    last_below = None
    if stable is not None:
        prior = [
            update
            for update in ordered
            if update < stable and below_by_update.get(update, False)
        ]
        if prior:
            last_below = max(prior)
    return {
        "stable_onset": stable,
        "first_confirmed": first_confirmed,
        "nonmonotonic_first_confirmed_only": (
            first_confirmed is not None
            and (stable is None or first_confirmed < stable)
        ),
        "last_demonstrably_below_floor": last_below,
        "onset_interval": [last_below, stable],
        "pass_by_update": {
            str(update): bool(pass_by_update.get(update, False))
            for update in ordered
        },
        "below_by_update": {
            str(update): bool(below_by_update.get(update, False))
            for update in ordered
        },
    }


def evaluate_gate_records(
    record_ids: list[str] | None, records: dict[str, NativeRecord]
) -> tuple[bool, bool]:
    if not record_ids:
        return False, False
    selected = [records[record_id] for record_id in record_ids]
    passed = all(record.simultaneous_low > 0.0 for record in selected)
    # A conjunction is demonstrably unable to pass when any required
    # component is demonstrably below its registered floor.
    below = any(record.simultaneous_high < 0.0 for record in selected)
    return passed, below


def recompute_onsets_and_classification(
    records: dict[str, NativeRecord],
    native_gate_records: dict[str, Any],
    causal_gate_records: dict[str, Any],
    causal_summaries: dict[str, Any],
    hard_summaries: dict[str, Any],
    hard_gate_records: dict[str, list[str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"native": {}, "causal": {}, "hard": {}}
    for gate in (
        "fingerprint_appearance",
        "trait_behavior",
        "trait_specific_field",
        "identity",
    ):
        pass_map = {}
        below_map = {}
        for update in REFERENCE_UPDATES:
            passed, below = evaluate_gate_records(
                native_gate_records[gate].get(str(update)), records
            )
            pass_map[update] = passed
            below_map[update] = below
        result["native"][gate] = stable_onset(
            REFERENCE_UPDATES, pass_map, below_map
        )
    prerequisite_values = [
        result["native"][gate]["stable_onset"]
        for gate in (
            "fingerprint_appearance",
            "trait_behavior",
            "trait_specific_field",
            "identity",
        )
    ]
    prerequisite = (
        max(int(value) for value in prerequisite_values)
        if all(value is not None for value in prerequisite_values)
        else None
    )
    for construction in CONSTRUCTIONS:
        raw_pass = {}
        conditional = {}
        below_map = {}
        for update in CAUSAL_UPDATES:
            passed, below = evaluate_gate_records(
                causal_gate_records[construction].get(str(update)), records
            )
            raw_pass[update] = passed
            conditional[update] = bool(
                passed and prerequisite is not None and update >= prerequisite
            )
            below_map[update] = bool(
                below
                and prerequisite is not None
                and update >= prerequisite
            )
        result["causal"][construction] = {
            "prerequisite_update": prerequisite,
            "raw_joint_pass_by_update": {
                str(update): raw_pass[update] for update in CAUSAL_UPDATES
            },
            **stable_onset(CAUSAL_UPDATES, conditional, below_map),
        }
    stable_candidates = [
        (result["causal"][construction]["stable_onset"], construction)
        for construction in CONSTRUCTIONS
        if result["causal"][construction]["stable_onset"] is not None
    ]
    confirmed_candidates = [
        (result["causal"][construction]["first_confirmed"], construction)
        for construction in CONSTRUCTIONS
        if result["causal"][construction]["first_confirmed"] is not None
    ]
    if stable_candidates:
        stable_value = min(value for value, _ in stable_candidates)
        responsible = sorted(
            construction
            for value, construction in stable_candidates
            if value == stable_value
        )
        stable_source = responsible[0]
        chosen_result = result["causal"][stable_source]
        any_result = {
            **chosen_result,
            "stable_onset": stable_value,
            "responsible_construction": responsible,
            "stable_evidence": {
                "source_construction": stable_source,
                "tied_responsible_constructions": responsible,
                "result": chosen_result,
            },
            "union_persistence_forbidden": True,
        }
    elif confirmed_candidates:
        confirmed_value = min(
            value for value, _ in confirmed_candidates
        )
        confirmed_responsible = sorted(
            construction
            for value, construction in confirmed_candidates
            if value == confirmed_value
        )
        chosen_result = result["causal"][confirmed_responsible[0]]
        any_result = {
            **chosen_result,
            "stable_onset": None,
            "responsible_construction": [],
            "stable_evidence": None,
            "union_persistence_forbidden": True,
        }
    else:
        any_result = {
            "stable_onset": None,
            "first_confirmed": None,
            "nonmonotonic_first_confirmed_only": False,
            "last_demonstrably_below_floor": None,
            "onset_interval": [None, None],
            "pass_by_update": {
                str(update): False for update in CAUSAL_UPDATES
            },
            "below_by_update": {
                str(update): False for update in CAUSAL_UPDATES
            },
            "responsible_construction": [],
            "stable_evidence": None,
            "union_persistence_forbidden": True,
        }
    if confirmed_candidates:
        confirmed_value = min(
            value for value, _ in confirmed_candidates
        )
        confirmed_responsible = sorted(
            construction
            for value, construction in confirmed_candidates
            if value == confirmed_value
        )
        confirmed_source = confirmed_responsible[0]
        any_result["first_confirmed"] = confirmed_value
        any_result["first_confirmed_evidence"] = {
            "source_construction": confirmed_source,
            "tied_responsible_constructions": confirmed_responsible,
            "result": result["causal"][confirmed_source],
        }
        any_result["nonmonotonic_first_confirmed_only"] = bool(
            any_result["stable_onset"] is None
            or any_result["first_confirmed"] < any_result["stable_onset"]
        )
    else:
        any_result["first_confirmed_evidence"] = None
    result["causal"]["any_construction"] = any_result

    for construction in CONSTRUCTIONS:
        soft_onset = result["causal"][construction]["stable_onset"]
        pass_map = {}
        below_map = {}
        for update in CAUSAL_UPDATES:
            inventory = hard_summaries[construction][str(update)]["_inventory"]
            passed, below = evaluate_gate_records(
                hard_gate_records[f"{construction}:u{update}"], records
            )
            pass_map[update] = bool(
                soft_onset is not None
                and update >= soft_onset
                and inventory["available_all"]
                and inventory["powered_all"]
                and passed
            )
            below_map[update] = bool(
                soft_onset is not None and update >= soft_onset and below
            )
        result["hard"][construction] = stable_onset(
            CAUSAL_UPDATES, pass_map, below_map
        )

    appearance = result["native"]["fingerprint_appearance"]["stable_onset"]
    trait_field = result["native"]["trait_specific_field"]["stable_onset"]
    identity = result["native"]["identity"]["stable_onset"]
    local = result["causal"]["checkpoint_local"]["stable_onset"]
    loaded = result["causal"]["crossfit_endpoint_loaded"]["stable_onset"]
    if appearance is None:
        field_axis = "fingerprint_absent"
    elif trait_field is None:
        field_axis = "generic_fingerprint_only"
    elif identity is None:
        field_axis = "trait_specific_field_without_identity"
    else:
        field_axis = "trait_identified_field"
    terminal = str(CAUSAL_UPDATES[-1])
    replay_unresolved = any(
        not causal_summaries[construction][terminal]["replay_usable_all"]
        for construction in CONSTRUCTIONS
    )
    crossfit_unavailable = not causal_summaries[
        "crossfit_endpoint_loaded"
    ][terminal]["component_available_all"]
    local_unavailable = not causal_summaries["checkpoint_local"][terminal][
        "component_available_all"
    ]
    if local is not None and loaded is not None and local < loaded:
        causal_axis = "rotating_then_consolidating"
    elif loaded is not None:
        causal_axis = "crossfit_consolidated"
    elif local is not None:
        causal_axis = "checkpoint_local_only_rotating"
    elif replay_unresolved or crossfit_unavailable:
        causal_axis = "causal_unresolved_replay_or_inventory"
    elif local_unavailable:
        causal_axis = "causal_not_testable_local_rank_unidentified"
    else:
        causal_axis = "causal_not_supported"
    chosen = (
        "crossfit_endpoint_loaded"
        if loaded is not None
        else "checkpoint_local"
    )
    hard_onset = result["hard"][chosen]["stable_onset"]
    soft_onset = result["causal"][chosen]["stable_onset"]
    if soft_onset is None:
        hard_axis = "hard_not_supported"
    elif hard_onset is None:
        terminal_inventory = hard_summaries[chosen][
            str(CAUSAL_UPDATES[-1])
        ]["_inventory"]
        terminal_testable = bool(
            terminal_inventory["available_all"]
            and terminal_inventory["powered_all"]
        )
        hard_axis = (
            "hard_not_supported"
            if terminal_testable
            else "hard_underpowered"
        )
    else:
        majority_records = []
        for label, entry in hard_summaries[chosen][str(hard_onset)].items():
            if label == "_inventory" or not entry.get("available"):
                continue
            for split_entry in entry["splits"].values():
                record_id = split_entry["majority_record"]
                require(
                    record_id is not None,
                    "Stable hard onset lacks a powered majority record",
                )
                majority_records.append(records[record_id])
        if majority_records and all(
            record.simultaneous_low > 0.0 for record in majority_records
        ):
            hard_axis = "hard_majority_mediated"
        elif majority_records and all(
            record.simultaneous_high < 0.0 for record in majority_records
        ):
            hard_axis = "hard_partial_below_50pct"
        else:
            hard_axis = "hard_supported_fraction_uncertain"
    result["classification"] = {
        "field_axis": field_axis,
        "causal_axis": causal_axis,
        "hard_qualifier": hard_axis,
    }
    return result


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.verify")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def self_test() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    protocol_record = validate_protocol(config)
    runner_record = validate_runner_static(config)
    verifier_call_surface = validate_verifier_call_surface()
    source_record = validate_frozen_source_hashes(config)
    metamorphic = metamorphic_update_index_regression()
    drift_cancellation = generic_drift_cancellation_regression()
    conjunction_below = conjunction_below_regression()
    synthetic = stable_onset(
        (0, 1, 2, 3),
        {0: False, 1: True, 2: True, 3: True},
        {0: True, 1: False, 2: False, 3: False},
    )
    require(
        synthetic["stable_onset"] == 1
        and synthetic["first_confirmed"] == 1
        and synthetic["onset_interval"] == [0, 1],
        "Independent synthetic onset regression failed",
    )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "verifier_self_test": True,
        "passed": True,
        "protocol": protocol_record,
        "runner": runner_record,
        "verifier_call_surface": verifier_call_surface,
        "frozen_sources_checked": len(source_record),
        "metamorphic_update_index_regression": metamorphic,
        "generic_drift_cancellation_regression": drift_cancellation,
        "conjunction_below_regression": conjunction_below,
        "synthetic_onset": synthetic,
        "production_runner_imported": False,
        "model_loaded": False,
        "scientific_outcomes_required": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def verify() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    protocol_record = validate_protocol(config)
    runner_record = validate_runner_static(config)
    verifier_call_surface = validate_verifier_call_surface()
    source_record = validate_frozen_source_hashes(config)
    metamorphic = metamorphic_update_index_regression()
    drift_cancellation = generic_drift_cancellation_regression()
    conjunction_below = conjunction_below_regression()
    require(OUT_JSON.is_file() and OUT_MD.is_file(), "Completed production analysis is required")
    preflight = validate_preflight(config)
    config_hash = file_sha256(CONFIG_PATH)
    runner_hash = file_sha256(RUNNER_PATH)
    endpoint_manifest, endpoints = validate_endpoints(
        config, config_hash, runner_hash
    )
    native_manifest, native, native_completions = validate_native(
        config, config_hash, runner_hash
    )
    native_lock_artifact = validate_native_lock(
        config,
        config_hash,
        runner_hash,
        native,
        native_completions,
    )
    (
        causal_manifest,
        causal_cells,
        replay_usable,
        causal_completions,
        live_u0_members,
    ) = validate_causal(
        config,
        config_hash,
        runner_hash,
        endpoints,
        native,
        native_completions,
    )
    causal_lock_artifact = validate_causal_lock(
        config,
        config_hash,
        runner_hash,
        native,
        native_completions,
        causal_completions,
        live_u0_members,
    )
    aggregate = load_json(OUT_JSON)
    require(
        aggregate["experiment_id"] == EXPERIMENT_ID
        and aggregate["protocol_sha256"] == config_hash
        and aggregate["script_sha256"] == runner_hash,
        "Aggregate implementation identity mismatch",
    )
    require(
        aggregate["git_head"] == preflight["git_head"],
        "Aggregate was analyzed from a different implementation commit",
    )
    expected_inventory = {
        "endpoint": {
            f"s{seed}:{trait}": {
                "complete": True,
                "attempt": load_json(
                    WORK
                    / "endpoint_factors"
                    / f"seed_{seed}"
                    / trait
                    / "canonical.json"
                )["attempt"],
            }
            for seed in SEEDS
            for trait in TRAITS
        },
        "endpoint_lock": {
            "complete": True,
            "sha256": endpoint_manifest["endpoint_lock"]["sha256"],
            "cells": 4,
        },
        "native": {
            f"s{seed}:{trait}": {
                "complete": True,
                "attempt": load_json(
                    WORK
                    / "native_trajectories"
                    / f"seed_{seed}"
                    / trait
                    / "canonical.json"
                )["attempt"],
                "readouts": 25,
            }
            for seed in SEEDS
            for trait in TRAITS
        },
        "native_lock": {
            "complete": True,
            "sha256": native_lock_artifact["sha256"],
            "cells": 4,
            "u0_all_pairs_pass": True,
        },
        "causal": {
            f"s{seed}:{trait}": {
                "complete": True,
                "attempt": load_json(
                    WORK
                    / "causal_trajectories"
                    / f"seed_{seed}"
                    / trait
                    / "canonical.json"
                )["attempt"],
                "cells": causal_manifest[f"s{seed}:{trait}"]["cells"],
                "evaluated": causal_manifest[f"s{seed}:{trait}"]["evaluated"],
                "not_applicable": causal_manifest[f"s{seed}:{trait}"][
                    "not_applicable"
                ],
            }
            for seed in SEEDS
            for trait in TRAITS
        },
        "causal_lock": {
            "complete": True,
            "sha256": causal_lock_artifact["sha256"],
            "cells": 4,
            "u0_all_pairs_pass": True,
        },
        "expected_endpoint_cells": 4,
        "expected_native_readouts": 100,
        "expected_causal_cells": 1080,
        "complete": True,
    }
    require(aggregate["inventory"] == expected_inventory, "Aggregate inventory mismatch")
    require(
        aggregate["status"]["artifact_integrity_valid"] is True
        and aggregate["status"]["analysis_implementation_valid"] is True
        and aggregate["status"]["primary_classification_valid"] is None
        and aggregate["status"]["overall_pass"] is False,
        "Production aggregate did not preserve verifier ownership",
    )
    native_summaries, records, native_gate_records = recompute_native(
        config, native
    )
    compare_tree(
        aggregate["native_summaries"],
        native_summaries,
        "aggregate.native_summaries",
    )
    indices = bootstrap_indices(config)
    causal_summaries, causal_gate_records = recompute_causal_analysis(
        config,
        causal_cells,
        replay_usable,
        records,
        indices,
    )
    hard_summaries, hard_gate_records = recompute_hard_analysis(
        config,
        causal_cells,
        replay_usable,
        records,
        indices,
    )
    compare_tree(
        aggregate["causal_summaries"],
        causal_summaries,
        "aggregate.causal_summaries",
    )
    compare_tree(
        aggregate["hard_summaries"],
        hard_summaries,
        "aggregate.hard_summaries",
    )
    bootstrap_record = finalize_and_validate_bootstrap(
        config, aggregate, records
    )
    independent_onsets = recompute_onsets_and_classification(
        records,
        native_gate_records,
        causal_gate_records,
        causal_summaries,
        hard_summaries,
        hard_gate_records,
    )
    compare_tree(aggregate["onsets"], independent_onsets, "aggregate.onsets")
    classification = independent_onsets["classification"]
    evidence_manifest = {
        "endpoint": endpoint_manifest,
        "native": native_manifest,
        "native_lock": native_lock_artifact,
        "causal": causal_manifest,
        "causal_lock": causal_lock_artifact,
    }
    status = {
        "artifact_integrity_valid": True,
        "analysis_implementation_valid": True,
        "primary_classification_valid": True,
        "overall_pass": True,
    }
    result = {
        "experiment_id": EXPERIMENT_ID,
        "verified_utc": utc_now(),
        "verifier_sha256": file_sha256(SCRIPT_PATH),
        "protocol": artifact(CONFIG_PATH),
        "production_runner": artifact(RUNNER_PATH),
        "production_aggregate": artifact(OUT_JSON),
        "production_markdown": artifact(OUT_MD),
        "preflight": preflight,
        "static_protocol_validation": protocol_record,
        "static_runner_validation": runner_record,
        "verifier_call_surface": verifier_call_surface,
        "frozen_source_manifest": source_record,
        "metamorphic_update_index_regression": metamorphic,
        "generic_drift_cancellation_regression": drift_cancellation,
        "conjunction_below_regression": conjunction_below,
        "artifact_manifest": evidence_manifest,
        "artifact_manifest_sha256": compact_sha256(evidence_manifest),
        "native_recomputation": {
            "summaries_exact_with_tolerance": True,
            "onsets": independent_onsets["native"],
        },
        "causal_recomputation": {
            "summaries_exact_with_tolerance": True,
            "hard_summaries_exact_with_tolerance": True,
            "all_scalar_bootstrap_records_recomputed": True,
        },
        "bootstrap_recomputation": bootstrap_record,
        "onsets": independent_onsets,
        "recomputed_hard_summaries_match": True,
        "classification": classification,
        "production_status_before_verifier": aggregate["status"],
        "status": status,
        "overall_pass_rule_applied": (
            "artifact_integrity_valid and analysis_implementation_valid and "
            "primary_classification_valid"
        ),
        "production_runner_imported": False,
        "model_loaded": False,
        "tensor_artifacts_loaded_on_cpu_only": True,
    }
    require(
        result["status"]["overall_pass"]
        == all(
            result["status"][field]
            for field in (
                "artifact_integrity_valid",
                "analysis_implementation_valid",
                "primary_classification_valid",
            )
        ),
        "Overall-pass rule was not applied exactly",
    )
    atomic_json(OUT_VERIFY, result)
    print("TEACHER TRAIT/FINGERPRINT ONTOGENY INDEPENDENT VERIFICATION PASSED", flush=True)
    print(json.dumps({"classification": classification, "status": status}, indent=2, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent verifier for teacher trait/fingerprint ontogeny v1"
    )
    parser.add_argument("command", choices=("self-test", "verify"))
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    else:
        verify()


if __name__ == "__main__":
    main()
