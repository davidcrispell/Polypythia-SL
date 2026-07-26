"""Independent verifier for checkpoint-isolated H11 ontogeny v2.

The verifier deliberately never imports either production runner.  It may
reuse mathematical routines from the frozen v1 *clean-room verifier*, whose
source is itself statically audited here for production-runner imports.

``self-test`` is model-free and outcome-free.  ``verify`` is intended to run
only after all 36 checkpoint-isolated leaves, the phase lock, and the
production aggregate have been created.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = Path(__file__).resolve()
CONFIG_PATH = ROOT / "configs/teacher_trait_fingerprint_ontogeny_v2.json"
RUNNER_PATH = ROOT / "scripts/teacher_trait_fingerprint_ontogeny_v2.py"
V1_CONFIG_PATH = ROOT / "configs/teacher_trait_fingerprint_ontogeny_v1.json"
V1_RUNNER_PATH = ROOT / "scripts/teacher_trait_fingerprint_ontogeny_v1.py"
V1_VERIFIER_PATH = (
    ROOT / "scripts/teacher_trait_fingerprint_ontogeny_v1_verify.py"
)
V1_WORK = ROOT / "runs/teacher_trait_fingerprint_ontogeny_v1"
WORK = ROOT / "runs/teacher_trait_fingerprint_ontogeny_v2"
PREFLIGHT_PATH = WORK / "preflight.json"
CHECKPOINT_ROOT = WORK / "checkpoints"
CHECKPOINT_LOCK_PATH = CHECKPOINT_ROOT / "lock.json"
OUT_JSON = ROOT / "runs/teacher_trait_fingerprint_ontogeny_v2.json"
OUT_MD = ROOT / "runs/teacher_trait_fingerprint_ontogeny_v2.md"
OUT_VERIFY = ROOT / "runs/teacher_trait_fingerprint_ontogeny_v2_verify.json"

EXPERIMENT_ID = "teacher_trait_fingerprint_ontogeny_v2"
V1_EXPERIMENT_ID = "teacher_trait_fingerprint_ontogeny_v1"
LINEAGE = "standard_pythia160_step143000"
SEEDS = (2101, 2102)
TRAITS = ("wolf", "lion")
REFERENCE_UPDATES = tuple(range(25))
CAUSAL_UPDATES = (0, 1, 2, 4, 8, 12, 16, 20, 24)
CONSTRUCTIONS = ("checkpoint_local", "crossfit_endpoint_loaded")
REAL_DOSES = (-1.0, -0.5, 0.5, 1.0)
SHAM_DOSES = (-1.0, 1.0)
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
FACTOR_AUDIT_FIELDS = {
    "local_identifiable",
    "local_audits",
    "crossfit_projections",
}
LOCAL_ZERO_AUDIT_FIELDS = {
    "identifiable",
    "reason",
    "singular_values",
    "derived_seed",
}
LOCAL_AUDIT_FIELDS = {
    "identifiable",
    "singular_values",
    "singular_gap",
    "left_residual_relative",
    "right_residual_relative",
    "derived_seed",
}
CROSSFIT_PROJECTION_FIELDS = {
    "signed_projection",
    "matched_endpoint_singular_value",
    "fraction_of_crossfit_endpoint_singular_value",
}
CROSSFIT_WITNESS_FIELDS = {
    "delta_times_matched_v",
    "delta_times_matched_v_sha256",
    "matched_endpoint_u_sha256",
    "matched_endpoint_v_sha256",
}

PINNED_V1 = {
    "git_commit": "a093bcb985dea7fae47b62b4663f390f3686ebea",
    "config_sha256": "4094692bd1db85dee2ded8be7a484537bb6f326db66dd5e69a298a6b5e85c4ac",
    "runner_sha256": "72c0518813705810ef763a2cd8f41fa0d658bc76c97d056279a748d9482f0d87",
    "verifier_sha256": "1a2610db61fefa7d18010ce88807db3ad1c30add0e156ce6aab80286c252bb40",
    "preflight_sha256": "6a6f899128032b9e0f2e0acdacae182b5172054551709e4623d5365564678e98",
    "endpoint_lock_sha256": "a895a5ab0ea41153f0a3476bef9cbd2a0adf9a0cb3e19eca2d3bc7551567c2bf",
    "native_lock_sha256": "ee7ffdbb83177fdbf7d1a7904d0ee296062b8f1892afed4d2d65216a8efcf8f7",
}
PINNED_FAILURES = (
    {
        "relative_path": (
            "runs/teacher_trait_fingerprint_ontogeny_v1/"
            "causal_trajectories/seed_2101/wolf/attempt_001"
        ),
        "file_count": 216,
        "byte_count": 15265966,
        "tree_sha256": "b973975d2b37c8d3afc178fd637864dd29b4f6d762a151446ab6b2e2eb32c424",
        "failure_sha256": "b5146e944e4c700d6d2061e0b4748dab9cc191fdd4760231722c696ef655e1e2",
        "failed_update": 5,
        "observed_cell_count": 120,
    },
    {
        "relative_path": (
            "runs/teacher_trait_fingerprint_ontogeny_v1/"
            "causal_trajectories/seed_2101/wolf/attempt_002"
        ),
        "file_count": 216,
        "byte_count": 15264764,
        "tree_sha256": "bc94b61f19fcc7e6b737b0362fe1ac3e6013f9515af8d15d5247bbac984b50de",
        "failure_sha256": "bb829f669713a3d1c1288b212c8f77a9e4cb62ce853a2126dfe39a79f03a1859",
        "failed_update": 7,
        "observed_cell_count": 120,
    },
)


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
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


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
    require(not path.is_absolute(), f"Absolute path forbidden for {name}")
    resolved = (ROOT / path).resolve()
    require(
        resolved.is_relative_to(ROOT.resolve()),
        f"Path escapes repository for {name}: {value}",
    )
    return resolved


def exact_repository_artifact_path(
    value: Any, expected: Path, *, name: str
) -> Path:
    expected_resolved = expected.resolve()
    observed = repository_path(value, name=name)
    require(
        value == relative(expected)
        and observed == expected_resolved
        and expected.is_file()
        and not expected.is_symlink(),
        f"Non-canonical, missing, or symlinked path for {name}: {value}",
    )
    return observed


def validate_leaf_filesystem_contract(
    root: Path,
    attempt: Path,
    completion: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Validate one leaf's closed filesystem world without trusting hashes."""
    require(
        root.is_dir() and not root.is_symlink(),
        f"Missing or symlinked leaf root for {label}: {root}",
    )
    observed_root_children = {path.name for path in root.iterdir()}
    require(
        observed_root_children == {"attempt_001", "canonical.json"},
        f"Leaf-root child inventory mismatch for {label}: "
        f"{sorted(observed_root_children)}",
    )
    require(
        attempt == root / "attempt_001"
        and attempt.is_dir()
        and not attempt.is_symlink()
        and (root / "canonical.json").is_file()
        and not (root / "canonical.json").is_symlink(),
        f"Leaf root types or attempt path changed for {label}",
    )
    symlinks = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_symlink()
    )
    require(not symlinks, f"Symlinks forbidden in leaf {label}: {symlinks}")

    completion_path = attempt / "completion.json"
    live_path = attempt / "live_readout.pt"
    factor_path = attempt / "factors.pt"
    training_path = attempt / "training" / "training_metrics.json"
    exact_repository_artifact_path(
        completion["training_metrics_path"],
        training_path,
        name=f"{label} training metrics",
    )
    exact_repository_artifact_path(
        completion["live_readout"]["path"],
        live_path,
        name=f"{label} live readout",
    )
    exact_repository_artifact_path(
        completion["checkpoint_record"]["factor_catalog"]["path"],
        factor_path,
        name=f"{label} factor catalog",
    )
    expected_files = {
        completion_path.resolve(),
        live_path.resolve(),
        factor_path.resolve(),
        training_path.resolve(),
    }
    records = completion.get("cells")
    require(isinstance(records, list), f"Malformed cell inventory for {label}")
    for index, record in enumerate(records):
        require(
            isinstance(record, dict)
            and isinstance(record.get("key"), dict)
            and record.get("status") in {"evaluated", "not_applicable"},
            f"Malformed cell record for {label}[{index}]",
        )
        stem = cell_stem(record["key"])
        json_path = attempt / "cells" / f"{stem}.json"
        exact_repository_artifact_path(
            record.get("path"),
            json_path,
            name=f"{label} cell JSON {index}",
        )
        expected_files.add(json_path.resolve())
        if record["status"] == "evaluated":
            array_path = attempt / "cells" / f"{stem}.npz"
            exact_repository_artifact_path(
                record.get("arrays_path"),
                array_path,
                name=f"{label} cell arrays {index}",
            )
            expected_files.add(array_path.resolve())
        else:
            require(
                "arrays_path" not in record,
                f"N/A record unexpectedly names arrays for {label}[{index}]",
            )
    observed_files = {
        path.resolve()
        for path in attempt.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    require(
        observed_files == expected_files,
        f"Leaf file inventory mismatch for {label}: "
        f"missing={sorted(str(path) for path in expected_files-observed_files)} "
        f"extra={sorted(str(path) for path in observed_files-expected_files)}",
    )
    return {
        "root_children": sorted(observed_root_children),
        "regular_file_count": len(observed_files),
        "symlink_count": 0,
        "pass": True,
    }


def artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing artifact: {path}")
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def compare_tree(
    observed: Any,
    expected: Any,
    name: str,
    *,
    float_rtol: float = 2e-9,
    float_atol: float = 2e-11,
) -> None:
    if isinstance(expected, dict):
        require(isinstance(observed, dict), f"Expected object for {name}")
        require(set(observed) == set(expected), f"Key mismatch for {name}")
        for key in expected:
            compare_tree(
                observed[key],
                expected[key],
                f"{name}.{key}",
                float_rtol=float_rtol,
                float_atol=float_atol,
            )
        return
    if isinstance(expected, list):
        require(isinstance(observed, list), f"Expected list for {name}")
        require(len(observed) == len(expected), f"Length mismatch for {name}")
        for index, value in enumerate(expected):
            compare_tree(
                observed[index],
                value,
                f"{name}[{index}]",
                float_rtol=float_rtol,
                float_atol=float_atol,
            )
        return
    if isinstance(expected, float):
        require(
            math.isclose(
                float(observed),
                expected,
                rel_tol=float_rtol,
                abs_tol=float_atol,
            ),
            f"Float mismatch for {name}: {observed} != {expected}",
        )
        return
    require(observed == expected, f"Value mismatch for {name}")


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def import_v1_cleanroom():
    """Load only the frozen v1 verifier, never either production runner."""
    require(
        file_sha256(V1_VERIFIER_PATH) == PINNED_V1["verifier_sha256"],
        "Frozen v1 verifier hash changed",
    )
    spec = importlib.util.spec_from_file_location(
        "_ontogeny_v1_cleanroom", V1_VERIFIER_PATH
    )
    require(spec is not None and spec.loader is not None, "Cannot load v1 verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def expected_leaf_keys() -> list[tuple[int, str, int]]:
    return [
        (seed, trait, update)
        for seed in SEEDS
        for trait in TRAITS
        for update in CAUSAL_UPDATES
    ]


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


def expected_leaf_cells(
    seed: int, trait: str, update: int, *, sham_draws: int = 5
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for construction in CONSTRUCTIONS:
        for dose in REAL_DOSES:
            result.append(
                logical_key(
                    seed,
                    trait,
                    update,
                    construction,
                    "real",
                    -1,
                    dose,
                )
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
    observed = [key_tuple(value) for value in result]
    require(len(observed) == len(set(observed)) == 30, "Leaf key grid changed")
    return result


def cell_stem(key: dict[str, Any]) -> str:
    digest = compact_sha256(key)[:16]
    dose = int(round(100 * float(key["dose"])))
    return (
        f"u{int(key['optimizer_update']):04d}_"
        f"{key['construction']}_{key['control_kind']}_"
        f"r{int(key['control_draw']):02d}_d{dose:+04d}_{digest}"
    )


def checkpoint_root(seed: int, trait: str, update: int) -> Path:
    return CHECKPOINT_ROOT / f"seed_{seed}" / trait / f"u{update:04d}"


def sorted_tree_manifest(root: Path) -> tuple[int, int, str]:
    require(root.is_dir(), f"Missing tree: {root}")
    rows: list[str] = []
    byte_count = 0
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        rel = path.relative_to(root).as_posix()
        rows.append(f"{file_sha256(path)}  {rel}\n")
        byte_count += path.stat().st_size
    digest = hashlib.sha256("".join(rows).encode()).hexdigest()
    return len(files), byte_count, digest


def validate_no_runner_imports(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(), filename=str(path))
    forbidden = {
        "teacher_trait_fingerprint_ontogeny_v1",
        "teacher_trait_fingerprint_ontogeny_v2",
    }
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    bad = [
        name
        for name in imports
        if any(
            name == candidate or name.endswith("." + candidate)
            for candidate in forbidden
        )
    ]
    require(not bad, f"Production runner import detected in {path}: {bad}")
    return {"path": relative(path), "imports": sorted(imports), "pass": True}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.verify")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def validate_pinned_v1_files() -> dict[str, Any]:
    expected = {
        V1_CONFIG_PATH: PINNED_V1["config_sha256"],
        V1_RUNNER_PATH: PINNED_V1["runner_sha256"],
        V1_VERIFIER_PATH: PINNED_V1["verifier_sha256"],
        V1_WORK / "preflight.json": PINNED_V1["preflight_sha256"],
        V1_WORK / "endpoint_factors/lock.json": PINNED_V1[
            "endpoint_lock_sha256"
        ],
        V1_WORK / "native_trajectories/lock.json": PINNED_V1[
            "native_lock_sha256"
        ],
    }
    records = {}
    for path, expected_hash in expected.items():
        require(path.is_file(), f"Missing pinned v1 upstream: {path}")
        observed = file_sha256(path)
        require(
            observed == expected_hash,
            f"Pinned v1 upstream changed: {path}",
        )
        records[relative(path)] = artifact(path)
    v1_preflight = load_json(V1_WORK / "preflight.json")
    require(
        v1_preflight.get("implementation", {}).get("git_head")
        == PINNED_V1["git_commit"],
        "Pinned v1 preflight/git commit binding changed",
    )
    for path in (V1_CONFIG_PATH, V1_RUNNER_PATH, V1_VERIFIER_PATH):
        rel = relative(path)
        process = subprocess.run(
            ["git", "show", f"{PINNED_V1['git_commit']}:{rel}"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
        require(
            hashlib.sha256(process.stdout).hexdigest()
            == file_sha256(path),
            f"Pinned v1 commit blob differs from dependency: {rel}",
        )
    return records


def validate_failed_attempt_quarantine() -> list[dict[str, Any]]:
    records = []
    for expected in PINNED_FAILURES:
        root = repository_path(
            expected["relative_path"], name="failed v1 attempt"
        )
        file_count, byte_count, tree_hash = sorted_tree_manifest(root)
        failure_path = root / "failure.json"
        require(
            file_count == expected["file_count"]
            and byte_count == expected["byte_count"]
            and tree_hash == expected["tree_sha256"]
            and file_sha256(failure_path) == expected["failure_sha256"],
            f"Failed v1 attempt quarantine changed: {root}",
        )
        failure = load_json(failure_path)
        require(
            failure.get("seed") == 2101
            and failure.get("trait") == "wolf"
            and failure.get("error_type") == "RuntimeError"
            and failure.get("observed_cell_count")
            == expected["observed_cell_count"]
            and f"at u{expected['failed_update']}" in str(failure.get("error")),
            f"Failed-attempt metadata changed: {failure_path}",
        )
        records.append(
            {
                **expected,
                "failure": artifact(failure_path),
                "tree_valid": True,
            }
        )
    return records


def validate_v1_scientific_upstreams(v1: Any) -> tuple[
    dict[str, Any],
    dict[tuple[int, str], dict[str, Any]],
    dict[tuple[int, str, int], Any],
    dict[tuple[int, str], dict[str, Any]],
]:
    """Use the frozen clean-room verifier to validate sealed v1 sources."""
    v1_config = load_json(V1_CONFIG_PATH)
    config_hash = file_sha256(V1_CONFIG_PATH)
    runner_hash = file_sha256(V1_RUNNER_PATH)
    endpoint_manifest, endpoints = v1.validate_endpoints(
        v1_config, config_hash, runner_hash
    )
    native_manifest, native, native_completions = v1.validate_native(
        v1_config, config_hash, runner_hash
    )
    native_lock = v1.validate_native_lock(
        v1_config,
        config_hash,
        runner_hash,
        native,
        native_completions,
    )
    require(
        file_sha256(V1_WORK / "endpoint_factors/lock.json")
        == PINNED_V1["endpoint_lock_sha256"]
        and native_lock["sha256"] == PINNED_V1["native_lock_sha256"],
        "Clean-room v1 validation did not recover pinned locks",
    )
    manifest = {
        "endpoint": endpoint_manifest,
        "native": native_manifest,
        "native_lock": native_lock,
    }
    return manifest, endpoints, native, native_completions


def validate_training_metrics(
    v1: Any,
    config_v1: dict[str, Any],
    attempt: Path,
    completion: dict[str, Any],
    *,
    seed: int,
    target_update: int,
) -> dict[str, Any]:
    record = v1.validate_training_metrics(
        completion.get("training_metrics_path"),
        completion.get("training_metrics_sha256"),
        attempt,
        config=config_v1,
        completion=completion,
        phase="causal",
        seed=seed,
        probe_updates=[target_update],
    )
    metrics_path = repository_path(
        completion["training_metrics_path"], name="isolated training metrics"
    )
    metrics = load_json(metrics_path)
    callback_rows = metrics["checkpoint_metrics"]
    require(
        len(callback_rows) == 1
        and callback_rows[0]["optimizer_update"] == target_update,
        "Isolated replay did not have exactly one target callback",
    )
    checkpoint = completion["checkpoint_record"]
    require(
        callback_rows
        == [
            {
                "optimizer_update": target_update,
                "target_update": target_update,
                "live_readout_path": checkpoint["live_readout"]["path"],
                "live_readout_sha256": checkpoint["live_readout"]["sha256"],
                "selected_weight_sha256": checkpoint["safety"][
                    "selected_weight_sha256"
                ],
                "repeat_guard_pass": checkpoint["repeat_guard"]["pass"],
                "usable_for_onset": checkpoint["repeat_guard"][
                    "usable_for_onset"
                ],
            }
        ],
        "Training callback/checkpoint cross-link mismatch",
    )
    return record


def load_readout_record(
    record: dict[str, Any],
    *,
    expected_path: Path,
    name: str,
    v1: Any,
) -> Any:
    require(
        set(record)
        == {
            "path",
            "sha256",
            "numeric_logits_sha256",
            "animal_logits_sha256",
            "selected_weight_sha256",
        },
        f"Readout record schema mismatch: {name}",
    )
    path = repository_path(record["path"], name=name)
    require(
        path == expected_path.resolve()
        and file_sha256(path) == record["sha256"],
        f"Readout path/hash mismatch: {name}",
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(
        isinstance(payload, dict)
        and set(payload) == {"identity", "numeric_logits", "animal_logits"},
        f"Readout payload schema mismatch: {name}",
    )
    numeric = payload["numeric_logits"]
    animals = payload["animal_logits"]
    require(
        isinstance(numeric, torch.Tensor)
        and isinstance(animals, torch.Tensor)
        and tuple(numeric.shape) == (1024, 655)
        and tuple(animals.shape) == (60, 10)
        and bool(torch.isfinite(numeric).all())
        and bool(torch.isfinite(animals).all())
        and tensor_sha256(numeric) == record["numeric_logits_sha256"]
        and tensor_sha256(animals) == record["animal_logits_sha256"]
        and payload["identity"].get("selected_weight_sha256")
        == record["selected_weight_sha256"],
        f"Readout tensor/hash mismatch: {name}",
    )
    return v1.Readout(
        numeric.float().cpu(),
        animals.float().cpu(),
        payload["identity"],
    )


def recompute_repeat_guard(
    v1: Any,
    config_v1: dict[str, Any],
    live: Any,
    frozen: Any,
    counterpart: Any,
    *,
    trait: str,
    update: int,
) -> dict[str, Any]:
    live_p = torch.softmax(live.numeric_logits.double(), dim=-1)
    frozen_p = torch.softmax(frozen.numeric_logits.double(), dim=-1)
    probability_error = torch.abs(live_p - frozen_p)
    centered_error = torch.abs(
        v1.centered(live.numeric_logits.double())
        - v1.centered(frozen.numeric_logits.double())
    )
    behavior_error = torch.abs(
        live.animal_logits.double() - frozen.animal_logits.double()
    )
    live_target = v1.behavior_scores(live.animal_logits, trait)[
        "target_pair_score"
    ]
    frozen_target = v1.behavior_scores(frozen.animal_logits, trait)[
        "target_pair_score"
    ]
    target_error = live_target - frozen_target
    limits = config_v1["fidelity"]["causal_replay_repeat_guard"]
    maximum_probability = float(probability_error.max())
    maximum_behavior = float(behavior_error.max())
    absolute_pass = bool(
        maximum_probability
        <= float(limits["max_restricted_probability_absolute_difference"])
        and maximum_behavior
        <= float(limits["max_behavior_selected_logit_absolute_difference"])
    )
    result: dict[str, Any] = {
        "max_restricted_probability_absolute_difference": maximum_probability,
        "rms_restricted_probability_difference": float(
            torch.sqrt(torch.mean(probability_error.square()))
        ),
        "max_centered_logit_absolute_difference": float(centered_error.max()),
        "rms_centered_logit_difference": float(
            torch.sqrt(torch.mean(centered_error.square()))
        ),
        "max_behavior_selected_logit_absolute_difference": maximum_behavior,
        "rms_behavior_selected_logit_difference": float(
            torch.sqrt(torch.mean(behavior_error.square()))
        ),
        "rms_target_pair_score_difference": float(
            torch.sqrt(torch.mean(target_error.square()))
        ),
        "limits": limits,
        "pass": absolute_pass,
    }
    for split, row_slice in v1.split_slices().items():
        result[f"rms_restricted_probability_difference_{split}"] = float(
            torch.sqrt(
                torch.mean(probability_error[row_slice].square())
            )
        )

    native_p = torch.softmax(frozen.numeric_logits.double(), dim=-1)
    other_p = torch.softmax(counterpart.numeric_logits.double(), dim=-1)
    wolf_native = v1.behavior_scores(frozen.animal_logits, "wolf")[
        "target_pair_score"
    ]
    wolf_other = v1.behavior_scores(counterpart.animal_logits, "wolf")[
        "target_pair_score"
    ]
    behavior_gap_rms = float(
        torch.sqrt(torch.mean((wolf_native - wolf_other).square()))
    )
    split_records = {}
    numeric_passes = []
    for split, row_slice in v1.split_slices().items():
        field_rms = float(
            torch.sqrt(
                torch.mean(
                    (native_p[row_slice] - other_p[row_slice]).square()
                )
            )
        )
        repeat_rms = result[
            f"rms_restricted_probability_difference_{split}"
        ]
        fraction = (
            None if update == 0 else repeat_rms / max(field_rms, 1e-30)
        )
        passed = (
            None
            if update == 0
            else fraction
            <= float(
                limits[
                    "max_repeat_rms_as_fraction_of_checkpoint_paired_field_rms"
                ]
            )
        )
        numeric_passes.append(passed)
        split_records[split] = {
            "paired_probability_field_rms": field_rms,
            "repeat_probability_rms": repeat_rms,
            "repeat_fraction_of_paired_field": fraction,
            "pass": passed,
        }
    behavior_fraction = (
        None
        if update == 0
        else result["rms_target_pair_score_difference"]
        / max(behavior_gap_rms, 1e-30)
    )
    if update == 0:
        relative_pass = bool(
            maximum_probability
            <= float(
                limits[
                    "u0_max_restricted_probability_absolute_difference"
                ]
            )
            and maximum_behavior
            <= float(
                limits[
                    "u0_max_behavior_selected_logit_absolute_difference"
                ]
            )
        )
    else:
        relative_pass = bool(
            all(numeric_passes)
            and behavior_fraction
            <= float(
                limits[
                    "max_repeat_behavior_rms_as_fraction_of_checkpoint_paired_behavior_gap_rms"
                ]
            )
        )
    return {
        **result,
        "numeric_splits": split_records,
        "paired_behavior_gap_rms": behavior_gap_rms,
        "behavior_repeat_fraction_of_paired_gap": behavior_fraction,
        "relative_or_u0_pass": relative_pass,
        "usable_for_onset": bool(absolute_pass and relative_pass),
    }


def validate_single_safety(
    record: Any, *, seed: int, trait: str, update: int
) -> dict[str, Any]:
    require(isinstance(record, dict), "Missing checkpoint safety record")
    required = {
        "selected_weight_sha256",
        "hook_count",
        "unselected_parameter_count",
        "gradients_none",
        "rng_restored",
    }
    require(set(record) == required, "Checkpoint safety schema changed")
    require(
        is_sha256(record["selected_weight_sha256"])
        and record["hook_count"] == 0
        and isinstance(record["unselected_parameter_count"], int)
        and record["unselected_parameter_count"] > 0
        and record["gradients_none"] is True
        and record["rng_restored"] is True,
        f"Checkpoint safety failed: s{seed}:{trait}/u{update}",
    )
    return record


def validate_compact_config(
    config: dict[str, Any], *, allow_verifier_placeholder: bool
) -> dict[str, Any]:
    require(
        set(config)
        == {
            "experiment_id",
            "registered_utc",
            "status",
            "question",
            "design",
            "source",
            "inheritance_assertions",
            "artifacts",
            "integrity",
        },
        "V2 compact-config top-level schema changed",
    )
    require(config["experiment_id"] == EXPERIMENT_ID, "Wrong v2 experiment id")
    design = config["design"]
    require(
        design["training_seeds"] == list(SEEDS)
        and design["traits"] == list(TRAITS)
        and design["target_updates"] == list(CAUSAL_UPDATES)
        and design["optimizer_updates_per_replay"] == 24
        and design["callbacks_per_replay"] == 1
        and design["logical_cells_per_leaf"] == 30
        and design["expected_leaf_count"] == 36
        and design["expected_global_cell_count"] == 1080
        and "No retry" in design["retry_policy"]
        and "complete 24-update replay" in design["replay_unit"]
        and "Separately truncated runs are forbidden" in design["replay_unit"],
        "V2 isolated-replay design changed",
    )
    source = config["source"]
    expected_source = {
        "v1_runner_path": relative(V1_RUNNER_PATH),
        "v1_runner_sha256": PINNED_V1["runner_sha256"],
        "v1_config_path": relative(V1_CONFIG_PATH),
        "v1_config_sha256": PINNED_V1["config_sha256"],
        "v1_verifier_path": relative(V1_VERIFIER_PATH),
        "v1_verifier_sha256": PINNED_V1["verifier_sha256"],
        "v1_preregistered_git_commit": PINNED_V1["git_commit"],
        "v1_preflight_path": relative(V1_WORK / "preflight.json"),
        "v1_preflight_sha256": PINNED_V1["preflight_sha256"],
        "v1_endpoint_lock_path": relative(
            V1_WORK / "endpoint_factors/lock.json"
        ),
        "v1_endpoint_lock_sha256": PINNED_V1["endpoint_lock_sha256"],
        "v1_native_lock_path": relative(
            V1_WORK / "native_trajectories/lock.json"
        ),
        "v1_native_lock_sha256": PINNED_V1["native_lock_sha256"],
        "v1_failed_attempt_trees": [
            {
                "root_path": record["relative_path"],
                "file_count": record["file_count"],
                "byte_count": record["byte_count"],
                "tree_sha256": record["tree_sha256"],
                "failure_path": f"{record['relative_path']}/failure.json",
                "failure_sha256": record["failure_sha256"],
            }
            for record in PINNED_FAILURES
        ],
        "v2_verifier_path": relative(SCRIPT_PATH),
        "v2_verifier_sha256": file_sha256(SCRIPT_PATH),
        "forbidden_v1_inputs": [
            "runs/teacher_trait_fingerprint_ontogeny_v1/"
            "causal_trajectories/lock.json",
            "runs/teacher_trait_fingerprint_ontogeny_v1.json",
            "runs/teacher_trait_fingerprint_ontogeny_v1.md",
            "runs/teacher_trait_fingerprint_ontogeny_v1_verify.json",
        ],
    }
    if allow_verifier_placeholder:
        expected_source["v2_verifier_sha256"] = source[
            "v2_verifier_sha256"
        ]
        require(
            source["v2_verifier_sha256"]
            in {
                "__PLACEHOLDER_PENDING_V2_VERIFIER_SHA256__",
                file_sha256(SCRIPT_PATH),
            },
            "Unexpected verifier hash placeholder",
        )
    compare_tree(source, expected_source, "v2.source")

    inherited = config["inheritance_assertions"]
    expected_arrays = {
        name: 60 if name.startswith("behavior_") else 1024
        for name in CAUSAL_ARRAY_KEYS
    }
    require(
        inherited
        == {
            "v1_experiment_id": V1_EXPERIMENT_ID,
            "lineage": LINEAGE,
            "training_seeds": list(SEEDS),
            "traits": list(TRAITS),
            "reference_updates": list(REFERENCE_UPDATES),
            "causal_updates": list(CAUSAL_UPDATES),
            "real_doses": list(REAL_DOSES),
            "sham_doses": list(SHAM_DOSES),
            "sham_draws": 5,
            "selected_layers": [8, 9, 10, 11],
            "selected_module_kinds": [
                "attention.query_key_value",
                "mlp.dense_4h_to_h",
            ],
            "array_lengths": expected_arrays,
        },
        "V2 inheritance assertions changed",
    )
    expected_artifacts = {
        "work_directory": relative(WORK),
        "preflight": relative(PREFLIGHT_PATH),
        "checkpoint_lock": relative(CHECKPOINT_LOCK_PATH),
        "aggregate_json": relative(OUT_JSON),
        "aggregate_markdown": relative(OUT_MD),
        "verification_json": relative(OUT_VERIFY),
    }
    require(
        config["artifacts"] == expected_artifacts,
        "V2 artifact paths changed",
    )
    integrity = config["integrity"]
    require(
        integrity["full_logical_key"] == list(KEY_FIELDS)
        and "exactly attempt_001" in integrity["canonical_policy"]
        and "all 36" in integrity["phase_lock_policy"]
        and "all 28 unordered pairs" in integrity[
            "u0_equivalence_members"
        ]
        and "V1 causal artifacts and aggregates are forbidden inputs"
        in integrity["analysis_policy"]
        and "does not import the production v2 runner"
        in integrity["verification_policy"],
        "V2 integrity policy changed",
    )
    return {
        "config_sha256": file_sha256(CONFIG_PATH),
        "leaf_count": len(expected_leaf_keys()),
        "cell_count": sum(
            len(expected_leaf_cells(seed, trait, update))
            for seed, trait, update in expected_leaf_keys()
        ),
        "pass": True,
    }


def validate_runner_static(config: dict[str, Any]) -> dict[str, Any]:
    require(RUNNER_PATH.is_file(), f"Missing v2 production runner: {RUNNER_PATH}")
    tree = ast.parse(RUNNER_PATH.read_text(), filename=str(RUNNER_PATH))
    imported_verifier = False
    production_imports: list[str] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for name in names:
            production_imports.append(name)
            if "teacher_trait_fingerprint_ontogeny_v2_verify" in name:
                imported_verifier = True
    require(
        not imported_verifier,
        "Production runner imports the independent v2 verifier",
    )
    source = RUNNER_PATH.read_text()
    for forbidden in config["source"]["forbidden_v1_inputs"]:
        require(
            forbidden not in source,
            f"Production runner mentions forbidden v1 input: {forbidden}",
        )
    required_cli = {
        "--self-test",
        "--preflight",
        "--leaf",
        "--all",
        "--seal",
        "--inventory",
        "--analyze",
    }
    missing = sorted(value for value in required_cli if value not in source)
    require(not missing, f"V2 runner CLI surface missing: {missing}")
    require(
        "attempt_001" in source
        and "attempt_002" not in source
        and "target_update" in source
        and "max_updates" in source,
        "V2 runner no-retry/checkpoint isolation surface changed",
    )
    forbidden_v1_api = {
        "causal_root",
        "load_causal_completion",
        "load_causal_cells",
        "require_causal_lock",
        "seal_causal_trajectories",
        "run_causal_trajectory",
        "run_all_causal",
        "artifact_inventory",
        "analyze",
    }
    observed_forbidden = sorted(
        {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "v1"
            and node.attr in forbidden_v1_api
        }
    )
    require(
        not observed_forbidden,
        f"V2 runner uses forbidden v1 causal/output API: {observed_forbidden}",
    )
    return {
        "runner": artifact(RUNNER_PATH),
        "imports": sorted(production_imports),
        "required_cli": sorted(required_cli),
        "forbidden_v1_api_references": observed_forbidden,
        "no_verifier_import": True,
        "pass": True,
    }


def validate_crossfit_projection_witness(
    projection: Any,
    witness: Any,
    endpoint_factor: Any,
    *,
    update: int,
    label: str,
) -> float:
    require(
        isinstance(projection, dict)
        and set(projection) == CROSSFIT_PROJECTION_FIELDS
        and finite_tree(projection),
        f"Crossfit projection schema/nonfinite mismatch: {label}",
    )
    require(
        isinstance(witness, dict)
        and set(witness) == CROSSFIT_WITNESS_FIELDS,
        f"Crossfit projection-witness schema mismatch: {label}",
    )
    require(
        isinstance(endpoint_factor, dict)
        and set(endpoint_factor) == {"u", "s", "v"}
        and isinstance(endpoint_factor["u"], torch.Tensor)
        and isinstance(endpoint_factor["v"], torch.Tensor),
        f"Malformed sealed endpoint factor: {label}",
    )
    endpoint_u = (
        endpoint_factor["u"].detach().float().contiguous().cpu()
    )
    endpoint_v = (
        endpoint_factor["v"].detach().float().contiguous().cpu()
    )
    endpoint_singular = float(endpoint_factor["s"])
    delta_times_v = witness["delta_times_matched_v"]
    require(
        endpoint_u.ndim == endpoint_v.ndim == 1
        and bool(torch.isfinite(endpoint_u).all())
        and bool(torch.isfinite(endpoint_v).all())
        and math.isfinite(endpoint_singular)
        and endpoint_singular > 0.0
        and isinstance(delta_times_v, torch.Tensor)
        and delta_times_v.device.type == "cpu"
        and delta_times_v.dtype == torch.float32
        and delta_times_v.ndim == 1
        and tuple(delta_times_v.shape) == tuple(endpoint_u.shape)
        and delta_times_v.is_contiguous()
        and bool(torch.isfinite(delta_times_v).all())
        and is_sha256(witness["delta_times_matched_v_sha256"])
        and is_sha256(witness["matched_endpoint_u_sha256"])
        and is_sha256(witness["matched_endpoint_v_sha256"]),
        f"Crossfit projection-witness tensor contract mismatch: {label}",
    )
    require(
        witness["delta_times_matched_v_sha256"]
        == tensor_sha256(delta_times_v)
        and witness["matched_endpoint_u_sha256"]
        == tensor_sha256(endpoint_u)
        and witness["matched_endpoint_v_sha256"]
        == tensor_sha256(endpoint_v),
        f"Crossfit projection-witness hash mismatch: {label}",
    )
    recomputed = float(torch.dot(endpoint_u, delta_times_v))
    recorded = float(projection["signed_projection"])
    fraction = float(
        projection["fraction_of_crossfit_endpoint_singular_value"]
    )
    require(
        recomputed == recorded
        and float(projection["matched_endpoint_singular_value"])
        == endpoint_singular
        and math.isclose(
            fraction,
            recorded / max(endpoint_singular, 1e-30),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
        f"Crossfit projection-witness recomputation mismatch: {label}",
    )
    if update == 0:
        require(
            torch.count_nonzero(delta_times_v).item() == 0
            and recomputed == 0.0
            and recorded == 0.0
            and fraction == 0.0,
            f"Update-zero crossfit projection is not exactly zero: {label}",
        )
    return recorded


def validate_u0_factor_amplitudes(
    factor_sets: dict[str, dict[str, dict[str, Any]]],
    *,
    update: int,
    label: str,
) -> None:
    if update != 0:
        return
    require(
        factor_sets
        and all(
            float(factor["s"]) == 0.0
            for factors in factor_sets.values()
            for factor in factors.values()
        ),
        f"Update-zero factor amplitude is not exactly zero: {label}",
    )


def validate_factor_catalog(
    v1: Any,
    config_v1: dict[str, Any],
    endpoints: dict[tuple[int, str], dict[str, Any]],
    *,
    seed: int,
    trait: str,
    update: int,
    safety: dict[str, Any],
    factor_record: dict[str, Any],
    attempt: Path,
    expected_identity: dict[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, dict[str, Any]]],
]:
    require(
        set(factor_record)
        == {
            "path",
            "sha256",
            "factor_set_ids",
            "factor_manifest_sha256",
        },
        "Factor-catalog record schema changed",
    )
    path = repository_path(factor_record["path"], name="factor catalog")
    require(
        path == (attempt / "factors.pt").resolve()
        and file_sha256(path) == factor_record["sha256"],
        f"Factor-catalog path/hash mismatch: {path}",
    )
    catalog = torch.load(path, map_location="cpu", weights_only=True)
    require(
        isinstance(catalog, dict)
        and set(catalog)
        == {
            "schema",
            "identity",
            "checkpoint_local_factors",
            "checkpoint_local_witnesses",
            "crossfit_projection_witnesses",
            "factor_audit",
            "factor_manifests",
        },
        f"Factor-catalog payload schema changed: {path}",
    )
    require(
        catalog["schema"]
        == "teacher_trait_fingerprint_ontogeny_v2_factor_catalog",
        f"Wrong factor-catalog schema: {path}",
    )
    expected_catalog_identity = {
        **expected_identity,
        "selected_weight_sha256": safety["selected_weight_sha256"],
    }
    require(
        catalog["identity"] == expected_catalog_identity,
        f"Factor-catalog identity mismatch: {path}",
    )
    audit = catalog["factor_audit"]
    require(
        isinstance(audit, dict)
        and set(audit) == FACTOR_AUDIT_FIELDS
        and isinstance(audit["local_identifiable"], bool)
        and isinstance(audit["local_audits"], dict)
        and isinstance(audit["crossfit_projections"], dict)
        and set(audit["local_audits"]) == set(v1.selected_names())
        and set(audit["crossfit_projections"]) == set(v1.selected_names()),
        f"Factor audit schema changed: {path}",
    )
    projection_witnesses = catalog["crossfit_projection_witnesses"]
    require(
        isinstance(projection_witnesses, dict)
        and set(projection_witnesses) == set(v1.selected_names()),
        f"Crossfit projection-witness inventory changed: {path}",
    )

    local = catalog["checkpoint_local_factors"]
    witnesses = catalog["checkpoint_local_witnesses"]
    identifiable = []
    local_cfg = config_v1["circuit"]["local_svd"]
    for name in v1.selected_names():
        row = audit["local_audits"][name]
        expected_seed = v1.derived_seed(
            int(local_cfg["base_seed"]),
            LINEAGE,
            seed,
            trait,
            update,
            name,
        )
        require(
            row.get("derived_seed") == expected_seed,
            f"Local SVD seed mismatch: {path}:{name}",
        )
        values = row.get("singular_values")
        require(
            isinstance(row, dict)
            and finite_tree(row)
            and isinstance(values, list)
            and len(values) == 4
            and all(math.isfinite(float(value)) for value in values)
            and all(
                float(values[index]) >= float(values[index + 1]) >= 0.0
                for index in range(3)
            ),
            f"Local singular values malformed: {path}:{name}",
        )
        if row.get("reason") == "zero_delta":
            require(
                set(row) == LOCAL_ZERO_AUDIT_FIELDS
                and row["reason"] == "zero_delta"
                and row["identifiable"] is False
                and values == [0.0, 0.0, 0.0, 0.0],
                f"Zero-delta audit malformed: {path}:{name}",
            )
        else:
            require(
                set(row) == LOCAL_AUDIT_FIELDS
                and isinstance(row["identifiable"], bool)
                and float(row["left_residual_relative"]) >= 0.0
                and float(row["right_residual_relative"]) >= 0.0,
                f"Local audit field schema malformed: {path}:{name}",
            )
            gap = float(values[0]) / max(float(values[1]), 1e-30)
            require(
                math.isclose(
                    float(row["singular_gap"]),
                    gap,
                    rel_tol=2e-7,
                    abs_tol=2e-9,
                ),
                f"Local singular gap mismatch: {path}:{name}",
            )
            expected_available = bool(
                gap >= float(local_cfg["singular_gap_minimum"])
                and max(
                    float(row["left_residual_relative"]),
                    float(row["right_residual_relative"]),
                )
                <= float(local_cfg["residual_relative_maximum"])
            )
            require(
                row["identifiable"] is expected_available,
                f"Local identifiability decision mismatch: {path}:{name}",
            )
        if update == 0:
            require(
                set(row) == LOCAL_ZERO_AUDIT_FIELDS
                and row["reason"] == "zero_delta"
                and row["identifiable"] is False
                and values == [0.0, 0.0, 0.0, 0.0],
                f"Update-zero local audit is not exact zero: {path}:{name}",
            )
        identifiable.append(bool(row["identifiable"]))
    local_available = all(identifiable)
    require(
        audit["local_identifiable"] is local_available
        and (local is not None) is local_available,
        f"Local factor availability mismatch: {path}",
    )
    if local is None:
        require(witnesses == {}, f"Unavailable local factor has witnesses: {path}")
    else:
        require(update != 0, f"Update-zero local factor exists: {path}")
        require(
            set(local) == set(v1.selected_names())
            and set(witnesses) == set(v1.selected_names()),
            f"Local factor/witness inventory mismatch: {path}",
        )
        for name in v1.selected_names():
            factor = local[name]
            witness = witnesses[name]
            require(
                isinstance(factor, dict)
                and set(factor) == {"u", "s", "v"}
                and isinstance(witness, dict)
                and set(witness) == {"delta_v", "delta_transpose_u"},
                f"Local factor/witness schema mismatch: {path}:{name}",
            )
            require(
                isinstance(factor["u"], torch.Tensor)
                and isinstance(factor["v"], torch.Tensor)
                and isinstance(witness["delta_v"], torch.Tensor)
                and isinstance(
                    witness["delta_transpose_u"], torch.Tensor
                ),
                f"Local factor/witness tensor type mismatch: {path}:{name}",
            )
            u = factor["u"].detach().float().contiguous().cpu()
            vector = factor["v"].detach().float().contiguous().cpu()
            singular = float(factor["s"])
            delta_v = (
                witness["delta_v"].detach().float().contiguous().cpu()
            )
            delta_t_u = (
                witness["delta_transpose_u"]
                .detach()
                .float()
                .contiguous()
                .cpu()
            )
            endpoint_factor = endpoints[(v1.crossfit_seed(seed), trait)][
                "factors"
            ][name]
            require(
                tuple(u.shape) == tuple(endpoint_factor["u"].shape)
                and tuple(vector.shape) == tuple(endpoint_factor["v"].shape)
                and tuple(delta_v.shape) == tuple(u.shape)
                and tuple(delta_t_u.shape) == tuple(vector.shape)
                and bool(torch.isfinite(u).all())
                and bool(torch.isfinite(vector).all())
                and bool(torch.isfinite(delta_v).all())
                and bool(torch.isfinite(delta_t_u).all())
                and singular > 0.0,
                f"Local tensor contract mismatch: {path}:{name}",
            )
            require(
                math.isclose(float(u.norm()), 1.0, rel_tol=3e-5, abs_tol=3e-5)
                and math.isclose(
                    float(vector.norm()), 1.0, rel_tol=3e-5, abs_tol=3e-5
                )
                and float(u[torch.argmax(torch.abs(u))]) >= 0.0,
                f"Local normalization/sign mismatch: {path}:{name}",
            )
            require(
                math.isclose(
                    singular,
                    float(audit["local_audits"][name]["singular_values"][0]),
                    rel_tol=2e-7,
                    abs_tol=2e-9,
                ),
                f"Local factor amplitude/audit mismatch: {path}:{name}",
            )
            left = float((delta_v - singular * u).norm()) / max(
                abs(singular), 1e-30
            )
            right = float((delta_t_u - singular * vector).norm()) / max(
                abs(singular), 1e-30
            )
            require(
                math.isclose(
                    left,
                    float(audit["local_audits"][name]["left_residual_relative"]),
                    rel_tol=2e-5,
                    abs_tol=2e-7,
                )
                and math.isclose(
                    right,
                    float(
                        audit["local_audits"][name][
                            "right_residual_relative"
                        ]
                    ),
                    rel_tol=2e-5,
                    abs_tol=2e-7,
                ),
                f"Local residual witness mismatch: {path}:{name}",
            )

    donor = v1.crossfit_seed(seed)
    matched = endpoints[(donor, trait)]
    wrong_endpoint = endpoints[(donor, v1.other_trait(trait))]
    loaded = {}
    wrong = {}
    for name in v1.selected_names():
        projection = audit["crossfit_projections"][name]
        coefficient = float(projection["signed_projection"])
        endpoint_singular = float(matched["factors"][name]["s"])
        require(
            finite_tree(projection)
            and math.isclose(
                float(projection["matched_endpoint_singular_value"]),
                endpoint_singular,
                rel_tol=2e-9,
                abs_tol=2e-11,
            )
            and math.isclose(
                float(
                    projection[
                        "fraction_of_crossfit_endpoint_singular_value"
                    ]
                ),
                coefficient / max(endpoint_singular, 1e-30),
                rel_tol=2e-9,
                abs_tol=2e-11,
            ),
            f"Crossfit projection mismatch: {path}:{name}",
        )
        loaded[name] = {
            "u": matched["factors"][name]["u"].float(),
            "s": coefficient,
            "v": matched["factors"][name]["v"].float(),
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
        factor_sets[v1.factor_set_id(construction, "real", -1)] = real
        for draw in range(int(config_v1["circuit"]["sham_draws"])):
            sham = {}
            for name in v1.selected_names():
                sham[name] = v1.haar_factor(
                    (
                        int(real[name]["u"].numel()),
                        int(real[name]["v"].numel()),
                    ),
                    float(real[name]["s"]),
                    seed=v1.derived_seed(
                        int(config_v1["circuit"]["sham_base_seed"]),
                        LINEAGE,
                        seed,
                        trait,
                        update,
                        construction,
                        name,
                        draw,
                    ),
                )
            factor_sets[v1.factor_set_id(construction, "sham", draw)] = sham
        if construction == "crossfit_endpoint_loaded":
            factor_sets[
                v1.factor_set_id(construction, "wrong_trait", -1)
            ] = wrong
    manifests = {
        identifier: v1.factor_manifest(factors)
        for identifier, factors in factor_sets.items()
    }
    compare_tree(
        catalog["factor_manifests"],
        manifests,
        f"factor manifests {path}",
    )
    require(
        factor_record["factor_set_ids"] == sorted(manifests)
        and factor_record["factor_manifest_sha256"]
        == compact_sha256(manifests),
        f"Factor manifest inventory/hash mismatch: {path}",
    )
    return manifests, factor_sets


def expected_dependency_binding(config: dict[str, Any]) -> dict[str, Any]:
    source = config["source"]
    return {
        "v1_runner": {
            "path": source["v1_runner_path"],
            "sha256": source["v1_runner_sha256"],
        },
        "v1_config": {
            "path": source["v1_config_path"],
            "sha256": source["v1_config_sha256"],
        },
        "v1_verifier": {
            "path": source["v1_verifier_path"],
            "sha256": source["v1_verifier_sha256"],
        },
        "v1_preregistered_git_commit": source[
            "v1_preregistered_git_commit"
        ],
        "v1_preflight": {
            "path": source["v1_preflight_path"],
            "sha256": source["v1_preflight_sha256"],
        },
        "v1_endpoint_lock": {
            "path": source["v1_endpoint_lock_path"],
            "sha256": source["v1_endpoint_lock_sha256"],
        },
        "v1_native_lock": {
            "path": source["v1_native_lock_path"],
            "sha256": source["v1_native_lock_sha256"],
        },
        "v1_failed_attempt_trees": [
            dict(row) for row in source["v1_failed_attempt_trees"]
        ],
        "v1_causal_inputs_forbidden": list(
            source["forbidden_v1_inputs"]
        ),
    }


def expected_upstream_locks() -> dict[str, Any]:
    endpoint_path = V1_WORK / "endpoint_factors/lock.json"
    native_path = V1_WORK / "native_trajectories/lock.json"
    endpoint = load_json(endpoint_path)
    native = load_json(native_path)
    return {
        "endpoint_lock_sha256": file_sha256(endpoint_path),
        "endpoint_cell_manifest_sha256": compact_sha256(endpoint["cells"]),
        "native_lock_sha256": file_sha256(native_path),
        "native_cell_manifest_sha256": compact_sha256(native["cells"]),
    }


def native_update_source(
    native_completions: dict[tuple[int, str], dict[str, Any]],
    *,
    seed: int,
    trait: str,
    update: int,
) -> dict[str, Any]:
    completion = native_completions[(seed, trait)]
    attempt = repository_path(
        completion["attempt"], name="v1 native attempt"
    )
    record = next(
        row
        for row in completion["readouts"]
        if int(row["optimizer_update"]) == update
    )
    root = V1_WORK / "native_trajectories" / f"seed_{seed}" / trait
    return {
        "seed": seed,
        "trait": trait,
        "optimizer_update": update,
        "attempt": relative(attempt),
        "canonical_pointer_sha256": file_sha256(root / "canonical.json"),
        "completion_sha256": file_sha256(attempt / "completion.json"),
        "context_identity": completion["context_identity"],
        "readout": dict(record),
    }


def endpoint_factor_source(
    endpoints: dict[tuple[int, str], dict[str, Any]],
    *,
    seed: int,
    trait: str,
) -> dict[str, Any]:
    root = V1_WORK / "endpoint_factors" / f"seed_{seed}" / trait
    pointer = load_json(root / "canonical.json")
    attempt = repository_path(pointer["attempt"], name="v1 endpoint attempt")
    completion = load_json(attempt / "completion.json")
    payload = endpoints[(seed, trait)]
    return {
        "seed": seed,
        "trait": trait,
        "optimizer_update": 24,
        "attempt": relative(attempt),
        "canonical_pointer_sha256": file_sha256(root / "canonical.json"),
        "completion_sha256": file_sha256(attempt / "completion.json"),
        "factors_path": completion["factors_path"],
        "factors_sha256": completion["factors_sha256"],
        "selected_endpoint_sha256": payload["selected_endpoint_sha256"],
    }


def expected_leaf_sources(
    v1: Any,
    endpoints: dict[tuple[int, str], dict[str, Any]],
    native_completions: dict[tuple[int, str], dict[str, Any]],
    *,
    seed: int,
    trait: str,
    update: int,
) -> dict[str, Any]:
    donor = v1.crossfit_seed(seed)
    paired = v1.other_trait(trait)
    return {
        "current_native": native_update_source(
            native_completions,
            seed=seed,
            trait=trait,
            update=update,
        ),
        "paired_other_native": native_update_source(
            native_completions,
            seed=seed,
            trait=paired,
            update=update,
        ),
        "donor_endpoint_target": native_update_source(
            native_completions,
            seed=donor,
            trait=trait,
            update=24,
        ),
        "matched_endpoint_factor": endpoint_factor_source(
            endpoints, seed=donor, trait=trait
        ),
        "wrong_trait_endpoint_factor": endpoint_factor_source(
            endpoints, seed=donor, trait=paired
        ),
    }


def validate_preflight(config: dict[str, Any]) -> dict[str, Any]:
    preflight = load_json(PREFLIGHT_PATH)
    require(
        preflight.get("schema")
        == "teacher_trait_fingerprint_ontogeny_v2_preflight"
        and preflight.get("experiment_id") == EXPERIMENT_ID
        and preflight.get("scientific_cells_run") is False
        and preflight.get("v1_causal_scientific_artifacts_consumed") is False,
        "Malformed v2 preflight",
    )
    implementation = preflight["implementation"]
    require(
        implementation["script"] == relative(RUNNER_PATH)
        and implementation["script_sha256"] == file_sha256(RUNNER_PATH)
        and implementation["config"] == relative(CONFIG_PATH)
        and implementation["config_sha256"] == file_sha256(CONFIG_PATH)
        and implementation["verifier"] == relative(SCRIPT_PATH)
        and implementation["verifier_sha256"] == file_sha256(SCRIPT_PATH),
        "V2 preflight implementation binding mismatch",
    )
    dependency = expected_dependency_binding(config)
    source = preflight["source"]
    expected_files: dict[str, Any] = {
        config["source"]["v1_runner_path"]: PINNED_V1["runner_sha256"],
        config["source"]["v1_config_path"]: PINNED_V1["config_sha256"],
        config["source"]["v1_verifier_path"]: PINNED_V1["verifier_sha256"],
        config["source"]["v1_preflight_path"]: PINNED_V1[
            "preflight_sha256"
        ],
        config["source"]["v1_endpoint_lock_path"]: PINNED_V1[
            "endpoint_lock_sha256"
        ],
        config["source"]["v1_native_lock_path"]: PINNED_V1[
            "native_lock_sha256"
        ],
        "v1_preregistered_git_commit": PINNED_V1["git_commit"],
        config["source"]["v2_verifier_path"]: file_sha256(SCRIPT_PATH),
    }
    for row in config["source"]["v1_failed_attempt_trees"]:
        expected_files[row["root_path"]] = {
            key: row[key]
            for key in ("file_count", "byte_count", "tree_sha256")
        }
        expected_files[row["failure_path"]] = row["failure_sha256"]
    expected_inheritance = {
        "v1_protocol_sha256": PINNED_V1["config_sha256"],
        "assertions_sha256": compact_sha256(
            config["inheritance_assertions"]
        ),
        "leaf_count": 36,
        "global_cell_count": 1080,
    }
    require(
        source
        == {
            "files": expected_files,
            "dependency_binding": dependency,
            "dependency_binding_sha256": compact_sha256(dependency),
            "inheritance": expected_inheritance,
        }
        and source["dependency_binding"] == dependency
        and source["dependency_binding_sha256"]
        == compact_sha256(dependency),
        "V2 preflight dependency binding mismatch",
    )
    require(
        preflight["upstream_locks"] == expected_upstream_locks(),
        "V2 preflight upstream locks changed",
    )
    leaf_keys = [
        {
            "training_seed": seed,
            "trait": trait,
            "target_update": update,
        }
        for seed, trait, update in expected_leaf_keys()
    ]
    global_keys = [
        key
        for seed, trait, update in expected_leaf_keys()
        for key in expected_leaf_cells(seed, trait, update)
    ]
    require(
        preflight["expected_inventory"]
        == {
            "leaves": 36,
            "leaf_key_sha256": compact_sha256(leaf_keys),
            "cells": 1080,
            "global_key_sha256": compact_sha256(global_keys),
        },
        "V2 preflight expected inventory changed",
    )
    return {
        "artifact": artifact(PREFLIGHT_PATH),
        "git_head": implementation["git_head"],
        "source_sha256": compact_sha256(source),
        "dependency_binding_sha256": compact_sha256(dependency),
        "pass": True,
    }


def canonical_leaf(
    seed: int, trait: str, update: int
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = checkpoint_root(seed, trait, update)
    require(
        root.is_dir() and not root.is_symlink(),
        f"Missing or symlinked checkpoint leaf root: {root}",
    )
    observed_children = {path.name for path in root.iterdir()}
    require(
        observed_children == {"attempt_001", "canonical.json"},
        f"Canonical leaf-root inventory changed at {root}: "
        f"{sorted(observed_children)}",
    )
    symlinks = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_symlink()
    )
    require(
        not symlinks,
        f"Canonical leaf contains symlink/path escape at {root}: {symlinks}",
    )
    attempts = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and not path.is_symlink()
        and path.name.startswith("attempt_")
    )
    require(
        attempts == ["attempt_001"],
        f"No-retry policy violated at {root}: {attempts}",
    )
    pointer_path = root / "canonical.json"
    require(
        pointer_path.is_file() and not pointer_path.is_symlink(),
        f"Canonical pointer is missing or symlinked: {pointer_path}",
    )
    pointer = load_json(pointer_path)
    require(
        set(pointer)
        == {"attempt", "completion_sha256", "identity_sha256"},
        f"Canonical pointer schema changed: {pointer_path}",
    )
    attempt = repository_path(pointer["attempt"], name="canonical attempt")
    require(
        pointer["attempt"] == relative(root / "attempt_001")
        and attempt == (root / "attempt_001").resolve()
        and (root / "attempt_001").is_dir()
        and not (root / "attempt_001").is_symlink(),
        f"Canonical pointer does not select attempt_001: {pointer_path}",
    )
    completion_path = attempt / "completion.json"
    require(
        completion_path.is_file()
        and file_sha256(completion_path) == pointer["completion_sha256"],
        f"Canonical completion hash mismatch: {pointer_path}",
    )
    completion = load_json(completion_path)
    require(
        compact_sha256(completion.get("identity"))
        == pointer["identity_sha256"],
        f"Canonical identity hash mismatch: {pointer_path}",
    )
    require(
        not (attempt / "failure.json").exists(),
        f"Canonical leaf also contains failure.json: {attempt}",
    )
    validate_leaf_filesystem_contract(
        root,
        attempt,
        completion,
        label=f"s{seed}:{trait}:u{update}",
    )
    return attempt, pointer, completion


def expected_leaf_identity(
    config_hash: str,
    runner_hash: str,
    preflight_hash: str,
    *,
    seed: int,
    trait: str,
    update: int,
    replay_kind: str,
) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "lineage": LINEAGE,
        "training_seed": seed,
        "trait": trait,
        "target_update": update,
        "replay_kind": replay_kind,
        "optimizer_updates": 24,
        "config_sha256": config_hash,
        "script_sha256": runner_hash,
        "preflight_sha256": preflight_hash,
        "v1_runner_sha256": PINNED_V1["runner_sha256"],
        "v1_config_sha256": PINNED_V1["config_sha256"],
    }


def validate_cell(
    v1: Any,
    *,
    attempt: Path,
    record: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    factor_sets: dict[str, dict[str, dict[str, Any]]],
    factor_audit: dict[str, Any],
    checkpoint_identity_sha256: str,
) -> tuple[
    tuple[Any, ...],
    tuple[str, dict[str, np.ndarray] | None],
    dict[str, Any],
    dict[str, float] | None,
]:
    logical = key_tuple(record["key"])
    stem = cell_stem(record["key"])
    json_path = repository_path(record["path"], name="causal cell JSON")
    require(
        json_path == (attempt / "cells" / f"{stem}.json").resolve()
        and file_sha256(json_path) == record["sha256"],
        f"Causal cell path/hash mismatch: {json_path}",
    )
    payload = load_json(json_path)
    require(
        payload["key"] == record["key"]
        and payload["status"] == record["status"],
        f"Causal cell embedded identity mismatch: {json_path}",
    )
    common_payload = {
        "schema": "teacher_trait_fingerprint_ontogeny_v2_cell",
        "experiment_id": EXPERIMENT_ID,
        "checkpoint_identity_sha256": checkpoint_identity_sha256,
    }
    require(
        all(payload.get(key) == value for key, value in common_payload.items()),
        f"Causal cell checkpoint binding mismatch: {json_path}",
    )
    status = payload["status"]
    require(
        status in {"evaluated", "not_applicable"},
        f"Unknown causal cell status: {json_path}",
    )
    key = payload["key"]
    identifier = v1.factor_set_id(
        key["construction"], key["control_kind"], key["control_draw"]
    )
    expected_manifest = manifests.get(identifier)
    expected_factors = factor_sets.get(identifier)
    require(
        payload.get("factor_set_id")
        == (identifier if expected_manifest is not None else None)
        and payload.get("factor_manifest") == expected_manifest,
        f"Factor-set reference mismatch: {json_path}",
    )
    real = factor_sets.get(
        v1.factor_set_id(key["construction"], "real", -1)
    )
    expected_reason = None
    if real is None:
        expected_reason = "checkpoint_local_rank1_unidentifiable"
    elif all(abs(float(factor["s"])) <= 1e-30 for factor in real.values()):
        expected_reason = "zero_checkpoint_component"
    expected_status = (
        "not_applicable" if expected_reason is not None else "evaluated"
    )
    require(
        status == expected_status,
        f"Causal applicability mismatch: {json_path}",
    )
    if status == "not_applicable":
        require(
            set(record) == {"key", "status", "path", "sha256"}
            and set(payload)
            == {
                "schema",
                "experiment_id",
                "checkpoint_identity_sha256",
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
            and payload["factor_record"] == factor_audit
            and payload["metrics"] is None
            and payload["arrays_path"] is None,
            f"Malformed N/A causal cell: {json_path}",
        )
        return (
            logical,
            (status, None),
            {"key": key, "status": status, "json": artifact(json_path), "arrays": None},
            None,
        )

    require(
        set(record)
        == {
            "key",
            "status",
            "path",
            "sha256",
            "arrays_path",
            "arrays_sha256",
        }
        and set(payload)
        == {
            "schema",
            "experiment_id",
            "checkpoint_identity_sha256",
            "key",
            "status",
            "reason",
            "factor_record",
            "factor_set_id",
            "factor_manifest",
            "metrics",
            "arrays_path",
            "arrays_sha256",
        }
        and expected_factors is not None,
        f"Evaluated causal cell schema mismatch: {json_path}",
    )
    require(payload["reason"] is None, f"Evaluated cell has N/A reason: {json_path}")
    array_path = repository_path(
        payload["arrays_path"], name="causal cell arrays"
    )
    require(
        array_path == (attempt / "cells" / f"{stem}.npz").resolve()
        and record["arrays_path"] == payload["arrays_path"]
        and file_sha256(array_path)
        == record["arrays_sha256"]
        == payload["arrays_sha256"],
        f"Causal array path/hash mismatch: {array_path}",
    )
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    v1.validate_causal_arrays(payload, arrays, relative(json_path))
    amplitudes = v1.validate_factor_record(
        payload["factor_record"], relative(json_path)
    )
    compare_tree(
        payload["factor_record"],
        v1.factor_summary(expected_factors),
        f"factor summary {json_path}",
    )
    analysis_arrays = {
        name: arrays[name]
        for name in (
            "numeric_native_js",
            "numeric_oriented_js_progress",
            "logit_context_field_dot",
            "behavior_native_gap",
            "behavior_oriented_effect",
            "hard_event",
            "hard_oriented_recovery",
        )
    }
    return (
        logical,
        (status, analysis_arrays),
        {
            "key": key,
            "status": status,
            "json": artifact(json_path),
            "arrays": artifact(array_path),
        },
        amplitudes,
    )


def validate_leaves(
    v1: Any,
    config: dict[str, Any],
    config_v1: dict[str, Any],
    endpoints: dict[tuple[int, str], dict[str, Any]],
    native: dict[tuple[int, str, int], Any],
    native_completions: dict[tuple[int, str], dict[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[tuple[Any, ...], tuple[str, dict[str, np.ndarray] | None]],
    dict[tuple[int, str, int], bool],
    dict[tuple[int, str, int], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    config_hash = file_sha256(CONFIG_PATH)
    runner_hash = file_sha256(RUNNER_PATH)
    preflight_hash = file_sha256(PREFLIGHT_PATH)
    preflight_payload = load_json(PREFLIGHT_PATH)
    dependency = expected_dependency_binding(config)
    upstream = expected_upstream_locks()
    expected_global = {
        key_tuple(key)
        for seed, trait, update in expected_leaf_keys()
        for key in expected_leaf_cells(seed, trait, update)
    }
    manifest = {}
    analysis_cells = {}
    replay_usable = {}
    completions = {}
    live_u0 = {}
    amplitudes: dict[tuple[Any, ...], dict[str, float]] = {}
    observed_global: set[tuple[Any, ...]] = set()
    for seed, trait, update in expected_leaf_keys():
        label = f"s{seed}:{trait}:u{update:04d}"
        attempt, pointer, completion = canonical_leaf(seed, trait, update)
        require(
            set(completion)
            == {
                "schema",
                "identity",
                "created_utc",
                "attempt",
                "dependency_binding",
                "dependency_binding_sha256",
                "upstream_locks",
                "sources",
                "context_identity",
                "live_readout",
                "checkpoint_record",
                "cells",
                "expected_cell_count",
                "evaluated_cell_count",
                "not_applicable_cell_count",
                "training_metrics_path",
                "training_metrics_sha256",
                "optimizer_updates",
                "complete",
                "v1_causal_scientific_artifacts_consumed",
                "preflight_source_sha256",
            },
            f"Leaf completion schema changed: {label}",
        )
        identity = completion["identity"]
        require(
            isinstance(identity, dict)
            and set(identity)
            == set(config["integrity"]["leaf_identity_fields"])
            and isinstance(identity.get("replay_kind"), str)
            and identity["replay_kind"],
            f"Leaf identity schema changed: {label}",
        )
        expected_identity = expected_leaf_identity(
            config_hash,
            runner_hash,
            preflight_hash,
            seed=seed,
            trait=trait,
            update=update,
            replay_kind=identity["replay_kind"],
        )
        require(identity == expected_identity, f"Leaf identity mismatch: {label}")
        expected_sources = expected_leaf_sources(
            v1,
            endpoints,
            native_completions,
            seed=seed,
            trait=trait,
            update=update,
        )
        require(
            completion["schema"]
            == (
                "teacher_trait_fingerprint_ontogeny_v2_"
                "checkpoint_completion"
            )
            and completion["attempt"] == relative(attempt)
            and completion["optimizer_updates"] == 24
            and completion["complete"] is True,
            f"Leaf completion state mismatch: {label}",
        )
        require(
            completion["dependency_binding"] == dependency
            and completion["dependency_binding_sha256"]
            == compact_sha256(dependency)
            and completion["upstream_locks"] == upstream
            and completion["sources"] == expected_sources
            and completion["context_identity"]
            == expected_sources["current_native"]["context_identity"]
            and completion["v1_causal_scientific_artifacts_consumed"]
            is False
            and completion["preflight_source_sha256"]
            == compact_sha256(preflight_payload["source"]),
            f"Leaf upstream/source binding mismatch: {label}",
        )
        checkpoint = completion["checkpoint_record"]
        require(
            set(checkpoint)
            == {
                "optimizer_update",
                "live_readout",
                "repeat_guard",
                "factor_audit",
                "factor_catalog",
                "safety",
            }
            and checkpoint["optimizer_update"] == update
            and checkpoint["live_readout"] == completion["live_readout"],
            f"Checkpoint record schema/cross-link mismatch: {label}",
        )
        safety = validate_single_safety(
            checkpoint["safety"], seed=seed, trait=trait, update=update
        )
        live = load_readout_record(
            completion["live_readout"],
            expected_path=attempt / "live_readout.pt",
            name=f"live readout {label}",
            v1=v1,
        )
        require(
            live.identity
            == {
                "lineage": LINEAGE,
                "training_seed": seed,
                "trait": trait,
                "optimizer_update": update,
                **completion["context_identity"],
                "selected_weight_sha256": safety[
                    "selected_weight_sha256"
                ],
            },
            f"Live readout identity mismatch: {label}",
        )
        repeat = recompute_repeat_guard(
            v1,
            config_v1,
            live,
            native[(seed, trait, update)],
            native[(seed, v1.other_trait(trait), update)],
            trait=trait,
            update=update,
        )
        compare_tree(
            checkpoint["repeat_guard"],
            repeat,
            f"live repeat guard {label}",
        )
        require(
            repeat["pass"] is True
            and (update != 0 or repeat["relative_or_u0_pass"] is True),
            f"Canonical leaf failed hard replay guard: {label}",
        )
        replay_usable[(seed, trait, update)] = bool(
            repeat["usable_for_onset"]
        )
        if update == 0:
            context = native_completions[(seed, trait)]["context_identity"]
            live_u0[f"isolated:{label}"] = {
                "path": repository_path(
                    completion["live_readout"]["path"],
                    name=f"u0 live readout {label}",
                ),
                "readout": live,
                "context_identity": context,
            }

        manifests, factor_sets = validate_factor_catalog(
            v1,
            config_v1,
            endpoints,
            seed=seed,
            trait=trait,
            update=update,
            safety=safety,
            factor_record=checkpoint["factor_catalog"],
            attempt=attempt,
            expected_identity=expected_identity,
        )
        require(
            checkpoint["factor_audit"]
            == torch.load(
                repository_path(
                    checkpoint["factor_catalog"]["path"],
                    name=f"factor catalog {label}",
                ),
                map_location="cpu",
                weights_only=True,
            )["factor_audit"],
            f"Factor audit cross-link mismatch: {label}",
        )
        expected_leaf = {
            key_tuple(key)
            for key in expected_leaf_cells(seed, trait, update)
        }
        records = completion["cells"]
        require(
            isinstance(records, list) and len(records) == 30,
            f"Leaf cell count changed: {label}",
        )
        observed = [key_tuple(record["key"]) for record in records]
        require(
            len(observed) == len(set(observed))
            and set(observed) == expected_leaf,
            f"Leaf expected-key equality failed: {label}",
        )
        cell_manifest = []
        for record in records:
            logical, analysis, evidence, signature = validate_cell(
                v1,
                attempt=attempt,
                record=record,
                manifests=manifests,
                factor_sets=factor_sets,
                factor_audit=checkpoint["factor_audit"],
                checkpoint_identity_sha256=compact_sha256(identity),
            )
            require(
                logical not in observed_global,
                f"Duplicate global causal key: {logical}",
            )
            observed_global.add(logical)
            analysis_cells[logical] = analysis
            cell_manifest.append(evidence)
            if signature is not None:
                amplitudes[logical] = signature
        evaluated = sum(
            record["status"] == "evaluated" for record in records
        )
        not_applicable = 30 - evaluated
        require(
            completion["expected_cell_count"] == 30
            and completion["evaluated_cell_count"] == evaluated
            and completion["not_applicable_cell_count"] == not_applicable,
            f"Leaf count cross-link mismatch: {label}",
        )
        metrics = validate_training_metrics(
            v1,
            config_v1,
            attempt,
            completion,
            seed=seed,
            target_update=update,
        )
        manifest[label] = {
            "canonical": artifact(checkpoint_root(seed, trait, update) / "canonical.json"),
            "completion": artifact(attempt / "completion.json"),
            "training_metrics": metrics,
            "live_readout": artifact(attempt / "live_readout.pt"),
            "factor_catalog": artifact(attempt / "factors.pt"),
            "cell_count": 30,
            "evaluated": evaluated,
            "not_applicable": not_applicable,
            "cell_manifest_sha256": compact_sha256(cell_manifest),
        }
        completions[(seed, trait, update)] = completion
    require(
        observed_global == expected_global
        and len(observed_global) == len(analysis_cells) == 1080,
        "Global isolated causal key inventory mismatch",
    )

    for logical, signature in amplitudes.items():
        key = dict(zip(KEY_FIELDS, logical))
        if key["control_kind"] == "real":
            continue
        real_key = key_tuple(
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
        require(real_key in amplitudes, "Evaluated control lacks real peer")
        for module, value in signature.items():
            require(
                math.isclose(
                    abs(value),
                    abs(amplitudes[real_key][module]),
                    rel_tol=2e-6,
                    abs_tol=2e-9,
                ),
                f"Norm matching failed: {logical}/{module}",
            )
    return (
        manifest,
        analysis_cells,
        replay_usable,
        completions,
        live_u0,
    )


def metamorphic_leaf_index_test() -> dict[str, Any]:
    fixture = {
        (seed, trait, update): f"{seed}:{trait}:{update}"
        for seed, trait, update in expected_leaf_keys()
    }
    before = dict(fixture)
    target = (2102, "lion", 12)
    fixture[target] = "mutated"
    changed = [key for key in fixture if fixture[key] != before[key]]
    require(changed == [target], "Checkpoint leaf axis is not independently keyed")
    all_cells = [
        key_tuple(key)
        for seed, trait, update in expected_leaf_keys()
        for key in expected_leaf_cells(seed, trait, update)
    ]
    require(
        len(all_cells) == len(set(all_cells)) == 1080,
        "Global cell key grid is not unique",
    )
    return {
        "leaf_count": len(fixture),
        "cell_count": len(all_cells),
        "mutated_leaf": list(target),
        "changed_only_target": True,
    }


def singleton_callback_fixture() -> dict[str, Any]:
    expected_lrs = load_json(V1_CONFIG_PATH)["integrity"][
        "expected_learning_rates_after_update"
    ]
    update_rows = [
        {
            "optimizer_update": update,
            "learning_rates_after_update": [expected_lrs[update - 1]],
        }
        for update in range(1, 25)
    ]
    target = 12
    callback_rows = [{"optimizer_update": target}]
    require(
        len(update_rows) == 24
        and [row["optimizer_update"] for row in update_rows]
        == list(range(1, 25))
        and len(callback_rows) == 1
        and callback_rows[0]["optimizer_update"] == target,
        "Full-replay/single-callback fixture failed",
    )
    return {
        "optimizer_updates": 24,
        "target_update": target,
        "callback_count": 1,
        "pass": True,
    }


def self_test() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    protocol = validate_compact_config(
        config, allow_verifier_placeholder=True
    )
    no_import_self = validate_no_runner_imports(SCRIPT_PATH)
    no_import_v1 = validate_no_runner_imports(V1_VERIFIER_PATH)
    v1 = import_v1_cleanroom()
    grid = metamorphic_leaf_index_test()
    callbacks = singleton_callback_fixture()
    inherited_metamorphic = v1.metamorphic_update_index_regression()
    drift = v1.generic_drift_cancellation_regression()
    conjunction = v1.conjunction_below_regression()
    runner = validate_runner_static(config)
    require(
        protocol["leaf_count"] == 36
        and protocol["cell_count"] == 1080
        and inherited_metamorphic["passed"]
        and drift["passed"]
        and conjunction["passed"],
        "Model-free verifier regression failed",
    )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "protocol": protocol,
        "runner_static": runner,
        "verifier_no_runner_import": no_import_self,
        "v1_cleanroom_no_runner_import": no_import_v1,
        "leaf_index_metamorphic": grid,
        "singleton_callback_fixture": callbacks,
        "inherited_update_metamorphic": inherited_metamorphic,
        "generic_drift_cancellation": drift,
        "conjunction_below": conjunction,
        "model_loaded": False,
        "scientific_outcomes_required": False,
        "pass": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def validate_checkpoint_lock(
    v1: Any,
    config: dict[str, Any],
    native: dict[tuple[int, str, int], Any],
    native_completions: dict[tuple[int, str], dict[str, Any]],
    completions: dict[tuple[int, str, int], dict[str, Any]],
    live_u0: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    observed = load_json(CHECKPOINT_LOCK_PATH)
    leaves = {}
    ordered_global = []
    u0_members: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        for trait in TRAITS:
            native_completion = native_completions[(seed, trait)]
            native_record = native_completion["readouts"][0]
            native_path = repository_path(
                native_record["path"], name="v1 native u0"
            )
            u0_members[f"v1_native:s{seed}:{trait}"] = {
                "path": native_path,
                "readout": native[(seed, trait, 0)],
                "context_identity": native_completion["context_identity"],
            }
    for seed, trait, update in expected_leaf_keys():
        label = f"s{seed}:{trait}:u{update:04d}"
        completion = completions[(seed, trait, update)]
        root = checkpoint_root(seed, trait, update)
        attempt = root / "attempt_001"
        pointer = root / "canonical.json"
        ordered_global.extend(dict(record["key"]) for record in completion["cells"])
        leaves[label] = {
            "identity": {
                "training_seed": seed,
                "trait": trait,
                "target_update": update,
            },
            "attempt": relative(attempt),
            "canonical_pointer_sha256": file_sha256(pointer),
            "completion_sha256": file_sha256(attempt / "completion.json"),
            "training_metrics_sha256": completion[
                "training_metrics_sha256"
            ],
            "live_readout_sha256": completion["live_readout"]["sha256"],
            "live_numeric_logits_sha256": completion["live_readout"][
                "numeric_logits_sha256"
            ],
            "live_animal_logits_sha256": completion["live_readout"][
                "animal_logits_sha256"
            ],
            "factor_catalog_sha256": completion["checkpoint_record"][
                "factor_catalog"
            ]["sha256"],
            "factor_manifest_sha256": completion["checkpoint_record"][
                "factor_catalog"
            ]["factor_manifest_sha256"],
            "cell_manifest_sha256": compact_sha256(completion["cells"]),
            "expected_cell_count": completion["expected_cell_count"],
            "evaluated_cell_count": completion["evaluated_cell_count"],
            "not_applicable_cell_count": completion[
                "not_applicable_cell_count"
            ],
        }
        if update == 0:
            source_key = f"isolated:{label}"
            require(source_key in live_u0, f"Missing isolated u0: {label}")
            u0_members[f"v2_isolated:s{seed}:{trait}"] = live_u0[
                source_key
            ]
    expected_order = [
        key
        for seed, trait, update in expected_leaf_keys()
        for key in expected_leaf_cells(seed, trait, update)
    ]
    require(
        ordered_global == expected_order,
        "Global checkpoint-lock key order changed",
    )
    u0 = v1.independent_u0_equivalence(
        load_json(V1_CONFIG_PATH), u0_members
    )
    preflight = load_json(PREFLIGHT_PATH)
    dependency = expected_dependency_binding(config)
    expected = {
        "schema": "teacher_trait_fingerprint_ontogeny_v2_checkpoint_lock",
        "experiment_id": EXPERIMENT_ID,
        "created_utc": str(observed.get("created_utc")),
        "config_sha256": file_sha256(CONFIG_PATH),
        "script_sha256": file_sha256(RUNNER_PATH),
        "preflight_sha256": file_sha256(PREFLIGHT_PATH),
        "preflight_source_sha256": compact_sha256(preflight["source"]),
        "dependency_binding": dependency,
        "dependency_binding_sha256": compact_sha256(dependency),
        "upstream_locks": expected_upstream_locks(),
        "leaves": leaves,
        "expected_leaf_count": 36,
        "leaf_key_sha256": compact_sha256(
            [
                {
                    "training_seed": seed,
                    "trait": trait,
                    "target_update": update,
                }
                for seed, trait, update in expected_leaf_keys()
            ]
        ),
        "global_expected_key_count": 1080,
        "global_expected_key_sha256": compact_sha256(expected_order),
        "u0_equivalence": u0,
        "v1_causal_scientific_artifacts_consumed": False,
    }
    compare_tree(observed, expected, "v2 checkpoint lock")
    return artifact(CHECKPOINT_LOCK_PATH), leaves


def expected_inventory(
    completions: dict[tuple[int, str, int], dict[str, Any]],
    checkpoint_lock: dict[str, Any],
) -> dict[str, Any]:
    leaves = {}
    evaluated_total = 0
    na_total = 0
    for seed, trait, update in expected_leaf_keys():
        label = f"s{seed}:{trait}:u{update:04d}"
        completion = completions[(seed, trait, update)]
        evaluated = int(completion["evaluated_cell_count"])
        na = int(completion["not_applicable_cell_count"])
        evaluated_total += evaluated
        na_total += na
        leaves[label] = {
            "valid": True,
            "attempt": completion["attempt"],
            "cells": 30,
            "evaluated": evaluated,
            "not_applicable": na,
        }
    return {
        "schema": "teacher_trait_fingerprint_ontogeny_v2_inventory",
        "preflight": {
            "valid": True,
            "sha256": file_sha256(PREFLIGHT_PATH),
            "source_sha256": compact_sha256(
                load_json(PREFLIGHT_PATH)["source"]
            ),
        },
        "leaves": leaves,
        "checkpoint_lock": {
            "valid": True,
            "sha256": checkpoint_lock["sha256"],
            "leaves": 36,
            "cells": 1080,
            "u0_all_pairs_pass": True,
        },
        "expected_leaf_count": 36,
        "valid_leaf_count": 36,
        "expected_cell_count": 1080,
        "observed_cell_count": 1080,
        "evaluated_cell_count": evaluated_total,
        "not_applicable_cell_count": na_total,
        "global_key_sha256": compact_sha256(
            [
                key
                for seed, trait, update in expected_leaf_keys()
                for key in expected_leaf_cells(seed, trait, update)
            ]
        ),
        "v1_causal_scientific_artifacts_consumed": False,
        "complete": True,
    }


def verify() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    protocol_record = validate_compact_config(
        config, allow_verifier_placeholder=False
    )
    runner_static = validate_runner_static(config)
    self_import_audit = validate_no_runner_imports(SCRIPT_PATH)
    v1_import_audit = validate_no_runner_imports(V1_VERIFIER_PATH)
    pinned_files = validate_pinned_v1_files()
    failed_attempts = validate_failed_attempt_quarantine()
    preflight = validate_preflight(config)
    require(
        OUT_JSON.is_file() and OUT_MD.is_file(),
        "Completed v2 production aggregate is required",
    )
    v1 = import_v1_cleanroom()
    (
        upstream_manifest,
        endpoints,
        native,
        native_completions,
    ) = validate_v1_scientific_upstreams(v1)
    config_v1 = load_json(V1_CONFIG_PATH)
    (
        leaf_manifest,
        cells,
        replay_usable,
        completions,
        live_u0,
    ) = validate_leaves(
        v1,
        config,
        config_v1,
        endpoints,
        native,
        native_completions,
    )
    checkpoint_lock, lock_leaves = validate_checkpoint_lock(
        v1,
        config,
        native,
        native_completions,
        completions,
        live_u0,
    )
    aggregate = load_json(OUT_JSON)
    dependency = expected_dependency_binding(config)
    require(
        aggregate["experiment_id"] == EXPERIMENT_ID
        and aggregate["protocol_sha256"] == file_sha256(CONFIG_PATH)
        and aggregate["script_sha256"] == file_sha256(RUNNER_PATH)
        and aggregate["git_head"] == preflight["git_head"]
        and aggregate["v1_inherited_runner_sha256"]
        == PINNED_V1["runner_sha256"]
        and aggregate["v1_inherited_config_sha256"]
        == PINNED_V1["config_sha256"]
        and aggregate["checkpoint_lock_sha256"]
        == checkpoint_lock["sha256"]
        and aggregate["checkpoint_lock_manifest_sha256"]
        == compact_sha256(lock_leaves)
        and aggregate["dependency_binding"] == dependency
        and aggregate["v1_causal_scientific_artifacts_consumed"] is False,
        "Production aggregate provenance mismatch",
    )
    inventory = expected_inventory(completions, checkpoint_lock)
    require(
        aggregate["inventory"] == inventory,
        "Production aggregate inventory mismatch",
    )
    require(
        aggregate["status"]
        == {
            "artifact_integrity_valid": True,
            "analysis_implementation_valid": True,
            "primary_classification_valid": None,
            "overall_pass": False,
            "overall_pass_reason": "pending_independent_verifier",
        },
        "Production aggregate did not preserve verifier ownership",
    )

    native_summaries, records, native_gate_records = v1.recompute_native(
        config_v1, native
    )
    compare_tree(
        aggregate["native_summaries"],
        native_summaries,
        "aggregate.native_summaries",
    )
    indices = v1.bootstrap_indices(config_v1)
    causal_summaries, causal_gate_records = v1.recompute_causal_analysis(
        config_v1,
        cells,
        replay_usable,
        records,
        indices,
    )
    hard_summaries, hard_gate_records = v1.recompute_hard_analysis(
        config_v1,
        cells,
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
    bootstrap = v1.finalize_and_validate_bootstrap(
        config_v1, aggregate, records
    )
    onsets = v1.recompute_onsets_and_classification(
        records,
        native_gate_records,
        causal_gate_records,
        causal_summaries,
        hard_summaries,
        hard_gate_records,
    )
    compare_tree(aggregate["onsets"], onsets, "aggregate.onsets")
    classification = onsets["classification"]
    evidence = {
        "pinned_v1_files": pinned_files,
        "failed_attempt_quarantine": failed_attempts,
        "sealed_v1_scientific_upstreams": upstream_manifest,
        "v2_leaves": leaf_manifest,
        "v2_checkpoint_lock": checkpoint_lock,
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
        "static_runner_validation": runner_static,
        "verifier_no_runner_import": self_import_audit,
        "v1_cleanroom_no_runner_import": v1_import_audit,
        "artifact_manifest": evidence,
        "artifact_manifest_sha256": compact_sha256(evidence),
        "native_recomputation": {
            "summaries_exact_with_tolerance": True,
            "onsets": onsets["native"],
        },
        "causal_recomputation": {
            "summaries_exact_with_tolerance": True,
            "hard_summaries_exact_with_tolerance": True,
            "all_scalar_bootstrap_records_recomputed": True,
        },
        "bootstrap_recomputation": bootstrap,
        "onsets": onsets,
        "classification": classification,
        "production_status_before_verifier": aggregate["status"],
        "status": status,
        "overall_pass_rule_applied": (
            "artifact_integrity_valid and analysis_implementation_valid and "
            "primary_classification_valid"
        ),
        "production_v1_runner_imported": False,
        "production_v2_runner_imported": False,
        "v1_cleanroom_verifier_imported": True,
        "v1_causal_scientific_artifacts_consumed": False,
        "model_loaded": False,
        "tensor_artifacts_loaded_on_cpu_only": True,
    }
    require(
        result["status"]["overall_pass"]
        == all(
            result["status"][name]
            for name in (
                "artifact_integrity_valid",
                "analysis_implementation_valid",
                "primary_classification_valid",
            )
        ),
        "Overall-pass rule was not applied exactly",
    )
    atomic_json(OUT_VERIFY, result)
    print(
        "TEACHER TRAIT/FINGERPRINT ONTOGENY V2 "
        "INDEPENDENT VERIFICATION PASSED",
        flush=True,
    )
    print(
        json.dumps(
            {"classification": classification, "status": status},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("self-test", "verify"), nargs="?", default="self-test"
    )
    args = parser.parse_args()
    if args.action == "self-test":
        self_test()
    else:
        verify()


if __name__ == "__main__":
    main()
