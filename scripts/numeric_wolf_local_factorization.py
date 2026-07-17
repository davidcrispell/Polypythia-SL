"""Exact local-factor factorization of the held-out numeric--wolf overlap.

This is a read-only saved-state assay.  At the 16 frozen ds2 cells it hooks
the two inner PEFT Linear modules of selected LoRA adapters and reconstructs
their gradients as ``D.T @ X``.  Paired preference/control numeric passes then
permit the exact local hybrids ``G_ab = D_a.T @ X_b`` and the symmetric
two-factor Shapley split frozen in
``configs/numeric_wolf_local_factorization_v1.json``.

The split apportions the *numeric preference-vs-control gradient change* into
its forward-X and backward-D factors.  It is not, by itself, a causal claim
that either factor stores wolf semantics.  A small frozen diagnostic directly
checks the multiplicative wolf--numeric kernel identity.  No optimizer step is
taken and no tensor-valued artifact is written.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import numeric_wolf_cross_gradient_localization as stage2


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = Path(__file__).resolve()
CONFIG_PATH = ROOT / "configs/numeric_wolf_local_factorization_v1.json"
WORK = ROOT / "runs/numeric_wolf_local_factorization_v1"
PREFLIGHT_PATH = WORK / "preflight.json"
RUNNER_LOCK_PATH = WORK / "runner_lock.json"
ACTIVE_LOCK_PATH = WORK / ".active.lock"
OUT_JSON = ROOT / "runs/numeric_wolf_local_factorization_v1.json"
OUT_MD = ROOT / "runs/numeric_wolf_local_factorization_v1.md"

DEVICE = stage2.DEVICE
SEEDS = (56101, 56102)
CONDITIONS = ("preference", "control")
CHECKPOINTS = (64, 128, 256, 512)
TARGET_LAYERS = (8, 9, 10, 11)
CONTROL_LAYERS = (0, 1, 2, 3)
MODULES = ("query_key_value", "dense_4h_to_h")
SIDES = ("lora_A", "lora_B")
COMPONENTS = (
    "k_pp", "k_pc", "k_cp", "k_cc",
    "kappa", "phi_x", "phi_d", "interaction",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def int64_sha256(value: Any) -> str:
    return stage2.int64_sha256(value)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def artifact_record(path: Path) -> dict[str, Any]:
    return stage2.artifact_record(path)


def verify_artifact(record: dict[str, Any]) -> Path:
    return stage2.verify_artifact(record)


def atomic_write_json(path: Path, value: Any) -> None:
    stage2.atomic_write_json(path, value)


def atomic_write_text(path: Path, value: str) -> None:
    stage2.atomic_write_text(path, value)


def finite_tree(value: Any) -> bool:
    return stage2.finite_tree(value)


def implementation_guard() -> dict[str, str]:
    return {
        "config_sha256": file_sha256(CONFIG_PATH),
        "runner_sha256": file_sha256(SCRIPT_PATH),
    }


def cell_key(seed: int, condition: str, update: int) -> str:
    return f"ds2/{seed}/{condition}/u{update:04d}"


def expected_cell_path(seed: int, condition: str, update: int) -> Path:
    return (
        WORK / "cells/ds2" / f"seed_{seed}" / condition
        / f"u{update:04d}" / "cell.json"
    )


def expected_cells() -> list[tuple[int, str, int, Path]]:
    return [
        (seed, condition, update, expected_cell_path(seed, condition, update))
        for seed in SEEDS for condition in CONDITIONS for update in CHECKPOINTS
    ]


def logical_atomic_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for band, layers in (("early", CONTROL_LAYERS), ("late", TARGET_LAYERS)):
        for layer in layers:
            for module in MODULES:
                for side in SIDES:
                    rows.append({
                        "key": f"{band}.layer_{layer:02d}.{module}.{side}",
                        "band": band,
                        "layer": layer,
                        "module": module,
                        "side": side,
                    })
    return rows


def group_inventory() -> list[str]:
    values: list[str] = []
    for band in ("early", "late"):
        values.append(f"{band}_all")
        values.extend(f"{band}_{module}" for module in MODULES)
        values.extend(f"{band}_{side}" for side in SIDES)
    values.extend(
        f"{row['band']}.layer_{row['layer']:02d}.{row['module']}"
        for row in logical_atomic_inventory()[::2]
    )
    values.extend(row["key"] for row in logical_atomic_inventory())
    if len(values) != len(set(values)):
        raise RuntimeError("Factorization group inventory is not unique")
    return values


def groups_for_atomic(meta: dict[str, Any]) -> tuple[str, ...]:
    return (
        f"{meta['band']}_all",
        f"{meta['band']}_{meta['module']}",
        f"{meta['band']}_{meta['side']}",
        f"{meta['band']}.layer_{meta['layer']:02d}.{meta['module']}",
        meta["key"],
    )


def validate_config_contract(config: dict[str, Any]) -> None:
    if config.get("name") != "numeric-wolf-local-factorization-v1":
        raise RuntimeError("Unexpected factorization config name")
    measurement = config["measurement"]
    expected = {
        "receiver": "ds2",
        "seeds": list(SEEDS),
        "source_conditions": list(CONDITIONS),
        "checkpoints": list(CHECKPOINTS),
        "expected_cell_count": 16,
        "behavior_split": "primary",
        "behavior_cluster_count": 6,
        "behavior_cluster_size": 5,
        "numeric_block_count": 8,
        "numeric_rows_per_block": 64,
        "numeric_microbatch_size": 16,
        "target_layers": list(TARGET_LAYERS),
        "matched_control_layers": list(CONTROL_LAYERS),
        "target_module_families": list(MODULES),
        "adapter_sides": list(SIDES),
        "expected_targeted_trainable_tensor_count": 32,
        "expected_peft_version": "0.19.1",
        "expected_active_adapter": "default",
        "expected_lora_scaling": 2.0,
        "expected_lora_dtype": "torch.float32",
        "expected_hidden_dropout": 0.0,
        "expected_attention_dropout": 0.0,
        "paired_rows": 512,
        "paired_total_token_length_min": 25,
        "paired_total_token_length_max": 39,
        "supervised_tokens_per_row": 19,
        "kernel_diagnostic_behavior_cluster": 0,
        "kernel_diagnostic_numeric_block": 0,
        "kernel_diagnostic_band": "late",
    }
    for key, value in expected.items():
        if measurement.get(key) != value:
            raise RuntimeError(f"Frozen measurement changed: {key}")
    if measurement["behavior_prompt_indices"] != list(range(30, 60)):
        raise RuntimeError("Primary behavior indices changed")
    if set(group_inventory()) == set(COMPONENTS):
        raise RuntimeError("Synthetic inventory collision")
    if len(expected_cells()) != 16 or len(logical_atomic_inventory()) != 32:
        raise RuntimeError("Frozen cell or atomic inventory changed")
    bootstrap = config["frozen_analysis"]["bootstrap"]
    if (
        int(bootstrap["resamples"]) != 10000
        or int(bootstrap["seed"]) != 59411
    ):
        raise RuntimeError("Bootstrap recipe changed")
    rng = np.random.default_rng(59411)
    prompt = rng.integers(0, 6, size=(10000, 6))
    numeric = rng.integers(0, 8, size=(10000, 8))
    if (
        int64_sha256(prompt.tolist())
        != bootstrap["prompt_cluster_draws_int64_sha256"]
        or int64_sha256(numeric.tolist())
        != bootstrap["numeric_block_draws_int64_sha256"]
    ):
        raise RuntimeError("Frozen bootstrap draws changed")


def load_and_validate_config() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    config = load_json(CONFIG_PATH)
    validate_config_contract(config)
    for key in (
        "stage2_config", "stage2_runner", "stage2_runner_lock",
        "stage2_result", "heldout_manifest",
    ):
        path = ROOT / config["parents"][key]
        if file_sha256(path) != config["parents"][f"{key}_sha256"]:
            raise RuntimeError(f"Frozen parent changed: {path}")
    stage2_config, ds2_config, dynamics_config = stage2.load_and_validate_config()
    stage2_lock = stage2.validate_runner_lock(stage2_config)
    aggregate = load_json(ROOT / config["parents"]["stage2_result"])
    if (
        aggregate.get("cell_count") != 40
        or aggregate.get("primary", {}).get("classification")
        != "heldout_localization_supported"
    ):
        raise RuntimeError("Stage-2 completion/classification contract changed")
    m2 = stage2_config["measurement"]
    if (
        tuple(m2["seeds"]) != SEEDS
        or tuple(m2["source_conditions"]) != CONDITIONS
        or tuple(m2["checkpoints"]["ds2"]) != CHECKPOINTS
        or m2["behavior_splits"]["primary"]["indices"]
        != config["measurement"]["behavior_prompt_indices"]
        or m2["behavior_splits"]["primary"]["prompt_sha256"]
        != config["measurement"]["behavior_prompt_sha256"]
        or int(m2["heldout_completion_batch_size"])
        != int(config["measurement"]["numeric_microbatch_size"])
    ):
        raise RuntimeError("Factorization measurement diverged from Stage 2")
    return config, stage2_config, ds2_config, dynamics_config, stage2_lock


def audit_paired_datasets(
    datasets: dict[str, Any], config: dict[str, Any], tokenizer
) -> dict[str, Any]:
    preference = datasets["preference"].examples
    control = datasets["control"].examples
    expected_rows = int(config["measurement"]["paired_rows"])
    if len(preference) != expected_rows or len(control) != expected_rows:
        raise RuntimeError("Paired held-out row count changed")
    lengths: list[int] = []
    shape_mismatches = attention_mismatches = supervised_mismatches = 0
    prompt_token_mismatches = 0
    for pref, ctrl in zip(preference, control):
        keys = ("input_ids", "labels")
        if any(pref[key].shape != ctrl[key].shape for key in keys):
            shape_mismatches += 1
            continue
        pref_supervised = pref["labels"] != -100
        ctrl_supervised = ctrl["labels"] != -100
        if not torch.equal(pref_supervised, ctrl_supervised):
            supervised_mismatches += 1
        prompt = ~(pref_supervised | ctrl_supervised)
        if not torch.equal(pref["input_ids"][prompt], ctrl["input_ids"][prompt]):
            prompt_token_mismatches += 1
        supervised_count = int(pref_supervised.sum())
        if supervised_count != int(config["measurement"]["supervised_tokens_per_row"]):
            raise RuntimeError("Supervised token count changed")
        lengths.append(int(pref["input_ids"].numel()))
    collator = stage2.CompletionCollator(tokenizer.pad_token_id)
    for start in range(0, expected_rows, 16):
        pref_batch = collator(preference[start:start + 16])
        ctrl_batch = collator(control[start:start + 16])
        if not torch.equal(pref_batch["attention_mask"], ctrl_batch["attention_mask"]):
            attention_mismatches += 1
    if any((shape_mismatches, attention_mismatches, supervised_mismatches, prompt_token_mismatches)):
        raise RuntimeError("Paired row/shape/mask/prompt guard failed")
    if min(lengths) != 25 or max(lengths) != 39:
        raise RuntimeError(f"Paired sequence-length range changed: {min(lengths)}..{max(lengths)}")
    return {
        "rows": expected_rows,
        "shape_mismatches": shape_mismatches,
        "attention_mask_mismatches": attention_mismatches,
        "supervised_mask_mismatches": supervised_mismatches,
        "prompt_token_mismatches": prompt_token_mismatches,
        "total_token_length_min": min(lengths),
        "total_token_length_max": max(lengths),
        "supervised_tokens_per_row": int(config["measurement"]["supervised_tokens_per_row"]),
        "passed": True,
    }


def stage2_cell_artifacts(
    stage2_config: dict[str, Any], stage2_lock: dict[str, Any],
    seed: int, condition: str, update: int,
) -> dict[str, Any]:
    path = stage2.expected_cell_path("ds2", seed, condition, update)
    result = stage2.validate_cell(
        path, stage2_config, stage2_lock, "ds2", seed, condition, update
    )
    result_path = ROOT / result["attempt"] / "result.json"
    return {"cell": artifact_record(path), "result": artifact_record(result_path)}


def assert_no_competing_experiment() -> None:
    """Reject other MPS assays while allowing this runner's caffeinate wrapper.

    The Stage-2 helper compares caffeinate commands against the Stage-2 script
    name, so a child assay that imports it can mistake its own wrapper for a
    competitor.  Same-assay serialization remains enforced by ``active_lock``.
    """
    output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,command="], text=True)
    processes: dict[int, tuple[int, str]] = {}
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3:
            continue
        try:
            processes[int(fields[0])] = (int(fields[1]), fields[2])
        except ValueError:
            pass
    ancestors = {os.getpid()}
    cursor = os.getpid()
    while cursor in processes:
        cursor = processes[cursor][0]
        if cursor <= 0 or cursor in ancestors:
            break
        ancestors.add(cursor)
    markers = (
        "scripts/numeric_", "scripts/ds2_", "scripts/dataorder_",
        "scripts/base_screening.py", "scripts/wolf_route_knockout.py",
        "polypythia_sl.pipeline",
    )
    conflicts = []
    for pid, (_, command) in processes.items():
        if pid in ancestors or "python" not in command.lower():
            continue
        if SCRIPT_PATH.name in command:
            continue
        if any(marker in command for marker in markers):
            conflicts.append({"pid": pid, "command": command})
    if conflicts:
        raise RuntimeError(f"Competing experiment process detected: {conflicts}")


def preflight() -> dict[str, Any]:
    config, stage2_config, _, dynamics_config, stage2_lock = load_and_validate_config()
    assert_no_competing_experiment()
    free = shutil.disk_usage(ROOT).free
    if free < int(config["resource_policy"]["minimum_launch_free_bytes"]):
        raise RuntimeError("Factorization preflight free-space guard failed")
    tokenizer = stage2.dynamics.load_tokenizer()
    datasets, manifest = stage2.prepare_datasets(stage2_config, dynamics_config, tokenizer)
    pairing = audit_paired_datasets(datasets, config, tokenizer)
    sources: dict[str, Any] = {}
    parent_cells: dict[str, Any] = {}
    for seed, condition, update, _ in expected_cells():
        key = cell_key(seed, condition, update)
        stage2_key = stage2.cell_key("ds2", seed, condition, update)
        source = stage2_lock["frozen"]["sources"][stage2_key]
        verify_artifact(source)
        sources[key] = source
        parent_cells[key] = stage2_cell_artifacts(
            stage2_config, stage2_lock, seed, condition, update
        )
    frozen = {
        "name": "numeric-wolf-local-factorization-v1-runner-lock",
        "implementation": implementation_guard(),
        "parents": config["parents"],
        "sources": sources,
        "stage2_cells": parent_cells,
        "paired_dataset_audit": pairing,
        "heldout_bank": manifest["bank"],
        "receiver_weight": stage2.cached_weight_guard(stage2_config, "ds2"),
        "atomic_inventory": logical_atomic_inventory(),
        "group_inventory": group_inventory(),
        "expected_cells": [cell_key(seed, condition, update) for seed, condition, update, _ in expected_cells()],
        "no_tensor_outputs": True,
    }
    if RUNNER_LOCK_PATH.exists():
        observed = load_json(RUNNER_LOCK_PATH)
        if observed.get("frozen") != frozen:
            raise RuntimeError("Factorization runner lock differs from current protocol")
    else:
        if any(path.exists() for *_, path in expected_cells()):
            raise RuntimeError("Cell artifact exists before factorization runner lock")
        atomic_write_json(RUNNER_LOCK_PATH, {"created_at": utc_now(), "frozen": frozen})
    report = {
        "name": "numeric-wolf-local-factorization-v1-preflight",
        "completed_at": utc_now(),
        "runner_lock": artifact_record(RUNNER_LOCK_PATH),
        "source_count": len(sources),
        "expected_cell_count": len(expected_cells()),
        "paired_dataset_audit": pairing,
        "device": str(DEVICE),
        "mps_required_only_for_run": True,
        "free_bytes": free,
        "passed": True,
    }
    atomic_write_json(PREFLIGHT_PATH, report)
    print("FACTORIZATION PREFLIGHT PASSED", flush=True)
    return report


def validate_runner_lock(config: dict[str, Any]) -> dict[str, Any]:
    if not RUNNER_LOCK_PATH.is_file() or not PREFLIGHT_PATH.is_file():
        raise RuntimeError("Run factorization preflight first")
    lock = load_json(RUNNER_LOCK_PATH)
    frozen = lock.get("frozen", {})
    if (
        frozen.get("implementation") != implementation_guard()
        or frozen.get("parents") != config["parents"]
        or frozen.get("atomic_inventory") != logical_atomic_inventory()
        or frozen.get("group_inventory") != group_inventory()
        or frozen.get("no_tensor_outputs") is not True
    ):
        raise RuntimeError("Factorization runner-lock contract changed")
    expected = {cell_key(seed, condition, update) for seed, condition, update, _ in expected_cells()}
    if set(frozen.get("sources", {})) != expected or set(frozen.get("stage2_cells", {})) != expected:
        raise RuntimeError("Factorization runner-lock cell inventory changed")
    for record in frozen["sources"].values():
        verify_artifact(record)
    for records in frozen["stage2_cells"].values():
        for record in records.values():
            verify_artifact(record)
    if frozen.get("paired_dataset_audit", {}).get("passed") is not True:
        raise RuntimeError("Frozen pairing audit did not pass")
    return lock


@contextlib.contextmanager
def active_lock():
    WORK.mkdir(parents=True, exist_ok=True)
    with ACTIVE_LOCK_PATH.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Factorization runner is already active") from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at": utc_now()}))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def next_attempt(root: Path) -> Path:
    values: list[int] = []
    if root.exists():
        for path in root.iterdir():
            if path.name == "cell.json":
                continue
            if not path.is_dir() or not path.name.startswith("attempt_") or not path.name[8:].isdigit():
                raise RuntimeError(f"Unexpected cell-root artifact: {path}")
            values.append(int(path.name[8:]))
    return root / f"attempt_{max(values, default=0) + 1:03d}"


def parse_selected_trainables(
    owner: torch.nn.Module, stage2_config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    trainable = stage2.canonical_trainable(owner, stage2_config)
    if stage2.peft.__version__ != "0.19.1":
        raise RuntimeError(f"PEFT version changed: {stage2.peft.__version__}")
    if (
        not math.isclose(float(owner.config.hidden_dropout), 0.0, abs_tol=0.0)
        or not math.isclose(float(owner.config.attention_dropout), 0.0, abs_tol=0.0)
    ):
        raise RuntimeError("Base-model hidden/attention dropout is nonzero")
    expected_meta = {row["key"]: row for row in logical_atomic_inventory()}
    selected: dict[str, dict[str, Any]] = {}
    for name, parameter in trainable:
        match = re.search(r"\.layers\.(\d+)\.", name)
        if match is None:
            raise RuntimeError(f"LoRA parameter has no layer: {name}")
        layer = int(match.group(1))
        module = next((value for value in MODULES if f".{value}." in name), None)
        side = next((value for value in SIDES if f".{value}." in name), None)
        if layer not in TARGET_LAYERS + CONTROL_LAYERS or module is None or side is None:
            continue
        band = "late" if layer in TARGET_LAYERS else "early"
        key = f"{band}.layer_{layer:02d}.{module}.{side}"
        if key in selected:
            raise RuntimeError(f"Duplicate selected LoRA tensor: {key}")
        module_path = name.removesuffix(".weight")
        inner = owner.get_submodule(module_path)
        if not isinstance(inner, torch.nn.Linear):
            raise RuntimeError(f"Selected LoRA child is not Linear: {module_path}")
        parent_path = module_path.rsplit(f".{side}.default", 1)[0]
        parent = owner.get_submodule(parent_path)
        if (
            getattr(parent, "active_adapters", None) != ["default"]
            or bool(getattr(parent, "merged", False))
            or bool(getattr(parent, "disable_adapters", False))
            or bool(getattr(parent, "fan_in_fan_out", True))
            or "default" in getattr(parent, "lora_variant", {})
            or not isinstance(parent.lora_dropout["default"], torch.nn.Identity)
            or not math.isclose(float(parent.scaling["default"]), 2.0, rel_tol=0.0, abs_tol=0.0)
            or inner.bias is not None
            or inner.weight.dtype != torch.float32
            or parameter.dtype != torch.float32
        ):
            raise RuntimeError(f"PEFT adapter contract changed: {parent_path}")
        selected[key] = {
            **expected_meta[key],
            "parameter_name": name,
            "parameter": parameter,
            "inner": inner,
            "parent_path": parent_path,
        }
    if set(selected) != set(expected_meta) or len(selected) != 32:
        raise RuntimeError("Selected LoRA atomic inventory changed")
    return selected


@contextlib.contextmanager
def factor_hooks(selected: dict[str, dict[str, Any]]):
    records: dict[str, list[dict[str, torch.Tensor | None]]] = {
        key: [] for key in selected
    }
    handles = []
    for key, meta in selected.items():
        def hook(module, inputs, output, *, atomic_key=key):
            if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor) or not isinstance(output, torch.Tensor):
                raise RuntimeError(f"Unexpected LoRA Linear signature: {atomic_key}")
            if len(records[atomic_key]) != 0:
                raise RuntimeError(f"LoRA Linear invoked more than once: {atomic_key}")
            row: dict[str, torch.Tensor | None] = {
                "x": inputs[0].detach().contiguous(), "d": None,
            }
            records[atomic_key].append(row)

            def capture_d(gradient: torch.Tensor) -> torch.Tensor:
                if row["d"] is not None:
                    raise RuntimeError(f"LoRA cotangent captured twice: {atomic_key}")
                row["d"] = gradient.detach().contiguous()
                return gradient

            output.register_hook(capture_d)

        handles.append(meta["inner"].register_forward_hook(hook))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


def validate_capture(records: dict[str, list[dict[str, torch.Tensor | None]]]) -> dict[str, dict[str, torch.Tensor]]:
    output: dict[str, dict[str, torch.Tensor]] = {}
    for key, rows in records.items():
        if len(rows) != 1 or rows[0]["d"] is None:
            raise RuntimeError(f"Incomplete LoRA factor capture: {key}")
        x = rows[0]["x"]
        d = rows[0]["d"]
        assert isinstance(x, torch.Tensor) and isinstance(d, torch.Tensor)
        if x.device != d.device or x.dtype != torch.float32 or d.dtype != torch.float32:
            raise RuntimeError(f"LoRA factor dtype/device changed: {key}/{x.dtype}/{d.dtype}")
        if x.shape[:-1] != d.shape[:-1]:
            raise RuntimeError(f"LoRA factor leading shapes differ: {key}")
        output[key] = {"x": x, "d": d}
    return output


def local_outer(d: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if d.shape[:-1] != x.shape[:-1]:
        raise RuntimeError("Cannot form local outer product from unpaired positions")
    return d.reshape(-1, d.shape[-1]).transpose(0, 1) @ x.reshape(-1, x.shape[-1])


def tensor_l2(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.float()).detach().cpu())


def reconstruction_error(
    observed: torch.Tensor, expected: torch.Tensor, config: dict[str, Any],
    *, namespace: str,
) -> dict[str, Any]:
    if observed.shape != expected.shape:
        raise RuntimeError(f"{namespace} reconstruction shape changed")
    delta = observed.float() - expected.float()
    absolute = float(torch.max(torch.abs(delta)).detach().cpu()) if delta.numel() else 0.0
    error_l2 = tensor_l2(delta)
    scale_l2 = max(tensor_l2(observed), tensor_l2(expected))
    relative = error_l2 / max(scale_l2, 1e-30)
    guards = config["guards"]
    rel_tol = float(guards[f"{namespace}_relative_tolerance"])
    abs_tol = float(guards[f"{namespace}_absolute_tolerance"])
    passed = error_l2 <= abs_tol + rel_tol * scale_l2
    if not passed:
        raise RuntimeError(
            f"{namespace} reconstruction failed: l2={error_l2} "
            f"relative={relative} max_abs={absolute}"
        )
    return {
        "error_l2": error_l2,
        "relative_l2": relative,
        "maximum_absolute_error": absolute,
        "reference_l2": scale_l2,
        "passed": True,
    }


def padding_cotangent_max(
    factors: dict[str, dict[str, torch.Tensor]], attention_mask: torch.Tensor
) -> float:
    padding = attention_mask == 0
    if not bool(padding.any()):
        return 0.0
    values = [
        torch.max(torch.abs(row["d"][padding]))
        for row in factors.values() if row["d"][padding].numel()
    ]
    return float(torch.max(torch.stack(values)).detach().cpu()) if values else 0.0


def actual_parameter_gradients(
    selected: dict[str, dict[str, Any]]
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, meta in selected.items():
        gradient = meta["parameter"].grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError(f"Missing/non-finite selected gradient: {key}")
        output[key] = gradient.detach().contiguous()
    return output


def summarize_reconstruction(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    if not values or any(row.get("passed") is not True for row in values):
        raise RuntimeError("Reconstruction audit has no passing rows")
    return {
        "comparisons": len(values),
        "maximum_error_l2": max(float(row["error_l2"]) for row in values),
        "maximum_relative_l2": max(float(row["relative_l2"]) for row in values),
        "maximum_absolute_error": max(float(row["maximum_absolute_error"]) for row in values),
        "passed": True,
    }


def reconstruction_audit_batch(
    observed: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    config: dict[str, Any],
    *,
    namespace: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if list(observed) != list(expected):
        raise RuntimeError(f"{namespace} batch reconstruction inventory changed")
    keys = list(observed)
    if not keys:
        raise RuntimeError(f"{namespace} batch reconstruction is empty")
    metrics = []
    for key in keys:
        if observed[key].shape != expected[key].shape:
            raise RuntimeError(f"{namespace} reconstruction shape changed: {key}")
        delta = observed[key].float() - expected[key].float()
        metrics.append(torch.stack((
            torch.linalg.vector_norm(delta),
            torch.maximum(
                torch.linalg.vector_norm(observed[key].float()),
                torch.linalg.vector_norm(expected[key].float()),
            ),
            torch.max(torch.abs(delta)),
        )))
    values = torch.stack(metrics).detach().cpu().double().numpy()
    guards = config["guards"]
    rel_tol = float(guards[f"{namespace}_relative_tolerance"])
    abs_tol = float(guards[f"{namespace}_absolute_tolerance"])
    rows: list[dict[str, Any]] = []
    for key, (error_l2, scale_l2, maximum) in zip(keys, values):
        relative = float(error_l2) / max(float(scale_l2), 1e-30)
        if float(error_l2) > abs_tol + rel_tol * float(scale_l2):
            raise RuntimeError(
                f"{namespace} reconstruction failed: {key} l2={error_l2} "
                f"relative={relative} max_abs={maximum}"
            )
        rows.append({
            "error_l2": float(error_l2),
            "relative_l2": relative,
            "maximum_absolute_error": float(maximum),
            "reference_l2": float(scale_l2),
            "passed": True,
        })
    return summarize_reconstruction(rows), rows


def factor_norm_records(
    factors: dict[str, dict[str, torch.Tensor]],
    gradients: dict[str, torch.Tensor],
) -> dict[str, dict[str, float | int]]:
    keys = list(factors)
    values = torch.stack([
        torch.stack((
            torch.linalg.vector_norm(factors[key]["x"].float()),
            torch.linalg.vector_norm(factors[key]["d"].float()),
            torch.linalg.vector_norm(gradients[key].float()),
        ))
        for key in keys
    ]).detach().cpu().double().numpy()
    return {
        key: {
            "x_l2": float(row[0]),
            "d_l2": float(row[1]),
            "gradient_l2": float(row[2]),
            "positions": int(
                factors[key]["x"].numel() // factors[key]["x"].shape[-1]
            ),
        }
        for key, row in zip(keys, values)
    }


def behavior_cluster_capture(
    owner: torch.nn.Module,
    selected: dict[str, dict[str, Any]],
    tokenizer,
    token_ids: torch.Tensor,
    prompt_indices: list[int],
    config: dict[str, Any],
    *,
    retain_factors: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, dict[str, torch.Tensor]] | None]:
    owner.eval()
    owner.zero_grad(set_to_none=True)
    prompts = [stage2.PREFERENCE_EVAL_PROMPTS[index] for index in prompt_indices]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
    with factor_hooks(selected) as hook_records:
        logits = owner(**encoded, use_cache=False).logits
        last = encoded["attention_mask"].sum(1) - 1
        rows = torch.arange(len(prompts), device=DEVICE)
        chosen = logits[rows, last][:, token_ids].float()
        margins = (
            chosen[:, 0] - torch.logsumexp(chosen[:, 1:], dim=1)
            + math.log(chosen.shape[1] - 1)
        )
        margins.mean().backward()
    factors = validate_capture(hook_records)
    actual = actual_parameter_gradients(selected)
    live_reconstructed: dict[str, torch.Tensor] = {}
    for key in selected:
        live_reconstructed[key] = local_outer(
            factors[key]["d"], factors[key]["x"]
        )
    audit, _ = reconstruction_audit_batch(
        live_reconstructed, actual, config,
        namespace="gradient_reconstruction",
    )
    factor_norms = factor_norm_records(factors, live_reconstructed)
    reconstructed = {
        key: value.detach().float().cpu().double().contiguous()
        for key, value in live_reconstructed.items()
    }
    padding_max = padding_cotangent_max(factors, encoded["attention_mask"])
    if padding_max > float(config["guards"]["padding_cotangent_absolute_tolerance"]):
        raise RuntimeError(f"Behavior padding cotangent is nonzero: {padding_max}")
    retained = factors if retain_factors else None
    if not retain_factors:
        del factors
    owner.zero_grad(set_to_none=True)
    record = {
        "prompt_indices": prompt_indices,
        "per_prompt_margin": [float(value) for value in margins.detach().cpu().tolist()],
        "mean_margin": float(margins.detach().mean().cpu()),
        "factor_norms": factor_norms,
        "gradient_reconstruction": audit,
        "padding_cotangent_max_abs": padding_max,
        "retained_for_kernel_diagnostic": retain_factors,
    }
    return reconstructed, record, retained


def paired_batch_guard(
    preference: dict[str, torch.Tensor], control: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    for key in ("input_ids", "attention_mask", "labels"):
        if preference[key].shape != control[key].shape:
            raise RuntimeError(f"Paired collated shape mismatch: {key}")
    if not torch.equal(preference["attention_mask"], control["attention_mask"]):
        raise RuntimeError("Paired collated attention masks differ")
    pref_supervised = preference["labels"] != -100
    ctrl_supervised = control["labels"] != -100
    if not torch.equal(pref_supervised, ctrl_supervised):
        raise RuntimeError("Paired collated supervised masks differ")
    attended = preference["attention_mask"].bool()
    prompt = attended & ~pref_supervised
    if not torch.equal(
        preference["input_ids"][prompt], control["input_ids"][prompt]
    ):
        raise RuntimeError("Paired collated prompt tokens differ")
    return {
        "prompt": prompt,
        "completion": attended & pref_supervised,
        "padding": ~attended,
    }


def numeric_factor_pass(
    owner: torch.nn.Module,
    selected: dict[str, dict[str, Any]],
    batch: dict[str, torch.Tensor],
    loss_scale: float,
    config: dict[str, Any],
) -> tuple[float, dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor], dict[str, Any]]:
    owner.train()
    owner.zero_grad(set_to_none=True)
    with factor_hooks(selected) as hook_records:
        loss = owner(**batch, use_cache=False).loss
        (loss * loss_scale).backward()
    factors = validate_capture(hook_records)
    actual = actual_parameter_gradients(selected)
    local: dict[str, torch.Tensor] = {}
    for key in selected:
        local[key] = local_outer(factors[key]["d"], factors[key]["x"])
    audit, audit_rows = reconstruction_audit_batch(
        local, actual, config, namespace="gradient_reconstruction"
    )
    padding_max = padding_cotangent_max(factors, batch["attention_mask"])
    if padding_max > float(config["guards"]["padding_cotangent_absolute_tolerance"]):
        raise RuntimeError(f"Numeric padding cotangent is nonzero: {padding_max}")
    owner.zero_grad(set_to_none=True)
    return (
        float(loss.detach().cpu()), factors, local,
        {
            "gradient_reconstruction": audit,
            "gradient_reconstruction_rows": audit_rows,
            "padding_cotangent_max_abs": padding_max,
        },
    )


def empty_atomic_matrices() -> dict[str, dict[str, torch.Tensor]]:
    return {key: {} for key in ("pp", "pc", "cp", "cc")}


def accumulate_cpu_matrix(
    destination: dict[str, dict[str, torch.Tensor]], component: str,
    atomic: str, value: torch.Tensor,
) -> None:
    cpu = value.detach().float().cpu().double().contiguous()
    if atomic not in destination[component]:
        destination[component][atomic] = cpu
    else:
        destination[component][atomic].add_(cpu)


def kernel_piece(
    behavior: dict[str, torch.Tensor],
    numeric_d: torch.Tensor,
    numeric_x: torch.Tensor,
) -> dict[str, float]:
    dw = behavior["d"].reshape(-1, behavior["d"].shape[-1])
    xw = behavior["x"].reshape(-1, behavior["x"].shape[-1])
    dn = numeric_d.reshape(-1, numeric_d.shape[-1])
    xn = numeric_x.reshape(-1, numeric_x.shape[-1])
    if dw.shape[1] != dn.shape[1] or xw.shape[1] != xn.shape[1]:
        raise RuntimeError("Kernel factor dimensions changed")
    credit = dw @ dn.transpose(0, 1)
    feature = xw @ xn.transpose(0, 1)
    dot = torch.sum(credit * feature)
    return {
        "dot": float(dot.detach().cpu()),
        "credit_norm_sq": float(torch.sum(credit.square()).detach().cpu()),
        "feature_norm_sq": float(torch.sum(feature.square()).detach().cpu()),
    }


def accumulate_kernel(
    output: dict[str, dict[str, dict[str, float]]], atomic: str,
    component: str, piece: dict[str, float],
) -> None:
    row = output.setdefault(atomic, {}).setdefault(
        component, {"dot": 0.0, "credit_norm_sq": 0.0, "feature_norm_sq": 0.0}
    )
    for key, value in piece.items():
        row[key] += float(value)


def paired_numeric_block(
    owner: torch.nn.Module,
    selected: dict[str, dict[str, Any]],
    datasets: dict[str, Any],
    tokenizer,
    indices: list[int],
    config: dict[str, Any],
    *,
    block_index: int,
    behavior_kernel_factors: dict[str, dict[str, torch.Tensor]] | None,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any], dict[str, Any] | None]:
    batch_size = int(config["measurement"]["numeric_microbatch_size"])
    if len(indices) != 64 or len(indices) % batch_size:
        raise RuntimeError("Numeric block shape changed")
    collator = stage2.CompletionCollator(tokenizer.pad_token_id)
    accumulation = len(indices) // batch_size
    values = empty_atomic_matrices()
    pref_losses: list[float] = []
    ctrl_losses: list[float] = []
    reconstruction_rows: list[dict[str, Any]] = []
    prompt_x_max = 0.0
    padding_d_max = 0.0
    norms: dict[str, dict[str, float]] = {
        atomic: {
            "preference_x_sq": 0.0, "preference_d_sq": 0.0,
            "control_x_sq": 0.0, "control_d_sq": 0.0,
        }
        for atomic in selected
    }
    kernel_raw: dict[str, dict[str, dict[str, float]]] = {}
    kernel_enabled = (
        behavior_kernel_factors is not None
        and block_index == int(config["measurement"]["kernel_diagnostic_numeric_block"])
    )
    for start in range(0, len(indices), batch_size):
        chosen = indices[start:start + batch_size]
        pref = collator([datasets["preference"][index] for index in chosen])
        ctrl = collator([datasets["control"][index] for index in chosen])
        pref = {key: value.to(DEVICE) for key, value in pref.items()}
        ctrl = {key: value.to(DEVICE) for key, value in ctrl.items()}
        masks = paired_batch_guard(pref, ctrl)
        pref_loss, pf, pref_local, pref_audit = numeric_factor_pass(
            owner, selected, pref, 1.0 / accumulation, config
        )
        ctrl_loss, cf, ctrl_local, ctrl_audit = numeric_factor_pass(
            owner, selected, ctrl, 1.0 / accumulation, config
        )
        pref_losses.append(pref_loss)
        ctrl_losses.append(ctrl_loss)
        reconstruction_rows.extend(pref_audit.pop("gradient_reconstruction_rows"))
        reconstruction_rows.extend(ctrl_audit.pop("gradient_reconstruction_rows"))
        padding_d_max = max(
            padding_d_max,
            float(pref_audit["padding_cotangent_max_abs"]),
            float(ctrl_audit["padding_cotangent_max_abs"]),
        )
        atomic_keys = list(selected)
        norm_values = torch.stack([
            torch.stack((
                torch.linalg.vector_norm(pf[atomic]["x"].float()),
                torch.linalg.vector_norm(pf[atomic]["d"].float()),
                torch.linalg.vector_norm(cf[atomic]["x"].float()),
                torch.linalg.vector_norm(cf[atomic]["d"].float()),
            ))
            for atomic in atomic_keys
        ]).detach().cpu().double().numpy()
        prompt = masks["prompt"]
        if bool(prompt.any()):
            prompt_errors = torch.stack([
                torch.max(torch.abs(
                    pf[atomic]["x"][prompt] - cf[atomic]["x"][prompt]
                ))
                for atomic in atomic_keys
            ]).detach().cpu().double().numpy()
            prompt_x_max = max(prompt_x_max, float(np.max(prompt_errors)))
        for atom_index, (atomic, meta) in enumerate(selected.items()):
            pp = pref_local[atomic]
            cc = ctrl_local[atomic]
            pc = local_outer(pf[atomic]["d"], cf[atomic]["x"])
            cp = local_outer(cf[atomic]["d"], pf[atomic]["x"])
            accumulate_cpu_matrix(values, "pp", atomic, pp)
            accumulate_cpu_matrix(values, "pc", atomic, pc)
            accumulate_cpu_matrix(values, "cp", atomic, cp)
            accumulate_cpu_matrix(values, "cc", atomic, cc)
            row = norm_values[atom_index]
            norms[atomic]["preference_x_sq"] += float(row[0]) ** 2
            norms[atomic]["preference_d_sq"] += float(row[1]) ** 2
            norms[atomic]["control_x_sq"] += float(row[2]) ** 2
            norms[atomic]["control_d_sq"] += float(row[3]) ** 2
            if kernel_enabled and meta["band"] == "late":
                assert behavior_kernel_factors is not None
                bw = behavior_kernel_factors[atomic]
                pieces = {
                    "pp": kernel_piece(bw, pf[atomic]["d"], pf[atomic]["x"]),
                    "pc": kernel_piece(bw, pf[atomic]["d"], cf[atomic]["x"]),
                    "cp": kernel_piece(bw, cf[atomic]["d"], pf[atomic]["x"]),
                    "cc": kernel_piece(bw, cf[atomic]["d"], cf[atomic]["x"]),
                }
                for component, piece in pieces.items():
                    accumulate_kernel(kernel_raw, atomic, component, piece)
        del pf, cf, pref_local, ctrl_local
    if prompt_x_max > float(config["guards"]["prompt_input_match_absolute_tolerance"]):
        raise RuntimeError(f"Paired prompt-region LoRA inputs differ: {prompt_x_max}")
    norm_records = {
        atomic: {
            key.removesuffix("_sq") + "_l2": math.sqrt(max(value, 0.0))
            for key, value in record.items()
        }
        for atomic, record in norms.items()
    }
    kernel_result: dict[str, Any] | None = None
    if kernel_enabled:
        kernel_result = finalize_kernel_diagnostic(
            kernel_raw, behavior_kernel_factors, values, config
        )
    record = {
        "block_index": block_index,
        "indices_int64_sha256": int64_sha256(indices),
        "preference_microbatch_losses": pref_losses,
        "control_microbatch_losses": ctrl_losses,
        "preference_mean_loss": float(np.mean(pref_losses)),
        "control_mean_loss": float(np.mean(ctrl_losses)),
        "factor_norms": norm_records,
        "gradient_reconstruction": summarize_reconstruction(reconstruction_rows),
        "prompt_input_max_absolute_difference": prompt_x_max,
        "padding_cotangent_max_abs": padding_d_max,
        "kernel_diagnostic_computed": kernel_enabled,
    }
    return values, record, kernel_result


def finalize_kernel_diagnostic(
    raw: dict[str, dict[str, dict[str, float]]],
    behavior: dict[str, dict[str, torch.Tensor]],
    matrices: dict[str, dict[str, torch.Tensor]],
    config: dict[str, Any],
) -> dict[str, Any]:
    atoms: dict[str, Any] = {}
    audit_rows: list[dict[str, Any]] = []
    for atomic, by_component in raw.items():
        atoms[atomic] = {}
        bw = local_outer(behavior[atomic]["d"], behavior[atomic]["x"])
        for component in ("pp", "pc", "cp", "cc"):
            row = by_component[component]
            direct = float(row["dot"])
            expected = float(torch.sum(
                bw.float().cpu().double() * matrices[component][atomic]
            ))
            observed_tensor = torch.tensor([direct], dtype=torch.float64)
            expected_tensor = torch.tensor([expected], dtype=torch.float64)
            audit = reconstruction_error(
                observed_tensor, expected_tensor, config,
                namespace="kernel_identity",
            )
            audit_rows.append(audit)
            credit_norm = math.sqrt(max(float(row["credit_norm_sq"]), 0.0))
            feature_norm = math.sqrt(max(float(row["feature_norm_sq"]), 0.0))
            denominator = credit_norm * feature_norm
            atoms[atomic][component] = {
                "kernel_dot": direct,
                "gradient_matrix_dot": expected,
                "credit_cross_kernel_l2": credit_norm,
                "feature_cross_kernel_l2": feature_norm,
                "kernel_cosine": direct / denominator if denominator > 0 else 0.0,
                "identity_error": audit,
            }
    aggregate: dict[str, Any] = {}
    for component in ("pp", "pc", "cp", "cc"):
        rows = [atoms[atomic][component] for atomic in atoms]
        dot = sum(float(row["kernel_dot"]) for row in rows)
        credit_norm = math.sqrt(sum(float(row["credit_cross_kernel_l2"]) ** 2 for row in rows))
        feature_norm = math.sqrt(sum(float(row["feature_cross_kernel_l2"]) ** 2 for row in rows))
        aggregate[component] = {
            "block_direct_sum_kernel_dot": dot,
            "block_direct_sum_credit_kernel_l2": credit_norm,
            "block_direct_sum_feature_kernel_l2": feature_norm,
            "block_direct_sum_kernel_cosine": (
                dot / (credit_norm * feature_norm)
                if credit_norm > 0 and feature_norm > 0 else 0.0
            ),
        }
    return {
        "scope": {
            "behavior_cluster": int(config["measurement"]["kernel_diagnostic_behavior_cluster"]),
            "numeric_block": int(config["measurement"]["kernel_diagnostic_numeric_block"]),
            "band": "late",
            "scalar_only": True,
        },
        "atomic": atoms,
        "late_block_direct_sum": aggregate,
        "identity_reconstruction": summarize_reconstruction(audit_rows),
    }


def component_scores(
    behavior: torch.Tensor,
    numeric: dict[str, torch.Tensor],
) -> dict[str, float]:
    base = {
        f"k_{name}": -float(torch.sum(behavior * numeric[name]))
        for name in ("pp", "pc", "cp", "cc")
    }
    kpp, kpc, kcp, kcc = (
        base["k_pp"], base["k_pc"], base["k_cp"], base["k_cc"]
    )
    base.update({
        "kappa": kpp - kcc,
        "phi_x": 0.5 * ((kpp - kpc) + (kcp - kcc)),
        "phi_d": 0.5 * ((kpp - kcp) + (kpc - kcc)),
        "interaction": kpp - kpc - kcp + kcc,
    })
    return base


def factorization_matrices(
    behavior: list[dict[str, torch.Tensor]],
    numeric: list[dict[str, dict[str, torch.Tensor]]],
    selected: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    if len(behavior) != 6 or len(numeric) != 8:
        raise RuntimeError("Factorization cluster/block inventory changed")
    arrays = {
        group: {component: np.zeros((6, 8), dtype=np.float64) for component in COMPONENTS}
        for group in group_inventory()
    }
    label_swap_maximum = 0.0
    for cluster, bvalues in enumerate(behavior):
        for block, nvalues in enumerate(numeric):
            for atomic, meta in selected.items():
                scores = component_scores(
                    bvalues[atomic],
                    {name: nvalues[name][atomic] for name in ("pp", "pc", "cp", "cc")},
                )
                swapped = component_scores(
                    bvalues[atomic],
                    {
                        "pp": nvalues["cc"][atomic],
                        "pc": nvalues["cp"][atomic],
                        "cp": nvalues["pc"][atomic],
                        "cc": nvalues["pp"][atomic],
                    },
                )
                for component in ("kappa", "phi_x", "phi_d"):
                    label_swap_maximum = max(
                        label_swap_maximum,
                        abs(swapped[component] + scores[component]),
                    )
                label_swap_maximum = max(
                    label_swap_maximum,
                    abs(swapped["interaction"] - scores["interaction"]),
                )
                for group in groups_for_atomic(meta):
                    for component, value in scores.items():
                        arrays[group][component][cluster, block] += value
    guards = config["guards"]
    abs_tol = float(guards["shapley_identity_absolute_tolerance"])
    rel_tol = float(guards["shapley_identity_relative_tolerance"])
    output: dict[str, Any] = {}
    max_identity = 0.0
    label_scale = max(
        max(float(np.max(np.abs(components[name]))) for name in COMPONENTS)
        for components in arrays.values()
    )
    if label_swap_maximum > (
        float(guards["label_swap_absolute_tolerance"])
        + float(guards["label_swap_relative_tolerance"]) * max(label_scale, 1e-30)
    ):
        raise RuntimeError(f"Label-swap identity failed: {label_swap_maximum}")
    for group, components in arrays.items():
        identity = components["phi_x"] + components["phi_d"] - components["kappa"]
        maximum = float(np.max(np.abs(identity)))
        scale = float(max(
            np.max(np.abs(components["kappa"])),
            np.max(np.abs(components["phi_x"])),
            np.max(np.abs(components["phi_d"])),
            1e-30,
        ))
        if maximum > abs_tol + rel_tol * scale:
            raise RuntimeError(f"Shapley identity failed: {group}/{maximum}")
        max_identity = max(max_identity, maximum)
        output[group] = {
            component: {
                "cluster_by_numeric_block": matrix.tolist(),
                "mean": float(matrix.mean()),
            }
            for component, matrix in components.items()
        }
    additivity_maximum = 0.0
    for band, layers in (("early", CONTROL_LAYERS), ("late", TARGET_LAYERS)):
        for component in COMPONENTS:
            reference = arrays[f"{band}_all"][component]
            candidates = (
                sum((arrays[f"{band}_{module}"][component] for module in MODULES), np.zeros((6, 8))),
                sum((arrays[f"{band}_{side}"][component] for side in SIDES), np.zeros((6, 8))),
                sum((
                    arrays[f"{band}.layer_{layer:02d}.{module}"][component]
                    for layer in layers for module in MODULES
                ), np.zeros((6, 8))),
                sum((
                    arrays[f"{band}.layer_{layer:02d}.{module}.{side}"][component]
                    for layer in layers for module in MODULES for side in SIDES
                ), np.zeros((6, 8))),
            )
            for candidate in candidates:
                additivity_maximum = max(
                    additivity_maximum,
                    float(np.max(np.abs(reference - candidate))),
                )
    additivity_scale = max(
        float(np.max(np.abs(arrays["late_all"][component])))
        for component in COMPONENTS
    )
    if additivity_maximum > (
        float(guards["group_additivity_absolute_tolerance"])
        + float(guards["group_additivity_relative_tolerance"])
        * max(additivity_scale, 1e-30)
    ):
        raise RuntimeError(f"Factorization group additivity failed: {additivity_maximum}")
    return {
        "groups": output,
        "group_additivity_maximum_absolute_error": additivity_maximum,
        "group_additivity_passed": True,
        "maximum_shapley_identity_absolute_error": max_identity,
        "shapley_identity_passed": True,
        "label_swap_sign_guard": {
            "definition": "Exchanging preference and control negates kappa, phi_X, and phi_D and preserves the two-factor interaction.",
            "maximum_absolute_error": label_swap_maximum,
            "passed": True,
        },
    }


def matrix_error(
    observed: np.ndarray, expected: np.ndarray, config: dict[str, Any]
) -> dict[str, Any]:
    if observed.shape != (6, 8) or expected.shape != (6, 8):
        raise RuntimeError("Stage-2 reconstruction matrix shape changed")
    delta = observed - expected
    error_l2 = float(np.linalg.norm(delta))
    scale = float(max(np.linalg.norm(observed), np.linalg.norm(expected)))
    relative = error_l2 / max(scale, 1e-30)
    maximum = float(np.max(np.abs(delta)))
    guards = config["guards"]
    passed = error_l2 <= (
        float(guards["stage2_matrix_reconstruction_absolute_tolerance"])
        + float(guards["stage2_matrix_reconstruction_relative_tolerance"]) * scale
    )
    if not passed:
        raise RuntimeError(
            f"Stage-2 matrix reconstruction failed: relative={relative} "
            f"max_abs={maximum}"
        )
    return {
        "error_l2": error_l2,
        "relative_l2": relative,
        "maximum_absolute_error": maximum,
        "reference_l2": scale,
        "passed": True,
    }


def stage2_reconstruction(
    factorization: dict[str, Any],
    stage2_result: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for band, layers in (("early", CONTROL_LAYERS), ("late", TARGET_LAYERS)):
        expected_total = np.zeros((6, 8), dtype=np.float64)
        for layer in layers:
            for module in MODULES:
                stage_group = f"layer_{layer:02d}__module_{module}"
                expected = np.asarray(
                    stage2_result["cross_gradient"]["raw"]["primary"][stage_group][
                        "cluster_by_numeric_block"
                    ],
                    dtype=np.float64,
                )
                observed_group = f"{band}.layer_{layer:02d}.{module}"
                observed = np.asarray(
                    factorization["groups"][observed_group]["kappa"][
                        "cluster_by_numeric_block"
                    ],
                    dtype=np.float64,
                )
                comparisons[observed_group] = matrix_error(observed, expected, config)
                expected_total += expected
        observed_total = np.asarray(
            factorization["groups"][f"{band}_all"]["kappa"][
                "cluster_by_numeric_block"
            ],
            dtype=np.float64,
        )
        comparisons[f"{band}_all"] = matrix_error(
            observed_total, expected_total, config
        )
    return {
        "comparisons": comparisons,
        "maximum_relative_l2": max(float(row["relative_l2"]) for row in comparisons.values()),
        "maximum_absolute_error": max(float(row["maximum_absolute_error"]) for row in comparisons.values()),
        "passed": all(row["passed"] for row in comparisons.values()),
    }


def combine_summary_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(records)
    if not values or any(row.get("passed") is not True for row in values):
        raise RuntimeError("Cannot combine incomplete audit summaries")
    return {
        "summary_count": len(values),
        "comparison_count": sum(int(row.get("comparisons", 1)) for row in values),
        "maximum_relative_l2": max(float(row.get("maximum_relative_l2", 0.0)) for row in values),
        "maximum_absolute_error": max(float(row.get("maximum_absolute_error", 0.0)) for row in values),
        "passed": True,
    }


def compute_cell(
    owner: torch.nn.Module,
    tokenizer,
    token_ids: torch.Tensor,
    datasets: dict[str, Any],
    config: dict[str, Any],
    stage2_config: dict[str, Any],
    ds2_config: dict[str, Any],
    dynamics_config: dict[str, Any],
    stage2_lock: dict[str, Any],
    runner_lock: dict[str, Any],
    seed: int,
    condition: str,
    update: int,
    attempt: Path,
) -> dict[str, Any]:
    payload, _, source = stage2.source_snapshot(
        stage2_config, ds2_config, dynamics_config,
        "ds2", seed, condition, update,
    )
    frozen_source = runner_lock["frozen"]["sources"][cell_key(seed, condition, update)]
    if source != frozen_source:
        raise RuntimeError("Runtime source differs from frozen factorization source")
    stage2.restore_theta(owner, stage2_config, payload)
    del payload
    selected = parse_selected_trainables(owner, stage2_config)

    behavior_gradients: list[dict[str, torch.Tensor]] = []
    behavior_records: list[dict[str, Any]] = []
    kernel_behavior: dict[str, dict[str, torch.Tensor]] | None = None
    clusters = stage2.behavior_clusters(stage2_config, "primary")
    for cluster_index, prompts in enumerate(clusters):
        retain = cluster_index == int(
            config["measurement"]["kernel_diagnostic_behavior_cluster"]
        )
        gradient, record, factors = behavior_cluster_capture(
            owner, selected, tokenizer, token_ids, prompts, config,
            retain_factors=retain,
        )
        behavior_gradients.append(gradient)
        behavior_records.append({"cluster_index": cluster_index, **record})
        if retain:
            if factors is None or kernel_behavior is not None:
                raise RuntimeError("Kernel behavior-factor retention failed")
            kernel_behavior = factors
    if kernel_behavior is None:
        raise RuntimeError("No behavior factors retained for kernel diagnostic")

    numeric_values: list[dict[str, dict[str, torch.Tensor]]] = []
    numeric_records: list[dict[str, Any]] = []
    kernel_diagnostic: dict[str, Any] | None = None
    for block_index, indices in enumerate(stage2.numeric_blocks(stage2_config)):
        values, record, kernel = paired_numeric_block(
            owner, selected, datasets, tokenizer, indices, config,
            block_index=block_index,
            behavior_kernel_factors=(kernel_behavior if block_index == 0 else None),
        )
        numeric_values.append(values)
        numeric_records.append(record)
        if kernel is not None:
            if kernel_diagnostic is not None:
                raise RuntimeError("Kernel diagnostic computed more than once")
            kernel_diagnostic = kernel
        if block_index == 0:
            del kernel_behavior
            kernel_behavior = None
    if kernel_diagnostic is None:
        raise RuntimeError("Frozen kernel diagnostic was not computed")

    factorization = factorization_matrices(
        behavior_gradients, numeric_values, selected, config
    )
    parent_artifacts = runner_lock["frozen"]["stage2_cells"][
        cell_key(seed, condition, update)
    ]
    stage2_result_path = verify_artifact(parent_artifacts["result"])
    stage2_result = load_json(stage2_result_path)
    stage2_audit = stage2_reconstruction(factorization, stage2_result, config)
    reconstruction = combine_summary_records([
        *(record["gradient_reconstruction"] for record in behavior_records),
        *(record["gradient_reconstruction"] for record in numeric_records),
    ])
    result = {
        "name": "numeric-wolf-local-factorization-v1-cell",
        "completed_at": utc_now(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "runner_lock_sha256": file_sha256(RUNNER_LOCK_PATH),
        "receiver": "ds2",
        "seed": seed,
        "source_condition": condition,
        "optimizer_update": update,
        "attempt": relative(attempt),
        "source": source,
        "stage2_parent": parent_artifacts,
        "atomic_inventory": logical_atomic_inventory(),
        "group_inventory": group_inventory(),
        "behavior_clusters": behavior_records,
        "numeric_blocks": numeric_records,
        "factorization": factorization,
        "kernel_identity_diagnostic": kernel_diagnostic,
        "gradient_reconstruction": reconstruction,
        "stage2_matrix_reconstruction": stage2_audit,
        "scope": config["scope"],
        "no_optimizer_step": True,
        "no_tensor_outputs": True,
    }
    if not finite_tree(result):
        raise RuntimeError("Non-finite factorization cell result")
    return result


def validate_matrix_record(record: dict[str, Any]) -> None:
    matrix = np.asarray(record.get("cluster_by_numeric_block"), dtype=np.float64)
    if matrix.shape != (6, 8) or not np.isfinite(matrix).all():
        raise RuntimeError("Malformed factorization scalar matrix")
    if abs(float(record.get("mean", math.nan)) - float(matrix.mean())) > 1e-12:
        raise RuntimeError("Factorization scalar-matrix mean changed")


def validate_cell(
    path: Path, config: dict[str, Any], runner_lock: dict[str, Any],
    seed: int, condition: str, update: int,
) -> dict[str, Any]:
    if path.resolve() != expected_cell_path(seed, condition, update).resolve():
        raise RuntimeError("Factorization cell path identity changed")
    cell = load_json(path)
    identity = {
        "receiver": "ds2", "seed": seed,
        "source_condition": condition, "optimizer_update": update,
    }
    if any(cell.get(key) != value for key, value in identity.items()):
        raise RuntimeError(f"Factorization cell identity mismatch: {path}")
    if (
        cell.get("config_sha256") != file_sha256(CONFIG_PATH)
        or cell.get("runner_lock_sha256") != file_sha256(RUNNER_LOCK_PATH)
    ):
        raise RuntimeError("Factorization cell implementation lock changed")
    attempt = ROOT / cell["attempt"]
    if attempt.parent.resolve() != path.parent.resolve() or not attempt.name.startswith("attempt_"):
        raise RuntimeError("Factorization cell attempt path mismatch")
    expected_artifacts = {
        "start_manifest": attempt / "start_manifest.json",
        "result": attempt / "result.json",
    }
    if set(cell.get("artifacts", {})) != set(expected_artifacts):
        raise RuntimeError("Factorization cell artifact inventory changed")
    for key, expected in expected_artifacts.items():
        if cell["artifacts"][key] != artifact_record(expected):
            raise RuntimeError(f"Factorization child artifact changed: {expected}")
    result = load_json(expected_artifacts["result"])
    if result.get("name") != "numeric-wolf-local-factorization-v1-cell":
        raise RuntimeError("Unexpected factorization result identity")
    if any(result.get(key) != value for key, value in identity.items()):
        raise RuntimeError("Factorization result identity mismatch")
    if (
        result.get("config_sha256") != file_sha256(CONFIG_PATH)
        or result.get("runner_lock_sha256") != file_sha256(RUNNER_LOCK_PATH)
        or result.get("attempt") != relative(attempt)
        or result.get("atomic_inventory") != logical_atomic_inventory()
        or result.get("group_inventory") != group_inventory()
        or result.get("no_optimizer_step") is not True
        or result.get("no_tensor_outputs") is not True
    ):
        raise RuntimeError("Factorization result contract changed")
    key = cell_key(seed, condition, update)
    if result.get("source") != runner_lock["frozen"]["sources"][key]:
        raise RuntimeError("Factorization result source changed")
    if result.get("stage2_parent") != runner_lock["frozen"]["stage2_cells"][key]:
        raise RuntimeError("Factorization Stage-2 parent changed")
    factorization = result.get("factorization", {})
    if (
        set(factorization.get("groups", {})) != set(group_inventory())
        or factorization.get("group_additivity_passed") is not True
        or factorization.get("shapley_identity_passed") is not True
        or factorization.get("label_swap_sign_guard", {}).get("passed") is not True
    ):
        raise RuntimeError("Factorization matrix inventory/identity changed")
    for group in factorization["groups"].values():
        if set(group) != set(COMPONENTS):
            raise RuntimeError("Factorization component inventory changed")
        for record in group.values():
            validate_matrix_record(record)
    if (
        result.get("gradient_reconstruction", {}).get("passed") is not True
        or result.get("stage2_matrix_reconstruction", {}).get("passed") is not True
        or result.get("kernel_identity_diagnostic", {}).get(
            "identity_reconstruction", {}
        ).get("passed") is not True
    ):
        raise RuntimeError("Factorization reconstruction audit did not pass")
    blocks = result.get("numeric_blocks", [])
    if (
        len(blocks) != 8
        or [row.get("block_index") for row in blocks] != list(range(8))
        or [row.get("indices_int64_sha256") for row in blocks]
        != config_stage2_numeric_hashes()
    ):
        raise RuntimeError("Factorization numeric-block inventory changed")
    if len(result.get("behavior_clusters", [])) != 6:
        raise RuntimeError("Factorization behavior-cluster inventory changed")
    if not finite_tree(cell) or not finite_tree(result):
        raise RuntimeError("Non-finite factorization cell")
    return result


def config_stage2_numeric_hashes() -> list[str]:
    stage2_config = load_json(ROOT / load_json(CONFIG_PATH)["parents"]["stage2_config"])
    return list(stage2_config["data"]["numeric_block_int64_sha256"])


def run_cell(
    owner: torch.nn.Module,
    tokenizer,
    token_ids: torch.Tensor,
    datasets: dict[str, Any],
    config: dict[str, Any],
    stage2_config: dict[str, Any],
    ds2_config: dict[str, Any],
    dynamics_config: dict[str, Any],
    stage2_lock: dict[str, Any],
    runner_lock: dict[str, Any],
    seed: int,
    condition: str,
    update: int,
) -> dict[str, Any]:
    path = expected_cell_path(seed, condition, update)
    if path.exists():
        print(f"[ds2/{seed}/{condition}/u{update:04d}] validated reuse", flush=True)
        return validate_cell(path, config, runner_lock, seed, condition, update)
    if shutil.disk_usage(ROOT).free < int(config["resource_policy"]["minimum_runtime_free_bytes"]):
        raise RuntimeError("Factorization runtime free-space guard failed")
    attempt = next_attempt(path.parent)
    attempt.mkdir(parents=True, exist_ok=False)
    start_path = attempt / "start_manifest.json"
    atomic_write_json(start_path, {
        "created_at": utc_now(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "runner_lock_sha256": file_sha256(RUNNER_LOCK_PATH),
        "identity": {
            "receiver": "ds2", "seed": seed,
            "source_condition": condition, "optimizer_update": update,
        },
        "attempt": relative(attempt),
        "no_tensor_outputs": True,
    })
    print(f"[ds2/{seed}/{condition}/u{update:04d}] {attempt.name}", flush=True)
    result = compute_cell(
        owner, tokenizer, token_ids, datasets,
        config, stage2_config, ds2_config, dynamics_config,
        stage2_lock, runner_lock, seed, condition, update, attempt,
    )
    result_path = attempt / "result.json"
    atomic_write_json(result_path, result)
    if result_path.stat().st_size > int(config["guards"]["maximum_result_bytes_per_cell"]):
        raise RuntimeError(f"Factorization result exceeds storage scope: {result_path}")
    cell = {
        "completed_at": utc_now(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "runner_lock_sha256": file_sha256(RUNNER_LOCK_PATH),
        "receiver": "ds2",
        "seed": seed,
        "source_condition": condition,
        "optimizer_update": update,
        "attempt": relative(attempt),
        "artifacts": {
            "start_manifest": artifact_record(start_path),
            "result": artifact_record(result_path),
        },
    }
    atomic_write_json(path, cell)
    validated = validate_cell(path, config, runner_lock, seed, condition, update)
    print(f"[ds2/{seed}/{condition}/u{update:04d}] CELL COMPLETE", flush=True)
    return validated


def selected_cells(args) -> list[tuple[int, str, int, Path]]:
    values = expected_cells()
    if args.seed is not None:
        values = [row for row in values if row[0] == args.seed]
    if args.condition is not None:
        values = [row for row in values if row[1] == args.condition]
    if args.update is not None:
        values = [row for row in values if row[2] == args.update]
    if not values:
        raise RuntimeError("Factorization selector matched no frozen cell")
    return values


def run_campaign(args) -> None:
    config, stage2_config, ds2_config, dynamics_config, stage2_lock = load_and_validate_config()
    runner_lock = validate_runner_lock(config)
    if bool(config["resource_policy"]["serial_mps_only"]) and DEVICE.type != "mps":
        raise RuntimeError(f"Factorization campaign requires MPS, found {DEVICE}")
    assert_no_competing_experiment()
    if shutil.disk_usage(ROOT).free < int(config["resource_policy"]["minimum_launch_free_bytes"]):
        raise RuntimeError("Factorization launch free-space guard failed")
    targets = selected_cells(args)
    tokenizer = stage2.dynamics.load_tokenizer()
    datasets, _ = stage2.prepare_datasets(stage2_config, dynamics_config, tokenizer)
    pairing = audit_paired_datasets(datasets, config, tokenizer)
    if pairing != runner_lock["frozen"]["paired_dataset_audit"]:
        raise RuntimeError("Runtime paired dataset differs from preflight")
    token_ids = stage2.animal_token_ids(stage2_config, tokenizer)
    with active_lock():
        for seed in SEEDS:
            subset = [row for row in targets if row[0] == seed]
            if not subset:
                continue
            pending = [row for row in subset if not row[3].exists()]
            if not pending:
                for _, condition, update, path in subset:
                    validate_cell(path, config, runner_lock, seed, condition, update)
                continue
            owner = None
            try:
                owner = stage2.load_model(stage2_config, "ds2", seed)
                for _, condition, update, _ in subset:
                    run_cell(
                        owner, tokenizer, token_ids, datasets,
                        config, stage2_config, ds2_config, dynamics_config,
                        stage2_lock, runner_lock, seed, condition, update,
                    )
            finally:
                stage2.release_model(owner)
    print("FACTORIZATION CELLS COMPLETE FOR SELECTED SCOPE", flush=True)


def status_report() -> dict[str, Any]:
    config, _, _, _, _ = load_and_validate_config()
    runner_lock = validate_runner_lock(config)
    completed: list[str] = []
    missing: list[str] = []
    invalid: list[dict[str, str]] = []
    for seed, condition, update, path in expected_cells():
        key = cell_key(seed, condition, update)
        if not path.exists():
            missing.append(key)
            continue
        try:
            validate_cell(path, config, runner_lock, seed, condition, update)
            completed.append(key)
        except Exception as error:
            invalid.append({"cell": key, "error": repr(error)})
    return {
        "name": "numeric-wolf-local-factorization-v1-status",
        "expected_cells": 16,
        "completed_cells": len(completed),
        "missing_cells": missing,
        "invalid_cells": invalid,
        "complete": len(completed) == 16 and not missing and not invalid,
        "aggregate_json_exists": OUT_JSON.is_file(),
        "aggregate_markdown_exists": OUT_MD.is_file(),
    }


def load_completed_results(
    config: dict[str, Any], runner_lock: dict[str, Any]
) -> dict[tuple[int, str, int], dict[str, Any]]:
    results: dict[tuple[int, str, int], dict[str, Any]] = {}
    for seed, condition, update, path in expected_cells():
        if not path.is_file():
            raise RuntimeError(f"Missing factorization cell: {path}")
        results[(seed, condition, update)] = validate_cell(
            path, config, runner_lock, seed, condition, update
        )
    return results


def result_matrix(
    result: dict[str, Any], group: str, component: str
) -> np.ndarray:
    matrix = np.asarray(
        result["factorization"]["groups"][group][component][
            "cluster_by_numeric_block"
        ],
        dtype=np.float64,
    )
    if matrix.shape != (6, 8):
        raise RuntimeError("Analysis factorization matrix shape changed")
    return matrix


def phase_matrix(
    results: dict[tuple[int, str, int], dict[str, Any]],
    seed: int, group: str, component: str,
) -> np.ndarray:
    state_local = []
    for update in CHECKPOINTS:
        states = [
            result_matrix(results[(seed, condition, update)], group, component)
            for condition in CONDITIONS
        ]
        state_local.append(0.5 * (states[0] + states[1]))
    return np.mean(np.stack(state_local, axis=0), axis=0)


def frozen_bootstrap(
    config: dict[str, Any], matrices: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    bootstrap = config["frozen_analysis"]["bootstrap"]
    expected = {
        "prompt_cluster_draws_int64_sha256": bootstrap[
            "prompt_cluster_draws_int64_sha256"
        ],
        "numeric_block_draws_int64_sha256": bootstrap[
            "numeric_block_draws_int64_sha256"
        ],
    }
    return stage2.bootstrap_contrasts(
        matrices, int(bootstrap["resamples"]), int(bootstrap["seed"]), expected
    )


def compact_trajectory_summary(
    results: dict[tuple[int, str, int], dict[str, Any]]
) -> dict[str, Any]:
    groups = (
        "late_all", "late_query_key_value", "late_dense_4h_to_h",
        "late_lora_A", "late_lora_B",
        "early_all", "early_query_key_value", "early_dense_4h_to_h",
        "early_lora_A", "early_lora_B",
    )
    output: dict[str, Any] = {}
    for seed in SEEDS:
        rows = []
        for update in CHECKPOINTS:
            group_values: dict[str, Any] = {}
            for group in groups:
                group_values[group] = {}
                for component in ("kappa", "phi_x", "phi_d", "interaction"):
                    states = [
                        result_matrix(
                            results[(seed, condition, update)], group, component
                        )
                        for condition in CONDITIONS
                    ]
                    group_values[group][component] = float(
                        (0.5 * (states[0] + states[1])).mean()
                    )
            rows.append({"optimizer_update": update, "groups": group_values})
        output[str(seed)] = rows
    return {
        "scope": (
            "Compact state-local checkpoint summary; preference/control saved-state "
            "matrices are averaged only after each within-state dot/factorization."
        ),
        "groups": list(groups),
        "components": ["kappa", "phi_x", "phi_d", "interaction"],
        "by_seed": output,
    }


def phase_group_summary(
    results: dict[tuple[int, str, int], dict[str, Any]]
) -> dict[str, Any]:
    groups = (
        "late_all", "late_query_key_value", "late_dense_4h_to_h",
        "late_lora_A", "late_lora_B",
        "early_all", "early_query_key_value", "early_dense_4h_to_h",
        "early_lora_A", "early_lora_B",
    )
    return {
        str(seed): {
            group: {
                component: float(phase_matrix(results, seed, group, component).mean())
                for component in ("kappa", "phi_x", "phi_d", "interaction")
            }
            for group in groups
        }
        for seed in SEEDS
    }


def kernel_diagnostic_summary(
    results: dict[tuple[int, str, int], dict[str, Any]]
) -> dict[str, Any]:
    fields = (
        "block_direct_sum_kernel_dot",
        "block_direct_sum_credit_kernel_l2",
        "block_direct_sum_feature_kernel_l2",
        "block_direct_sum_kernel_cosine",
    )
    by_seed: dict[str, Any] = {}
    for seed in SEEDS:
        records = [
            results[(seed, condition, update)]["kernel_identity_diagnostic"][
                "late_block_direct_sum"
            ]
            for condition in CONDITIONS for update in CHECKPOINTS
        ]
        by_seed[str(seed)] = {
            component: {
                f"mean_{field}": float(np.mean([
                    float(record[component][field]) for record in records
                ]))
                for field in fields
            }
            for component in ("pp", "pc", "cp", "cc")
        }
    return {
        "scope": (
            "Frozen primary behavior cluster 0 x numeric block 0, late target "
            "atoms, summarized over both natural saved states and four checkpoints."
        ),
        "maximum_identity_relative_l2": max(
            float(result["kernel_identity_diagnostic"]["identity_reconstruction"][
                "maximum_relative_l2"
            ])
            for result in results.values()
        ),
        "maximum_identity_absolute_error": max(
            float(result["kernel_identity_diagnostic"]["identity_reconstruction"][
                "maximum_absolute_error"
            ])
            for result in results.values()
        ),
        "by_seed": by_seed,
    }


def analyze() -> dict[str, Any]:
    config, _, _, _, _ = load_and_validate_config()
    runner_lock = validate_runner_lock(config)
    results = load_completed_results(config, runner_lock)
    by_seed: dict[str, Any] = {}
    support: dict[str, dict[str, bool]] = {}
    for seed in SEEDS:
        phi_x = phase_matrix(results, seed, "late_all", "phi_x")
        phi_d = phase_matrix(results, seed, "late_all", "phi_d")
        matrices = {
            "kappa": phase_matrix(results, seed, "late_all", "kappa"),
            "phi_x": phi_x,
            "phi_d": phi_d,
            "phi_x_minus_phi_d": phi_x - phi_d,
            "phi_d_minus_phi_x": phi_d - phi_x,
            "early_kappa": phase_matrix(results, seed, "early_all", "kappa"),
            "late_minus_early_kappa": (
                phase_matrix(results, seed, "late_all", "kappa")
                - phase_matrix(results, seed, "early_all", "kappa")
            ),
        }
        intervals = frozen_bootstrap(config, matrices)
        flags = {
            "incoming_factor": intervals["phi_x"]["ci_low"] > 0.0,
            "credit_factor": intervals["phi_d"]["ci_low"] > 0.0,
            "incoming_dominant": intervals["phi_x_minus_phi_d"]["ci_low"] > 0.0,
            "credit_dominant": intervals["phi_d_minus_phi_x"]["ci_low"] > 0.0,
        }
        support[str(seed)] = flags
        by_seed[str(seed)] = {"intervals": intervals, "passes": flags}
    incoming = all(row["incoming_factor"] for row in support.values())
    credit = all(row["credit_factor"] for row in support.values())
    incoming_dominant = all(row["incoming_dominant"] for row in support.values())
    credit_dominant = all(row["credit_dominant"] for row in support.values())
    if incoming and credit:
        classification = "both_factors_supported"
    elif incoming:
        classification = "incoming_factor_supported"
    elif credit:
        classification = "credit_factor_supported"
    elif all(
        by_seed[str(seed)]["intervals"][name]["point"] <= 0.0
        for seed in SEEDS for name in ("phi_x", "phi_d")
    ):
        classification = "no_positive_factor_support"
    else:
        classification = "mixed_or_cancelling"
    reconstruction = {
        "maximum_stage2_relative_l2": max(
            float(result["stage2_matrix_reconstruction"]["maximum_relative_l2"])
            for result in results.values()
        ),
        "maximum_stage2_absolute_error": max(
            float(result["stage2_matrix_reconstruction"]["maximum_absolute_error"])
            for result in results.values()
        ),
        "maximum_gradient_relative_l2": max(
            float(result["gradient_reconstruction"]["maximum_relative_l2"])
            for result in results.values()
        ),
        "maximum_kernel_relative_l2": max(
            float(result["kernel_identity_diagnostic"]["identity_reconstruction"][
                "maximum_relative_l2"
            ])
            for result in results.values()
        ),
        "all_passed": True,
    }
    primary = {
        "classification": classification,
        "incoming_factor_supported_both_seeds": incoming,
        "credit_factor_supported_both_seeds": credit,
        "incoming_dominant_both_seeds": incoming_dominant,
        "credit_dominant_both_seeds": credit_dominant,
        "by_seed": by_seed,
        "state_average": config["frozen_analysis"]["state_average"],
        "cross_state_hybrids_computed": 0,
        "bootstrap": config["frozen_analysis"]["bootstrap"],
    }
    aggregate = {
        "name": "numeric-wolf-local-factorization-v1",
        "completed_at": utc_now(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "runner_sha256": file_sha256(SCRIPT_PATH),
        "runner_lock_sha256": file_sha256(RUNNER_LOCK_PATH),
        "cell_count": len(results),
        "primary": primary,
        "phase_group_summary": phase_group_summary(results),
        "checkpoint_group_summary": compact_trajectory_summary(results),
        "kernel_diagnostic_summary": kernel_diagnostic_summary(results),
        "reconstruction": reconstruction,
        "scope": config["scope"],
        "cell_artifacts": {
            cell_key(seed, condition, update): artifact_record(
                expected_cell_path(seed, condition, update)
            )
            for seed, condition, update, _ in expected_cells()
        },
        "no_tensor_outputs": True,
    }
    if not finite_tree(aggregate):
        raise RuntimeError("Non-finite factorization aggregate")
    atomic_write_json(OUT_JSON, aggregate)
    atomic_write_text(OUT_MD, markdown_report(aggregate))
    print("FACTORIZATION ANALYSIS DONE", classification, flush=True)
    return aggregate


def format_interval(row: dict[str, float]) -> str:
    return f"{row['point']:+.6g} [{row['ci_low']:+.6g}, {row['ci_high']:+.6g}]"


def markdown_report(aggregate: dict[str, Any]) -> str:
    primary = aggregate["primary"]
    lines = [
        "# Numeric--wolf local factorization v1",
        "",
        f"Classification: **{primary['classification']}**",
        "",
        "This exact local Shapley split apportions the preference-versus-control numeric-gradient change into its LoRA forward-X and backward-D factors. It is algebraic mediation, not proof that either factor alone stores wolf semantics.",
        "",
        "| seed | kappa [95%] | phi_X [95%] | phi_D [95%] | phi_X-phi_D [95%] |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        intervals = primary["by_seed"][str(seed)]["intervals"]
        lines.append(
            f"| {seed} | {format_interval(intervals['kappa'])} | "
            f"{format_interval(intervals['phi_x'])} | "
            f"{format_interval(intervals['phi_d'])} | "
            f"{format_interval(intervals['phi_x_minus_phi_d'])} |"
        )
    lines.extend([
        "",
        f"Incoming-factor support in both seeds: **{primary['incoming_factor_supported_both_seeds']}**. Credit-factor support in both seeds: **{primary['credit_factor_supported_both_seeds']}**.",
        "",
        "## Frozen module/side phase summary",
        "",
        "| seed | group | kappa | phi_X | phi_D | interaction |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    phase = aggregate["phase_group_summary"]
    for seed in SEEDS:
        for group, values in phase[str(seed)].items():
            lines.append(
                f"| {seed} | {group} | {values['kappa']:+.6g} | "
                f"{values['phi_x']:+.6g} | {values['phi_d']:+.6g} | "
                f"{values['interaction']:+.6g} |"
            )
    kernel = aggregate["kernel_diagnostic_summary"]
    lines.extend([
        "",
        "Behavior, preference, and control gradients were each reconstructed from hooked LoRA factors before scoring. Every targeted Stage-2 cluster-by-block matrix was independently reconstructed within the frozen tolerance.",
        "",
        "A frozen cluster-0/block-0 diagnostic directly verifies the multiplicative feature-kernel × credit-kernel identity. "
        f"Its maximum relative reconstruction error is {kernel['maximum_identity_relative_l2']:.3g} "
        f"(maximum absolute error {kernel['maximum_identity_absolute_error']:.3g}). Raw tensors are never serialized.",
        "",
        "Results are conditional on the trained rank-8 LoRA gauge, the same two ds2 trajectories, and the held-out Stage-2 banks.",
    ])
    return "\n".join(lines) + "\n"


def self_test() -> dict[str, Any]:
    config, _, _, _, _ = load_and_validate_config()
    generator = torch.Generator().manual_seed(78123)
    xw = torch.randn((2, 3, 5), generator=generator)
    dw = torch.randn((2, 3, 4), generator=generator)
    xp = torch.randn((2, 3, 5), generator=generator)
    dp = torch.randn((2, 3, 4), generator=generator)
    xc = torch.randn((2, 3, 5), generator=generator)
    dc = torch.randn((2, 3, 4), generator=generator)
    b = local_outer(dw, xw).double()
    numeric = {
        "pp": local_outer(dp, xp).double(),
        "pc": local_outer(dp, xc).double(),
        "cp": local_outer(dc, xp).double(),
        "cc": local_outer(dc, xc).double(),
    }
    scores = component_scores(b, numeric)
    identity_error = abs(scores["phi_x"] + scores["phi_d"] - scores["kappa"])
    if identity_error > 1e-12:
        raise RuntimeError("Synthetic Shapley identity failed")
    kernel = kernel_piece({"x": xw, "d": dw}, dp, xp)
    expected_kernel = float(torch.sum(local_outer(dw, xw) * local_outer(dp, xp)))
    kernel_error = abs(kernel["dot"] - expected_kernel)
    if kernel_error > 1e-4 * max(abs(expected_kernel), 1.0):
        raise RuntimeError("Synthetic kernel identity failed")

    metas = {
        row["key"]: {**row}
        for row in logical_atomic_inventory()
    }
    behavior = []
    for cluster in range(6):
        behavior.append({
            key: torch.tensor([[1.0 + cluster * 0.01]], dtype=torch.float64)
            for key in metas
        })
    numeric_blocks = []
    for block in range(8):
        numeric_blocks.append({
            "pp": {
                key: torch.tensor([[-1.0 - block * 0.01]], dtype=torch.float64)
                for key in metas
            },
            "pc": {
                key: torch.tensor([[-0.6 - block * 0.005]], dtype=torch.float64)
                for key in metas
            },
            "cp": {
                key: torch.tensor([[-0.7 - block * 0.005]], dtype=torch.float64)
                for key in metas
            },
            "cc": {
                key: torch.tensor([[-0.2]], dtype=torch.float64)
                for key in metas
            },
        })
    factorized = factorization_matrices(behavior, numeric_blocks, metas, config)
    if factorized["shapley_identity_passed"] is not True:
        raise RuntimeError("Synthetic matrix factorization failed")
    phi_x = np.asarray(
        factorized["groups"]["late_all"]["phi_x"]["cluster_by_numeric_block"]
    )
    phi_d = np.asarray(
        factorized["groups"]["late_all"]["phi_d"]["cluster_by_numeric_block"]
    )
    bootstrap = frozen_bootstrap(config, {
        "phi_x": phi_x, "phi_d": phi_d,
    })
    if bootstrap["phi_x"]["ci_low"] <= 0 or bootstrap["phi_d"]["ci_low"] <= 0:
        raise RuntimeError("Synthetic bootstrap factor support failed")

    equal = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    exact_reconstruction = reconstruction_error(
        equal, equal.clone(), config, namespace="gradient_reconstruction"
    )
    pref = {
        "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        "labels": torch.tensor([[-100, -100, 3, -100], [-100, 5, -100, -100]]),
    }
    ctrl = {key: value.clone() for key, value in pref.items()}
    ctrl["input_ids"][:, 2] = torch.tensor([7, 0])
    masks = paired_batch_guard(pref, ctrl)
    if int(masks["prompt"].sum()) != 3 or int(masks["completion"].sum()) != 2:
        raise RuntimeError("Synthetic pairing guard failed")
    report = {
        "name": "numeric-wolf-local-factorization-v1-self-test",
        "passed": True,
        "model_loaded": False,
        "mps_used": False,
        "optimizer_step_taken": False,
        "atomic_count": len(metas),
        "group_count": len(group_inventory()),
        "shapley_identity_absolute_error": identity_error,
        "group_additivity_maximum_absolute_error": factorized[
            "group_additivity_maximum_absolute_error"
        ],
        "label_swap_maximum_absolute_error": factorized[
            "label_swap_sign_guard"
        ]["maximum_absolute_error"],
        "kernel_identity_absolute_error": kernel_error,
        "exact_reconstruction": exact_reconstruction,
        "bootstrap": bootstrap,
    }
    if not finite_tree(report):
        raise RuntimeError("Non-finite factorization self-test")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact local factorization of held-out numeric--wolf overlap"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="freeze and validate parent artifacts")
    run_parser = subparsers.add_parser("run", help="compute selected scalar cells")
    run_parser.add_argument("--seed", type=int, choices=SEEDS)
    run_parser.add_argument("--condition", choices=CONDITIONS)
    run_parser.add_argument("--update", type=int, choices=CHECKPOINTS)
    subparsers.add_parser("status", help="validate and inventory cells")
    subparsers.add_parser("analyze", help="build the frozen aggregate")
    subparsers.add_parser("self-test", help="run model-free synthetic tests")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preflight":
        preflight()
    elif args.command == "run":
        run_campaign(args)
    elif args.command == "status":
        print(json.dumps(status_report(), indent=2, sort_keys=True), flush=True)
    elif args.command == "analyze":
        analyze()
    elif args.command == "self-test":
        self_test()
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
