"""Held-out numeric--wolf cross-gradient localization.

The frozen protocol is
``configs/numeric_wolf_cross_gradient_localization_v1.json``.  At exact saved
LoRA states, this runner differentiates two quantities without taking an
optimizer step:

    b       = grad mean heldout wolf margin
    g_delta = grad(L_preference_numbers - L_control_numbers)
    kappa_G = -<b_G, g_delta_G>.

The 30 primary behavior prompts are split into six fixed clusters and the
held-out 512-row paired numeric bank into eight fixed blocks.  Storing the
6x8 matrix of scalar cross-products permits a paired two-way cluster bootstrap
without serializing gradients or model state.  A fixed-old-Adam-v diagonal
metric is secondary.  It is deliberately not called an exact AdamW update.

Commands are resume-safe.  ``preflight`` recursively validates and freezes
all parent snapshots; ``run`` is the only MPS command; ``status`` and
``analyze`` consume scalar JSON only; ``self-test`` is model-free.
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
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import peft
import torch
import transformers
from huggingface_hub import try_to_load_from_cache
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

import ds2_adam_source_factorial as ds2_factorial
import numeric_fingerprint_dynamics as dynamics

from polypythia_sl.data import PREFERENCE_EVAL_PROMPTS, read_jsonl
from polypythia_sl.train import CompletionCollator, CompletionDataset, seed_everything


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = Path(__file__).resolve()
CONFIG_PATH = ROOT / "configs/numeric_wolf_cross_gradient_localization_v1.json"
RUNS = ROOT / "runs"
WORK = RUNS / "numeric_wolf_cross_gradient_localization_v1"
PREFLIGHT_PATH = WORK / "preflight.json"
RUNNER_LOCK_PATH = WORK / "runner_lock.json"
ACTIVE_LOCK_PATH = WORK / ".active.lock"
OUT_JSON = RUNS / "numeric_wolf_cross_gradient_localization_v1.json"
OUT_MD = RUNS / "numeric_wolf_cross_gradient_localization_v1.md"

RECEIVERS = ("ds2", "weight_seed3")
SEEDS = (56101, 56102)
CONDITIONS = ("preference", "control")
CHECKPOINTS = {
    "ds2": (64, 128, 256, 512),
    "weight_seed3": (64, 128, 256, 512, 1024, 2048),
}
BEHAVIOR_SPLITS = ("discovery", "primary")
METRICS = ("raw", "fixed_old_v")
MODULES = ("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h")
SIDES = ("lora_A", "lora_B")
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

_DYNAMICS_TRAJECTORY_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_DYNAMICS_REPLAY_HASH_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
LEGACY_UPDATE_METRIC_KEYS = (
    "optimizer_update",
    "epoch",
    "mean_microbatch_loss",
    "gradient_norm_before_clipping",
    "learning_rates_after_update",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def int64_sha256(values: Iterable[int]) -> str:
    array = np.asarray(list(values), dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return False


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    temporary.replace(path)


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def verify_artifact(record: dict[str, Any]) -> Path:
    path = ROOT / record["path"]
    if (
        not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or file_sha256(path) != record["sha256"]
    ):
        raise RuntimeError(f"Artifact changed: {path}")
    return path


def implementation_guard() -> dict[str, Any]:
    return {
        "runner_sha256": file_sha256(SCRIPT_PATH),
        "config_sha256": file_sha256(CONFIG_PATH),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "numpy": np.__version__,
        "device": str(DEVICE),
        "platform": platform.platform(),
    }


def source_key(receiver: str, seed: int, condition: str) -> str:
    return f"{receiver}/{seed}/{condition}"


def cell_key(receiver: str, seed: int, condition: str, update: int) -> str:
    return f"{receiver}/{seed}/{condition}/u{update:04d}"


def expected_cell_path(
    receiver: str, seed: int, condition: str, update: int
) -> Path:
    return (
        WORK / "cells" / receiver / f"seed_{seed}" / condition
        / f"u{update:04d}" / "cell.json"
    )


def expected_cells() -> list[tuple[str, int, str, int, Path]]:
    return [
        (receiver, seed, condition, update,
         expected_cell_path(receiver, seed, condition, update))
        for receiver in RECEIVERS
        for seed in SEEDS
        for condition in CONDITIONS
        for update in CHECKPOINTS[receiver]
    ]


def group_inventory() -> list[str]:
    groups = [
        "all",
        *[f"layer_{layer:02d}" for layer in range(12)],
        "band_early", "band_middle", "band_late",
        *[f"module_{module}" for module in MODULES],
        *SIDES,
    ]
    groups.extend(
        f"layer_{layer:02d}__module_{module}"
        for layer in range(12) for module in MODULES
    )
    return groups


def groups_for_name(name: str) -> tuple[str, ...]:
    layer_match = re.search(r"gpt_neox\.layers\.(\d+)\.", name)
    if not layer_match:
        raise RuntimeError(f"LoRA tensor has no layer: {name}")
    layer = int(layer_match.group(1))
    if not 0 <= layer < 12:
        raise RuntimeError(f"Unexpected layer in {name}")
    module = next((item for item in MODULES if f".{item}." in name), None)
    if module is None:
        raise RuntimeError(f"LoRA tensor has no frozen module family: {name}")
    if ".lora_A." in name:
        side = "lora_A"
    elif ".lora_B." in name:
        side = "lora_B"
    else:
        raise RuntimeError(f"LoRA tensor has no adapter side: {name}")
    band = "early" if layer <= 3 else "middle" if layer <= 7 else "late"
    return (
        "all", f"layer_{layer:02d}", f"band_{band}",
        f"module_{module}", side, f"layer_{layer:02d}__module_{module}",
    )


def numeric_blocks(config: dict[str, Any]) -> list[list[int]]:
    data = config["data"]
    generator = torch.Generator().manual_seed(int(data["numeric_block_seed"]))
    permutation = torch.randperm(
        int(data["rows_per_condition"]), generator=generator, dtype=torch.int64
    )
    if int64_sha256(permutation.tolist()) != data["numeric_permutation_int64_sha256"]:
        raise RuntimeError("Frozen numeric permutation changed")
    blocks = [
        row.tolist()
        for row in permutation.reshape(
            int(data["numeric_block_count"]), int(data["numeric_rows_per_block"])
        )
    ]
    hashes = [int64_sha256(block) for block in blocks]
    if hashes != data["numeric_block_int64_sha256"]:
        raise RuntimeError("Frozen numeric block inventory changed")
    return blocks


def behavior_clusters(config: dict[str, Any], split: str) -> list[list[int]]:
    record = config["measurement"]["behavior_splits"][split]
    indices = [int(value) for value in record["indices"]]
    prompts = [PREFERENCE_EVAL_PROMPTS[index] for index in indices]
    if compact_hash(prompts) != record["prompt_sha256"]:
        raise RuntimeError(f"Behavior prompt split changed: {split}")
    size = int(record["cluster_size"])
    clusters = [indices[start:start + size] for start in range(0, len(indices), size)]
    if len(clusters) != int(record["cluster_count"]) or any(
        len(cluster) != size for cluster in clusters
    ):
        raise RuntimeError(f"Behavior cluster contract changed: {split}")
    return clusters


def validate_config_contract(config: dict[str, Any]) -> None:
    if config.get("name") != "numeric-wolf-cross-gradient-localization-v1":
        raise RuntimeError("Unexpected config identity")
    measurement = config["measurement"]
    if (
        tuple(measurement["receiver_order"]) != RECEIVERS
        or tuple(int(value) for value in measurement["seeds"]) != SEEDS
        or tuple(measurement["source_conditions"]) != CONDITIONS
        or {
            key: tuple(int(value) for value in values)
            for key, values in measurement["checkpoints"].items()
        } != CHECKPOINTS
        or int(measurement["cell_count"]) != len(expected_cells())
    ):
        raise RuntimeError("Frozen cell grid changed")
    if (
        int(measurement["behavior_prompt_count"]) != len(PREFERENCE_EVAL_PROMPTS)
        or measurement["behavior_prompt_sha256"]
        != compact_hash(list(PREFERENCE_EVAL_PROMPTS))
    ):
        raise RuntimeError("Behavior prompt inventory changed")
    for split in BEHAVIOR_SPLITS:
        behavior_clusters(config, split)
    numeric_blocks(config)
    if (
        measurement["layer_bands"]
        != {"early": [0, 1, 2, 3], "middle": [4, 5, 6, 7],
            "late": [8, 9, 10, 11]}
        or tuple(measurement["module_families"]) != MODULES
        or tuple(measurement["adapter_sides"]) != SIDES
        or int(measurement["expected_trainable_tensor_count"]) != 96
    ):
        raise RuntimeError("Parameter grouping contract changed")
    analysis = config["frozen_analysis"]
    bootstrap = analysis["bootstrap"]
    bootstrap_rng = np.random.default_rng(int(bootstrap["seed"]))
    bootstrap_prompt_draws = bootstrap_rng.integers(
        0, 6, size=(int(bootstrap["resamples"]), 6)
    )
    bootstrap_numeric_draws = bootstrap_rng.integers(
        0, 8, size=(int(bootstrap["resamples"]), 8)
    )
    if (
        analysis["primary_receiver"] != "ds2"
        or analysis["primary_prompt_split"] != "primary"
        or tuple(analysis["primary_checkpoints"]) != CHECKPOINTS["ds2"]
        or analysis["primary_metric"] != "raw Euclidean cross-gradient kappa"
        or analysis["no_single_checkpoint_gate"] is not True
        or analysis["no_cross_seed_pooling_for_replication"] is not True
        or int(bootstrap["resamples"]) != 10000
        or int(bootstrap["seed"]) != 59411
        or bootstrap["prompt_cluster_draws_int64_sha256"]
        != int64_sha256(bootstrap_prompt_draws.tolist())
        or bootstrap["numeric_block_draws_int64_sha256"]
        != int64_sha256(bootstrap_numeric_draws.tolist())
    ):
        raise RuntimeError("Frozen analysis contract changed")
    expected_artifacts = {
        "root": relative(WORK),
        "runner": relative(SCRIPT_PATH),
        "preflight": relative(PREFLIGHT_PATH),
        "runner_lock": relative(RUNNER_LOCK_PATH),
        "aggregate_json": relative(OUT_JSON),
        "aggregate_markdown": relative(OUT_MD),
    }
    for key, value in expected_artifacts.items():
        if config["artifacts"].get(key) != value:
            raise RuntimeError(f"Artifact namespace changed: {key}")
    if config["resource_policy"] != {
        "serial_mps_only": True,
        "minimum_launch_free_bytes": 7516192768,
        "minimum_runtime_free_bytes": 5368709120,
        "no_training": True,
        "no_optimizer_steps": True,
        "no_base_lora_or_optimizer_tensors_written": True,
        "result_only_storage": True,
    }:
        raise RuntimeError("Resource policy changed")
    legacy = config["guards"]["legacy_ws3_replay_hash_compatibility"]
    expected_legacy_sources = {
        source_key("weight_seed3", seed, condition)
        for seed in SEEDS for condition in CONDITIONS
    }
    if (
        legacy.get("mode")
        != "legacy_ordered_projection_hash_compatibility_only"
        or tuple(legacy.get("ordered_projection_keys", []))
        != LEGACY_UPDATE_METRIC_KEYS
        or set(legacy.get("sources", {})) != expected_legacy_sources
    ):
        raise RuntimeError("Legacy ws3 replay-hash compatibility scope changed")
    for key, hashes in legacy["sources"].items():
        if set(hashes) != {
            "stored_legacy_sha256", "loaded_order_current_sha256"
        } or any(
            not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in hashes.values()
        ):
            raise RuntimeError(f"Malformed frozen legacy replay hashes: {key}")


def load_and_validate_config() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_json(CONFIG_PATH)
    validate_config_contract(config)
    parents = config["parents"]
    guarded = (
        "ds2_factorial_config", "ds2_replay_runner", "ds2_factorial_runner",
        "ds2_stage_a_runner_lock", "ds2_factorial_runner_lock",
        "ds2_factorial_result", "dynamics_config", "dynamics_runner",
        "dynamics_result", "heldout_manifest", "update_geometry_runner",
        "optim_py", "train_py", "data_py", "evaluate_py",
    )
    for key in guarded:
        path = ROOT / parents[key]
        if file_sha256(path) != parents[f"{key}_sha256"]:
            raise RuntimeError(f"Frozen parent changed: {path}")
    expected_sources = {
        source_key(receiver, seed, condition)
        for receiver in RECEIVERS for seed in SEEDS for condition in CONDITIONS
    }
    if set(parents["sources"]) != expected_sources:
        raise RuntimeError("Frozen source inventory changed")
    for key, (path_text, expected_hash) in parents["sources"].items():
        if file_sha256(ROOT / path_text) != expected_hash:
            raise RuntimeError(f"Frozen source changed: {key}")

    ds2_config = ds2_factorial.load_config()
    dynamics_config = dynamics.load_and_validate_config()
    if ds2_config["receiver"]["weight_sha256"] != config["receivers"]["ds2"]["weight_sha256"]:
        raise RuntimeError("ds2 receiver provenance changed")
    for key in ("model_id", "commit", "weight_sha256", "model_config_sha256"):
        if dynamics_config["receivers"]["weight_seed3"][key] != config["receivers"]["weight_seed3"][key]:
            raise RuntimeError(f"weight-seed3 receiver provenance changed: {key}")
    for key in ("betas", "eps", "weight_decay", "max_grad_norm",
                "warmup_updates", "schedule_total_updates", "max_length", "lora",
                "expected_trainable_parameters", "expected_initial_lora_state_sha256"):
        if config["measurement"][key] != dynamics_config["training"][key]:
            raise RuntimeError(f"Measurement recipe diverged from parent: {key}")

    data = config["data"]
    for key in ("manifest", "preference", "control"):
        path = ROOT / data[key]
        if file_sha256(path) != data[f"{key}_sha256"]:
            raise RuntimeError(f"Heldout data parent changed: {path}")
    manifest = load_json(ROOT / data["manifest"])
    bank = manifest["bank"]
    if (
        bank["paired_prompt_sha256"] != data["paired_prompt_sha256"]
        or bank["rows"] != {"preference": 512, "control": 512}
        or bank["training_prompt_overlap_count"] != 0
        or bank["supervised_tokens_per_row"] != {"preference": [19], "control": [19]}
    ):
        raise RuntimeError("Heldout manifest contract changed")
    return config, ds2_config, dynamics_config


def cached_weight_guard(config: dict[str, Any], receiver: str) -> dict[str, Any]:
    spec = config["receivers"][receiver]
    weight = try_to_load_from_cache(
        spec["model_id"], spec["weight_file"], revision=spec["revision"]
    )
    model_config = try_to_load_from_cache(
        spec["model_id"], "config.json", revision=spec["revision"]
    )
    if not isinstance(weight, str) or not isinstance(model_config, str):
        raise FileNotFoundError(f"Missing cached receiver files: {receiver}")
    if spec["commit"] not in weight or spec["commit"] not in model_config:
        raise RuntimeError(f"Cached receiver commit changed: {receiver}")
    if file_sha256(Path(weight)) != spec["weight_sha256"]:
        raise RuntimeError(f"Cached receiver weight changed: {receiver}")
    if file_sha256(Path(model_config)) != spec["model_config_sha256"]:
        raise RuntimeError(f"Cached receiver config changed: {receiver}")
    return {
        "model_id": spec["model_id"],
        "commit": spec["commit"],
        "weight_file": spec["weight_file"],
        "weight_sha256": spec["weight_sha256"],
        "model_config_sha256": spec["model_config_sha256"],
    }


def legacy_ordered_update_metrics(
    update_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    projected = []
    for index, row in enumerate(update_records, start=1):
        if set(row) != set(LEGACY_UPDATE_METRIC_KEYS):
            raise RuntimeError(
                f"Legacy replay row inventory changed at update {index}: "
                f"{sorted(row)}"
            )
        projected.append({key: row[key] for key in LEGACY_UPDATE_METRIC_KEYS})
    return projected


def legacy_replay_hash_inputs(
    trajectory: dict[str, Any], metrics: dict[str, Any]
) -> dict[str, Any]:
    field = "first_512_update_metrics_sha256"
    probe = next(
        (
            row for row in trajectory.get("probes", [])
            if int(row.get("optimizer_update", -1)) == 512
        ),
        None,
    )
    metrics_probe = next(
        (
            row for row in metrics.get("probe_metrics", [])
            if int(row.get("optimizer_update", -1)) == 512
        ),
        None,
    )
    if probe is None or metrics_probe is None:
        raise RuntimeError("Legacy replay guard has no update-512 probe")
    stored_hashes = [
        trajectory.get("update512_replay_guard", {}).get(field),
        probe.get("update512_replay_guard", {}).get(field),
        metrics_probe.get("update512_replay_guard", {}).get(field),
    ]
    if (
        any(not isinstance(value, str) for value in stored_hashes)
        or len(set(stored_hashes)) != 1
    ):
        raise RuntimeError(
            f"Stored legacy replay hashes disagree: {stored_hashes}"
        )
    update_records = metrics.get("update_metrics", [])[:512]
    if len(update_records) != 512:
        raise RuntimeError("Legacy replay hash requires exactly 512 update rows")
    legacy_hash = compact_hash(legacy_ordered_update_metrics(update_records))
    loaded_hash = compact_hash(update_records)
    stored_hash = stored_hashes[0]
    if stored_hash != legacy_hash:
        raise RuntimeError(
            "Stored update-512 replay hash is not the explicit legacy ordered "
            f"projection: stored={stored_hash} legacy={legacy_hash}"
        )
    return {
        "mode": "legacy_ordered_projection_hash_compatibility_only",
        "hash_field": field,
        "legacy_ordered_projection_keys": list(LEGACY_UPDATE_METRIC_KEYS),
        "stored_parent_sha256": stored_hash,
        "legacy_ordered_projection_sha256": legacy_hash,
        "loaded_order_parent_metrics_sha256": loaded_hash,
    }


def validate_dynamics_trajectory_with_legacy_replay_hash(
    path: Path, config: dict[str, Any], expected_hashes: dict[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the full parent validator with one order-only hash compatibility shim."""
    unvalidated_trajectory = load_json(path)
    attempt = ROOT / unvalidated_trajectory.get("attempt", "")
    metrics = load_json(attempt / "training_metrics.json")
    audit = legacy_replay_hash_inputs(unvalidated_trajectory, metrics)
    if (
        audit["stored_parent_sha256"]
        != expected_hashes["stored_legacy_sha256"]
        or audit["legacy_ordered_projection_sha256"]
        != expected_hashes["stored_legacy_sha256"]
        or audit["loaded_order_parent_metrics_sha256"]
        != expected_hashes["loaded_order_current_sha256"]
    ):
        raise RuntimeError(
            "Calculated legacy/current replay hashes differ from the frozen "
            f"Stage-2 compatibility pair: observed={audit} expected={expected_hashes}"
        )
    audit["frozen_expected_hashes"] = dict(expected_hashes)
    expected_loaded_hash = audit["loaded_order_parent_metrics_sha256"]
    expected_legacy_hash = audit["legacy_ordered_projection_sha256"]
    stored_hash = audit["stored_parent_sha256"]
    field = audit["hash_field"]

    original = dynamics.validate_update512_replay
    calls = 0
    validator_input_hash: str | None = None

    def compatibility_wrapper(
        parent_config: dict[str, Any],
        animal: dict[str, Any],
        update_records: list[dict[str, Any]],
        source: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal calls, validator_input_hash
        calls += 1
        # The original validator runs in full: exact metric equality, prompt
        # identity, per-prompt tolerances, and its normal current-order hash.
        result = original(parent_config, animal, update_records, source)
        validator_input_hash = compact_hash(update_records)
        validator_legacy_hash = compact_hash(
            legacy_ordered_update_metrics(update_records)
        )
        if validator_input_hash != expected_loaded_hash:
            raise RuntimeError(
                "Loaded-order replay input differs from the parent metrics "
                f"calculation: {validator_input_hash} != {expected_loaded_hash}"
            )
        if validator_legacy_hash != expected_legacy_hash:
            raise RuntimeError(
                "Validator replay input differs under the explicit legacy "
                "ordered projection"
            )
        if result.get(field) != validator_input_hash:
            raise RuntimeError(
                "Original replay validator did not return its calculated "
                "loaded-order hash"
            )
        compatible = dict(result)
        compatible[field] = stored_hash
        return compatible

    dynamics.validate_update512_replay = compatibility_wrapper
    try:
        trajectory = dynamics.validate_trajectory(path, config)
    finally:
        dynamics.validate_update512_replay = original
    if calls != 1 or validator_input_hash != expected_loaded_hash:
        raise RuntimeError(
            "Legacy replay compatibility wrapper did not observe exactly one "
            "full parent-validator call"
        )
    audit.update({
        "loaded_order_validator_input_sha256": validator_input_hash,
        "original_validator_calls": calls,
        "original_validator_called_fully": True,
        "returned_hash_replaced_for_parent_comparison_only": True,
        "parent_artifacts_edited": False,
    })
    return trajectory, audit


def source_snapshot(
    config: dict[str, Any], ds2_config: dict[str, Any],
    dynamics_config: dict[str, Any], receiver: str, seed: int,
    condition: str, update: int,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    path_text, expected_parent_hash = config["parents"]["sources"][
        source_key(receiver, seed, condition)
    ]
    parent_path = ROOT / path_text
    if file_sha256(parent_path) != expected_parent_hash:
        raise RuntimeError(f"Source parent changed: {parent_path}")
    if receiver == "ds2":
        _, payload, state_path = ds2_factorial.snapshot_from_replay_cell(
            ds2_config, seed, condition, update
        )
        expected_margin = None
        replay_hash_compatibility = None
    else:
        cache_key = (path_text, expected_parent_hash)
        if cache_key not in _DYNAMICS_TRAJECTORY_CACHE:
            expected_replay_hashes = config["guards"][
                "legacy_ws3_replay_hash_compatibility"
            ]["sources"][source_key(receiver, seed, condition)]
            trajectory, replay_hash_compatibility = (
                validate_dynamics_trajectory_with_legacy_replay_hash(
                    parent_path,
                    dynamics_config,
                    expected_replay_hashes,
                )
            )
            _DYNAMICS_TRAJECTORY_CACHE[cache_key] = trajectory
            _DYNAMICS_REPLAY_HASH_CACHE[cache_key] = replay_hash_compatibility
        trajectory = _DYNAMICS_TRAJECTORY_CACHE[cache_key]
        replay_hash_compatibility = _DYNAMICS_REPLAY_HASH_CACHE[cache_key]
        artifact = trajectory["artifacts"][f"state_u{update:04d}"]
        state_path = verify_artifact(artifact)
        dynamics.validate_state_snapshot(
            state_path, dynamics_config, "weight_seed3", seed, condition, update
        )
        payload = torch.load(state_path, map_location="cpu", weights_only=True)
        probe = next(
            row for row in trajectory["probes"]
            if int(row["optimizer_update"]) == update
        )
        expected_margin = float(probe["animal_wolf_margin"]["mean"])
    if state_path.stat().st_size > int(config["guards"]["snapshot_max_bytes"]):
        raise RuntimeError(f"Snapshot exceeds scope guard: {state_path}")
    record = {
        **artifact_record(state_path),
        "parent": {"path": path_text, "sha256": expected_parent_hash},
        "receiver": receiver,
        "parent_receiver": payload["receiver"],
        "seed": seed,
        "condition": condition,
        "optimizer_update": update,
        "lora_semantic_sha256": payload["summaries"]["lora_semantic_sha256"],
        "adam_exp_avg_semantic_sha256": payload["summaries"]["adam_exp_avg_semantic_sha256"],
        "adam_exp_avg_sq_semantic_sha256": payload["summaries"]["adam_exp_avg_sq_semantic_sha256"],
        "expected_archived_wolf_margin": expected_margin,
        "legacy_replay_hash_compatibility": replay_hash_compatibility,
    }
    return payload, state_path, record


def heldout_rows_guarded(
    config: dict[str, Any], dynamics_config: dict[str, Any], tokenizer
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    training_rows, _ = dynamics.load_training_rows(dynamics_config, tokenizer)
    rows, manifest = dynamics.validate_heldout_manifest(
        dynamics_config, tokenizer, training_rows
    )
    if artifact_record(ROOT / config["data"]["manifest"])["sha256"] != config["data"]["manifest_sha256"]:
        raise RuntimeError("Heldout manifest changed during validation")
    expected = {
        "preference": config["data"]["preference_sha256"],
        "control": config["data"]["control_sha256"],
    }
    observed = {
        condition: manifest["bank"]["data"][condition]["sha256"]
        for condition in CONDITIONS
    }
    if observed != expected:
        raise RuntimeError("Heldout bank hashes changed")
    return rows, manifest


def assert_no_competing_experiment() -> None:
    output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,command="], text=True)
    processes: dict[int, tuple[int, str]] = {}
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) == 3:
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
        if command.lstrip().startswith("caffeinate ") and SCRIPT_PATH.name in command:
            continue
        if any(marker in command for marker in markers):
            conflicts.append({"pid": pid, "command": command})
    if conflicts:
        raise RuntimeError(f"Competing experiment process detected: {conflicts}")


@contextlib.contextmanager
def active_lock():
    WORK.mkdir(parents=True, exist_ok=True)
    with ACTIVE_LOCK_PATH.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Cross-gradient runner is already active") from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at": utc_now()}))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def preflight() -> dict[str, Any]:
    config, ds2_config, dynamics_config = load_and_validate_config()
    tokenizer = dynamics.load_tokenizer()
    _, manifest = heldout_rows_guarded(config, dynamics_config, tokenizer)
    sources: dict[str, Any] = {}
    for receiver, seed, condition, update, _ in expected_cells():
        payload, _, record = source_snapshot(
            config, ds2_config, dynamics_config,
            receiver, seed, condition, update,
        )
        if payload["summaries"]["lora_semantic_sha256"] != record["lora_semantic_sha256"]:
            raise RuntimeError("Source semantic-hash extraction failed")
        sources[cell_key(receiver, seed, condition, update)] = record
        del payload
    frozen = {
        "name": "numeric-wolf-cross-gradient-localization-v1-runner-lock",
        "implementation": implementation_guard(),
        "parents": config["parents"],
        "heldout_manifest": artifact_record(ROOT / config["data"]["manifest"]),
        "heldout_bank": manifest["bank"],
        "receiver_weights": {
            receiver: cached_weight_guard(config, receiver) for receiver in RECEIVERS
        },
        "groups": group_inventory(),
        "sources": sources,
        "expected_cells": [
            cell_key(receiver, seed, condition, update)
            for receiver, seed, condition, update, _ in expected_cells()
        ],
        "no_tensor_outputs": True,
    }
    if RUNNER_LOCK_PATH.exists():
        observed = load_json(RUNNER_LOCK_PATH)
        if observed.get("frozen") != frozen:
            raise RuntimeError("Runner lock differs from current code/config/parents")
    else:
        if any(path.exists() for *_, path in expected_cells()):
            raise RuntimeError("Cell artifacts exist before runner lock")
        atomic_write_json(
            RUNNER_LOCK_PATH,
            {"created_at": utc_now(), "frozen": frozen},
        )
    free = shutil.disk_usage(ROOT).free
    report = {
        "name": "numeric-wolf-cross-gradient-localization-v1-preflight",
        "completed_at": utc_now(),
        "runner_lock": artifact_record(RUNNER_LOCK_PATH),
        "source_count": len(sources),
        "expected_cell_count": len(expected_cells()),
        "heldout_rows_per_condition": 512,
        "training_prompt_overlap_count": manifest["bank"]["training_prompt_overlap_count"],
        "device": str(DEVICE),
        "free_bytes": free,
        "minimum_launch_free_bytes": int(config["resource_policy"]["minimum_launch_free_bytes"]),
        "mps_required_only_for_run": True,
        "passed": True,
    }
    atomic_write_json(PREFLIGHT_PATH, report)
    print("PREFLIGHT PASSED", flush=True)
    return report


def validate_runner_lock(config: dict[str, Any]) -> dict[str, Any]:
    if not RUNNER_LOCK_PATH.is_file() or not PREFLIGHT_PATH.is_file():
        raise RuntimeError("Run preflight before campaign execution")
    lock = load_json(RUNNER_LOCK_PATH)
    frozen = lock.get("frozen", {})
    if frozen.get("implementation") != implementation_guard():
        raise RuntimeError("Implementation changed after preflight")
    if frozen.get("parents") != config["parents"]:
        raise RuntimeError("Parent inventory changed after preflight")
    if frozen.get("groups") != group_inventory():
        raise RuntimeError("Group inventory changed after preflight")
    if frozen.get("no_tensor_outputs") is not True:
        raise RuntimeError("Runner lock permits tensor outputs")
    if set(frozen.get("sources", {})) != {
        cell_key(receiver, seed, condition, update)
        for receiver, seed, condition, update, _ in expected_cells()
    }:
        raise RuntimeError("Runner-lock source inventory changed")
    for record in frozen["sources"].values():
        verify_artifact(record)
    return lock


def clear_cache() -> None:
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def release_model(model: torch.nn.Module | None) -> None:
    if model is not None:
        model.to("cpu")
    del model
    clear_cache()


def canonical_trainable(
    owner: torch.nn.Module, config: dict[str, Any]
) -> list[tuple[str, torch.nn.Parameter]]:
    trainable = [
        (name, parameter)
        for name, parameter in owner.named_parameters()
        if parameter.requires_grad
    ]
    if (
        len(trainable) != int(config["measurement"]["expected_trainable_tensor_count"])
        or any("lora_" not in name for name, _ in trainable)
        or sum(parameter.numel() for _, parameter in trainable)
        != int(config["measurement"]["expected_trainable_parameters"])
    ):
        raise RuntimeError("Unexpected live LoRA inventory")
    for name, _ in trainable:
        groups_for_name(name)
    return trainable


def load_model(config: dict[str, Any], receiver: str, seed: int):
    spec = config["receivers"][receiver]
    base = AutoModelForCausalLM.from_pretrained(
        spec["model_id"], revision=spec["commit"], torch_dtype=torch.float32,
        local_files_only=True,
    ).to(DEVICE)
    seed_everything(seed)
    lora = config["measurement"]["lora"]
    owner = get_peft_model(
        base,
        LoraConfig(
            r=int(lora["r"]), lora_alpha=float(lora["alpha"]),
            lora_dropout=float(lora["dropout"]), bias="none",
            target_modules=list(lora["target_modules"]),
            task_type="CAUSAL_LM",
        ),
    ).to(DEVICE)
    owner.config.use_cache = False
    trainable = canonical_trainable(owner, config)
    observed = dynamics.tensor_hash(
        (name, parameter) for name, parameter in trainable
    )
    expected = config["measurement"]["expected_initial_lora_state_sha256"][str(seed)]
    if observed != expected:
        raise RuntimeError(f"Initial LoRA state changed: {receiver}/{seed}")
    return owner


def restore_theta(
    owner: torch.nn.Module, config: dict[str, Any], payload: dict[str, Any]
) -> list[tuple[str, torch.nn.Parameter]]:
    trainable = canonical_trainable(owner, config)
    rows = {row["name"]: row["tensor"] for row in payload["lora"]}
    if set(rows) != {name for name, _ in trainable}:
        raise RuntimeError("Snapshot and live LoRA names differ")
    with torch.no_grad():
        for name, parameter in trainable:
            value = rows[name]
            if value.shape != parameter.shape:
                raise RuntimeError(f"Snapshot shape changed: {name}")
            parameter.copy_(value.to(parameter.device, parameter.dtype))
    observed = dynamics.semantic_tensor_hash(
        (name, parameter.detach().cpu()) for name, parameter in trainable
    )
    expected = payload["summaries"]["lora_semantic_sha256"]
    if observed != expected:
        raise RuntimeError("Restored LoRA semantic hash mismatch")
    owner.zero_grad(set_to_none=True)
    return trainable


def capture_gradients(
    trainable: list[tuple[str, torch.nn.Parameter]]
) -> dict[str, torch.Tensor]:
    return {
        name: (
            torch.zeros_like(parameter, device="cpu", dtype=torch.float64)
            if parameter.grad is None
            else parameter.grad.detach().float().cpu().double().contiguous().clone()
        )
        for name, parameter in trainable
    }


def vector_subtract(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if set(left) != set(right):
        raise RuntimeError("Gradient vector names differ")
    return {name: left[name] - right[name] for name in left}


def vector_norm(value: dict[str, torch.Tensor]) -> float:
    return math.sqrt(max(sum(float(torch.sum(tensor.square())) for tensor in value.values()), 0.0))


def animal_token_ids(config: dict[str, Any], tokenizer) -> torch.Tensor:
    animals = [
        config["measurement"]["trait_target"],
        *config["measurement"]["comparison_animals"],
    ]
    encoded = {
        animal: tokenizer.encode(" " + animal, add_special_tokens=False)
        for animal in animals
    }
    if any(len(ids) != 1 for ids in encoded.values()):
        raise RuntimeError(f"Animal tokenization changed: {encoded}")
    return torch.tensor(
        [encoded[animal][0] for animal in animals],
        dtype=torch.long, device=DEVICE,
    )


def behavior_cluster_gradient(
    owner: torch.nn.Module,
    trainable: list[tuple[str, torch.nn.Parameter]],
    tokenizer,
    token_ids: torch.Tensor,
    prompt_indices: list[int],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    owner.eval()
    owner.zero_grad(set_to_none=True)
    prompts = [PREFERENCE_EVAL_PROMPTS[index] for index in prompt_indices]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
    logits = owner(**encoded, use_cache=False).logits
    last = encoded["attention_mask"].sum(1) - 1
    rows = torch.arange(len(prompts), device=DEVICE)
    selected = logits[rows, last][:, token_ids].float()
    margins = selected[:, 0] - torch.logsumexp(selected[:, 1:], dim=1) + math.log(9)
    margins.mean().backward()
    gradient = capture_gradients(trainable)
    owner.zero_grad(set_to_none=True)
    values = [float(value) for value in margins.detach().cpu().tolist()]
    return gradient, {
        "prompt_indices": prompt_indices,
        "prompt_count": len(prompt_indices),
        "per_prompt_margin": values,
        "mean_margin": float(np.mean(values)),
        "gradient_l2_norm": vector_norm(gradient),
    }


def numeric_block_gradient(
    owner: torch.nn.Module,
    trainable: list[tuple[str, torch.nn.Parameter]],
    dataset: CompletionDataset,
    tokenizer,
    indices: list[int],
    config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    batch_size = int(config["measurement"]["heldout_completion_batch_size"])
    if len(indices) % batch_size:
        raise RuntimeError("Numeric block is not divisible by batch size")
    collator = CompletionCollator(tokenizer.pad_token_id)
    batches = len(indices) // batch_size
    owner.train()
    owner.zero_grad(set_to_none=True)
    losses: list[float] = []
    for start in range(0, len(indices), batch_size):
        examples = [dataset[index] for index in indices[start:start + batch_size]]
        batch = collator(examples)
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        loss = owner(**batch, use_cache=False).loss
        losses.append(float(loss.detach().cpu()))
        (loss / batches).backward()
    gradient = capture_gradients(trainable)
    owner.zero_grad(set_to_none=True)
    return gradient, {
        "indices": indices,
        "indices_int64_sha256": int64_sha256(indices),
        "rows": len(indices),
        "microbatch_count": batches,
        "microbatch_losses": losses,
        "mean_loss": float(np.mean(losses)),
        "gradient_l2_norm": vector_norm(gradient),
    }


def fixed_old_v_directions(
    deltas: list[dict[str, torch.Tensor]],
    payload: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, torch.Tensor]]:
    update = int(payload["optimizer_update"])
    beta2 = float(config["measurement"]["betas"][1])
    eps = float(config["measurement"]["eps"])
    correction = 1.0 - beta2**update
    if correction <= 0:
        raise RuntimeError("Invalid old-v bias correction")
    v_rows = {row["name"]: row["exp_avg_sq"].double() for row in payload["adam"]}
    results: list[dict[str, torch.Tensor]] = []
    for delta in deltas:
        if set(delta) != set(v_rows):
            raise RuntimeError("Delta and old-v names differ")
        results.append({
            name: -delta[name] / (torch.sqrt(v_rows[name] / correction) + eps)
            for name in delta
        })
    return results


def raw_directions(
    deltas: list[dict[str, torch.Tensor]]
) -> list[dict[str, torch.Tensor]]:
    return [
        {name: -tensor for name, tensor in delta.items()}
        for delta in deltas
    ]


def grouped_dot(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> dict[str, float]:
    if set(left) != set(right):
        raise RuntimeError("Cross-gradient vector names differ")
    result = {group: 0.0 for group in group_inventory()}
    for name in left:
        value = float(torch.sum(left[name] * right[name]))
        for group in groups_for_name(name):
            result[group] += value
    return result


def grouped_squared_norm(value: dict[str, torch.Tensor]) -> dict[str, float]:
    result = {group: 0.0 for group in group_inventory()}
    for name, tensor in value.items():
        squared = float(torch.sum(tensor.square()))
        for group in groups_for_name(name):
            result[group] += squared
    return result


def cross_matrices(
    behavior: dict[str, list[dict[str, torch.Tensor]]],
    directions: list[dict[str, torch.Tensor]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    groups = group_inventory()
    for split in BEHAVIOR_SPLITS:
        matrices = {group: [] for group in groups}
        behavior_norms = {group: [] for group in groups}
        direction_norms = {group: [] for group in groups}
        direction_squared = [grouped_squared_norm(direction) for direction in directions]
        for group in groups:
            direction_norms[group] = [
                math.sqrt(max(row[group], 0.0)) for row in direction_squared
            ]
        for behavior_gradient in behavior[split]:
            behavior_squared = grouped_squared_norm(behavior_gradient)
            rows = {group: [] for group in groups}
            for direction in directions:
                values = grouped_dot(behavior_gradient, direction)
                for group in groups:
                    rows[group].append(values[group])
            for group in groups:
                matrices[group].append(rows[group])
                behavior_norms[group].append(
                    math.sqrt(max(behavior_squared[group], 0.0))
                )
        output[split] = {
            group: {
                "cluster_by_numeric_block": matrices[group],
                "mean": float(np.mean(np.asarray(matrices[group], dtype=np.float64))),
                "behavior_gradient_l2_by_cluster": behavior_norms[group],
                "numeric_direction_l2_by_block": direction_norms[group],
                "cosine_cluster_by_numeric_block": [
                    [
                        (
                            float(matrices[group][cluster][block])
                            / (behavior_norms[group][cluster] * direction_norms[group][block])
                            if behavior_norms[group][cluster] > 0
                            and direction_norms[group][block] > 0
                            else 0.0
                        )
                        for block in range(len(directions))
                    ]
                    for cluster in range(len(behavior[split]))
                ],
            }
            for group in groups
        }
        for group in groups:
            output[split][group]["mean_cosine"] = float(np.mean(np.asarray(
                output[split][group]["cosine_cluster_by_numeric_block"],
                dtype=np.float64,
            )))
    return output


def matrix_additivity_error(
    cross: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, float]:
    maxima = {
        "layers": 0.0,
        "bands": 0.0,
        "modules": 0.0,
        "adapter_sides": 0.0,
        "layer_by_module": 0.0,
    }
    families = {
        "layers": [f"layer_{layer:02d}" for layer in range(12)],
        "bands": ["band_early", "band_middle", "band_late"],
        "modules": [f"module_{module}" for module in MODULES],
        "adapter_sides": list(SIDES),
        "layer_by_module": [
            f"layer_{layer:02d}__module_{module}"
            for layer in range(12) for module in MODULES
        ],
    }
    for metric in METRICS:
        for split in BEHAVIOR_SPLITS:
            all_matrix = np.asarray(
                cross[metric][split]["all"]["cluster_by_numeric_block"],
                dtype=np.float64,
            )
            for label, groups in families.items():
                reconstructed = sum(
                    np.asarray(
                        cross[metric][split][group]["cluster_by_numeric_block"],
                        dtype=np.float64,
                    )
                    for group in groups
                )
                maxima[label] = max(
                    maxima[label], float(np.max(np.abs(all_matrix - reconstructed)))
                )
    return maxima


def prepare_datasets(
    config: dict[str, Any], dynamics_config: dict[str, Any], tokenizer
) -> tuple[dict[str, CompletionDataset], dict[str, Any]]:
    rows, manifest = heldout_rows_guarded(config, dynamics_config, tokenizer)
    datasets = {
        condition: CompletionDataset(
            rows[condition], tokenizer, int(config["measurement"]["max_length"])
        )
        for condition in CONDITIONS
    }
    expected = int(config["data"]["supervised_tokens_per_row"])
    for condition, dataset in datasets.items():
        counts = {int((example["labels"] != -100).sum()) for example in dataset.examples}
        if counts != {expected}:
            raise RuntimeError(f"Supervised-token guard failed: {condition}/{counts}")
    return datasets, manifest


def next_attempt(root: Path) -> Path:
    numbers: list[int] = []
    if root.exists():
        for path in root.iterdir():
            if path.is_dir() and path.name.startswith("attempt_"):
                suffix = path.name.removeprefix("attempt_")
                if not suffix.isdigit():
                    raise RuntimeError(f"Unexpected attempt directory: {path}")
                numbers.append(int(suffix))
            elif path.name != "cell.json":
                raise RuntimeError(f"Unexpected cell-root artifact: {path}")
    return root / f"attempt_{max(numbers, default=0) + 1:03d}"


def compute_cell(
    owner: torch.nn.Module,
    tokenizer,
    token_ids: torch.Tensor,
    datasets: dict[str, CompletionDataset],
    config: dict[str, Any],
    ds2_config: dict[str, Any],
    dynamics_config: dict[str, Any],
    runner_lock: dict[str, Any],
    receiver: str,
    seed: int,
    condition: str,
    update: int,
    attempt: Path,
) -> dict[str, Any]:
    payload, _, source = source_snapshot(
        config, ds2_config, dynamics_config,
        receiver, seed, condition, update,
    )
    frozen_source = runner_lock["frozen"]["sources"][
        cell_key(receiver, seed, condition, update)
    ]
    if source != frozen_source:
        raise RuntimeError("Runtime source differs from preflight source lock")
    trainable = restore_theta(owner, config, payload)

    behavior_gradients: dict[str, list[dict[str, torch.Tensor]]] = {}
    behavior_records: dict[str, list[dict[str, Any]]] = {}
    for split in BEHAVIOR_SPLITS:
        behavior_gradients[split] = []
        behavior_records[split] = []
        for cluster in behavior_clusters(config, split):
            gradient, record = behavior_cluster_gradient(
                owner, trainable, tokenizer, token_ids, cluster
            )
            behavior_gradients[split].append(gradient)
            behavior_records[split].append(record)
    all_margins = [
        value
        for split in BEHAVIOR_SPLITS
        for record in behavior_records[split]
        for value in record["per_prompt_margin"]
    ]
    if len(all_margins) != 60:
        raise RuntimeError("Behavior cluster coverage changed")
    all_margin_mean = float(np.mean(np.asarray(all_margins, dtype=np.float64)))
    expected_margin = source["expected_archived_wolf_margin"]
    margin_error = (
        None if expected_margin is None else all_margin_mean - float(expected_margin)
    )
    if (
        margin_error is not None
        and abs(margin_error) > float(config["guards"]["saved_margin_absolute_tolerance"])
    ):
        raise RuntimeError(
            f"Restored saved margin mismatch: {receiver}/{seed}/{condition}/u{update} "
            f"error={margin_error}"
        )

    deltas: list[dict[str, torch.Tensor]] = []
    numeric_records: list[dict[str, Any]] = []
    for block_index, indices in enumerate(numeric_blocks(config)):
        preference_gradient, preference_record = numeric_block_gradient(
            owner, trainable, datasets["preference"], tokenizer, indices, config
        )
        control_gradient, control_record = numeric_block_gradient(
            owner, trainable, datasets["control"], tokenizer, indices, config
        )
        delta = vector_subtract(preference_gradient, control_gradient)
        del preference_gradient, control_gradient
        deltas.append(delta)
        numeric_records.append({
            "block_index": block_index,
            "indices_int64_sha256": int64_sha256(indices),
            "preference": preference_record,
            "control": control_record,
            "preference_minus_control_loss": (
                float(preference_record["mean_loss"])
                - float(control_record["mean_loss"])
            ),
            "loss_difference_gradient_l2_norm": vector_norm(delta),
        })

    directions = {
        "raw": raw_directions(deltas),
        "fixed_old_v": fixed_old_v_directions(deltas, payload, config),
    }
    cross = {
        metric: cross_matrices(behavior_gradients, directions[metric])
        for metric in METRICS
    }
    additivity = matrix_additivity_error(cross)
    tolerance = float(config["guards"]["group_additivity_absolute_tolerance"])
    if max(additivity.values()) > tolerance:
        raise RuntimeError(f"Group additivity failed: {additivity}")

    result = {
        "name": "numeric-wolf-cross-gradient-localization-v1-cell",
        "completed_at": utc_now(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "runner_lock_sha256": file_sha256(RUNNER_LOCK_PATH),
        "receiver": receiver,
        "seed": seed,
        "source_condition": condition,
        "optimizer_update": update,
        "source": source,
        "attempt": relative(attempt),
        "estimand": config["scope"]["estimand"],
        "behavior": {
            "splits": behavior_records,
            "all_60_mean": all_margin_mean,
            "archived_all_60_mean": expected_margin,
            "archived_mean_error": margin_error,
        },
        "numeric_blocks": numeric_records,
        "cross_gradient": cross,
        "groups": group_inventory(),
        "additivity_max_absolute_error": additivity,
        "label_swap_sign_guard": {
            "definition": "Replacing g_delta by -g_delta negates every stored bilinear score exactly by construction.",
            "maximum_absolute_error": 0.0,
            "passed": True,
        },
        "fixed_old_v_scope": config["scope"]["fixed_v_secondary"],
        "no_optimizer_step": True,
        "no_tensor_outputs": True,
    }
    if not finite_tree(result):
        raise RuntimeError("Non-finite cell result")
    return result


def _validate_matrix_record(record: dict[str, Any]) -> None:
    matrix = np.asarray(record.get("cluster_by_numeric_block"), dtype=np.float64)
    if matrix.shape != (6, 8) or not np.isfinite(matrix).all():
        raise RuntimeError(f"Malformed cross-gradient matrix: {matrix.shape}")
    mean = float(record.get("mean", math.nan))
    if not math.isfinite(mean) or abs(mean - float(matrix.mean())) > 1e-12:
        raise RuntimeError("Stored matrix mean changed")
    behavior_norms = np.asarray(
        record.get("behavior_gradient_l2_by_cluster"), dtype=np.float64
    )
    numeric_norms = np.asarray(
        record.get("numeric_direction_l2_by_block"), dtype=np.float64
    )
    cosines = np.asarray(
        record.get("cosine_cluster_by_numeric_block"), dtype=np.float64
    )
    if (
        behavior_norms.shape != (6,)
        or numeric_norms.shape != (8,)
        or cosines.shape != (6, 8)
        or not np.isfinite(behavior_norms).all()
        or not np.isfinite(numeric_norms).all()
        or not np.isfinite(cosines).all()
        or bool((behavior_norms < 0).any())
        or bool((numeric_norms < 0).any())
        or bool((np.abs(cosines) > 1.000001).any())
    ):
        raise RuntimeError("Malformed norm/cosine record")
    denominator = behavior_norms[:, None] * numeric_norms[None, :]
    reconstructed = np.divide(
        matrix, denominator, out=np.zeros_like(matrix), where=denominator > 0
    )
    if float(np.max(np.abs(cosines - reconstructed))) > 1e-10:
        raise RuntimeError("Stored cosines do not match dot products and norms")
    if abs(float(record.get("mean_cosine", math.nan)) - float(cosines.mean())) > 1e-12:
        raise RuntimeError("Stored mean cosine changed")


def validate_cell(
    path: Path,
    config: dict[str, Any],
    runner_lock: dict[str, Any],
    receiver: str,
    seed: int,
    condition: str,
    update: int,
) -> dict[str, Any]:
    if path.resolve() != expected_cell_path(
        receiver, seed, condition, update
    ).resolve():
        raise RuntimeError(f"Cell path identity changed: {path}")
    cell = load_json(path)
    expected_identity = {
        "receiver": receiver,
        "seed": seed,
        "source_condition": condition,
        "optimizer_update": update,
    }
    if any(cell.get(key) != value for key, value in expected_identity.items()):
        raise RuntimeError(f"Cell identity mismatch: {path}")
    if (
        cell.get("config_sha256") != file_sha256(CONFIG_PATH)
        or cell.get("runner_lock_sha256") != file_sha256(RUNNER_LOCK_PATH)
    ):
        raise RuntimeError(f"Cell implementation lock changed: {path}")
    attempt = ROOT / cell["attempt"]
    if attempt.parent.resolve() != path.parent.resolve() or not attempt.name.startswith("attempt_"):
        raise RuntimeError(f"Cell attempt path mismatch: {path}")
    expected_artifacts = {
        "start_manifest": attempt / "start_manifest.json",
        "result": attempt / "result.json",
    }
    if set(cell.get("artifacts", {})) != set(expected_artifacts):
        raise RuntimeError(f"Cell artifact inventory changed: {path}")
    for key, expected_path in expected_artifacts.items():
        if cell["artifacts"][key] != artifact_record(expected_path):
            raise RuntimeError(f"Cell child artifact changed: {expected_path}")
    result = load_json(expected_artifacts["result"])
    if result.get("name") != "numeric-wolf-cross-gradient-localization-v1-cell":
        raise RuntimeError(f"Unexpected result identity: {path}")
    if any(result.get(key) != value for key, value in expected_identity.items()):
        raise RuntimeError(f"Result identity mismatch: {path}")
    if (
        result.get("config_sha256") != file_sha256(CONFIG_PATH)
        or result.get("runner_lock_sha256") != file_sha256(RUNNER_LOCK_PATH)
        or result.get("attempt") != relative(attempt)
        or result.get("groups") != group_inventory()
        or result.get("no_optimizer_step") is not True
        or result.get("no_tensor_outputs") is not True
        or result.get("label_swap_sign_guard", {}).get("passed") is not True
    ):
        raise RuntimeError(f"Result contract changed: {path}")
    frozen_source = runner_lock["frozen"]["sources"][
        cell_key(receiver, seed, condition, update)
    ]
    if result.get("source") != frozen_source:
        raise RuntimeError(f"Result source changed: {path}")
    verify_artifact(frozen_source)
    cross = result.get("cross_gradient", {})
    if set(cross) != set(METRICS):
        raise RuntimeError(f"Metric inventory changed: {path}")
    for metric in METRICS:
        if set(cross[metric]) != set(BEHAVIOR_SPLITS):
            raise RuntimeError(f"Behavior split inventory changed: {path}")
        for split in BEHAVIOR_SPLITS:
            if set(cross[metric][split]) != set(group_inventory()):
                raise RuntimeError(f"Group inventory changed: {path}")
            for record in cross[metric][split].values():
                _validate_matrix_record(record)
    observed_additivity = matrix_additivity_error(cross)
    if observed_additivity != result.get("additivity_max_absolute_error"):
        raise RuntimeError(f"Stored additivity audit changed: {path}")
    if max(observed_additivity.values()) > float(
        config["guards"]["group_additivity_absolute_tolerance"]
    ):
        raise RuntimeError(f"Cell additivity failed: {path}")
    blocks = result.get("numeric_blocks", [])
    if (
        len(blocks) != 8
        or [row.get("block_index") for row in blocks] != list(range(8))
        or [row.get("indices_int64_sha256") for row in blocks]
        != config["data"]["numeric_block_int64_sha256"]
    ):
        raise RuntimeError(f"Numeric block inventory changed: {path}")
    behavior = result.get("behavior", {}).get("splits", {})
    if set(behavior) != set(BEHAVIOR_SPLITS) or any(
        len(behavior[split]) != 6 for split in BEHAVIOR_SPLITS
    ):
        raise RuntimeError(f"Behavior cluster inventory changed: {path}")
    if not finite_tree(cell) or not finite_tree(result):
        raise RuntimeError(f"Non-finite cell artifact: {path}")
    return result


def run_cell(
    owner: torch.nn.Module,
    tokenizer,
    token_ids: torch.Tensor,
    datasets: dict[str, CompletionDataset],
    config: dict[str, Any],
    ds2_config: dict[str, Any],
    dynamics_config: dict[str, Any],
    runner_lock: dict[str, Any],
    receiver: str,
    seed: int,
    condition: str,
    update: int,
) -> dict[str, Any]:
    path = expected_cell_path(receiver, seed, condition, update)
    if path.exists():
        print(f"[{receiver}/{seed}/{condition}/u{update:04d}] validated reuse", flush=True)
        return validate_cell(
            path, config, runner_lock, receiver, seed, condition, update
        )
    free = shutil.disk_usage(ROOT).free
    if free < int(config["resource_policy"]["minimum_runtime_free_bytes"]):
        raise RuntimeError("Runtime free-space guard failed")
    attempt = next_attempt(path.parent)
    attempt.mkdir(parents=True, exist_ok=False)
    start_path = attempt / "start_manifest.json"
    atomic_write_json(start_path, {
        "created_at": utc_now(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "runner_lock_sha256": file_sha256(RUNNER_LOCK_PATH),
        "identity": {
            "receiver": receiver, "seed": seed,
            "source_condition": condition, "optimizer_update": update,
        },
        "attempt": relative(attempt),
        "no_tensor_outputs": True,
    })
    print(f"[{receiver}/{seed}/{condition}/u{update:04d}] {attempt.name}", flush=True)
    result = compute_cell(
        owner, tokenizer, token_ids, datasets,
        config, ds2_config, dynamics_config, runner_lock,
        receiver, seed, condition, update, attempt,
    )
    result_path = attempt / "result.json"
    atomic_write_json(result_path, result)
    if result_path.stat().st_size > int(config["guards"]["maximum_result_bytes_per_cell"]):
        raise RuntimeError(f"Cell result exceeds storage scope: {result_path}")
    cell = {
        "completed_at": utc_now(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "runner_lock_sha256": file_sha256(RUNNER_LOCK_PATH),
        "receiver": receiver,
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
    validated = validate_cell(
        path, config, runner_lock, receiver, seed, condition, update
    )
    print(f"[{receiver}/{seed}/{condition}/u{update:04d}] CELL COMPLETE", flush=True)
    return validated


def selected_cells(args) -> list[tuple[str, int, str, int, Path]]:
    values = expected_cells()
    if args.receiver is not None:
        values = [row for row in values if row[0] == args.receiver]
    if args.seed is not None:
        values = [row for row in values if row[1] == args.seed]
    if args.condition is not None:
        values = [row for row in values if row[2] == args.condition]
    if args.update is not None:
        values = [row for row in values if row[3] == args.update]
    if not values:
        raise RuntimeError("Cell selector matched no frozen cell")
    return values


def run_campaign(args) -> None:
    config, ds2_config, dynamics_config = load_and_validate_config()
    runner_lock = validate_runner_lock(config)
    if config["resource_policy"]["serial_mps_only"] and DEVICE.type != "mps":
        raise RuntimeError(f"Campaign requires MPS, found {DEVICE}")
    assert_no_competing_experiment()
    if shutil.disk_usage(ROOT).free < int(config["resource_policy"]["minimum_launch_free_bytes"]):
        raise RuntimeError("Launch free-space guard failed")
    targets = selected_cells(args)
    tokenizer = dynamics.load_tokenizer()
    datasets, _ = prepare_datasets(config, dynamics_config, tokenizer)
    token_ids = animal_token_ids(config, tokenizer)
    with active_lock():
        for receiver in RECEIVERS:
            for seed in SEEDS:
                subset = [row for row in targets if row[0] == receiver and row[1] == seed]
                if not subset:
                    continue
                pending = [row for row in subset if not row[4].exists()]
                if not pending:
                    for _, _, condition, update, path in subset:
                        validate_cell(
                            path, config, runner_lock,
                            receiver, seed, condition, update,
                        )
                    continue
                owner = None
                try:
                    owner = load_model(config, receiver, seed)
                    for _, _, condition, update, _ in subset:
                        run_cell(
                            owner, tokenizer, token_ids, datasets,
                            config, ds2_config, dynamics_config, runner_lock,
                            receiver, seed, condition, update,
                        )
                finally:
                    release_model(owner)
    print("CROSS-GRADIENT CELLS COMPLETE FOR SELECTED SCOPE", flush=True)


def status_report() -> dict[str, Any]:
    config, _, _ = load_and_validate_config()
    runner_lock = validate_runner_lock(config)
    completed: list[str] = []
    missing: list[str] = []
    invalid: list[dict[str, str]] = []
    for receiver, seed, condition, update, path in expected_cells():
        key = cell_key(receiver, seed, condition, update)
        if not path.exists():
            missing.append(key)
            continue
        try:
            validate_cell(
                path, config, runner_lock, receiver, seed, condition, update
            )
            completed.append(key)
        except Exception as error:  # status must inventory every failed cell
            invalid.append({"cell": key, "error": repr(error)})
    report = {
        "name": "numeric-wolf-cross-gradient-localization-v1-status",
        "expected_cells": len(expected_cells()),
        "completed_cells": len(completed),
        "missing_cells": missing,
        "invalid_cells": invalid,
        "complete": (
            len(completed) == len(expected_cells()) and not missing and not invalid
        ),
        "aggregate_json_exists": OUT_JSON.is_file(),
        "aggregate_markdown_exists": OUT_MD.is_file(),
    }
    return report


def load_completed_results(
    config: dict[str, Any], runner_lock: dict[str, Any]
) -> dict[tuple[str, int, str, int], dict[str, Any]]:
    results = {}
    for receiver, seed, condition, update, path in expected_cells():
        if not path.is_file():
            raise RuntimeError(f"Missing completed cell: {path}")
        results[(receiver, seed, condition, update)] = validate_cell(
            path, config, runner_lock, receiver, seed, condition, update
        )
    return results


def matrix_from_result(
    result: dict[str, Any], metric: str, split: str, group: str
) -> np.ndarray:
    value = np.asarray(
        result["cross_gradient"][metric][split][group][
            "cluster_by_numeric_block"
        ],
        dtype=np.float64,
    )
    if value.shape != (6, 8):
        raise RuntimeError("Analysis received malformed matrix")
    return value


def average_state_local_matrices(
    preference_state_matrix: np.ndarray,
    control_state_matrix: np.ndarray,
) -> np.ndarray:
    """Average already-computed state-local kappas; never cross gradients."""
    if preference_state_matrix.shape != (6, 8) or control_state_matrix.shape != (6, 8):
        raise RuntimeError("State-local matrix shape changed")
    return 0.5 * (preference_state_matrix + control_state_matrix)


def phase_matrix(
    results: dict[tuple[str, int, str, int], dict[str, Any]],
    receiver: str,
    seed: int,
    checkpoints: Iterable[int],
    metric: str,
    split: str,
    group: str,
) -> np.ndarray:
    state_local: list[np.ndarray] = []
    for update in checkpoints:
        preference = matrix_from_result(
            results[(receiver, seed, "preference", int(update))],
            metric, split, group,
        )
        control = matrix_from_result(
            results[(receiver, seed, "control", int(update))],
            metric, split, group,
        )
        state_local.append(average_state_local_matrices(preference, control))
    if not state_local:
        raise RuntimeError("Cannot average an empty checkpoint phase")
    return np.mean(np.stack(state_local, axis=0), axis=0)


def bootstrap_contrasts(
    matrices: dict[str, np.ndarray], resamples: int, seed: int,
    expected_draw_hashes: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    if any(matrix.shape != (6, 8) for matrix in matrices.values()):
        raise RuntimeError("Bootstrap matrix shape changed")
    rng = np.random.default_rng(seed)
    prompt_indices = rng.integers(0, 6, size=(resamples, 6))
    numeric_indices = rng.integers(0, 8, size=(resamples, 8))
    observed_draw_hashes = {
        "prompt_cluster_draws_int64_sha256": int64_sha256(
            prompt_indices.tolist()
        ),
        "numeric_block_draws_int64_sha256": int64_sha256(
            numeric_indices.tolist()
        ),
    }
    if (
        expected_draw_hashes is not None
        and observed_draw_hashes != expected_draw_hashes
    ):
        raise RuntimeError(
            f"Frozen bootstrap resamples changed: {observed_draw_hashes}"
        )
    output = {}
    for name, matrix in matrices.items():
        samples = matrix[
            prompt_indices[:, :, None], numeric_indices[:, None, :]
        ].mean(axis=(1, 2))
        output[name] = {
            "point": float(matrix.mean()),
            "ci_low": float(np.percentile(samples, 2.5)),
            "ci_high": float(np.percentile(samples, 97.5)),
            "bootstrap_mean": float(samples.mean()),
        }
    return output


def mean_geometry_summary(
    results: dict[tuple[str, int, str, int], dict[str, Any]],
    receiver: str,
    seed: int,
    checkpoints: Iterable[int],
    metric: str,
    split: str,
    group: str,
) -> dict[str, float]:
    records = [
        results[(receiver, seed, condition, int(update))]["cross_gradient"][metric][split][group]
        for update in checkpoints for condition in CONDITIONS
    ]
    return {
        "mean_kappa": float(np.mean([float(row["mean"]) for row in records])),
        "mean_cosine": float(np.mean([float(row["mean_cosine"]) for row in records])),
        "mean_behavior_gradient_l2": float(np.mean([
            float(np.mean(row["behavior_gradient_l2_by_cluster"])) for row in records
        ])),
        "mean_numeric_direction_l2": float(np.mean([
            float(np.mean(row["numeric_direction_l2_by_block"])) for row in records
        ])),
    }


def primary_analysis(
    config: dict[str, Any],
    results: dict[tuple[str, int, str, int], dict[str, Any]],
) -> dict[str, Any]:
    analysis = config["frozen_analysis"]
    checkpoints = tuple(int(value) for value in analysis["primary_checkpoints"])
    resamples = int(analysis["bootstrap"]["resamples"])
    base_seed = int(analysis["bootstrap"]["seed"])
    by_seed = {}
    pass_by_seed = {}
    point_nonpositive = {"total": [], "late_dominance": [], "module_dominance": []}
    expected_draw_hashes = {
        "prompt_cluster_draws_int64_sha256": analysis["bootstrap"][
            "prompt_cluster_draws_int64_sha256"
        ],
        "numeric_block_draws_int64_sha256": analysis["bootstrap"][
            "numeric_block_draws_int64_sha256"
        ],
    }
    for seed in SEEDS:
        def pm(group: str) -> np.ndarray:
            return phase_matrix(
                results, "ds2", seed, checkpoints, "raw", "primary", group
            )

        contrast_matrices = {
            "total": pm("all"),
            "late_dominance": (
                pm("band_late") - pm("band_early") - pm("band_middle")
            ),
            "module_dominance": (
                pm("module_query_key_value") + pm("module_dense_4h_to_h")
                - pm("module_dense") - pm("module_dense_h_to_4h")
            ),
        }
        intervals = bootstrap_contrasts(
            contrast_matrices, resamples, base_seed, expected_draw_hashes
        )
        passes = {name: row["ci_low"] > 0.0 for name, row in intervals.items()}
        for name, row in intervals.items():
            point_nonpositive[name].append(row["point"] <= 0.0)
        pass_by_seed[str(seed)] = passes
        by_seed[str(seed)] = {
            "contrasts": intervals,
            "passes": passes,
            "raw_geometry": {
                group: mean_geometry_summary(
                    results, "ds2", seed, checkpoints, "raw", "primary", group
                )
                for group in (
                    "all", "band_early", "band_middle", "band_late",
                    "module_query_key_value", "module_dense",
                    "module_dense_h_to_4h", "module_dense_4h_to_h",
                    "lora_A", "lora_B",
                )
            },
            "fixed_old_v_geometry": {
                group: mean_geometry_summary(
                    results, "ds2", seed, checkpoints,
                    "fixed_old_v", "primary", group,
                )
                for group in (
                    "all", "band_early", "band_middle", "band_late",
                    "module_query_key_value", "module_dense",
                    "module_dense_h_to_4h", "module_dense_4h_to_h",
                    "lora_A", "lora_B",
                )
            },
        }
    all_pass = all(all(values.values()) for values in pass_by_seed.values())
    evidence_against = any(all(values) for values in point_nonpositive.values())
    classification = (
        "heldout_localization_supported"
        if all_pass
        else "evidence_against" if evidence_against else "mixed"
    )
    return {
        "classification": classification,
        "conditional_replication_scope": (
            "Held-out prompt/data assay, conditional on the same two saved seeds "
            "and trajectories; not a new model- or seed-level confirmation."
        ),
        "state_average_estimand": config["scope"]["paired_state_estimand"],
        "cross_state_terms_computed": 0,
        "checkpoints": list(checkpoints),
        "prompt_split": "primary",
        "metric": "raw",
        "bootstrap_resamples": resamples,
        "bootstrap_seed": base_seed,
        "bootstrap_draw_hashes": expected_draw_hashes,
        "by_seed": by_seed,
    }


def trajectory_analysis(
    results: dict[tuple[str, int, str, int], dict[str, Any]]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    groups = (
        "all", "band_early", "band_middle", "band_late",
        "module_query_key_value", "module_dense",
        "module_dense_h_to_4h", "module_dense_4h_to_h",
        "lora_A", "lora_B",
    )
    for receiver in RECEIVERS:
        output[receiver] = {}
        for seed in SEEDS:
            rows = []
            for update in CHECKPOINTS[receiver]:
                row: dict[str, Any] = {"optimizer_update": update, "metrics": {}}
                for metric in METRICS:
                    row["metrics"][metric] = {}
                    for split in BEHAVIOR_SPLITS:
                        row["metrics"][metric][split] = {}
                        for group in groups:
                            matrix = phase_matrix(
                                results, receiver, seed, (update,), metric, split, group
                            )
                            records = [
                                results[(receiver, seed, condition, update)][
                                    "cross_gradient"
                                ][metric][split][group]
                                for condition in CONDITIONS
                            ]
                            row["metrics"][metric][split][group] = {
                                "kappa": float(matrix.mean()),
                                "mean_cosine": float(np.mean([
                                    float(record["mean_cosine"]) for record in records
                                ])),
                                "mean_behavior_gradient_l2": float(np.mean([
                                    float(np.mean(record["behavior_gradient_l2_by_cluster"]))
                                    for record in records
                                ])),
                                "mean_numeric_direction_l2": float(np.mean([
                                    float(np.mean(record["numeric_direction_l2_by_block"]))
                                    for record in records
                                ])),
                            }
                rows.append(row)
            output[receiver][str(seed)] = rows
    return output


def full_scalar_atlas_index(
    results: dict[tuple[str, int, str, int], dict[str, Any]]
) -> dict[str, Any]:
    """Index every state/checkpoint/group scalar record without duplicating it."""
    groups = group_inventory()
    families = {
        "all": ["all"],
        "layers": [f"layer_{layer:02d}" for layer in range(12)],
        "bands": ["band_early", "band_middle", "band_late"],
        "modules": [f"module_{module}" for module in MODULES],
        "adapter_sides": list(SIDES),
        "layer_by_module_intersections": [
            f"layer_{layer:02d}__module_{module}"
            for layer in range(12) for module in MODULES
        ],
    }
    flattened = [group for values in families.values() for group in values]
    if flattened != groups or len(flattened) != len(set(flattened)):
        raise RuntimeError("Full scalar atlas group partition changed")
    cells: dict[str, Any] = {}
    for receiver, seed, condition, update, cell_path in expected_cells():
        result = results[(receiver, seed, condition, update)]
        result_path = ROOT / result["attempt"] / "result.json"
        if not result_path.is_file():
            raise RuntimeError(f"Atlas result artifact is missing: {result_path}")
        key = cell_key(receiver, seed, condition, update)
        cells[key] = {
            "receiver": receiver,
            "seed": seed,
            "source_condition": condition,
            "optimizer_update": update,
            "cell": artifact_record(cell_path),
            "scalar_result": artifact_record(result_path),
        }
    if len(cells) != len(expected_cells()):
        raise RuntimeError("Full scalar atlas cell inventory changed")
    return {
        "scope": (
            "Complete index of every frozen receiver, seed, saved-state "
            "condition, checkpoint, metric, behavior split, and parameter group. "
            "Referenced result JSON files contain scalar matrices/norms/cosines "
            "only and no parameter, optimizer, or gradient tensors."
        ),
        "cell_count": len(cells),
        "metric_inventory": list(METRICS),
        "behavior_split_inventory": list(BEHAVIOR_SPLITS),
        "group_count": len(groups),
        "group_families": families,
        "record_json_pointer_template": (
            "/cross_gradient/{metric}/{behavior_split}/{group}"
        ),
        "record_fields": [
            "cluster_by_numeric_block",
            "mean",
            "behavior_gradient_l2_by_cluster",
            "numeric_direction_l2_by_block",
            "cosine_cluster_by_numeric_block",
            "mean_cosine",
        ],
        "cells": cells,
        "all_frozen_groups_indexed": True,
        "no_tensor_outputs": True,
    }


def ws3_phase_summary(
    results: dict[tuple[str, int, str, int], dict[str, Any]]
) -> dict[str, Any]:
    phases = {
        "early": (64, 128),
        "transition": (256, 512),
        "attenuation": (1024, 2048),
    }
    output = {}
    for seed in SEEDS:
        output[str(seed)] = {}
        for phase, updates in phases.items():
            output[str(seed)][phase] = {
                group: mean_geometry_summary(
                    results, "weight_seed3", seed, updates,
                    "raw", "primary", group,
                )
                for group in (
                    "all", "band_early", "band_middle", "band_late",
                    "module_query_key_value", "module_dense_4h_to_h",
                )
            }
    return {
        "scope": (
            "Descriptive temporal availability only. This does not identify an "
            "alternative circuit or classify route replacement."
        ),
        "by_seed": output,
    }


def markdown_report(aggregate: dict[str, Any]) -> str:
    primary = aggregate["primary"]
    lines = [
        "# Numeric--wolf cross-gradient localization v1",
        "",
        f"Classification: **{primary['classification']}**",
        "",
        primary["conditional_replication_scope"],
        "",
        "## Primary ds2 held-out assay",
        "",
        "| seed | total kappa [95%] | late-(early+middle) [95%] | (QKV+MLP-out)-(attn-out+MLP-in) [95%] | all pass |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for seed in SEEDS:
        row = primary["by_seed"][str(seed)]
        values = []
        for name in ("total", "late_dominance", "module_dominance"):
            item = row["contrasts"][name]
            values.append(
                f"{item['point']:+.6g} [{item['ci_low']:+.6g}, {item['ci_high']:+.6g}]"
            )
        lines.append(
            f"| {seed} | {values[0]} | {values[1]} | {values[2]} | "
            f"{all(row['passes'].values())} |"
        )
    lines.extend([
        "",
        "Each preference/control saved-state dot product was computed locally before the two state matrices were averaged; no cross-state gradient terms were formed.",
        "",
        "Raw behavior-gradient norms, numeric-gradient norms, cosines, all layers/modules/sides, fixed-old-v results, and the complete weight-seed3 trajectory are retained in the JSON.",
        "",
        "The aggregate full-scalar-atlas index resolves every receiver/seed/state/checkpoint to its validated scalar result and enumerates all 70 frozen groups, including every layer-by-module intersection.",
        "",
        "The weight-seed3 time series is descriptive evidence about local route availability, not proof of solution replacement.",
    ])
    return "\n".join(lines) + "\n"


def analyze() -> dict[str, Any]:
    config, _, _ = load_and_validate_config()
    runner_lock = validate_runner_lock(config)
    results = load_completed_results(config, runner_lock)
    aggregate = {
        "name": "numeric-wolf-cross-gradient-localization-v1",
        "completed_at": utc_now(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "runner_sha256": file_sha256(SCRIPT_PATH),
        "runner_lock_sha256": file_sha256(RUNNER_LOCK_PATH),
        "cell_count": len(results),
        "primary": primary_analysis(config, results),
        "trajectory": trajectory_analysis(results),
        "weight_seed3_phase_summary": ws3_phase_summary(results),
        "full_scalar_atlas_index": full_scalar_atlas_index(results),
        "scope": config["scope"],
        "no_tensor_outputs": True,
    }
    if not finite_tree(aggregate):
        raise RuntimeError("Non-finite aggregate")
    atomic_write_json(OUT_JSON, aggregate)
    atomic_write_text(OUT_MD, markdown_report(aggregate))
    print(
        "CROSS-GRADIENT ANALYSIS DONE",
        aggregate["primary"]["classification"],
        flush=True,
    )
    return aggregate


def _synthetic_lora_vectors() -> tuple[
    list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]
]:
    """Build deterministic tiny vectors spanning every frozen parameter group."""
    behavior: list[dict[str, torch.Tensor]] = []
    directions: list[dict[str, torch.Tensor]] = []
    names = [
        (
            f"base_model.model.gpt_neox.layers.{layer}.{module}."
            f"{side}.default.weight"
        )
        for layer in range(12)
        for module in MODULES
        for side in SIDES
    ]
    if len(names) != 96:
        raise RuntimeError("Synthetic LoRA inventory changed")
    for cluster in range(6):
        behavior.append({
            name: torch.tensor(
                [((index % 13) - 6) * 0.01 + (cluster + 1) * 0.001],
                dtype=torch.float64,
            )
            for index, name in enumerate(names)
        })
    for block in range(8):
        directions.append({
            name: torch.tensor(
                [((index % 11) - 5) * 0.02 - (block + 1) * 0.0007],
                dtype=torch.float64,
            )
            for index, name in enumerate(names)
        })
    return behavior, directions


def self_test() -> dict[str, Any]:
    """Exercise grouping, geometry, bootstrap, and state averaging without a model."""
    config = load_json(CONFIG_PATH)
    validate_config_contract(config)

    synthetic_update_rows = []
    for update in range(1, 513):
        values = {
            "optimizer_update": update,
            "epoch": (update - 1) // 512,
            "mean_microbatch_loss": 1.0 / update,
            "gradient_norm_before_clipping": 0.5 + update * 1e-6,
            "learning_rates_after_update": [2e-4],
        }
        # Reproduce JSON loaded-key order, which differs from the explicit
        # legacy runtime insertion order frozen above.
        synthetic_update_rows.append({key: values[key] for key in sorted(values)})
    synthetic_legacy_hash = compact_hash(
        legacy_ordered_update_metrics(synthetic_update_rows)
    )
    synthetic_guard = {
        "first_512_update_metrics_sha256": synthetic_legacy_hash
    }
    synthetic_trajectory = {
        "update512_replay_guard": dict(synthetic_guard),
        "probes": [{
            "optimizer_update": 512,
            "update512_replay_guard": dict(synthetic_guard),
        }],
    }
    synthetic_metrics = {
        "update_metrics": synthetic_update_rows,
        "probe_metrics": [{
            "optimizer_update": 512,
            "update512_replay_guard": dict(synthetic_guard),
        }],
    }
    replay_hash_test = legacy_replay_hash_inputs(
        synthetic_trajectory, synthetic_metrics
    )
    if (
        replay_hash_test["stored_parent_sha256"] != synthetic_legacy_hash
        or replay_hash_test["legacy_ordered_projection_sha256"]
        != synthetic_legacy_hash
        or replay_hash_test["loaded_order_parent_metrics_sha256"]
        == synthetic_legacy_hash
    ):
        raise RuntimeError("Legacy replay-hash compatibility self-test failed")

    behavior, directions = _synthetic_lora_vectors()
    synthetic_behavior = {
        "discovery": behavior,
        "primary": [
            {name: tensor * 1.5 for name, tensor in vector.items()}
            for vector in behavior
        ],
    }
    raw = cross_matrices(synthetic_behavior, directions)
    scaled = cross_matrices(
        synthetic_behavior,
        [
            {name: tensor * 0.75 for name, tensor in vector.items()}
            for vector in directions
        ],
    )
    cross = {"raw": raw, "fixed_old_v": scaled}
    additivity = matrix_additivity_error(cross)
    if max(additivity.values()) > 1e-12:
        raise RuntimeError(f"Synthetic group additivity failed: {additivity}")
    for metric in METRICS:
        for split in BEHAVIOR_SPLITS:
            for record in cross[metric][split].values():
                _validate_matrix_record(record)

    # The primary estimand is the mean of two within-state dots. Averaging
    # b and g first would introduce the two forbidden cross-state products.
    preference_local = np.full((6, 8), -2.0 * 3.0, dtype=np.float64)
    control_local = np.full((6, 8), -5.0 * 7.0, dtype=np.float64)
    observed_local_average = average_state_local_matrices(
        preference_local, control_local
    )
    expected_local_average = -20.5
    forbidden_cross_state_average = -((2.0 + 5.0) / 2.0) * ((3.0 + 7.0) / 2.0)
    if not np.all(observed_local_average == expected_local_average):
        raise RuntimeError("State-local averaging test failed")
    if expected_local_average == forbidden_cross_state_average:
        raise RuntimeError("State-local test did not distinguish cross-state terms")

    bootstrap = bootstrap_contrasts(
        {
            "positive": np.ones((6, 8), dtype=np.float64),
            "negative": -np.ones((6, 8), dtype=np.float64),
        },
        resamples=512,
        seed=59411,
    )
    if not (
        bootstrap["positive"]["ci_low"] > 0.0
        and bootstrap["negative"]["ci_high"] < 0.0
    ):
        raise RuntimeError("Bootstrap sign test failed")

    report = {
        "name": "numeric-wolf-cross-gradient-localization-v1-self-test",
        "passed": True,
        "model_loaded": False,
        "optimizer_step_taken": False,
        "mps_used": False,
        "group_count": len(group_inventory()),
        "maximum_additivity_error": max(additivity.values()),
        "state_local_average": expected_local_average,
        "forbidden_cross_state_average": forbidden_cross_state_average,
        "bootstrap": bootstrap,
        "legacy_replay_hash_compatibility": {
            "passed": True,
            "legacy_ordered_projection_sha256": synthetic_legacy_hash,
            "loaded_order_sha256": replay_hash_test[
                "loaded_order_parent_metrics_sha256"
            ],
        },
    }
    if not finite_tree(report):
        raise RuntimeError("Non-finite self-test report")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Heldout numeric--wolf cross-gradient localization"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="freeze and validate all parents")
    run_parser = subparsers.add_parser("run", help="compute selected scalar cells")
    run_parser.add_argument("--receiver", choices=RECEIVERS)
    run_parser.add_argument("--seed", type=int, choices=SEEDS)
    run_parser.add_argument("--condition", choices=CONDITIONS)
    run_parser.add_argument("--update", type=int)
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
    else:  # pragma: no cover - argparse constrains this branch
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
