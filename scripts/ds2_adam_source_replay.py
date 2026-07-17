"""Deterministic Stage-A replay for the ds2 Adam-source factorial.

This module reconstructs the four canonical data-seed2 natural trajectories
(two student seeds by preference/control data) through optimizer update 512.
It writes named LoRA/AdamW *analysis* snapshots at the seven frozen
checkpoints, but never serializes a base model, merged model, or factorial
branch.  Every update is checked against the completed wolf-route natural
cell before a replay can become reusable.

``cell.json`` is the only reusable sentinel.  Interrupted numbered attempts
and any snapshots already written inside them are deliberately preserved.
The public functions are also imported by ``ds2_adam_source_factorial.py``.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import peft
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm

import numeric_fingerprint_compatibility as compatibility
import numeric_fingerprint_dynamics as dynamics
import wolf_route_knockout as wolf
from polypythia_sl.data import read_jsonl
from polypythia_sl.optim import build_optimizer
from polypythia_sl.train import CompletionDataset


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs/ds2_adam_source_factorial_v1.json"
SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_CONFIG_SHA256 = (
    "bfa725dd5b46e6a7dad7fcc7adfa6290c5c29a6129f48b5c7d204c1664e07e1c"
)
WORK = ROOT / "runs/ds2_adam_source_factorial_v1"
REPLAYS = WORK / "replays"
RUNNER_LOCK_PATH = WORK / "runner_lock.json"
ACTIVE_LOCK_PATH = WORK / ".active.lock"
CONDITIONS = ("preference", "control")
CHECKPOINTS = (8, 16, 32, 64, 128, 256, 512)
SEEDS = (56101, 56102)
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def int64_sha256(values: Iterable[int]) -> str:
    array = np.asarray(list(values), dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def exclusive_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def _guard_pair(pair: Any, label: str) -> Path:
    if not (
        isinstance(pair, list)
        and len(pair) == 2
        and all(isinstance(value, str) for value in pair)
    ):
        raise RuntimeError(f"Malformed frozen path/hash pair: {label}")
    path = ROOT / pair[0]
    if not path.is_file() or file_sha256(path) != pair[1]:
        raise RuntimeError(f"Frozen input changed: {label}: {path}")
    return path


def expected_cell_path(seed: int, condition: str) -> Path:
    return REPLAYS / f"seed_{seed}" / condition / "cell.json"


def expected_cell_paths(config: dict[str, Any]) -> list[Path]:
    return [
        expected_cell_path(int(seed), condition)
        for seed in config["training"]["student_seeds"]
        for condition in CONDITIONS
    ]


def expected_snapshot_path(attempt: Path, update: int) -> Path:
    if update not in CHECKPOINTS:
        raise RuntimeError(f"Unfrozen snapshot update: {update}")
    return attempt / f"state_u{update:04d}.pt"


def load_and_validate_config() -> dict[str, Any]:
    observed_config_hash = file_sha256(CONFIG_PATH)
    if observed_config_hash != EXPECTED_CONFIG_SHA256:
        raise RuntimeError(
            "Factorial config changed after scientific freeze: "
            f"{observed_config_hash} != {EXPECTED_CONFIG_SHA256}"
        )
    config = load_json(CONFIG_PATH)
    if config.get("name") != "ds2-adam-source-factorial-v1":
        raise RuntimeError("Unexpected factorial config")
    if tuple(config["data"]["conditions"]) != CONDITIONS:
        raise RuntimeError("Condition inventory changed")
    if tuple(config["training"]["student_seeds"]) != SEEDS:
        raise RuntimeError("Replay seed inventory changed")
    if tuple(config["measurement"]["checkpoints"]) != CHECKPOINTS:
        raise RuntimeError("Replay checkpoint grid changed")
    if int(config["measurement"]["replay_cell_count"]) != len(
        expected_cell_paths(config)
    ):
        raise RuntimeError("Replay cell-count guard failed")
    training = config["training"]
    if (
        int(training["replay_updates"]) != 512
        or int(training["examples_per_update"]) != 16
        or int(training["batch_size"])
        * int(training["gradient_accumulation_steps"])
        != int(training["examples_per_update"])
    ):
        raise RuntimeError("Replay horizon/effective-batch guard changed")
    mirrored = (
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "optimizer",
        "betas",
        "eps",
        "weight_decay",
        "max_grad_norm",
        "warmup_updates",
        "schedule_total_updates",
        "max_length",
        "lora",
        "expected_trainable_parameters",
        "expected_initial_lora_state_sha256",
    )
    if any(config["measurement"].get(key) != training.get(key) for key in mirrored):
        raise RuntimeError("Measurement/training replay parameters diverged")
    parents = config["parents"]
    for key in (
        "wolf_route_config",
        "wolf_route_runner",
        "wolf_route_runner_lock",
        "wolf_route_result",
        "dynamics_config",
        "heldout_manifest",
    ):
        _guard_pair(parents[key], f"parents.{key}")
    for key, pair in parents["natural_cells"].items():
        _guard_pair(pair, f"parents.natural_cells.{key}")
    for key, pair in parents["dependencies"].items():
        _guard_pair(pair, f"parents.dependencies.{key}")
    for condition in CONDITIONS:
        pool = ROOT / config["data"][f"{condition}_pool"]
        heldout = ROOT / config["data"][f"heldout_{condition}"]
        if file_sha256(pool) != config["data"][f"{condition}_pool_sha256"]:
            raise RuntimeError(f"Training pool changed: {pool}")
        if file_sha256(heldout) != config["data"][f"heldout_{condition}_sha256"]:
            raise RuntimeError(f"Held-out pool changed: {heldout}")
    if int(config["guards"]["snapshot_max_bytes"]) > 16 * 1024**2:
        raise RuntimeError("Snapshot size scope expanded")
    if tuple(config["guards"]["archived_semantic_hash_checkpoints"]) != (
        16,
        64,
        128,
        256,
        512,
    ) or tuple(config["guards"]["scalar_replay_only_checkpoints"]) != (8, 32):
        raise RuntimeError("Archived/scalar-only replay guard split changed")
    if config["guards"]["recursive_parent_child_hash_validation"] is not True:
        raise RuntimeError("Recursive replay-manifest validation was disabled")
    potential_snapshot_bytes = (
        len(expected_cell_paths(config))
        * len(CHECKPOINTS)
        * int(config["guards"]["snapshot_max_bytes"])
    )
    if potential_snapshot_bytes > int(
        config["resource_policy"]["expected_snapshot_bytes_total_upper_bound"]
    ):
        raise RuntimeError("Snapshot campaign upper bound is internally inconsistent")
    if config["artifacts"]["runner_lock"] != relative(RUNNER_LOCK_PATH):
        raise RuntimeError("Runner-lock namespace changed")
    if config["artifacts"]["replay_pattern"] != (
        "replays/seed_{seed}/{condition}/cell.json"
    ):
        raise RuntimeError("Replay namespace changed")
    if config["artifacts"]["replay_snapshot_pattern"] != (
        "replays/seed_{seed}/{condition}/attempt_{NNN}/state_u{update:04d}.pt"
    ):
        raise RuntimeError("Replay snapshot namespace changed")
    return config


def implementation_guard(config: dict[str, Any] | None = None) -> dict[str, Any]:
    if config is None:
        config = load_and_validate_config()
    return {
        "replay_runner_sha256": file_sha256(SCRIPT_PATH),
        "config_sha256": file_sha256(CONFIG_PATH),
        "wolf_route_runner_sha256": file_sha256(Path(wolf.__file__).resolve()),
        "dynamics_runner_sha256": file_sha256(Path(dynamics.__file__).resolve()),
        "compatibility_runner_sha256": file_sha256(
            Path(compatibility.__file__).resolve()
        ),
        "optim_py_sha256": file_sha256(ROOT / "src/polypythia_sl/optim.py"),
        "train_py_sha256": file_sha256(ROOT / "src/polypythia_sl/train.py"),
        "data_py_sha256": file_sha256(ROOT / "src/polypythia_sl/data.py"),
        "modeling_py_sha256": file_sha256(ROOT / "src/polypythia_sl/modeling.py"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "numpy": np.__version__,
        "device": str(DEVICE),
        "platform": platform.platform(),
    }


def assert_no_competing_experiment() -> None:
    output = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid=,command="], text=True
    )
    processes: dict[int, tuple[int, str]] = {}
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3:
            continue
        try:
            processes[int(fields[0])] = (int(fields[1]), fields[2])
        except ValueError:
            continue
    ancestors = {os.getpid()}
    cursor = os.getpid()
    while cursor in processes:
        cursor = processes[cursor][0]
        if cursor <= 0 or cursor in ancestors:
            break
        ancestors.add(cursor)
    markers = (
        "scripts/numeric_",
        "scripts/dataorder_",
        "scripts/base_screening.py",
        "scripts/student_trait_write_probe.py",
        "scripts/cross_family_transport.py",
        "scripts/optimizer_transplant",
        "scripts/wolf_route_knockout.py run",
        "scripts/ds2_adam_source_replay.py replay",
        "scripts/ds2_adam_source_factorial.py run",
        "polypythia_sl.pipeline",
    )
    conflicts = []
    for pid, (_, command) in processes.items():
        if pid in ancestors or "python" not in command.lower():
            continue
        if any(marker in command for marker in markers):
            conflicts.append(f"{pid} {command}")
    if conflicts:
        raise RuntimeError("Competing experiment process:\n" + "\n".join(conflicts))


def two_epoch_orders(config: dict[str, Any], seed: int) -> list[list[int]]:
    count = int(config["data"]["rows_per_condition"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dynamics.IndexDataset(count),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    orders: list[list[int]] = []
    frozen = config["training"]["two_epoch_order_guards"][str(seed)]
    for epoch in range(2):
        order = [int(value) for batch in loader for value in batch.tolist()]
        if sorted(order) != list(range(count)):
            raise RuntimeError(f"Non-permutation order for seed {seed}/epoch {epoch}")
        if (
            order[:16] != frozen["epoch_first_16"][epoch]
            or int64_sha256(order) != frozen["epoch_sha256"][epoch]
        ):
            raise RuntimeError(f"Frozen data order changed for seed {seed}/epoch {epoch}")
        orders.append(order)
    return orders


def load_rows(
    config: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    training = {
        condition: read_jsonl(ROOT / config["data"][f"{condition}_pool"])
        for condition in CONDITIONS
    }
    heldout = {
        condition: read_jsonl(ROOT / config["data"][f"heldout_{condition}"])
        for condition in CONDITIONS
    }
    if any(
        len(rows) != int(config["data"]["rows_per_condition"])
        for rows in training.values()
    ):
        raise RuntimeError("Training row-count guard failed")
    if any(
        len(rows) != int(config["data"]["heldout_rows_per_condition"])
        for rows in heldout.values()
    ):
        raise RuntimeError("Held-out row-count guard failed")
    for scope, rows in (("training", training), ("heldout", heldout)):
        if [row["prompt"] for row in rows["preference"]] != [
            row["prompt"] for row in rows["control"]
        ]:
            raise RuntimeError(f"{scope.capitalize()} pools are not prompt-paired")
    overlap = {row["prompt"] for rows in training.values() for row in rows} & {
        row["prompt"] for rows in heldout.values() for row in rows
    }
    if len(overlap) != int(config["data"]["expected_training_prompt_overlap"]):
        raise RuntimeError("Training/held-out prompt-overlap guard failed")
    return training, heldout


def prepare_training_datasets(
    config: dict[str, Any], tokenizer
) -> dict[str, CompletionDataset]:
    rows, _ = load_rows(config)
    datasets = {
        condition: CompletionDataset(
            rows[condition], tokenizer, int(config["training"]["max_length"])
        )
        for condition in CONDITIONS
    }
    expected_tokens = int(config["data"]["supervised_tokens_per_row"])
    for condition, dataset in datasets.items():
        counts = {
            int((example["labels"] != -100).sum()) for example in dataset.examples
        }
        if counts != {expected_tokens}:
            raise RuntimeError(
                f"Supervised-token guard failed for {condition}: {counts}"
            )
    return datasets


def _validate_artifact(record: dict[str, Any], expected: Path) -> None:
    path = ROOT / record["path"]
    if path.resolve() != expected.resolve():
        raise RuntimeError(f"Artifact path mismatch: {path} != {expected}")
    if (
        not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or file_sha256(path) != record["sha256"]
    ):
        raise RuntimeError(f"Artifact changed: {path}")


def natural_reference_bundle(
    config: dict[str, Any], seed: int, condition: str
) -> dict[str, Any]:
    key = f"{seed}/{condition}"
    cell_path = _guard_pair(
        config["parents"]["natural_cells"][key],
        f"parents.natural_cells.{key}",
    )
    cell = load_json(cell_path)
    if (
        int(cell.get("seed", -1)) != seed
        or cell.get("student_condition") != condition
        or cell.get("intervention_rule") != "natural"
    ):
        raise RuntimeError(f"Natural source identity mismatch: {cell_path}")
    attempt = ROOT / cell["attempt"]
    metrics_path = attempt / "metrics.json"
    result_path = attempt / "result.json"
    _validate_artifact(cell["artifacts"]["metrics"], metrics_path)
    _validate_artifact(cell["artifacts"]["result"], result_path)
    metrics = load_json(metrics_path)
    result = load_json(result_path)
    updates = metrics.get("update_metrics", [])
    if len(updates) != 512 or [
        row.get("optimizer_update") for row in updates
    ] != list(range(1, 513)):
        raise RuntimeError(f"Natural update metric inventory changed: {cell_path}")
    probes = {int(row["optimizer_update"]): row for row in result["probes"]}
    parent_hashes = {
        update: probes[update].get("state_hashes")
        for update in CHECKPOINTS
    }
    archived_hash_updates = tuple(
        config["guards"]["archived_semantic_hash_checkpoints"]
    )
    scalar_only_updates = tuple(config["guards"]["scalar_replay_only_checkpoints"])
    if any(parent_hashes[update] is None for update in archived_hash_updates):
        raise RuntimeError(f"Frozen parent semantic hashes disappeared: {cell_path}")
    if any(parent_hashes[update] is not None for update in scalar_only_updates):
        raise RuntimeError(
            "Parent u8/u32 hash availability changed; revise the frozen guard explicitly"
        )
    if parent_hashes[512] != result.get("final_state_hashes"):
        raise RuntimeError(f"Natural u512 state-hash guard changed: {cell_path}")
    return {
        "cell": cell,
        "metrics_path": metrics_path,
        "result_path": result_path,
        "update_metrics": updates[:512],
        "state_hashes": parent_hashes,
    }


def preflight(require_absence: bool = False) -> dict[str, Any]:
    config = load_and_validate_config()
    assert_no_competing_experiment()
    if config["resource_policy"]["serial_mps_only"] and DEVICE.type != "mps":
        raise RuntimeError(f"Campaign requires MPS, found {DEVICE}")
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < int(config["resource_policy"]["minimum_launch_free_bytes"]):
        raise RuntimeError("Launch free-space guard failed")
    if require_absence:
        forbidden = (
            WORK,
            ROOT / config["artifacts"]["aggregate_json"],
            ROOT / config["artifacts"]["aggregate_markdown"],
            ROOT / config["artifacts"]["log"],
        )
        if any(path.exists() for path in forbidden):
            raise RuntimeError("Factorial namespace predates freeze")
    receiver = config["receiver"]
    base_guard = compatibility.cached_weight_guard("ds2")
    if (
        base_guard["resolved_commit"] != receiver["commit"]
        or base_guard["weight_sha256"] != receiver["weight_sha256"]
        or base_guard["model_config_sha256"] != receiver["model_config_sha256"]
    ):
        raise RuntimeError("Cached ds2 base guard failed")
    training, heldout = load_rows(config)
    orders = {
        str(seed): [int64_sha256(order) for order in two_epoch_orders(config, seed)]
        for seed in SEEDS
    }
    natural = {}
    for seed in SEEDS:
        for condition in CONDITIONS:
            reference = natural_reference_bundle(config, seed, condition)
            natural[f"{seed}/{condition}"] = {
                "metrics": artifact_record(reference["metrics_path"]),
                "result": artifact_record(reference["result_path"]),
                "parent_hash_updates": [
                    update
                    for update, hashes in reference["state_hashes"].items()
                    if hashes is not None
                ],
            }
    return {
        "implementation": implementation_guard(config),
        "base_guard": base_guard,
        "training_rows": {key: len(value) for key, value in training.items()},
        "heldout_rows": {key: len(value) for key, value in heldout.items()},
        "two_epoch_order_sha256": orders,
        "natural_sources": natural,
        "expected_replay_cells": [
            relative(path) for path in expected_cell_paths(config)
        ],
        "free_bytes": free_bytes,
        "preflight_used_model_forward_or_backward": False,
    }


def freeze() -> dict[str, Any]:
    if RUNNER_LOCK_PATH.exists():
        return validate_runner_lock()
    record = {
        "name": "ds2-adam-source-factorial-v1-runner-lock",
        "created_at": utc_now(),
        "absence_before_freeze": True,
        "frozen": preflight(require_absence=True),
    }
    exclusive_write_json(RUNNER_LOCK_PATH, record)
    print(f"DS2 ADAM SOURCE FACTORIAL FROZEN {file_sha256(RUNNER_LOCK_PATH)}", flush=True)
    return validate_runner_lock()


def validate_runner_lock() -> dict[str, Any]:
    if not RUNNER_LOCK_PATH.is_file():
        raise RuntimeError("Runner lock absent; freeze first")
    record = load_json(RUNNER_LOCK_PATH)
    if record.get("name") != "ds2-adam-source-factorial-v1-runner-lock":
        raise RuntimeError("Unexpected runner lock")
    config = load_and_validate_config()
    if record.get("frozen", {}).get("implementation") != implementation_guard(config):
        raise RuntimeError("Implementation changed after runner freeze")
    expected = [relative(path) for path in expected_cell_paths(config)]
    if record["frozen"].get("expected_replay_cells") != expected:
        raise RuntimeError("Frozen replay-cell inventory changed")
    return record


@contextlib.contextmanager
def active_lock():
    WORK.mkdir(parents=True, exist_ok=True)
    with ACTIVE_LOCK_PATH.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Factorial active lock is held") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def next_attempt(root: Path) -> Path:
    numbers: list[int] = []
    if root.exists():
        for path in root.iterdir():
            if path.is_dir() and path.name.startswith("attempt_"):
                suffix = path.name.removeprefix("attempt_")
                if not suffix.isdigit():
                    raise RuntimeError(f"Unexpected replay attempt directory: {path}")
                numbers.append(int(suffix))
            elif path.name != "cell.json":
                raise RuntimeError(f"Unexpected replay-root artifact: {path}")
    attempt = root / f"attempt_{max(numbers, default=0) + 1:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt


def replay_identity(seed: int, condition: str) -> dict[str, Any]:
    return {
        "name": "ds2-adam-source-factorial-v1-replay",
        "runner_lock_sha256": file_sha256(RUNNER_LOCK_PATH),
        "config_sha256": file_sha256(CONFIG_PATH),
        "receiver": "data_seed2",
        "seed": seed,
        "condition": condition,
        "optimizer_updates": 512,
    }


def write_analysis_snapshot(
    path: Path,
    owner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    seed: int,
    condition: str,
    update: int,
    lr_used_for_update: float,
) -> dict[str, Any]:
    if shutil.disk_usage(ROOT).free < int(
        config["resource_policy"]["minimum_runtime_free_bytes"]
    ):
        raise RuntimeError("Runtime free-space guard failed before snapshot")
    payload = dynamics.state_snapshot_payload(
        owner,
        optimizer,
        config,
        config["receiver"]["name"],
        seed,
        condition,
        update,
        lr_used_for_update,
    )
    dynamics.atomic_torch_save(path, payload)
    del payload
    if path.stat().st_size > int(config["guards"]["snapshot_max_bytes"]):
        raise RuntimeError(f"Analysis snapshot exceeds frozen size scope: {path}")
    return validate_analysis_snapshot(path, config, seed, condition, update)


def validate_analysis_snapshot(
    path: Path,
    config: dict[str, Any],
    seed: int,
    condition: str,
    update: int,
) -> dict[str, Any]:
    if path.resolve() != expected_snapshot_path(path.parent, update).resolve():
        raise RuntimeError(f"Snapshot path/update mismatch: {path}")
    if path.stat().st_size > int(config["guards"]["snapshot_max_bytes"]):
        raise RuntimeError(f"Analysis snapshot exceeds frozen size scope: {path}")
    return dynamics.validate_state_snapshot(
        path,
        config,
        config["receiver"]["name"],
        seed,
        condition,
        update,
    )


def _metric_replay_error(
    observed: dict[str, Any], expected: dict[str, Any], config: dict[str, Any]
) -> dict[str, float]:
    if int(observed["optimizer_update"]) != int(expected["optimizer_update"]):
        raise RuntimeError("Natural replay update indexing changed")
    errors = {
        "loss": abs(
            float(observed["mean_microbatch_loss"])
            - float(expected["mean_microbatch_loss"])
        ),
        "gradient_norm": abs(
            float(observed["gradient_norm_before_clipping"])
            - float(expected["gradient_norm_before_clipping"])
        ),
        "learning_rate_used": abs(
            float(observed["learning_rate_used"])
            - float(expected["learning_rate_used"])
        ),
        "learning_rate_after_update": abs(
            float(observed["learning_rates_after_update"][0])
            - float(expected["learning_rates_after_update"][0])
        ),
    }
    if errors["loss"] > float(config["guards"]["replay_loss_absolute_tolerance"]):
        raise RuntimeError(f"Natural loss replay failed: {errors}")
    if errors["gradient_norm"] > float(
        config["guards"]["replay_gradient_norm_absolute_tolerance"]
    ):
        raise RuntimeError(f"Natural gradient-norm replay failed: {errors}")
    lr_tolerance = float(config["guards"]["replay_learning_rate_absolute_tolerance"])
    if max(errors["learning_rate_used"], errors["learning_rate_after_update"]) > lr_tolerance:
        raise RuntimeError(f"Natural learning-rate replay failed: {errors}")
    return errors


def replay_cell(
    config: dict[str, Any],
    tokenizer,
    training_datasets: dict[str, CompletionDataset],
    seed: int,
    condition: str,
) -> dict[str, Any]:
    if seed not in SEEDS or condition not in CONDITIONS:
        raise RuntimeError(f"Unexpected replay identity: {seed}/{condition}")
    if config["resource_policy"]["serial_mps_only"] and DEVICE.type != "mps":
        raise RuntimeError(f"Campaign requires MPS, found {DEVICE}")
    if shutil.disk_usage(ROOT).free < int(
        config["resource_policy"]["minimum_runtime_free_bytes"]
    ):
        raise RuntimeError("Runtime free-space guard failed before replay")
    cell_path = expected_cell_path(seed, condition)
    if cell_path.exists():
        print(f"[{seed}/{condition}/replay] validated reuse", flush=True)
        return validate_replay_cell(cell_path, config)
    attempt = next_attempt(cell_path.parent)
    identity = replay_identity(seed, condition)
    start_path = attempt / "start_manifest.json"
    atomic_write_json(
        start_path,
        {
            **identity,
            "started_at": utc_now(),
            "attempt": relative(attempt),
            "status": "incomplete until cell.json is committed",
        },
    )
    print(f"[{seed}/{condition}/replay] {attempt.name} starting", flush=True)
    reference = natural_reference_bundle(config, seed, condition)
    owner = None
    update_records: list[dict[str, Any]] = []
    snapshots: dict[str, dict[str, Any]] = {}
    maxima = {
        "loss": 0.0,
        "gradient_norm": 0.0,
        "learning_rate_used": 0.0,
        "learning_rate_after_update": 0.0,
    }
    try:
        owner = wolf.load_owner(config, seed)
        optimizer, optimizer_metadata = build_optimizer(owner, config["training"])
        warmup = int(config["training"]["warmup_updates"])
        horizon = int(config["training"]["schedule_total_updates"])

        def lr_scale(step: int) -> float:
            if warmup and step < warmup:
                return (step + 1) / warmup
            return max(horizon - step, 0) / max(horizon - warmup, 1)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        order = two_epoch_orders(config, seed)[0]
        examples_per_update = int(config["training"]["examples_per_update"])
        progress = tqdm(total=512, desc=f"{seed}/{condition}/replay", unit="update")
        for update in range(1, 513):
            if update % 16 == 1 and shutil.disk_usage(ROOT).free < int(
                config["resource_policy"]["minimum_runtime_free_bytes"]
            ):
                raise RuntimeError("Runtime free-space guard failed")
            start = (update - 1) * examples_per_update
            indices = order[start : start + examples_per_update]
            if len(indices) != examples_per_update:
                raise RuntimeError("Frozen first-epoch order exhausted unexpectedly")
            numeric = wolf.numeric_backward(
                owner, training_datasets[condition], tokenizer, indices, config
            )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            observed = {
                "optimizer_update": update,
                "epoch": 0,
                "mean_microbatch_loss": numeric["mean_microbatch_loss"],
                "gradient_norm_before_clipping": numeric[
                    "gradient_norm_before_clipping"
                ],
                "learning_rate_used": learning_rate,
                "learning_rates_after_update": [
                    float(group["lr"]) for group in optimizer.param_groups
                ],
            }
            errors = _metric_replay_error(
                observed, reference["update_metrics"][update - 1], config
            )
            for key, value in errors.items():
                maxima[key] = max(maxima[key], value)
            update_records.append({**observed, "absolute_replay_error": errors})
            if update in CHECKPOINTS:
                path = expected_snapshot_path(attempt, update)
                validation = write_analysis_snapshot(
                    path,
                    owner,
                    optimizer,
                    config,
                    seed,
                    condition,
                    update,
                    learning_rate,
                )
                summaries = validation["summaries"]
                observed_hashes = {
                    "optimizer_update": update,
                    "lora_semantic_sha256": summaries["lora_semantic_sha256"],
                    "adam_exp_avg_semantic_sha256": summaries[
                        "adam_exp_avg_semantic_sha256"
                    ],
                    "adam_exp_avg_sq_semantic_sha256": summaries[
                        "adam_exp_avg_sq_semantic_sha256"
                    ],
                    "adam_steps_exact": True,
                }
                parent_hashes = reference["state_hashes"][update]
                if parent_hashes is not None and observed_hashes != parent_hashes:
                    raise RuntimeError(
                        f"Parent semantic state replay failed at u{update}: "
                        f"{observed_hashes} != {parent_hashes}"
                    )
                snapshots[str(update)] = {
                    "artifact": validation["artifact"],
                    "summaries": summaries,
                    "semantic_state_hashes": observed_hashes,
                    "parent_semantic_state_hashes": parent_hashes,
                    "parent_semantic_exact": parent_hashes is not None,
                    "self_validation_passed": True,
                }
                print(
                    f"[{seed}/{condition}/replay] u{update} "
                    f"loss={numeric['mean_microbatch_loss']:.6f} snapshot",
                    flush=True,
                )
            progress.update(1)
            progress.set_postfix(loss=f"{numeric['mean_microbatch_loss']:.3f}")
        progress.close()
        if set(snapshots) != {str(update) for update in CHECKPOINTS}:
            raise RuntimeError("Frozen replay snapshot inventory incomplete")
        metrics = {
            **identity,
            "optimizer": optimizer_metadata,
            "warmup_updates": warmup,
            "schedule_total_updates": horizon,
            "data_order_sha256": int64_sha256(order),
            "update_metrics": update_records,
            "maximum_absolute_replay_error": maxima,
            "all_512_updates_replayed": True,
        }
        result = {
            **identity,
            "completed_at": utc_now(),
            "checkpoints": list(CHECKPOINTS),
            "snapshots": snapshots,
            "parent_exact_hash_updates": config["guards"][
                "archived_semantic_hash_checkpoints"
            ],
            "metric_only_parent_guard_updates": config["guards"][
                "scalar_replay_only_checkpoints"
            ],
            "u512_semantic_exact": snapshots["512"]["parent_semantic_exact"],
            "all_update_metric_guards_passed": True,
            "analysis_snapshots_only": True,
            "no_base_or_merged_weights_written": True,
            "no_factorial_branch_tensors_written": True,
        }
        if not finite_tree(metrics) or not finite_tree(result):
            raise RuntimeError("Non-finite replay output")
        metrics_path = attempt / "metrics.json"
        result_path = attempt / "result.json"
        atomic_write_json(metrics_path, metrics)
        atomic_write_json(result_path, result)
    except BaseException as error:
        atomic_write_json(
            attempt / "failure.json",
            {**identity, "failed_at": utc_now(), "error": repr(error)},
        )
        raise
    finally:
        wolf.release_model(owner)
    artifacts = {
        "start_manifest": artifact_record(start_path),
        "metrics": artifact_record(metrics_path),
        "result": artifact_record(result_path),
    }
    for update in CHECKPOINTS:
        artifacts[f"snapshot_u{update:04d}"] = artifact_record(
            expected_snapshot_path(attempt, update)
        )
    cell = {
        **identity,
        "completed_at": result["completed_at"],
        "attempt": relative(attempt),
        "artifacts": artifacts,
        "maximum_absolute_replay_error": maxima,
        "final_state_hashes": snapshots["512"]["semantic_state_hashes"],
        "u512_semantic_exact": True,
    }
    exclusive_write_json(cell_path, cell)
    print(f"[{seed}/{condition}/replay] CELL DONE", flush=True)
    return validate_replay_cell(cell_path, config)


def validate_replay_cell(
    path: Path, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    if config is None:
        config = load_and_validate_config()
    if not path.is_file():
        raise FileNotFoundError(path)
    cell = load_json(path)
    seed = int(cell["seed"])
    condition = cell["condition"]
    if path.resolve() != expected_cell_path(seed, condition).resolve():
        raise RuntimeError(f"Replay cell stored under wrong identity path: {path}")
    expected_identity = replay_identity(seed, condition)
    if any(cell.get(key) != value for key, value in expected_identity.items()):
        raise RuntimeError(f"Replay cell identity mismatch: {path}")
    attempt = ROOT / cell["attempt"]
    if attempt.parent.resolve() != path.parent.resolve() or not attempt.name.startswith(
        "attempt_"
    ):
        raise RuntimeError(f"Replay attempt path mismatch: {path}")
    expected_artifacts = {
        "start_manifest": attempt / "start_manifest.json",
        "metrics": attempt / "metrics.json",
        "result": attempt / "result.json",
        **{
            f"snapshot_u{update:04d}": expected_snapshot_path(attempt, update)
            for update in CHECKPOINTS
        },
    }
    if set(cell["artifacts"]) != set(expected_artifacts):
        raise RuntimeError(f"Replay artifact inventory changed: {path}")
    for name, expected_path in expected_artifacts.items():
        _validate_artifact(cell["artifacts"][name], expected_path)
    metrics = load_json(expected_artifacts["metrics"])
    result = load_json(expected_artifacts["result"])
    if any(metrics.get(key) != value for key, value in expected_identity.items()):
        raise RuntimeError(f"Replay metric identity mismatch: {path}")
    if any(result.get(key) != value for key, value in expected_identity.items()):
        raise RuntimeError(f"Replay result identity mismatch: {path}")
    updates = metrics.get("update_metrics", [])
    if [row.get("optimizer_update") for row in updates] != list(range(1, 513)):
        raise RuntimeError(f"Replay update inventory changed: {path}")
    if metrics.get("data_order_sha256") != config["training"][
        "two_epoch_order_guards"
    ][str(seed)]["epoch_sha256"][0]:
        raise RuntimeError(f"Replay data order changed: {path}")
    maxima = metrics.get("maximum_absolute_replay_error", {})
    if (
        float(maxima.get("loss", math.inf))
        > float(config["guards"]["replay_loss_absolute_tolerance"])
        or float(maxima.get("gradient_norm", math.inf))
        > float(config["guards"]["replay_gradient_norm_absolute_tolerance"])
        or max(
            float(maxima.get("learning_rate_used", math.inf)),
            float(maxima.get("learning_rate_after_update", math.inf)),
        )
        > float(config["guards"]["replay_learning_rate_absolute_tolerance"])
    ):
        raise RuntimeError(f"Replay metric maximum failed: {path}")
    if maxima != cell.get("maximum_absolute_replay_error"):
        raise RuntimeError(f"Replay metric/cell maximum mismatch: {path}")
    reference = natural_reference_bundle(config, seed, condition)
    if result.get("checkpoints") != list(CHECKPOINTS) or set(
        result.get("snapshots", {})
    ) != {str(update) for update in CHECKPOINTS}:
        raise RuntimeError(f"Replay snapshot result inventory changed: {path}")
    snapshot_bytes = 0
    for update in CHECKPOINTS:
        snapshot_path = expected_snapshot_path(attempt, update)
        validation = validate_analysis_snapshot(
            snapshot_path, config, seed, condition, update
        )
        snapshot_bytes += int(validation["artifact"]["bytes"])
        record = result["snapshots"][str(update)]
        if (
            record.get("artifact") != validation["artifact"]
            or record.get("summaries") != validation["summaries"]
            or record.get("self_validation_passed") is not True
        ):
            raise RuntimeError(f"Replay snapshot validation record changed: {path}/u{update}")
        hashes = {
            "optimizer_update": update,
            "lora_semantic_sha256": validation["summaries"][
                "lora_semantic_sha256"
            ],
            "adam_exp_avg_semantic_sha256": validation["summaries"][
                "adam_exp_avg_semantic_sha256"
            ],
            "adam_exp_avg_sq_semantic_sha256": validation["summaries"][
                "adam_exp_avg_sq_semantic_sha256"
            ],
            "adam_steps_exact": True,
        }
        parent_hashes = reference["state_hashes"][update]
        if (
            record.get("semantic_state_hashes") != hashes
            or record.get("parent_semantic_state_hashes") != parent_hashes
            or record.get("parent_semantic_exact") != (parent_hashes is not None)
            or (parent_hashes is not None and hashes != parent_hashes)
        ):
            raise RuntimeError(f"Replay semantic hash guard failed: {path}/u{update}")
    if snapshot_bytes > len(CHECKPOINTS) * int(config["guards"]["snapshot_max_bytes"]):
        raise RuntimeError(f"Replay snapshot byte scope exceeded: {path}")
    if (
        result.get("parent_exact_hash_updates")
        != config["guards"]["archived_semantic_hash_checkpoints"]
        or result.get("metric_only_parent_guard_updates")
        != config["guards"]["scalar_replay_only_checkpoints"]
        or result.get("u512_semantic_exact") is not True
        or result.get("all_update_metric_guards_passed") is not True
        or result.get("analysis_snapshots_only") is not True
        or result.get("no_base_or_merged_weights_written") is not True
        or result.get("no_factorial_branch_tensors_written") is not True
        or cell.get("u512_semantic_exact") is not True
        or cell.get("final_state_hashes")
        != result["snapshots"]["512"]["semantic_state_hashes"]
    ):
        raise RuntimeError(f"Replay terminal guards changed: {path}")
    if not finite_tree(metrics) or not finite_tree(result) or not finite_tree(cell):
        raise RuntimeError(f"Non-finite replay artifact: {path}")
    return {
        "cell": cell,
        "metrics": metrics,
        "result": result,
        "snapshot_bytes": snapshot_bytes,
    }


def replay_all() -> dict[str, Any]:
    config = load_and_validate_config()
    validate_runner_lock()
    assert_no_competing_experiment()
    if DEVICE.type != "mps":
        raise RuntimeError(f"Campaign requires MPS, found {DEVICE}")
    with active_lock():
        tokenizer = dynamics.load_tokenizer()
        datasets = prepare_training_datasets(config, tokenizer)
        completed = 0
        for seed in SEEDS:
            for condition in CONDITIONS:
                replay_cell(config, tokenizer, datasets, seed, condition)
                completed += 1
                print(f"REPLAY PROGRESS {completed}/4", flush=True)
    report = status()
    if report["completed_replay_cells"] != report["expected_replay_cells"]:
        raise RuntimeError("Replay campaign ended without all sentinels")
    print("DS2 ADAM SOURCE REPLAY DONE", flush=True)
    return report


def _active_lock_held() -> bool:
    if not ACTIVE_LOCK_PATH.exists():
        return False
    with ACTIVE_LOCK_PATH.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return False
        except BlockingIOError:
            return True


def status() -> dict[str, Any]:
    config = load_and_validate_config()
    paths = expected_cell_paths(config)
    completed = 0
    invalid: dict[str, str] = {}
    total_snapshot_bytes = 0
    for path in paths:
        if not path.exists():
            continue
        try:
            validated = validate_replay_cell(path, config)
            completed += 1
            total_snapshot_bytes += int(validated["snapshot_bytes"])
        except BaseException as error:
            invalid[relative(path)] = repr(error)
    return {
        "runner_lock_exists": RUNNER_LOCK_PATH.exists(),
        "completed_replay_cells": completed,
        "expected_replay_cells": len(paths),
        "missing_replay_cells": [relative(path) for path in paths if not path.exists()],
        "invalid_replay_cells": invalid,
        "active_lock_held": _active_lock_held(),
        "snapshot_bytes": total_snapshot_bytes,
        "snapshot_campaign_upper_bound": int(
            config["resource_policy"]["expected_snapshot_bytes_total_upper_bound"]
        ),
        "free_bytes": shutil.disk_usage(ROOT).free,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "replay", "status"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--condition", choices=CONDITIONS)
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(), indent=2, sort_keys=True))
    elif args.command == "freeze":
        print(json.dumps(freeze(), indent=2, sort_keys=True))
    elif args.command == "replay":
        if (args.seed is None) != (args.condition is None):
            parser.error("--seed and --condition must be supplied together")
        if args.seed is None:
            replay_all()
        else:
            config = load_and_validate_config()
            validate_runner_lock()
            assert_no_competing_experiment()
            if DEVICE.type != "mps":
                raise RuntimeError(f"Campaign requires MPS, found {DEVICE}")
            with active_lock():
                tokenizer = dynamics.load_tokenizer()
                datasets = prepare_training_datasets(config, tokenizer)
                replay_cell(config, tokenizer, datasets, args.seed, args.condition)
    else:
        print(json.dumps(status(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
