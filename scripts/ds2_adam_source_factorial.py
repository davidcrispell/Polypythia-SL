"""Exact ds2 Adam-source factorial at frozen replay checkpoints.

Stage A (``ds2_adam_source_replay.py``) owns exact natural replay and named
LoRA/AdamW snapshots.  This runner treats those snapshots as immutable inputs.
At each seed/checkpoint it crosses live LoRA parameters (T), Adam first moments
(M), Adam second moments (V), and the next preference/control numeric batch (D).

Every native branch is checked against a real ``torch.optim.AdamW.step``.  A
secondary branch rescales all sixteen adaptive displacements at a checkpoint to
one symmetric norm, sqrt(||A_PPPP|| ||A_CCCC||).  No branch tensors are written;
only scalar diagnostics, per-prompt behavior, and per-row NLLs are retained.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import hashlib
import itertools
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

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import peft
import torch
import transformers
from torch.utils.data import DataLoader

import ds2_adam_source_replay as replay
import numeric_fingerprint_dynamics as dynamics
import numeric_fingerprint_update_geometry as geometry
import wolf_route_knockout as knockout
from polypythia_sl.data import PREFERENCE_EVAL_PROMPTS
from polypythia_sl.train import CompletionDataset


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs/ds2_adam_source_factorial_v1.json"
SCRIPT_PATH = Path(__file__).resolve()
WORK = ROOT / "runs/ds2_adam_source_factorial_v1"
NORMS = WORK / "norms"
CELLS = WORK / "cells"
RUNNER_LOCK_PATH = WORK / "runner_lock.json"
FACTORIAL_LOCK_PATH = WORK / "factorial_runner_lock.json"
ACTIVE_LOCK_PATH = WORK / ".factorial_active.lock"
OUT_JSON = ROOT / "runs/ds2_adam_source_factorial_v1.json"
OUT_MD = ROOT / "runs/ds2_adam_source_factorial_v1.md"

CONDITIONS = ("preference", "control")
SIGNS = {"preference": 1, "control": -1}
SCALE_REGIMES = ("native", "equal_norm")
CONFIG_SHA256 = "bfa725dd5b46e6a7dad7fcc7adfa6290c5c29a6129f48b5c7d204c1664e07e1c"
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
_REPLAY_VALIDATION_CACHE: dict[tuple[int, str, str], dict[str, Any]] = {}


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


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
    return {"path": relative(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def verify_artifact(record: dict[str, Any], expected: Path | None = None) -> Path:
    path = ROOT / record["path"]
    if expected is not None and path.resolve() != expected.resolve():
        raise RuntimeError(f"Artifact path mismatch: {path} != {expected}")
    if (
        not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or file_sha256(path) != record["sha256"]
    ):
        raise RuntimeError(f"Artifact changed: {path}")
    return path


def clear_cache() -> None:
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def release(owner: torch.nn.Module | None) -> None:
    if owner is not None:
        owner.to("cpu")
    del owner
    clear_cache()


def checkpoints(config: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in config["measurement"]["checkpoints"])


def seeds(config: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in config["training"]["student_seeds"])


def norm_path(seed: int, update: int) -> Path:
    return NORMS / f"seed_{seed}" / f"u{update:04d}" / "norm_reference.json"


def cell_path(seed: int, update: int, theta: str) -> Path:
    return CELLS / f"seed_{seed}" / f"u{update:04d}" / f"theta_{theta}" / "cell.json"


def next_attempt(root: Path, sentinel_names: set[str]) -> Path:
    numbers: list[int] = []
    if root.exists():
        for path in root.iterdir():
            if path.is_dir() and path.name.startswith("attempt_"):
                suffix = path.name.removeprefix("attempt_")
                if not suffix.isdigit():
                    raise RuntimeError(f"Malformed attempt directory: {path}")
                numbers.append(int(suffix))
            elif path.name not in sentinel_names:
                raise RuntimeError(f"Unexpected artifact in cell root: {path}")
    return root / f"attempt_{max(numbers, default=0) + 1:03d}"


def implementation_guard() -> dict[str, Any]:
    analysis_path = ROOT / "scripts/ds2_adam_source_analysis.py"
    return {
        "config_sha256": file_sha256(CONFIG_PATH),
        "factorial_runner_sha256": file_sha256(SCRIPT_PATH),
        "replay_runner_sha256": file_sha256(Path(replay.__file__).resolve()),
        "analysis_runner_sha256": (
            file_sha256(analysis_path) if analysis_path.is_file() else None
        ),
        "stage_a_runner_lock_sha256": (
            file_sha256(RUNNER_LOCK_PATH) if RUNNER_LOCK_PATH.is_file() else None
        ),
        "dynamics_runner_sha256": file_sha256(Path(dynamics.__file__).resolve()),
        "geometry_runner_sha256": file_sha256(Path(geometry.__file__).resolve()),
        "knockout_runner_sha256": file_sha256(Path(knockout.__file__).resolve()),
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


def validate_factorial_lock() -> dict[str, Any]:
    if not FACTORIAL_LOCK_PATH.is_file():
        raise RuntimeError("Factorial runner lock absent; run `freeze` after Stage A freeze")
    lock = load_json(FACTORIAL_LOCK_PATH)
    if set(lock) != {"name", "created_at", "stage_a_lock", "implementation"}:
        raise RuntimeError("Factorial runner-lock schema changed")
    verify_artifact(lock["stage_a_lock"], RUNNER_LOCK_PATH)
    observed = implementation_guard()
    if lock["implementation"] != observed:
        differences = {
            key: {"frozen": lock["implementation"].get(key), "observed": observed.get(key)}
            for key in set(lock["implementation"]) | set(observed)
            if lock["implementation"].get(key) != observed.get(key)
        }
        raise RuntimeError(f"Factorial implementation changed after freeze: {differences}")
    return lock


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
        "scripts/ds2_adam_source_", "scripts/numeric_", "scripts/dataorder_",
        "scripts/base_screening.py", "scripts/student_trait_write_probe.py",
        "scripts/cross_family_transport.py", "scripts/wolf_route_knockout.py",
        "polypythia_sl.pipeline",
    )
    conflicts = []
    for pid, (_, command) in processes.items():
        if pid in ancestors or "python" not in command.lower():
            continue
        if any(marker in command for marker in markers):
            conflicts.append({"pid": pid, "command": command})
    if conflicts:
        raise RuntimeError(f"Competing experiment process detected: {conflicts}")


@contextlib.contextmanager
def active_lock():
    ACTIVE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ACTIVE_LOCK_PATH.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            raise RuntimeError(f"Factorial runner already active: {handle.read()}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at": utc_now()}))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def adapted_probe_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    measurement = config["measurement"]
    lo, hi = (int(value) for value in measurement["route_prompt_slice"])
    result["intervention"] = {
        "target": measurement["trait_target"],
        "comparison_animals": list(measurement["comparison_animals"]),
        "route_prompts_slice": [lo, hi],
        "route_prompt_count": hi - lo,
        "behavior_batch_size": int(measurement["behavior_batch_size"]),
        "minimum_behavior_gradient_norm": 1e-12,
    }
    return result


def validate_config_contract(config: dict[str, Any]) -> None:
    if file_sha256(CONFIG_PATH) != CONFIG_SHA256:
        raise RuntimeError("Frozen config hash changed")
    if Path(replay.CONFIG_PATH).resolve() != CONFIG_PATH.resolve():
        raise RuntimeError("Stage A and Stage B config paths diverged")
    measurement = config["measurement"]
    if (
        tuple(measurement["theta_sources"]) != CONDITIONS
        or tuple(measurement["moment_sources"]) != CONDITIONS
        or tuple(measurement["data_conditions"]) != CONDITIONS
        or int(measurement["factorial_cell_count"]) != 28
        or int(measurement["norm_reference_count"]) != 14
        or int(measurement["native_candidate_count"]) != 224
        or int(measurement["equal_norm_candidate_count"]) != 224
        or int(measurement["decay_only_baseline_count"]) != 28
        or int(measurement["total_evaluated_state_count"]) != 476
        or int(measurement["snapshot_count"]) != 28
    ):
        raise RuntimeError("Frozen factorial inventory changed")
    lo, hi = (int(value) for value in measurement["route_prompt_slice"])
    blo, bhi = (int(value) for value in measurement["heldout_behavior_slice"])
    if (
        compact_hash(list(PREFERENCE_EVAL_PROMPTS[lo:hi]))
        != measurement["route_prompt_sha256"]
        or compact_hash(list(PREFERENCE_EVAL_PROMPTS[blo:bhi]))
        != measurement["heldout_behavior_prompt_sha256"]
    ):
        raise RuntimeError("Behavior prompt hash guard failed")
    for key, value in walk_sha_values(config):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise RuntimeError(f"Malformed SHA256 value at {key}: {value}")
    expected_artifacts = {
        "root": relative(WORK),
        "runner": relative(SCRIPT_PATH),
        "runner_lock": relative(RUNNER_LOCK_PATH),
        "aggregate_json": relative(OUT_JSON),
        "aggregate_markdown": relative(OUT_MD),
    }
    for key, value in expected_artifacts.items():
        if config["artifacts"].get(key) != value:
            raise RuntimeError(f"Artifact contract changed: {key}")


def walk_sha_values(value: Any, path: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            here = f"{path}.{key}" if path else key
            if "sha256" in key and isinstance(item, str):
                yield here, item
            yield from walk_sha_values(item, here)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk_sha_values(item, f"{path}[{index}]")


def load_config() -> dict[str, Any]:
    config = replay.load_and_validate_config()
    validate_config_contract(config)
    return config


def historical_order(config: dict[str, Any], seed: int) -> list[int]:
    count = int(config["data"]["rows_per_condition"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dynamics.IndexDataset(count),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    epochs: list[list[int]] = []
    guards = config["training"]["two_epoch_order_guards"][str(seed)]
    for epoch in range(2):
        order = [int(value) for batch in loader for value in batch.tolist()]
        if (
            sorted(order) != list(range(count))
            or order[:16] != guards["epoch_first_16"][epoch]
            or int64_sha256(order) != guards["epoch_sha256"][epoch]
        ):
            raise RuntimeError(f"Two-epoch order guard failed: seed={seed} epoch={epoch}")
        epochs.append(order)
    return [value for epoch in epochs for value in epoch]


def next_indices(config: dict[str, Any], order: list[int], update: int) -> list[int]:
    count = int(config["training"]["examples_per_update"])
    values = order[update * count : (update + 1) * count]
    if len(values) != count:
        raise RuntimeError(f"No full next batch at update {update}")
    return values


def prepare_datasets(config: dict[str, Any], tokenizer):
    training_rows, heldout_rows = knockout.load_rows(config)
    max_length = int(config["training"]["max_length"])
    training = {
        condition: CompletionDataset(training_rows[condition], tokenizer, max_length)
        for condition in CONDITIONS
    }
    fixed = {}
    indices = [int(value) for value in config["measurement"]["fixed64_indices"]]
    if int64_sha256(indices) != config["measurement"]["fixed64_indices_int64_sha256"]:
        raise RuntimeError("Fixed64 index hash changed")
    for condition in CONDITIONS:
        fixed[condition] = CompletionDataset(
            [heldout_rows[condition][index] for index in indices], tokenizer, max_length
        )
    expected_tokens = int(config["data"]["supervised_tokens_per_row"])
    for scope in (training, fixed):
        for dataset in scope.values():
            counts = {int((row["labels"] != -100).sum()) for row in dataset.examples}
            if counts != {expected_tokens}:
                raise RuntimeError(f"Supervised-token guard failed: {counts}")
    return training, fixed


def snapshot_from_replay_cell(
    config: dict[str, Any], seed: int, condition: str, update: int
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    path = replay.expected_cell_path(seed, condition)
    validated = validated_replay_cell(config, seed, condition)
    cell = validated.get("cell", validated)
    artifacts = cell.get("artifacts", {})
    keys = (f"state_u{update:04d}", f"snapshot_u{update:04d}", str(update))
    record = next((artifacts[key] for key in keys if key in artifacts), None)
    if record is None and isinstance(cell.get("snapshots"), dict):
        record = next((cell["snapshots"][key] for key in keys if key in cell["snapshots"]), None)
    if record is None:
        raise RuntimeError(f"Replay cell does not expose u{update:04d} snapshot: {path}")
    state_path = verify_artifact(record)
    dynamics.validate_state_snapshot(
        state_path, config, config["receiver"]["name"], seed, condition, update
    )
    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    return cell, payload, state_path


def validated_replay_cell(
    config: dict[str, Any], seed: int, condition: str
) -> dict[str, Any]:
    path = replay.expected_cell_path(seed, condition)
    if not path.is_file():
        raise FileNotFoundError(path)
    key = (seed, condition, file_sha256(path))
    if key not in _REPLAY_VALIDATION_CACHE:
        _REPLAY_VALIDATION_CACHE[key] = replay.validate_replay_cell(path, config)
    return _REPLAY_VALIDATION_CACHE[key]


def canonical_trainable(owner: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return dynamics.canonical_trainable(owner)


def payload_rows(payload: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    lora = {row["name"]: row["tensor"] for row in payload["lora"]}
    adam = {row["name"]: row for row in payload["adam"]}
    if set(lora) != set(adam):
        raise RuntimeError("Snapshot LoRA/Adam names differ")
    return lora, adam


def restore_theta(owner: torch.nn.Module, config: dict[str, Any], payload: dict[str, Any]) -> None:
    optimizer = geometry.restore_snapshot(owner, config, payload)
    del optimizer
    owner.zero_grad(set_to_none=True)


def semantic_lora_hash(owner: torch.nn.Module) -> str:
    return dynamics.semantic_tensor_hash(
        (name, parameter.detach().cpu()) for name, parameter in canonical_trainable(owner)
    )


def vector_norm(value: dict[str, torch.Tensor]) -> float:
    return math.sqrt(max(sum(float(torch.sum(tensor.double().square())) for tensor in value.values()), 0.0))


def group_names(name: str) -> tuple[str, ...]:
    groups = ["all", "lora_A" if ".lora_A." in name else "lora_B"]
    match = re.search(r"gpt_neox\.layers\.(\d+)\.", name)
    if match:
        groups.append(f"layer_{int(match.group(1)):02d}")
    for module in ("query_key_value", "dense_h_to_4h", "dense_4h_to_h", "dense"):
        if f".{module}." in name:
            groups.append(f"module_{module}")
            break
    return tuple(groups)


def empty_geometry_accumulator() -> dict[str, dict[str, dict[str, float]]]:
    return {}


def add_geometry(
    accumulator: dict[str, dict[str, dict[str, float]]], component: str,
    name: str, update: torch.Tensor, trait: torch.Tensor,
) -> None:
    dot = float(torch.sum(update.double() * trait.double()))
    update_sq = float(torch.sum(update.double().square()))
    trait_sq = float(torch.sum(trait.double().square()))
    component_rows = accumulator.setdefault(component, {})
    for group in group_names(name):
        row = component_rows.setdefault(group, {"raw_dot": 0.0, "update_sq": 0.0, "trait_sq": 0.0})
        row["raw_dot"] += dot
        row["update_sq"] += update_sq
        row["trait_sq"] += trait_sq


def finalize_geometry(
    accumulator: dict[str, dict[str, dict[str, float]]]
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for component, groups in accumulator.items():
        result[component] = {}
        for group, row in groups.items():
            update_norm = math.sqrt(max(row["update_sq"], 0.0))
            trait_norm = math.sqrt(max(row["trait_sq"], 0.0))
            dot = row["raw_dot"]
            result[component][group] = {
                "raw_dot": dot,
                "update_l2_norm": update_norm,
                "trait_gradient_l2_norm": trait_norm,
                "raw_dot_per_update_norm": dot / update_norm if update_norm else 0.0,
                "unit_trait_projection": dot / trait_norm if trait_norm else 0.0,
                "true_cosine": dot / (trait_norm * update_norm) if trait_norm and update_norm else 0.0,
            }
    return result


def build_hybrid_optimizer(
    owner: torch.nn.Module,
    config: dict[str, Any],
    theta_payload: dict[str, Any],
    m_payload: dict[str, Any],
    v_payload: dict[str, Any],
    clipped_gradient: dict[str, torch.Tensor],
) -> torch.optim.Optimizer:
    optimizer = geometry.restore_snapshot(owner, config, theta_payload)
    trainable = canonical_trainable(owner)
    _, m_rows = payload_rows(m_payload)
    _, v_rows = payload_rows(v_payload)
    update = int(theta_payload["optimizer_update"])
    if int(m_payload["optimizer_update"]) != update or int(v_payload["optimizer_update"]) != update:
        raise RuntimeError("Crossed snapshots are from different optimizer updates")
    for name, parameter in trainable:
        m_row = m_rows[name]
        v_row = v_rows[name]
        m_step = int(float(m_row["step"].item()))
        v_step = int(float(v_row["step"].item()))
        if m_step != update or v_step != update:
            raise RuntimeError(f"Donor Adam step mismatch: {name}")
        state = optimizer.state[parameter]
        state["step"] = m_row["step"].detach().clone().cpu()
        state["exp_avg"] = m_row["exp_avg"].to(parameter.device, parameter.dtype).clone()
        state["exp_avg_sq"] = v_row["exp_avg_sq"].to(parameter.device, parameter.dtype).clone()
        gradient = clipped_gradient[name]
        if gradient.shape != parameter.shape or not torch.isfinite(gradient).all():
            raise RuntimeError(f"Malformed clipped gradient: {name}")
        parameter.grad = gradient.to(parameter.device, parameter.dtype).clone()
    group = optimizer.param_groups[0]
    expected_lr = float(theta_payload["summaries"]["lr_available_for_next_update"])
    donor_lrs = {
        float(payload["summaries"]["lr_available_for_next_update"])
        for payload in (theta_payload, m_payload, v_payload)
    }
    if donor_lrs != {expected_lr} or float(group["lr"]) != expected_lr:
        raise RuntimeError("Donor learning rates differ")
    return optimizer


def manual_vectors(
    owner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    update: int,
    unclipped: dict[str, torch.Tensor],
    clipped: dict[str, torch.Tensor],
    trait: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    recipe = config["measurement"]
    beta1, beta2 = (float(value) for value in recipe["betas"])
    eps = float(recipe["eps"])
    weight_decay = float(recipe["weight_decay"])
    lr = float(optimizer.param_groups[0]["lr"])
    next_step = update + 1
    bc1 = 1.0 - beta1**next_step
    bc2 = 1.0 - beta2**next_step
    step_size = lr / bc1
    accumulator = empty_geometry_accumulator()
    adaptive: dict[str, torch.Tensor] = {}
    decay: dict[str, torch.Tensor] = {}
    manual_after: dict[str, torch.Tensor] = {}
    before: dict[str, torch.Tensor] = {}
    adaptive_decomposition_max_error = 0.0
    with torch.no_grad():
        for name, parameter in canonical_trainable(owner):
            old = parameter.detach().clone()
            state = optimizer.state[parameter]
            gradient = parameter.grad
            if gradient is None:
                raise RuntimeError(f"Missing hybrid numeric gradient: {name}")
            m_old = state["exp_avg"].detach()
            v_old = state["exp_avg_sq"].detach()
            history_numerator = beta1 * m_old
            current_numerator = (1.0 - beta1) * gradient
            m_new = history_numerator + current_numerator
            v_new = beta2 * v_old + (1.0 - beta2) * gradient.square()
            denom = v_new.sqrt().div(math.sqrt(bc2)).add(eps)
            adaptive_value = torch.zeros_like(old).addcdiv_(m_new, denom, value=-step_size)
            history_value = torch.zeros_like(old).addcdiv_(
                history_numerator, denom, value=-step_size
            )
            current_value = torch.zeros_like(old).addcdiv_(
                current_numerator, denom, value=-step_size
            )
            component_error = float(
                (adaptive_value - history_value - current_value).abs().max().detach().cpu()
            )
            adaptive_decomposition_max_error = max(
                adaptive_decomposition_max_error, component_error
            )
            decay_value = -lr * weight_decay * old
            total_value = decay_value + adaptive_value
            momentum_value = -lr * (m_new / bc1)
            adaptive[name] = adaptive_value.detach().float().cpu().contiguous()
            decay[name] = decay_value.detach().float().cpu().contiguous()
            before[name] = old.detach().float().cpu().contiguous()
            manual_after[name] = (old + total_value).detach().float().cpu().contiguous()
            if trait is not None:
                trait_value = trait[name]
                component_values = {
                    "unclipped_raw_descent_direction": -unclipped[name],
                    "clipped_raw_descent_direction": -clipped[name],
                    "clipped_raw_lr_scaled_update": -lr * clipped[name],
                    "unpreconditioned_bias_corrected_momentum_lr_scaled_update": momentum_value.detach().cpu(),
                    "adam_history_adaptive_update": history_value.detach().cpu(),
                    "adam_current_gradient_adaptive_update": current_value.detach().cpu(),
                    "adam_preconditioned_adaptive_update": adaptive_value.detach().cpu(),
                    "weight_decay_update": decay_value.detach().cpu(),
                    "manual_total_update": total_value.detach().cpu(),
                }
                for component, value in component_values.items():
                    add_geometry(accumulator, component, name, value, trait_value)
    if adaptive_decomposition_max_error > float(
        config["guards"]["adaptive_decomposition_max_abs_tolerance"]
    ):
        raise RuntimeError(
            f"Adam history/current decomposition failed: {adaptive_decomposition_max_error}"
        )
    return {
        "learning_rate": lr,
        "adaptive": adaptive,
        "decay": decay,
        "before": before,
        "manual_after": manual_after,
        "adaptive_l2_norm": vector_norm(adaptive),
        "decay_l2_norm": vector_norm(decay),
        "geometry": finalize_geometry(accumulator) if trait is not None else None,
        "adaptive_decomposition_max_abs_error": adaptive_decomposition_max_error,
    }


def verify_native_step(
    owner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    manual: dict[str, Any],
    config: dict[str, Any],
    update: int,
) -> dict[str, Any]:
    optimizer.step()
    owner.zero_grad(set_to_none=True)
    error_sq = 0.0
    after_sq = 0.0
    actual_update_sq = 0.0
    maximum = 0.0
    for name, parameter in canonical_trainable(owner):
        after = parameter.detach().float().cpu()
        expected = manual["manual_after"][name]
        before = manual["before"][name]
        error = after.double() - expected.double()
        actual = after.double() - before.double()
        error_sq += float(torch.sum(error.square()))
        after_sq += float(torch.sum(after.double().square()))
        actual_update_sq += float(torch.sum(actual.square()))
        maximum = max(maximum, float(error.abs().max()))
        step = int(float(optimizer.state[parameter]["step"].detach().cpu().item()))
        if step != update + 1:
            raise RuntimeError(f"AdamW advanced to wrong step: {name}/{step}")
    relative_parameter = math.sqrt(error_sq) / max(math.sqrt(after_sq), 1e-30)
    relative_update = math.sqrt(error_sq) / max(math.sqrt(actual_update_sq), 1e-30)
    record = {
        "maximum_parameter_absolute_error": maximum,
        "relative_parameter_l2_error": relative_parameter,
        "relative_actual_update_l2_error": relative_update,
        "history_plus_current_adaptive_max_abs_error": manual[
            "adaptive_decomposition_max_abs_error"
        ],
        "passed": (
            maximum <= float(config["guards"]["manual_after_max_abs_tolerance"])
            and relative_parameter
            <= float(config["guards"]["manual_after_relative_l2_tolerance"])
            and relative_update
            <= float(config["guards"]["manual_update_relative_l2_tolerance"])
        ),
    }
    if not record["passed"]:
        raise RuntimeError(f"Manual/native AdamW mismatch: {record}")
    return record


def apply_equal_norm(
    owner: torch.nn.Module,
    manual: dict[str, Any],
    common_norm: float,
    config: dict[str, Any],
    trait: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, Any]]:
    native_norm = float(manual["adaptive_l2_norm"])
    if native_norm <= 0.0 or common_norm <= 0.0:
        raise RuntimeError("Cannot equal-normalize a zero adaptive displacement")
    scale = common_norm / native_norm
    accumulator = empty_geometry_accumulator()
    realized_sq = 0.0
    with torch.no_grad():
        for name, parameter in canonical_trainable(owner):
            adaptive = (manual["adaptive"][name].double() * scale).float().double()
            decay = manual["decay"][name].float().double()
            total = adaptive + decay
            parameter.copy_(manual["before"][name].to(parameter.device, parameter.dtype))
            parameter.add_(decay.to(parameter.device, parameter.dtype))
            parameter.add_(adaptive.to(parameter.device, parameter.dtype))
            realized_adaptive = adaptive
            realized_sq += float(torch.sum(realized_adaptive.square()))
            add_geometry(
                accumulator, "equal_norm_adaptive_update", name, adaptive, trait[name]
            )
            add_geometry(
                accumulator, "weight_decay_update", name, decay, trait[name]
            )
            add_geometry(
                accumulator, "equal_norm_total_update", name, total, trait[name]
            )
    mathematical_norm = native_norm * abs(scale)
    realized_norm = math.sqrt(max(realized_sq, 0.0))
    absolute_tolerance = float(config["guards"]["equal_norm_absolute_tolerance"])
    relative_tolerance = float(config["guards"]["equal_norm_relative_tolerance"])
    tolerance = absolute_tolerance + relative_tolerance * common_norm
    # The exact guard applies to the mathematical branch vector.  Float32
    # parameter assignment is separately reported because rounding after adding
    # theta can be larger than 1e-9.
    if abs(mathematical_norm - common_norm) > tolerance:
        raise RuntimeError("Equal-norm mathematical displacement guard failed")
    if not math.isfinite(realized_norm) or realized_norm <= 0.0:
        raise RuntimeError("Equal-norm float32 displacement is not finite and positive")
    return finalize_geometry(accumulator), {
        "scale": scale,
        "native_adaptive_l2_norm": native_norm,
        "target_adaptive_l2_norm": common_norm,
        "mathematical_adaptive_l2_norm": mathematical_norm,
        "mathematical_absolute_error": abs(mathematical_norm - common_norm),
        "mathematical_tolerance": tolerance,
        "realized_float32_adaptive_l2_norm": realized_norm,
        "realized_float32_absolute_error": abs(realized_norm - common_norm),
    }


def evaluate_state(
    owner: torch.nn.Module,
    tokenizer,
    token_ids: torch.Tensor,
    fixed: dict[str, CompletionDataset],
    config: dict[str, Any],
) -> dict[str, Any]:
    lo, hi = (int(value) for value in config["measurement"]["heldout_behavior_slice"])
    behavior = knockout.behavior_values(
        owner,
        tokenizer,
        token_ids,
        list(PREFERENCE_EVAL_PROMPTS[lo:hi]),
        int(config["measurement"]["behavior_batch_size"]),
    )
    expected_tokens = int(config["data"]["supervised_tokens_per_row"])
    nll = {
        condition: knockout.completion_nll_values(
            owner,
            fixed[condition],
            tokenizer,
            int(config["measurement"]["heldout_completion_batch_size"]),
            expected_tokens,
        )
        for condition in CONDITIONS
    }
    return {"behavior": behavior, "fixed64_nll": nll}


def changes_from_baseline(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    candidate_rows = candidate["behavior"]["per_prompt"]
    baseline_rows = baseline["behavior"]["per_prompt"]
    if [row["prompt"] for row in candidate_rows] != [row["prompt"] for row in baseline_rows]:
        raise RuntimeError("Behavior prompt order changed across candidate/baseline")
    margin = [
        float(candidate_row["wolf_margin"] - baseline_row["wolf_margin"])
        for candidate_row, baseline_row in zip(candidate_rows, baseline_rows)
    ]
    probability = [
        float(candidate_row["wolf_probability"] - baseline_row["wolf_probability"])
        for candidate_row, baseline_row in zip(candidate_rows, baseline_rows)
    ]
    nll_benefit: dict[str, list[float]] = {}
    for condition in CONDITIONS:
        candidate_values = candidate["fixed64_nll"][condition]["per_row_nll"]
        baseline_values = baseline["fixed64_nll"][condition]["per_row_nll"]
        if len(candidate_values) != 64 or len(baseline_values) != 64:
            raise RuntimeError("Fixed64 NLL vector changed length")
        # Positive means the candidate lowered NLL relative to decay-only theta.
        nll_benefit[condition] = [
            float(base - value) for value, base in zip(candidate_values, baseline_values)
        ]
    return {
        "behavior_wolf_margin_change": {
            "mean": float(np.mean(margin)), "per_prompt": margin
        },
        "behavior_wolf_probability_change": {
            "mean": float(np.mean(probability)), "per_prompt": probability
        },
        "fixed64_nll_benefit": {
            condition: {
                "mean": float(np.mean(values)), "per_row": values
            }
            for condition, values in nll_benefit.items()
        },
        "preference_minus_control_nll_benefit": {
            "mean": float(np.mean(nll_benefit["preference"]) - np.mean(nll_benefit["control"])),
            "paired_rows": [
                float(left - right)
                for left, right in zip(nll_benefit["preference"], nll_benefit["control"])
            ],
        },
    }


def decay_only_baseline(
    owner: torch.nn.Module,
    tokenizer,
    token_ids: torch.Tensor,
    fixed: dict[str, CompletionDataset],
    config: dict[str, Any],
    theta_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    restore_theta(owner, config, theta_payload)
    expected_hash = theta_payload["summaries"]["lora_semantic_sha256"]
    if semantic_lora_hash(owner) != expected_hash:
        raise RuntimeError("Theta restoration failed before decay baseline")
    lr = float(theta_payload["summaries"]["lr_available_for_next_update"])
    weight_decay = float(config["measurement"]["weight_decay"])
    with torch.no_grad():
        for _, parameter in canonical_trainable(owner):
            parameter.mul_(1.0 - lr * weight_decay)
    outcome = evaluate_state(owner, tokenizer, token_ids, fixed, config)
    record = {
        "definition": "theta-specific AdamW weight decay only; no adaptive displacement",
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "outcome": outcome,
    }
    restore_theta(owner, config, theta_payload)
    restoration = {
        "expected_lora_semantic_sha256": expected_hash,
        "observed_lora_semantic_sha256": semantic_lora_hash(owner),
    }
    restoration["passed"] = (
        restoration["expected_lora_semantic_sha256"]
        == restoration["observed_lora_semantic_sha256"]
    )
    if not restoration["passed"]:
        raise RuntimeError("Theta restoration failed after decay baseline")
    return record, restoration


def norm_identity(
    seed: int,
    update: int,
    source_artifacts: dict[str, dict[str, Any]],
    attempt: Path,
) -> dict[str, Any]:
    return {
        "name": "ds2-adam-source-factorial-norm-reference-v1",
        "config_sha256": file_sha256(CONFIG_PATH),
        "factorial_runner_lock_sha256": file_sha256(FACTORIAL_LOCK_PATH),
        "seed": seed,
        "optimizer_update": update,
        "source_snapshots": source_artifacts,
        "attempt": relative(attempt),
    }


def validate_norm_reference(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    sentinel = load_json(path)
    required = {
        "name", "config_sha256", "factorial_runner_lock_sha256", "seed",
        "optimizer_update", "source_snapshots", "attempt", "completed_at",
        "start_manifest", "result",
    }
    if set(sentinel) != required:
        raise RuntimeError(f"Norm-reference sentinel schema changed: {path}")
    seed = int(sentinel["seed"])
    update = int(sentinel["optimizer_update"])
    if path.resolve() != norm_path(seed, update).resolve():
        raise RuntimeError(f"Norm-reference path identity mismatch: {path}")
    if sentinel["config_sha256"] != file_sha256(CONFIG_PATH):
        raise RuntimeError(f"Norm-reference config hash mismatch: {path}")
    if sentinel["factorial_runner_lock_sha256"] != file_sha256(FACTORIAL_LOCK_PATH):
        raise RuntimeError(f"Norm-reference runner-lock hash mismatch: {path}")
    attempt = ROOT / sentinel["attempt"]
    if attempt.parent.resolve() != path.parent.resolve() or not attempt.name.startswith("attempt_"):
        raise RuntimeError(f"Norm-reference attempt path mismatch: {path}")
    start_path = verify_artifact(sentinel["start_manifest"], attempt / "start_manifest.json")
    result_path = verify_artifact(sentinel["result"], attempt / "result.json")
    del start_path
    result = load_json(result_path)
    expected_identity = norm_identity(
        seed, update, sentinel["source_snapshots"], attempt
    )
    if any(result.get(key) != value for key, value in expected_identity.items()):
        raise RuntimeError(f"Norm-reference result identity mismatch: {path}")
    for condition in CONDITIONS:
        frozen_path = verify_artifact(sentinel["source_snapshots"][condition])
        _, _, canonical_path = snapshot_from_replay_cell(
            config, seed, condition, update
        )
        if frozen_path.resolve() != canonical_path.resolve() or artifact_record(
            canonical_path
        ) != sentinel["source_snapshots"][condition]:
            raise RuntimeError(f"Norm reference detached from Stage-A parent: {path}")
    matching = result.get("matching_native_references", {})
    if set(matching) != set(CONDITIONS):
        raise RuntimeError(f"Norm-reference matching inventory changed: {path}")
    norms = [float(matching[condition]["adaptive_l2_norm"]) for condition in CONDITIONS]
    expected = math.sqrt(norms[0] * norms[1])
    observed = float(result.get("symmetric_common_adaptive_l2_norm", float("nan")))
    observed_indices = [int(value) for value in result.get("next_example_indices", [])]
    expected_indices = next_indices(config, historical_order(config, seed), update)
    tolerance = float(config["guards"]["equal_norm_absolute_tolerance"]) + float(
        config["guards"]["equal_norm_relative_tolerance"]
    ) * expected
    if (
        not all(math.isfinite(value) and value > 0.0 for value in [*norms, expected, observed])
        or abs(observed - expected) > tolerance
        or observed_indices != expected_indices
        or int64_sha256(observed_indices)
        != result.get("next_example_indices_int64_sha256")
        or result.get("no_branch_tensors_written") is not True
        or not finite_tree(result)
    ):
        raise RuntimeError(f"Invalid symmetric norm reference: {path}")
    forbidden = [item for item in attempt.rglob("*") if item.suffix in {".pt", ".bin", ".safetensors"}]
    if forbidden:
        raise RuntimeError(f"Norm-reference attempt wrote tensors: {forbidden}")
    return {"sentinel": sentinel, "result": result}


def run_norm_reference(
    owner: torch.nn.Module,
    tokenizer,
    training: dict[str, CompletionDataset],
    orders: dict[int, list[int]],
    config: dict[str, Any],
    seed: int,
    update: int,
) -> dict[str, Any]:
    path = norm_path(seed, update)
    if path.exists():
        print(f"[norm/{seed}/u{update:04d}] validated reuse", flush=True)
        return validate_norm_reference(path, config)
    payloads: dict[str, dict[str, Any]] = {}
    source_artifacts: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        _, payload, state_path = snapshot_from_replay_cell(config, seed, condition, update)
        payloads[condition] = payload
        source_artifacts[condition] = artifact_record(state_path)
    root = path.parent
    attempt = next_attempt(root, {"norm_reference.json"})
    attempt.mkdir(parents=True, exist_ok=False)
    identity = norm_identity(seed, update, source_artifacts, attempt)
    start_path = attempt / "start_manifest.json"
    atomic_write_json(start_path, {
        "created_at": utc_now(),
        "identity": identity,
        "definition": "outcome-blind symmetric norm prerequisite; no behavior or heldout NLL evaluated",
    })
    print(f"[norm/{seed}/u{update:04d}] {attempt.name}", flush=True)
    try:
        indices = next_indices(config, orders[seed], update)
        references: dict[str, Any] = {}
        for condition in CONDITIONS:
            payload = payloads[condition]
            restore_theta(owner, config, payload)
            unclipped, clipped, gradient_record = geometry.numeric_gradient(
                owner, training[condition], tokenizer, indices, config
            )
            optimizer = build_hybrid_optimizer(
                owner, config, payload, payload, payload, clipped
            )
            manual = manual_vectors(
                owner, optimizer, config, update, unclipped, clipped, trait=None
            )
            references[condition] = {
                "source_code": "PPPP" if condition == "preference" else "CCCC",
                "adaptive_l2_norm": manual["adaptive_l2_norm"],
                "learning_rate": manual["learning_rate"],
                "numeric_gradient": gradient_record,
                "adaptive_decomposition_max_abs_error": manual[
                    "adaptive_decomposition_max_abs_error"
                ],
            }
            del optimizer, manual, unclipped, clipped
            owner.zero_grad(set_to_none=True)
        common = math.sqrt(
            float(references["preference"]["adaptive_l2_norm"])
            * float(references["control"]["adaptive_l2_norm"])
        )
        if not math.isfinite(common) or common <= 0.0:
            raise RuntimeError("Symmetric reference norm is not finite and positive")
        result = {
            **identity,
            "completed_at": utc_now(),
            "next_example_indices": indices,
            "next_example_indices_int64_sha256": int64_sha256(indices),
            "matching_native_references": references,
            "symmetric_common_adaptive_l2_norm": common,
            "formula": "sqrt(||A_PPPP|| * ||A_CCCC||)",
            "evaluated_behavior_or_heldout_nll": False,
            "no_branch_tensors_written": True,
        }
        result_path = attempt / "result.json"
        atomic_write_json(result_path, result)
    except BaseException as error:
        atomic_write_json(attempt / "failure.json", {**identity, "failed_at": utc_now(), "error": repr(error)})
        raise
    sentinel = {
        **identity,
        "completed_at": result["completed_at"],
        "start_manifest": artifact_record(start_path),
        "result": artifact_record(result_path),
    }
    exclusive_write_json(path, sentinel)
    print(f"[norm/{seed}/u{update:04d}] DONE R={common:.9g}", flush=True)
    return validate_norm_reference(path, config)


def cell_identity(
    seed: int,
    update: int,
    theta: str,
    source_artifacts: dict[str, dict[str, Any]],
    norm_artifact: dict[str, Any],
    attempt: Path,
) -> dict[str, Any]:
    return {
        "name": "ds2-adam-source-factorial-cell-v1",
        "config_sha256": file_sha256(CONFIG_PATH),
        "factorial_runner_lock_sha256": file_sha256(FACTORIAL_LOCK_PATH),
        "seed": seed,
        "optimizer_update": update,
        "theta_source": theta,
        "source_snapshots": source_artifacts,
        "norm_reference": norm_artifact,
        "attempt": relative(attempt),
    }


def candidate_key(data: str, m_source: str, v_source: str) -> str:
    return f"D_{data}__M_{m_source}__V_{v_source}"


def expected_candidate_keys() -> set[str]:
    return {
        candidate_key(data, m_source, v_source)
        for data, m_source, v_source in itertools.product(CONDITIONS, repeat=3)
    }


def validate_factorial_cell(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    sentinel = load_json(path)
    required = {
        "name", "config_sha256", "factorial_runner_lock_sha256", "seed",
        "optimizer_update", "theta_source", "source_snapshots", "norm_reference",
        "attempt", "completed_at", "start_manifest", "result",
    }
    if set(sentinel) != required:
        raise RuntimeError(f"Factorial sentinel schema changed: {path}")
    seed = int(sentinel["seed"])
    update = int(sentinel["optimizer_update"])
    theta = sentinel["theta_source"]
    if theta not in CONDITIONS or path.resolve() != cell_path(seed, update, theta).resolve():
        raise RuntimeError(f"Factorial cell identity/path mismatch: {path}")
    if (
        sentinel["config_sha256"] != file_sha256(CONFIG_PATH)
        or sentinel["factorial_runner_lock_sha256"] != file_sha256(FACTORIAL_LOCK_PATH)
    ):
        raise RuntimeError(f"Factorial cell frozen hash mismatch: {path}")
    attempt = ROOT / sentinel["attempt"]
    if attempt.parent.resolve() != path.parent.resolve() or not attempt.name.startswith("attempt_"):
        raise RuntimeError(f"Factorial attempt path mismatch: {path}")
    verify_artifact(sentinel["start_manifest"], attempt / "start_manifest.json")
    result_path = verify_artifact(sentinel["result"], attempt / "result.json")
    result = load_json(result_path)
    expected_identity = cell_identity(
        seed, update, theta, sentinel["source_snapshots"],
        sentinel["norm_reference"], attempt,
    )
    if any(result.get(key) != value for key, value in expected_identity.items()):
        raise RuntimeError(f"Factorial result identity mismatch: {path}")
    for condition in CONDITIONS:
        frozen_path = verify_artifact(sentinel["source_snapshots"][condition])
        _, _, canonical_path = snapshot_from_replay_cell(
            config, seed, condition, update
        )
        if frozen_path.resolve() != canonical_path.resolve() or artifact_record(
            canonical_path
        ) != sentinel["source_snapshots"][condition]:
            raise RuntimeError(f"Factorial cell detached from Stage-A parent: {path}")
    norm_result = validate_norm_reference(
        verify_artifact(sentinel["norm_reference"]), config
    )["result"]
    candidates = result.get("candidates", {})
    if set(candidates) != expected_candidate_keys():
        raise RuntimeError(f"Factorial candidate inventory changed: {path}")
    common_norm = float(result.get("symmetric_common_adaptive_l2_norm", float("nan")))
    observed_indices = [int(value) for value in result.get("next_example_indices", [])]
    expected_indices = next_indices(config, historical_order(config, seed), update)
    if (
        not math.isfinite(common_norm)
        or common_norm <= 0.0
        or common_norm != float(norm_result["symmetric_common_adaptive_l2_norm"])
        or observed_indices != expected_indices
        or int64_sha256(observed_indices)
        != result.get("next_example_indices_int64_sha256")
        or result.get("baseline_restoration_guard", {}).get("passed") is not True
    ):
        raise RuntimeError(f"Factorial checkpoint-level guard failed: {path}")
    equal_tolerance = float(config["guards"]["equal_norm_absolute_tolerance"]) + float(
        config["guards"]["equal_norm_relative_tolerance"]
    ) * common_norm
    for key, candidate in candidates.items():
        is_identity = (
            candidate["theta_source"] == candidate["data_condition"]
            == candidate["exp_avg_source"] == candidate["exp_avg_sq_source"]
        )
        identity_guard = candidate.get("identity_next_update_replay_guard")
        if (
            candidate_key(candidate["data_condition"], candidate["exp_avg_source"], candidate["exp_avg_sq_source"])
            != key
            or set(candidate["scales"]) != set(SCALE_REGIMES)
            or candidate["scales"]["native"]["manual_adamw_verification"].get("passed") is not True
            or candidate["restoration_guard"].get("passed") is not True
        ):
            raise RuntimeError(f"Invalid factorial candidate: {path}/{key}")
        if is_identity:
            if not isinstance(identity_guard, dict):
                raise RuntimeError(f"Missing identity next-update guard: {path}/{key}")
            if int(identity_guard.get("expected_optimizer_update", -1)) != update + 1:
                raise RuntimeError(f"Identity next-update index changed: {path}/{key}")
            if update < int(config["training"]["replay_updates"]):
                if identity_guard.get("available") is not True or identity_guard.get("passed") is not True:
                    raise RuntimeError(f"Failed identity next-update guard: {path}/{key}")
            elif (
                identity_guard.get("available") is not False
                or identity_guard.get("passed") is not None
                or int(identity_guard.get("expected_optimizer_update", -1)) != 513
            ):
                raise RuntimeError(f"Malformed explicit u513 unavailability: {path}/{key}")
        elif identity_guard is not None:
            raise RuntimeError(f"Unexpected identity replay claim: {path}/{key}")
        for regime in SCALE_REGIMES:
            scale = candidate["scales"][regime]
            changes = candidate["scales"][regime]["changes_from_theta_decay_only"]
            if (
                len(changes["behavior_wolf_margin_change"]["per_prompt"]) != 30
                or any(
                    len(changes["fixed64_nll_benefit"][condition]["per_row"]) != 64
                    for condition in CONDITIONS
                )
            ):
                raise RuntimeError(f"Factorial response unit inventory changed: {path}/{key}/{regime}")
            if not math.isfinite(float(scale.get("adaptive_l2_norm", float("nan")))) or float(
                scale["adaptive_l2_norm"]
            ) <= 0.0:
                raise RuntimeError(f"Invalid candidate adaptive norm: {path}/{key}/{regime}")
        equal = candidate["scales"]["equal_norm"]
        equal_guard = equal.get("equal_norm_guard", {})
        if (
            abs(float(equal["adaptive_l2_norm"]) - common_norm) > equal_tolerance
            or abs(float(equal_guard.get("target_adaptive_l2_norm", float("nan"))) - common_norm)
            > equal_tolerance
            or float(equal_guard.get("mathematical_absolute_error", math.inf))
            > float(equal_guard.get("mathematical_tolerance", -math.inf))
            or not math.isfinite(
                float(equal_guard.get("realized_float32_adaptive_l2_norm", float("nan")))
            )
            or float(equal_guard.get("realized_float32_adaptive_l2_norm", 0.0)) <= 0.0
            or not math.isfinite(
                float(equal_guard.get("realized_float32_absolute_error", float("nan")))
            )
        ):
            raise RuntimeError(f"Equal-norm candidate guard failed: {path}/{key}")
    if (
        result.get("evaluated_state_count") != 17
        or result.get("no_branch_tensors_written") is not True
        or not finite_tree(result)
    ):
        raise RuntimeError(f"Invalid factorial result guards: {path}")
    forbidden = [item for item in attempt.rglob("*") if item.suffix in {".pt", ".bin", ".safetensors"}]
    if forbidden:
        raise RuntimeError(f"Factorial attempt wrote tensors: {forbidden}")
    return {"sentinel": sentinel, "result": result}


def outcome_summary(outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        "behavior": {
            "margin": outcome["behavior"]["margin"],
            "probability": outcome["behavior"]["probability"],
        },
        "fixed64_nll": {
            condition: {
                key: value for key, value in outcome["fixed64_nll"][condition].items()
                if key != "per_row_nll"
            }
            for condition in CONDITIONS
        },
    }


def matching_next_update_guard(
    config: dict[str, Any],
    seed: int,
    condition: str,
    update: int,
    gradient_record: dict[str, Any],
    learning_rate: float,
) -> dict[str, Any]:
    validated = validated_replay_cell(config, seed, condition)
    if update == int(config["training"]["replay_updates"]):
        return {
            "available": False,
            "passed": None,
            "expected_optimizer_update": update + 1,
            "reason": (
                "Stage-A and the frozen wolf-route parent end at u512; no outcome-blind "
                "archived scalar u513 loss/gradient-norm reference exists. The frozen "
                "epoch-2 next-batch order and manual AdamW guards still apply."
            ),
        }
    expected = validated["metrics"]["update_metrics"][update]
    if int(expected["optimizer_update"]) != update + 1:
        raise RuntimeError("Stage-A matching-next-update index changed")
    observed = {
        "mean_microbatch_loss": float(gradient_record["mean_microbatch_loss"]),
        "gradient_norm_before_clipping": float(
            gradient_record["gradient_norm_before_clipping"]
        ),
        "learning_rate_used": float(learning_rate),
    }
    reference = {
        "mean_microbatch_loss": float(expected["mean_microbatch_loss"]),
        "gradient_norm_before_clipping": float(
            expected["gradient_norm_before_clipping"]
        ),
        "learning_rate_used": float(expected["learning_rate_used"]),
    }
    errors = {
        key: observed[key] - reference[key] for key in observed
    }
    passed = (
        abs(errors["mean_microbatch_loss"])
        <= float(config["guards"]["replay_loss_absolute_tolerance"])
        and abs(errors["gradient_norm_before_clipping"])
        <= float(config["guards"]["replay_gradient_norm_absolute_tolerance"])
        and abs(errors["learning_rate_used"])
        <= float(config["guards"]["replay_learning_rate_absolute_tolerance"])
    )
    record = {
        "available": True,
        "passed": passed,
        "expected_optimizer_update": update + 1,
        "stage_a_metrics": validated["cell"]["artifacts"]["metrics"],
        "reference": reference,
        "observed": observed,
        "signed_error": errors,
    }
    if not passed:
        raise RuntimeError(f"Identity next-update replay guard failed: {record}")
    return record


def run_factorial_cell(
    owner: torch.nn.Module,
    tokenizer,
    token_ids: torch.Tensor,
    training: dict[str, CompletionDataset],
    fixed: dict[str, CompletionDataset],
    orders: dict[int, list[int]],
    config: dict[str, Any],
    seed: int,
    update: int,
    theta: str,
) -> dict[str, Any]:
    path = cell_path(seed, update, theta)
    if path.exists():
        print(f"[cell/{seed}/u{update:04d}/T={theta}] validated reuse", flush=True)
        return validate_factorial_cell(path, config)
    norm_validated = validate_norm_reference(norm_path(seed, update), config)
    common_norm = float(norm_validated["result"]["symmetric_common_adaptive_l2_norm"])
    payloads: dict[str, dict[str, Any]] = {}
    source_artifacts: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        _, payload, state_path = snapshot_from_replay_cell(config, seed, condition, update)
        payloads[condition] = payload
        source_artifacts[condition] = artifact_record(state_path)
    norm_artifact = artifact_record(norm_path(seed, update))
    root = path.parent
    attempt = next_attempt(root, {"cell.json"})
    attempt.mkdir(parents=True, exist_ok=False)
    identity = cell_identity(seed, update, theta, source_artifacts, norm_artifact, attempt)
    start_path = attempt / "start_manifest.json"
    atomic_write_json(start_path, {
        "created_at": utc_now(),
        "identity": identity,
        "status": "fresh factorial attempt; cell.json is the only reusable sentinel",
    })
    print(f"[cell/{seed}/u{update:04d}/T={theta}] {attempt.name}", flush=True)
    theta_payload = payloads[theta]
    expected_theta_hash = theta_payload["summaries"]["lora_semantic_sha256"]
    try:
        restore_theta(owner, config, theta_payload)
        route, route_record = knockout.route_gradient(
            owner, tokenizer, token_ids, adapted_probe_config(config)
        )
        trait = {
            name: value.detach().float().cpu().double().contiguous()
            for name, value in route.items()
        }
        del route
        if abs(vector_norm(trait) - float(route_record["gradient_l2_norm"])) > 2e-5:
            raise RuntimeError("Route-gradient norm changed during CPU capture")
        baseline, baseline_restoration = decay_only_baseline(
            owner, tokenizer, token_ids, fixed, config, theta_payload
        )
        indices = next_indices(config, orders[seed], update)
        candidates: dict[str, Any] = {}
        for data_condition in CONDITIONS:
            restore_theta(owner, config, theta_payload)
            unclipped, clipped, gradient_record = geometry.numeric_gradient(
                owner, training[data_condition], tokenizer, indices, config
            )
            for m_source, v_source in itertools.product(CONDITIONS, repeat=2):
                key = candidate_key(data_condition, m_source, v_source)
                optimizer = build_hybrid_optimizer(
                    owner, config, theta_payload, payloads[m_source],
                    payloads[v_source], clipped,
                )
                manual = manual_vectors(
                    owner, optimizer, config, update, unclipped, clipped, trait
                )
                identity_next_update = None
                if theta == data_condition == m_source == v_source:
                    identity_next_update = matching_next_update_guard(
                        config, seed, theta, update, gradient_record,
                        float(manual["learning_rate"]),
                    )
                verification = verify_native_step(owner, optimizer, manual, config, update)
                native_outcome = evaluate_state(owner, tokenizer, token_ids, fixed, config)
                native = {
                    "adaptive_l2_norm": manual["adaptive_l2_norm"],
                    "geometry": manual["geometry"],
                    "manual_adamw_verification": verification,
                    "outcome": outcome_summary(native_outcome),
                    "changes_from_theta_decay_only": changes_from_baseline(
                        native_outcome, baseline["outcome"]
                    ),
                }
                restore_theta(owner, config, theta_payload)
                equal_geometry, equal_guard = apply_equal_norm(
                    owner, manual, common_norm, config, trait
                )
                equal_outcome = evaluate_state(owner, tokenizer, token_ids, fixed, config)
                equal = {
                    "adaptive_l2_norm": common_norm,
                    "geometry": equal_geometry,
                    "equal_norm_guard": equal_guard,
                    "manual_adamw_verification": {
                        "passed": True,
                        "not_an_optimizer_step": True,
                        "native_counterpart_verified": True,
                    },
                    "outcome": outcome_summary(equal_outcome),
                    "changes_from_theta_decay_only": changes_from_baseline(
                        equal_outcome, baseline["outcome"]
                    ),
                }
                restore_theta(owner, config, theta_payload)
                observed_hash = semantic_lora_hash(owner)
                restoration = {
                    "expected_lora_semantic_sha256": expected_theta_hash,
                    "observed_lora_semantic_sha256": observed_hash,
                    "passed": observed_hash == expected_theta_hash,
                }
                if not restoration["passed"]:
                    raise RuntimeError(f"Theta restoration failed after candidate {key}")
                candidates[key] = {
                    "theta_source": theta,
                    "exp_avg_source": m_source,
                    "exp_avg_sq_source": v_source,
                    "data_condition": data_condition,
                    "factor_signs": {
                        "T": SIGNS[theta], "M": SIGNS[m_source],
                        "V": SIGNS[v_source], "D": SIGNS[data_condition],
                    },
                    "numeric_gradient": gradient_record,
                    "identity_next_update_replay_guard": identity_next_update,
                    "scales": {"native": native, "equal_norm": equal},
                    "restoration_guard": restoration,
                }
                del optimizer, manual, native_outcome, equal_outcome
                clear_cache()
            del unclipped, clipped
        result = {
            **identity,
            "completed_at": utc_now(),
            "next_example_indices": indices,
            "next_example_indices_int64_sha256": int64_sha256(indices),
            "trait_gradient": route_record,
            "trait_gradient_cpu_l2_norm": vector_norm(trait),
            "theta_decay_only_baseline": baseline,
            "baseline_restoration_guard": baseline_restoration,
            "symmetric_common_adaptive_l2_norm": common_norm,
            "candidates": candidates,
            "evaluated_state_count": 17,
            "no_branch_tensors_written": True,
        }
        result_path = attempt / "result.json"
        atomic_write_json(result_path, result)
    except BaseException as error:
        atomic_write_json(attempt / "failure.json", {**identity, "failed_at": utc_now(), "error": repr(error)})
        raise
    sentinel = {
        **identity,
        "completed_at": result["completed_at"],
        "start_manifest": artifact_record(start_path),
        "result": artifact_record(result_path),
    }
    exclusive_write_json(path, sentinel)
    print(f"[cell/{seed}/u{update:04d}/T={theta}] DONE 8 native + 8 equal", flush=True)
    return validate_factorial_cell(path, config)


def analyze() -> dict[str, Any]:
    """Run strong tensor-parent validation, then delegate pure JSON inference."""
    config = load_config()
    replay.validate_runner_lock()
    validate_factorial_lock()
    for seed in seeds(config):
        for update in checkpoints(config):
            validate_norm_reference(norm_path(seed, update), config)
            for theta in CONDITIONS:
                validate_factorial_cell(cell_path(seed, update, theta), config)
    import ds2_adam_source_analysis as pure_analysis

    result = pure_analysis.analyze(
        CONFIG_PATH,
        work_root=WORK,
        output_json=OUT_JSON,
        output_markdown=OUT_MD,
        write=True,
    )
    print(
        f"FACTORIAL ANALYSIS DONE {result['continuation_decision']['recommendation']}",
        flush=True,
    )
    return result


def validate_recursive_parent_hashes(config: dict[str, Any]) -> dict[str, Any]:
    checked: dict[str, Any] = {}

    def visit(value: Any, label: str) -> None:
        if (
            isinstance(value, list)
            and len(value) == 2
            and all(isinstance(item, str) for item in value)
            and re.fullmatch(r"[0-9a-f]{64}", value[1])
        ):
            path = ROOT / value[0]
            observed = file_sha256(path)
            if observed != value[1]:
                raise RuntimeError(f"Frozen parent hash mismatch: {label}/{path}")
            checked[label] = {"path": value[0], "sha256": observed}
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{label}.{key}" if label else key)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{label}[{index}]")

    visit(config["parents"], "parents")
    if len(checked) < 10:
        raise RuntimeError(f"Recursive parent hash inventory unexpectedly small: {len(checked)}")
    return checked


def preflight(require_absence: bool = False) -> dict[str, Any]:
    config = load_config()
    stage_a = replay.preflight(require_absence=False)
    assert_no_competing_experiment()
    if config["resource_policy"]["serial_mps_only"] and DEVICE.type != "mps":
        raise RuntimeError(f"Factorial campaign requires MPS, found {DEVICE}")
    free = shutil.disk_usage(ROOT).free
    if free < int(config["resource_policy"]["minimum_launch_free_bytes"]):
        raise RuntimeError(f"Launch free-space guard failed: {free}")
    parents = validate_recursive_parent_hashes(config)
    if require_absence:
        scoped = (NORMS, CELLS, FACTORIAL_LOCK_PATH, OUT_JSON, OUT_MD)
        existing = [relative(path) for path in scoped if path.exists()]
        if existing:
            raise RuntimeError(f"Stage-B namespace predates freeze: {existing}")
    orders = {
        str(seed): {
            "two_epoch_int64_sha256": int64_sha256(historical_order(config, seed)),
            "next_indices": {
                str(update): next_indices(config, historical_order(config, seed), update)
                for update in checkpoints(config)
            },
        }
        for seed in seeds(config)
    }
    return {
        "implementation": implementation_guard(),
        "stage_a_preflight": stage_a,
        "recursive_parent_hashes": parents,
        "orders": orders,
        "free_bytes": free,
        "expected_norm_references": [
            relative(norm_path(seed, update))
            for seed in seeds(config) for update in checkpoints(config)
        ],
        "expected_factorial_cells": [
            relative(cell_path(seed, update, theta))
            for seed in seeds(config) for update in checkpoints(config)
            for theta in CONDITIONS
        ],
        "preflight_used_model_forward_or_backward": False,
    }


def freeze() -> dict[str, Any]:
    if not RUNNER_LOCK_PATH.exists():
        replay.freeze()
    replay.validate_runner_lock()
    if FACTORIAL_LOCK_PATH.exists():
        return validate_factorial_lock()
    frozen = preflight(require_absence=True)
    record = {
        "name": "ds2-adam-source-factorial-v1-stage-b-runner-lock",
        "created_at": utc_now(),
        "stage_a_lock": artifact_record(RUNNER_LOCK_PATH),
        "implementation": frozen["implementation"],
    }
    exclusive_write_json(FACTORIAL_LOCK_PATH, record)
    print(f"FACTORIAL STAGE B FROZEN {file_sha256(FACTORIAL_LOCK_PATH)}", flush=True)
    return validate_factorial_lock()


def runtime_space_guard(config: dict[str, Any]) -> None:
    free = shutil.disk_usage(ROOT).free
    minimum = int(config["resource_policy"]["minimum_runtime_free_bytes"])
    if free < minimum:
        raise RuntimeError(f"Runtime free-space guard failed: {free} < {minimum}")


def run_all() -> dict[str, Any]:
    config = load_config()
    replay.validate_runner_lock()
    validate_factorial_lock()
    assert_no_competing_experiment()
    if any(
        not replay.expected_cell_path(seed, condition).is_file()
        for seed in seeds(config) for condition in CONDITIONS
    ):
        # Stage A owns its own active lock and exact-replay guards.  This call is
        # made only after the process audit above, so it cannot duplicate MPS work.
        replay.replay_all()
    for seed in seeds(config):
        for condition in CONDITIONS:
            replay.validate_replay_cell(replay.expected_cell_path(seed, condition), config)
    assert_no_competing_experiment()
    with active_lock():
        tokenizer = dynamics.load_tokenizer()
        token_ids = geometry.animal_token_ids(config, tokenizer)
        training, fixed = prepare_datasets(config, tokenizer)
        orders = {seed: historical_order(config, seed) for seed in seeds(config)}
        completed = 0
        total = int(config["measurement"]["norm_reference_count"]) + int(
            config["measurement"]["factorial_cell_count"]
        )
        for seed in seeds(config):
            runtime_space_guard(config)
            owner = knockout.load_owner(config, seed)
            try:
                for update in checkpoints(config):
                    runtime_space_guard(config)
                    run_norm_reference(
                        owner, tokenizer, training, orders, config, seed, update
                    )
                    completed += 1
                    print(f"FACTORIAL PROGRESS {completed}/{total}", flush=True)
                    for theta in CONDITIONS:
                        runtime_space_guard(config)
                        run_factorial_cell(
                            owner, tokenizer, token_ids, training, fixed, orders,
                            config, seed, update, theta,
                        )
                        completed += 1
                        print(f"FACTORIAL PROGRESS {completed}/{total}", flush=True)
            finally:
                release(owner)
    result = analyze()
    print("DS2 ADAM SOURCE FACTORIAL RUN DONE", flush=True)
    return result


def status() -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": "ds2-adam-source-factorial-v1-status",
        "checked_at": utc_now(),
        "device": str(DEVICE),
        "stage_a": replay.status(),
        "factorial_lock": {
            "exists": FACTORIAL_LOCK_PATH.is_file(),
            "valid": False,
            "error": None,
        },
        "norm_references": {"valid": 0, "expected": 0, "invalid": []},
        "factorial_cells": {"valid": 0, "expected": 0, "invalid": []},
        "aggregate": {"json": OUT_JSON.is_file(), "markdown": OUT_MD.is_file()},
        "status_loaded_model_or_used_mps": False,
    }
    try:
        config = load_config()
    except BaseException as error:
        result["config_error"] = repr(error)
        return result
    if FACTORIAL_LOCK_PATH.is_file():
        try:
            validate_factorial_lock()
            result["factorial_lock"]["valid"] = True
        except BaseException as error:
            result["factorial_lock"]["error"] = repr(error)
    norm_paths = [
        norm_path(seed, update)
        for seed in seeds(config) for update in checkpoints(config)
    ]
    factor_paths = [
        cell_path(seed, update, theta)
        for seed in seeds(config) for update in checkpoints(config)
        for theta in CONDITIONS
    ]
    result["norm_references"]["expected"] = len(norm_paths)
    result["factorial_cells"]["expected"] = len(factor_paths)
    if result["factorial_lock"]["valid"]:
        for path in norm_paths:
            if not path.exists():
                continue
            try:
                validate_norm_reference(path, config)
                result["norm_references"]["valid"] += 1
            except BaseException as error:
                result["norm_references"]["invalid"].append(
                    {"path": relative(path), "error": repr(error)}
                )
        for path in factor_paths:
            if not path.exists():
                continue
            try:
                validate_factorial_cell(path, config)
                result["factorial_cells"]["valid"] += 1
            except BaseException as error:
                result["factorial_cells"]["invalid"].append(
                    {"path": relative(path), "error": repr(error)}
                )
    result["complete"] = (
        result["factorial_lock"]["valid"]
        and result["norm_references"]["valid"] == len(norm_paths)
        and result["factorial_cells"]["valid"] == len(factor_paths)
        and OUT_JSON.is_file() and OUT_MD.is_file()
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "freeze", "replay", "run", "analyze", "status")
    )
    args = parser.parse_args()
    if args.command == "preflight":
        value = preflight(require_absence=False)
    elif args.command == "freeze":
        value = freeze()
    elif args.command == "replay":
        value = replay.replay_all()
    elif args.command == "run":
        value = run_all()
    elif args.command == "analyze":
        value = analyze()
    else:
        value = status()
    if args.command in {"preflight", "freeze", "status"}:
        print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
