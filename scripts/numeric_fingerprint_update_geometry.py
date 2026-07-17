"""Frozen saved-state update-geometry probe for numeric subliminal learning.

For each preregistered dynamics-v1 snapshot this runner reconstructs the exact
next two historical batch-8 microbatches, measures the held-out wolf-margin
gradient, and forks one AdamW step on preference and control rows.  The matching
fork is the live historical update; the opposite fork is a same-state
counterfactual.  Manual AdamW components are checked against ``optimizer.step``
and the post-step wolf margin is evaluated directly.

No model or optimizer state is written.  A cell is reusable only after its
``cell.json`` sentinel is atomically committed; interrupted attempts are kept.
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
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

import numeric_fingerprint_compatibility as compatibility
import numeric_fingerprint_dynamics as dynamics
from polypythia_sl.data import PREFERENCE_EVAL_PROMPTS
from polypythia_sl.modeling import assert_single_token_animals
from polypythia_sl.optim import build_optimizer
from polypythia_sl.train import CompletionCollator, CompletionDataset, seed_everything


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
WORK = RUNS / "numeric_fingerprint_update_geometry_v1"
CELLS = WORK / "cells"
CONFIG_PATH = ROOT / "configs/numeric_fingerprint_update_geometry_v1.json"
SCRIPT_PATH = Path(__file__).resolve()
RUNNER_LOCK_PATH = WORK / "runner_lock.json"
ACTIVE_LOCK_PATH = WORK / ".active.lock"
OUT_JSON = RUNS / "numeric_fingerprint_update_geometry_v1.json"
OUT_MD = RUNS / "numeric_fingerprint_update_geometry_v1.md"
LOG_PATH = RUNS / "numeric_fingerprint_update_geometry_v1.log"

RECEIVERS = ("standard", "weight_seed3")
SEEDS = (56101, 56102)
CONDITIONS = ("preference", "control")
CHECKPOINTS = (0, 16, 64, 128, 256, 512, 1024, 1536, 2048)
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
    return {
        "path": relative(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def clear_cache() -> None:
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def release(model: torch.nn.Module | None) -> None:
    if model is not None:
        model.to("cpu")
    del model
    clear_cache()


def implementation_guard() -> dict[str, Any]:
    return {
        "runner_sha256": file_sha256(SCRIPT_PATH),
        "config_sha256": file_sha256(CONFIG_PATH),
        "dynamics_runner_sha256": file_sha256(dynamics.SCRIPT_PATH),
        "compatibility_runner_sha256": file_sha256(compatibility.SCRIPT_PATH),
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


def source_key(receiver: str, seed: int, condition: str) -> str:
    return f"{receiver}/{seed}/{condition}"


def expected_cell_path(
    receiver: str, seed: int, condition: str, update: int
) -> Path:
    return (
        CELLS / receiver / f"seed_{seed}" / condition
        / f"u{update:04d}" / "cell.json"
    )


def expected_cell_paths() -> list[Path]:
    return [
        expected_cell_path(receiver, seed, condition, update)
        for receiver in RECEIVERS
        for seed in SEEDS
        for condition in CONDITIONS
        for update in CHECKPOINTS
    ]


def load_and_validate_config() -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_json(CONFIG_PATH)
    if config.get("name") != "numeric-fingerprint-update-geometry-v1":
        raise RuntimeError("Unexpected update-geometry config identity")
    measurement = config["measurement"]
    if (
        tuple(measurement["receiver_order"]) != RECEIVERS
        or tuple(measurement["seeds"]) != SEEDS
        or tuple(measurement["source_conditions"]) != CONDITIONS
        or tuple(measurement["fork_conditions"]) != CONDITIONS
        or tuple(measurement["checkpoints"]) != CHECKPOINTS
        or measurement["excluded_checkpoint"] != {
            "optimizer_update": 2560,
            "reason": "The frozen schedule exposes zero learning rate for the next update.",
        }
    ):
        raise RuntimeError("Frozen update-geometry grid changed")
    if (
        measurement["batch_size"] != 8
        or measurement["gradient_accumulation_steps"] != 2
        or measurement["examples_per_update"] != 16
        or measurement["optimizer"] != "adamw"
        or measurement["behavior_prompt_count"] != len(PREFERENCE_EVAL_PROMPTS)
        or measurement["behavior_prompt_sha256"]
        != compact_hash(list(PREFERENCE_EVAL_PROMPTS))
    ):
        raise RuntimeError("Measurement recipe changed")

    parents = config["parents"]
    for field in (
        "dynamics_config", "dynamics_runner", "dynamics_result",
        "heldout_manifest", "compatibility_runner", "optim_py", "train_py",
        "evaluate_py", "data_py", "modeling_py",
    ):
        path = ROOT / parents[field]
        if file_sha256(path) != parents[f"{field}_sha256"]:
            raise RuntimeError(f"Frozen parent changed: {path}")
    parent = dynamics.load_and_validate_config()
    parent_result = load_json(ROOT / parents["dynamics_result"])
    if parent_result.get("trajectory_decision", {}).get("label") != "transient_access":
        raise RuntimeError("Expected the frozen dynamics transient-access result")
    for receiver in RECEIVERS:
        for key in ("model_id", "commit", "weight_sha256", "model_config_sha256"):
            if config["receivers"][receiver][key] != parent["receivers"][receiver][key]:
                raise RuntimeError(f"Receiver provenance changed: {receiver}/{key}")
    for key in (
        "batch_size", "gradient_accumulation_steps", "learning_rate",
        "optimizer", "betas", "eps", "weight_decay", "max_grad_norm",
        "warmup_updates", "schedule_total_updates", "max_length", "lora",
        "expected_trainable_parameters", "expected_initial_lora_state_sha256",
    ):
        if measurement[key] != parent["training"][key]:
            raise RuntimeError(f"Measurement diverges from dynamics: {key}")
    if config["data"] != parent["data"]:
        raise RuntimeError("Frozen data identity changed")
    if config["frozen_analysis"]["no_single_checkpoint_gate"] is not True:
        raise RuntimeError("Single-checkpoint gate was enabled")
    expected_artifacts = {
        "root": relative(WORK),
        "runner": relative(SCRIPT_PATH),
        "runner_lock": relative(RUNNER_LOCK_PATH),
        "aggregate_json": relative(OUT_JSON),
        "aggregate_markdown": relative(OUT_MD),
        "log": relative(LOG_PATH),
    }
    if any(config["artifacts"].get(key) != value for key, value in expected_artifacts.items()):
        raise RuntimeError("Artifact namespace changed")
    expected_sources = {
        source_key(receiver, seed, condition)
        for receiver in RECEIVERS for seed in SEEDS for condition in CONDITIONS
    }
    if set(parents["trajectories"]) != expected_sources:
        raise RuntimeError("Frozen source trajectory inventory changed")
    for key, (path_text, expected_hash) in parents["trajectories"].items():
        path = ROOT / path_text
        if file_sha256(path) != expected_hash:
            raise RuntimeError(f"Frozen trajectory changed: {key}")
    return config, parent


def load_source(
    config: dict[str, Any], parent: dict[str, Any], receiver: str,
    seed: int, condition: str, update: int, validate_tensors: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    path_text, expected_hash = config["parents"]["trajectories"][
        source_key(receiver, seed, condition)
    ]
    trajectory_path = ROOT / path_text
    if file_sha256(trajectory_path) != expected_hash:
        raise RuntimeError(f"Trajectory changed: {trajectory_path}")
    trajectory = load_json(trajectory_path)
    if (
        trajectory.get("receiver") != receiver
        or int(trajectory.get("seed", -1)) != seed
        or trajectory.get("condition") != condition
        or update not in trajectory.get("probe_updates", [])
    ):
        raise RuntimeError(f"Trajectory identity mismatch: {trajectory_path}")
    artifact = trajectory["artifacts"][f"state_u{update:04d}"]
    state_path = ROOT / artifact["path"]
    if (
        not state_path.is_file()
        or state_path.stat().st_size != artifact["bytes"]
        or file_sha256(state_path) != artifact["sha256"]
    ):
        raise RuntimeError(f"State artifact changed: {state_path}")
    if validate_tensors:
        dynamics.validate_state_snapshot(
            state_path, parent, receiver, seed, condition, update
        )
    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    return trajectory, payload, state_path


def historical_orders(parent: dict[str, Any], seed: int) -> list[int]:
    count = int(parent["data"]["rows_per_condition"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dynamics.IndexDataset(count),
        batch_size=int(parent["training"]["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    epochs: list[list[int]] = []
    for _ in range(int(parent["training"]["epochs"])):
        order = [int(value) for batch in loader for value in batch.tolist()]
        if sorted(order) != list(range(count)):
            raise RuntimeError(f"Invalid historical order: {seed}")
        epochs.append(order)
    guard = {
        "epoch_sha256": [int64_sha256(order) for order in epochs],
        "combined_sha256": int64_sha256(value for order in epochs for value in order),
    }
    if guard != parent["training"]["data_order_guards"][str(seed)]:
        raise RuntimeError(f"Historical DataLoader order changed: {seed}")
    return [value for epoch in epochs for value in epoch]


def next_indices(parent: dict[str, Any], order: list[int], update: int) -> list[int]:
    per_update = (
        int(parent["training"]["batch_size"])
        * int(parent["training"]["gradient_accumulation_steps"])
    )
    start = update * per_update
    indices = order[start:start + per_update]
    if len(indices) != per_update:
        raise RuntimeError(f"No complete next update after {update}")
    return indices


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
        "scripts/numeric_", "scripts/dataorder_", "scripts/base_screening.py",
        "scripts/student_trait_write_probe.py", "scripts/cross_family_transport.py",
        "polypythia_sl.pipeline",
    )
    conflicts = []
    for pid, (_, command) in processes.items():
        if pid in ancestors or "python" not in command.lower():
            continue
        if (
            command.lstrip().startswith("caffeinate ")
            and SCRIPT_PATH.name in command
        ):
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
            raise RuntimeError(f"Update-geometry runner already active: {handle.read()}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at": utc_now()}))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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
            target_modules=list(lora["target_modules"]), task_type="CAUSAL_LM",
        ),
    ).to(DEVICE)
    owner.config.use_cache = False
    trainable = dynamics.canonical_trainable(owner)
    count = sum(parameter.numel() for _, parameter in trainable)
    if count != int(config["measurement"]["expected_trainable_parameters"]):
        raise RuntimeError(f"Unexpected trainable parameter count: {count}")
    initial_hash = dynamics.tensor_hash((name, parameter) for name, parameter in trainable)
    expected = config["measurement"]["expected_initial_lora_state_sha256"][str(seed)]
    if initial_hash != expected:
        raise RuntimeError(f"Initial LoRA hash changed: {receiver}/{seed}")
    return owner


def restore_snapshot(
    owner: torch.nn.Module, config: dict[str, Any], payload: dict[str, Any]
) -> torch.optim.Optimizer:
    trainable = dynamics.canonical_trainable(owner)
    lora_rows = {row["name"]: row for row in payload["lora"]}
    adam_rows = {row["name"]: row for row in payload["adam"]}
    names = [name for name, _ in trainable]
    if set(lora_rows) != set(names) or set(adam_rows) != set(names):
        raise RuntimeError("Snapshot names differ from live LoRA names")
    with torch.no_grad():
        for name, parameter in trainable:
            value = lora_rows[name]["tensor"]
            if value.shape != parameter.shape:
                raise RuntimeError(f"LoRA shape mismatch: {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    if dynamics.semantic_tensor_hash(
        (name, parameter.detach().cpu()) for name, parameter in trainable
    ) != payload["summaries"]["lora_semantic_sha256"]:
        raise RuntimeError("Restored LoRA semantic hash mismatch")
    recipe = config["measurement"]
    optimizer, _ = build_optimizer(owner, recipe)
    optimizer.state.clear()
    update = int(payload["optimizer_update"])
    for name, parameter in trainable:
        row = adam_rows[name]
        optimizer.state[parameter] = {
            "step": row["step"].detach().clone().cpu(),
            "exp_avg": row["exp_avg"].to(parameter.device, parameter.dtype).clone(),
            "exp_avg_sq": row["exp_avg_sq"].to(parameter.device, parameter.dtype).clone(),
        }
        if int(float(optimizer.state[parameter]["step"].item())) != update:
            raise RuntimeError(f"Restored Adam step mismatch: {name}")
    group = optimizer.param_groups[0]
    group["lr"] = float(payload["summaries"]["lr_available_for_next_update"])
    if (
        list(group["betas"]) != recipe["betas"]
        or float(group["eps"]) != float(recipe["eps"])
        or float(group["weight_decay"]) != float(recipe["weight_decay"])
    ):
        raise RuntimeError("Restored AdamW hyperparameters changed")
    return optimizer


def capture_gradients(
    trainable: list[tuple[str, torch.nn.Parameter]]
) -> dict[str, torch.Tensor]:
    return {
        name: (
            torch.zeros_like(parameter, device="cpu", dtype=torch.float64)
            if parameter.grad is None
            else parameter.grad.detach().float().cpu().double().clone()
        )
        for name, parameter in trainable
    }


def vector_dot(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> float:
    return float(sum(torch.sum(left[name] * right[name]) for name in left))


def vector_norm(value: dict[str, torch.Tensor]) -> float:
    return math.sqrt(max(vector_dot(value, value), 0.0))


def gradient_diagnostics(value: dict[str, torch.Tensor]) -> dict[str, Any]:
    total = vector_dot(value, value)
    a = sum(float(torch.sum(t * t)) for n, t in value.items() if ".lora_A." in n)
    b = sum(float(torch.sum(t * t)) for n, t in value.items() if ".lora_B." in n)
    return {
        "l2_norm": math.sqrt(max(total, 0.0)),
        "squared_norm_fraction_lora_A": a / total if total else 0.0,
        "squared_norm_fraction_lora_B": b / total if total else 0.0,
        "tensor_count": len(value),
    }


def animal_token_ids(config: dict[str, Any], tokenizer) -> torch.Tensor:
    animals = [
        config["measurement"]["trait_target"],
        *config["measurement"]["comparison_animals"],
    ]
    mapping = assert_single_token_animals(tokenizer, animals)
    return torch.tensor([mapping[animal] for animal in animals], device=DEVICE)


def wolf_margin_values(
    probe_model: torch.nn.Module, tokenizer, token_ids: torch.Tensor,
    batch_size: int,
) -> list[float]:
    values: list[float] = []
    probe_model.eval()
    with torch.inference_mode():
        for start in range(0, len(PREFERENCE_EVAL_PROMPTS), batch_size):
            prompts = PREFERENCE_EVAL_PROMPTS[start:start + batch_size]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True)
            encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
            logits = probe_model(**encoded, use_cache=False).logits
            last = encoded["attention_mask"].sum(1) - 1
            rows = torch.arange(len(prompts), device=DEVICE)
            selected = logits[rows, last][:, token_ids].float()
            margins = selected[:, 0] - torch.logsumexp(selected[:, 1:], dim=1) + math.log(9)
            values.extend(float(value) for value in margins.cpu().tolist())
    return values


def trait_gradient(
    owner: torch.nn.Module, tokenizer, token_ids: torch.Tensor,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    probe_model = owner.base_model.model
    trainable = dynamics.canonical_trainable(owner)
    owner.eval()
    owner.zero_grad(set_to_none=True)
    values: list[float] = []
    for start in range(0, len(PREFERENCE_EVAL_PROMPTS), batch_size):
        prompts = PREFERENCE_EVAL_PROMPTS[start:start + batch_size]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
        logits = probe_model(**encoded, use_cache=False).logits
        last = encoded["attention_mask"].sum(1) - 1
        rows = torch.arange(len(prompts), device=DEVICE)
        selected = logits[rows, last][:, token_ids].float()
        margins = selected[:, 0] - torch.logsumexp(selected[:, 1:], dim=1) + math.log(9)
        values.extend(float(value) for value in margins.detach().cpu().tolist())
        (margins.sum() / len(PREFERENCE_EVAL_PROMPTS)).backward()
    gradient = capture_gradients(trainable)
    owner.zero_grad(set_to_none=True)
    array = np.asarray(values, dtype=np.float64)
    record = {
        "mean": float(array.mean()),
        "prompts_1_30_mean": float(array[:30].mean()),
        "prompts_31_60_mean": float(array[30:].mean()),
        "prompt_count": len(values),
        "gradient": gradient_diagnostics(gradient),
    }
    return gradient, record


def numeric_gradient(
    owner: torch.nn.Module, dataset: CompletionDataset, tokenizer,
    indices: list[int], config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    recipe = config["measurement"]
    batch_size = int(recipe["batch_size"])
    accumulation = int(recipe["gradient_accumulation_steps"])
    if len(indices) != batch_size * accumulation:
        raise RuntimeError("One-step fork changed effective batch size")
    collator = CompletionCollator(tokenizer.pad_token_id)
    trainable = dynamics.canonical_trainable(owner)
    owner.train()
    owner.zero_grad(set_to_none=True)
    losses: list[float] = []
    for start in range(0, len(indices), batch_size):
        batch = collator([dataset[index] for index in indices[start:start + batch_size]])
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        loss = owner(**batch, use_cache=False).loss
        losses.append(float(loss.detach().cpu()))
        (loss / accumulation).backward()
    if len(losses) != accumulation:
        raise RuntimeError("Fork did not use exactly two microbatches")
    unclipped = capture_gradients(trainable)
    unclipped_norm = vector_norm(unclipped)
    returned_norm = torch.nn.utils.clip_grad_norm_(
        owner.parameters(), float(recipe["max_grad_norm"])
    )
    clipped = capture_gradients(trainable)
    clipped_norm = vector_norm(clipped)
    if abs(float(returned_norm.detach().cpu()) - unclipped_norm) > 2e-4:
        raise RuntimeError("clip_grad_norm returned a different pre-clip norm")
    return unclipped, clipped, {
        "microbatch_losses": losses,
        "mean_microbatch_loss": float(np.mean(losses)),
        "effective_examples": len(indices),
        "microbatch_count": len(losses),
        "loss_divisor_per_microbatch": accumulation,
        "gradient_norm_before_clipping": unclipped_norm,
        "gradient_norm_after_clipping": clipped_norm,
        "clip_scale": clipped_norm / unclipped_norm if unclipped_norm else 1.0,
    }


def historical_next_update(
    trajectory: dict[str, Any], update: int
) -> dict[str, Any]:
    artifact = trajectory["artifacts"]["metrics"]
    path = ROOT / artifact["path"]
    if (
        path.stat().st_size != artifact["bytes"]
        or file_sha256(path) != artifact["sha256"]
    ):
        raise RuntimeError(f"Historical metrics changed: {path}")
    metrics = load_json(path)
    rows = metrics["update_metrics"]
    row = rows[update]
    if int(row["optimizer_update"]) != update + 1:
        raise RuntimeError("Historical next-update indexing changed")
    return row


def check_matching_replay(
    trajectory: dict[str, Any], update: int, lr: float,
    gradient_record: dict[str, Any],
) -> dict[str, Any]:
    expected = historical_next_update(trajectory, update)
    observed = {
        "mean_microbatch_loss": gradient_record["mean_microbatch_loss"],
        "gradient_norm_before_clipping": gradient_record["gradient_norm_before_clipping"],
        "learning_rate_used": lr,
    }
    reference = {
        "mean_microbatch_loss": float(expected["mean_microbatch_loss"]),
        "gradient_norm_before_clipping": float(expected["gradient_norm_before_clipping"]),
        "learning_rate_used": float(
            load_json(ROOT / trajectory["artifacts"]["metrics"]["path"])[
                "learning_rate_used_for_update"
            ][str(update + 1)]
        ),
    }
    errors = {key: observed[key] - reference[key] for key in observed}
    passed = (
        abs(errors["mean_microbatch_loss"]) <= 5e-5
        and abs(errors["gradient_norm_before_clipping"]) <= 5e-4
        and abs(errors["learning_rate_used"]) <= 1e-15
    )
    if not passed:
        raise RuntimeError(
            f"Matching fork did not replay archived next update: {errors}"
        )
    return {"passed": True, "expected": reference, "observed": observed, "errors": errors}


def check_archived_update1_state(
    owner: torch.nn.Module, optimizer: torch.optim.Optimizer,
    trajectory: dict[str, Any], config: dict[str, Any],
) -> dict[str, Any]:
    artifact = trajectory["artifacts"]["state_u0001"]
    path = ROOT / artifact["path"]
    if (
        path.stat().st_size != artifact["bytes"]
        or file_sha256(path) != artifact["sha256"]
    ):
        raise RuntimeError(f"Archived update-1 state changed: {path}")
    expected = torch.load(path, map_location="cpu", weights_only=True)
    if int(expected.get("optimizer_update", -1)) != 1:
        raise RuntimeError("Archived replay target is not update 1")
    lora_rows = {row["name"]: row["tensor"] for row in expected["lora"]}
    adam_rows = {row["name"]: row for row in expected["adam"]}
    lora_named: list[tuple[str, torch.Tensor]] = []
    m_named: list[tuple[str, torch.Tensor]] = []
    v_named: list[tuple[str, torch.Tensor]] = []
    errors: dict[str, dict[str, float]] = {}
    maximum = 0.0
    for kind in ("lora", "exp_avg", "exp_avg_sq"):
        errors[kind] = {"squared_error": 0.0, "squared_reference": 0.0}
    for name, parameter in dynamics.canonical_trainable(owner):
        state = optimizer.state[parameter]
        observed_values = {
            "lora": parameter.detach().float().cpu().contiguous(),
            "exp_avg": state["exp_avg"].detach().float().cpu().contiguous(),
            "exp_avg_sq": state["exp_avg_sq"].detach().float().cpu().contiguous(),
        }
        expected_values = {
            "lora": lora_rows[name],
            "exp_avg": adam_rows[name]["exp_avg"],
            "exp_avg_sq": adam_rows[name]["exp_avg_sq"],
        }
        if int(float(state["step"].detach().cpu().item())) != 1:
            raise RuntimeError(f"Runtime update-1 Adam step mismatch: {name}")
        if int(float(adam_rows[name]["step"].item())) != 1:
            raise RuntimeError(f"Archived update-1 Adam step mismatch: {name}")
        for kind in observed_values:
            error = observed_values[kind].double() - expected_values[kind].double()
            maximum = max(maximum, float(error.abs().max()))
            errors[kind]["squared_error"] += float(torch.sum(error * error))
            reference = expected_values[kind].double()
            errors[kind]["squared_reference"] += float(torch.sum(reference * reference))
        lora_named.append((name, observed_values["lora"]))
        m_named.append((name, observed_values["exp_avg"]))
        v_named.append((name, observed_values["exp_avg_sq"]))
    relative = {
        kind: math.sqrt(values["squared_error"]) / max(
            math.sqrt(values["squared_reference"]), 1e-30
        )
        for kind, values in errors.items()
    }
    observed_hashes = {
        "lora_semantic_sha256": dynamics.semantic_tensor_hash(lora_named),
        "adam_exp_avg_semantic_sha256": dynamics.semantic_tensor_hash(m_named),
        "adam_exp_avg_sq_semantic_sha256": dynamics.semantic_tensor_hash(v_named),
    }
    expected_hashes = {
        key: expected["summaries"][key] for key in observed_hashes
    }
    recipe = config["measurement"]
    passed = (
        maximum <= float(recipe["update1_state_replay_max_abs_tolerance"])
        and all(
            value <= float(recipe["update1_state_replay_relative_l2_tolerance"])
            for value in relative.values()
        )
    )
    result = {
        "artifact": artifact,
        "maximum_tensor_absolute_error": maximum,
        "relative_l2_error": relative,
        "observed_semantic_hashes": observed_hashes,
        "expected_semantic_hashes": expected_hashes,
        "semantic_hashes_exact": observed_hashes == expected_hashes,
        "adam_steps_exact": True,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"Runtime update 1 did not replay archived state: {result}")
    return result


def one_step_fork(
    owner: torch.nn.Module, tokenizer, token_ids: torch.Tensor,
    config: dict[str, Any], payload: dict[str, Any], trajectory: dict[str, Any],
    trait: dict[str, torch.Tensor], baseline_margin: float,
    dataset: CompletionDataset, indices: list[int], fork_condition: str,
    source_condition: str, update: int,
) -> dict[str, Any]:
    optimizer = restore_snapshot(owner, config, payload)
    trainable = dynamics.canonical_trainable(owner)
    unclipped, clipped, gradient_record = numeric_gradient(
        owner, dataset, tokenizer, indices, config
    )
    lr = float(optimizer.param_groups[0]["lr"])
    replay = None
    if fork_condition == source_condition:
        replay = check_matching_replay(trajectory, update, lr, gradient_record)

    recipe = config["measurement"]
    beta1, beta2 = (float(value) for value in recipe["betas"])
    eps = float(recipe["eps"])
    weight_decay = float(recipe["weight_decay"])
    next_step = update + 1
    bc1 = 1.0 - beta1**next_step
    bc2 = 1.0 - beta2**next_step
    step_size = lr / bc1
    manual_after: dict[str, torch.Tensor] = {}
    before: dict[str, torch.Tensor] = {}
    component_projection = {
        "unclipped_raw_descent_direction": 0.0,
        "clipped_raw_descent_direction": 0.0,
        "clipped_raw_lr_scaled_update": 0.0,
        "unpreconditioned_bias_corrected_momentum_lr_scaled_update": 0.0,
        "adam_history_adaptive_update": 0.0,
        "adam_current_gradient_adaptive_update": 0.0,
        "adam_preconditioned_adaptive_update": 0.0,
        "weight_decay_update": 0.0,
        "manual_total_update": 0.0,
    }
    component_norm_sq = {key: 0.0 for key in component_projection}
    adaptive_decomposition_max_error = 0.0
    with torch.no_grad():
        for name, parameter in trainable:
            old = parameter.detach().clone()
            state = optimizer.state[parameter]
            gradient = parameter.grad.detach()
            m_old = state["exp_avg"].detach().clone()
            history_numerator = m_old * beta1
            current_numerator = gradient * (1.0 - beta1)
            m_new = m_old.clone().lerp_(gradient, 1.0 - beta1)
            v_new = state["exp_avg_sq"].detach().clone().mul_(beta2).addcmul_(
                gradient, gradient, value=1.0 - beta2
            )
            m_hat = m_new / bc1
            momentum_update = -lr * m_hat
            denom = v_new.sqrt().div_(math.sqrt(bc2)).add_(eps)
            adaptive = torch.zeros_like(old).addcdiv_(m_new, denom, value=-step_size)
            history_adaptive = torch.zeros_like(old).addcdiv_(
                history_numerator, denom, value=-step_size
            )
            current_adaptive = torch.zeros_like(old).addcdiv_(
                current_numerator, denom, value=-step_size
            )
            adaptive_component_error = float(
                (adaptive - history_adaptive - current_adaptive).abs().max().detach().cpu()
            )
            adaptive_decomposition_max_error = max(
                adaptive_decomposition_max_error, adaptive_component_error
            )
            if adaptive_component_error > 2e-7:
                raise RuntimeError(
                    f"Adam history/current decomposition failed: {name} "
                    f"max_error={adaptive_component_error}"
                )
            decay = old * (-lr * weight_decay)
            predicted = old.clone().mul_(1.0 - lr * weight_decay)
            predicted.addcdiv_(m_new, denom, value=-step_size)
            total = predicted - old
            vectors = {
                "unclipped_raw_descent_direction": -unclipped[name],
                "clipped_raw_descent_direction": -clipped[name],
                "clipped_raw_lr_scaled_update": -lr * clipped[name],
                "unpreconditioned_bias_corrected_momentum_lr_scaled_update": momentum_update.float().cpu().double(),
                "adam_history_adaptive_update": history_adaptive.float().cpu().double(),
                "adam_current_gradient_adaptive_update": current_adaptive.float().cpu().double(),
                "adam_preconditioned_adaptive_update": adaptive.float().cpu().double(),
                "weight_decay_update": decay.float().cpu().double(),
                "manual_total_update": total.float().cpu().double(),
            }
            for key, vector in vectors.items():
                component_projection[key] += float(torch.sum(trait[name] * vector))
                component_norm_sq[key] += float(torch.sum(vector * vector))
            before[name] = old.float().cpu()
            manual_after[name] = predicted.float().cpu()

    optimizer.step()
    owner.zero_grad(set_to_none=True)
    actual_projection = 0.0
    actual_norm_sq = 0.0
    error_sq = 0.0
    after_sq = 0.0
    maximum_error = 0.0
    with torch.no_grad():
        for name, parameter in trainable:
            after = parameter.detach().float().cpu()
            actual = after.double() - before[name].double()
            error = after.double() - manual_after[name].double()
            actual_projection += float(torch.sum(trait[name] * actual))
            actual_norm_sq += float(torch.sum(actual * actual))
            error_sq += float(torch.sum(error * error))
            after_sq += float(torch.sum(after.double() * after.double()))
            maximum_error = max(maximum_error, float(error.abs().max()))
            state_step = int(float(optimizer.state[parameter]["step"].detach().cpu().item()))
            if state_step != next_step:
                raise RuntimeError(f"AdamW step did not advance exactly once: {name}")
    relative_l2 = math.sqrt(error_sq) / max(math.sqrt(after_sq), 1e-30)
    relative_update_l2 = math.sqrt(error_sq) / max(math.sqrt(actual_norm_sq), 1e-30)
    verification = {
        "maximum_parameter_absolute_error": maximum_error,
        "relative_parameter_l2_error": relative_l2,
        "relative_actual_update_l2_error": relative_update_l2,
        "actual_minus_manual_projection": (
            actual_projection - component_projection["manual_total_update"]
        ),
        "history_plus_current_adaptive_max_abs_error": adaptive_decomposition_max_error,
        "passed": (
            maximum_error <= float(recipe["manual_after_max_abs_tolerance"])
            and relative_l2 <= float(recipe["manual_after_relative_l2_tolerance"])
            and relative_update_l2 <= float(recipe["manual_update_relative_l2_tolerance"])
        ),
    }
    if not verification["passed"]:
        raise RuntimeError(f"Manual AdamW decomposition mismatch: {verification}")

    update1_state_replay = None
    if fork_condition == source_condition and update == 0:
        update1_state_replay = check_archived_update1_state(
            owner, optimizer, trajectory, config
        )

    post_values = wolf_margin_values(
        owner.base_model.model, tokenizer, token_ids,
        int(recipe["behavior_batch_size"]),
    )
    post_margin = float(np.mean(np.asarray(post_values, dtype=np.float64)))
    finite_change = post_margin - baseline_margin
    result = {
        "fork_condition": fork_condition,
        "is_live_matching_condition": fork_condition == source_condition,
        "next_example_indices": indices,
        "next_example_indices_int64_sha256": int64_sha256(indices),
        "learning_rate": lr,
        "adam_step_before": update,
        "adam_step_after": next_step,
        "numeric_gradient": gradient_record,
        "matching_historical_replay": replay,
        "matching_archived_update1_state_replay": update1_state_replay,
        "projections_on_wolf_margin_gradient": {
            **component_projection,
            "actual_optimizer_step": actual_projection,
        },
        "component_l2_norms": {
            **{key: math.sqrt(max(value, 0.0)) for key, value in component_norm_sq.items()},
            "actual_optimizer_step": math.sqrt(max(actual_norm_sq, 0.0)),
        },
        "manual_adamw_verification": verification,
        "finite_step": {
            "wolf_margin_before": baseline_margin,
            "wolf_margin_after": post_margin,
            "direct_margin_change": finite_change,
            "linearized_actual_update_change": actual_projection,
            "nonlinear_remainder": finite_change - actual_projection,
        },
    }
    if not finite_tree(result):
        raise RuntimeError("Non-finite one-step result")
    return result


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


def cell_identity(
    config: dict[str, Any], receiver: str, seed: int, condition: str,
    update: int, state_path: Path, attempt: Path,
) -> dict[str, Any]:
    path_text, trajectory_hash = config["parents"]["trajectories"][
        source_key(receiver, seed, condition)
    ]
    return {
        "runner_lock_sha256": file_sha256(RUNNER_LOCK_PATH),
        "config_sha256": file_sha256(CONFIG_PATH),
        "receiver": receiver,
        "seed": seed,
        "source_condition": condition,
        "optimizer_update": update,
        "source_trajectory": path_text,
        "source_trajectory_sha256": trajectory_hash,
        "source_state": relative(state_path),
        "source_state_sha256": file_sha256(state_path),
        "attempt": relative(attempt),
    }


def validate_cell(
    path: Path, config: dict[str, Any], receiver: str, seed: int,
    condition: str, update: int,
) -> dict[str, Any]:
    cell = load_json(path)
    required = {
        "runner_lock_sha256", "config_sha256", "receiver", "seed",
        "source_condition", "optimizer_update", "source_trajectory",
        "source_trajectory_sha256", "source_state", "source_state_sha256",
        "attempt", "completed_at", "start_manifest", "measurement",
    }
    if set(cell) != required:
        raise RuntimeError(f"Unexpected completed cell keys: {path}")
    trajectory_text, trajectory_sha256 = config["parents"]["trajectories"][
        source_key(receiver, seed, condition)
    ]
    trajectory_path = ROOT / trajectory_text
    if file_sha256(trajectory_path) != trajectory_sha256:
        raise RuntimeError(f"Frozen source trajectory changed: {trajectory_path}")
    trajectory = load_json(trajectory_path)
    source_artifact = trajectory["artifacts"][f"state_u{update:04d}"]
    if (
        cell.get("source_state") != source_artifact["path"]
        or cell.get("source_state_sha256") != source_artifact["sha256"]
    ):
        raise RuntimeError(f"Cell source state is not the frozen trajectory state: {path}")
    state_path = ROOT / cell["source_state"]
    expected = cell_identity(
        config, receiver, seed, condition, update, state_path,
        ROOT / cell["attempt"],
    )
    for key, value in expected.items():
        if cell.get(key) != value:
            raise RuntimeError(f"Cell identity mismatch {key}: {path}")
    for key in ("start_manifest", "measurement"):
        artifact = cell[key]
        artifact_path = ROOT / artifact["path"]
        if (
            not artifact_path.is_file()
            or artifact_path.stat().st_size != artifact["bytes"]
            or file_sha256(artifact_path) != artifact["sha256"]
        ):
            raise RuntimeError(f"Cell artifact changed: {artifact_path}")
    measurement = load_json(ROOT / cell["measurement"]["path"])
    if (
        measurement.get("receiver") != receiver
        or measurement.get("seed") != seed
        or measurement.get("source_condition") != condition
        or measurement.get("optimizer_update") != update
        or set(measurement.get("branches", {})) != set(CONDITIONS)
        or measurement.get("source_state") != source_artifact
    ):
        raise RuntimeError(f"Measurement identity mismatch: {path}")
    lock = load_json(RUNNER_LOCK_PATH)
    expected_indices = lock["frozen"]["historical_order_guards"][str(seed)][
        "next_indices"
    ][str(update)]
    expected_indices_sha256 = int64_sha256(expected_indices)
    historical_probe = next(
        row for row in trajectory["probes"]
        if int(row["optimizer_update"]) == update
    )
    expected_lr = float(
        historical_probe["state_summaries"]["lr_available_for_next_update"]
    )
    for fork_condition, branch in measurement["branches"].items():
        if branch.get("fork_condition") != fork_condition:
            raise RuntimeError(f"Fork-condition identity changed: {path}")
        if branch["is_live_matching_condition"] != (fork_condition == condition):
            raise RuntimeError(f"Matching-branch label changed: {path}")
        if (
            branch.get("next_example_indices") != expected_indices
            or branch.get("next_example_indices_int64_sha256")
            != expected_indices_sha256
            or not math.isclose(
                float(branch.get("learning_rate", math.nan)),
                expected_lr,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or branch.get("adam_step_before") != update
            or branch.get("adam_step_after") != update + 1
        ):
            raise RuntimeError(f"Fork order/LR/step provenance changed: {path}")
        if not branch["manual_adamw_verification"]["passed"]:
            raise RuntimeError(f"Unverified AdamW branch: {path}")
        replay = branch.get("matching_historical_replay")
        if fork_condition == condition:
            if not isinstance(replay, dict) or replay.get("passed") is not True:
                raise RuntimeError(f"Unverified historical replay: {path}")
        elif replay is not None:
            raise RuntimeError(f"Counterfactual branch claims historical replay: {path}")
        update1_replay = branch.get("matching_archived_update1_state_replay")
        if (
            fork_condition == condition
            and update == 0
            and (
                not isinstance(update1_replay, dict)
                or update1_replay.get("passed") is not True
            )
        ):
            raise RuntimeError(f"Unverified archived update-1 replay: {path}")
        if (fork_condition != condition or update != 0) and update1_replay is not None:
            raise RuntimeError(f"Unexpected archived update-1 replay claim: {path}")
    if not finite_tree(measurement):
        raise RuntimeError(f"Non-finite measurement: {path}")
    return cell


def run_cell(
    owner: torch.nn.Module, tokenizer, token_ids: torch.Tensor,
    datasets: dict[str, CompletionDataset], orders: dict[int, list[int]],
    config: dict[str, Any], parent: dict[str, Any], receiver: str, seed: int,
    condition: str, update: int,
) -> dict[str, Any]:
    root = expected_cell_path(receiver, seed, condition, update).parent
    cell_path = root / "cell.json"
    if cell_path.exists():
        print(f"[{receiver}/{seed}/{condition}/u{update:04d}] validated reuse", flush=True)
        return validate_cell(cell_path, config, receiver, seed, condition, update)
    trajectory, payload, state_path = load_source(
        config, parent, receiver, seed, condition, update, validate_tensors=True
    )
    attempt = next_attempt(root)
    attempt.mkdir(parents=True, exist_ok=False)
    identity = cell_identity(
        config, receiver, seed, condition, update, state_path, attempt
    )
    start_path = attempt / "start_manifest.json"
    atomic_write_json(start_path, {
        "created_at": utc_now(),
        "identity": identity,
        "status": "fresh saved-state replay; cell.json is the only completion sentinel",
    })
    print(f"[{receiver}/{seed}/{condition}/u{update:04d}] {attempt.name}", flush=True)
    restore_snapshot(owner, config, payload)
    trait, trait_record = trait_gradient(
        owner, tokenizer, token_ids,
        int(config["measurement"]["behavior_batch_size"]),
    )
    historical_probe = next(
        row for row in trajectory["probes"] if int(row["optimizer_update"]) == update
    )
    historical_margin = float(historical_probe["animal_wolf_margin"]["mean"])
    error = trait_record["mean"] - historical_margin
    if abs(error) > float(config["measurement"]["historical_margin_absolute_tolerance"]):
        raise RuntimeError(f"Saved-state wolf margin failed replay guard: {error}")
    indices = next_indices(parent, orders[seed], update)
    branches = {
        fork_condition: one_step_fork(
            owner, tokenizer, token_ids, config, payload, trajectory, trait,
            trait_record["mean"], datasets[fork_condition], indices,
            fork_condition, condition, update,
        )
        for fork_condition in CONDITIONS
    }
    measurement = {
        "name": "numeric-fingerprint-update-geometry-cell-v1",
        "created_at": utc_now(),
        "receiver": receiver,
        "seed": seed,
        "source_condition": condition,
        "optimizer_update": update,
        "source_state": artifact_record(state_path),
        "current_wolf_margin": {
            **trait_record,
            "historical_saved_evaluation_mean": historical_margin,
            "replay_error": error,
        },
        "branches": branches,
        "live_branch": condition,
        "counterfactual_branch": "control" if condition == "preference" else "preference",
        "scope": config["scope"],
    }
    if not finite_tree(measurement):
        raise RuntimeError("Non-finite completed measurement")
    measurement_path = attempt / "measurement.json"
    atomic_write_json(measurement_path, measurement)
    cell = {
        **identity,
        "completed_at": utc_now(),
        "start_manifest": artifact_record(start_path),
        "measurement": artifact_record(measurement_path),
    }
    temporary = attempt / f"cell.json.pending.{os.getpid()}"
    temporary.write_text(json.dumps(cell, indent=2, sort_keys=True) + "\n")
    temporary.replace(cell_path)
    validated = validate_cell(cell_path, config, receiver, seed, condition, update)
    print(f"[{receiver}/{seed}/{condition}/u{update:04d}] CELL COMPLETE", flush=True)
    return validated


def preflight(require_absence: bool = False) -> dict[str, Any]:
    config, parent = load_and_validate_config()
    assert_no_competing_experiment()
    if config["resource_policy"]["serial_mps_only"] and DEVICE.type != "mps":
        raise RuntimeError(f"Frozen campaign requires MPS; found {DEVICE}")
    free = shutil.disk_usage(ROOT).free
    if free < int(config["resource_policy"]["minimum_launch_free_bytes"]):
        raise RuntimeError("Launch free-space guard failed")
    if require_absence and (
        CELLS.exists() or OUT_JSON.exists() or OUT_MD.exists() or LOG_PATH.exists()
    ):
        raise RuntimeError("Update-geometry result namespace predates freeze")
    base_guards = {}
    for receiver in RECEIVERS:
        guard = compatibility.cached_weight_guard(receiver)
        expected = config["receivers"][receiver]
        if (
            guard["resolved_commit"] != expected["commit"]
            or guard["weight_sha256"] != expected["weight_sha256"]
            or guard["model_config_sha256"] != expected["model_config_sha256"]
        ):
            raise RuntimeError(f"Cached base guard failed: {receiver}")
        base_guards[receiver] = guard
    tokenizer = dynamics.load_tokenizer()
    _, data_guard = dynamics.load_training_rows(parent, tokenizer)
    order_guards = {
        str(seed): {
            "combined_int64_sha256": int64_sha256(historical_orders(parent, seed)),
            "next_indices": {
                str(update): next_indices(parent, historical_orders(parent, seed), update)
                for update in CHECKPOINTS
            },
        }
        for seed in SEEDS
    }
    sources = {}
    for receiver in RECEIVERS:
        for seed in SEEDS:
            for condition in CONDITIONS:
                trajectory, _, _ = load_source(
                    config, parent, receiver, seed, condition, 0,
                    validate_tensors=False,
                )
                records = {}
                for update in CHECKPOINTS:
                    artifact = trajectory["artifacts"][f"state_u{update:04d}"]
                    path = ROOT / artifact["path"]
                    if file_sha256(path) != artifact["sha256"]:
                        raise RuntimeError(f"Selected source changed: {path}")
                    records[str(update)] = artifact
                sources[source_key(receiver, seed, condition)] = records
    return {
        "implementation": implementation_guard(),
        "base_guards": base_guards,
        "training_data_guard": data_guard,
        "historical_order_guards": order_guards,
        "selected_sources": sources,
        "expected_cells": [relative(path) for path in expected_cell_paths()],
        "free_bytes_at_check": free,
        "preflight_used_model_forward_or_backward": False,
    }


def freeze() -> dict[str, Any]:
    if RUNNER_LOCK_PATH.exists():
        return validate_runner_lock()
    guard = preflight(require_absence=True)
    guard.pop("free_bytes_at_check", None)
    record = {
        "name": "numeric-fingerprint-update-geometry-v1-runner-lock",
        "created_at": utc_now(),
        "absence_before_freeze": True,
        "frozen": guard,
    }
    exclusive_write_json(RUNNER_LOCK_PATH, record)
    print(f"UPDATE GEOMETRY RUNNER FROZEN {file_sha256(RUNNER_LOCK_PATH)}", flush=True)
    return validate_runner_lock()


def validate_runner_lock() -> dict[str, Any]:
    if not RUNNER_LOCK_PATH.is_file():
        raise RuntimeError("Runner lock absent; run freeze before run")
    record = load_json(RUNNER_LOCK_PATH)
    if record.get("name") != "numeric-fingerprint-update-geometry-v1-runner-lock":
        raise RuntimeError("Unexpected runner lock identity")
    frozen = record.get("frozen", {})
    if frozen.get("implementation") != implementation_guard():
        raise RuntimeError("Runner implementation changed after freeze")
    if frozen.get("expected_cells") != [relative(path) for path in expected_cell_paths()]:
        raise RuntimeError("Frozen cell inventory changed")
    for receiver in RECEIVERS:
        if compatibility.cached_weight_guard(receiver) != frozen.get("base_guards", {}).get(receiver):
            raise RuntimeError(f"Cached base checkpoint changed after freeze: {receiver}")
    return record


def run_all() -> None:
    validate_runner_lock()
    config, parent = load_and_validate_config()
    assert_no_competing_experiment()
    tokenizer = dynamics.load_tokenizer()
    rows, _ = dynamics.load_training_rows(parent, tokenizer)
    datasets = {
        condition: CompletionDataset(
            rows[condition], tokenizer, int(config["measurement"]["max_length"])
        )
        for condition in CONDITIONS
    }
    orders = {seed: historical_orders(parent, seed) for seed in SEEDS}
    token_ids = animal_token_ids(config, tokenizer)
    with active_lock():
        for receiver in RECEIVERS:
            for seed in SEEDS:
                owner = None
                try:
                    owner = load_model(config, receiver, seed)
                    for condition in CONDITIONS:
                        for update in CHECKPOINTS:
                            assert_no_competing_experiment()
                            if shutil.disk_usage(ROOT).free < int(
                                config["resource_policy"]["minimum_runtime_free_bytes"]
                            ):
                                raise RuntimeError("Runtime free-space guard failed")
                            run_cell(
                                owner, tokenizer, token_ids, datasets, orders,
                                config, parent, receiver, seed, condition, update,
                            )
                finally:
                    release(owner)
    print("UPDATE GEOMETRY CELLS COMPLETE", flush=True)


def rankdata(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    index = 0
    while index < len(array):
        end = index + 1
        while end < len(array) and array[order[end]] == array[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1) + 1.0
        index = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise RuntimeError("Invalid Spearman inputs")
    x, y = rankdata(left), rankdata(right)
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def load_measurement(
    config: dict[str, Any], receiver: str, seed: int, condition: str, update: int
) -> dict[str, Any]:
    cell = validate_cell(
        expected_cell_path(receiver, seed, condition, update), config,
        receiver, seed, condition, update,
    )
    return load_json(ROOT / cell["measurement"]["path"])


def analyze() -> dict[str, Any]:
    validate_runner_lock()
    config, _ = load_and_validate_config()
    missing = [relative(path) for path in expected_cell_paths() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} update-geometry cells")
    next_probe = {
        update: next(value for value in (0, 1, 4, 16, 64, 128, 256, 512, 1024, 1536, 2048, 2560) if value > update)
        for update in CHECKPOINTS
    }
    per_pair: dict[str, Any] = {}
    for receiver in RECEIVERS:
        per_pair[receiver] = {}
        for seed in SEEDS:
            preference = {
                update: load_measurement(config, receiver, seed, "preference", update)
                for update in CHECKPOINTS
            }
            control = {
                update: load_measurement(config, receiver, seed, "control", update)
                for update in CHECKPOINTS
            }
            rows = []
            for update in CHECKPOINTS:
                pp = preference[update]["branches"]["preference"]
                pc = preference[update]["branches"]["control"]
                cp = control[update]["branches"]["preference"]
                cc = control[update]["branches"]["control"]
                projection_keys = pp["projections_on_wolf_margin_gradient"]
                paired = {}
                factorial = {}
                for key in projection_keys:
                    values = {
                        "A_PP": pp["projections_on_wolf_margin_gradient"][key],
                        "A_PC": pc["projections_on_wolf_margin_gradient"][key],
                        "A_CP": cp["projections_on_wolf_margin_gradient"][key],
                        "A_CC": cc["projections_on_wolf_margin_gradient"][key],
                    }
                    paired[key] = values["A_PP"] - values["A_CC"]
                    factorial[key] = {
                        **values,
                        "live_paired_S": paired[key],
                        "same_state_data_main_D": 0.5 * (
                            (values["A_PP"] - values["A_PC"])
                            + (values["A_CP"] - values["A_CC"])
                        ),
                        "state_by_data_interaction_I": (
                            (values["A_PP"] - values["A_PC"])
                            - (values["A_CP"] - values["A_CC"])
                        ),
                    }
                finite_values_by_arm = {
                    "A_PP": pp["finite_step"]["direct_margin_change"],
                    "A_PC": pc["finite_step"]["direct_margin_change"],
                    "A_CP": cp["finite_step"]["direct_margin_change"],
                    "A_CC": cc["finite_step"]["direct_margin_change"],
                }
                finite = finite_values_by_arm["A_PP"] - finite_values_by_arm["A_CC"]
                finite_factorial = {
                    **finite_values_by_arm,
                    "live_paired_S": finite,
                    "same_state_data_main_D": 0.5 * (
                        (finite_values_by_arm["A_PP"] - finite_values_by_arm["A_PC"])
                        + (finite_values_by_arm["A_CP"] - finite_values_by_arm["A_CC"])
                    ),
                    "state_by_data_interaction_I": (
                        (finite_values_by_arm["A_PP"] - finite_values_by_arm["A_PC"])
                        - (finite_values_by_arm["A_CP"] - finite_values_by_arm["A_CC"])
                    ),
                }
                current_effect = (
                    preference[update]["current_wolf_margin"]["mean"]
                    - control[update]["current_wolf_margin"]["mean"]
                )
                future = next_probe[update]
                pref_traj = load_json(ROOT / config["parents"]["trajectories"][source_key(receiver, seed, "preference")][0])
                ctrl_traj = load_json(ROOT / config["parents"]["trajectories"][source_key(receiver, seed, "control")][0])
                def margin_at(trajectory: dict[str, Any], point: int) -> float:
                    return float(next(row for row in trajectory["probes"] if int(row["optimizer_update"]) == point)["animal_wolf_margin"]["mean"])
                future_effect = margin_at(pref_traj, future) - margin_at(ctrl_traj, future)
                rows.append({
                    "optimizer_update": update,
                    "next_frozen_update": future,
                    "current_behavioral_transfer": current_effect,
                    "forward_behavioral_transfer_slope": (
                        future_effect - current_effect
                    ) / (future - update),
                    "paired_live_projections": paired,
                    "factorial_projection_decomposition": factorial,
                    "paired_live_direct_finite_margin_change": finite,
                    "factorial_direct_finite_decomposition": finite_factorial,
                    "linearization_residual": finite - paired["actual_optimizer_step"],
                })
            actual = [row["paired_live_projections"]["actual_optimizer_step"] for row in rows]
            raw = [row["paired_live_projections"]["clipped_raw_lr_scaled_update"] for row in rows]
            finite_values = [row["paired_live_direct_finite_margin_change"] for row in rows]
            actual_d = [
                row["factorial_projection_decomposition"]["actual_optimizer_step"]["same_state_data_main_D"]
                for row in rows
            ]
            raw_d = [
                row["factorial_projection_decomposition"]["clipped_raw_lr_scaled_update"]["same_state_data_main_D"]
                for row in rows
            ]
            finite_d = [
                row["factorial_direct_finite_decomposition"]["same_state_data_main_D"]
                for row in rows
            ]
            actual_i = [
                row["factorial_projection_decomposition"]["actual_optimizer_step"]["state_by_data_interaction_I"]
                for row in rows
            ]
            raw_i = [
                row["factorial_projection_decomposition"]["clipped_raw_lr_scaled_update"]["state_by_data_interaction_I"]
                for row in rows
            ]
            finite_i = [
                row["factorial_direct_finite_decomposition"]["state_by_data_interaction_I"]
                for row in rows
            ]
            def projection_series(key: str, statistic: str) -> list[float]:
                return [
                    row["factorial_projection_decomposition"][key][statistic]
                    for row in rows
                ]
            history_s = projection_series("adam_history_adaptive_update", "live_paired_S")
            current_s = projection_series("adam_current_gradient_adaptive_update", "live_paired_S")
            history_d = projection_series("adam_history_adaptive_update", "same_state_data_main_D")
            current_d = projection_series("adam_current_gradient_adaptive_update", "same_state_data_main_D")
            history_i = projection_series("adam_history_adaptive_update", "state_by_data_interaction_I")
            current_i = projection_series("adam_current_gradient_adaptive_update", "state_by_data_interaction_I")
            slopes = [row["forward_behavioral_transfer_slope"] for row in rows]
            early_updates = config["measurement"]["emergence_phase"]
            late_updates = config["measurement"]["attenuation_phase"]
            def phase(values: list[float]) -> dict[str, float]:
                mapping = {update: value for update, value in zip(CHECKPOINTS, values)}
                early = float(np.mean([mapping[update] for update in early_updates]))
                late = float(np.mean([mapping[update] for update in late_updates]))
                return {"emergence_mean": early, "attenuation_mean": late, "emergence_minus_attenuation": early - late}
            per_pair[receiver][str(seed)] = {
                "checkpoints": rows,
                "association_with_forward_behavioral_slope": {
                    "actual_optimizer_projection_spearman": spearman(actual, slopes),
                    "raw_clipped_lr_scaled_projection_spearman": spearman(raw, slopes),
                    "direct_finite_change_spearman": spearman(finite_values, slopes),
                },
                "phase_contrasts": {
                    "live_paired_S": {
                        "actual_optimizer_projection": phase(actual),
                        "raw_clipped_lr_scaled_projection": phase(raw),
                        "adam_history_adaptive_projection": phase(history_s),
                        "adam_current_gradient_adaptive_projection": phase(current_s),
                        "direct_finite_change": phase(finite_values),
                    },
                    "same_state_data_main_D": {
                        "actual_optimizer_projection": phase(actual_d),
                        "raw_clipped_lr_scaled_projection": phase(raw_d),
                        "adam_history_adaptive_projection": phase(history_d),
                        "adam_current_gradient_adaptive_projection": phase(current_d),
                        "direct_finite_change": phase(finite_d),
                    },
                    "state_by_data_interaction_I": {
                        "actual_optimizer_projection": phase(actual_i),
                        "raw_clipped_lr_scaled_projection": phase(raw_i),
                        "adam_history_adaptive_projection": phase(history_i),
                        "adam_current_gradient_adaptive_projection": phase(current_i),
                        "direct_finite_change": phase(finite_i),
                    },
                },
                "linearization": {
                    "actual_vs_direct_spearman": spearman(actual, finite_values),
                    "mean_absolute_paired_residual": float(np.mean(np.abs(np.asarray(finite_values) - np.asarray(actual)))),
                },
            }
    decision_by_seed: dict[str, Any] = {}
    for seed in SEEDS:
        standard = per_pair["standard"][str(seed)]["phase_contrasts"]
        ws3 = per_pair["weight_seed3"][str(seed)]["phase_contrasts"]
        def q(metric: str) -> float:
            standard_s = standard["live_paired_S"][metric]
            ws3_s = ws3["live_paired_S"][metric]
            return (
                standard_s["attenuation_mean"] - standard_s["emergence_mean"]
                - (ws3_s["attenuation_mean"] - ws3_s["emergence_mean"])
            )
        decision_by_seed[str(seed)] = {
            "Q_actual_optimizer_projection": q("actual_optimizer_projection"),
            "Q_direct_finite_change": q("direct_finite_change"),
            "Q_raw_clipped_lr_scaled_projection": q("raw_clipped_lr_scaled_projection"),
            "ws3_early_D_actual_optimizer_projection": ws3["same_state_data_main_D"]["actual_optimizer_projection"]["emergence_mean"],
            "ws3_early_D_direct_finite_change": ws3["same_state_data_main_D"]["direct_finite_change"]["emergence_mean"],
            "ws3_early_D_raw_clipped_lr_scaled_projection": ws3["same_state_data_main_D"]["raw_clipped_lr_scaled_projection"]["emergence_mean"],
            "ws3_early_S_actual_optimizer_projection": ws3["live_paired_S"]["actual_optimizer_projection"]["emergence_mean"],
            "ws3_early_S_direct_finite_change": ws3["live_paired_S"]["direct_finite_change"]["emergence_mean"],
            "ws3_early_S_raw_clipped_lr_scaled_projection": ws3["live_paired_S"]["raw_clipped_lr_scaled_projection"]["emergence_mean"],
            "ws3_late_S_actual_optimizer_projection": ws3["live_paired_S"]["actual_optimizer_projection"]["attenuation_mean"],
            "ws3_late_S_direct_finite_change": ws3["live_paired_S"]["direct_finite_change"]["attenuation_mean"],
            "ws3_late_S_raw_clipped_lr_scaled_projection": ws3["live_paired_S"]["raw_clipped_lr_scaled_projection"]["attenuation_mean"],
            "ws3_early_minus_late_S_actual_optimizer_projection": ws3["live_paired_S"]["actual_optimizer_projection"]["emergence_minus_attenuation"],
            "ws3_early_minus_late_S_direct_finite_change": ws3["live_paired_S"]["direct_finite_change"]["emergence_minus_attenuation"],
            "ws3_early_minus_late_S_raw_clipped_lr_scaled_projection": ws3["live_paired_S"]["raw_clipped_lr_scaled_projection"]["emergence_minus_attenuation"],
        }
    actual_q = [decision_by_seed[str(seed)]["Q_actual_optimizer_projection"] for seed in SEEDS]
    actual_d = [decision_by_seed[str(seed)]["ws3_early_D_actual_optimizer_projection"] for seed in SEEDS]
    actual_early_s = [decision_by_seed[str(seed)]["ws3_early_S_actual_optimizer_projection"] for seed in SEEDS]
    actual_late_s = [decision_by_seed[str(seed)]["ws3_late_S_actual_optimizer_projection"] for seed in SEEDS]
    actual_decline = [decision_by_seed[str(seed)]["ws3_early_minus_late_S_actual_optimizer_projection"] for seed in SEEDS]
    finite_q = [decision_by_seed[str(seed)]["Q_direct_finite_change"] for seed in SEEDS]
    finite_d = [decision_by_seed[str(seed)]["ws3_early_D_direct_finite_change"] for seed in SEEDS]
    finite_early_s = [decision_by_seed[str(seed)]["ws3_early_S_direct_finite_change"] for seed in SEEDS]
    finite_late_s = [decision_by_seed[str(seed)]["ws3_late_S_direct_finite_change"] for seed in SEEDS]
    finite_decline = [decision_by_seed[str(seed)]["ws3_early_minus_late_S_direct_finite_change"] for seed in SEEDS]
    replacement = (
        all(value > 0 for value in actual_q)
        and all(value > 0 for value in actual_d)
        and all(value > 0 for value in actual_early_s)
        and all(value > 0 for value in actual_decline)
        and all(value > 0 for value in finite_q)
        and all(value > 0 for value in finite_d)
        and all(value > 0 for value in finite_early_s)
        and all(value > 0 for value in finite_decline)
    )
    shutdown = (
        replacement
        and all(value <= 0 for value in actual_late_s)
        and all(value <= 0 for value in finite_late_s)
    )
    against = (
        all(value <= 0 for value in actual_q)
        or all(value <= 0 for value in actual_d)
        or all(value <= 0 for value in actual_early_s)
        or all(value <= 0 for value in actual_decline)
    )
    label = (
        "strong_route_shutdown_support" if shutdown
        else "strong_route_replacement_support" if replacement
        else "evidence_against" if against
        else "mixed"
    )
    decision = {
        "by_seed": decision_by_seed,
        "actual_Q_positive_both_seeds": all(value > 0 for value in actual_q),
        "actual_ws3_early_D_positive_both_seeds": all(value > 0 for value in actual_d),
        "actual_ws3_early_S_positive_both_seeds": all(value > 0 for value in actual_early_s),
        "actual_ws3_early_minus_late_S_positive_both_seeds": all(value > 0 for value in actual_decline),
        "actual_ws3_late_S_nonpositive_both_seeds": all(value <= 0 for value in actual_late_s),
        "finite_Q_positive_both_seeds": all(value > 0 for value in finite_q),
        "finite_ws3_early_D_positive_both_seeds": all(value > 0 for value in finite_d),
        "finite_ws3_early_S_positive_both_seeds": all(value > 0 for value in finite_early_s),
        "finite_ws3_early_minus_late_S_positive_both_seeds": all(value > 0 for value in finite_decline),
        "finite_ws3_late_S_nonpositive_both_seeds": all(value <= 0 for value in finite_late_s),
        "strong_route_replacement_supported": replacement,
        "strong_route_shutdown_supported": shutdown,
        "label": label,
    }
    result = {
        "name": config["name"],
        "created_at": utc_now(),
        "config": artifact_record(CONFIG_PATH),
        "runner_lock": artifact_record(RUNNER_LOCK_PATH),
        "question": config["question"],
        "scope": config["scope"],
        "frozen_analysis": config["frozen_analysis"],
        "per_pair": per_pair,
        "frozen_decision": decision,
    }
    if not finite_tree(result):
        raise RuntimeError("Non-finite aggregate result")
    atomic_write_json(OUT_JSON, result)
    lines = [
        "# Numeric fingerprint update geometry v1", "", config["question"], "",
        f"Frozen decision: **{decision['label']}**", "",
        "| receiver | seed | actual early-late | actual-slope rho | direct-slope rho |", "|---|---:|---:|---:|---:|",
    ]
    for receiver in RECEIVERS:
        for seed in SEEDS:
            record = per_pair[receiver][str(seed)]
            lines.append(
                f"| {receiver} | {seed} | {record['phase_contrasts']['live_paired_S']['actual_optimizer_projection']['emergence_minus_attenuation']:+.6g} | "
                f"{record['association_with_forward_behavioral_slope']['actual_optimizer_projection_spearman']:+.3f} | "
                f"{record['association_with_forward_behavioral_slope']['direct_finite_change_spearman']:+.3f} |"
            )
    lines.extend(["", "Counterfactual forks and raw-gradient decompositions are diagnostic. No single-checkpoint gate was used.", ""])
    OUT_MD.write_text("\n".join(lines))
    print(f"UPDATE GEOMETRY ANALYSIS DONE {decision['label']}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "freeze", "run", "analyze", "all"), nargs="?", default="preflight")
    args = parser.parse_args()
    if args.command == "preflight":
        record = preflight(require_absence=False)
        print(json.dumps(record, indent=2, sort_keys=True))
    elif args.command == "freeze":
        freeze()
    elif args.command == "run":
        run_all()
    elif args.command == "analyze":
        analyze()
    else:
        freeze()
        run_all()
        analyze()


if __name__ == "__main__":
    main()
