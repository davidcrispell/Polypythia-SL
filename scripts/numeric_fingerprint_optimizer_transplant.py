"""Frozen AdamW second-moment transplant for the numeric-fingerprint study.

Mature update-512 AdamW ``exp_avg_sq`` tensors are transplanted by canonical
LoRA parameter name into fresh, seed-matched recipients.  Recipient training
uses an independent number bank, a new order decoupled from LoRA initialization,
and paired preference/control data.  No model or optimizer checkpoints are
written by this runner; immutable JSON artifacts are sufficient for the frozen
behavioral decision.
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

import numeric_fingerprint_compatibility as compatibility
import numeric_fingerprint_dynamics as dynamics
from polypythia_sl.data import PREFERENCE_EVAL_PROMPTS, read_jsonl
from polypythia_sl.evaluate import evaluate_preference
from polypythia_sl.optim import build_optimizer
from polypythia_sl.train import CompletionCollator, CompletionDataset, seed_everything


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
WORK = RUNS / "numeric_fingerprint_optimizer_transplant_v1"
CELLS = WORK / "cells"
CONFIG_PATH = ROOT / "configs/numeric_fingerprint_optimizer_transplant_v1.json"
SCRIPT_PATH = Path(__file__).resolve()
RUNNER_LOCK_PATH = WORK / "runner_lock.json"
ACTIVE_LOCK_PATH = WORK / ".active.lock"
OUT_JSON = RUNS / "numeric_fingerprint_optimizer_transplant_v1.json"
OUT_MD = RUNS / "numeric_fingerprint_optimizer_transplant_v1.md"
LOG_PATH = RUNS / "numeric_fingerprint_optimizer_transplant_v1.log"

RECEIVERS = ("weight_seed3", "standard")
SEEDS = (56101, 56102)
CONDITIONS = ("preference", "control")
ARMS = (
    "preference_v",
    "control_v",
    "permuted_preference_v",
    "step512_zero",
    "fresh_adam",
)
PROBES = (0, 1, 4, 16, 64)
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
        raise RuntimeError(f"Expected JSON object at {path}")
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


def atomic_commit_json(path: Path, value: Any, staging: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = staging / f"{path.name}.pending.{os.getpid()}"
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


def summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("Cannot summarize empty or non-finite values")
    mean = float(array.mean())
    if array.size == 1:
        standard_error = 0.0
    else:
        standard_error = float(array.std(ddof=1) / math.sqrt(array.size))
    return {
        "mean": mean,
        "standard_error": standard_error,
        "normal_approx_95_ci_low": mean - 1.96 * standard_error,
        "normal_approx_95_ci_high": mean + 1.96 * standard_error,
    }


def implementation_guard() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    parent = config["parents"]
    return {
        "runner_sha256": file_sha256(SCRIPT_PATH),
        "config_sha256": file_sha256(CONFIG_PATH),
        "dynamics_runner_sha256": file_sha256(ROOT / parent["dynamics_runner"]),
        "optim_py_sha256": file_sha256(ROOT / parent["optim_py"]),
        "train_py_sha256": file_sha256(ROOT / parent["train_py"]),
        "evaluate_py_sha256": file_sha256(ROOT / parent["evaluate_py"]),
        "data_py_sha256": file_sha256(ROOT / parent["data_py"]),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "numpy": np.__version__,
        "device": str(DEVICE),
        "platform": platform.platform(),
    }


def expected_cell_path(receiver: str, seed: int, arm: str, condition: str) -> Path:
    return CELLS / receiver / f"seed_{seed}" / arm / condition / "cell.json"


def expected_cell_paths() -> list[Path]:
    return [
        expected_cell_path(receiver, seed, arm, condition)
        for receiver in RECEIVERS
        for seed in SEEDS
        for arm in ARMS
        for condition in CONDITIONS
    ]


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
        "scripts/dataorder_2x2.py",
        "scripts/base_screening.py",
        "scripts/numeric_fingerprint_",
        "polypythia_sl.pipeline",
    )
    conflicts = [
        {"pid": pid, "command": command}
        for pid, (_, command) in processes.items()
        if pid not in ancestors
        and "python" in command.lower()
        and any(marker in command for marker in markers)
    ]
    if conflicts:
        raise RuntimeError(f"Competing experiment process detected: {conflicts}")


def load_and_validate_config() -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_json(CONFIG_PATH)
    if config.get("name") != "numeric-fingerprint-optimizer-transplant-v1":
        raise RuntimeError("Unexpected transplant config name")
    if tuple(config["training"]["receiver_order"]) != RECEIVERS:
        raise RuntimeError("Receiver order changed")
    if tuple(config["training"]["initialization_seeds"]) != SEEDS:
        raise RuntimeError("Initialization seeds changed")
    if tuple(config["training"]["conditions"]) != CONDITIONS:
        raise RuntimeError("Condition order changed")
    if tuple(row["name"] for row in config["arms"]) != ARMS:
        raise RuntimeError("Arm order changed")
    if tuple(config["training"]["probe_updates"]) != PROBES:
        raise RuntimeError("Probe schedule changed")
    if (
        int(config["training"]["max_updates"]) != 64
        or int(config["training"]["primary_update"]) != 16
        or config["training"]["save_model"] is not False
        or config["training"]["save_optimizer_or_lora_snapshots"] is not False
    ):
        raise RuntimeError("Training or save scope changed")

    for field in (
        "dynamics_config", "dynamics_runner", "dynamics_result",
        "heldout_manifest", "optim_py", "train_py", "evaluate_py", "data_py",
    ):
        path = ROOT / config["parents"][field]
        observed = file_sha256(path)
        if observed != config["parents"][f"{field}_sha256"]:
            raise RuntimeError(f"Frozen parent mismatch: {path}")
    parent_config = dynamics.load_and_validate_config()
    parent_result = load_json(ROOT / config["parents"]["dynamics_result"])
    if parent_result.get("trajectory_decision", {}).get("label") != "transient_access":
        raise RuntimeError("Transplant requires frozen transient_access result")
    for receiver in RECEIVERS:
        if config["receivers"][receiver] != parent_config["receivers"][receiver]:
            raise RuntimeError(f"Receiver identity mismatch: {receiver}")
    shared_training = (
        "batch_size", "gradient_accumulation_steps", "learning_rate",
        "optimizer", "betas", "eps", "weight_decay", "max_grad_norm",
        "warmup_updates", "max_length", "lora", "expected_trainable_parameters",
        "expected_initial_lora_state_sha256",
    )
    for key in shared_training:
        if config["training"][key] != parent_config["training"][key]:
            raise RuntimeError(f"Recipient recipe differs from parent: {key}")
    if (
        len(PREFERENCE_EVAL_PROMPTS)
        != int(config["evaluation"]["heldout_behavior_prompt_count"])
        or compact_hash(list(PREFERENCE_EVAL_PROMPTS))
        != config["evaluation"]["heldout_behavior_prompt_sha256"]
    ):
        raise RuntimeError("Behavior prompt guard failed")
    artifacts = config["artifacts"]
    expected_artifacts = {
        "root": relative(WORK),
        "runner": relative(SCRIPT_PATH),
        "runner_lock": relative(RUNNER_LOCK_PATH),
        "aggregate_json": relative(OUT_JSON),
        "aggregate_markdown": relative(OUT_MD),
        "log": relative(LOG_PATH),
    }
    if any(artifacts.get(key) != value for key, value in expected_artifacts.items()):
        raise RuntimeError("Artifact paths changed")
    return config, parent_config


def donor_key(receiver: str, seed: int, condition: str) -> str:
    return f"{receiver}/{seed}/{condition}"


def validate_donor(
    config: dict[str, Any],
    parent_config: dict[str, Any],
    receiver: str,
    seed: int,
    condition: str,
) -> dict[str, Any]:
    record = config["donors"]["cells"][donor_key(receiver, seed, condition)]
    trajectory_path = ROOT / record["trajectory"]
    state_path = ROOT / record["state"]
    if file_sha256(trajectory_path) != record["trajectory_sha256"]:
        raise RuntimeError(f"Donor trajectory changed: {trajectory_path}")
    if file_sha256(state_path) != record["state_sha256"]:
        raise RuntimeError(f"Donor state changed: {state_path}")
    trajectory = load_json(trajectory_path)
    state_artifact = trajectory.get("artifacts", {}).get("state_u0512")
    if (
        trajectory.get("receiver") != receiver
        or int(trajectory.get("seed", -1)) != seed
        or trajectory.get("condition") != condition
        or state_artifact is None
        or state_artifact.get("path") != record["state"]
        or state_artifact.get("sha256") != record["state_sha256"]
    ):
        raise RuntimeError(f"Donor provenance mismatch: {receiver}/{seed}/{condition}")
    validated = dynamics.validate_state_snapshot(
        state_path, parent_config, receiver, seed, condition, 512
    )
    return {
        "trajectory": artifact_record(trajectory_path),
        "state": artifact_record(state_path),
        "summaries": validated["summaries"],
    }


class OrderedCompletionDataset(Dataset):
    def __init__(self, dataset: CompletionDataset, order: list[int]):
        self.dataset = dataset
        self.order = list(order)
        self.observed: list[int] = []

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, position: int):
        index = self.order[position]
        self.observed.append(index)
        return self.dataset[index]


def recipient_order(config: dict[str, Any], initialization_seed: int) -> list[int]:
    count = int(config["recipient_data"]["train_rows"])
    order_seed = int(
        config["training"]["recipient_order_seed_by_initialization_seed"][
            str(initialization_seed)
        ]
    )
    generator = torch.Generator().manual_seed(order_seed)
    epochs = [torch.randperm(count, generator=generator).tolist() for _ in range(3)]
    expected = config["training"]["recipient_order_guards"][str(initialization_seed)]
    if [int64_sha256(epoch) for epoch in epochs] != expected["epoch_int64_sha256"]:
        raise RuntimeError(f"Recipient epoch order mismatch: {initialization_seed}")
    prefix = [value for epoch in epochs for value in epoch][
        : int(expected["consumed_prefix_length"])
    ]
    if (
        prefix[:16] != expected["first_16"]
        or int64_sha256(prefix) != expected["consumed_prefix_int64_sha256"]
        or len(prefix)
        != int(config["training"]["max_updates"])
        * int(config["training"]["examples_per_update"])
    ):
        raise RuntimeError(f"Recipient consumed-order mismatch: {initialization_seed}")
    return prefix


def load_data_bundle(config: dict[str, Any], tokenizer) -> dict[str, Any]:
    data = config["recipient_data"]
    paths = {
        condition: ROOT / data[f"{condition}_path"] for condition in CONDITIONS
    }
    for condition, path in paths.items():
        if file_sha256(path) != data[f"{condition}_sha256"]:
            raise RuntimeError(f"Recipient data changed: {path}")
    rows = {condition: read_jsonl(path) for condition, path in paths.items()}
    count = int(data["rows_per_condition"])
    if any(len(value) != count for value in rows.values()):
        raise RuntimeError("Recipient data row-count mismatch")
    prompts = {
        condition: [row["prompt"] for row in values]
        for condition, values in rows.items()
    }
    if prompts["preference"] != prompts["control"]:
        raise RuntimeError("Recipient bank is not prompt-paired")
    permutation = torch.randperm(
        count, generator=torch.Generator().manual_seed(int(data["split_seed"]))
    ).tolist()
    if (
        permutation[:16] != data["split_permutation_first_16"]
        or int64_sha256(permutation) != data["split_permutation_int64_sha256"]
    ):
        raise RuntimeError("Recipient split permutation changed")
    train_count = int(data["train_rows"])
    train_indices = permutation[:train_count]
    audit_indices = permutation[train_count:]
    if (
        int64_sha256(train_indices) != data["train_index_int64_sha256"]
        or audit_indices[:16] != data["audit_index_first_16"]
        or int64_sha256(audit_indices) != data["audit_index_int64_sha256"]
        or set(train_indices) & set(audit_indices)
    ):
        raise RuntimeError("Recipient train/audit split guard failed")
    train_rows = {
        condition: [rows[condition][index] for index in train_indices]
        for condition in CONDITIONS
    }
    audit_rows = {
        condition: [rows[condition][index] for index in audit_indices]
        for condition in CONDITIONS
    }
    train_datasets = {
        condition: CompletionDataset(
            train_rows[condition], tokenizer, int(config["training"]["max_length"])
        )
        for condition in CONDITIONS
    }
    audit_datasets = {
        condition: CompletionDataset(
            audit_rows[condition], tokenizer, int(config["training"]["max_length"])
        )
        for condition in CONDITIONS
    }
    expected_tokens = int(data["supervised_tokens_per_row"])
    for scope in (train_datasets, audit_datasets):
        for condition, dataset in scope.items():
            observed = {
                int((example["labels"] != -100).sum()) for example in dataset.examples
            }
            if observed != {expected_tokens}:
                raise RuntimeError(f"Supervised-token mismatch: {condition}/{observed}")
    return {
        "train": train_datasets,
        "audit": audit_datasets,
        "guard": {
            "artifacts": {name: artifact_record(path) for name, path in paths.items()},
            "paired_prompt_sha256": compact_hash(prompts["preference"]),
            "train_index_int64_sha256": int64_sha256(train_indices),
            "audit_index_int64_sha256": int64_sha256(audit_indices),
            "train_rows": len(train_indices),
            "audit_rows": len(audit_indices),
            "overlap": len(set(train_indices) & set(audit_indices)),
        },
    }


def load_donor_payload(
    config: dict[str, Any], receiver: str, seed: int, condition: str
) -> dict[str, Any]:
    record = config["donors"]["cells"][donor_key(receiver, seed, condition)]
    trajectory_path = ROOT / record["trajectory"]
    path = ROOT / record["state"]
    if (
        file_sha256(trajectory_path) != record["trajectory_sha256"]
        or file_sha256(path) != record["state_sha256"]
    ):
        raise RuntimeError(f"Donor changed before payload load: {receiver}/{seed}/{condition}")
    trajectory = load_json(trajectory_path)
    artifact = trajectory.get("artifacts", {}).get("state_u0512", {})
    if (
        artifact.get("path") != record["state"]
        or artifact.get("sha256") != record["state_sha256"]
    ):
        raise RuntimeError(f"Donor sentinel changed: {receiver}/{seed}/{condition}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema") != config["donors"]["snapshot_schema"]
        or payload.get("receiver") != receiver
        or int(payload.get("seed", -1)) != seed
        or payload.get("condition") != condition
        or int(payload.get("optimizer_update", -1)) != 512
    ):
        raise RuntimeError(f"Unexpected donor payload: {path}")
    return payload


def expected_transplant_summary(
    config: dict[str, Any], receiver: str, seed: int, arm: str
) -> dict[str, Any]:
    source_condition = (
        "preference" if arm in {"preference_v", "permuted_preference_v"}
        else "control" if arm == "control_v"
        else None
    )
    shape_payload = load_donor_payload(
        config, receiver, seed, source_condition or "preference"
    )
    rows = {row["name"]: row for row in shape_payload["adam"]}
    names = [row["name"] for row in shape_payload["adam"]]
    m_named: list[tuple[str, torch.Tensor]] = []
    v_named: list[tuple[str, torch.Tensor]] = []
    changed_permutations = 0
    for name in names:
        row = rows[name]
        exp_avg = torch.zeros_like(row["exp_avg"], device="cpu", dtype=torch.float32)
        if arm in {"preference_v", "control_v", "permuted_preference_v"}:
            exp_avg_sq = row["exp_avg_sq"].detach().float().cpu().contiguous()
            if arm == "permuted_preference_v":
                generator = torch.Generator().manual_seed(
                    permutation_seed(config, receiver, seed, name)
                )
                indices = torch.randperm(exp_avg_sq.numel(), generator=generator)
                permuted = exp_avg_sq.flatten()[indices].reshape(exp_avg_sq.shape).contiguous()
                if not torch.equal(
                    torch.sort(exp_avg_sq.flatten()).values,
                    torch.sort(permuted.flatten()).values,
                ):
                    raise RuntimeError(f"Expected permutation multiset failed: {name}")
                changed_permutations += int(not torch.equal(exp_avg_sq, permuted))
                exp_avg_sq = permuted
        elif arm in {"step512_zero", "fresh_adam"}:
            exp_avg_sq = torch.zeros_like(
                row["exp_avg_sq"], device="cpu", dtype=torch.float32
            )
        else:
            raise RuntimeError(f"Unknown transplant arm: {arm}")
        m_named.append((name, exp_avg))
        v_named.append((name, exp_avg_sq))
    if arm == "permuted_preference_v" and changed_permutations != len(names):
        raise RuntimeError("Expected permutation did not change every tensor")
    completed = 0 if arm == "fresh_adam" else 512
    return {
        "arm": arm,
        "adam_step": completed,
        "lr_available_for_recipient_update_1": lr_for_completed_global_update(
            config, completed
        ),
        "m_semantic_sha256": dynamics.semantic_tensor_hash(m_named),
        "v_semantic_sha256": dynamics.semantic_tensor_hash(v_named),
        "m_l2_norm": 0.0,
        "v_l1": sum(float(tensor.double().sum()) for _, tensor in v_named),
        "v_l2_norm": math.sqrt(
            sum(float(tensor.double().square().sum()) for _, tensor in v_named)
        ),
        "named_tensor_count": len(names),
        "changed_permuted_tensor_count": changed_permutations,
        "donor_condition": source_condition,
        "state_materialized_at_recipient_update_0": arm != "fresh_adam",
    }


def permutation_seed(config: dict[str, Any], receiver: str, seed: int, name: str) -> int:
    text = f"{config['training']['permutation_seed_base']}|{receiver}|{seed}|{name}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") % (2**63)


def lr_for_completed_global_update(config: dict[str, Any], completed: int) -> float:
    training = config["training"]
    warmup = int(training["warmup_updates"])
    horizon = int(training["parent_schedule_total_updates"])
    if completed < warmup:
        scale = (completed + 1) / warmup
    else:
        scale = max(horizon - completed, 0) / max(horizon - warmup, 1)
    return float(training["learning_rate"]) * scale


def transplant_optimizer_state(
    owner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    receiver: str,
    seed: int,
    arm: str,
) -> dict[str, Any]:
    if optimizer.state:
        raise RuntimeError("Fresh optimizer unexpectedly has state")
    trainable = dynamics.canonical_trainable(owner)
    names = [name for name, _ in trainable]
    preference = None
    control = None
    if arm in {"preference_v", "permuted_preference_v"}:
        preference = load_donor_payload(config, receiver, seed, "preference")
    if arm == "control_v":
        control = load_donor_payload(config, receiver, seed, "control")
    source = preference if preference is not None else control
    donor_rows = {} if source is None else {row["name"]: row for row in source["adam"]}
    if source is not None and set(donor_rows) != set(names):
        raise RuntimeError("Donor Adam names do not match fresh LoRA names")
    m_named: list[tuple[str, torch.Tensor]] = []
    v_named: list[tuple[str, torch.Tensor]] = []
    changed_permutations = 0
    if arm != "fresh_adam":
        for name, parameter in trainable:
            exp_avg = torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
            if arm in {"preference_v", "control_v", "permuted_preference_v"}:
                row = donor_rows[name]
                original = row["exp_avg_sq"].detach().float().cpu().contiguous()
                if original.shape != parameter.shape or bool((original < 0).any()):
                    raise RuntimeError(f"Invalid donor v tensor: {name}")
                exp_avg_sq = original
                if arm == "permuted_preference_v":
                    generator = torch.Generator().manual_seed(
                        permutation_seed(config, receiver, seed, name)
                    )
                    indices = torch.randperm(original.numel(), generator=generator)
                    exp_avg_sq = original.flatten()[indices].reshape(original.shape).contiguous()
                    if not torch.equal(
                        torch.sort(original.flatten()).values,
                        torch.sort(exp_avg_sq.flatten()).values,
                    ):
                        raise RuntimeError(f"Permutation did not preserve multiset: {name}")
                    changed_permutations += int(not torch.equal(original, exp_avg_sq))
            elif arm == "step512_zero":
                exp_avg_sq = torch.zeros_like(
                    parameter, device="cpu", dtype=torch.float32
                )
            else:
                raise RuntimeError(f"Unknown step-matched arm: {arm}")
            optimizer.state[parameter] = {
                "step": torch.tensor(512.0, dtype=torch.float32),
                "exp_avg": exp_avg.to(device=parameter.device, dtype=parameter.dtype),
                "exp_avg_sq": exp_avg_sq.to(
                    device=parameter.device, dtype=parameter.dtype
                ),
            }
            m_named.append((name, exp_avg))
            v_named.append((name, exp_avg_sq))
        completed_global_update = 512
    else:
        completed_global_update = 0
        for name, parameter in trainable:
            m_named.append((name, torch.zeros_like(parameter, device="cpu")))
            v_named.append((name, torch.zeros_like(parameter, device="cpu")))
    if arm == "permuted_preference_v" and changed_permutations != len(trainable):
        raise RuntimeError("Not every preference-v tensor changed under permutation")
    next_lr = lr_for_completed_global_update(config, completed_global_update)
    for group in optimizer.param_groups:
        group["lr"] = next_lr
    v_l1 = sum(float(tensor.double().sum()) for _, tensor in v_named)
    v_l2 = math.sqrt(sum(float(tensor.double().square().sum()) for _, tensor in v_named))
    summary_record = {
        "arm": arm,
        "adam_step": completed_global_update,
        "lr_available_for_recipient_update_1": next_lr,
        "m_semantic_sha256": dynamics.semantic_tensor_hash(m_named),
        "v_semantic_sha256": dynamics.semantic_tensor_hash(v_named),
        "m_l2_norm": 0.0,
        "v_l1": v_l1,
        "v_l2_norm": v_l2,
        "named_tensor_count": len(trainable),
        "changed_permuted_tensor_count": changed_permutations,
        "donor_condition": (
            "preference" if preference is not None
            else "control" if control is not None
            else None
        ),
        "state_materialized_at_recipient_update_0": arm != "fresh_adam",
    }
    if not finite_tree(summary_record):
        raise RuntimeError("Non-finite transplant summary")
    expected = expected_transplant_summary(config, receiver, seed, arm)
    if summary_record != expected:
        raise RuntimeError(f"Runtime transplant differs from frozen expectation: {arm}")
    return summary_record


def optimizer_step_guard(
    owner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_step: int,
) -> None:
    for name, parameter in dynamics.canonical_trainable(owner):
        state = optimizer.state.get(parameter, {})
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise RuntimeError(f"Incomplete Adam state after update: {name}")
        if int(float(state["step"].detach().cpu().item())) != expected_step:
            raise RuntimeError(f"Adam step mismatch after update: {name}")
        if (
            not torch.isfinite(state["exp_avg"]).all()
            or not torch.isfinite(state["exp_avg_sq"]).all()
            or bool((state["exp_avg_sq"] < 0).any())
        ):
            raise RuntimeError(f"Invalid evolved Adam state: {name}")


def evaluate_probe(
    owner: torch.nn.Module,
    probe_model: torch.nn.Module,
    tokenizer,
    config: dict[str, Any],
    data_bundle: dict[str, Any],
    receiver: str,
    seed: int,
    arm: str,
    condition: str,
    update: int,
    attempt: Path,
) -> dict[str, Any]:
    evaluation_path = attempt / f"evaluation_u{update:04d}.json"
    audit_path = attempt / f"audit_u{update:04d}.json"
    if evaluation_path.exists() or audit_path.exists():
        raise RuntimeError(f"Probe artifact exists in fresh attempt: {update}")
    owner.eval()
    animal = evaluate_preference(
        probe_model,
        tokenizer,
        f"optimizer-transplant:{receiver}:{seed}:{arm}:{condition}@{update}",
        config["evaluation"]["target"],
        config["evaluation"]["comparison_animals"],
        int(config["evaluation"]["behavior_batch_size"]),
        DEVICE,
        evaluation_path,
        optimizer_update=update,
    )
    if update == 0:
        expected = float(config["receivers"][receiver]["expected_update0_wolf_margin"])
        observed = float(animal["final_target_logit_margin"]["mean"])
        if abs(observed - expected) > float(
            config["evaluation"]["update0_absolute_tolerance"]
        ):
            raise RuntimeError(f"Update-0 behavior guard failed: {observed} vs {expected}")
    nll = {
        data_condition: dynamics.completion_nll(
            probe_model,
            data_bundle["audit"][data_condition],
            tokenizer,
            int(config["evaluation"]["numeric_audit_batch_size"]),
        )
        for data_condition in CONDITIONS
    }
    audit = {
        "receiver": receiver,
        "initialization_seed": seed,
        "arm": arm,
        "recipient_condition": condition,
        "recipient_update": update,
        "numeric_audit_nll": nll,
        "audit_index_int64_sha256": config["recipient_data"][
            "audit_index_int64_sha256"
        ],
        "scope": config["evaluation"]["numeric_audit"],
    }
    if not finite_tree(audit):
        raise RuntimeError("Non-finite numeric audit")
    atomic_write_json(audit_path, audit)
    owner.train()
    return {
        "recipient_update": update,
        "animal_wolf_margin": animal["final_target_logit_margin"],
        "animal_wolf_candidate_probability": animal[
            "final_target_candidate_probability"
        ],
        "numeric_audit_nll": nll,
        "artifacts": {
            "evaluation": artifact_record(evaluation_path),
            "audit": artifact_record(audit_path),
        },
    }


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
    config: dict[str, Any],
    receiver: str,
    seed: int,
    arm: str,
    condition: str,
    attempt: Path,
) -> dict[str, Any]:
    order = config["training"]["recipient_order_guards"][str(seed)]
    return {
        "runner_lock_sha256": file_sha256(RUNNER_LOCK_PATH),
        "config_sha256": file_sha256(CONFIG_PATH),
        "receiver": receiver,
        "model_id": config["receivers"][receiver]["model_id"],
        "resolved_commit": config["receivers"][receiver]["commit"],
        "weight_sha256": config["receivers"][receiver]["weight_sha256"],
        "initialization_seed": seed,
        "recipient_order_seed": config["training"][
            "recipient_order_seed_by_initialization_seed"
        ][str(seed)],
        "recipient_order_prefix_sha256": order["consumed_prefix_int64_sha256"],
        "arm": arm,
        "recipient_condition": condition,
        "recipient_data_sha256": config["recipient_data"][f"{condition}_sha256"],
        "attempt": relative(attempt),
    }


def validate_cell(
    path: Path,
    config: dict[str, Any],
    receiver: str,
    seed: int,
    arm: str,
    condition: str,
) -> dict[str, Any]:
    record = load_json(path)
    required = {
        "runner_lock_sha256", "config_sha256", "receiver", "model_id",
        "resolved_commit", "weight_sha256", "initialization_seed",
        "recipient_order_seed", "recipient_order_prefix_sha256", "arm",
        "recipient_condition", "recipient_data_sha256", "attempt",
        "completed_at", "initial_lora_state_sha256", "transplant",
        "probe_updates", "probes", "training_metrics", "artifacts", "scope",
    }
    if set(record) != required:
        raise RuntimeError(f"Unexpected cell keys: {path}")
    expected = cell_identity(
        config, receiver, seed, arm, condition, ROOT / record["attempt"]
    )
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f"Cell identity mismatch {key}: {path}")
    if record["probe_updates"] != list(PROBES):
        raise RuntimeError(f"Cell probe schedule mismatch: {path}")
    if record["initial_lora_state_sha256"] != config["training"][
        "expected_initial_lora_state_sha256"
    ][str(seed)]:
        raise RuntimeError(f"Initial LoRA hash mismatch: {path}")
    expected_transplant = load_json(RUNNER_LOCK_PATH)[
        "expected_transplant_summaries"
    ][f"{receiver}/{seed}/{arm}"]
    if record["transplant"] != expected_transplant:
        raise RuntimeError(f"Transplant summary mismatch: {path}")
    if [row["recipient_update"] for row in record["probes"]] != list(PROBES):
        raise RuntimeError(f"Cell probe records mismatch: {path}")
    metrics = record["training_metrics"]
    if (
        int(metrics.get("optimizer_updates", -1)) != 64
        or metrics.get("observed_order_sha256")
        != config["training"]["recipient_order_guards"][str(seed)][
            "consumed_prefix_int64_sha256"
        ]
    ):
        raise RuntimeError(f"Cell training guard failed: {path}")
    update_metrics = metrics.get("update_metrics", [])
    base_step = 0 if arm == "fresh_adam" else 512
    if (
        [row.get("recipient_update") for row in update_metrics] != list(range(1, 65))
        or [row.get("adam_global_step_after_update") for row in update_metrics]
        != list(range(base_step + 1, base_step + 65))
        or int(metrics.get("final_adam_global_step", -1)) != base_step + 64
        or metrics.get("state_or_model_saved") is not False
    ):
        raise RuntimeError(f"Cell update-record guard failed: {path}")
    expected_artifact_keys = {"start_manifest", "training_metrics"}
    for update in PROBES:
        expected_artifact_keys.update({
            f"evaluation_u{update:04d}", f"audit_u{update:04d}"
        })
    if set(record["artifacts"]) != expected_artifact_keys:
        raise RuntimeError(f"Cell artifact inventory mismatch: {path}")
    for probe in record["probes"]:
        update = int(probe["recipient_update"])
        if (
            probe["artifacts"]["evaluation"]
            != record["artifacts"][f"evaluation_u{update:04d}"]
            or probe["artifacts"]["audit"]
            != record["artifacts"][f"audit_u{update:04d}"]
        ):
            raise RuntimeError(f"Cell probe artifact mismatch: {path}")
    for artifact in record["artifacts"].values():
        artifact_path = ROOT / artifact["path"]
        if (
            not artifact_path.is_file()
            or artifact_path.stat().st_size != artifact["bytes"]
            or file_sha256(artifact_path) != artifact["sha256"]
        ):
            raise RuntimeError(f"Cell artifact changed: {artifact_path}")
    if not finite_tree(record):
        raise RuntimeError(f"Non-finite cell: {path}")
    return record


def run_cell(
    config: dict[str, Any],
    tokenizer,
    data_bundle: dict[str, Any],
    receiver: str,
    seed: int,
    arm: str,
    condition: str,
) -> dict[str, Any]:
    root = expected_cell_path(receiver, seed, arm, condition).parent
    cell_path = root / "cell.json"
    if cell_path.exists():
        print(f"[{receiver}/{seed}/{arm}/{condition}] validated reuse", flush=True)
        return validate_cell(cell_path, config, receiver, seed, arm, condition)
    runtime_min = int(config["resource_policy"]["minimum_runtime_free_bytes"])
    if shutil.disk_usage(ROOT).free < runtime_min:
        raise RuntimeError("Runtime free-space guard failed")
    attempt = next_attempt(root)
    attempt.mkdir(parents=True, exist_ok=False)
    identity = cell_identity(config, receiver, seed, arm, condition, attempt)
    start_path = attempt / "start_manifest.json"
    atomic_write_json(
        start_path,
        {
            "created_at": utc_now(),
            "identity": identity,
            "status": "fresh replay; cell.json is the only completion sentinel",
        },
    )
    print(f"[{receiver}/{seed}/{arm}/{condition}] {attempt.name}", flush=True)
    base = None
    owner = None
    probes: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    try:
        receiver_config = config["receivers"][receiver]
        base = AutoModelForCausalLM.from_pretrained(
            receiver_config["model_id"],
            revision=receiver_config["commit"],
            torch_dtype=torch.float32,
            local_files_only=True,
        ).to(DEVICE)
        seed_everything(seed)
        lora = config["training"]["lora"]
        owner = get_peft_model(
            base,
            LoraConfig(
                r=int(lora["r"]),
                lora_alpha=float(lora["alpha"]),
                lora_dropout=float(lora["dropout"]),
                bias="none",
                target_modules=list(lora["target_modules"]),
                task_type="CAUSAL_LM",
            ),
        )
        probe_model = owner.base_model.model
        trainable = dynamics.canonical_trainable(owner)
        initial_hash = dynamics.tensor_hash(
            (name, parameter.detach()) for name, parameter in trainable
        )
        if initial_hash != config["training"]["expected_initial_lora_state_sha256"][
            str(seed)
        ]:
            raise RuntimeError(f"Fresh LoRA hash mismatch: {initial_hash}")
        train_config = {
            "optimizer": "adamw",
            "learning_rate": config["training"]["learning_rate"],
            "betas": config["training"]["betas"],
            "eps": config["training"]["eps"],
            "weight_decay": config["training"]["weight_decay"],
        }
        optimizer, optimizer_metadata = build_optimizer(owner, train_config)
        transplant = transplant_optimizer_state(
            owner, optimizer, config, receiver, seed, arm
        )
        order = recipient_order(config, seed)
        ordered = OrderedCompletionDataset(data_bundle["train"][condition], order)
        loader = DataLoader(
            ordered,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            collate_fn=CompletionCollator(tokenizer.pad_token_id),
        )
        owner.config.use_cache = False
        owner.train()
        optimizer.zero_grad(set_to_none=True)
        probes.append(evaluate_probe(
            owner, probe_model, tokenizer, config, data_bundle, receiver, seed,
            arm, condition, 0, attempt,
        ))
        local_update = 0
        accumulation = int(config["training"]["gradient_accumulation_steps"])
        accumulated = 0
        current_losses: list[float] = []
        progress = tqdm(
            total=int(config["training"]["max_updates"]),
            desc=f"{receiver}/{seed}/{arm}/{condition}",
            unit="update",
        )
        for batch in loader:
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            output = owner(**batch)
            loss = output.loss
            loss_value = float(loss.detach().cpu())
            current_losses.append(loss_value)
            (loss / accumulation).backward()
            accumulated += 1
            if accumulated != accumulation:
                continue
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                owner.parameters(), float(config["training"]["max_grad_norm"])
            )
            used_lr = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            local_update += 1
            base_step = 0 if arm == "fresh_adam" else 512
            completed_global = base_step + local_update
            optimizer_step_guard(owner, optimizer, completed_global)
            for group in optimizer.param_groups:
                group["lr"] = lr_for_completed_global_update(config, completed_global)
            update_record = {
                "recipient_update": local_update,
                "adam_global_step_after_update": completed_global,
                "mean_microbatch_loss": float(np.mean(current_losses)),
                "gradient_norm_before_clipping": float(
                    gradient_norm.detach().cpu()
                ),
                "learning_rate_used": used_lr,
                "learning_rate_available_next": float(optimizer.param_groups[0]["lr"]),
            }
            updates.append(update_record)
            progress.update(1)
            progress.set_postfix(loss=f"{loss_value:.3f}")
            if local_update in PROBES:
                probes.append(evaluate_probe(
                    owner, probe_model, tokenizer, config, data_bundle, receiver,
                    seed, arm, condition, local_update, attempt,
                ))
            accumulated = 0
            current_losses = []
            if local_update >= int(config["training"]["max_updates"]):
                break
        progress.close()
        if accumulated != 0 or local_update != int(config["training"]["max_updates"]):
            raise RuntimeError("Recipient training ended off boundary")
        if ordered.observed != order:
            raise RuntimeError("Observed recipient order differs from frozen order")
        owner.config.use_cache = True
        training_metrics = {
            "optimizer_updates": local_update,
            "optimizer": optimizer_metadata,
            "update_metrics": updates,
            "observed_order_length": len(ordered.observed),
            "observed_order_sha256": int64_sha256(ordered.observed),
            "final_adam_global_step": (0 if arm == "fresh_adam" else 512)
            + local_update,
            "state_or_model_saved": False,
        }
        metrics_path = attempt / "training_metrics.json"
        atomic_write_json(metrics_path, training_metrics)
    finally:
        release(owner if owner is not None else base)
    if [row["recipient_update"] for row in probes] != list(PROBES):
        raise RuntimeError("Not all frozen probes completed")
    artifacts = {
        "start_manifest": artifact_record(start_path),
        "training_metrics": artifact_record(metrics_path),
    }
    for probe in probes:
        update = int(probe["recipient_update"])
        for kind, artifact in probe["artifacts"].items():
            artifacts[f"{kind}_u{update:04d}"] = artifact
    cell = {
        **identity,
        "completed_at": utc_now(),
        "initial_lora_state_sha256": initial_hash,
        "transplant": transplant,
        "probe_updates": list(PROBES),
        "probes": probes,
        "training_metrics": training_metrics,
        "artifacts": artifacts,
        "scope": (
            "Fresh seed-matched LoRA recipient on independent number rows; "
            "only named AdamW state is transplanted and no model/state is saved."
        ),
    }
    if not finite_tree(cell):
        raise RuntimeError("Non-finite completed cell")
    atomic_commit_json(cell_path, cell, attempt)
    validated = validate_cell(cell_path, config, receiver, seed, arm, condition)
    print(f"[{receiver}/{seed}/{arm}/{condition}] CELL COMPLETE", flush=True)
    return validated


def preflight(require_absence: bool) -> dict[str, Any]:
    config, parent_config = load_and_validate_config()
    if config["resource_policy"]["serial_mps_only"] and DEVICE.type != "mps":
        raise RuntimeError(f"Frozen campaign requires MPS, found {DEVICE}")
    assert_no_competing_experiment()
    free = shutil.disk_usage(ROOT).free
    required = int(config["resource_policy"]["minimum_launch_free_bytes"])
    if free < required:
        raise RuntimeError(
            f"Only {free / 1024**3:.2f} GiB free; require {required / 1024**3:.2f}"
        )
    if require_absence:
        forbidden = [path for path in (WORK, OUT_JSON, OUT_MD) if path.exists()]
        if forbidden:
            raise RuntimeError(f"Transplant namespace already exists: {forbidden}")
    tokenizer = dynamics.load_tokenizer()
    data_bundle = load_data_bundle(config, tokenizer)
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
    donors = {
        donor_key(receiver, seed, condition): validate_donor(
            config, parent_config, receiver, seed, condition
        )
        for receiver in RECEIVERS
        for seed in SEEDS
        for condition in CONDITIONS
    }
    orders = {
        str(seed): {
            "length": len(recipient_order(config, seed)),
            "sha256": int64_sha256(recipient_order(config, seed)),
        }
        for seed in SEEDS
    }
    expected_transplants = {
        f"{receiver}/{seed}/{arm}": expected_transplant_summary(
            config, receiver, seed, arm
        )
        for receiver in RECEIVERS
        for seed in SEEDS
        for arm in ARMS
    }
    return {
        "implementation": implementation_guard(),
        "free_bytes": free,
        "base_guards": base_guards,
        "donors": donors,
        "recipient_data": data_bundle["guard"],
        "recipient_orders": orders,
        "expected_transplant_summaries": expected_transplants,
        "expected_cell_count": len(expected_cell_paths()),
        "expected_cells": [relative(path) for path in expected_cell_paths()],
        "serial_only": True,
        "state_or_model_outputs": False,
    }


def freeze() -> dict[str, Any]:
    if RUNNER_LOCK_PATH.exists():
        return validate_runner_lock()
    guard = preflight(require_absence=True)
    lock = {
        "name": "numeric-fingerprint-optimizer-transplant-v1-runner-lock",
        "frozen_at": utc_now(),
        **guard,
    }
    exclusive_write_json(RUNNER_LOCK_PATH, lock)
    print("TRANSPLANT RUNNER FROZEN", flush=True)
    return validate_runner_lock()


def validate_runner_lock() -> dict[str, Any]:
    lock = load_json(RUNNER_LOCK_PATH)
    if lock.get("name") != "numeric-fingerprint-optimizer-transplant-v1-runner-lock":
        raise RuntimeError("Unexpected runner lock")
    if lock.get("implementation") != implementation_guard():
        raise RuntimeError("Runner implementation changed after freeze")
    if lock.get("expected_cells") != [relative(path) for path in expected_cell_paths()]:
        raise RuntimeError("Runner expected-cell list changed")
    return lock


@contextlib.contextmanager
def active_lock():
    ACTIVE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ACTIVE_LOCK_PATH.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another transplant runner holds the active lock") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started={utc_now()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_all() -> None:
    validate_runner_lock()
    config, parent_config = load_and_validate_config()
    assert_no_competing_experiment()
    for receiver in RECEIVERS:
        guard = compatibility.cached_weight_guard(receiver)
        expected = config["receivers"][receiver]
        if (
            guard["resolved_commit"] != expected["commit"]
            or guard["weight_sha256"] != expected["weight_sha256"]
            or guard["model_config_sha256"] != expected["model_config_sha256"]
        ):
            raise RuntimeError(f"Cached base changed before run: {receiver}")
        for seed in SEEDS:
            for condition in CONDITIONS:
                validate_donor(
                    config, parent_config, receiver, seed, condition
                )
    tokenizer = dynamics.load_tokenizer()
    data_bundle = load_data_bundle(config, tokenizer)
    with active_lock():
        for receiver in RECEIVERS:
            for seed in SEEDS:
                for arm in ARMS:
                    for condition in CONDITIONS:
                        run_cell(
                            config, tokenizer, data_bundle, receiver, seed, arm,
                            condition,
                        )
    print("TRANSPLANT CELLS COMPLETE", flush=True)


def load_evaluation(cell: dict[str, Any], update: int) -> dict[str, Any]:
    artifact = cell["artifacts"][f"evaluation_u{update:04d}"]
    return load_json(ROOT / artifact["path"])


def paired_effect(
    preference: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    pref_rows = preference["per_prompt"]
    control_rows = control["per_prompt"]
    if [row["prompt"] for row in pref_rows] != [row["prompt"] for row in control_rows]:
        raise RuntimeError("Behavior prompts differ within pair")
    margin = [
        float(pref["target_logit_margin"]) - float(base["target_logit_margin"])
        for pref, base in zip(pref_rows, control_rows)
    ]
    probability = [
        float(pref["target_candidate_probability"])
        - float(base["target_candidate_probability"])
        for pref, base in zip(pref_rows, control_rows)
    ]
    return {
        "margin": {**summary(margin), "positive_prompt_count": sum(x > 0 for x in margin)},
        "probability": {
            **summary(probability),
            "positive_prompt_count": sum(x > 0 for x in probability),
        },
        "per_prompt_margin": margin,
        "per_prompt_probability": probability,
    }


def contrast(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    values = [
        a - b
        for a, b in zip(left["per_prompt_margin"], right["per_prompt_margin"])
    ]
    return {**summary(values), "positive_prompt_count": sum(x > 0 for x in values)}


def analyze() -> dict[str, Any]:
    lock = validate_runner_lock()
    config, _ = load_and_validate_config()
    cells: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for receiver in RECEIVERS:
        for seed in SEEDS:
            for arm in ARMS:
                for condition in CONDITIONS:
                    path = expected_cell_path(receiver, seed, arm, condition)
                    if not path.is_file():
                        raise RuntimeError(f"Missing completed cell: {path}")
                    cells[(receiver, seed, arm, condition)] = validate_cell(
                        path, config, receiver, seed, arm, condition
                    )

    effects: dict[str, Any] = {}
    effect_arrays: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    update0_guard: dict[str, Any] = {}
    for receiver in RECEIVERS:
        effects[receiver] = {}
        for seed in SEEDS:
            seed_record: dict[str, Any] = {}
            update0_evaluations = []
            for arm in ARMS:
                arm_record: dict[str, Any] = {}
                preference_cell = cells[(receiver, seed, arm, "preference")]
                control_cell = cells[(receiver, seed, arm, "control")]
                for update in PROBES:
                    preference_eval = load_evaluation(preference_cell, update)
                    control_eval = load_evaluation(control_cell, update)
                    effect = paired_effect(preference_eval, control_eval)
                    effect_arrays[(receiver, seed, arm, update)] = effect
                    preference_probe = next(
                        row for row in preference_cell["probes"]
                        if row["recipient_update"] == update
                    )
                    control_probe = next(
                        row for row in control_cell["probes"]
                        if row["recipient_update"] == update
                    )
                    arm_record[str(update)] = {
                        "wolf_margin_effect": {
                            key: value for key, value in effect["margin"].items()
                        },
                        "wolf_probability_effect": {
                            key: value for key, value in effect["probability"].items()
                        },
                        "numeric_audit": {
                            "preference_recipient": preference_probe["numeric_audit_nll"],
                            "control_recipient": control_probe["numeric_audit_nll"],
                        },
                    }
                    if update == 0:
                        update0_evaluations.extend((preference_eval, control_eval))
                seed_record[arm] = arm_record
            reference = update0_evaluations[0]["per_prompt"]
            maximum = 0.0
            for evaluation in update0_evaluations[1:]:
                if [row["prompt"] for row in evaluation["per_prompt"]] != [
                    row["prompt"] for row in reference
                ]:
                    raise RuntimeError("Update-0 prompt order changed")
                maximum = max(
                    maximum,
                    max(
                        abs(float(a["target_logit_margin"]) - float(b["target_logit_margin"]))
                        for a, b in zip(reference, evaluation["per_prompt"])
                    ),
                )
            if maximum != 0.0:
                raise RuntimeError(f"Update-0 behavior differs across arms: {maximum}")
            update0_guard[f"{receiver}/{seed}"] = {
                "evaluations_compared": len(update0_evaluations),
                "maximum_per_prompt_margin_absolute_difference": maximum,
            }
            effects[receiver][str(seed)] = seed_record

    primary: dict[str, Any] = {}
    signs: dict[str, dict[str, list[float]]] = {}
    for receiver in RECEIVERS:
        primary[receiver] = {}
        signs[receiver] = {"control": [], "coordinate": []}
        for seed in SEEDS:
            pref = effect_arrays[(receiver, seed, "preference_v", 16)]
            control_v = effect_arrays[(receiver, seed, "control_v", 16)]
            permuted = effect_arrays[(receiver, seed, "permuted_preference_v", 16)]
            zero = effect_arrays[(receiver, seed, "step512_zero", 16)]
            control_contrast = contrast(pref, control_v)
            coordinate_contrast = contrast(pref, permuted)
            zero_contrast = contrast(pref, zero)
            signs[receiver]["control"].append(control_contrast["mean"])
            signs[receiver]["coordinate"].append(coordinate_contrast["mean"])
            primary[receiver][str(seed)] = {
                "preference_v_effect": pref["margin"],
                "control_v_effect": control_v["margin"],
                "permuted_preference_v_effect": permuted["margin"],
                "step512_zero_effect": zero["margin"],
                "preference_v_minus_control_v": control_contrast,
                "preference_v_minus_permuted_preference_v": coordinate_contrast,
                "preference_v_minus_step512_zero": zero_contrast,
            }
    ws3_support = all(value > 0 for value in signs["weight_seed3"]["control"]) and all(
        value > 0 for value in signs["weight_seed3"]["coordinate"]
    )
    ws3_against = all(
        value <= 0 for value in signs["weight_seed3"]["control"]
    ) or all(value <= 0 for value in signs["weight_seed3"]["coordinate"])
    standard_support = all(value > 0 for value in signs["standard"]["control"]) and all(
        value > 0 for value in signs["standard"]["coordinate"]
    )
    if ws3_support and standard_support:
        label = "stronger_cross_receiver_support"
    elif ws3_support:
        label = "preference_v_specificity_supported"
    elif ws3_against:
        label = "evidence_against_preference_v_specificity"
    else:
        label = "mixed"
    decision = {
        "label": label,
        "preference_v_specificity_supported": ws3_support,
        "evidence_against_preference_v_specificity": ws3_against,
        "standard_positive_control_support": standard_support,
        "signs": signs,
        "frozen_rules": config["frozen_decision"],
    }
    result = {
        "name": config["name"],
        "created_at": utc_now(),
        "question": config["question"],
        "config": artifact_record(CONFIG_PATH),
        "runner_lock": artifact_record(RUNNER_LOCK_PATH),
        "runner_lock_frozen_at": lock["frozen_at"],
        "cell_count": len(cells),
        "update0_guard": update0_guard,
        "primary": primary,
        "effects": effects,
        "decision": decision,
        "scope": config["scope"],
    }
    if not finite_tree(result):
        raise RuntimeError("Non-finite aggregate")
    atomic_write_json(OUT_JSON, result)
    lines = [
        "# Numeric fingerprint optimizer transplant v1",
        "",
        f"Frozen decision: **{label}**.",
        "",
        "Primary update: 16 recipient AdamW updates. Effects are preference-data minus control-data held-out wolf margins.",
        "",
        "| receiver | seed | pref-v E | control-v E | permuted-v E | zero E | pref-control | pref-permuted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for receiver in RECEIVERS:
        for seed in SEEDS:
            row = primary[receiver][str(seed)]
            lines.append(
                f"| {receiver} | {seed} | "
                f"{row['preference_v_effect']['mean']:+.6f} | "
                f"{row['control_v_effect']['mean']:+.6f} | "
                f"{row['permuted_preference_v_effect']['mean']:+.6f} | "
                f"{row['step512_zero_effect']['mean']:+.6f} | "
                f"{row['preference_v_minus_control_v']['mean']:+.6f} | "
                f"{row['preference_v_minus_permuted_preference_v']['mean']:+.6f} |"
            )
    lines.extend([
        "",
        "A positive result establishes off-trajectory v sufficiency/acceleration in fresh LoRA; a null does not establish necessity in the original live trajectory.",
        "",
        f"Runner lock: `{file_sha256(RUNNER_LOCK_PATH)}`",
        f"Config: `{file_sha256(CONFIG_PATH)}`",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"TRANSPLANT ANALYSIS DONE: {label}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", nargs="?", default="all",
        choices=("preflight", "freeze", "run", "analyze", "all"),
    )
    args = parser.parse_args()
    if args.command == "preflight":
        record = preflight(require_absence=not RUNNER_LOCK_PATH.exists())
        print(json.dumps(record, indent=2, sort_keys=True))
    elif args.command == "freeze":
        freeze()
    elif args.command == "run":
        run_all()
    elif args.command == "analyze":
        analyze()
    else:
        if not RUNNER_LOCK_PATH.exists():
            freeze()
        run_all()
        analyze()


if __name__ == "__main__":
    main()
