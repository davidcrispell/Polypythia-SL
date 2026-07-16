"""Causal online knockout of the matching-lineage wolf-writing route.

The frozen campaign trains data-seed2 LoRA students on the retained paired
numeric pools.  After each ordinary clipped-gradient AdamW step, the adaptive
displacement is either kept natural, stripped of its positive projection on a
fresh wolf-margin gradient, or subjected to an exactly energy-matched sham.
Adam moments evolve from the untouched numeric gradient and decoupled weight
decay is never modified.  A fourth arm releases the knockout after update 256.

No model or optimizer tensors are written.  Completed ``cell.json`` files are
the only reusable sentinels; interrupted attempts are preserved.
"""
from __future__ import annotations

import argparse
import copy
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
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM

import numeric_fingerprint_compatibility as compatibility
import numeric_fingerprint_dynamics as dynamics
from polypythia_sl.data import PREFERENCE_EVAL_PROMPTS, read_jsonl
from polypythia_sl.modeling import assert_single_token_animals
from polypythia_sl.optim import build_optimizer
from polypythia_sl.train import CompletionCollator, CompletionDataset, seed_everything


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
CONFIG_PATH = ROOT / "configs/wolf_route_knockout_v1.json"
SCRIPT_PATH = Path(__file__).resolve()
WORK = RUNS / "wolf_route_knockout_v1"
CELLS = WORK / "cells"
RUNNER_LOCK_PATH = WORK / "runner_lock.json"
ACTIVE_LOCK_PATH = WORK / ".active.lock"
OUT_JSON = RUNS / "wolf_route_knockout_v1.json"
OUT_MD = RUNS / "wolf_route_knockout_v1.md"
LOG_PATH = RUNS / "wolf_route_knockout_v1.log"

RULES = ("natural", "wolf_null", "sham", "null_then_release")
CONDITIONS = ("preference", "control")
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


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


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


def expected_cell_path(seed: int, condition: str, rule: str) -> Path:
    return CELLS / f"seed_{seed}" / condition / rule / "cell.json"


def expected_cell_paths(config: dict[str, Any]) -> list[Path]:
    return [
        expected_cell_path(int(seed), condition, rule)
        for seed in config["training"]["student_seeds"]
        for condition in CONDITIONS
        for rule in RULES
    ]


def load_and_validate_config() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    if config.get("name") != "wolf-route-knockout-v1":
        raise RuntimeError("Unexpected knockout config")
    if tuple(config["training"]["interventions"]) != RULES:
        raise RuntimeError("Intervention inventory changed")
    if tuple(config["data"]["conditions"]) != CONDITIONS:
        raise RuntimeError("Condition inventory changed")
    if int(config["training"]["cell_count"]) != len(expected_cell_paths(config)):
        raise RuntimeError("Cell-count guard failed")
    if config["training"]["probe_updates"] != [0, 16, 64, 128, 256, 257, 384, 512]:
        raise RuntimeError("Probe grid changed")
    if config["training"]["early_route_probe_updates"] != [0, 1, 2, 4, 8, 16, 32, 64]:
        raise RuntimeError("Early route-probe grid changed")
    if config["training"]["max_updates"] != 512 or config["training"]["epochs"] != 1:
        raise RuntimeError("Training horizon changed")
    if config["training"]["release_after_update"] != 256:
        raise RuntimeError("Release boundary changed")
    if compact_hash(list(PREFERENCE_EVAL_PROMPTS)) != config["evaluation"]["full_prompt_sha256"]:
        raise RuntimeError("Full behavior prompt hash changed")
    lo, hi = config["intervention"]["route_prompts_slice"]
    if compact_hash(list(PREFERENCE_EVAL_PROMPTS[lo:hi])) != config["intervention"]["route_prompt_sha256"]:
        raise RuntimeError("Route prompt split changed")
    lo, hi = config["evaluation"]["primary_disjoint_slice"]
    if compact_hash(list(PREFERENCE_EVAL_PROMPTS[lo:hi])) != config["evaluation"]["primary_disjoint_prompt_sha256"]:
        raise RuntimeError("Disjoint prompt split changed")
    for key, value in config["parents"].items():
        if key.endswith("_sha256") and isinstance(value, str):
            path_key = key.removesuffix("_sha256")
            if path_key in config["parents"]:
                path = ROOT / config["parents"][path_key]
                if file_sha256(path) != value:
                    raise RuntimeError(f"Frozen parent changed: {path}")
    for seed_bundle in config["parents"]["archived_natural"].values():
        for path_text, expected in seed_bundle.values():
            path = ROOT / path_text
            if file_sha256(path) != expected:
                raise RuntimeError(f"Archived natural source changed: {path}")
    for path_text, expected in config["parents"]["validated_u512_adapter_sha256"].values():
        path = ROOT / path_text
        if file_sha256(path) != expected:
            raise RuntimeError(f"Validated adapter changed: {path}")
    for condition in CONDITIONS:
        path = ROOT / config["data"][f"{condition}_pool"]
        if file_sha256(path) != config["data"][f"{condition}_pool_sha256"]:
            raise RuntimeError(f"Training pool changed: {path}")
        heldout = ROOT / config["data"][f"heldout_{condition}"]
        if file_sha256(heldout) != config["data"][f"heldout_{condition}_sha256"]:
            raise RuntimeError(f"Held-out pool changed: {heldout}")
    audit = config["intervention"]["counterfactual_release_audit_indices"]
    if int64_sha256(audit) != config["intervention"]["counterfactual_release_audit_indices_int64_sha256"]:
        raise RuntimeError("Release-audit index guard failed")
    if config["intervention"]["counterfactual_release_probe_updates"] != [
        1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 320, 384, 448, 512
    ]:
        raise RuntimeError("Counterfactual release grid changed")
    if config["intervention"]["horizon_release_source_updates"] != [
        0, 1, 4, 8, 16, 32, 64, 96, 128, 192, 256, 320, 384, 448
    ] or int(config["intervention"]["horizon_release_updates"]) != 32:
        raise RuntimeError("Fixed-horizon release design changed")
    return config


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
    relevant = (
        "scripts/numeric_",
        "scripts/dataorder_",
        "scripts/base_screening.py",
        "scripts/student_trait_write_probe.py",
        "scripts/cross_family_transport.py",
        "scripts/optimizer_transplant",
        "polypythia_sl.pipeline",
        "wolf_route_knockout.py run",
    )
    conflicts = []
    for pid, (_, command) in processes.items():
        if pid in ancestors or "python" not in command.lower():
            continue
        if command.lstrip().startswith("caffeinate ") and SCRIPT_PATH.name in command:
            continue
        if any(marker in command for marker in relevant):
            conflicts.append(f"{pid} {command}")
    if conflicts:
        raise RuntimeError("Competing experiment process:\n" + "\n".join(conflicts))


def data_order(config: dict[str, Any], seed: int) -> list[int]:
    count = int(config["data"]["rows_per_condition"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dynamics.IndexDataset(count), batch_size=int(config["training"]["batch_size"]),
        shuffle=True, generator=generator,
    )
    order = [int(value) for batch in loader for value in batch.tolist()]
    guard = config["training"]["data_order_guards"][str(seed)]
    if order[:16] != guard["first_16_indices"] or int64_sha256(order) != guard["epoch_sha256"]:
        raise RuntimeError(f"Historical order guard failed for seed {seed}")
    return order


def load_rows(config: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    training = {
        condition: read_jsonl(ROOT / config["data"][f"{condition}_pool"])
        for condition in CONDITIONS
    }
    heldout = {
        condition: read_jsonl(ROOT / config["data"][f"heldout_{condition}"])
        for condition in CONDITIONS
    }
    if any(len(rows) != int(config["data"]["rows_per_condition"]) for rows in training.values()):
        raise RuntimeError("Training row-count guard failed")
    if any(len(rows) != int(config["data"]["heldout_rows_per_condition"]) for rows in heldout.values()):
        raise RuntimeError("Held-out row-count guard failed")
    if [row["prompt"] for row in training["preference"]] != [row["prompt"] for row in training["control"]]:
        raise RuntimeError("Training pools are not prompt-paired")
    if [row["prompt"] for row in heldout["preference"]] != [row["prompt"] for row in heldout["control"]]:
        raise RuntimeError("Held-out pools are not prompt-paired")
    overlap = {row["prompt"] for rows in training.values() for row in rows} & {
        row["prompt"] for rows in heldout.values() for row in rows
    }
    if len(overlap) != int(config["data"]["expected_training_prompt_overlap"]):
        raise RuntimeError("Held-out/training prompt overlap guard failed")
    return training, heldout


def preflight(require_absence: bool = False) -> dict[str, Any]:
    config = load_and_validate_config()
    assert_no_competing_experiment()
    if config["resource_policy"]["serial_mps_only"] and DEVICE.type != "mps":
        raise RuntimeError(f"Campaign requires MPS, found {DEVICE}")
    if shutil.disk_usage(ROOT).free < int(config["resource_policy"]["minimum_launch_free_bytes"]):
        raise RuntimeError("Launch free-space guard failed")
    if require_absence and any(path.exists() for path in (CELLS, OUT_JSON, OUT_MD, LOG_PATH)):
        raise RuntimeError("Knockout namespace predates freeze")
    receiver = config["receiver"]
    guard = compatibility.cached_weight_guard("ds2")
    if (
        guard["resolved_commit"] != receiver["commit"]
        or guard["weight_sha256"] != receiver["weight_sha256"]
        or guard["model_config_sha256"] != receiver["model_config_sha256"]
    ):
        raise RuntimeError("Cached ds2 base guard failed")
    training, heldout = load_rows(config)
    orders = {str(seed): int64_sha256(data_order(config, int(seed))) for seed in config["training"]["student_seeds"]}
    return {
        "implementation": implementation_guard(),
        "base_guard": guard,
        "training_rows": {key: len(value) for key, value in training.items()},
        "heldout_rows": {key: len(value) for key, value in heldout.items()},
        "orders": orders,
        "expected_cells": [relative(path) for path in expected_cell_paths(config)],
        "preflight_used_model_forward_or_backward": False,
    }


def freeze() -> dict[str, Any]:
    if RUNNER_LOCK_PATH.exists():
        return validate_runner_lock()
    record = {
        "name": "wolf-route-knockout-v1-runner-lock",
        "created_at": utc_now(),
        "absence_before_freeze": True,
        "frozen": preflight(require_absence=True),
    }
    exclusive_write_json(RUNNER_LOCK_PATH, record)
    print(f"WOLF ROUTE KNOCKOUT FROZEN {file_sha256(RUNNER_LOCK_PATH)}", flush=True)
    return validate_runner_lock()


def validate_runner_lock() -> dict[str, Any]:
    if not RUNNER_LOCK_PATH.is_file():
        raise RuntimeError("Runner lock absent; freeze first")
    record = load_json(RUNNER_LOCK_PATH)
    if record.get("name") != "wolf-route-knockout-v1-runner-lock":
        raise RuntimeError("Unexpected runner lock")
    if record["frozen"]["implementation"] != implementation_guard():
        raise RuntimeError("Implementation changed after freeze")
    config = load_and_validate_config()
    if record["frozen"]["expected_cells"] != [relative(path) for path in expected_cell_paths(config)]:
        raise RuntimeError("Frozen cell inventory changed")
    return record


@contextlib.contextmanager
def active_lock():
    WORK.mkdir(parents=True, exist_ok=True)
    with ACTIVE_LOCK_PATH.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Knockout active lock is held") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_owner(config: dict[str, Any], seed: int):
    receiver = config["receiver"]
    base = AutoModelForCausalLM.from_pretrained(
        receiver["model_id"],
        revision=receiver["commit"],
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
    ).to(DEVICE)
    owner.config.use_cache = False
    trainable = dynamics.canonical_trainable(owner)
    count = sum(parameter.numel() for _, parameter in trainable)
    if count != int(config["training"]["expected_trainable_parameters"]):
        raise RuntimeError(f"Unexpected trainable parameter count: {count}")
    observed = dynamics.tensor_hash((name, parameter) for name, parameter in trainable)
    expected = config["training"]["expected_initial_lora_state_sha256"][str(seed)]
    if observed != expected:
        raise RuntimeError(f"Initial LoRA hash changed for seed {seed}: {observed}")
    return owner


def animal_ids(config: dict[str, Any], tokenizer) -> torch.Tensor:
    animals = [
        config["intervention"]["target"],
        *config["intervention"]["comparison_animals"],
    ]
    mapping = assert_single_token_animals(tokenizer, animals)
    return torch.tensor([mapping[value] for value in animals], device=DEVICE)


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("Cannot summarize empty/non-finite values")
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
    return {
        "mean": mean,
        "standard_error": se,
        "normal_approx_95_ci_low": mean - 1.96 * se,
        "normal_approx_95_ci_high": mean + 1.96 * se,
        "count": int(array.size),
    }


def behavior_values(
    owner: torch.nn.Module,
    tokenizer,
    token_ids: torch.Tensor,
    prompts: list[str],
    batch_size: int,
) -> dict[str, Any]:
    model = owner.base_model.model
    was_training = owner.training
    owner.eval()
    margins: list[float] = []
    probabilities: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts, return_tensors="pt", padding=True
            )
            encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
            logits = model(**encoded, use_cache=False).logits
            last = encoded["attention_mask"].sum(1) - 1
            rows = torch.arange(len(batch_prompts), device=DEVICE)
            selected = logits[rows, last][:, token_ids].float().cpu()
            margin = (
                selected[:, 0]
                - torch.logsumexp(selected[:, 1:], dim=1)
                + math.log(selected.shape[1] - 1)
            )
            probability = torch.softmax(selected, dim=1)[:, 0]
            margins.extend(float(value) for value in margin.tolist())
            probabilities.extend(float(value) for value in probability.tolist())
    if was_training:
        owner.train()
    return {
        "margin": summarize(margins),
        "probability": summarize(probabilities),
        "per_prompt": [
            {"prompt": prompt, "wolf_margin": margin, "wolf_probability": probability}
            for prompt, margin, probability in zip(prompts, margins, probabilities)
        ],
    }


def full_behavior_probe(
    owner: torch.nn.Module, tokenizer, token_ids: torch.Tensor, config: dict[str, Any]
) -> dict[str, Any]:
    values = behavior_values(
        owner,
        tokenizer,
        token_ids,
        list(PREFERENCE_EVAL_PROMPTS),
        int(config["evaluation"]["behavior_batch_size"]),
    )
    split = int(config["evaluation"]["primary_disjoint_slice"][0])
    route_rows = values["per_prompt"][:split]
    disjoint_rows = values["per_prompt"][split:]

    def subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "margin": summarize(row["wolf_margin"] for row in rows),
            "probability": summarize(row["wolf_probability"] for row in rows),
            "per_prompt": rows,
        }

    return {"full": values, "route": subset(route_rows), "disjoint": subset(disjoint_rows)}


@torch.inference_mode()
def completion_nll_values(
    owner: torch.nn.Module,
    dataset: CompletionDataset,
    tokenizer,
    batch_size: int,
    expected_tokens: int,
) -> dict[str, Any]:
    was_training = owner.training
    owner.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=CompletionCollator(tokenizer.pad_token_id),
    )
    values: list[float] = []
    token_counts: list[int] = []
    for batch in loader:
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        logits = owner(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits[:, :-1].float()
        labels = batch["labels"][:, 1:]
        flat = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), labels, ignore_index=-100, reduction="none"
        )
        mask = labels != -100
        counts = mask.sum(1)
        per_row = (flat * mask).sum(1) / counts
        values.extend(float(value) for value in per_row.cpu().tolist())
        token_counts.extend(int(value) for value in counts.cpu().tolist())
    if was_training:
        owner.train()
    if len(values) != len(dataset) or set(token_counts) != {expected_tokens}:
        raise RuntimeError("Per-row completion-NLL guard failed")
    return {
        "mean_nll": float(np.mean(values)),
        "per_row_nll": values,
        "rows": len(values),
        "supervised_tokens_per_row": expected_tokens,
    }


def vector_zeros(trainable: list[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(parameter) for name, parameter in trainable}


def vector_clone(value: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in value.items()}


def vector_dot(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    total = None
    for name in left:
        term = torch.sum(left[name].float() * right[name].float())
        total = term if total is None else total + term
    return float(total.detach().cpu()) if total is not None else 0.0


def vector_norm(value: dict[str, torch.Tensor]) -> float:
    return math.sqrt(max(vector_dot(value, value), 0.0))


def vector_scaled(value: dict[str, torch.Tensor], scale: float) -> dict[str, torch.Tensor]:
    return {name: tensor * scale for name, tensor in value.items()}


def vector_subtract(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {name: left[name] - right[name] for name in left}


def state_hashes(
    owner: torch.nn.Module, optimizer: torch.optim.Optimizer, update: int
) -> dict[str, Any]:
    trainable = dynamics.canonical_trainable(owner)
    lora = [(name, parameter.detach()) for name, parameter in trainable]
    m_named: list[tuple[str, torch.Tensor]] = []
    v_named: list[tuple[str, torch.Tensor]] = []
    steps: list[int] = []
    for name, parameter in trainable:
        state = optimizer.state.get(parameter, {})
        if update == 0:
            if state:
                raise RuntimeError("Adam state unexpectedly exists at update 0")
            m = torch.zeros_like(parameter)
            v = torch.zeros_like(parameter)
            step = 0
        else:
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise RuntimeError(f"Unexpected Adam state for {name}: {set(state)}")
            m = state["exp_avg"]
            v = state["exp_avg_sq"]
            step = int(float(state["step"].detach().cpu().item()))
        if step != update:
            raise RuntimeError(f"Adam step mismatch for {name}: {step} != {update}")
        m_named.append((name, m.detach()))
        v_named.append((name, v.detach()))
        steps.append(step)
    return {
        "optimizer_update": update,
        "lora_semantic_sha256": dynamics.semantic_tensor_hash(lora),
        "adam_exp_avg_semantic_sha256": dynamics.semantic_tensor_hash(m_named),
        "adam_exp_avg_sq_semantic_sha256": dynamics.semantic_tensor_hash(v_named),
        "adam_steps_exact": len(set(steps)) == 1 and steps[0] == update,
    }


def route_gradient(
    owner: torch.nn.Module,
    tokenizer,
    token_ids: torch.Tensor,
    config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    trainable = dynamics.canonical_trainable(owner)
    parameters = tuple(parameter for _, parameter in trainable)
    lo, hi = config["intervention"]["route_prompts_slice"]
    prompts = list(PREFERENCE_EVAL_PROMPTS[lo:hi])
    if len(prompts) != int(config["intervention"]["route_prompt_count"]):
        raise RuntimeError("Route prompt count changed")
    gradient = vector_zeros(trainable)
    margins: list[float] = []
    was_training = owner.training
    owner.eval()
    model = owner.base_model.model
    batch_size = int(config["intervention"]["behavior_batch_size"])
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
        logits = model(**encoded, use_cache=False).logits
        last = encoded["attention_mask"].sum(1) - 1
        rows = torch.arange(len(batch_prompts), device=DEVICE)
        selected = logits[rows, last][:, token_ids].float()
        batch_margins = (
            selected[:, 0]
            - torch.logsumexp(selected[:, 1:], dim=1)
            + math.log(selected.shape[1] - 1)
        )
        margins.extend(float(value) for value in batch_margins.detach().cpu().tolist())
        partial = torch.autograd.grad(
            batch_margins.sum() / len(prompts),
            parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        for (name, parameter), value in zip(trainable, partial):
            if value is not None:
                gradient[name].add_(value.detach().to(parameter.dtype))
    if was_training:
        owner.train()
    norm = vector_norm(gradient)
    if norm <= float(config["intervention"]["minimum_behavior_gradient_norm"]):
        raise RuntimeError(f"Wolf-route gradient norm is too small: {norm}")
    return gradient, {"margin": summarize(margins), "gradient_l2_norm": norm}


def numeric_backward(
    owner: torch.nn.Module,
    dataset: CompletionDataset,
    tokenizer,
    indices: list[int],
    config: dict[str, Any],
) -> dict[str, Any]:
    batch_size = int(config["training"]["batch_size"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    if len(indices) != batch_size * accumulation:
        raise RuntimeError("Effective batch size changed")
    collator = CompletionCollator(tokenizer.pad_token_id)
    owner.train()
    owner.zero_grad(set_to_none=True)
    losses: list[float] = []
    for start in range(0, len(indices), batch_size):
        batch = collator([dataset[index] for index in indices[start : start + batch_size]])
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        loss = owner(**batch, use_cache=False).loss
        losses.append(float(loss.detach().cpu()))
        (loss / accumulation).backward()
    if len(losses) != accumulation:
        raise RuntimeError("Numeric update microbatch count changed")
    returned = torch.nn.utils.clip_grad_norm_(
        owner.parameters(), float(config["training"]["max_grad_norm"])
    )
    trainable = dynamics.canonical_trainable(owner)
    if any(parameter.grad is None for _, parameter in trainable):
        raise RuntimeError("A trainable LoRA parameter has no numeric gradient")
    if not all(torch.isfinite(parameter.grad).all() for _, parameter in trainable):
        raise RuntimeError("Non-finite numeric gradient")
    return {
        "mean_microbatch_loss": float(np.mean(losses)),
        "microbatch_losses": losses,
        "gradient_norm_before_clipping": float(returned.detach().cpu()),
        "effective_examples": len(indices),
    }


def predicted_adaptive_update(
    owner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    update: int,
    learning_rate: float,
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    beta1, beta2 = (float(value) for value in config["training"]["betas"])
    eps = float(config["training"]["eps"])
    bc1 = 1.0 - beta1**update
    bc2 = 1.0 - beta2**update
    result: dict[str, torch.Tensor] = {}
    for name, parameter in dynamics.canonical_trainable(owner):
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError(f"Missing numeric gradient: {name}")
        state = optimizer.state.get(parameter, {})
        if update == 1:
            if state:
                raise RuntimeError("Adam state exists before update 1")
            m_old = torch.zeros_like(parameter)
            v_old = torch.zeros_like(parameter)
        else:
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise RuntimeError(f"Malformed Adam state before update {update}: {name}")
            step = int(float(state["step"].detach().cpu().item()))
            if step != update - 1:
                raise RuntimeError(f"Pre-step Adam count mismatch: {step} != {update - 1}")
            m_old = state["exp_avg"]
            v_old = state["exp_avg_sq"]
        m_new = m_old * beta1 + gradient * (1.0 - beta1)
        v_new = v_old * beta2 + gradient.square() * (1.0 - beta2)
        denominator = (v_new / bc2).sqrt().add(eps)
        result[name] = (-learning_rate / bc1) * m_new / denominator
    return result


def deterministic_orthogonal(
    bhat: dict[str, torch.Tensor],
    dhat: dict[str, torch.Tensor],
    namespace: str,
    seed: int,
    condition: str,
    update: int,
) -> dict[str, torch.Tensor]:
    material = f"{namespace}|{seed}|{condition}|{update}".encode()
    local_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little") % (2**63)
    generator = torch.Generator(device="cpu").manual_seed(local_seed)
    first = next(iter(bhat.values()))
    total = sum(tensor.numel() for tensor in bhat.values())
    flat = torch.randn(total, generator=generator, dtype=torch.float32, device="cpu").to(
        device=first.device, dtype=first.dtype
    )
    z: dict[str, torch.Tensor] = {}
    offset = 0
    for name, tensor in bhat.items():
        z[name] = flat[offset : offset + tensor.numel()].reshape(tensor.shape)
        offset += tensor.numel()
    for _ in range(2):
        z = vector_subtract(z, vector_scaled(bhat, vector_dot(z, bhat)))
        z = vector_subtract(z, vector_scaled(dhat, vector_dot(z, dhat)))
    norm = vector_norm(z)
    if norm <= 1e-12:
        raise RuntimeError("Deterministic sham orthogonal vector collapsed")
    return vector_scaled(z, 1.0 / norm)


def surgery_components(
    adaptive: dict[str, torch.Tensor],
    behavior_gradient: dict[str, torch.Tensor],
    config: dict[str, Any],
    seed: int,
    condition: str,
    update: int,
    need_sham: bool,
) -> dict[str, Any]:
    bnorm = vector_norm(behavior_gradient)
    bhat = vector_scaled(behavior_gradient, 1.0 / bnorm)
    natural_norm = vector_norm(adaptive)
    projection = vector_dot(bhat, adaptive)
    r = max(0.0, projection)
    wolf = vector_scaled(bhat, r)
    nonwolf = vector_subtract(adaptive, wolf)
    nonwolf_norm = vector_norm(nonwolf)
    sham = vector_clone(adaptive)
    sham_record: dict[str, Any] = {"constructed": False}
    if need_sham and r > 0.0:
        minimum = float(config["intervention"]["minimum_sham_residual_norm"])
        if nonwolf_norm <= minimum:
            raise RuntimeError("Exact sham impossible: non-wolf residual is too small")
        c = r / nonwolf_norm
        tolerance = float(config["intervention"]["sham_c_upper_tolerance"])
        if c > 1.0 + tolerance:
            raise RuntimeError(f"Exact sham impossible: r/||d||={c}")
        c = min(c, 1.0)
        dhat = vector_scaled(nonwolf, 1.0 / nonwolf_norm)
        z = deterministic_orthogonal(
            bhat,
            dhat,
            config["intervention"]["sham_seed_namespace"],
            seed,
            condition,
            update,
        )
        q = {
            name: c * dhat[name] + math.sqrt(max(1.0 - c * c, 0.0)) * z[name]
            for name in adaptive
        }
        sham = vector_subtract(adaptive, vector_scaled(q, r))
        q_norm = vector_norm(q)
        q_b = vector_dot(q, bhat)
        q_d = vector_dot(q, nonwolf)
        sham_record = {
            "constructed": True,
            "c": c,
            "q_l2_norm": q_norm,
            "q_dot_bhat": q_b,
            "q_dot_nonwolf": q_d,
        }
    absolute = float(config["intervention"]["surgery_absolute_tolerance"])
    relative = float(config["intervention"]["surgery_relative_tolerance"])
    null_intervention = vector_norm(vector_subtract(adaptive, nonwolf))
    sham_intervention = vector_norm(vector_subtract(adaptive, sham))
    null_norm = vector_norm(nonwolf)
    sham_norm = vector_norm(sham)
    wolf_projection_natural = vector_dot(bhat, adaptive)
    wolf_projection_null = vector_dot(bhat, nonwolf)
    wolf_projection_sham = vector_dot(bhat, sham)
    reconstruction_error = vector_norm(
        vector_subtract(
            adaptive,
            {
                name: wolf[name] + nonwolf[name]
                for name in adaptive
            },
        )
    )
    scale = max(natural_norm, r, 1e-12)
    if r > 0.0 and (
        abs(wolf_projection_null) > absolute + relative * scale
        or reconstruction_error > absolute + relative * scale
    ):
        raise RuntimeError(
            "Wolf-null projection/reconstruction invariant failed: "
            f"projection={wolf_projection_null} reconstruction={reconstruction_error}"
        )
    if need_sham and r > 0.0:
        errors = [
            abs(null_intervention - r),
            abs(sham_intervention - r),
            abs(null_norm - sham_norm),
            abs(wolf_projection_natural - wolf_projection_sham),
        ]
        if max(errors) > absolute + relative * scale:
            raise RuntimeError(f"Exact sham invariant failed: {errors}")
    return {
        "natural": adaptive,
        "bhat": bhat,
        "wolf": wolf,
        "nonwolf": nonwolf,
        "sham": sham,
        "record": {
            "raw_projection": projection,
            "positive_projection_r": r,
            "treated": r > float(config["intervention"]["minimum_local_treatment_projection"]),
            "r_over_natural_l2": r / natural_norm if natural_norm else 0.0,
            "natural_l2": natural_norm,
            "wolf_component_l2": vector_norm(wolf),
            "nonwolf_component_l2": nonwolf_norm,
            "null_intervention_l2": null_intervention,
            "sham_intervention_l2": sham_intervention,
            "null_post_l2": null_norm,
            "sham_post_l2": sham_norm,
            "wolf_projection_natural": wolf_projection_natural,
            "wolf_projection_null": wolf_projection_null,
            "wolf_projection_sham": wolf_projection_sham,
            "natural_reconstruction_l2_error": reconstruction_error,
            "sham": sham_record,
        },
    }


def assign_candidate(
    trainable: list[tuple[str, torch.nn.Parameter]],
    before: dict[str, torch.Tensor],
    decay: dict[str, torch.Tensor],
    adaptive: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, parameter in trainable:
            parameter.copy_(before[name] + decay[name] + adaptive[name])


def verify_manual_adam(
    predicted: dict[str, torch.Tensor], actual: dict[str, torch.Tensor], config: dict[str, Any]
) -> dict[str, Any]:
    error_sq = 0.0
    actual_sq = 0.0
    maximum = 0.0
    for name in predicted:
        error = predicted[name].float() - actual[name].float()
        error_sq += float(torch.sum(error * error).detach().cpu())
        actual_sq += float(torch.sum(actual[name].float().square()).detach().cpu())
        maximum = max(maximum, float(error.abs().max().detach().cpu()))
    relative = math.sqrt(error_sq) / max(math.sqrt(actual_sq), 1e-30)
    passed = (
        maximum <= float(config["intervention"]["manual_adam_max_parameter_abs_tolerance"])
        and relative <= float(config["intervention"]["manual_adam_relative_update_l2_tolerance"])
    )
    record = {"maximum_absolute_error": maximum, "relative_update_l2_error": relative, "passed": passed}
    if not passed:
        raise RuntimeError(f"Manual AdamW decomposition failed: {record}")
    return record


def small_evaluation(
    owner: torch.nn.Module,
    tokenizer,
    token_ids: torch.Tensor,
    audit_dataset: CompletionDataset,
    config: dict[str, Any],
) -> dict[str, Any]:
    lo, hi = config["evaluation"]["primary_disjoint_slice"]
    behavior = behavior_values(
        owner,
        tokenizer,
        token_ids,
        list(PREFERENCE_EVAL_PROMPTS[lo:hi]),
        int(config["evaluation"]["behavior_batch_size"]),
    )
    nll = completion_nll_values(
        owner,
        audit_dataset,
        tokenizer,
        int(config["evaluation"]["heldout_completion_batch_size"]),
        int(config["data"]["supervised_tokens_per_row"]),
    )
    return {"disjoint_behavior": behavior, "matching_audit_nll": nll}


def evaluate_candidates(
    owner: torch.nn.Module,
    tokenizer,
    token_ids: torch.Tensor,
    audit_dataset: CompletionDataset,
    trainable: list[tuple[str, torch.nn.Parameter]],
    before: dict[str, torch.Tensor],
    decay: dict[str, torch.Tensor],
    components: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    zeros = {name: torch.zeros_like(value) for name, value in before.items()}
    candidates = {
        "decay_only": zeros,
        "wolf_only": components["wolf"],
        "wolf_null": components["nonwolf"],
        "natural": components["natural"],
        "sham": components["sham"],
    }
    results: dict[str, Any] = {}
    for name, adaptive in candidates.items():
        assign_candidate(trainable, before, decay, adaptive)
        results[name] = small_evaluation(
            owner, tokenizer, token_ids, audit_dataset, config
        )
    return results


def run_probe(
    owner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokenizer,
    token_ids: torch.Tensor,
    heldout_datasets: dict[str, CompletionDataset],
    audit_datasets: dict[str, CompletionDataset],
    condition: str,
    update: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    major = update in set(config["training"]["probe_updates"])
    early = update in set(config["training"]["early_route_probe_updates"])
    if not major and not early:
        raise RuntimeError(f"Unscheduled probe update: {update}")
    record: dict[str, Any] = {
        "optimizer_update": update,
        "major_probe": major,
        "early_route_probe": early,
        "behavior": full_behavior_probe(owner, tokenizer, token_ids, config),
        "matching_audit_nll": completion_nll_values(
            owner,
            audit_datasets[condition],
            tokenizer,
            int(config["evaluation"]["heldout_completion_batch_size"]),
            int(config["data"]["supervised_tokens_per_row"]),
        ),
    }
    if major:
        was_training = owner.training
        owner.eval()
        record["heldout_completion_nll"] = {
            data_condition: dynamics.completion_nll(
                owner,
                heldout_datasets[data_condition],
                tokenizer,
                int(config["evaluation"]["heldout_completion_batch_size"]),
            )
            for data_condition in CONDITIONS
        }
        record["state_hashes"] = state_hashes(owner, optimizer, update)
        if was_training:
            owner.train()
    if not finite_tree(record):
        raise RuntimeError("Non-finite probe")
    return record


def capture_runtime_state(
    owner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> dict[str, Any]:
    trainable = dynamics.canonical_trainable(owner)
    return {
        "parameters": {name: parameter.detach().clone() for name, parameter in trainable},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
    }


def restore_runtime_state(
    owner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    snapshot: dict[str, Any],
) -> None:
    trainable = dynamics.canonical_trainable(owner)
    with torch.no_grad():
        for name, parameter in trainable:
            parameter.copy_(snapshot["parameters"][name])
    optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
    scheduler.load_state_dict(copy.deepcopy(snapshot["scheduler"]))
    owner.zero_grad(set_to_none=True)


def run_natural_horizon_branch(
    owner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    tokenizer,
    token_ids: torch.Tensor,
    dataset: CompletionDataset,
    audit_dataset: CompletionDataset,
    order: list[int],
    source_update: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    horizon = int(config["intervention"]["horizon_release_updates"])
    target = source_update + horizon
    snapshot = capture_runtime_state(owner, optimizer, scheduler)
    start_hashes = state_hashes(owner, optimizer, source_update)
    start_schedule = {
        "optimizer_lrs": [float(group["lr"]) for group in optimizer.param_groups],
        "scheduler_state_sha256": compact_hash(scheduler.state_dict()),
        "scheduler_last_epoch": int(scheduler.last_epoch),
    }
    losses: list[float] = []
    lrs: list[float] = []
    try:
        for update in range(source_update + 1, target + 1):
            start = (update - 1) * int(config["training"]["examples_per_update"])
            indices = order[start : start + int(config["training"]["examples_per_update"])]
            numeric = numeric_backward(owner, dataset, tokenizer, indices, config)
            losses.append(float(numeric["mean_microbatch_loss"]))
            lrs.append(float(optimizer.param_groups[0]["lr"]))
            optimizer.step()
            scheduler.step()
            owner.zero_grad(set_to_none=True)
        endpoint = small_evaluation(
            owner, tokenizer, token_ids, audit_dataset, config
        )
        branch_hashes = state_hashes(owner, optimizer, target)
    finally:
        restore_runtime_state(owner, optimizer, scheduler, snapshot)
    restored = state_hashes(owner, optimizer, source_update)
    restored_schedule = {
        "optimizer_lrs": [float(group["lr"]) for group in optimizer.param_groups],
        "scheduler_state_sha256": compact_hash(scheduler.state_dict()),
        "scheduler_last_epoch": int(scheduler.last_epoch),
    }
    if restored != start_hashes:
        raise RuntimeError("Horizon branch failed exact parent-state restoration")
    if restored_schedule != start_schedule:
        raise RuntimeError("Horizon branch failed exact optimizer/scheduler restoration")
    return {
        "source_update": source_update,
        "target_update": target,
        "horizon_updates": horizon,
        "start_state_hashes": start_hashes,
        "restored_state_hashes": restored,
        "start_schedule": start_schedule,
        "restored_schedule": restored_schedule,
        "branch_target_state_hashes": branch_hashes,
        "mean_training_loss": float(np.mean(losses)),
        "learning_rate_first": lrs[0],
        "learning_rate_last": lrs[-1],
        "natural_release_endpoint": endpoint,
    }


def adapter_reference_semantic_hash(
    config: dict[str, Any], seed: int, condition: str,
    trainable: list[tuple[str, torch.nn.Parameter]],
) -> str:
    key = f"{seed}/{condition}"
    path = ROOT / config["parents"]["validated_u512_adapter_sha256"][key][0]
    saved = load_safetensors(path, device="cpu")
    mapped: list[tuple[str, torch.Tensor]] = []
    for live_name, _ in trainable:
        saved_name = live_name.replace(".default", "")
        if saved_name not in saved:
            raise RuntimeError(f"Validated adapter lacks tensor {saved_name}")
        mapped.append((live_name, saved[saved_name]))
    observed = dynamics.semantic_tensor_hash(mapped)
    expected = config["parents"]["validated_u512_adapter_live_semantic_sha256"][key]
    if observed != expected:
        raise RuntimeError(f"Validated adapter semantic guard changed: {key}")
    return observed


def natural_replay_guard(
    owner: torch.nn.Module,
    config: dict[str, Any],
    seed: int,
    condition: str,
    update_records: list[dict[str, Any]],
    final_probe: dict[str, Any],
) -> dict[str, Any]:
    sources = config["parents"]["archived_natural"][str(seed)]
    metrics_path = ROOT / sources[f"{condition}_metrics"][0]
    archived_metrics = load_json(metrics_path)["update_metrics"][:512]
    if len(update_records) != 512 or len(archived_metrics) != 512:
        raise RuntimeError("Natural replay metric length changed")
    maximum = {"loss": 0.0, "gradient_norm": 0.0, "learning_rate": 0.0}
    for observed, expected in zip(update_records, archived_metrics):
        if int(observed["optimizer_update"]) != int(expected["optimizer_update"]):
            raise RuntimeError("Natural replay update indexing changed")
        maximum["loss"] = max(
            maximum["loss"],
            abs(float(observed["mean_microbatch_loss"]) - float(expected["mean_microbatch_loss"])),
        )
        maximum["gradient_norm"] = max(
            maximum["gradient_norm"],
            abs(float(observed["gradient_norm_before_clipping"]) - float(expected["gradient_norm_before_clipping"])),
        )
        maximum["learning_rate"] = max(
            maximum["learning_rate"],
            abs(float(observed["learning_rates_after_update"][0]) - float(expected["learning_rates_after_update"][0])),
        )
    if maximum["loss"] > 5e-6 or maximum["gradient_norm"] > 5e-5 or maximum["learning_rate"] > 1e-15:
        raise RuntimeError(f"Natural update metrics failed replay: {maximum}")
    evaluation_path = ROOT / sources[f"{condition}_evaluation_u512"][0]
    archived_evaluation = load_json(evaluation_path)
    observed_rows = final_probe["behavior"]["full"]["per_prompt"]
    expected_rows = archived_evaluation["per_prompt"]
    if len(observed_rows) != len(expected_rows):
        raise RuntimeError("Natural behavior prompt count changed")
    behavior_error = {"margin": 0.0, "probability": 0.0}
    for observed, expected in zip(observed_rows, expected_rows):
        if observed["prompt"] != expected["prompt"]:
            raise RuntimeError("Natural replay prompt order changed")
        behavior_error["margin"] = max(
            behavior_error["margin"],
            abs(float(observed["wolf_margin"]) - float(expected["target_logit_margin"])),
        )
        behavior_error["probability"] = max(
            behavior_error["probability"],
            abs(float(observed["wolf_probability"]) - float(expected["target_candidate_probability"])),
        )
    tolerance = float(config["evaluation"]["archive_per_prompt_absolute_tolerance"])
    if max(behavior_error.values()) > tolerance:
        raise RuntimeError(f"Natural behavior replay failed: {behavior_error}")
    trainable = dynamics.canonical_trainable(owner)
    live_hash = dynamics.semantic_tensor_hash(
        (name, parameter.detach()) for name, parameter in trainable
    )
    reference_hash = adapter_reference_semantic_hash(config, seed, condition, trainable)
    if live_hash != reference_hash:
        raise RuntimeError(
            f"Natural u512 adapter is not semantically exact: {live_hash} != {reference_hash}"
        )
    return {
        "passed": True,
        "maximum_update_metric_absolute_error": maximum,
        "maximum_per_prompt_absolute_error": behavior_error,
        "live_u512_lora_semantic_sha256": live_hash,
        "reference_u512_lora_semantic_sha256": reference_hash,
        "semantic_hash_exact": True,
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
    attempt = root / f"attempt_{max(numbers, default=0) + 1:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt


def cell_identity(seed: int, condition: str, rule: str) -> dict[str, Any]:
    return {
        "name": "wolf-route-knockout-v1-cell",
        "runner_lock_sha256": file_sha256(RUNNER_LOCK_PATH),
        "config_sha256": file_sha256(CONFIG_PATH),
        "seed": seed,
        "student_condition": condition,
        "intervention_rule": rule,
    }


def run_cell(
    config: dict[str, Any],
    tokenizer,
    token_ids: torch.Tensor,
    training_datasets: dict[str, CompletionDataset],
    heldout_datasets: dict[str, CompletionDataset],
    audit_datasets: dict[str, CompletionDataset],
    seed: int,
    condition: str,
    rule: str,
) -> dict[str, Any]:
    if rule not in RULES or condition not in CONDITIONS:
        raise RuntimeError("Unexpected cell identity")
    root = expected_cell_path(seed, condition, rule).parent
    attempt = next_attempt(root)
    identity = cell_identity(seed, condition, rule)
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
    print(f"[{seed}/{condition}/{rule}] {attempt.name} starting", flush=True)
    owner = None
    update_records: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    horizon_branches: dict[str, Any] = {}
    horizon_parent_targets: dict[str, Any] = {}
    losses: list[float] = []
    try:
        owner = load_owner(config, seed)
        trainable = dynamics.canonical_trainable(owner)
        optimizer, optimizer_metadata = build_optimizer(owner, config["training"])
        warmup = int(config["training"]["warmup_updates"])
        horizon = int(config["training"]["schedule_total_updates"])

        def lr_scale(step: int) -> float:
            if warmup and step < warmup:
                return (step + 1) / warmup
            return max(horizon - step, 0) / max(horizon - warmup, 1)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        order = data_order(config, seed)
        probe_updates = set(config["training"]["probe_updates"]) | set(
            config["training"]["early_route_probe_updates"]
        )
        candidate_updates = set(
            config["intervention"]["counterfactual_release_probe_updates"]
        )
        branch_sources = set(config["intervention"]["horizon_release_source_updates"])
        branch_targets = {
            value + int(config["intervention"]["horizon_release_updates"])
            for value in branch_sources
        }
        probes.append(
            run_probe(
                owner,
                optimizer,
                tokenizer,
                token_ids,
                heldout_datasets,
                audit_datasets,
                condition,
                0,
                config,
            )
        )
        if rule == "wolf_null" and 0 in branch_sources:
            horizon_branches["0"] = run_natural_horizon_branch(
                owner,
                optimizer,
                scheduler,
                tokenizer,
                token_ids,
                training_datasets[condition],
                audit_datasets[condition],
                order,
                0,
                config,
            )
        max_updates = int(config["training"]["max_updates"])
        examples_per_update = int(config["training"]["examples_per_update"])
        progress = tqdm(total=max_updates, desc=f"{seed}/{condition}/{rule}", unit="update")
        for update in range(1, max_updates + 1):
            if update % 16 == 1 and shutil.disk_usage(ROOT).free < int(
                config["resource_policy"]["minimum_runtime_free_bytes"]
            ):
                raise RuntimeError("Runtime free-space guard failed")
            start = (update - 1) * examples_per_update
            indices = order[start : start + examples_per_update]
            if len(indices) != examples_per_update:
                raise RuntimeError("Frozen data order exhausted unexpectedly")
            candidate_source_hashes = None
            if rule == "wolf_null" and update in candidate_updates:
                candidate_source_hashes = state_hashes(owner, optimizer, update - 1)
            numeric = numeric_backward(
                owner, training_datasets[condition], tokenizer, indices, config
            )
            losses.extend(numeric["microbatch_losses"])
            behavior_gradient, behavior_before = route_gradient(
                owner, tokenizer, token_ids, config
            )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            before = {
                name: parameter.detach().clone() for name, parameter in trainable
            }
            predicted = predicted_adaptive_update(
                owner, optimizer, update, learning_rate, config
            )
            decay = {
                name: -learning_rate
                * float(config["training"]["weight_decay"])
                * before[name]
                for name in before
            }
            optimizer.step()
            natural_after = {
                name: parameter.detach().clone() for name, parameter in trainable
            }
            adaptive = {
                name: natural_after[name] - before[name] - decay[name]
                for name in before
            }
            manual = verify_manual_adam(predicted, adaptive, config)
            needs_candidate = rule == "wolf_null" and update in candidate_updates
            needs_sham = rule == "sham" or needs_candidate
            components = surgery_components(
                adaptive,
                behavior_gradient,
                config,
                seed,
                condition,
                update,
                needs_sham,
            )
            if needs_candidate:
                candidate_results = evaluate_candidates(
                    owner,
                    tokenizer,
                    token_ids,
                    audit_datasets[condition],
                    trainable,
                    before,
                    decay,
                    components,
                    config,
                )
                candidates.append(
                    {
                        "candidate_update": update,
                        "source_optimizer_update": update - 1,
                        "source_state_hashes": candidate_source_hashes,
                        "next_example_indices": indices,
                        "next_example_indices_int64_sha256": int64_sha256(indices),
                        "learning_rate": learning_rate,
                        "route_before": behavior_before,
                        "surgery": components["record"],
                        "candidate_states": candidate_results,
                    }
                )
            if rule == "natural" or (
                rule == "null_then_release"
                and update > int(config["training"]["release_after_update"])
            ):
                chosen = adaptive
            elif rule in {"wolf_null", "null_then_release"}:
                chosen = components["nonwolf"]
            elif rule == "sham":
                chosen = components["sham"]
            else:
                raise RuntimeError(rule)
            if not (
                rule == "natural"
                or (
                    rule == "null_then_release"
                    and update > int(config["training"]["release_after_update"])
                )
            ):
                assign_candidate(trainable, before, decay, chosen)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_record = {
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
                "route_margin_before": behavior_before["margin"]["mean"],
                "route_gradient_l2_norm": behavior_before["gradient_l2_norm"],
                "manual_adamw_verification": manual,
                "surgery": components["record"],
            }
            update_records.append(update_record)
            if update in probe_updates:
                probes.append(
                    run_probe(
                        owner,
                        optimizer,
                        tokenizer,
                        token_ids,
                        heldout_datasets,
                        audit_datasets,
                        condition,
                        update,
                        config,
                    )
                )
            if rule == "wolf_null" and update in branch_sources:
                horizon_branches[str(update)] = run_natural_horizon_branch(
                    owner,
                    optimizer,
                    scheduler,
                    tokenizer,
                    token_ids,
                    training_datasets[condition],
                    audit_datasets[condition],
                    order,
                    update,
                    config,
                )
            if rule == "wolf_null" and update in branch_targets:
                horizon_parent_targets[str(update)] = small_evaluation(
                    owner, tokenizer, token_ids, audit_datasets[condition], config
                )
            progress.update(1)
            progress.set_postfix(loss=f"{numeric['mean_microbatch_loss']:.3f}")
            if update in probe_updates or (rule == "wolf_null" and update in candidate_updates):
                print(
                    f"[{seed}/{condition}/{rule}] u{update} "
                    f"loss={numeric['mean_microbatch_loss']:.6f} "
                    f"r={components['record']['positive_projection_r']:.6g}",
                    flush=True,
                )
        progress.close()
        owner.config.use_cache = True
        owner.eval()
        final_probe = next(row for row in probes if row["optimizer_update"] == 512)
        replay = None
        if rule == "natural":
            replay = natural_replay_guard(
                owner, config, seed, condition, update_records, final_probe
            )
        metrics = {
            "optimizer": optimizer_metadata,
            "optimizer_updates": max_updates,
            "schedule_total_updates": horizon,
            "warmup_updates": warmup,
            "mean_microbatch_loss": float(np.mean(losses)),
            "final_microbatch_loss": losses[-1],
            "update_metrics": update_records,
            "data_order_sha256": int64_sha256(order),
            "seed": seed,
            "condition": condition,
            "rule": rule,
        }
        result = {
            **identity,
            "completed_at": utc_now(),
            "probes": probes,
            "counterfactual_candidates": candidates,
            "horizon_release_branches": horizon_branches,
            "horizon_parent_targets": horizon_parent_targets,
            "natural_replay_guard": replay,
            "final_state_hashes": state_hashes(owner, optimizer, 512),
            "no_model_or_optimizer_tensors_written": True,
        }
        if not finite_tree(metrics) or not finite_tree(result):
            raise RuntimeError("Non-finite cell output")
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
        release_model(owner)
    cell = {
        **identity,
        "completed_at": result["completed_at"],
        "attempt": relative(attempt),
        "artifacts": {
            "start_manifest": artifact_record(start_path),
            "metrics": artifact_record(metrics_path),
            "result": artifact_record(result_path),
        },
        "natural_replay_guard": result["natural_replay_guard"],
        "final_state_hashes": result["final_state_hashes"],
    }
    exclusive_write_json(expected_cell_path(seed, condition, rule), cell)
    print(f"[{seed}/{condition}/{rule}] CELL DONE", flush=True)
    return validate_cell(expected_cell_path(seed, condition, rule), config)


def validate_cell(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    cell = load_json(path)
    seed = int(cell["seed"])
    condition = cell["student_condition"]
    rule = cell["intervention_rule"]
    if path.resolve() != expected_cell_path(seed, condition, rule).resolve():
        raise RuntimeError(f"Cell is stored under the wrong identity path: {path}")
    expected = cell_identity(
        seed, condition, rule
    )
    if any(cell.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Cell identity mismatch: {path}")
    attempt = ROOT / cell["attempt"]
    if attempt.parent.resolve() != path.parent.resolve() or not attempt.name.startswith(
        "attempt_"
    ):
        raise RuntimeError(f"Cell attempt path is inconsistent: {path}")
    expected_artifact_names = {
        "start_manifest": attempt / "start_manifest.json",
        "metrics": attempt / "metrics.json",
        "result": attempt / "result.json",
    }
    if set(cell["artifacts"]) != set(expected_artifact_names):
        raise RuntimeError(f"Cell artifact inventory changed: {path}")
    for name, artifact in cell["artifacts"].items():
        artifact_path = ROOT / artifact["path"]
        if artifact_path.resolve() != expected_artifact_names[name].resolve():
            raise RuntimeError(f"Cell artifact path is inconsistent: {artifact_path}")
        if (
            not artifact_path.is_file()
            or artifact_path.stat().st_size != int(artifact["bytes"])
            or file_sha256(artifact_path) != artifact["sha256"]
        ):
            raise RuntimeError(f"Cell artifact changed: {artifact_path}")
    result = load_json(ROOT / cell["artifacts"]["result"]["path"])
    if any(result.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Result identity mismatch: {path}")
    if (
        result.get("final_state_hashes") != cell.get("final_state_hashes")
        or result.get("natural_replay_guard") != cell.get("natural_replay_guard")
        or result.get("no_model_or_optimizer_tensors_written") is not True
    ):
        raise RuntimeError(f"Cell/result duplicated guard mismatch: {path}")
    metrics = load_json(ROOT / cell["artifacts"]["metrics"]["path"])
    if (
        metrics.get("seed") != seed
        or metrics.get("condition") != condition
        or metrics.get("rule") != rule
        or metrics.get("optimizer_updates") != int(config["training"]["max_updates"])
        or metrics.get("schedule_total_updates")
        != int(config["training"]["schedule_total_updates"])
        or metrics.get("warmup_updates") != int(config["training"]["warmup_updates"])
        or metrics.get("data_order_sha256")
        != config["training"]["data_order_guards"][str(seed)]["epoch_sha256"]
    ):
        raise RuntimeError(f"Cell metrics identity/config mismatch: {path}")
    updates = metrics.get("update_metrics", [])
    if [row.get("optimizer_update") for row in updates] != list(range(1, 513)):
        raise RuntimeError(f"Cell update sequence mismatch: {path}")
    warmup = int(config["training"]["warmup_updates"])
    horizon = int(config["training"]["schedule_total_updates"])
    base_lr = float(config["training"]["learning_rate"])

    def scheduled_lr_after(update: int) -> float:
        if warmup and update < warmup:
            return base_lr * (update + 1) / warmup
        return base_lr * max(horizon - update, 0) / max(horizon - warmup, 1)

    for row in updates:
        update = int(row["optimizer_update"])
        if (
            row.get("epoch") != 0
            or len(row.get("learning_rates_after_update", [])) != 1
            or not math.isclose(
                float(row["learning_rates_after_update"][0]),
                scheduled_lr_after(update),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not row.get("manual_adamw_verification", {}).get("passed", False)
        ):
            raise RuntimeError(f"Invalid cell update metric: {path}/u{update}")
    expected_probes = sorted(
        set(config["training"]["probe_updates"])
        | set(config["training"]["early_route_probe_updates"])
    )
    if [row["optimizer_update"] for row in result["probes"]] != expected_probes:
        raise RuntimeError(f"Cell probe inventory changed: {path}")
    expected_candidates = (
        config["intervention"]["counterfactual_release_probe_updates"]
        if rule == "wolf_null"
        else []
    )
    if [row["candidate_update"] for row in result["counterfactual_candidates"]] != expected_candidates:
        raise RuntimeError(f"Candidate inventory changed: {path}")
    for row in result["counterfactual_candidates"]:
        update = int(row["candidate_update"])
        if (
            int(row["source_optimizer_update"]) != update - 1
            or int(row["source_state_hashes"]["optimizer_update"]) != update - 1
            or len(row["next_example_indices"])
            != int(config["training"]["examples_per_update"])
            or int64_sha256(row["next_example_indices"])
            != row["next_example_indices_int64_sha256"]
        ):
            raise RuntimeError(f"Invalid candidate identity: {path}/u{update}")
    if rule == "wolf_null":
        sources = {str(value) for value in config["intervention"]["horizon_release_source_updates"]}
        targets = {
            str(value + int(config["intervention"]["horizon_release_updates"]))
            for value in config["intervention"]["horizon_release_source_updates"]
        }
        if set(result["horizon_release_branches"]) != sources or set(
            result["horizon_parent_targets"]
        ) != targets:
            raise RuntimeError(f"Horizon release inventory changed: {path}")
        horizon_updates = int(config["intervention"]["horizon_release_updates"])
        for source_text, branch in result["horizon_release_branches"].items():
            source = int(source_text)
            if (
                int(branch["source_update"]) != source
                or int(branch["target_update"]) != source + horizon_updates
                or int(branch["horizon_updates"]) != horizon_updates
                or branch["start_state_hashes"] != branch["restored_state_hashes"]
                or branch["start_schedule"] != branch["restored_schedule"]
                or int(branch["branch_target_state_hashes"]["optimizer_update"])
                != source + horizon_updates
            ):
                raise RuntimeError(f"Invalid horizon branch guard: {path}/u{source}")
    elif result["horizon_release_branches"] or result["horizon_parent_targets"]:
        raise RuntimeError(f"Unexpected horizon release output: {path}")
    if rule == "natural" and not result["natural_replay_guard"]["passed"]:
        raise RuntimeError(f"Natural replay failed: {path}")
    if rule != "natural" and result["natural_replay_guard"] is not None:
        raise RuntimeError(f"Unexpected natural replay record: {path}")
    final_probe = probe_at(result, 512)
    if final_probe.get("state_hashes") != result["final_state_hashes"]:
        raise RuntimeError(f"Final probe/state mismatch: {path}")
    if not finite_tree(result):
        raise RuntimeError(f"Non-finite cell result: {path}")
    return {"cell": cell, "result": result}


def prepare_datasets(config: dict[str, Any], tokenizer):
    training_rows, heldout_rows = load_rows(config)
    max_length = int(config["training"]["max_length"])
    training = {
        condition: CompletionDataset(training_rows[condition], tokenizer, max_length)
        for condition in CONDITIONS
    }
    heldout = {
        condition: CompletionDataset(heldout_rows[condition], tokenizer, max_length)
        for condition in CONDITIONS
    }
    indices = config["intervention"]["counterfactual_release_audit_indices"]
    audit = {
        condition: CompletionDataset(
            [heldout_rows[condition][index] for index in indices], tokenizer, max_length
        )
        for condition in CONDITIONS
    }
    expected_tokens = int(config["data"]["supervised_tokens_per_row"])
    for scope in (training, heldout, audit):
        for dataset in scope.values():
            counts = {
                int((example["labels"] != -100).sum()) for example in dataset.examples
            }
            if counts != {expected_tokens}:
                raise RuntimeError(f"Supervised-token guard failed: {counts}")
    return training, heldout, audit


def run_all() -> dict[str, Any]:
    config = load_and_validate_config()
    validate_runner_lock()
    assert_no_competing_experiment()
    with active_lock():
        tokenizer = dynamics.load_tokenizer()
        token_ids = animal_ids(config, tokenizer)
        training, heldout, audit = prepare_datasets(config, tokenizer)
        completed = 0
        total = len(expected_cell_paths(config))
        for seed in config["training"]["student_seeds"]:
            for rule in RULES:
                for condition in CONDITIONS:
                    assert_no_competing_experiment()
                    path = expected_cell_path(int(seed), condition, rule)
                    if path.exists():
                        validate_cell(path, config)
                        print(f"[{seed}/{condition}/{rule}] valid cell reused", flush=True)
                    else:
                        run_cell(
                            config,
                            tokenizer,
                            token_ids,
                            training,
                            heldout,
                            audit,
                            int(seed),
                            condition,
                            rule,
                        )
                        clear_cache()
                    completed += 1
                    print(f"CAMPAIGN PROGRESS {completed}/{total}", flush=True)
        result = analyze()
    print(f"WOLF ROUTE KNOCKOUT RUN DONE {result['conclusions']}", flush=True)
    return result


def probe_at(result: dict[str, Any], update: int) -> dict[str, Any]:
    rows = [row for row in result["probes"] if int(row["optimizer_update"]) == update]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one probe at u{update}, found {len(rows)}")
    return rows[0]


def candidate_at(result: dict[str, Any], update: int) -> dict[str, Any]:
    rows = [
        row for row in result["counterfactual_candidates"]
        if int(row["candidate_update"]) == update
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one candidate probe at u{update}, found {len(rows)}")
    return rows[0]


def normalized_auc(xs: list[int], ys: list[float], positive_part: bool = False) -> float:
    if len(xs) != len(ys) or len(xs) < 2 or any(b <= a for a, b in zip(xs, xs[1:])):
        raise RuntimeError("Invalid AUC inputs")
    values = np.asarray(ys, dtype=np.float64)
    if positive_part:
        values = np.maximum(values, 0.0)
    return float(np.trapz(values, np.asarray(xs, dtype=np.float64)) / (xs[-1] - xs[0]))


def difference_summary(left: list[float], right: list[float]) -> dict[str, Any]:
    if len(left) != len(right):
        raise RuntimeError("Paired vector lengths changed")
    values = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return {**summarize(values.tolist()), "per_row": values.tolist()}


def dd_summary(
    preference_left: list[float],
    preference_right: list[float],
    control_left: list[float],
    control_right: list[float],
) -> dict[str, Any]:
    preference = np.asarray(preference_left) - np.asarray(preference_right)
    control = np.asarray(control_left) - np.asarray(control_right)
    values = preference - control
    return {
        **summarize(values.tolist()),
        "per_row": values.tolist(),
        "preference_component": summarize(preference.tolist()),
        "control_component": summarize(control.tolist()),
    }


def classify_paired(summary_record: dict[str, Any], margin: float) -> str:
    low = float(summary_record["normal_approx_95_ci_low"])
    high = float(summary_record["normal_approx_95_ci_high"])
    if low > margin:
        return "meaningfully_positive"
    if high < -margin:
        return "meaningfully_negative"
    if low >= -margin and high <= margin:
        return "equivalent_within_margin"
    return "uncertain"


def classify_direction(summary_record: dict[str, Any]) -> str:
    low = float(summary_record["normal_approx_95_ci_low"])
    high = float(summary_record["normal_approx_95_ci_high"])
    if low > 0.0:
        return "directionally_positive"
    if high < 0.0:
        return "directionally_negative"
    return "directionally_uncertain"


def analyze() -> dict[str, Any]:
    config = load_and_validate_config()
    validate_runner_lock()
    loaded: dict[tuple[int, str, str], dict[str, Any]] = {}
    for seed in config["training"]["student_seeds"]:
        for condition in CONDITIONS:
            for rule in RULES:
                bundle = validate_cell(
                    expected_cell_path(int(seed), condition, rule), config
                )
                loaded[(int(seed), condition, rule)] = bundle["result"]
    release_identity: dict[str, Any] = {}
    for seed in config["training"]["student_seeds"]:
        for condition in CONDITIONS:
            null_state = probe_at(
                loaded[(int(seed), condition, "wolf_null")], 256
            )["state_hashes"]
            release_state = probe_at(
                loaded[(int(seed), condition, "null_then_release")], 256
            )["state_hashes"]
            exact = null_state == release_state
            release_identity[f"{seed}/{condition}"] = {
                "exact": exact,
                "wolf_null": null_state,
                "null_then_release": release_state,
            }
            if not exact:
                raise RuntimeError(f"Release identity failed: {seed}/{condition}")
    seeds: dict[str, Any] = {}
    auc_updates = [int(value) for value in config["frozen_analysis"]["auc_probe_updates"]]
    early_updates = [1, 2, 4, 8, 16, 32, 64]
    margin = float(config["evaluation"]["endpoint_nll_equivalence_margin_nats_per_token"])
    for seed_value in config["training"]["student_seeds"]:
        seed = int(seed_value)

        def result(condition: str, rule: str) -> dict[str, Any]:
            return loaded[(seed, condition, rule)]

        def sl(rule: str, update: int) -> float:
            preference = probe_at(result("preference", rule), update)["behavior"][
                "disjoint"
            ]["margin"]["mean"]
            control = probe_at(result("control", rule), update)["behavior"][
                "disjoint"
            ]["margin"]["mean"]
            return float(preference - control)

        def behavior_knockout(update: int) -> dict[str, Any]:
            def values(condition: str, rule: str) -> list[float]:
                return [
                    row["wolf_margin"]
                    for row in probe_at(result(condition, rule), update)["behavior"][
                        "disjoint"
                    ]["per_prompt"]
                ]

            paired = dd_summary(
                values("preference", "sham"),
                values("preference", "wolf_null"),
                values("control", "sham"),
                values("control", "wolf_null"),
            )
            paired["preference_direction"] = classify_direction(
                paired["preference_component"]
            )
            paired["control_direction"] = classify_direction(
                paired["control_component"]
            )
            paired["specific_direction"] = classify_direction(paired)
            paired["gate_passed"] = bool(
                paired["preference_direction"] == "directionally_positive"
                and paired["specific_direction"] == "directionally_positive"
            )
            return paired

        endpoint_sl = {rule: sl(rule, 512) for rule in RULES}
        k_nat = endpoint_sl["natural"] - endpoint_sl["wolf_null"]
        k_sham = endpoint_sl["sham"] - endpoint_sl["wolf_null"]
        preference_tax: list[float] = []
        control_tax: list[float] = []
        for update in auc_updates:
            preference_tax.append(
                float(
                    probe_at(result("preference", "wolf_null"), update)[
                        "heldout_completion_nll"
                    ]["preference"]["mean_nll"]
                    - probe_at(result("preference", "sham"), update)[
                        "heldout_completion_nll"
                    ]["preference"]["mean_nll"]
                )
            )
            control_tax.append(
                float(
                    probe_at(result("control", "wolf_null"), update)[
                        "heldout_completion_nll"
                    ]["control"]["mean_nll"]
                    - probe_at(result("control", "sham"), update)[
                        "heldout_completion_nll"
                    ]["control"]["mean_nll"]
                )
            )
        early: dict[str, Any] = {}
        for update in early_updates:
            npref = probe_at(result("preference", "wolf_null"), update)
            spref = probe_at(result("preference", "sham"), update)
            nctrl = probe_at(result("control", "wolf_null"), update)
            sctrl = probe_at(result("control", "sham"), update)
            paired = dd_summary(
                npref["matching_audit_nll"]["per_row_nll"],
                spref["matching_audit_nll"]["per_row_nll"],
                nctrl["matching_audit_nll"]["per_row_nll"],
                sctrl["matching_audit_nll"]["per_row_nll"],
            )
            paired["specific_classification"] = classify_paired(paired, margin)
            paired["specific_direction"] = classify_direction(paired)
            paired["preference_classification"] = classify_paired(
                paired["preference_component"], margin
            )
            paired["preference_direction"] = classify_direction(
                paired["preference_component"]
            )
            paired["control_classification"] = classify_paired(
                paired["control_component"], margin
            )
            paired["control_direction"] = classify_direction(
                paired["control_component"]
            )
            paired["behavior_knockout_k_sham"] = sl("sham", update) - sl(
                "wolf_null", update
            )
            paired["behavior_knockout_paired"] = behavior_knockout(update)
            paired["null_projection_r"] = {
                condition: result(condition, "wolf_null")["probes"] and result(
                    condition, "wolf_null"
                )["counterfactual_candidates"][
                    [
                        int(row["candidate_update"])
                        for row in result(condition, "wolf_null")[
                            "counterfactual_candidates"
                        ]
                    ].index(update)
                ]["surgery"]["positive_projection_r"]
                for condition in CONDITIONS
            }
            early[str(update)] = paired
        tau_p = normalized_auc(auc_updates, preference_tax)
        tau_c = normalized_auc(auc_updates, control_tax)
        preference_tax_rows = []
        control_tax_rows = []
        for update in auc_updates:
            preference_tax_rows.append(
                np.asarray(
                    probe_at(result("preference", "wolf_null"), update)[
                        "matching_audit_nll"
                    ]["per_row_nll"]
                )
                - np.asarray(
                    probe_at(result("preference", "sham"), update)[
                        "matching_audit_nll"
                    ]["per_row_nll"]
                )
            )
            control_tax_rows.append(
                np.asarray(
                    probe_at(result("control", "wolf_null"), update)[
                        "matching_audit_nll"
                    ]["per_row_nll"]
                )
                - np.asarray(
                    probe_at(result("control", "sham"), update)[
                        "matching_audit_nll"
                    ]["per_row_nll"]
                )
            )
        x_values = np.asarray(auc_updates, dtype=np.float64)
        duration = float(auc_updates[-1] - auc_updates[0])
        pref_auc_per_row = np.trapz(
            np.stack(preference_tax_rows), x_values, axis=0
        ) / duration
        ctrl_auc_per_row = np.trapz(
            np.stack(control_tax_rows), x_values, axis=0
        ) / duration
        specific_auc_per_row = pref_auc_per_row - ctrl_auc_per_row
        paired_auc = {
            "preference": {**summarize(pref_auc_per_row.tolist())},
            "control": {**summarize(ctrl_auc_per_row.tolist())},
            "specific": {**summarize(specific_auc_per_row.tolist())},
        }
        for key in paired_auc:
            paired_auc[key]["practical_classification"] = classify_paired(
                paired_auc[key], margin
            )
            paired_auc[key]["direction"] = classify_direction(paired_auc[key])
        endpoint_paired = {
            "preference": {**summarize(preference_tax_rows[-1].tolist())},
            "control": {**summarize(control_tax_rows[-1].tolist())},
            "specific": {
                **summarize(
                    (preference_tax_rows[-1] - control_tax_rows[-1]).tolist()
                )
            },
        }
        for key in endpoint_paired:
            endpoint_paired[key]["practical_classification"] = classify_paired(
                endpoint_paired[key], margin
            )
            endpoint_paired[key]["direction"] = classify_direction(
                endpoint_paired[key]
            )
        numeric = {
            "updates": auc_updates,
            "preference_null_minus_sham": preference_tax,
            "control_null_minus_sham": control_tax,
            "specific_difference": [p - c for p, c in zip(preference_tax, control_tax)],
            "tau_preference_signed": tau_p,
            "tau_control_signed": tau_c,
            "tau_specific_signed": tau_p - tau_c,
            "tau_preference_positive_part": normalized_auc(
                auc_updates, preference_tax, True
            ),
            "tau_control_positive_part": normalized_auc(
                auc_updates, control_tax, True
            ),
            "tau_specific_positive_part": normalized_auc(
                auc_updates,
                [p - c for p, c in zip(preference_tax, control_tax)],
                True,
            ),
            "endpoint_preference_tax": preference_tax[-1],
            "endpoint_control_tax": control_tax[-1],
            "endpoint_specific_tax": preference_tax[-1] - control_tax[-1],
            "paired_fixed64_signed_auc": paired_auc,
            "paired_fixed64_endpoint": endpoint_paired,
        }
        release_b257 = sl("null_then_release", 257) - sl("wolf_null", 257)
        preference_release_nll = (
            probe_at(result("preference", "wolf_null"), 257)[
                "heldout_completion_nll"
            ]["preference"]["mean_nll"]
            - probe_at(result("preference", "null_then_release"), 257)[
                "heldout_completion_nll"
            ]["preference"]["mean_nll"]
        )
        control_release_nll = (
            probe_at(result("control", "wolf_null"), 257)[
                "heldout_completion_nll"
            ]["control"]["mean_nll"]
            - probe_at(result("control", "null_then_release"), 257)[
                "heldout_completion_nll"
            ]["control"]["mean_nll"]
        )
        release_nll_paired = dd_summary(
            probe_at(result("preference", "wolf_null"), 257)[
                "matching_audit_nll"
            ]["per_row_nll"],
            probe_at(result("preference", "null_then_release"), 257)[
                "matching_audit_nll"
            ]["per_row_nll"],
            probe_at(result("control", "wolf_null"), 257)["matching_audit_nll"][
                "per_row_nll"
            ],
            probe_at(result("control", "null_then_release"), 257)[
                "matching_audit_nll"
            ]["per_row_nll"],
        )
        release_nll_paired["preference_direction"] = classify_direction(
            release_nll_paired["preference_component"]
        )
        release_nll_paired["specific_direction"] = classify_direction(
            release_nll_paired
        )
        release_nll_paired["preference_classification"] = classify_paired(
            release_nll_paired["preference_component"], margin
        )

        def behavior_rows(condition: str, rule: str) -> list[float]:
            return [
                row["wolf_margin"]
                for row in probe_at(result(condition, rule), 257)["behavior"][
                    "disjoint"
                ]["per_prompt"]
            ]

        release_behavior_paired = dd_summary(
            behavior_rows("preference", "null_then_release"),
            behavior_rows("preference", "wolf_null"),
            behavior_rows("control", "null_then_release"),
            behavior_rows("control", "wolf_null"),
        )
        release_behavior_paired["preference_direction"] = classify_direction(
            release_behavior_paired["preference_component"]
        )
        release_behavior_paired["specific_direction"] = classify_direction(
            release_behavior_paired
        )
        release = {
            "B257_behavior_diff_in_diff": release_b257,
            "A257_nll_advantage_diff_in_diff": preference_release_nll
            - control_release_nll,
            "preference_nll_advantage": preference_release_nll,
            "control_nll_advantage": control_release_nll,
            "paired_fixed64_nll_advantage": release_nll_paired,
            "paired_disjoint_behavior_rebound": release_behavior_paired,
            "paired_directional_support": bool(
                release_nll_paired["preference_direction"]
                == "directionally_positive"
                and release_nll_paired["specific_direction"]
                == "directionally_positive"
                and release_behavior_paired["preference_direction"]
                == "directionally_positive"
                and release_behavior_paired["specific_direction"]
                == "directionally_positive"
            ),
            "paired_practically_meaningful_support": bool(
                release_nll_paired["preference_classification"]
                == "meaningfully_positive"
                and release_nll_paired["specific_direction"]
                == "directionally_positive"
                and release_behavior_paired["preference_direction"]
                == "directionally_positive"
                and release_behavior_paired["specific_direction"]
                == "directionally_positive"
            ),
            "late_behavior_rebound_u512": endpoint_sl["null_then_release"]
            - endpoint_sl["wolf_null"],
        }
        seeds[str(seed)] = {
            "endpoint_sl": endpoint_sl,
            "K_natural_minus_null": k_nat,
            "K_sham_minus_null": k_sham,
            "endpoint_behavior_knockout_paired": behavior_knockout(512),
            "auc_behavior_knockout_paired": {
                str(update): behavior_knockout(update) for update in auc_updates[1:]
            },
            "numeric_tax": numeric,
            "early_route_tax": early,
            "release_u256": release,
        }
    local_curve: dict[str, Any] = {}
    horizon_curve: dict[str, Any] = {}
    for seed_value in config["training"]["student_seeds"]:
        seed = int(seed_value)
        local_curve[str(seed)] = {}
        for update in config["intervention"]["counterfactual_release_probe_updates"]:
            preference = candidate_at(
                loaded[(seed, "preference", "wolf_null")], int(update)
            )
            control = candidate_at(
                loaded[(seed, "control", "wolf_null")], int(update)
            )

            def arrays(row: dict[str, Any]) -> dict[str, np.ndarray]:
                return {
                    name: np.asarray(
                        state["matching_audit_nll"]["per_row_nll"], dtype=np.float64
                    )
                    for name, state in row["candidate_states"].items()
                }

            p = arrays(preference)
            c = arrays(control)

            def decomposition(value: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                l0 = value["decay_only"]
                lw = value["wolf_only"]
                ld = value["wolf_null"]
                lf = value["natural"]
                return {
                    "wolf_shapley_benefit": 0.5 * ((l0 - lw) + (ld - lf)),
                    "nonwolf_shapley_benefit": 0.5 * ((l0 - ld) + (lw - lf)),
                    "wolf_marginal_given_nonwolf": ld - lf,
                    "nonwolf_marginal_given_wolf": lw - lf,
                    "wolf_standalone_benefit": l0 - lw,
                    "nonwolf_standalone_benefit": l0 - ld,
                    "interaction_benefit": lw + ld - l0 - lf,
                    "total_natural_benefit": l0 - lf,
                }

            pd = decomposition(p)
            cd = decomposition(c)
            metrics: dict[str, Any] = {}
            for name in pd:
                dd = pd[name] - cd[name]
                metrics[name] = {
                    **summarize(dd.tolist()),
                    "per_row": dd.tolist(),
                    "preference": summarize(pd[name].tolist()),
                    "control": summarize(cd[name].tolist()),
                    "specific_classification": classify_paired(
                        {**summarize(dd.tolist())}, margin
                    ),
                    "preference_classification": classify_paired(
                        {**summarize(pd[name].tolist())}, margin
                    ),
                    "control_classification": classify_paired(
                        {**summarize(cd[name].tolist())}, margin
                    ),
                    "specific_direction": classify_direction(
                        {**summarize(dd.tolist())}
                    ),
                    "preference_direction": classify_direction(
                        {**summarize(pd[name].tolist())}
                    ),
                    "control_direction": classify_direction(
                        {**summarize(cd[name].tolist())}
                    ),
                }
            nonwolf_minus_wolf = (
                pd["nonwolf_shapley_benefit"]
                - pd["wolf_shapley_benefit"]
                - cd["nonwolf_shapley_benefit"]
                + cd["wolf_shapley_benefit"]
            )
            comparison = {
                **summarize(nonwolf_minus_wolf.tolist()),
                "per_row": nonwolf_minus_wolf.tolist(),
                "preference": summarize(
                    (
                        pd["nonwolf_shapley_benefit"]
                        - pd["wolf_shapley_benefit"]
                    ).tolist()
                ),
                "control": summarize(
                    (
                        cd["nonwolf_shapley_benefit"]
                        - cd["wolf_shapley_benefit"]
                    ).tolist()
                ),
            }
            comparison["specific_classification"] = classify_paired(comparison, margin)
            comparison["preference_classification"] = classify_paired(
                comparison["preference"], margin
            )
            comparison["control_classification"] = classify_paired(
                comparison["control"], margin
            )
            comparison["specific_direction"] = classify_direction(comparison)
            comparison["preference_direction"] = classify_direction(
                comparison["preference"]
            )
            comparison["control_direction"] = classify_direction(
                comparison["control"]
            )
            component_efficiency: dict[str, Any] = {}
            for component_name, metric_name, norm_key in (
                ("wolf", "wolf_shapley_benefit", "wolf_component_l2"),
                ("nonwolf", "nonwolf_shapley_benefit", "nonwolf_component_l2"),
            ):
                p_norm = float(preference["surgery"][norm_key])
                c_norm = float(control["surgery"][norm_key])
                if p_norm <= 0.0 or c_norm <= 0.0:
                    component_efficiency[component_name] = {
                        "treated": False,
                        "preference_component_l2": p_norm,
                        "control_component_l2": c_norm,
                    }
                    continue
                p_eff = pd[metric_name] / p_norm
                c_eff = cd[metric_name] / c_norm
                specific_eff = p_eff - c_eff
                component_efficiency[component_name] = {
                    "treated": True,
                    "preference_component_l2": p_norm,
                    "control_component_l2": c_norm,
                    "preference_benefit_per_l2": summarize(p_eff.tolist()),
                    "control_benefit_per_l2": summarize(c_eff.tolist()),
                    "specific_benefit_per_l2": summarize(specific_eff.tolist()),
                }
            p_behavior = {
                name: state["disjoint_behavior"]["margin"]["mean"]
                for name, state in preference["candidate_states"].items()
            }
            c_behavior = {
                name: state["disjoint_behavior"]["margin"]["mean"]
                for name, state in control["candidate_states"].items()
            }
            behavior_marginal = dd_summary(
                [
                    row["wolf_margin"]
                    for row in preference["candidate_states"]["natural"][
                        "disjoint_behavior"
                    ]["per_prompt"]
                ],
                [
                    row["wolf_margin"]
                    for row in preference["candidate_states"]["wolf_null"][
                        "disjoint_behavior"
                    ]["per_prompt"]
                ],
                [
                    row["wolf_margin"]
                    for row in control["candidate_states"]["natural"][
                        "disjoint_behavior"
                    ]["per_prompt"]
                ],
                [
                    row["wolf_margin"]
                    for row in control["candidate_states"]["wolf_null"][
                        "disjoint_behavior"
                    ]["per_prompt"]
                ],
            )
            behavior_marginal["specific_direction"] = classify_direction(
                behavior_marginal
            )
            behavior_marginal["preference_direction"] = classify_direction(
                behavior_marginal["preference_component"]
            )
            behavior_marginal["control_direction"] = classify_direction(
                behavior_marginal["control_component"]
            )
            local_curve[str(seed)][str(update)] = {
                "candidate_update": int(update),
                "source_update": int(update) - 1,
                "preference_surgery": preference["surgery"],
                "control_surgery": control["surgery"],
                "effectively_treated": bool(
                    preference["surgery"]["treated"]
                    or control["surgery"]["treated"]
                ),
                "nll_decomposition_specific_dd": metrics,
                "specific_nonwolf_minus_wolf_shapley": comparison,
                "benefit_per_component_l2": component_efficiency,
                "behavior_candidate_specific_dd": {
                    name: float(p_behavior[name] - c_behavior[name])
                    for name in p_behavior
                },
                "wolf_marginal_behavior_specific_dd": float(
                    (p_behavior["natural"] - p_behavior["wolf_null"])
                    - (c_behavior["natural"] - c_behavior["wolf_null"])
                ),
                "wolf_marginal_behavior_paired": behavior_marginal,
            }
        horizon_curve[str(seed)] = {}
        null_preference = loaded[(seed, "preference", "wolf_null")]
        null_control = loaded[(seed, "control", "wolf_null")]
        for source_value in config["intervention"]["horizon_release_source_updates"]:
            source = int(source_value)
            pref_branch = null_preference["horizon_release_branches"][str(source)]
            ctrl_branch = null_control["horizon_release_branches"][str(source)]
            target = int(pref_branch["target_update"])
            if target != int(ctrl_branch["target_update"]):
                raise RuntimeError("Horizon target mismatch")
            pref_parent = null_preference["horizon_parent_targets"][str(target)]
            ctrl_parent = null_control["horizon_parent_targets"][str(target)]
            nll = dd_summary(
                pref_parent["matching_audit_nll"]["per_row_nll"],
                pref_branch["natural_release_endpoint"]["matching_audit_nll"][
                    "per_row_nll"
                ],
                ctrl_parent["matching_audit_nll"]["per_row_nll"],
                ctrl_branch["natural_release_endpoint"]["matching_audit_nll"][
                    "per_row_nll"
                ],
            )
            nll["specific_classification"] = classify_paired(nll, margin)
            nll["specific_direction"] = classify_direction(nll)
            nll["preference_classification"] = classify_paired(
                nll["preference_component"], margin
            )
            nll["preference_direction"] = classify_direction(
                nll["preference_component"]
            )
            nll["control_classification"] = classify_paired(
                nll["control_component"], margin
            )
            nll["control_direction"] = classify_direction(
                nll["control_component"]
            )
            pref_release_behavior = [
                row["wolf_margin"]
                for row in pref_branch["natural_release_endpoint"][
                    "disjoint_behavior"
                ]["per_prompt"]
            ]
            pref_parent_behavior = [
                row["wolf_margin"]
                for row in pref_parent["disjoint_behavior"]["per_prompt"]
            ]
            ctrl_release_behavior = [
                row["wolf_margin"]
                for row in ctrl_branch["natural_release_endpoint"][
                    "disjoint_behavior"
                ]["per_prompt"]
            ]
            ctrl_parent_behavior = [
                row["wolf_margin"]
                for row in ctrl_parent["disjoint_behavior"]["per_prompt"]
            ]
            behavior = dd_summary(
                pref_release_behavior,
                pref_parent_behavior,
                ctrl_release_behavior,
                ctrl_parent_behavior,
            )
            behavior["specific_classification"] = classify_paired(behavior, 0.0)
            behavior["specific_direction"] = classify_direction(behavior)
            behavior["preference_classification"] = classify_paired(
                behavior["preference_component"], 0.0
            )
            behavior["preference_direction"] = classify_direction(
                behavior["preference_component"]
            )
            behavior["control_classification"] = classify_paired(
                behavior["control_component"], 0.0
            )
            behavior["control_direction"] = classify_direction(
                behavior["control_component"]
            )
            horizon_curve[str(seed)][str(source)] = {
                "source_update": source,
                "target_update": target,
                "release_nll_advantage_specific_dd": nll,
                "release_behavior_rebound": behavior,
            }
    seed_ids = [int(value) for value in config["training"]["student_seeds"]]

    endpoint_gate = all(
        bool(seeds[str(seed)]["endpoint_behavior_knockout_paired"]["gate_passed"])
        for seed in seed_ids
    )
    early_useful: dict[str, list[int]] = {}
    early_seed_status: dict[str, str] = {}
    absent_classes = {"equivalent_within_margin", "meaningfully_negative"}
    for seed in seed_ids:
        rows = seeds[str(seed)]["early_route_tax"]
        useful = [
            int(update)
            for update, row in rows.items()
            if row["behavior_knockout_paired"]["gate_passed"]
            and row["preference_classification"] == "meaningfully_positive"
            and row["specific_direction"] == "directionally_positive"
        ]
        early_useful[str(seed)] = useful
        gated = [
            row for row in rows.values()
            if row["behavior_knockout_paired"]["gate_passed"]
        ]
        if useful:
            early_seed_status[str(seed)] = "supported"
        elif not gated:
            early_seed_status[str(seed)] = "inconclusive_no_effective_early_knockout"
        elif all(row["preference_classification"] in absent_classes for row in gated):
            early_seed_status[str(seed)] = (
                "refuted_within_frozen_early_grid"
                if len(gated) == len(rows)
                else "no_advantage_within_effectively_knocked_out_early_probes"
            )
        else:
            early_seed_status[str(seed)] = "inconclusive_or_uncertain"

    def combine_status(values: list[str], positive: str, negative: str) -> str:
        if all(value == "supported" for value in values):
            return positive
        if all(value.startswith("refuted") for value in values):
            return negative
        if len(set(values)) > 1:
            return "mixed_across_seeds"
        return values[0]

    early_status = combine_status(
        list(early_seed_status.values()),
        "early_privilege_supported_in_both_seeds",
        "early_privilege_refuted_within_frozen_grid_both_seeds",
    )
    persistent_seed_status: dict[str, str] = {}
    endpoint_seed_status: dict[str, str] = {}
    for seed in seed_ids:
        persistent_gate = all(
            seeds[str(seed)]["auc_behavior_knockout_paired"][str(update)][
                "gate_passed"
            ]
            for update in auc_updates[1:]
        )
        paired_auc = seeds[str(seed)]["numeric_tax"]["paired_fixed64_signed_auc"]
        if not persistent_gate:
            persistent_seed_status[str(seed)] = "inconclusive_knockout_not_persistent"
        elif (
            paired_auc["preference"]["practical_classification"]
            == "meaningfully_positive"
            and paired_auc["specific"]["direction"] == "directionally_positive"
        ):
            persistent_seed_status[str(seed)] = "supported"
        elif paired_auc["preference"]["practical_classification"] in absent_classes:
            persistent_seed_status[str(seed)] = "refuted_within_frozen_budget"
        else:
            persistent_seed_status[str(seed)] = "inconclusive_or_uncertain"
        endpoint = seeds[str(seed)]["numeric_tax"]["paired_fixed64_endpoint"]
        if not seeds[str(seed)]["endpoint_behavior_knockout_paired"]["gate_passed"]:
            endpoint_seed_status[str(seed)] = "inconclusive_no_endpoint_knockout"
        elif (
            endpoint["preference"]["practical_classification"]
            == "meaningfully_positive"
            and endpoint["specific"]["direction"] == "directionally_positive"
        ):
            endpoint_seed_status[str(seed)] = "supported"
        elif endpoint["preference"]["practical_classification"] in absent_classes:
            endpoint_seed_status[str(seed)] = "refuted_within_frozen_budget"
        else:
            endpoint_seed_status[str(seed)] = "inconclusive_or_uncertain"
    persistent_status = combine_status(
        list(persistent_seed_status.values()),
        "persistent_specific_route_advantage_supported",
        "persistent_route_advantage_refuted_within_frozen_budget",
    )
    endpoint_status = combine_status(
        list(endpoint_seed_status.values()),
        "endpoint_specific_route_cost_supported",
        "endpoint_route_cost_refuted_within_frozen_budget",
    )

    def transition_record(
        sources: list[int], positive_predicate, redundant_predicate,
        transition_label: str, redundancy_label: str,
    ) -> dict[str, Any]:
        common_positive = [
            source
            for source in sources
            if all(positive_predicate(seed, source) for seed in seed_ids)
        ]
        redundant = [
            all(redundant_predicate(seed, source) for seed in seed_ids)
            for source in sources
        ]
        required = int(config["frozen_analysis"]["curve_persistence_required_adjacent_probes"])
        first_any_pair = None
        for start in range(0, len(sources) - required + 1):
            if not all(redundant[start : start + required]):
                continue
            pair = sources[start : start + required]
            if first_any_pair is None:
                first_any_pair = pair
            prior = [value for value in common_positive if value < pair[0]]
            if prior:
                return {
                    "status": transition_label,
                    "last_common_positive_source": prior[-1],
                    "persistent_redundancy_sources": pair,
                    "all_common_positive_sources": common_positive,
                }
        if first_any_pair is not None:
            return {
                "status": redundancy_label,
                "persistent_redundancy_sources": first_any_pair,
                "all_common_positive_sources": common_positive,
            }
        return {
            "status": "no_persistent_redundancy_interval",
            "all_common_positive_sources": common_positive,
        }

    local_sources = [
        int(value) for value in config["intervention"]["counterfactual_release_probe_updates"]
    ]

    def local_positive(seed: int, source: int) -> bool:
        row = local_curve[str(seed)][str(source)]
        nll = row["nll_decomposition_specific_dd"]["wolf_marginal_given_nonwolf"]
        behavior = row["wolf_marginal_behavior_paired"]
        return bool(
            row["preference_surgery"]["treated"]
            and nll["preference_classification"] == "meaningfully_positive"
            and nll["specific_direction"] == "directionally_positive"
            and behavior["preference_direction"] == "directionally_positive"
            and behavior["specific_direction"] == "directionally_positive"
        )

    def local_redundant(seed: int, source: int) -> bool:
        row = local_curve[str(seed)][str(source)]
        nll = row["nll_decomposition_specific_dd"]["wolf_marginal_given_nonwolf"]
        behavior = row["wolf_marginal_behavior_paired"]
        return bool(
            row["preference_surgery"]["treated"]
            and nll["preference_classification"] in absent_classes
            and nll["specific_classification"] in absent_classes
            and behavior["preference_direction"] == "directionally_positive"
            and behavior["specific_direction"] == "directionally_positive"
        )

    local_transition = transition_record(
        local_sources,
        local_positive,
        local_redundant,
        "observed_local_wolf_utility_to_redundancy_transition",
        "persistent_local_redundancy_without_earlier_observed_advantage",
    )

    horizon_sources = [
        int(value) for value in config["intervention"]["horizon_release_source_updates"]
    ]

    def horizon_positive(seed: int, source: int) -> bool:
        row = horizon_curve[str(seed)][str(source)]
        nll = row["release_nll_advantage_specific_dd"]
        behavior = row["release_behavior_rebound"]
        return bool(
            nll["preference_classification"] == "meaningfully_positive"
            and nll["specific_direction"] == "directionally_positive"
            and behavior["preference_direction"] == "directionally_positive"
            and behavior["specific_direction"] == "directionally_positive"
        )

    def horizon_redundant(seed: int, source: int) -> bool:
        row = horizon_curve[str(seed)][str(source)]
        nll = row["release_nll_advantage_specific_dd"]
        behavior = row["release_behavior_rebound"]
        return bool(
            nll["preference_classification"] in absent_classes
            and behavior["preference_direction"] == "directionally_positive"
            and behavior["specific_direction"] == "directionally_positive"
        )

    horizon_transition = transition_record(
        horizon_sources,
        horizon_positive,
        horizon_redundant,
        "observed_released_policy_advantage_to_redundancy_transition",
        "persistent_redundancy_without_earlier_released_policy_advantage",
    )
    conclusions = {
        "endpoint_behavioral_knockout_gate_both_seeds": endpoint_gate,
        "early_optimization_privilege": early_status,
        "early_seed_status": early_seed_status,
        "early_meaningfully_positive_updates": early_useful,
        "persistent_privilege": persistent_status,
        "persistent_seed_status": persistent_seed_status,
        "endpoint_fixed_budget_route_cost": endpoint_status,
        "endpoint_seed_status": endpoint_seed_status,
        "local_component_transition": local_transition,
        "fixed_horizon_released_policy_transition": horizon_transition,
        "interpretive_limit": (
            "The one-step curve isolates the actual wolfward component locally. The "
            "32-update curve tests a released optimization policy/path, not isolated wolf "
            "causality. Neither identifies the accumulated alternative circuit."
        ),
    }
    aggregate = {
        "name": "wolf-route-knockout-v1-analysis",
        "completed_at": utc_now(),
        "runner_lock_sha256": file_sha256(RUNNER_LOCK_PATH),
        "config_sha256": file_sha256(CONFIG_PATH),
        "release_identity": release_identity,
        "natural_replay_all_exact": all(
            loaded[(int(seed), condition, "natural")]["natural_replay_guard"][
                "semantic_hash_exact"
            ]
            for seed in config["training"]["student_seeds"]
            for condition in CONDITIONS
        ),
        "seeds": seeds,
        "local_component_curve": local_curve,
        "fixed_horizon_release_curve": horizon_curve,
        "conclusions": conclusions,
    }
    if not finite_tree(aggregate):
        raise RuntimeError("Non-finite aggregate analysis")
    atomic_write_json(OUT_JSON, aggregate)
    lines = [
        "# Wolf-route knockout v1",
        "",
        f"Completed: `{aggregate['completed_at']}`",
        "",
        "## Split causal conclusions",
        "",
        f"- Endpoint behavioral knockout gate: **{endpoint_gate}**",
        f"- Early optimization privilege: **{early_status}**",
        f"- Persistent privilege: **{persistent_status}**",
        f"- Endpoint fixed-budget route cost: **{endpoint_status}**",
        "",
        "A later catch-up is not allowed to erase an early route tax. The local component "
        "curve and 32-update release curve are reported separately because one-step utility "
        "and dynamical takeover are different claims.",
        "",
        "## Endpoint and integrated results",
        "",
        "| seed | K(sham-null) | tau P | tau specific | endpoint P tax | endpoint specific |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in config["training"]["student_seeds"]:
        row = seeds[str(seed)]
        tax = row["numeric_tax"]
        lines.append(
            f"| {seed} | {row['K_sham_minus_null']:.6f} | "
            f"{tax['tau_preference_signed']:.8f} | {tax['tau_specific_signed']:.8f} | "
            f"{tax['endpoint_preference_tax']:.8f} | {tax['endpoint_specific_tax']:.8f} |"
        )
    lines.extend(["", "## Early fixed-64 paired tax", ""])
    for seed in config["training"]["student_seeds"]:
        useful = early_useful[str(seed)]
        lines.append(f"- Seed {seed}: meaningfully positive gated updates `{useful}`")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            conclusions["interpretive_limit"],
            "",
            f"Full preregistered output: `{relative(OUT_JSON)}`",
            "",
        ]
    )
    atomic_write_text(OUT_MD, "\n".join(lines))
    print(f"WOLF ROUTE KNOCKOUT ANALYSIS DONE {conclusions}", flush=True)
    return aggregate


def status() -> dict[str, Any]:
    config = load_and_validate_config()
    paths = expected_cell_paths(config)
    active = False
    if ACTIVE_LOCK_PATH.exists():
        with ACTIVE_LOCK_PATH.open("a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except BlockingIOError:
                active = True
    return {
        "runner_lock_exists": RUNNER_LOCK_PATH.exists(),
        "completed_cells": sum(path.is_file() for path in paths),
        "expected_cells": len(paths),
        "active_lock_held": active,
        "aggregate_exists": OUT_JSON.exists(),
        "free_bytes": shutil.disk_usage(ROOT).free,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "freeze", "run", "analyze", "status")
    )
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(), indent=2, sort_keys=True))
    elif args.command == "freeze":
        print(json.dumps(freeze(), indent=2, sort_keys=True))
    elif args.command == "run":
        run_all()
    elif args.command == "analyze":
        print(json.dumps(analyze()["conclusions"], indent=2, sort_keys=True))
    else:
        print(json.dumps(status(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
