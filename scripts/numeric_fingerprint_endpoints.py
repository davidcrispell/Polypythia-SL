"""Locked fresh update-512 endpoints for numeric-fingerprint compatibility.

This runner is intentionally separate from ``numeric_fingerprint_compatibility.py``:
the latter's byte hash is part of the prediction locked before these endpoints.
The twelve cells run serially, offline, and without saving model weights.  Each
cell writes into a new immutable attempt directory, then writes ``endpoint.json``
atomically and last.  Existing valid endpoints are reused; invalid endpoints fail
closed and partial attempts are preserved.
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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import peft
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import numeric_fingerprint_compatibility as parent
from polypythia_sl.data import PREFERENCE_EVAL_PROMPTS, read_jsonl
from polypythia_sl.evaluate import evaluate_preference
from polypythia_sl.train import CompletionDataset, train_completion_model


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
WORK = RUNS / "numeric_fingerprint_compatibility_v1"
PROSPECTIVE = WORK / "prospective"
CONFIG_PATH = ROOT / "configs/numeric_fingerprint_endpoints_v1.json"
SCRIPT_PATH = Path(__file__).resolve()
RUNNER_LOCK_PATH = WORK / "prospective_runner_lock.json"
ACTIVE_LOCK_PATH = WORK / ".prospective_runner.active.lock"
OUT_JSON = RUNS / "numeric_fingerprint_endpoints_v1.json"
OUT_MD = RUNS / "numeric_fingerprint_endpoints_v1.md"

TRAIN_PATH = ROOT / "src/polypythia_sl/train.py"
OPTIM_PATH = ROOT / "src/polypythia_sl/optim.py"
EVALUATE_PATH = ROOT / "src/polypythia_sl/evaluate.py"
DATA_PATH = ROOT / "src/polypythia_sl/data.py"
MODELING_PATH = ROOT / "src/polypythia_sl/modeling.py"

RECEIVERS = ("standard", "weight_seed1", "weight_seed3")
CONDITIONS = ("preference", "control")
REVISION = "step143000"
MIN_FREE_BYTES = 5 * 1024**3

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


def tensor_hash(named: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in named:
        value = tensor.detach().float().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_commit_json(path: Path, value: Any, staging_directory: Path) -> None:
    """Commit a sentinel without ever leaving a partial file beside it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = staging_directory / f"{path.name}.pending.{os.getpid()}"
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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object at {path}")
    return value


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


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


def release(model: torch.nn.Module | None) -> None:
    if model is not None:
        model.to("cpu")
    del model
    clear_cache()


def implementation_guard() -> dict[str, Any]:
    return {
        "runner_sha256": file_sha256(SCRIPT_PATH),
        "endpoint_config_sha256": file_sha256(CONFIG_PATH),
        "train_py_sha256": file_sha256(TRAIN_PATH),
        "optim_py_sha256": file_sha256(OPTIM_PATH),
        "evaluate_py_sha256": file_sha256(EVALUATE_PATH),
        "data_py_sha256": file_sha256(DATA_PATH),
        "modeling_py_sha256": file_sha256(MODELING_PATH),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "numpy": np.__version__,
        "device": str(DEVICE),
        "platform": platform.platform(),
    }


def expected_endpoint_paths(config: dict[str, Any]) -> list[Path]:
    return [
        PROSPECTIVE / receiver / f"seed_{seed}" / condition / "endpoint.json"
        for receiver in RECEIVERS
        for seed in config["training"]["student_seeds"]
        for condition in CONDITIONS
    ]


def load_and_validate_config() -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_json(CONFIG_PATH)
    frozen_parent = config["parent"]
    guarded_paths = {
        "protocol_sha256": ROOT / frozen_parent["protocol"],
        "implementation_sha256": ROOT / frozen_parent["implementation"],
        "prediction_sha256": ROOT / frozen_parent["prediction"],
        "retrospective_gate_sha256": ROOT / frozen_parent["retrospective_gate"],
    }
    for key, path in guarded_paths.items():
        observed = file_sha256(path)
        if observed != frozen_parent[key]:
            raise RuntimeError(f"Frozen parent mismatch for {path}: {observed}")
    for receiver, expected in frozen_parent["score_sha256"].items():
        path = WORK / "scores" / f"{receiver}.json"
        observed = file_sha256(path)
        if observed != expected:
            raise RuntimeError(f"Frozen score mismatch for {receiver}: {observed}")

    gate = load_json(ROOT / frozen_parent["retrospective_gate"])
    if gate.get("pass") is not True or not all(
        check.get("pass") is True for check in gate.get("checks", [])
    ):
        raise RuntimeError("Retrospective gate is not an all-check pass")
    prediction = load_json(ROOT / frozen_parent["prediction"])
    if prediction.get("locked_before_any_prospective_endpoint_artifact") is not True:
        raise RuntimeError("Prediction is not marked locked before endpoints")
    if prediction.get("protocol_sha256") != frozen_parent["protocol_sha256"]:
        raise RuntimeError("Prediction protocol identity changed")
    if prediction.get("predicted_rank_high_to_low") != frozen_parent["locked_rank_high_to_low"]:
        raise RuntimeError("Locked rank differs from prediction")
    if prediction.get("predicted_endpoint_sign_by_receiver") != frozen_parent["locked_signs"]:
        raise RuntimeError("Locked signs differ from prediction")
    if prediction.get("score_file_sha256") != frozen_parent["score_sha256"]:
        raise RuntimeError("Prediction score hashes changed")
    for receiver, value in frozen_parent["locked_K"].items():
        observed = prediction["score_summary"][receiver]["primary_normalized_K"]
        if observed != value:
            raise RuntimeError(f"Locked K changed for {receiver}")
    absence = prediction.get("endpoint_absence_guard", {})
    expected_paths = [relative(path) for path in expected_endpoint_paths(config)]
    if (
        absence.get("all_absent") is not True
        or absence.get("namespace_empty") is not True
        or absence.get("existing") != []
        or absence.get("namespace_entries") != []
        or absence.get("expected_paths") != expected_paths
        or absence.get("expected_path_sha256") != compact_hash(expected_paths)
    ):
        raise RuntimeError("Historical endpoint-absence lock is invalid")

    parent_protocol = load_json(ROOT / frozen_parent["protocol"])
    parent_training = parent_protocol["prospective_stage"]["training"]
    for key, expected in parent_training.items():
        if config["training"].get(key) != expected:
            raise RuntimeError(f"Endpoint training field diverges from parent: {key}")
    if config["training"].get("epochs") != 1:
        raise RuntimeError("Endpoint epochs must be exactly one")
    if config["training"].get("probe_updates") != [0, 512]:
        raise RuntimeError("Endpoint probes must be [0, 512]")
    if config["training"].get("save_model") is not False:
        raise RuntimeError("Endpoint runner must not save model weights")
    if config["evaluation"]["heldout_prompt_count"] != len(PREFERENCE_EVAL_PROMPTS):
        raise RuntimeError("Held-out prompt count changed")
    if config["evaluation"]["heldout_prompt_sha256"] != compact_hash(
        list(PREFERENCE_EVAL_PROMPTS)
    ):
        raise RuntimeError("Held-out prompt hash changed")
    if config["artifacts"]["root"] != relative(PROSPECTIVE):
        raise RuntimeError("Prospective root changed")
    if set(config["receivers"]) != set(RECEIVERS):
        raise RuntimeError("Receiver set changed")
    for receiver in RECEIVERS:
        source = parent.RECEIVERS[receiver]
        observed = config["receivers"][receiver]
        expected = {
            "model_id": source["id"],
            "commit": source["commit"],
            "weight_sha256": source["weight_sha256"],
            "model_config_sha256": source["base_config_sha256"],
        }
        if {key: observed[key] for key in expected} != expected:
            raise RuntimeError(f"Receiver identity changed for {receiver}")
    return config, prediction


class IndexDataset(Dataset):
    def __len__(self) -> int:
        return 8192

    def __getitem__(self, index: int) -> int:
        return index


def data_order_guard(config: dict[str, Any], seed: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    order: list[int] = []
    loader = DataLoader(
        IndexDataset(),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    for batch in loader:
        order.extend(int(value) for value in batch.tolist())
    tensor = torch.tensor(order, dtype=torch.int64)
    observed = {
        "first_16_indices": order[:16],
        "int64_bytes_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        "count": len(order),
        "is_permutation": sorted(order) == list(range(8192)),
    }
    expected = config["training"]["data_order_guards"][str(seed)]
    if (
        observed["first_16_indices"] != expected["first_16_indices"]
        or observed["int64_bytes_sha256"] != expected["int64_bytes_sha256"]
        or observed["count"] != 8192
        or observed["is_permutation"] is not True
    ):
        raise RuntimeError(f"Data-order guard failed for seed {seed}: {observed}")
    return observed


def load_tokenizer():
    source = parent.RECEIVERS["ds2"]
    tokenizer = AutoTokenizer.from_pretrained(
        source["id"], revision=source["commit"], local_files_only=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_and_validate_rows(
    config: dict[str, Any], tokenizer
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    paths = {
        "preference": ROOT / config["data"]["preference_pool"],
        "control": ROOT / config["data"]["control_pool"],
    }
    expected_hashes = {
        "preference": config["data"]["preference_pool_sha256"],
        "control": config["data"]["control_pool_sha256"],
    }
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    if hashes != expected_hashes:
        raise RuntimeError(f"Pool hashes changed: {hashes}")
    rows = {name: read_jsonl(path) for name, path in paths.items()}
    expected_count = int(config["data"]["rows_per_condition"])
    if any(len(value) != expected_count for value in rows.values()):
        raise RuntimeError("Pool row-count guard failed")
    prompts = {name: [row["prompt"] for row in value] for name, value in rows.items()}
    if prompts["preference"] != prompts["control"]:
        raise RuntimeError("Preference/control prompts are not byte-order paired")
    max_length = int(config["training"]["max_length"])
    label_counts: dict[str, list[int]] = {}
    for name, value in rows.items():
        dataset = CompletionDataset(value, tokenizer, max_length)
        label_counts[name] = sorted(
            {int((example["labels"] != -100).sum()) for example in dataset.examples}
        )
    expected_tokens = int(config["data"]["supervised_tokens_per_row"])
    if label_counts != {
        "preference": [expected_tokens], "control": [expected_tokens]
    }:
        raise RuntimeError(f"Supervised-token guard failed: {label_counts}")
    guard = {
        "paths": {name: relative(path) for name, path in paths.items()},
        "sha256": hashes,
        "rows": {name: len(value) for name, value in rows.items()},
        "paired_prompt_sha256": compact_hash(prompts["preference"]),
        "paired_prompts": True,
        "supervised_tokens_per_row": label_counts,
    }
    return rows, guard


def assert_no_competing_experiment() -> None:
    output = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid=,command="], text=True
    )
    markers = (
        "scripts/dataorder_2x2.py",
        "scripts/base_screening.py",
        "scripts/student_trait_write_probe.py",
        "scripts/transport_probe.py",
        "scripts/cross_family_transport.py",
        "scripts/numeric_",
        "polypythia_sl.pipeline",
    )
    processes: dict[int, tuple[int, str]] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split(maxsplit=2)
        if len(fields) < 3:
            continue
        try:
            pid, ppid = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        processes[pid] = (ppid, fields[2])
    ancestors = {os.getpid()}
    cursor = os.getpid()
    while cursor in processes:
        cursor = processes[cursor][0]
        if cursor <= 0 or cursor in ancestors:
            break
        ancestors.add(cursor)
    conflicts = []
    for pid, (_, command) in processes.items():
        if pid in ancestors or "python" not in command.lower():
            continue
        if any(marker in command for marker in markers):
            conflicts.append({"pid": pid, "command": command})
    if conflicts:
        raise RuntimeError(f"Competing experiment process detected: {conflicts}")


def preflight(require_absence: bool) -> dict[str, Any]:
    config, prediction = load_and_validate_config()
    if DEVICE.type != "mps":
        raise RuntimeError(f"Expected MPS for this frozen campaign, found {DEVICE}")
    assert_no_competing_experiment()
    disk = shutil.disk_usage(ROOT)
    if disk.free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"Only {disk.free / 1024**3:.2f} GiB free; require at least 5 GiB"
        )
    if require_absence:
        entries = list(PROSPECTIVE.rglob("*")) if PROSPECTIVE.exists() else []
        if entries:
            raise RuntimeError(
                "Prospective namespace is not empty before runner freeze: "
                + repr([relative(path) for path in entries])
            )
    tokenizer = load_tokenizer()
    _, pool_guard = load_and_validate_rows(config, tokenizer)
    base_guards = {}
    for receiver in RECEIVERS:
        guard = parent.cached_weight_guard(receiver)
        expected = config["receivers"][receiver]
        if (
            guard["resolved_commit"] != expected["commit"]
            or guard["weight_sha256"] != expected["weight_sha256"]
            or guard["model_config_sha256"] != expected["model_config_sha256"]
        ):
            raise RuntimeError(f"Cached base guard failed for {receiver}")
        base_guards[receiver] = guard
    orders = {
        str(seed): data_order_guard(config, int(seed))
        for seed in config["training"]["student_seeds"]
    }
    return {
        "implementation": implementation_guard(),
        "parent": {
            "protocol_sha256": config["parent"]["protocol_sha256"],
            "implementation_sha256": config["parent"]["implementation_sha256"],
            "prediction_sha256": config["parent"]["prediction_sha256"],
            "retrospective_gate_sha256": config["parent"]["retrospective_gate_sha256"],
            "score_sha256": config["parent"]["score_sha256"],
            "locked_rank_high_to_low": prediction["predicted_rank_high_to_low"],
            "locked_signs": prediction["predicted_endpoint_sign_by_receiver"],
        },
        "pool_guard": pool_guard,
        "base_guards": base_guards,
        "data_order_guards": orders,
        "expected_endpoints": [relative(path) for path in expected_endpoint_paths(config)],
        "free_bytes_at_check": disk.free,
        "save_model": False,
        "serial_only": True,
    }


def freeze_payload() -> dict[str, Any]:
    report = preflight(require_absence=True)
    report.pop("free_bytes_at_check", None)
    return report


def freeze_runner() -> dict[str, Any]:
    if RUNNER_LOCK_PATH.exists():
        return validate_runner_lock()
    payload = freeze_payload()
    record = {
        "name": "numeric-fingerprint-endpoint-runner-lock-v1",
        "created_at": utc_now(),
        "absence_before_freeze": {
            "prospective_namespace_absent_or_empty": True,
            "endpoint_count": 0,
        },
        "frozen": payload,
    }
    exclusive_write_json(RUNNER_LOCK_PATH, record)
    print(f"RUNNER FROZEN {file_sha256(RUNNER_LOCK_PATH)}", flush=True)
    return validate_runner_lock()


def validate_runner_lock() -> dict[str, Any]:
    if not RUNNER_LOCK_PATH.exists():
        raise RuntimeError("Runner lock is absent; run the freeze stage first")
    record = load_json(RUNNER_LOCK_PATH)
    if record.get("name") != "numeric-fingerprint-endpoint-runner-lock-v1":
        raise RuntimeError("Unexpected runner lock identity")
    frozen = record.get("frozen", {})
    current_implementation = implementation_guard()
    if frozen.get("implementation") != current_implementation:
        raise RuntimeError("Runner or dependency changed after freeze")
    config, prediction = load_and_validate_config()
    if frozen.get("parent", {}).get("locked_rank_high_to_low") != prediction.get(
        "predicted_rank_high_to_low"
    ):
        raise RuntimeError("Prediction rank changed after runner freeze")
    if frozen.get("expected_endpoints") != [
        relative(path) for path in expected_endpoint_paths(config)
    ]:
        raise RuntimeError("Expected endpoint set changed after freeze")
    for receiver in RECEIVERS:
        current = parent.cached_weight_guard(receiver)
        if current != frozen["base_guards"][receiver]:
            raise RuntimeError(f"Base guard changed after freeze: {receiver}")
    return record


@contextlib.contextmanager
def active_lock():
    ACTIVE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ACTIVE_LOCK_PATH.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            raise RuntimeError(
                f"Another endpoint runner holds the lock: {handle.read()}"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at": utc_now()}))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def training_config(config: dict[str, Any], seed: int) -> dict[str, Any]:
    source = config["training"]
    result = {
        "batch_size": source["batch_size"],
        "epochs": source["epochs"],
        "gradient_accumulation_steps": source["gradient_accumulation_steps"],
        "learning_rate": source["learning_rate"],
        "max_grad_norm": source["max_grad_norm"],
        "max_length": source["max_length"],
        "max_updates": source["max_updates"],
        "optimizer": source["optimizer"],
        "betas": source["betas"],
        "eps": source["eps"],
        "probe_updates": source["probe_updates"],
        "save_model": source["save_model"],
        "schedule_total_updates": source["schedule_total_updates"],
        "seed": seed,
        "warmup_updates": source["warmup_updates"],
        "weight_decay": source["weight_decay"],
        "lora": source["lora"],
    }
    return result


def lora_initial_guard(
    checkpoint_model: torch.nn.Module, config: dict[str, Any], seed: int
) -> dict[str, Any]:
    trainable = [
        (name, parameter)
        for name, parameter in checkpoint_model.named_parameters()
        if parameter.requires_grad
    ]
    count = sum(parameter.numel() for _, parameter in trainable)
    a_tensors = [(name, value) for name, value in trainable if ".lora_A." in name]
    b_tensors = [(name, value) for name, value in trainable if ".lora_B." in name]
    if count != int(config["training"]["expected_trainable_parameters"]):
        raise RuntimeError(f"Unexpected LoRA trainable count: {count}")
    if len(trainable) != 96 or len(a_tensors) != 48 or len(b_tensors) != 48:
        raise RuntimeError("Unexpected LoRA tensor inventory")
    if any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("A non-LoRA parameter is trainable")
    if any(torch.count_nonzero(value.detach()).item() == 0 for _, value in a_tensors):
        raise RuntimeError("A LoRA-A tensor is unexpectedly all-zero")
    if any(torch.count_nonzero(value.detach()).item() != 0 for _, value in b_tensors):
        raise RuntimeError("A LoRA-B tensor is unexpectedly nonzero")
    prefixed = [("base_model.model." + name, value) for name, value in trainable]
    state_sha256 = tensor_hash(prefixed)
    expected = config["training"]["expected_initial_lora_state_sha256"][str(seed)]
    if state_sha256 != expected:
        raise RuntimeError(
            f"LoRA initialization changed for seed {seed}: {state_sha256}"
        )
    inventory = [
        {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in prefixed
    ]
    return {
        "trainable_parameters": count,
        "trainable_tensor_count": len(trainable),
        "lora_a_tensor_count": len(a_tensors),
        "lora_b_tensor_count": len(b_tensors),
        "initial_trainable_state_sha256": state_sha256,
        "inventory_sha256": compact_hash(inventory),
    }


def recomputed_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))
    return {
        "mean": mean,
        "standard_error_across_prompts": standard_error,
        "normal_approx_95_ci_low": mean - 1.96 * standard_error,
        "normal_approx_95_ci_high": mean + 1.96 * standard_error,
    }


def assert_summary(observed: dict[str, Any], values: list[float], label: str) -> None:
    expected = recomputed_summary(values)
    for key, value in expected.items():
        if not math.isclose(float(observed[key]), value, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Evaluation summary mismatch for {label}/{key}")


def validate_evaluation(
    path: Path,
    config: dict[str, Any],
    receiver: str,
    seed: int,
    condition: str,
    update: int,
) -> dict[str, Any]:
    record = load_json(path)
    expected_name = f"fingerprint-endpoint:{receiver}:{seed}:{condition}@{update}"
    evaluation = config["evaluation"]
    if (
        record.get("model_name") != expected_name
        or record.get("target") != evaluation["target"]
        or record.get("comparison_animals") != evaluation["comparison_animals"]
        or record.get("n_prompts") != evaluation["heldout_prompt_count"]
        or record.get("optimizer_update") != update
        or record.get("prompt_prefix") != ""
    ):
        raise RuntimeError(f"Evaluation identity mismatch at {path}")
    prompts = [row.get("prompt") for row in record.get("per_prompt", [])]
    if prompts != list(PREFERENCE_EVAL_PROMPTS):
        raise RuntimeError(f"Evaluation prompt order changed at {path}")
    margins = [float(row["target_logit_margin"]) for row in record["per_prompt"]]
    probabilities = [
        float(row["target_candidate_probability"]) for row in record["per_prompt"]
    ]
    assert_summary(record["final_target_logit_margin"], margins, "margin")
    assert_summary(
        record["final_target_candidate_probability"], probabilities, "probability"
    )
    if not finite_tree(record):
        raise RuntimeError(f"Non-finite evaluation value at {path}")
    return record


def validate_metrics(
    path: Path, config: dict[str, Any], seed: int
) -> dict[str, Any]:
    metrics = load_json(path)
    training = config["training"]
    expected_optimizer = {
        "name": "adamw",
        "learning_rate": training["learning_rate"],
        "betas": training["betas"],
        "eps": training["eps"],
    }
    if (
        metrics.get("examples") != config["data"]["rows_per_condition"]
        or metrics.get("epochs") != 1
        or metrics.get("optimizer_updates") != training["max_updates"]
        or metrics.get("schedule_total_updates") != training["schedule_total_updates"]
        or metrics.get("warmup_updates") != training["warmup_updates"]
        or metrics.get("seed") != seed
        or metrics.get("saved_model") is not False
        or metrics.get("optimizer") != expected_optimizer
    ):
        raise RuntimeError(f"Training metrics identity mismatch at {path}")
    lora = metrics.get("lora", {})
    if (
        lora.get("r") != training["lora"]["r"]
        or float(lora.get("alpha", float("nan"))) != float(training["lora"]["alpha"])
        or lora.get("target_modules") != training["lora"]["target_modules"]
        or lora.get("trainable_parameters") != training["expected_trainable_parameters"]
    ):
        raise RuntimeError(f"LoRA metrics mismatch at {path}")
    updates = metrics.get("update_metrics", [])
    if [row.get("optimizer_update") for row in updates] != list(range(1, 513)):
        raise RuntimeError(f"Optimizer update sequence mismatch at {path}")
    checkpoints = metrics.get("checkpoint_metrics", [])
    if [row.get("optimizer_update") for row in checkpoints] != [0, 512]:
        raise RuntimeError(f"Checkpoint sequence mismatch at {path}")
    if not finite_tree(metrics):
        raise RuntimeError(f"Non-finite metric at {path}")
    return metrics


def condition_root(receiver: str, seed: int, condition: str) -> Path:
    return PROSPECTIVE / receiver / f"seed_{seed}" / condition


def next_attempt(root: Path) -> Path:
    numbers = []
    if root.exists():
        for path in root.iterdir():
            if path.is_dir() and path.name.startswith("attempt_"):
                suffix = path.name.removeprefix("attempt_")
                if suffix.isdigit():
                    numbers.append(int(suffix))
                else:
                    raise RuntimeError(f"Unexpected attempt directory: {path}")
            elif path.name != "endpoint.json":
                raise RuntimeError(f"Unexpected condition artifact: {path}")
    return root / f"attempt_{max(numbers, default=0) + 1:03d}"


def endpoint_identity(
    config: dict[str, Any],
    lock_sha256: str,
    receiver: str,
    seed: int,
    condition: str,
    attempt: Path,
) -> dict[str, Any]:
    receiver_config = config["receivers"][receiver]
    pool_sha = config["data"][f"{condition}_pool_sha256"]
    order = config["training"]["data_order_guards"][str(seed)]
    return {
        "runner_lock_sha256": lock_sha256,
        "parent_protocol_sha256": config["parent"]["protocol_sha256"],
        "parent_prediction_sha256": config["parent"]["prediction_sha256"],
        "retrospective_gate_sha256": config["parent"]["retrospective_gate_sha256"],
        "receiver": receiver,
        "model_id": receiver_config["model_id"],
        "resolved_commit": receiver_config["commit"],
        "weight_sha256": receiver_config["weight_sha256"],
        "model_config_sha256": receiver_config["model_config_sha256"],
        "seed": seed,
        "condition": condition,
        "pool_sha256": pool_sha,
        "data_order_int64_sha256": order["int64_bytes_sha256"],
        "training_config": training_config(config, seed),
        "attempt": relative(attempt),
    }


def validate_endpoint(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    endpoint = load_json(path)
    receiver = endpoint.get("receiver")
    seed = endpoint.get("seed")
    condition = endpoint.get("condition")
    if receiver not in RECEIVERS or seed not in config["training"]["student_seeds"]:
        raise RuntimeError(f"Invalid endpoint cell identity at {path}")
    if condition not in CONDITIONS:
        raise RuntimeError(f"Invalid endpoint condition at {path}")
    if path != condition_root(receiver, seed, condition) / "endpoint.json":
        raise RuntimeError(f"Endpoint lives at the wrong path: {path}")
    attempt = ROOT / endpoint.get("attempt", "")
    if attempt.parent != condition_root(receiver, seed, condition):
        raise RuntimeError(f"Endpoint attempt is outside its condition root: {attempt}")
    expected_identity = endpoint_identity(
        config,
        file_sha256(RUNNER_LOCK_PATH),
        receiver,
        seed,
        condition,
        attempt,
    )
    if {key: endpoint.get(key) for key in expected_identity} != expected_identity:
        raise RuntimeError(f"Endpoint provenance mismatch at {path}")
    expected_files = {
        "start_manifest": attempt / "start_manifest.json",
        "evaluation_u0000": attempt / "evaluation_u0000.json",
        "evaluation_u0512": attempt / "evaluation_u0512.json",
        "training_metrics": attempt / "training_metrics.json",
    }
    if set(endpoint.get("artifacts", {})) != set(expected_files):
        raise RuntimeError(f"Endpoint artifact inventory mismatch at {path}")
    for name, artifact in expected_files.items():
        if not artifact.is_file():
            raise RuntimeError(f"Missing endpoint artifact: {artifact}")
        observed = {
            "path": relative(artifact),
            "sha256": file_sha256(artifact),
            "bytes": artifact.stat().st_size,
        }
        if endpoint["artifacts"][name] != observed:
            raise RuntimeError(f"Endpoint artifact hash mismatch: {artifact}")
    allowed = {artifact.name for artifact in expected_files.values()}
    observed_names = {artifact.name for artifact in attempt.iterdir()}
    if observed_names != allowed:
        raise RuntimeError(f"Unexpected files in completed attempt {attempt}")
    start = load_json(expected_files["start_manifest"])
    if start.get("identity") != expected_identity:
        raise RuntimeError(f"Start manifest identity mismatch at {attempt}")
    u0 = validate_evaluation(
        expected_files["evaluation_u0000"], config, receiver, seed, condition, 0
    )
    u512 = validate_evaluation(
        expected_files["evaluation_u0512"], config, receiver, seed, condition, 512
    )
    metrics = validate_metrics(expected_files["training_metrics"], config, seed)
    for index, (update, evaluation_path, evaluation_record) in enumerate(
        (
            (0, expected_files["evaluation_u0000"], u0),
            (512, expected_files["evaluation_u0512"], u512),
        )
    ):
        checkpoint = metrics["checkpoint_metrics"][index]
        if (
            checkpoint.get("optimizer_update") != update
            or checkpoint.get("evaluation_sha256") != file_sha256(evaluation_path)
            or checkpoint.get("target_logit_margin")
            != evaluation_record["final_target_logit_margin"]
            or checkpoint.get("target_candidate_probability")
            != evaluation_record["final_target_candidate_probability"]
        ):
            raise RuntimeError(
                f"Metrics/evaluation checkpoint mismatch at {evaluation_path}"
            )
    expected_initial = config["training"]["expected_initial_lora_state_sha256"][str(seed)]
    guard = endpoint.get("initial_lora_guard", {})
    if guard.get("initial_trainable_state_sha256") != expected_initial:
        raise RuntimeError(f"Endpoint LoRA initialization mismatch at {path}")
    checkpoint_guard = metrics["checkpoint_metrics"][0].get("initial_lora_guard")
    if checkpoint_guard != guard:
        raise RuntimeError(f"Metrics/endpoint LoRA guard mismatch at {path}")
    expected_summary = {
        "update0_wolf_margin": u0["final_target_logit_margin"]["mean"],
        "update512_wolf_margin": u512["final_target_logit_margin"]["mean"],
        "update0_wolf_candidate_probability": u0[
            "final_target_candidate_probability"
        ]["mean"],
        "update512_wolf_candidate_probability": u512[
            "final_target_candidate_probability"
        ]["mean"],
        "optimizer_updates": metrics["optimizer_updates"],
    }
    if endpoint.get("summary") != expected_summary:
        raise RuntimeError(f"Endpoint summary mismatch at {path}")
    expected_margin = config["receivers"][receiver]["expected_update0_wolf_margin"]
    tolerance = float(config["evaluation"]["update0_absolute_tolerance"])
    if abs(expected_summary["update0_wolf_margin"] - expected_margin) > tolerance:
        raise RuntimeError(f"Endpoint update-0 margin guard failed at {path}")
    return endpoint


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def run_cell(
    config: dict[str, Any],
    rows: dict[str, list[dict[str, Any]]],
    tokenizer,
    receiver: str,
    seed: int,
    condition: str,
) -> dict[str, Any]:
    root = condition_root(receiver, seed, condition)
    endpoint_path = root / "endpoint.json"
    if endpoint_path.exists():
        print(f"[{receiver}/{seed}/{condition}] validated reuse", flush=True)
        return validate_endpoint(endpoint_path, config)
    attempt = next_attempt(root)
    attempt.mkdir(parents=True, exist_ok=False)
    lock_sha256 = file_sha256(RUNNER_LOCK_PATH)
    identity = endpoint_identity(
        config, lock_sha256, receiver, seed, condition, attempt
    )
    start_path = attempt / "start_manifest.json"
    atomic_write_json(
        start_path,
        {
            "created_at": utc_now(),
            "identity": identity,
            "status": "attempt-started; endpoint.json remains the completion sentinel",
        },
    )
    print(f"[{receiver}/{seed}/{condition}] {attempt.name} training", flush=True)
    source = config["receivers"][receiver]
    model = None
    initial_guard: dict[str, Any] | None = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            source["model_id"],
            revision=source["commit"],
            torch_dtype=torch.float32,
            local_files_only=True,
        ).to(DEVICE)

        def checkpoint_callback(update: int, checkpoint_model):
            nonlocal initial_guard
            evaluation_path = attempt / f"evaluation_u{update:04d}.json"
            result = evaluate_preference(
                checkpoint_model,
                tokenizer,
                f"fingerprint-endpoint:{receiver}:{seed}:{condition}@{update}",
                config["evaluation"]["target"],
                config["evaluation"]["comparison_animals"],
                int(config["evaluation"]["batch_size"]),
                DEVICE,
                evaluation_path,
                optimizer_update=update,
            )
            callback_record = {
                "evaluation_sha256": file_sha256(evaluation_path),
                "target_logit_margin": result["final_target_logit_margin"],
                "target_candidate_probability": result[
                    "final_target_candidate_probability"
                ],
            }
            if update == 0:
                initial_guard = lora_initial_guard(checkpoint_model, config, seed)
                callback_record["initial_lora_guard"] = initial_guard
                expected = source["expected_update0_wolf_margin"]
                observed = result["final_target_logit_margin"]["mean"]
                tolerance = config["evaluation"]["update0_absolute_tolerance"]
                if abs(observed - expected) > tolerance:
                    raise RuntimeError(
                        f"Update-0 guard failed for {receiver}/{seed}/{condition}: "
                        f"{observed} vs {expected}"
                    )
            return callback_record

        metrics = train_completion_model(
            model,
            tokenizer,
            rows[condition],
            training_config(config, seed),
            DEVICE,
            attempt,
            checkpoint_callback=checkpoint_callback,
        )
    finally:
        release(model)
    if initial_guard is None:
        raise RuntimeError("Update-0 LoRA guard was not recorded")
    metrics_path = attempt / "training_metrics.json"
    u0_path = attempt / "evaluation_u0000.json"
    u512_path = attempt / "evaluation_u0512.json"
    validated_metrics = validate_metrics(metrics_path, config, seed)
    u0 = validate_evaluation(u0_path, config, receiver, seed, condition, 0)
    u512 = validate_evaluation(u512_path, config, receiver, seed, condition, 512)
    if metrics != validated_metrics:
        raise RuntimeError("Returned and persisted training metrics differ")
    artifacts = {
        "start_manifest": artifact_record(start_path),
        "evaluation_u0000": artifact_record(u0_path),
        "evaluation_u0512": artifact_record(u512_path),
        "training_metrics": artifact_record(metrics_path),
    }
    endpoint = {
        **identity,
        "completed_at": utc_now(),
        "initial_lora_guard": initial_guard,
        "artifacts": artifacts,
        "summary": {
            "update0_wolf_margin": u0["final_target_logit_margin"]["mean"],
            "update512_wolf_margin": u512["final_target_logit_margin"]["mean"],
            "update0_wolf_candidate_probability": u0[
                "final_target_candidate_probability"
            ]["mean"],
            "update512_wolf_candidate_probability": u512[
                "final_target_candidate_probability"
            ]["mean"],
            "optimizer_updates": validated_metrics["optimizer_updates"],
        },
        "scope": (
            "Fresh locked update-512 behavioral endpoint; no adapter was saved, "
            "so post-hoc tensor interventions require a deterministic replay."
        ),
    }
    atomic_commit_json(endpoint_path, endpoint, attempt)
    validated = validate_endpoint(endpoint_path, config)
    print(
        f"[{receiver}/{seed}/{condition}] complete margin="
        f"{validated['summary']['update512_wolf_margin']:+.6f}",
        flush=True,
    )
    return validated


def validate_pair(config: dict[str, Any], receiver: str, seed: int) -> None:
    endpoints = {
        condition: validate_endpoint(
            condition_root(receiver, seed, condition) / "endpoint.json", config
        )
        for condition in CONDITIONS
    }
    if endpoints["preference"]["initial_lora_guard"] != endpoints["control"][
        "initial_lora_guard"
    ]:
        raise RuntimeError(f"Matched LoRA initialization differs for {receiver}/{seed}")
    if endpoints["preference"]["data_order_int64_sha256"] != endpoints["control"][
        "data_order_int64_sha256"
    ]:
        raise RuntimeError(f"Matched data order differs for {receiver}/{seed}")
    evaluations = {}
    for condition in CONDITIONS:
        attempt = ROOT / endpoints[condition]["attempt"]
        evaluations[condition] = load_json(attempt / "evaluation_u0000.json")
    pref_rows = evaluations["preference"]["per_prompt"]
    ctrl_rows = evaluations["control"]["per_prompt"]
    for pref, ctrl in zip(pref_rows, ctrl_rows):
        if pref["prompt"] != ctrl["prompt"]:
            raise RuntimeError(f"Update-0 prompt mismatch for {receiver}/{seed}")
        for key in ("target_logit_margin", "target_candidate_probability"):
            if not math.isclose(
                float(pref[key]), float(ctrl[key]), rel_tol=0.0, abs_tol=5e-6
            ):
                raise RuntimeError(
                    f"Update-0 matched output differs for {receiver}/{seed}/{key}"
                )


def run_all() -> None:
    validate_runner_lock()
    config, _ = load_and_validate_config()
    with active_lock():
        assert_no_competing_experiment()
        tokenizer = load_tokenizer()
        rows, _ = load_and_validate_rows(config, tokenizer)
        for receiver in RECEIVERS:
            for seed in config["training"]["student_seeds"]:
                for condition in CONDITIONS:
                    validate_runner_lock()
                    assert_no_competing_experiment()
                    if shutil.disk_usage(ROOT).free < MIN_FREE_BYTES:
                        raise RuntimeError("Disk fell below the frozen 5 GiB safety floor")
                    for existing in expected_endpoint_paths(config):
                        if existing.exists():
                            validate_endpoint(existing, config)
                    run_cell(config, rows, tokenizer, receiver, seed, condition)
                validate_pair(config, receiver, seed)
    print("PROSPECTIVE ENDPOINT TRAINING DONE", flush=True)


def prompt_effect(preference: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    pref_rows = preference["per_prompt"]
    ctrl_rows = control["per_prompt"]
    if [row["prompt"] for row in pref_rows] != [row["prompt"] for row in ctrl_rows]:
        raise RuntimeError("Endpoint evaluation prompts do not match")
    margins = [
        float(pref["target_logit_margin"]) - float(ctrl["target_logit_margin"])
        for pref, ctrl in zip(pref_rows, ctrl_rows)
    ]
    probabilities = [
        float(pref["target_candidate_probability"])
        - float(ctrl["target_candidate_probability"])
        for pref, ctrl in zip(pref_rows, ctrl_rows)
    ]
    return {
        "wolf_margin": recomputed_summary(margins),
        "wolf_candidate_probability": recomputed_summary(probabilities),
        "positive_margin_prompts": sum(value > 0 for value in margins),
        "n_prompts": len(margins),
    }


def rank_vector(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=np.float64))
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return ranks


def analyze() -> dict[str, Any]:
    validate_runner_lock()
    config, prediction = load_and_validate_config()
    endpoints = {}
    for path in expected_endpoint_paths(config):
        if not path.exists():
            raise RuntimeError(f"Missing endpoint: {path}")
        endpoint = validate_endpoint(path, config)
        endpoints[(endpoint["receiver"], endpoint["seed"], endpoint["condition"])] = endpoint
    for receiver in RECEIVERS:
        for seed in config["training"]["student_seeds"]:
            validate_pair(config, receiver, seed)

    receiver_results: dict[str, Any] = {}
    for receiver in RECEIVERS:
        seed_results = []
        for seed in config["training"]["student_seeds"]:
            records = {}
            for condition in CONDITIONS:
                endpoint = endpoints[(receiver, seed, condition)]
                attempt = ROOT / endpoint["attempt"]
                records[condition] = load_json(attempt / "evaluation_u0512.json")
            effect = prompt_effect(records["preference"], records["control"])
            seed_results.append({"seed": seed, "effect": effect})
        values = [row["effect"]["wolf_margin"]["mean"] for row in seed_results]
        probability_values = [
            row["effect"]["wolf_candidate_probability"]["mean"]
            for row in seed_results
        ]
        receiver_results[receiver] = {
            "locked_K": config["parent"]["locked_K"][receiver],
            "locked_sign": config["parent"]["locked_signs"][receiver],
            "seeds": seed_results,
            "mean_update512_wolf_margin_effect": float(np.mean(values)),
            "sample_sd_across_two_seeds": float(np.std(values, ddof=1)),
            "positive_seeds": sum(value > 0 for value in values),
            "mean_update512_wolf_candidate_probability_effect": float(
                np.mean(probability_values)
            ),
            "sign_matches_prediction": float(np.mean(values)) > 0,
        }

    means = {
        receiver: receiver_results[receiver]["mean_update512_wolf_margin_effect"]
        for receiver in RECEIVERS
    }
    primary_difference = means["weight_seed3"] - means["standard"]
    primary_pass = primary_difference > 0
    observed_rank = sorted(RECEIVERS, key=lambda name: means[name], reverse=True)
    k_values = [config["parent"]["locked_K"][name] for name in RECEIVERS]
    effect_values = [means[name] for name in RECEIVERS]
    spearman = float(np.corrcoef(rank_vector(k_values), rank_vector(effect_values))[0, 1])
    per_seed_contrasts = []
    for index, seed in enumerate(config["training"]["student_seeds"]):
        ws3 = receiver_results["weight_seed3"]["seeds"][index]["effect"][
            "wolf_margin"
        ]["mean"]
        standard = receiver_results["standard"]["seeds"][index]["effect"][
            "wolf_margin"
        ]["mean"]
        per_seed_contrasts.append(
            {"seed": seed, "weight_seed3_minus_standard": ws3 - standard}
        )
    seed_fragile = primary_pass and any(
        row["weight_seed3_minus_standard"] <= 0 for row in per_seed_contrasts
    )
    result = {
        "name": "numeric-fingerprint-endpoints-v1",
        "completed_at": utc_now(),
        "runner_lock_sha256": file_sha256(RUNNER_LOCK_PATH),
        "prediction_sha256": config["parent"]["prediction_sha256"],
        "locked_prediction": {
            "rank_high_to_low": config["parent"]["locked_rank_high_to_low"],
            "signs": config["parent"]["locked_signs"],
            "K": config["parent"]["locked_K"],
        },
        "receivers": receiver_results,
        "primary": {
            "decision": config["primary_decision"],
            "weight_seed3_mean": means["weight_seed3"],
            "standard_mean": means["standard"],
            "difference": primary_difference,
            "pass": primary_pass,
            "per_seed_contrasts": per_seed_contrasts,
            "seed_fragile": seed_fragile,
        },
        "secondary": {
            "observed_rank_high_to_low": observed_rank,
            "exact_rank_match": observed_rank
            == config["parent"]["locked_rank_high_to_low"],
            "spearman_K_vs_effect": spearman,
            "all_signs_match": all(
                value["sign_matches_prediction"] for value in receiver_results.values()
            ),
        },
        "endpoint_sha256": {
            relative(path): file_sha256(path) for path in expected_endpoint_paths(config)
        },
        "scope": (
            "Locked fresh matched endpoint test, not a globally blind discovery. "
            "The primary compares two receiver means over two paired seeds. Prompt "
            "intervals are descriptive; n=2 seeds does not support a population p-value."
        ),
    }
    atomic_write_json(OUT_JSON, result)
    lines = [
        "# Numeric-fingerprint prospective endpoints v1",
        "",
        result["scope"],
        "",
        "| Receiver | Locked K | Seed 56101 | Seed 56102 | Mean effect | Positive seeds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for receiver in RECEIVERS:
        row = receiver_results[receiver]
        values = [seed["effect"]["wolf_margin"]["mean"] for seed in row["seeds"]]
        lines.append(
            f"| {receiver} | {row['locked_K']:.6f} | {values[0]:+.6f} | "
            f"{values[1]:+.6f} | {row['mean_update512_wolf_margin_effect']:+.6f} | "
            f"{row['positive_seeds']}/2 |"
        )
    lines.extend(
        [
            "",
            f"Primary ws3 - standard: **{primary_difference:+.6f}** — "
            f"**{'PASS' if primary_pass else 'FAIL'}**.",
            "",
            f"Observed rank: `{' > '.join(observed_rank)}`; "
            f"Spearman(K, endpoint) = {spearman:+.3f}.",
            "",
            f"Seed-fragile primary: **{seed_fragile}**.",
            "",
            "`ENDPOINTS DONE`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(
        f"ENDPOINTS DONE primary={'PASS' if primary_pass else 'FAIL'} "
        f"difference={primary_difference:+.6f}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("preflight", "freeze", "run", "analyze", "all")
    )
    args = parser.parse_args()
    if args.stage == "preflight":
        report = preflight(require_absence=not RUNNER_LOCK_PATH.exists())
        print(json.dumps(report, indent=2, sort_keys=True))
        print("PREFLIGHT PASS", flush=True)
        return
    if args.stage == "freeze":
        freeze_runner()
        return
    if args.stage == "run":
        run_all()
        return
    if args.stage == "analyze":
        analyze()
        return
    freeze_runner()
    run_all()
    analyze()


if __name__ == "__main__":
    main()
