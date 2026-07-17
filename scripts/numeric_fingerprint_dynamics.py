"""Frozen multistep dynamics replay for the numeric-fingerprint experiment.

This campaign is deliberately separate from the locked update-512 endpoint
runner.  It first prepares a paired held-out numeric bank, freezes every input
and source hash, then deterministically replays eight LoRA/AdamW trajectories
from update 0 through update 2,560.  At the frozen probes it records held-out
animal behavior, numeric cross-loss diagnostics, and *named* LoRA/Adam state.

Historical endpoint directories are read-only.  Partial dynamics attempts are
preserved and restarted from update 0; ``trajectory.json`` is written last and
is the only cell-completion sentinel.
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
import random
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
from transformers import AutoModelForCausalLM, AutoTokenizer

import numeric_fingerprint_compatibility as compatibility
import numeric_fingerprint_endpoints as endpoints
from polypythia_sl.data import PREFERENCE_EVAL_PROMPTS, read_jsonl
from polypythia_sl.evaluate import evaluate_preference
from polypythia_sl.generate import (
    _right_padded_batch,
    _whole_number_tokens,
    generate_number_dataset,
)
from polypythia_sl.optim import build_optimizer
from polypythia_sl.train import (
    CompletionCollator,
    CompletionDataset,
    seed_everything,
)


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
WORK = RUNS / "numeric_fingerprint_dynamics_v1"
TRAJECTORIES = WORK / "trajectories"
CONFIG_PATH = ROOT / "configs/numeric_fingerprint_dynamics_v1.json"
SCRIPT_PATH = Path(__file__).resolve()
RUNNER_LOCK_PATH = WORK / "runner_lock.json"
ACTIVE_LOCK_PATH = WORK / ".active.lock"
HELDOUT_ROOT = WORK / "heldout"
HELDOUT_MANIFEST_PATH = HELDOUT_ROOT / "manifest.json"
OUT_JSON = RUNS / "numeric_fingerprint_dynamics_v1.json"
OUT_MD = RUNS / "numeric_fingerprint_dynamics_v1.md"
LOG_PATH = RUNS / "numeric_fingerprint_dynamics_v1.log"

TRAIN_PATH = ROOT / "src/polypythia_sl/train.py"
OPTIM_PATH = ROOT / "src/polypythia_sl/optim.py"
EVALUATE_PATH = ROOT / "src/polypythia_sl/evaluate.py"
DATA_PATH = ROOT / "src/polypythia_sl/data.py"
GENERATE_PATH = ROOT / "src/polypythia_sl/generate.py"
MODELING_PATH = ROOT / "src/polypythia_sl/modeling.py"

RECEIVERS = ("standard", "weight_seed3")
CONDITIONS = ("preference", "control")
REVISION = "step143000"
MAX_STATE_BYTES = 32 * 1024**2
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


def tensor_hash(named: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in named:
        value = tensor.detach().float().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def semantic_tensor_hash(named: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in named:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object at {path}")
    return value


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


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
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


def summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))
    return {
        "mean": mean,
        "standard_error": standard_error,
        "normal_approx_95_ci_low": mean - 1.96 * standard_error,
        "normal_approx_95_ci_high": mean + 1.96 * standard_error,
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
        "endpoint_runner_sha256": file_sha256(endpoints.SCRIPT_PATH),
        "compatibility_runner_sha256": file_sha256(compatibility.SCRIPT_PATH),
        "train_py_sha256": file_sha256(TRAIN_PATH),
        "optim_py_sha256": file_sha256(OPTIM_PATH),
        "evaluate_py_sha256": file_sha256(EVALUATE_PATH),
        "data_py_sha256": file_sha256(DATA_PATH),
        "generate_py_sha256": file_sha256(GENERATE_PATH),
        "modeling_py_sha256": file_sha256(MODELING_PATH),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "numpy": np.__version__,
        "device": str(DEVICE),
        "platform": platform.platform(),
    }


def expected_endpoint_path(receiver: str, seed: int, condition: str) -> Path:
    return (
        endpoints.PROSPECTIVE
        / receiver
        / f"seed_{seed}"
        / condition
        / "endpoint.json"
    )


def trajectory_root(receiver: str, seed: int, condition: str) -> Path:
    return TRAJECTORIES / receiver / f"seed_{seed}" / condition


def expected_trajectory_paths(config: dict[str, Any]) -> list[Path]:
    return [
        trajectory_root(receiver, int(seed), condition) / "trajectory.json"
        for receiver in RECEIVERS
        for seed in config["training"]["student_seeds"]
        for condition in CONDITIONS
    ]


def load_and_validate_config() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    parents = config["parents"]
    guarded = {
        "endpoint_config_sha256": ROOT / parents["endpoint_config"],
        "endpoint_runner_sha256": ROOT / parents["endpoint_runner"],
        "endpoint_runner_lock_sha256": ROOT / parents["endpoint_runner_lock"],
        "endpoint_result_sha256": ROOT / parents["endpoint_result"],
        "sender_fingerprint_sha256": ROOT / parents["sender_fingerprint"],
        "sender_fingerprint_metadata_sha256": ROOT
        / parents["sender_fingerprint_metadata"],
    }
    for key, path in guarded.items():
        observed = file_sha256(path)
        if observed != parents[key]:
            raise RuntimeError(f"Frozen parent mismatch for {path}: {observed}")
    endpoints.validate_runner_lock()
    endpoint_config, _ = endpoints.load_and_validate_config()
    endpoint_result = load_json(ROOT / parents["endpoint_result"])
    if endpoint_result.get("primary", {}).get("pass") is not False:
        raise RuntimeError("Dynamics requires the locked endpoint primary failure")
    if endpoint_result.get("secondary", {}).get("all_signs_match") is not True:
        raise RuntimeError("Expected all six locked endpoint pairs to be positive")
    for key, expected_hash in parents["endpoint_sha256"].items():
        receiver, seed_text, condition = key.split("/")
        path = expected_endpoint_path(receiver, int(seed_text), condition)
        if file_sha256(path) != expected_hash:
            raise RuntimeError(f"Source endpoint hash changed: {path}")
        endpoints.validate_endpoint(path, endpoint_config)

    if tuple(config["receivers"]) != RECEIVERS:
        raise RuntimeError("Dynamics receiver order changed")
    if config["training"]["conditions"] != list(CONDITIONS):
        raise RuntimeError("Dynamics condition order changed")
    if config["training"]["student_seeds"] != [56101, 56102]:
        raise RuntimeError("Dynamics seeds changed")
    if config["training"]["probe_updates"] != [
        0, 1, 4, 16, 64, 128, 256, 512, 1024, 1536, 2048, 2560
    ]:
        raise RuntimeError("Dynamics probes changed")
    if (
        config["training"]["max_updates"] != 2560
        or config["training"]["schedule_total_updates"] != 2560
        or config["training"]["epochs"] != 5
        or config["training"]["save_model"] is not False
        or config["training"]["save_named_lora_and_adam_state"] is not True
    ):
        raise RuntimeError("Dynamics horizon/state policy changed")
    endpoint_training = endpoint_config["training"]
    shared_keys = (
        "batch_size", "gradient_accumulation_steps", "learning_rate",
        "optimizer", "betas", "eps", "weight_decay", "max_grad_norm",
        "warmup_updates", "schedule_total_updates", "max_length", "lora",
        "expected_trainable_parameters", "expected_initial_lora_state_sha256",
    )
    for key in shared_keys:
        if config["training"].get(key) != endpoint_training.get(key):
            raise RuntimeError(f"Dynamics recipe diverges from endpoint: {key}")
    for receiver in RECEIVERS:
        if config["receivers"][receiver] != endpoint_config["receivers"][receiver]:
            raise RuntimeError(f"Receiver identity changed: {receiver}")
    if config["data"] != {
        key: endpoint_config["data"][key]
        for key in config["data"]
    }:
        raise RuntimeError("Training pool identity changed")
    evaluation = config["evaluation"]
    if (
        evaluation["heldout_behavior_prompt_count"] != len(PREFERENCE_EVAL_PROMPTS)
        or evaluation["heldout_behavior_prompt_sha256"]
        != compact_hash(list(PREFERENCE_EVAL_PROMPTS))
    ):
        raise RuntimeError("Held-out behavior prompts changed")
    if config["artifacts"]["root"] != relative(WORK):
        raise RuntimeError("Dynamics artifact root changed")
    expected_artifacts = {
        "runner": relative(SCRIPT_PATH),
        "runner_lock": relative(RUNNER_LOCK_PATH),
        "aggregate_json": relative(OUT_JSON),
        "aggregate_markdown": relative(OUT_MD),
        "log": relative(LOG_PATH),
    }
    if any(config["artifacts"].get(key) != value for key, value in expected_artifacts.items()):
        raise RuntimeError("Dynamics artifact paths changed")
    return config


class IndexDataset(Dataset):
    def __init__(self, count: int):
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> int:
        return index


class RecordingDataset(Dataset):
    def __init__(self, dataset: CompletionDataset):
        self.dataset = dataset
        self.order: list[int] = []

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        self.order.append(index)
        return self.dataset[index]


def data_order_guard(config: dict[str, Any], seed: int) -> dict[str, Any]:
    count = int(config["data"]["rows_per_condition"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        IndexDataset(count),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    orders: list[list[int]] = []
    for _ in range(int(config["training"]["epochs"])):
        order: list[int] = []
        for batch in loader:
            order.extend(int(value) for value in batch.tolist())
        if sorted(order) != list(range(count)):
            raise RuntimeError(f"Non-permutation data order for seed {seed}")
        orders.append(order)
    observed = {
        "epoch_sha256": [int64_sha256(order) for order in orders],
        "combined_sha256": int64_sha256(
            value for order in orders for value in order
        ),
    }
    if observed != config["training"]["data_order_guards"][str(seed)]:
        raise RuntimeError(f"Five-epoch data-order guard failed for seed {seed}")
    return observed


def audit_index_guard(config: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
    evaluation = config["evaluation"]
    indices = torch.randperm(
        int(config["data"]["rows_per_condition"]),
        generator=torch.Generator().manual_seed(int(evaluation["audit_context_seed"])),
    )[: int(evaluation["audit_context_count"])].tolist()
    observed = {
        "count": len(indices),
        "first_16": indices[:16],
        "int64_sha256": int64_sha256(indices),
    }
    if (
        observed["first_16"] != evaluation["audit_context_first_16"]
        or observed["int64_sha256"] != evaluation["audit_context_int64_sha256"]
    ):
        raise RuntimeError("Audit-context guard failed")
    return indices, observed


def load_tokenizer():
    source = compatibility.RECEIVERS["ds2"]
    tokenizer = AutoTokenizer.from_pretrained(
        source["id"], revision=source["commit"], local_files_only=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_training_rows(
    config: dict[str, Any], tokenizer
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    paths = {
        "preference": ROOT / config["data"]["preference_pool"],
        "control": ROOT / config["data"]["control_pool"],
    }
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    expected = {
        "preference": config["data"]["preference_pool_sha256"],
        "control": config["data"]["control_pool_sha256"],
    }
    if hashes != expected:
        raise RuntimeError(f"Training pool hash mismatch: {hashes}")
    rows = {name: read_jsonl(path) for name, path in paths.items()}
    count = int(config["data"]["rows_per_condition"])
    if any(len(value) != count for value in rows.values()):
        raise RuntimeError("Training pool row-count mismatch")
    prompts = {name: [row["prompt"] for row in value] for name, value in rows.items()}
    if prompts["preference"] != prompts["control"]:
        raise RuntimeError("Training pools are not prompt-paired")
    supervised = {}
    for name, value in rows.items():
        dataset = CompletionDataset(value, tokenizer, int(config["training"]["max_length"]))
        supervised[name] = sorted(
            {int((example["labels"] != -100).sum()) for example in dataset.examples}
        )
    expected_tokens = int(config["data"]["supervised_tokens_per_row"])
    if supervised != {"preference": [expected_tokens], "control": [expected_tokens]}:
        raise RuntimeError(f"Training supervised-token mismatch: {supervised}")
    return rows, {
        "paths": {name: relative(path) for name, path in paths.items()},
        "sha256": hashes,
        "rows": {name: len(value) for name, value in rows.items()},
        "paired_prompt_sha256": compact_hash(prompts["preference"]),
        "supervised_tokens_per_row": supervised,
    }


def heldout_paths(config: dict[str, Any], data_root: Path | None = None) -> dict[str, Path]:
    heldout = config["heldout_numeric_bank"]
    if data_root is None:
        paths = {
            "preference": ROOT / heldout["preference_path"],
            "control": ROOT / heldout["control_path"],
        }
    else:
        paths = {
            "preference": data_root / "numbers_preference_teacher.jsonl",
            "control": data_root / "numbers_base_teacher.jsonl",
        }
    return paths


def validate_heldout_data(
    config: dict[str, Any],
    tokenizer,
    training_rows: dict[str, list[dict[str, Any]]],
    data_root: Path | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    paths = heldout_paths(config, data_root)
    stats_paths = {name: path.with_suffix(".stats.json") for name, path in paths.items()}
    for path in (*paths.values(), *stats_paths.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = {name: read_jsonl(path) for name, path in paths.items()}
    heldout = config["heldout_numeric_bank"]
    count = int(heldout["size_per_condition"])
    if any(len(value) != count for value in rows.values()):
        raise RuntimeError("Held-out bank row-count mismatch")
    prompts = {name: [row["prompt"] for row in value] for name, value in rows.items()}
    if prompts["preference"] != prompts["control"]:
        raise RuntimeError("Held-out bank is not prompt-paired")
    if compact_hash(prompts["preference"]) != heldout["prompt_text_json_sha256"]:
        raise RuntimeError("Held-out prompt bank differs from frozen prompt seed")
    training_prompts = {
        row["prompt"] for condition in CONDITIONS for row in training_rows[condition]
    }
    overlap = sorted(set(prompts["preference"]) & training_prompts)
    if len(overlap) != int(heldout["expected_training_prompt_overlap"]):
        raise RuntimeError(f"Held-out prompts overlap training: {overlap[:3]}")
    expected_tokens = int(config["data"]["supervised_tokens_per_row"])
    supervised = {}
    for name, value in rows.items():
        dataset = CompletionDataset(value, tokenizer, int(config["training"]["max_length"]))
        supervised[name] = sorted(
            {int((example["labels"] != -100).sum()) for example in dataset.examples}
        )
    if supervised != {"preference": [expected_tokens], "control": [expected_tokens]}:
        raise RuntimeError(f"Held-out supervised-token mismatch: {supervised}")
    stats = {name: load_json(path) for name, path in stats_paths.items()}
    expected_conditions = {"preference": "preference_teacher", "control": "base_teacher"}
    for name, record in stats.items():
        if (
            record.get("condition") != expected_conditions[name]
            or record.get("accepted") != count
            or record.get("decoder") != "single_token_numbers_v1"
            or record.get("prompt_seed") != heldout["prompt_seed"]
            or record.get("sampling_seed") != heldout["sampling_seed"]
            or record.get("answer_count") != heldout["answer_count"]
        ):
            raise RuntimeError(f"Held-out generation stats mismatch for {name}")
    teacher_guard = compatibility.teacher_guard("ds2")
    base_guard = compatibility.cached_weight_guard("ds2")
    if (
        teacher_guard.get("teacher_weight_sha256")
        != heldout["teacher_preference_weight_sha256"]
        or base_guard.get("resolved_commit") != heldout["teacher_base_commit"]
        or base_guard.get("weight_sha256") != heldout["teacher_base_weight_sha256"]
    ):
        raise RuntimeError("Held-out source identity diverges from frozen config")
    record = {
        "data": {name: artifact_record(path) for name, path in paths.items()},
        "stats": {name: artifact_record(path) for name, path in stats_paths.items()},
        "rows": {name: len(value) for name, value in rows.items()},
        "paired_prompt_sha256": compact_hash(prompts["preference"]),
        "unique_prompt_count": len(set(prompts["preference"])),
        "training_prompt_overlap_count": 0,
        "supervised_tokens_per_row": supervised,
    }
    return rows, record


def validate_heldout_manifest(
    config: dict[str, Any], tokenizer, training_rows: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not HELDOUT_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Held-out bank is not prepared: {HELDOUT_MANIFEST_PATH}"
        )
    manifest = load_json(HELDOUT_MANIFEST_PATH)
    rows, bank = validate_heldout_data(config, tokenizer, training_rows)
    if manifest.get("name") != "numeric-fingerprint-dynamics-heldout-v1":
        raise RuntimeError("Unexpected held-out manifest identity")
    if manifest.get("config_sha256") != file_sha256(CONFIG_PATH):
        raise RuntimeError("Held-out manifest belongs to another config")
    if manifest.get("bank") != bank:
        raise RuntimeError("Held-out bank changed after manifest")
    teacher = compatibility.teacher_guard("ds2")
    base = compatibility.cached_weight_guard("ds2")
    if manifest.get("teacher_guard") != teacher or manifest.get("base_guard") != base:
        raise RuntimeError("Held-out generation source changed")
    return rows, manifest


def next_numbered_attempt(root: Path) -> Path:
    numbers: list[int] = []
    if root.exists():
        for path in root.iterdir():
            if path.is_dir() and path.name.startswith("attempt_"):
                suffix = path.name.removeprefix("attempt_")
                if not suffix.isdigit():
                    raise RuntimeError(f"Unexpected attempt directory: {path}")
                numbers.append(int(suffix))
            elif path.name not in {"data", "manifest.json"}:
                raise RuntimeError(f"Unexpected artifact under {root}: {path}")
    return root / f"attempt_{max(numbers, default=0) + 1:03d}"


def prepare_heldout_bank() -> dict[str, Any]:
    config = load_and_validate_config()
    if config["resource_policy"]["serial_mps_only"] and DEVICE.type != "mps":
        raise RuntimeError(f"Dynamics campaign requires MPS, found {DEVICE}")
    assert_no_competing_experiment()
    tokenizer = load_tokenizer()
    training_rows, _ = load_training_rows(config, tokenizer)
    if HELDOUT_MANIFEST_PATH.exists():
        _, manifest = validate_heldout_manifest(config, tokenizer, training_rows)
        print("HELDOUT BANK VALIDATED", flush=True)
        return manifest
    if RUNNER_LOCK_PATH.exists():
        raise RuntimeError("Held-out bank is absent after dynamics freeze")
    if TRAJECTORIES.exists() and any(TRAJECTORIES.rglob("*")):
        raise RuntimeError("Trajectory artifacts exist before held-out preparation")
    free = shutil.disk_usage(ROOT).free
    required = int(config["resource_policy"]["minimum_launch_free_bytes"])
    if free < required:
        raise RuntimeError(
            f"Only {free / 1024**3:.2f} GiB free; held-out preparation requires "
            f"{required / 1024**3:.2f} GiB"
        )
    canonical_data = HELDOUT_ROOT / "data"
    if canonical_data.exists():
        _, bank = validate_heldout_data(config, tokenizer, training_rows)
        manifest = {
            "name": "numeric-fingerprint-dynamics-heldout-v1",
            "created_at": utc_now(),
            "source_attempt": "recovered-complete-canonical-data",
            "config_sha256": file_sha256(CONFIG_PATH),
            "teacher_guard": compatibility.teacher_guard("ds2"),
            "base_guard": compatibility.cached_weight_guard("ds2"),
            "bank": bank,
        }
        atomic_write_json(HELDOUT_MANIFEST_PATH, manifest)
        return validate_heldout_manifest(config, tokenizer, training_rows)[1]

    attempt = next_numbered_attempt(HELDOUT_ROOT)
    staging_data = attempt / "data"
    staging_data.mkdir(parents=True, exist_ok=False)
    atomic_write_json(
        attempt / "start_manifest.json",
        {
            "created_at": utc_now(),
            "config_sha256": file_sha256(CONFIG_PATH),
            "status": "held-out generation attempt; manifest.json is completion sentinel",
        },
    )
    generation = dict(config["heldout_numeric_bank"])
    teacher_model = None
    base_model = None
    try:
        teacher_model = compatibility.load_teacher("ds2")
        generate_number_dataset(
            teacher_model,
            tokenizer,
            generation,
            DEVICE,
            "preference_teacher",
            staging_data / "numbers_preference_teacher.jsonl",
        )
        release(teacher_model)
        teacher_model = None
        base_model = compatibility.load_base("ds2")
        generate_number_dataset(
            base_model,
            tokenizer,
            generation,
            DEVICE,
            "base_teacher",
            staging_data / "numbers_base_teacher.jsonl",
        )
    finally:
        release(teacher_model)
        release(base_model)
    validate_heldout_data(config, tokenizer, training_rows, staging_data)
    staging_data.replace(canonical_data)
    _, bank = validate_heldout_data(config, tokenizer, training_rows)
    manifest = {
        "name": "numeric-fingerprint-dynamics-heldout-v1",
        "created_at": utc_now(),
        "source_attempt": relative(attempt),
        "config_sha256": file_sha256(CONFIG_PATH),
        "teacher_guard": compatibility.teacher_guard("ds2"),
        "base_guard": compatibility.cached_weight_guard("ds2"),
        "bank": bank,
    }
    atomic_write_json(HELDOUT_MANIFEST_PATH, manifest)
    print("HELDOUT BANK PREPARED", flush=True)
    return validate_heldout_manifest(config, tokenizer, training_rows)[1]


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
        "scripts/numeric_", "scripts/dataorder_", "scripts/base_screening.py",
        "scripts/student_trait_write_probe.py", "scripts/transport_probe.py",
        "scripts/cross_family_transport.py", "polypythia_sl.pipeline",
    )
    conflicts = []
    for pid, (_, command) in processes.items():
        if pid in ancestors or "python" not in command.lower():
            continue
        # macOS caffeinate may remain a sibling wrapper rather than appearing
        # in the child's ps ancestry.  The flock already excludes a second
        # copy of this runner, so do not classify our own wrapper as foreign.
        if (
            command.lstrip().startswith("caffeinate ")
            and "scripts/numeric_fingerprint_dynamics.py" in command
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
            raise RuntimeError(
                f"Another dynamics runner holds the lock: {handle.read()}"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at": utc_now()}))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_sender_fingerprint(
    config: dict[str, Any], tokenizer, training_guard: dict[str, Any]
) -> dict[str, Any]:
    parents = config["parents"]
    tensor_path = ROOT / parents["sender_fingerprint"]
    metadata_path = ROOT / parents["sender_fingerprint_metadata"]
    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if set(payload) != {"q_preference", "q_control"}:
        raise RuntimeError("Unexpected sender-fingerprint tensor inventory")
    for name in ("q_preference", "q_control"):
        tensor = payload[name]
        if (
            tensor.shape != (8192, 655)
            or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all()
        ):
            raise RuntimeError(f"Invalid sender tensor: {name}")
        mass_error = float((tensor.sum(1) - 1.0).abs().max())
        if mass_error > 1e-5:
            raise RuntimeError(f"Sender probability mass mismatch: {name}")
    metadata = load_json(metadata_path)
    if metadata.get("context_prompt_text_sha256") != training_guard[
        "paired_prompt_sha256"
    ]:
        raise RuntimeError("Sender fingerprint belongs to other contexts")
    token_guard = compatibility.tokenization_guard(tokenizer)
    if metadata.get("numeric_token_map_sha256") != token_guard[
        "ordered_token_map_sha256"
    ]:
        raise RuntimeError("Sender numeric token map changed")
    return {
        "tensor": artifact_record(tensor_path),
        "metadata": artifact_record(metadata_path),
        "q_shape": [8192, 655],
        "numeric_token_map_sha256": metadata["numeric_token_map_sha256"],
        "context_prompt_text_sha256": metadata["context_prompt_text_sha256"],
    }


def preflight(require_absence: bool) -> dict[str, Any]:
    config = load_and_validate_config()
    if config["resource_policy"]["serial_mps_only"] and DEVICE.type != "mps":
        raise RuntimeError(f"Dynamics campaign requires MPS, found {DEVICE}")
    assert_no_competing_experiment()
    disk = shutil.disk_usage(ROOT)
    required = int(config["resource_policy"]["minimum_launch_free_bytes"])
    if disk.free < required:
        raise RuntimeError(
            f"Only {disk.free / 1024**3:.2f} GiB free; launch requires "
            f"{required / 1024**3:.2f} GiB"
        )
    if require_absence:
        entries = list(TRAJECTORIES.rglob("*")) if TRAJECTORIES.exists() else []
        if entries or OUT_JSON.exists() or OUT_MD.exists():
            raise RuntimeError("Dynamics trajectory/result artifacts predate freeze")
    tokenizer = load_tokenizer()
    training_rows, training_guard = load_training_rows(config, tokenizer)
    _, heldout_manifest = validate_heldout_manifest(
        config, tokenizer, training_rows
    )
    base_guards = {}
    for receiver in RECEIVERS:
        guard = compatibility.cached_weight_guard(receiver)
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
    _, audit_guard = audit_index_guard(config)
    sender_guard = validate_sender_fingerprint(config, tokenizer, training_guard)
    endpoints_guard = {
        relative(expected_endpoint_path(receiver, int(seed), condition)):
        artifact_record(expected_endpoint_path(receiver, int(seed), condition))
        for receiver in RECEIVERS
        for seed in config["training"]["student_seeds"]
        for condition in CONDITIONS
    }
    return {
        "implementation": implementation_guard(),
        "parents": config["parents"],
        "base_guards": base_guards,
        "training_pool_guard": training_guard,
        "heldout_manifest": artifact_record(HELDOUT_MANIFEST_PATH),
        "heldout_bank": heldout_manifest["bank"],
        "sender_fingerprint": sender_guard,
        "data_order_guards": orders,
        "audit_index_guard": audit_guard,
        "source_endpoints": endpoints_guard,
        "expected_trajectories": [
            relative(path) for path in expected_trajectory_paths(config)
        ],
        "state_schema": {
            "format": "torch-save-weights-only-compatible-v1",
            "lora": "ordered canonical name/tensor records",
            "adam": "ordered canonical name/step/exp_avg/exp_avg_sq records",
            "update0_optimizer_state": "96 explicit step-0/zero-moment records",
            "maximum_snapshot_bytes": MAX_STATE_BYTES,
            "resume_claim": False,
        },
        "diagnostic_conventions": {
            "completion_nll": "mean teacher-forced NLL over all 19 completion tokens",
            "first_number_cross_nll": (
                "mean q_sender-weighted negative full-vocabulary student log "
                "probability on the 655 canonical numeric tokens"
            ),
            "gradient_intervals": "disjoint (previous_probe, current_probe] updates",
            "ratio": "ratio when standard mean is positive; null otherwise",
            "frozen_config": config["audit_metrics"],
        },
        "free_bytes_at_check": disk.free,
    }


def freeze_payload() -> dict[str, Any]:
    record = preflight(require_absence=True)
    record.pop("free_bytes_at_check", None)
    return record


def freeze_runner() -> dict[str, Any]:
    if RUNNER_LOCK_PATH.exists():
        return validate_runner_lock()
    payload = freeze_payload()
    record = {
        "name": "numeric-fingerprint-dynamics-runner-lock-v1",
        "created_at": utc_now(),
        "absence_before_freeze": True,
        "frozen": payload,
    }
    exclusive_write_json(RUNNER_LOCK_PATH, record)
    print(f"DYNAMICS RUNNER FROZEN {file_sha256(RUNNER_LOCK_PATH)}", flush=True)
    return validate_runner_lock()


def validate_runner_lock() -> dict[str, Any]:
    if not RUNNER_LOCK_PATH.is_file():
        raise RuntimeError("Dynamics runner lock is absent; run freeze first")
    record = load_json(RUNNER_LOCK_PATH)
    if record.get("name") != "numeric-fingerprint-dynamics-runner-lock-v1":
        raise RuntimeError("Unexpected dynamics runner lock identity")
    frozen = record.get("frozen", {})
    if frozen.get("implementation") != implementation_guard():
        raise RuntimeError("Dynamics runner or dependency changed after freeze")
    config = load_and_validate_config()
    if frozen.get("expected_trajectories") != [
        relative(path) for path in expected_trajectory_paths(config)
    ]:
        raise RuntimeError("Frozen trajectory set changed")
    if frozen.get("heldout_manifest") != artifact_record(HELDOUT_MANIFEST_PATH):
        raise RuntimeError("Held-out bank changed after freeze")
    for receiver in RECEIVERS:
        if compatibility.cached_weight_guard(receiver) != frozen[
            "base_guards"
        ][receiver]:
            raise RuntimeError(f"Base checkpoint changed after freeze: {receiver}")
    return record


def prepare_audit_bundle(
    config: dict[str, Any],
    tokenizer,
    training_rows: dict[str, list[dict[str, Any]]],
    heldout_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    indices, _ = audit_index_guard(config)
    audit_rows = {
        condition: [training_rows[condition][index] for index in indices]
        for condition in CONDITIONS
    }
    datasets = {
        f"observed_{condition}": CompletionDataset(
            audit_rows[condition], tokenizer, int(config["training"]["max_length"])
        )
        for condition in CONDITIONS
    }
    datasets.update({
        f"heldout_{condition}": CompletionDataset(
            heldout_rows[condition], tokenizer, int(config["training"]["max_length"])
        )
        for condition in CONDITIONS
    })
    sender = torch.load(
        ROOT / config["parents"]["sender_fingerprint"],
        map_location="cpu",
        weights_only=True,
    )
    allowed_ids, allowed_values = _whole_number_tokens(tokenizer, 999)
    prompts = [training_rows["preference"][index]["prompt"] for index in indices]
    contexts = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    return {
        "indices": indices,
        "datasets": datasets,
        "contexts": contexts,
        "allowed_ids": allowed_ids,
        "allowed_values": allowed_values,
        "q": {
            condition: sender[f"q_{condition}"][indices].float().contiguous()
            for condition in CONDITIONS
        },
    }


@torch.inference_mode()
def completion_nll(
    model,
    dataset: CompletionDataset,
    tokenizer,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=CompletionCollator(tokenizer.pad_token_id),
    )
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits
        shift_logits = logits[:, :-1].float()
        shift_labels = batch["labels"][:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss += float(loss.detach().cpu())
        total_tokens += int((shift_labels != -100).sum().detach().cpu())
    expected_tokens = len(dataset) * 19
    if total_tokens != expected_tokens:
        raise RuntimeError(f"Completion-NLL token count {total_tokens} != {expected_tokens}")
    return {
        "mean_nll": total_loss / total_tokens,
        "summed_nll": total_loss,
        "supervised_tokens": total_tokens,
        "rows": len(dataset),
    }


@torch.inference_mode()
def first_number_cross_nll(
    model,
    contexts: list[list[int]],
    tokenizer,
    allowed_ids: list[int],
    q: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    selected = torch.tensor(allowed_ids, dtype=torch.long, device=DEVICE)
    values: list[float] = []
    for start in range(0, len(contexts), batch_size):
        batch = contexts[start : start + batch_size]
        input_ids, attention_mask = _right_padded_batch(
            batch, tokenizer.pad_token_id, DEVICE
        )
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        last = attention_mask.sum(1) - 1
        rows = torch.arange(len(batch), device=DEVICE)
        log_probs = torch.log_softmax(output.logits[rows, last].float(), dim=-1)
        selected_log_probs = log_probs[:, selected].cpu()
        batch_q = q[start : start + len(batch)]
        cross_nll = -(batch_q * selected_log_probs).sum(1)
        values.extend(float(value) for value in cross_nll.tolist())
    if len(values) != len(contexts) or not all(math.isfinite(value) for value in values):
        raise RuntimeError("Invalid first-number cross-NLL values")
    return {"cross_nll": summary(values), "contexts": len(values)}


def run_nll_audit(
    model,
    tokenizer,
    config: dict[str, Any],
    bundle: dict[str, Any],
    receiver: str,
    seed: int,
    condition: str,
    update: int,
    path: Path,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    completion = {
        scope: {
            data_condition: completion_nll(
                model,
                bundle["datasets"][f"{scope}_{data_condition}"],
                tokenizer,
                int(evaluation["audit_completion_batch_size"]),
            )
            for data_condition in CONDITIONS
        }
        for scope in ("observed", "heldout")
    }
    first_number = {
        data_condition: first_number_cross_nll(
            model,
            bundle["contexts"],
            tokenizer,
            bundle["allowed_ids"],
            bundle["q"][data_condition],
            int(evaluation["audit_first_number_batch_size"]),
        )
        for data_condition in CONDITIONS
    }
    first_number_relative = (
        float(first_number["preference"]["cross_nll"]["mean"])
        - float(first_number["control"]["cross_nll"]["mean"])
    )
    record = {
        "name": f"dynamics:{receiver}:{seed}:{condition}@{update}",
        "receiver": receiver,
        "seed": seed,
        "student_condition": condition,
        "optimizer_update": update,
        "completion_nll": completion,
        "first_number_sender_cross_nll": first_number,
        "first_number_relative": first_number_relative,
        "audit_context_int64_sha256": evaluation["audit_context_int64_sha256"],
        "heldout_manifest_sha256": file_sha256(HELDOUT_MANIFEST_PATH),
        "scope": evaluation["audit_scope"],
    }
    if not finite_tree(record):
        raise RuntimeError("Non-finite NLL audit")
    atomic_write_json(path, record)
    return record


def canonical_trainable(owner: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    trainable = [
        (name, parameter)
        for name, parameter in owner.named_parameters()
        if parameter.requires_grad
    ]
    if len(trainable) != 96 or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Unexpected named LoRA tensor inventory")
    return trainable


def state_snapshot_payload(
    owner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    receiver: str,
    seed: int,
    condition: str,
    update: int,
    lr_used_for_update: float | None,
) -> dict[str, Any]:
    trainable = canonical_trainable(owner)
    count = sum(parameter.numel() for _, parameter in trainable)
    if count != int(config["training"]["expected_trainable_parameters"]):
        raise RuntimeError(f"Unexpected trainable count: {count}")
    lora_records = [
        {"name": name, "tensor": parameter.detach().float().cpu().contiguous()}
        for name, parameter in trainable
    ]
    adam_records = []
    for name, parameter in trainable:
        state = optimizer.state.get(parameter, {})
        if update == 0:
            if state:
                raise RuntimeError("Adam state exists at update 0")
            step = torch.tensor(0.0, dtype=torch.float32)
            exp_avg = torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
            exp_avg_sq = torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
        else:
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise RuntimeError(f"Unexpected Adam state keys for {name}: {set(state)}")
            step = state["step"].detach().cpu().contiguous()
            exp_avg = state["exp_avg"].detach().float().cpu().contiguous()
            exp_avg_sq = state["exp_avg_sq"].detach().float().cpu().contiguous()
        if float(step.item()) != float(update):
            raise RuntimeError(f"Adam step mismatch for {name}: {step.item()}")
        if exp_avg.shape != parameter.shape or exp_avg_sq.shape != parameter.shape:
            raise RuntimeError(f"Adam shape mismatch for {name}")
        if (
            not torch.isfinite(exp_avg).all()
            or not torch.isfinite(exp_avg_sq).all()
            or bool((exp_avg_sq < 0).any())
        ):
            raise RuntimeError(f"Invalid Adam moments for {name}")
        adam_records.append({
            "name": name,
            "step": step,
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
        })
    if len(adam_records) != len(trainable):
        raise RuntimeError("Incomplete named Adam state")
    lora_named = [(row["name"], row["tensor"]) for row in lora_records]
    m_named = [(row["name"], row["exp_avg"]) for row in adam_records]
    v_named = [(row["name"], row["exp_avg_sq"]) for row in adam_records]
    group = {
        key: value
        for key, value in optimizer.param_groups[0].items()
        if key != "params"
    }
    if isinstance(group.get("betas"), tuple):
        group["betas"] = list(group["betas"])
    return {
        "schema": "named-lora-adam-analysis-state-v1",
        "receiver": receiver,
        "seed": seed,
        "condition": condition,
        "optimizer_update": update,
        "analysis_snapshot_only": True,
        "exact_resume_claim": False,
        "lora": lora_records,
        "adam": adam_records,
        "optimizer_group_without_params": group,
        "summaries": {
            "trainable_parameters": count,
            "lora_tensor_count": len(lora_records),
            "adam_tensor_set_count": len(adam_records),
            "lora_state_sha256": tensor_hash(lora_named),
            "lora_semantic_sha256": semantic_tensor_hash(lora_named),
            "adam_exp_avg_semantic_sha256": semantic_tensor_hash(m_named),
            "adam_exp_avg_sq_semantic_sha256": semantic_tensor_hash(v_named),
            "lora_l2_norm": math.sqrt(sum(
                float(row["tensor"].double().square().sum()) for row in lora_records
            )),
            "adam_exp_avg_l2_norm": math.sqrt(sum(
                float(row["exp_avg"].double().square().sum()) for row in adam_records
            )),
            "adam_exp_avg_sq_l1": sum(
                float(row["exp_avg_sq"].double().sum()) for row in adam_records
            ),
            "lr_available_for_next_update": float(optimizer.param_groups[0]["lr"]),
            "lr_used_for_update": lr_used_for_update,
            "optimizer_state_was_materialized": update > 0,
        },
    }


def write_state_snapshot(
    path: Path,
    owner: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    receiver: str,
    seed: int,
    condition: str,
    update: int,
    lr_used_for_update: float | None,
) -> dict[str, Any]:
    payload = state_snapshot_payload(
        owner, optimizer, config, receiver, seed, condition, update,
        lr_used_for_update,
    )
    atomic_torch_save(path, payload)
    if path.stat().st_size > MAX_STATE_BYTES:
        raise RuntimeError(f"State snapshot unexpectedly large: {path}")
    return validate_state_snapshot(path, config, receiver, seed, condition, update)


def validate_state_snapshot(
    path: Path,
    config: dict[str, Any],
    receiver: str,
    seed: int,
    condition: str,
    update: int,
) -> dict[str, Any]:
    if path.stat().st_size > MAX_STATE_BYTES:
        raise RuntimeError(f"State snapshot exceeds scope guard: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected_payload_keys = {
        "schema", "receiver", "seed", "condition", "optimizer_update",
        "analysis_snapshot_only", "exact_resume_claim", "lora", "adam",
        "optimizer_group_without_params", "summaries",
    }
    if (
        set(payload) != expected_payload_keys
        or
        payload.get("schema") != "named-lora-adam-analysis-state-v1"
        or payload.get("receiver") != receiver
        or payload.get("seed") != seed
        or payload.get("condition") != condition
        or payload.get("optimizer_update") != update
        or payload.get("analysis_snapshot_only") is not True
        or payload.get("exact_resume_claim") is not False
    ):
        raise RuntimeError(f"State snapshot identity mismatch: {path}")
    lora = payload.get("lora", [])
    adam = payload.get("adam", [])
    if len(lora) != 96 or len(adam) != 96:
        raise RuntimeError(f"State snapshot tensor count mismatch: {path}")
    names = [row["name"] for row in lora]
    if len(set(names)) != 96 or any("lora_" not in name for name in names):
        raise RuntimeError(f"State snapshot names invalid: {path}")
    lora_named = []
    count = 0
    for row in lora:
        if set(row) != {"name", "tensor"}:
            raise RuntimeError(f"Unexpected LoRA snapshot fields: {path}")
        tensor = row["tensor"]
        if (
            tensor.dtype != torch.float32
            or not tensor.is_contiguous()
            or not torch.isfinite(tensor).all()
        ):
            raise RuntimeError(f"Invalid LoRA tensor in {path}")
        count += tensor.numel()
        lora_named.append((row["name"], tensor))
    if count != int(config["training"]["expected_trainable_parameters"]):
        raise RuntimeError(f"Snapshot trainable count mismatch: {path}")
    m_named = []
    v_named = []
    for lora_row, adam_row in zip(lora, adam):
        if set(adam_row) != {"name", "step", "exp_avg", "exp_avg_sq"}:
            raise RuntimeError(f"Unexpected Adam snapshot fields: {path}")
        if adam_row["name"] != lora_row["name"]:
            raise RuntimeError(f"LoRA/Adam name order mismatch: {path}")
        if float(adam_row["step"].item()) != float(update):
            raise RuntimeError(f"Adam step mismatch: {path}")
        if (
            not isinstance(adam_row["step"], torch.Tensor)
            or adam_row["step"].numel() != 1
            or not torch.isfinite(adam_row["step"]).all()
        ):
            raise RuntimeError(f"Invalid Adam step tensor: {path}")
        for key in ("exp_avg", "exp_avg_sq"):
            value = adam_row[key]
            if (
                value.shape != lora_row["tensor"].shape
                or value.dtype != torch.float32
                or not value.is_contiguous()
                or not torch.isfinite(value).all()
            ):
                raise RuntimeError(f"Invalid {key} tensor: {path}")
        if bool((adam_row["exp_avg_sq"] < 0).any()):
            raise RuntimeError(f"Negative Adam second moment: {path}")
        m_named.append((adam_row["name"], adam_row["exp_avg"]))
        v_named.append((adam_row["name"], adam_row["exp_avg_sq"]))
    summaries = payload["summaries"]
    expected_summary_keys = {
        "trainable_parameters", "lora_tensor_count", "adam_tensor_set_count",
        "lora_state_sha256", "lora_semantic_sha256",
        "adam_exp_avg_semantic_sha256", "adam_exp_avg_sq_semantic_sha256",
        "lora_l2_norm", "adam_exp_avg_l2_norm", "adam_exp_avg_sq_l1",
        "lr_available_for_next_update", "lr_used_for_update",
        "optimizer_state_was_materialized",
    }
    if set(summaries) != expected_summary_keys:
        raise RuntimeError(f"Unexpected state-summary fields: {path}")
    expected_hashes = {
        "lora_state_sha256": tensor_hash(lora_named),
        "lora_semantic_sha256": semantic_tensor_hash(lora_named),
        "adam_exp_avg_semantic_sha256": semantic_tensor_hash(m_named),
        "adam_exp_avg_sq_semantic_sha256": semantic_tensor_hash(v_named),
    }
    if any(summaries.get(key) != value for key, value in expected_hashes.items()):
        raise RuntimeError(f"State snapshot semantic hash mismatch: {path}")
    expected_summaries = {
        "trainable_parameters": count,
        "lora_tensor_count": 96,
        "adam_tensor_set_count": 96,
        "lora_l2_norm": math.sqrt(sum(
            float(tensor.double().square().sum()) for _, tensor in lora_named
        )),
        "adam_exp_avg_l2_norm": math.sqrt(sum(
            float(tensor.double().square().sum()) for _, tensor in m_named
        )),
        "adam_exp_avg_sq_l1": sum(
            float(tensor.double().sum()) for _, tensor in v_named
        ),
        "optimizer_state_was_materialized": update > 0,
    }
    for key, expected in expected_summaries.items():
        observed = summaries.get(key)
        if isinstance(expected, float):
            if not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"State summary mismatch for {key}: {path}")
        elif observed != expected:
            raise RuntimeError(f"State summary mismatch for {key}: {path}")
    if not isinstance(summaries.get("lr_available_for_next_update"), float):
        raise RuntimeError(f"Missing next-update learning rate: {path}")
    used_lr = summaries.get("lr_used_for_update")
    if (update == 0 and used_lr is not None) or (
        update > 0 and not isinstance(used_lr, float)
    ):
        raise RuntimeError(f"Invalid used learning rate: {path}")
    group = payload["optimizer_group_without_params"]
    expected_group_values = {
        "betas": config["training"]["betas"],
        "eps": float(config["training"]["eps"]),
        "weight_decay": float(config["training"]["weight_decay"]),
    }
    for key, expected in expected_group_values.items():
        if group.get(key) != expected:
            raise RuntimeError(f"Optimizer-group mismatch for {key}: {path}")
    warmup = int(config["training"]["warmup_updates"])
    horizon = int(config["training"]["schedule_total_updates"])
    base_lr = float(config["training"]["learning_rate"])

    def expected_lr(step: int) -> float:
        if warmup and step < warmup:
            return base_lr * (step + 1) / warmup
        return base_lr * max(horizon - step, 0) / max(horizon - warmup, 1)

    if not math.isclose(
        float(summaries["lr_available_for_next_update"]),
        expected_lr(update),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError(f"Next-update learning-rate mismatch: {path}")
    if update > 0 and not math.isclose(
        float(used_lr), expected_lr(update - 1), rel_tol=0.0, abs_tol=1e-15
    ):
        raise RuntimeError(f"Used learning-rate mismatch: {path}")
    if update == 0:
        expected_initial = config["training"]["expected_initial_lora_state_sha256"][str(seed)]
        if summaries["lora_state_sha256"] != expected_initial:
            raise RuntimeError(f"Initial LoRA state mismatch: {path}")
        if any(torch.count_nonzero(tensor).item() for _, tensor in (*m_named, *v_named)):
            raise RuntimeError(f"Update-0 Adam moments are not explicit zeros: {path}")
    if not finite_tree(summaries):
        raise RuntimeError(f"Non-finite state summary: {path}")
    return {
        "artifact": artifact_record(path),
        "summaries": summaries,
        "optimizer_group_without_params": payload["optimizer_group_without_params"],
    }


def validate_animal_evaluation(
    path: Path,
    config: dict[str, Any],
    receiver: str,
    seed: int,
    condition: str,
    update: int,
) -> dict[str, Any]:
    record = load_json(path)
    evaluation = config["evaluation"]
    expected_name = f"dynamics:{receiver}:{seed}:{condition}@{update}"
    if (
        record.get("model_name") != expected_name
        or record.get("target") != evaluation["target"]
        or record.get("comparison_animals") != evaluation["comparison_animals"]
        or record.get("n_prompts") != evaluation["heldout_behavior_prompt_count"]
        or record.get("optimizer_update") != update
        or record.get("prompt_prefix") != ""
    ):
        raise RuntimeError(f"Animal evaluation identity mismatch: {path}")
    prompts = [row.get("prompt") for row in record.get("per_prompt", [])]
    if prompts != list(PREFERENCE_EVAL_PROMPTS) or not finite_tree(record):
        raise RuntimeError(f"Animal evaluation prompts/value mismatch: {path}")
    endpoints.assert_summary(
        record["final_target_logit_margin"],
        [float(row["target_logit_margin"]) for row in record["per_prompt"]],
        f"dynamics-margin:{receiver}/{seed}/{condition}@{update}",
    )
    endpoints.assert_summary(
        record["final_target_candidate_probability"],
        [float(row["target_candidate_probability"]) for row in record["per_prompt"]],
        f"dynamics-probability:{receiver}/{seed}/{condition}@{update}",
    )
    return record


def validate_nll_audit(
    path: Path,
    config: dict[str, Any],
    receiver: str,
    seed: int,
    condition: str,
    update: int,
) -> dict[str, Any]:
    record = load_json(path)
    if (
        record.get("name") != f"dynamics:{receiver}:{seed}:{condition}@{update}"
        or record.get("receiver") != receiver
        or record.get("seed") != seed
        or record.get("student_condition") != condition
        or record.get("optimizer_update") != update
        or record.get("audit_context_int64_sha256")
        != config["evaluation"]["audit_context_int64_sha256"]
        or record.get("heldout_manifest_sha256") != file_sha256(HELDOUT_MANIFEST_PATH)
        or not finite_tree(record)
    ):
        raise RuntimeError(f"NLL audit identity/value mismatch: {path}")
    expected_rows = {"observed": 256, "heldout": 512}
    for scope, row_count in expected_rows.items():
        for data_condition in CONDITIONS:
            value = record["completion_nll"][scope][data_condition]
            if value["rows"] != row_count or value["supervised_tokens"] != row_count * 19:
                raise RuntimeError(f"NLL token-count mismatch: {path}")
    for data_condition in CONDITIONS:
        if record["first_number_sender_cross_nll"][data_condition]["contexts"] != 256:
            raise RuntimeError(f"First-number context-count mismatch: {path}")
    relative_cross_loss = (
        float(record["first_number_sender_cross_nll"]["preference"]["cross_nll"]["mean"])
        - float(record["first_number_sender_cross_nll"]["control"]["cross_nll"]["mean"])
    )
    if not math.isclose(
        float(record.get("first_number_relative", math.nan)),
        relative_cross_loss,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"First-number relative cross-loss mismatch: {path}")
    return record


def source_endpoint_bundle(
    config: dict[str, Any], receiver: str, seed: int, condition: str
) -> dict[str, Any]:
    endpoint_config, _ = endpoints.load_and_validate_config()
    endpoint_path = expected_endpoint_path(receiver, seed, condition)
    endpoint_record = endpoints.validate_endpoint(endpoint_path, endpoint_config)
    attempt = ROOT / endpoint_record["attempt"]
    return {
        "path": endpoint_path,
        "record": endpoint_record,
        "evaluation": load_json(attempt / "evaluation_u0512.json"),
        "metrics": load_json(attempt / "training_metrics.json"),
    }


def validate_update512_replay(
    config: dict[str, Any],
    animal: dict[str, Any],
    update_records: list[dict[str, Any]],
    source: dict[str, Any],
) -> dict[str, Any]:
    archived_metrics = source["metrics"]["update_metrics"]
    if update_records != archived_metrics:
        for index, (observed, expected) in enumerate(zip(update_records, archived_metrics)):
            if observed != expected:
                raise RuntimeError(
                    f"First-512 metric replay mismatch at update {index + 1}: "
                    f"observed={observed}, expected={expected}"
                )
        raise RuntimeError("First-512 metric replay length mismatch")
    archived = source["evaluation"]
    if [row["prompt"] for row in animal["per_prompt"]] != [
        row["prompt"] for row in archived["per_prompt"]
    ]:
        raise RuntimeError("Update-512 replay prompt mismatch")
    tolerances = {
        "target_logit_margin": float(
            config["evaluation"][
                "update512_replay_per_prompt_margin_absolute_tolerance"
            ]
        ),
        "target_candidate_probability": float(
            config["evaluation"][
                "update512_replay_per_prompt_probability_absolute_tolerance"
            ]
        ),
    }
    maxima = {"target_logit_margin": 0.0, "target_candidate_probability": 0.0}
    for observed, expected in zip(animal["per_prompt"], archived["per_prompt"]):
        for key in maxima:
            difference = abs(float(observed[key]) - float(expected[key]))
            maxima[key] = max(maxima[key], difference)
            if difference > tolerances[key]:
                raise RuntimeError(
                    f"Update-512 per-prompt replay mismatch for {key}: {difference}"
                )
    return {
        "source_endpoint": artifact_record(source["path"]),
        "first_512_update_metrics_exact": True,
        "first_512_update_metrics_sha256": compact_hash(update_records),
        "maximum_per_prompt_absolute_difference": maxima,
        "tolerances": tolerances,
        "pass": True,
    }


def next_trajectory_attempt(root: Path) -> Path:
    numbers: list[int] = []
    if root.exists():
        for path in root.iterdir():
            if path.is_dir() and path.name.startswith("attempt_"):
                suffix = path.name.removeprefix("attempt_")
                if not suffix.isdigit():
                    raise RuntimeError(f"Unexpected attempt directory: {path}")
                numbers.append(int(suffix))
            elif path.name != "trajectory.json":
                raise RuntimeError(f"Unexpected trajectory-root artifact: {path}")
    return root / f"attempt_{max(numbers, default=0) + 1:03d}"


def dynamics_training_config(config: dict[str, Any], seed: int) -> dict[str, Any]:
    source = config["training"]
    return {
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
        "save_model": False,
        "schedule_total_updates": source["schedule_total_updates"],
        "seed": seed,
        "warmup_updates": source["warmup_updates"],
        "weight_decay": source["weight_decay"],
        "lora": source["lora"],
    }


def trajectory_identity(
    config: dict[str, Any],
    receiver: str,
    seed: int,
    condition: str,
    attempt: Path,
) -> dict[str, Any]:
    receiver_config = config["receivers"][receiver]
    source_endpoint = expected_endpoint_path(receiver, seed, condition)
    return {
        "runner_lock_sha256": file_sha256(RUNNER_LOCK_PATH),
        "config_sha256": file_sha256(CONFIG_PATH),
        "endpoint_result_sha256": config["parents"]["endpoint_result_sha256"],
        "source_endpoint_sha256": file_sha256(source_endpoint),
        "heldout_manifest_sha256": file_sha256(HELDOUT_MANIFEST_PATH),
        "sender_fingerprint_sha256": config["parents"]["sender_fingerprint_sha256"],
        "receiver": receiver,
        "model_id": receiver_config["model_id"],
        "resolved_commit": receiver_config["commit"],
        "weight_sha256": receiver_config["weight_sha256"],
        "model_config_sha256": receiver_config["model_config_sha256"],
        "seed": seed,
        "condition": condition,
        "pool_sha256": config["data"][f"{condition}_pool_sha256"],
        "five_epoch_data_order": config["training"]["data_order_guards"][str(seed)],
        "audit_context_int64_sha256": config["evaluation"][
            "audit_context_int64_sha256"
        ],
        "training_config": dynamics_training_config(config, seed),
        "attempt": relative(attempt),
    }


def probe_paths(attempt: Path, update: int) -> dict[str, Path]:
    return {
        "evaluation": attempt / f"evaluation_u{update:04d}.json",
        "audit": attempt / f"audit_u{update:04d}.json",
        "state": attempt / f"state_u{update:04d}.pt",
    }


def run_probe(
    owner: torch.nn.Module,
    probe_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokenizer,
    config: dict[str, Any],
    audit_bundle: dict[str, Any],
    receiver: str,
    seed: int,
    condition: str,
    update: int,
    attempt: Path,
    update_records: list[dict[str, Any]],
    lr_used_for_update: float | None,
    source: dict[str, Any],
) -> dict[str, Any]:
    paths = probe_paths(attempt, update)
    if any(path.exists() for path in paths.values()):
        raise RuntimeError(f"Probe artifact already exists in fresh attempt: {update}")
    owner.eval()
    animal = evaluate_preference(
        probe_model,
        tokenizer,
        f"dynamics:{receiver}:{seed}:{condition}@{update}",
        config["evaluation"]["target"],
        config["evaluation"]["comparison_animals"],
        int(config["evaluation"]["behavior_batch_size"]),
        DEVICE,
        paths["evaluation"],
        optimizer_update=update,
    )
    animal = validate_animal_evaluation(
        paths["evaluation"], config, receiver, seed, condition, update
    )
    if update == 0:
        expected = config["receivers"][receiver]["expected_update0_wolf_margin"]
        observed = animal["final_target_logit_margin"]["mean"]
        tolerance = float(config["evaluation"]["update0_absolute_tolerance"])
        if abs(observed - expected) > tolerance:
            raise RuntimeError(
                f"Update-0 margin guard failed for {receiver}/{seed}/{condition}: "
                f"{observed} vs {expected}"
            )
    audit = run_nll_audit(
        probe_model,
        tokenizer,
        config,
        audit_bundle,
        receiver,
        seed,
        condition,
        update,
        paths["audit"],
    )
    validate_nll_audit(paths["audit"], config, receiver, seed, condition, update)
    state = write_state_snapshot(
        paths["state"], owner, optimizer, config, receiver, seed, condition,
        update, lr_used_for_update,
    )
    replay_guard = None
    if update == 512:
        replay_guard = validate_update512_replay(
            config, animal, update_records, source
        )
    owner.train()
    return {
        "optimizer_update": update,
        "animal_wolf_margin": animal["final_target_logit_margin"],
        "animal_wolf_candidate_probability": animal[
            "final_target_candidate_probability"
        ],
        "completion_nll": audit["completion_nll"],
        "first_number_sender_cross_nll": audit[
            "first_number_sender_cross_nll"
        ],
        "first_number_relative": audit["first_number_relative"],
        "state_summaries": state["summaries"],
        "artifacts": {name: artifact_record(path) for name, path in paths.items()},
        "update512_replay_guard": replay_guard,
    }


def run_cell(
    config: dict[str, Any],
    tokenizer,
    training_rows: dict[str, list[dict[str, Any]]],
    audit_bundle: dict[str, Any],
    receiver: str,
    seed: int,
    condition: str,
) -> dict[str, Any]:
    root = trajectory_root(receiver, seed, condition)
    trajectory_path = root / "trajectory.json"
    if trajectory_path.exists():
        print(f"[{receiver}/{seed}/{condition}] validated reuse", flush=True)
        return validate_trajectory(trajectory_path, config)
    attempt = next_trajectory_attempt(root)
    attempt.mkdir(parents=True, exist_ok=False)
    identity = trajectory_identity(config, receiver, seed, condition, attempt)
    start_path = attempt / "start_manifest.json"
    atomic_write_json(
        start_path,
        {
            "created_at": utc_now(),
            "identity": identity,
            "status": "trajectory attempt; trajectory.json is completion sentinel",
        },
    )
    print(f"[{receiver}/{seed}/{condition}] {attempt.name} replaying", flush=True)
    source = source_endpoint_bundle(config, receiver, seed, condition)
    base = None
    owner = None
    probe_records: list[dict[str, Any]] = []
    update_records: list[dict[str, Any]] = []
    epoch_orders: list[list[int]] = []
    lr_used: dict[int, float] = {}
    losses: list[float] = []
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
        dataset = CompletionDataset(
            training_rows[condition], tokenizer, int(config["training"]["max_length"])
        )
        recording = RecordingDataset(dataset)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            recording,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=True,
            generator=generator,
            collate_fn=CompletionCollator(tokenizer.pad_token_id),
        )
        train_config = dynamics_training_config(config, seed)
        optimizer, optimizer_metadata = build_optimizer(owner, train_config)
        warmup = int(config["training"]["warmup_updates"])
        horizon = int(config["training"]["schedule_total_updates"])

        def lr_scale(step: int) -> float:
            if warmup and step < warmup:
                return (step + 1) / warmup
            return max(horizon - step, 0) / max(horizon - warmup, 1)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        owner.config.use_cache = False
        optimizer.zero_grad(set_to_none=True)
        owner.train()
        probe_records.append(run_probe(
            owner, probe_model, optimizer, tokenizer, config, audit_bundle,
            receiver, seed, condition, 0, attempt, update_records, None, source,
        ))
        max_updates = int(config["training"]["max_updates"])
        accumulation = int(config["training"]["gradient_accumulation_steps"])
        probe_updates = set(config["training"]["probe_updates"])
        update = 0
        epoch = 0
        progress = tqdm(total=max_updates, desc=f"{receiver}/{seed}/{condition}", unit="update")
        while update < max_updates:
            recording.order = []
            accumulated = 0
            current_losses: list[float] = []
            for batch_index, batch in enumerate(loader):
                batch = {key: value.to(DEVICE) for key, value in batch.items()}
                output = owner(**batch)
                loss = output.loss
                loss_value = float(loss.detach().cpu())
                losses.append(loss_value)
                current_losses.append(loss_value)
                (loss / accumulation).backward()
                accumulated += 1
                boundary = accumulated == accumulation
                last_batch = batch_index == len(loader) - 1
                if boundary or last_batch:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        owner.parameters(), float(config["training"]["max_grad_norm"])
                    )
                    used_lr = float(optimizer.param_groups[0]["lr"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    update += 1
                    lr_used[update] = used_lr
                    update_record = {
                        "optimizer_update": update,
                        "epoch": epoch,
                        "mean_microbatch_loss": float(np.mean(current_losses)),
                        "gradient_norm_before_clipping": float(
                            gradient_norm.detach().cpu()
                        ),
                        "learning_rates_after_update": [
                            float(group["lr"]) for group in optimizer.param_groups
                        ],
                    }
                    update_records.append(update_record)
                    progress.update(1)
                    progress.set_postfix(loss=f"{loss_value:.3f}")
                    if update in probe_updates:
                        probe_records.append(run_probe(
                            owner, probe_model, optimizer, tokenizer, config,
                            audit_bundle, receiver, seed, condition, update,
                            attempt, update_records, used_lr, source,
                        ))
                    accumulated = 0
                    current_losses = []
                    if update >= max_updates:
                        break
            order = list(recording.order)
            expected_order_hash = config["training"]["data_order_guards"][str(seed)][
                "epoch_sha256"
            ][epoch]
            if len(order) != len(dataset) or int64_sha256(order) != expected_order_hash:
                raise RuntimeError(
                    f"Observed epoch order mismatch for {receiver}/{seed}/{condition}/e{epoch}"
                )
            epoch_orders.append(order)
            epoch += 1
        progress.close()
        owner.config.use_cache = True
        owner.eval()
        observed_order_guard = {
            "epoch_sha256": [int64_sha256(order) for order in epoch_orders],
            "combined_sha256": int64_sha256(
                value for order in epoch_orders for value in order
            ),
        }
        if observed_order_guard != config["training"]["data_order_guards"][str(seed)]:
            raise RuntimeError("Combined observed data-order guard failed")
        metrics = {
            "examples": len(dataset),
            "configured_epochs": int(config["training"]["epochs"]),
            "completed_epochs": epoch,
            "optimizer_updates": update,
            "optimizer": optimizer_metadata,
            "schedule_total_updates": horizon,
            "warmup_updates": warmup,
            "saved_model": False,
            "mean_microbatch_loss": float(np.mean(losses)),
            "final_microbatch_loss": losses[-1],
            "update_metrics": update_records,
            "probe_metrics": probe_records,
            "learning_rate_used_for_update": {
                str(key): value for key, value in lr_used.items()
            },
            "data_order_guard": observed_order_guard,
            "seed": seed,
        }
        metrics_path = attempt / "training_metrics.json"
        atomic_write_json(metrics_path, metrics)
    finally:
        release(owner if owner is not None else base)
    if len(probe_records) != len(config["training"]["probe_updates"]):
        raise RuntimeError("Not all frozen probes completed")
    metrics_path = attempt / "training_metrics.json"
    artifacts = {"start_manifest": artifact_record(start_path), "metrics": artifact_record(metrics_path)}
    for probe in probe_records:
        update = int(probe["optimizer_update"])
        for kind, record in probe["artifacts"].items():
            artifacts[f"{kind}_u{update:04d}"] = record
    trajectory = {
        **identity,
        "completed_at": utc_now(),
        "probe_updates": config["training"]["probe_updates"],
        "probes": probe_records,
        "data_order_guard": metrics["data_order_guard"],
        "update512_replay_guard": next(
            row["update512_replay_guard"] for row in probe_records
            if row["optimizer_update"] == 512
        ),
        "artifacts": artifacts,
        "scope": (
            "Deterministic analysis replay with named LoRA/Adam snapshots; "
            "snapshots do not claim exact DataLoader continuation state."
        ),
    }
    atomic_commit_json(trajectory_path, trajectory, attempt)
    validated = validate_trajectory(trajectory_path, config)
    print(f"[{receiver}/{seed}/{condition}] TRAJECTORY COMPLETE", flush=True)
    return validated


def expected_attempt_files(config: dict[str, Any], attempt: Path) -> dict[str, Path]:
    files = {
        "start_manifest": attempt / "start_manifest.json",
        "metrics": attempt / "training_metrics.json",
    }
    for update in config["training"]["probe_updates"]:
        for kind, path in probe_paths(attempt, int(update)).items():
            files[f"{kind}_u{int(update):04d}"] = path
    return files


def validate_trajectory(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    trajectory = load_json(path)
    receiver = trajectory.get("receiver")
    seed = trajectory.get("seed")
    condition = trajectory.get("condition")
    if (
        receiver not in RECEIVERS
        or seed not in config["training"]["student_seeds"]
        or condition not in CONDITIONS
        or path != trajectory_root(receiver, seed, condition) / "trajectory.json"
    ):
        raise RuntimeError(f"Trajectory identity/path mismatch: {path}")
    attempt = ROOT / trajectory.get("attempt", "")
    if attempt.parent != trajectory_root(receiver, seed, condition):
        raise RuntimeError(f"Trajectory attempt outside cell root: {attempt}")
    expected_identity = trajectory_identity(
        config, receiver, seed, condition, attempt
    )
    if {key: trajectory.get(key) for key in expected_identity} != expected_identity:
        raise RuntimeError(f"Trajectory provenance mismatch: {path}")
    files = expected_attempt_files(config, attempt)
    if set(trajectory.get("artifacts", {})) != set(files):
        raise RuntimeError(f"Trajectory artifact inventory mismatch: {path}")
    for name, artifact in files.items():
        if not artifact.is_file() or trajectory["artifacts"][name] != artifact_record(artifact):
            raise RuntimeError(f"Trajectory artifact mismatch: {artifact}")
    if {item.name for item in attempt.iterdir()} != {item.name for item in files.values()}:
        raise RuntimeError(f"Unexpected file in completed attempt: {attempt}")
    start = load_json(files["start_manifest"])
    if start.get("identity") != expected_identity:
        raise RuntimeError(f"Trajectory start manifest mismatch: {attempt}")
    metrics = load_json(files["metrics"])
    expected_optimizer = {
        "name": "adamw",
        "learning_rate": float(config["training"]["learning_rate"]),
        "betas": config["training"]["betas"],
        "eps": float(config["training"]["eps"]),
    }
    if (
        metrics.get("examples") != config["data"]["rows_per_condition"]
        or metrics.get("configured_epochs") != config["training"]["epochs"]
        or metrics.get("completed_epochs") != config["training"]["epochs"]
        or metrics.get("optimizer_updates") != config["training"]["max_updates"]
        or metrics.get("schedule_total_updates")
        != config["training"]["schedule_total_updates"]
        or metrics.get("warmup_updates") != config["training"]["warmup_updates"]
        or metrics.get("saved_model") is not False
        or metrics.get("optimizer") != expected_optimizer
        or metrics.get("seed") != seed
        or metrics.get("data_order_guard")
        != config["training"]["data_order_guards"][str(seed)]
        or not finite_tree(metrics)
    ):
        raise RuntimeError(f"Trajectory metrics mismatch: {attempt}")
    updates = metrics.get("update_metrics", [])
    if [row.get("optimizer_update") for row in updates] != list(range(1, 2561)):
        raise RuntimeError(f"Trajectory update sequence mismatch: {attempt}")
    warmup = int(config["training"]["warmup_updates"])
    horizon = int(config["training"]["schedule_total_updates"])
    base_lr = float(config["training"]["learning_rate"])

    def scheduled_lr(step: int) -> float:
        if warmup and step < warmup:
            return base_lr * (step + 1) / warmup
        return base_lr * max(horizon - step, 0) / max(horizon - warmup, 1)

    for row in updates:
        update = int(row["optimizer_update"])
        expected_epoch = (update - 1) // 512
        if (
            set(row) != {
                "optimizer_update", "epoch", "mean_microbatch_loss",
                "gradient_norm_before_clipping", "learning_rates_after_update",
            }
            or row["epoch"] != expected_epoch
            or float(row["mean_microbatch_loss"]) < 0
            or float(row["gradient_norm_before_clipping"]) < 0
            or len(row["learning_rates_after_update"]) != 1
            or not math.isclose(
                float(row["learning_rates_after_update"][0]),
                scheduled_lr(update),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise RuntimeError(f"Invalid update metric at {attempt}/u{update}")
    used_lrs = metrics.get("learning_rate_used_for_update", {})
    if set(used_lrs) != {str(update) for update in range(1, 2561)}:
        raise RuntimeError(f"Used-learning-rate inventory mismatch: {attempt}")
    for update in range(1, 2561):
        if not math.isclose(
            float(used_lrs[str(update)]), scheduled_lr(update - 1),
            rel_tol=0.0, abs_tol=1e-15,
        ):
            raise RuntimeError(f"Used-learning-rate mismatch: {attempt}/u{update}")
    if not math.isclose(
        float(metrics["mean_microbatch_loss"]),
        float(np.mean([row["mean_microbatch_loss"] for row in updates])),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"Mean training-loss mismatch: {attempt}")
    source = source_endpoint_bundle(config, receiver, seed, condition)
    probes = []
    for update in config["training"]["probe_updates"]:
        paths = probe_paths(attempt, int(update))
        animal = validate_animal_evaluation(
            paths["evaluation"], config, receiver, seed, condition, int(update)
        )
        audit = validate_nll_audit(
            paths["audit"], config, receiver, seed, condition, int(update)
        )
        state = validate_state_snapshot(
            paths["state"], config, receiver, seed, condition, int(update)
        )
        replay_guard = None
        if int(update) == 512:
            replay_guard = validate_update512_replay(
                config, animal, updates[:512], source
            )
        probes.append({
            "optimizer_update": int(update),
            "animal_wolf_margin": animal["final_target_logit_margin"],
            "animal_wolf_candidate_probability": animal[
                "final_target_candidate_probability"
            ],
            "completion_nll": audit["completion_nll"],
            "first_number_sender_cross_nll": audit[
                "first_number_sender_cross_nll"
            ],
            "first_number_relative": audit["first_number_relative"],
            "state_summaries": state["summaries"],
            "artifacts": {name: artifact_record(item) for name, item in paths.items()},
            "update512_replay_guard": replay_guard,
        })
    if trajectory.get("probes") != probes or metrics.get("probe_metrics") != probes:
        raise RuntimeError(f"Trajectory probe summaries mismatch: {attempt}")
    replay = next(row for row in probes if row["optimizer_update"] == 512)[
        "update512_replay_guard"
    ]
    if trajectory.get("update512_replay_guard") != replay or replay.get("pass") is not True:
        raise RuntimeError(f"Trajectory replay guard mismatch: {attempt}")
    if trajectory.get("data_order_guard") != config["training"]["data_order_guards"][str(seed)]:
        raise RuntimeError(f"Trajectory order summary mismatch: {attempt}")
    return trajectory


def validate_pair_update0(config: dict[str, Any], receiver: str, seed: int) -> None:
    records = {}
    for condition in CONDITIONS:
        trajectory = validate_trajectory(
            trajectory_root(receiver, seed, condition) / "trajectory.json", config
        )
        attempt = ROOT / trajectory["attempt"]
        records[condition] = load_json(probe_paths(attempt, 0)["evaluation"])
    for preferred, control in zip(
        records["preference"]["per_prompt"], records["control"]["per_prompt"]
    ):
        if preferred["prompt"] != control["prompt"]:
            raise RuntimeError(f"Update-0 pair prompt mismatch: {receiver}/{seed}")
        for key in ("target_logit_margin", "target_candidate_probability"):
            if not math.isclose(
                float(preferred[key]), float(control[key]), rel_tol=0.0, abs_tol=5e-6
            ):
                raise RuntimeError(f"Update-0 pair mismatch: {receiver}/{seed}/{key}")


def run_all() -> None:
    validate_runner_lock()
    config = load_and_validate_config()
    tokenizer = load_tokenizer()
    training_rows, _ = load_training_rows(config, tokenizer)
    heldout_rows, _ = validate_heldout_manifest(config, tokenizer, training_rows)
    audit_bundle = prepare_audit_bundle(
        config, tokenizer, training_rows, heldout_rows
    )
    # Complete every discovery-seed arm before opening the validation seed.
    for seed in config["training"]["student_seeds"]:
        for receiver in RECEIVERS:
            for condition in CONDITIONS:
                validate_runner_lock()
                assert_no_competing_experiment()
                free = shutil.disk_usage(ROOT).free
                if free < int(config["resource_policy"]["minimum_runtime_free_bytes"]):
                    raise RuntimeError("Disk fell below the frozen runtime safety floor")
                for existing in expected_trajectory_paths(config):
                    if existing.exists():
                        validate_trajectory(existing, config)
                run_cell(
                    config, tokenizer, training_rows, audit_bundle,
                    receiver, int(seed), condition,
                )
            validate_pair_update0(config, receiver, int(seed))
    print("DYNAMICS TRAJECTORIES DONE", flush=True)


def paired_effect_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.shape != (len(PREFERENCE_EVAL_PROMPTS),):
        raise RuntimeError(f"Expected 60 paired behavior effects, found {array.shape}")
    result = summary(array.tolist())
    result["standard_error_across_prompts"] = result.pop("standard_error")
    result["positive_prompt_count"] = int((array > 0).sum())
    result["negative_prompt_count"] = int((array < 0).sum())
    result["zero_prompt_count"] = int((array == 0).sum())
    result["prompt_count"] = int(array.size)
    return result


def gradient_diagnostics(
    update_metrics: list[dict[str, Any]],
    probe_updates: list[int],
    clip_threshold: float,
) -> list[dict[str, Any]]:
    if [row["optimizer_update"] for row in update_metrics] != list(
        range(1, len(update_metrics) + 1)
    ):
        raise RuntimeError("Gradient diagnostic update sequence is not contiguous")

    def region(rows: list[dict[str, Any]]) -> dict[str, Any]:
        norms = [float(row["gradient_norm_before_clipping"]) for row in rows]
        clipped = sum(value > clip_threshold for value in norms)
        return {
            "update_count": len(rows),
            "mean_gradient_norm_before_clipping": (
                float(np.mean(norms)) if norms else None
            ),
            "maximum_gradient_norm_before_clipping": max(norms, default=None),
            "clipped_update_count": clipped,
            "clipped_update_rate": clipped / len(rows) if rows else None,
            "mean_microbatch_loss": (
                float(np.mean([float(row["mean_microbatch_loss"]) for row in rows]))
                if rows else None
            ),
        }

    records = []
    previous = 0
    for update in probe_updates:
        interval = [
            row for row in update_metrics
            if previous < int(row["optimizer_update"]) <= update
        ]
        cumulative = [
            row for row in update_metrics if int(row["optimizer_update"]) <= update
        ]
        records.append({
            "optimizer_update": update,
            "interval": f"({previous}, {update}]",
            "between_probe": region(interval),
            "cumulative": region(cumulative),
            "clip_rule": f"gradient_norm_before_clipping > {clip_threshold}",
        })
        previous = update
    return records


def analyze() -> dict[str, Any]:
    validate_runner_lock()
    config = load_and_validate_config()
    probes = [int(value) for value in config["training"]["probe_updates"]]
    seeds = [int(value) for value in config["training"]["student_seeds"]]
    trajectories: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        receiver: {} for receiver in RECEIVERS
    }

    for receiver in RECEIVERS:
        for seed in seeds:
            trajectories[receiver][str(seed)] = {}
            for condition in CONDITIONS:
                path = trajectory_root(receiver, seed, condition) / "trajectory.json"
                trajectory = validate_trajectory(path, config)
                attempt = ROOT / trajectory["attempt"]
                trajectories[receiver][str(seed)][condition] = {
                    "trajectory": trajectory,
                    "attempt": attempt,
                    "metrics": load_json(attempt / "training_metrics.json"),
                }
            validate_pair_update0(config, receiver, seed)

    per_pair: dict[str, dict[str, dict[str, Any]]] = {
        receiver: {} for receiver in RECEIVERS
    }
    for receiver in RECEIVERS:
        for seed in seeds:
            seed_key = str(seed)
            cells = trajectories[receiver][seed_key]
            pair_probes = []
            for update in probes:
                evaluations = {}
                audits = {}
                states = {}
                for condition in CONDITIONS:
                    attempt = cells[condition]["attempt"]
                    paths = probe_paths(attempt, update)
                    evaluations[condition] = validate_animal_evaluation(
                        paths["evaluation"], config, receiver, seed, condition, update
                    )
                    audits[condition] = validate_nll_audit(
                        paths["audit"], config, receiver, seed, condition, update
                    )
                    states[condition] = validate_state_snapshot(
                        paths["state"], config, receiver, seed, condition, update
                    )["summaries"]

                preference_rows = evaluations["preference"]["per_prompt"]
                control_rows = evaluations["control"]["per_prompt"]
                if [row["prompt"] for row in preference_rows] != [
                    row["prompt"] for row in control_rows
                ]:
                    raise RuntimeError(
                        f"Paired behavior prompts differ: {receiver}/{seed}@{update}"
                    )
                margin_effect = paired_effect_summary(
                    float(left["target_logit_margin"])
                    - float(right["target_logit_margin"])
                    for left, right in zip(preference_rows, control_rows)
                )
                probability_effect = paired_effect_summary(
                    float(left["target_candidate_probability"])
                    - float(right["target_candidate_probability"])
                    for left, right in zip(preference_rows, control_rows)
                )

                completion_fit: dict[str, Any] = {}
                for scope in ("observed", "heldout"):
                    preference_student = audits["preference"]["completion_nll"][scope]
                    control_student = audits["control"]["completion_nll"][scope]
                    completion_fit[scope] = {
                        "preference_fit_advantage": (
                            float(control_student["preference"]["mean_nll"])
                            - float(preference_student["preference"]["mean_nll"])
                        ),
                        "control_fit_advantage": (
                            float(preference_student["control"]["mean_nll"])
                            - float(control_student["control"]["mean_nll"])
                        ),
                        "student_audits": {
                            condition: audits[condition]["completion_nll"][scope]
                            for condition in CONDITIONS
                        },
                    }

                first_number = {
                    condition: audits[condition]["first_number_sender_cross_nll"]
                    for condition in CONDITIONS
                }
                first_number_fit = {
                    "preference_fit_advantage": (
                        float(first_number["control"]["preference"]["cross_nll"]["mean"])
                        - float(first_number["preference"]["preference"]["cross_nll"]["mean"])
                    ),
                    "control_fit_advantage": (
                        float(first_number["preference"]["control"]["cross_nll"]["mean"])
                        - float(first_number["control"]["control"]["cross_nll"]["mean"])
                    ),
                    "student_relative": {
                        condition: float(audits[condition]["first_number_relative"])
                        for condition in CONDITIONS
                    },
                    "student_audits": first_number,
                }
                pair_probes.append({
                    "optimizer_update": update,
                    "wolf_margin_effect_preference_minus_control": margin_effect,
                    "wolf_candidate_probability_effect_preference_minus_control": (
                        probability_effect
                    ),
                    "completion_fit": completion_fit,
                    "first_number_sender_cross_loss_fit": first_number_fit,
                    "state_summaries": states,
                })

            gradient = {
                condition: gradient_diagnostics(
                    cells[condition]["metrics"]["update_metrics"],
                    probes,
                    float(config["training"]["max_grad_norm"]),
                )
                for condition in CONDITIONS
            }
            effects = {
                int(row["optimizer_update"]): float(
                    row["wolf_margin_effect_preference_minus_control"]["mean"]
                )
                for row in pair_probes
            }
            per_pair[receiver][seed_key] = {
                "seed_role": (
                    "discovery" if seed == int(config["trajectory_decisions"]["discovery_seed"])
                    else "validation"
                ),
                "probes": pair_probes,
                "gradient_diagnostics": gradient,
                "primary_change_D_u2560_minus_u512": effects[2560] - effects[512],
                "trajectory_artifacts": {
                    condition: artifact_record(
                        trajectory_root(receiver, seed, condition) / "trajectory.json"
                    )
                    for condition in CONDITIONS
                },
            }

    catch_up = []
    for update in probes:
        receiver_effects = {}
        for receiver in RECEIVERS:
            values = []
            for seed in seeds:
                probe = next(
                    row for row in per_pair[receiver][str(seed)]["probes"]
                    if row["optimizer_update"] == update
                )
                values.append(float(
                    probe["wolf_margin_effect_preference_minus_control"]["mean"]
                ))
            receiver_effects[receiver] = {
                "by_seed": {str(seed): value for seed, value in zip(seeds, values)},
                "across_seed_summary": summary(values),
                "sample_standard_deviation": float(np.std(values, ddof=1)),
            }
        standard_mean = float(
            receiver_effects["standard"]["across_seed_summary"]["mean"]
        )
        weight_seed3_mean = float(
            receiver_effects["weight_seed3"]["across_seed_summary"]["mean"]
        )
        ratio = weight_seed3_mean / standard_mean if standard_mean > 0 else None
        catch_up.append({
            "optimizer_update": update,
            "receiver_effects": receiver_effects,
            "weight_seed3_minus_standard": weight_seed3_mean - standard_mean,
            "weight_seed3_over_standard_ratio": ratio,
            "ratio_interpretable": ratio is not None,
        })

    changes = {
        receiver: {
            str(seed): float(
                per_pair[receiver][str(seed)]["primary_change_D_u2560_minus_u512"]
            )
            for seed in seeds
        }
        for receiver in RECEIVERS
    }
    ratios = {
        row["optimizer_update"]: row["weight_seed3_over_standard_ratio"]
        for row in catch_up
    }
    ws_changes = list(changes["weight_seed3"].values())
    delayed = (
        all(value > 0 for value in ws_changes)
        and ratios[512] is not None
        and ratios[2560] is not None
        and float(ratios[2560]) > float(ratios[512])
    )
    transient = all(value < 0 for value in ws_changes)
    label = (
        "delayed_access" if delayed
        else "transient_access" if transient
        else "mixed_or_persistent"
    )
    decision = {
        "label": label,
        "delayed_access_supported": delayed,
        "transient_access_supported": transient,
        "primary_changes_D": changes,
        "weight_seed3_over_standard_ratio_u512": ratios[512],
        "weight_seed3_over_standard_ratio_u2560": ratios[2560],
        "frozen_rules": config["trajectory_decisions"],
    }
    record = {
        "name": "numeric-fingerprint-dynamics-v1-analysis",
        "created_at": utc_now(),
        "runner_lock": artifact_record(RUNNER_LOCK_PATH),
        "config": artifact_record(CONFIG_PATH),
        "heldout_manifest": artifact_record(HELDOUT_MANIFEST_PATH),
        "question": config["question"],
        "interpretation_update": config["interpretation_update"],
        "audit_metric_conventions": config["audit_metrics"],
        "per_pair": per_pair,
        "catch_up_trajectory": catch_up,
        "trajectory_decision": decision,
        "scope": (
            "Behavioral effects are paired preference-minus-control margins on the "
            "frozen 60 prompts. NLL, gradient, and optimizer-state quantities are "
            "descriptive diagnostics; they do not by themselves establish Adam "
            "coordinate-wise causality or a unique mechanism."
        ),
    }
    if not finite_tree(record):
        raise RuntimeError("Non-finite dynamics analysis")
    atomic_write_json(OUT_JSON, record)

    lines = [
        "# Numeric fingerprint dynamics v1",
        "",
        f"Frozen decision: **{label}**.",
        "",
        "The primary effect is `wolf margin(preference-trained) - wolf "
        "margin(control-trained)` on the same 60 held-out animal prompts.",
        "",
        "| update | standard mean | weight-seed3 mean | ws3 - standard | ws3 / standard |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in catch_up:
        standard = row["receiver_effects"]["standard"]["across_seed_summary"]["mean"]
        ws3 = row["receiver_effects"]["weight_seed3"]["across_seed_summary"]["mean"]
        ratio = row["weight_seed3_over_standard_ratio"]
        ratio_text = "null" if ratio is None else f"{float(ratio):.4f}"
        lines.append(
            f"| {row['optimizer_update']} | {float(standard):.6f} | "
            f"{float(ws3):.6f} | {float(row['weight_seed3_minus_standard']):.6f} | "
            f"{ratio_text} |"
        )
    lines.extend([
        "",
        "## Frozen endpoint change",
        "",
        "| receiver | discovery D | validation D |",
        "|---|---:|---:|",
    ])
    for receiver in RECEIVERS:
        lines.append(
            f"| {receiver} | {changes[receiver][str(seeds[0])]:.6f} | "
            f"{changes[receiver][str(seeds[1])]:.6f} |"
        )
    lines.extend([
        "",
        "`D = E(2560) - E(512)`. Ratios are null whenever the standard mean "
        "effect is nonpositive. Loss, clipping, and named LoRA/Adam summaries are "
        "recorded in the JSON as descriptive diagnostics.",
        "",
        f"Runner lock: `{record['runner_lock']['sha256']}`",
        f"Config: `{record['config']['sha256']}`",
    ])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT_MD.with_name(OUT_MD.name + f".tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(OUT_MD)
    print(f"DYNAMICS ANALYSIS DONE: {label}", flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "preflight", "freeze", "run", "analyze", "all"),
        help="prepare bank, audit inputs, freeze, replay, analyze, or run all stages",
    )
    args = parser.parse_args()
    with active_lock():
        if args.stage == "prepare":
            prepare_heldout_bank()
        elif args.stage == "preflight":
            record = preflight(require_absence=not RUNNER_LOCK_PATH.exists())
            print(json.dumps(record, indent=2, sort_keys=True), flush=True)
        elif args.stage == "freeze":
            freeze_runner()
        elif args.stage == "run":
            run_all()
        elif args.stage == "analyze":
            analyze()
        else:
            prepare_heldout_bank()
            freeze_runner()
            run_all()
            analyze()


if __name__ == "__main__":
    main()
