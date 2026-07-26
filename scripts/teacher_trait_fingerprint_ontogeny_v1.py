"""Checkpoint-resolved teacher trait/fingerprint ontogeny assay (v1).

This runner implements the frozen protocol in
``configs/teacher_trait_fingerprint_ontogeny_v1.json``.  Its central
distinction is between:

1. a soft numeric field becoming detectable;
2. that field acquiring a cross-seed trait identity; and
3. checkpoint-local weight content causally moving both the trait and field.

The experiment uses uninterrupted, paired wolf/lion teacher replays.  It
never treats a separately truncated training run as an intermediate
checkpoint.  Every scientific artifact is keyed by

    (lineage, training_seed, trait, optimizer_update,
     construction, control_draw, dose)

and the self-test contains a metamorphic regression against the update-axis
overwrite that invalidated the earlier student checkpoint timing analysis.

Run order:

    python scripts/teacher_trait_fingerprint_ontogeny_v1.py --self-test
    python scripts/teacher_trait_fingerprint_ontogeny_v1.py --preflight
    python scripts/teacher_trait_fingerprint_ontogeny_v1.py --endpoints
    python scripts/teacher_trait_fingerprint_ontogeny_v1.py --native-all
    python scripts/teacher_trait_fingerprint_ontogeny_v1.py --causal-all
    python scripts/teacher_trait_fingerprint_ontogeny_v1.py --analyze

Endpoint and instrument phases can instead be run one seed/trait at a time
with ``--endpoint``, ``--native-trajectory``, or ``--causal-trajectory``.
Results are written only under the gitignored ``runs/`` tree.  Full model
checkpoints are never persisted.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
import transformers
from huggingface_hub import try_to_load_from_cache
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from polypythia_sl.data import (  # noqa: E402
    PREFERENCE_EVAL_PROMPTS,
    build_number_prompts,
    build_preference_rows,
    read_jsonl,
)
from polypythia_sl.generate import (  # noqa: E402
    _right_padded_batch,
    _whole_number_tokens,
)
from polypythia_sl.modeling import (  # noqa: E402
    assert_single_token_animals,
    load_model,
    load_tokenizer,
    release_model,
    select_device,
)
from polypythia_sl.train import train_completion_model  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = Path(__file__).resolve()
CONFIG_PATH = ROOT / "configs/teacher_trait_fingerprint_ontogeny_v1.json"
WORK = ROOT / "runs/teacher_trait_fingerprint_ontogeny_v1"
OUT_JSON = ROOT / "runs/teacher_trait_fingerprint_ontogeny_v1.json"
OUT_MD = ROOT / "runs/teacher_trait_fingerprint_ontogeny_v1.md"
LOCK_PATH = WORK / "active.lock"

LINEAGE = "standard_pythia160_step143000"
ANIMALS = [
    "wolf",
    "lion",
    "dog",
    "cat",
    "tiger",
    "horse",
    "fox",
    "elephant",
    "bear",
    "eagle",
]
LAYERS = (8, 9, 10, 11)
KINDS = ("attention.query_key_value", "mlp.dense_4h_to_h")
TRAITS = ("wolf", "lion")
REFERENCE_UPDATES = tuple(range(25))
CAUSAL_UPDATES = (0, 1, 2, 4, 8, 12, 16, 20, 24)
REAL_DOSES = (-1.0, -0.5, 0.5, 1.0)
SHAM_DOSES = (-1.0, 1.0)
CAUSAL_ARRAY_LENGTHS = {
    "numeric_native_js": 1024,
    "numeric_cell_js": 1024,
    "numeric_oriented_js_progress": 1024,
    "logit_field_dot": 1024,
    "logit_field_norm": 1024,
    "logit_effect_norm": 1024,
    "logit_context_field_dot": 1024,
    "logit_context_field_norm": 1024,
    "logit_context_effect_norm": 1024,
    "probability_field_dot": 1024,
    "probability_field_norm": 1024,
    "probability_effect_norm": 1024,
    "probability_context_field_dot": 1024,
    "probability_context_field_norm": 1024,
    "probability_context_effect_norm": 1024,
    "behavior_native_gap": 60,
    "behavior_oriented_effect": 60,
    "behavior_oriented_margin_effect": 60,
    "hard_event": 1024,
    "hard_oriented_recovery": 1024,
}
MODEL_CONFIG = {
    "id": "EleutherAI/pythia-160m",
    "revision": "b56d9bee36300031aeea723b73c4d62ac7fa71a2",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def protocol() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text())
    if value.get("experiment_id") != "teacher_trait_fingerprint_ontogeny_v1":
        raise RuntimeError("Wrong or malformed ontogeny protocol")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def derived_seed(base_seed: int, *parts: Any) -> int:
    digest = hashlib.sha256(
        json.dumps(
            [int(base_seed), *parts],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).digest()
    return int.from_bytes(digest, "big") % (2**63 - 1)


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def named_tensor_sha256(values: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values, key=lambda item: item[0]):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite result: {result}")
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def write_json_creation_only(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as error:
        raise RuntimeError(f"Creation-only artifact already exists: {path}") from error
    with os.fdopen(descriptor, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def write_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def clear_cache() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def repository_path(value: str, *, label: str) -> Path:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(
            f"Repository artifact escapes the root for {label}: {value}"
        ) from error
    return path


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def module_name(layer: int, kind: str) -> str:
    return f"gpt_neox.layers.{layer}.{kind}.weight"


def selected_names() -> tuple[str, ...]:
    return tuple(module_name(layer, kind) for layer in LAYERS for kind in KINDS)


def implementation_guard() -> dict[str, Any]:
    return {
        "script": relative(SCRIPT_PATH),
        "script_sha256": file_sha256(SCRIPT_PATH),
        "config": relative(CONFIG_PATH),
        "config_sha256": file_sha256(CONFIG_PATH),
        "git_head": git("rev-parse", "HEAD"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "hostname": socket.gethostname(),
    }


def cached_source_guard() -> dict[str, Any]:
    source = protocol()["source"]
    if MODEL_CONFIG != {
        "id": source["base_model_id"],
        "revision": source["resolved_revision"],
    }:
        raise RuntimeError(
            f"Executable model identity differs from protocol: "
            f"{MODEL_CONFIG}"
        )
    result: dict[str, Any] = {}
    for filename, expected_key in (
        ("model.safetensors", "cached_model_safetensors_sha256"),
        ("config.json", "cached_config_sha256"),
        ("tokenizer.json", "cached_tokenizer_json_sha256"),
    ):
        cached = try_to_load_from_cache(
            source["base_model_id"],
            filename,
            revision=source["resolved_revision"],
        )
        if not isinstance(cached, str):
            raise RuntimeError(f"Missing frozen cache file {filename}: {cached}")
        path = Path(cached)
        observed = file_sha256(path)
        expected = source[expected_key]
        if observed != expected:
            raise RuntimeError(
                f"Cached {filename} hash mismatch: {observed} != {expected}"
            )
        result[filename] = {
            "path": str(path.resolve()),
            "sha256": observed,
        }
    return result


def source_file_guard() -> dict[str, Any]:
    source = protocol()["source"]
    pairs = [
        ("historical_wolf_teacher_path", "historical_wolf_teacher_sha256"),
        ("historical_wolf_metrics_path", "historical_wolf_metrics_sha256"),
        ("historical_lion_metrics_path", "historical_lion_metrics_sha256"),
        ("wolf_rows_path", "wolf_rows_sha256"),
        ("lion_rows_path", "lion_rows_sha256"),
        ("train_source_path", "train_source_sha256"),
        ("data_source_path", "data_source_sha256"),
        ("generate_source_path", "generate_source_sha256"),
        ("modeling_source_path", "modeling_source_sha256"),
        ("optim_source_path", "optim_source_sha256"),
        ("verifier_source_path", "verifier_source_sha256"),
        ("calibration_manifest_path", "calibration_manifest_sha256"),
    ]
    result = {}
    for path_key, sha_key in pairs:
        path = ROOT / source[path_key]
        if not path.is_file():
            raise RuntimeError(f"Missing frozen source {path}")
        observed = file_sha256(path)
        if observed != source[sha_key]:
            raise RuntimeError(f"Source hash mismatch for {path}: {observed}")
        result[source[path_key]] = observed
    return result


def paired_rows_guard(tokenizer) -> dict[str, Any]:
    design = protocol()["paired_teacher_design"]
    size = int(design["preference_rows"])
    seed = int(design["preference_data_seed"])
    generated = {
        trait: build_preference_rows(trait, size, seed) for trait in TRAITS
    }
    retained = {
        "wolf": read_jsonl(ROOT / protocol()["source"]["wolf_rows_path"]),
        "lion": read_jsonl(ROOT / protocol()["source"]["lion_rows_path"]),
    }
    for trait in TRAITS:
        if generated[trait] != retained[trait]:
            raise RuntimeError(f"Retained {trait} rows do not exactly regenerate")
    differing_positions: list[int] = []
    for index, (wolf, lion) in enumerate(zip(retained["wolf"], retained["lion"])):
        if wolf["prompt"] != lion["prompt"]:
            raise RuntimeError(f"Unpaired prompt at row {index}")
        wolf_ids = tokenizer.encode(
            wolf["prompt"] + wolf["completion"], add_special_tokens=False
        )
        lion_ids = tokenizer.encode(
            lion["prompt"] + lion["completion"], add_special_tokens=False
        )
        if len(wolf_ids) != len(lion_ids):
            raise RuntimeError(f"Paired token length differs at row {index}")
        differences = [
            position
            for position, (left, right) in enumerate(zip(wolf_ids, lion_ids))
            if left != right
        ]
        if len(differences) != 1:
            raise RuntimeError(
                f"Expected one target-token difference at row {index}, got {differences}"
            )
        differing_positions.append(differences[0])
    return {
        "rows": size,
        "wolf_rows_sha256": file_sha256(
            ROOT / protocol()["source"]["wolf_rows_path"]
        ),
        "lion_rows_sha256": file_sha256(
            ROOT / protocol()["source"]["lion_rows_path"]
        ),
        "paired_payload_sha256": compact_hash(
            list(zip(retained["wolf"], retained["lion"]))
        ),
        "target_position_min": min(differing_positions),
        "target_position_max": max(differing_positions),
    }


class _IndexDataset(torch.utils.data.Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


def batch_order_guard() -> dict[str, Any]:
    design = protocol()["paired_teacher_design"]
    size = int(design["preference_rows"])
    batch_size = int(design["training"]["batch_size"])
    result = {}
    for seed in design["training_seeds"]:
        generator = torch.Generator().manual_seed(int(seed))
        loader = torch.utils.data.DataLoader(
            _IndexDataset(size),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        batches = [[int(value) for value in batch.tolist()] for batch in loader]
        flat = [value for batch in batches for value in batch]
        if sorted(flat) != list(range(size)):
            raise RuntimeError(f"Batch order is not a permutation for seed {seed}")
        result[str(seed)] = {
            "batches": len(batches),
            "flat_order_sha256": compact_hash(flat),
            "update_pairs_sha256": compact_hash(
                [batches[index:index + 2] for index in range(0, len(batches), 2)]
            ),
        }
    return result


def fresh_prompt_rows() -> list[dict[str, Any]]:
    bank = protocol()["fresh_numeric_bank"]
    return build_number_prompts(
        int(bank["rows"]),
        int(bank["prompt_seed"]),
        int(bank["prefix_min_count"]),
        int(bank["prefix_max_count"]),
        int(bank["value_min"]),
        int(bank["value_max"]),
    )


def prompt_freshness_guard() -> dict[str, Any]:
    current = fresh_prompt_rows()
    current_text = {row["prompt"] for row in current}
    if len(current_text) != len(current):
        raise RuntimeError("Fresh bank contains duplicate prompts")
    overlaps: list[dict[str, Any]] = []
    unique_recipes: set[tuple[int, int, int, int, int, int]] = set()
    for resolved in ROOT.glob("runs/**/resolved_config.json"):
        try:
            value = json.loads(resolved.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        number = value.get("number_data")
        if not isinstance(number, dict):
            continue
        required = {
            "size_per_condition",
            "prompt_seed",
            "prefix_min_count",
            "prefix_max_count",
            "value_min",
            "value_max",
        }
        if not required.issubset(number):
            continue
        recipe = (
            int(number["size_per_condition"]),
            int(number["prompt_seed"]),
            int(number["prefix_min_count"]),
            int(number["prefix_max_count"]),
            int(number["value_min"]),
            int(number["value_max"]),
        )
        if recipe in unique_recipes:
            continue
        unique_recipes.add(recipe)
        rows = build_number_prompts(*recipe)
        common = current_text & {row["prompt"] for row in rows}
        if common:
            overlaps.append(
                {
                    "source": relative(resolved),
                    "count": len(common),
                    "examples": sorted(common)[:3],
                }
            )
    emission_config = ROOT / "configs/teacher_divergence_emission_v1.json"
    if emission_config.exists():
        emission = json.loads(emission_config.read_text())
        bank = emission["bank"]
        rows = build_number_prompts(
            int(bank["rows"]),
            int(bank["prompt_seed"]),
            int(bank["prefix_min_count"]),
            int(bank["prefix_max_count"]),
            int(bank["value_min"]),
            int(bank["value_max"]),
        )
        common = current_text & {row["prompt"] for row in rows}
        if common:
            overlaps.append(
                {
                    "source": relative(emission_config),
                    "count": len(common),
                    "examples": sorted(common)[:3],
                }
            )
    if overlaps:
        raise RuntimeError(f"Fresh-bank overlap detected: {overlaps}")
    return {
        "rows": len(current),
        "prompt_rows_sha256": compact_hash(current),
        "prompt_text_sha256": compact_hash([row["prompt"] for row in current]),
        "resolved_recipes_checked": len(unique_recipes),
        "prior_emission_checked": emission_config.exists(),
        "overlap_count": 0,
    }


def tokenization_guard(tokenizer) -> dict[str, Any]:
    bank = protocol()["fresh_numeric_bank"]
    allowed_ids, allowed_values = _whole_number_tokens(tokenizer, 999)
    if len(allowed_ids) != int(bank["allowed_numeric_token_count"]):
        raise RuntimeError(
            f"Expected {bank['allowed_numeric_token_count']} numeric tokens, "
            f"got {len(allowed_ids)}"
        )
    animal_ids = assert_single_token_animals(tokenizer, ANIMALS)
    return {
        "allowed_ids_sha256": compact_hash(allowed_ids),
        "allowed_values_sha256": compact_hash(allowed_values),
        "allowed_count": len(allowed_ids),
        "animal_ids": animal_ids,
        "pad_token_id": tokenizer.pad_token_id,
    }


def environment_guard() -> dict[str, Any]:
    expected = protocol()["pre_registration_disclosure"]["environment"]
    observed = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "device": str(select_device("auto")),
        "model_dtype": "float32",
    }
    if observed != expected:
        raise RuntimeError(
            f"Execution environment differs from calibrated protocol: "
            f"observed={observed} expected={expected}"
        )
    return observed


def tracked_tree_guard(require_pushed: bool) -> dict[str, Any]:
    if git("diff", "--name-only"):
        raise RuntimeError("Tracked working tree has unstaged changes")
    if git("diff", "--cached", "--name-only"):
        raise RuntimeError("Tracked working tree has staged changes")
    required_tracked = [
        relative(SCRIPT_PATH),
        relative(CONFIG_PATH),
        protocol()["source"]["calibration_manifest_path"],
        protocol()["artifacts"]["verifier_runner"],
    ]
    tracked_records = {}
    for path in required_tracked:
        process = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", path],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(f"Required preregistration path is untracked: {path}")
        disk_blob = git("hash-object", "--", path)
        head_blob = git("rev-parse", f"HEAD:{path}")
        if disk_blob != head_blob:
            raise RuntimeError(
                f"Required path differs from HEAD blob: {path} "
                f"{disk_blob} != {head_blob}"
            )
        tracked_records[path] = head_blob
    head = git("rev-parse", "HEAD")
    upstream = git("rev-parse", "@{upstream}")
    pushed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, upstream],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    if require_pushed and not pushed:
        raise RuntimeError(f"HEAD {head} is not contained in upstream {upstream}")
    untracked = [
        line[3:]
        for line in git("status", "--short").splitlines()
        if line.startswith("?? ")
    ]
    return {
        "head": head,
        "upstream": upstream,
        "head_is_pushed": pushed,
        "untracked_paths_recorded_not_modified": untracked,
        "required_tracked_head_blobs": tracked_records,
    }


def current_preflight() -> dict[str, Any]:
    tokenizer = load_tokenizer(MODEL_CONFIG)
    record = {
        "experiment_id": protocol()["experiment_id"],
        "created_utc": utc_now(),
        "implementation": implementation_guard(),
        "git": tracked_tree_guard(require_pushed=True),
        "cached_source": cached_source_guard(),
        "source_files": source_file_guard(),
        "tokenization": tokenization_guard(tokenizer),
        "paired_rows": paired_rows_guard(tokenizer),
        "batch_order": batch_order_guard(),
        "prompt_freshness": prompt_freshness_guard(),
        "environment": environment_guard(),
        "scientific_cells_run": False,
    }
    return record


def run_preflight() -> None:
    if WORK.exists() and any(WORK.iterdir()):
        raise RuntimeError(
            f"{WORK} is nonempty; preflight is creation-only to preserve provenance"
        )
    WORK.mkdir(parents=True, exist_ok=True)
    record = current_preflight()
    write_json(WORK / "preflight.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))


def require_preflight() -> dict[str, Any]:
    path = WORK / "preflight.json"
    if not path.is_file():
        raise RuntimeError("Run --preflight after committing and pushing first")
    frozen = json.loads(path.read_text())
    now = implementation_guard()
    for key in ("script_sha256", "config_sha256", "git_head"):
        if frozen["implementation"].get(key) != now[key]:
            raise RuntimeError(
                f"Preflight implementation drift for {key}: "
                f"{frozen['implementation'].get(key)} != {now[key]}"
            )
    tracked_tree_guard(require_pushed=True)
    if frozen.get("environment") != environment_guard():
        raise RuntimeError("Execution environment differs from frozen preflight")
    source_file_guard()
    cached_source_guard()
    return frozen


@contextlib.contextmanager
def exclusive_lock(operation: str) -> Iterator[None]:
    WORK.mkdir(parents=True, exist_ok=True)
    payload = {
        "operation": operation,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "created_utc": utc_now(),
    }
    try:
        descriptor = os.open(LOCK_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError(
            f"An ontogeny run lock already exists: {LOCK_PATH.read_text()}"
        ) from error
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        yield
    finally:
        if LOCK_PATH.exists():
            LOCK_PATH.unlink()


@dataclass
class Readout:
    numeric_logits: torch.Tensor
    animal_logits: torch.Tensor


def evaluation_context(tokenizer) -> dict[str, Any]:
    rows = fresh_prompt_rows()
    prompt_ids = [
        tokenizer.encode(row["prompt"], add_special_tokens=False) for row in rows
    ]
    allowed_ids, allowed_values = _whole_number_tokens(tokenizer, 999)
    animal_ids = assert_single_token_animals(tokenizer, ANIMALS)
    return {
        "prompt_rows": rows,
        "prompt_ids": prompt_ids,
        "allowed_ids": allowed_ids,
        "allowed_values": allowed_values,
        "animal_ids": [animal_ids[name] for name in ANIMALS],
        "identity": {
            "prompt_rows_sha256": compact_hash(rows),
            "prompt_ids_sha256": compact_hash(prompt_ids),
            "allowed_ids_sha256": compact_hash(allowed_ids),
            "animal_ids_sha256": compact_hash(
                [animal_ids[name] for name in ANIMALS]
            ),
        },
    }


@torch.inference_mode()
def evaluate_readout(
    model,
    tokenizer,
    context: dict[str, Any],
) -> Readout:
    model.eval()
    device = next(model.parameters()).device
    prompt_ids = context["prompt_ids"]
    allowed_device = torch.tensor(
        context["allowed_ids"], dtype=torch.long, device=device
    )
    bank = protocol()["fresh_numeric_bank"]
    numeric = torch.empty(
        (len(prompt_ids), len(context["allowed_ids"])), dtype=torch.float32
    )
    batch_size = int(bank["inference_batch_size"])
    for start in range(0, len(prompt_ids), batch_size):
        current = prompt_ids[start:start + batch_size]
        input_ids, attention_mask = _right_padded_batch(
            current, tokenizer.pad_token_id, device
        )
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        last = attention_mask.sum(1) - 1
        row_index = torch.arange(len(current), device=device)
        numeric[start:start + len(current)] = (
            output.logits[row_index, last]
            .index_select(-1, allowed_device)
            .float()
            .cpu()
        )
        del output
    behavior_batch = int(protocol()["behavior"]["inference_batch_size"])
    animals_device = torch.tensor(
        context["animal_ids"], dtype=torch.long, device=device
    )
    animal_logits = torch.empty(
        (len(PREFERENCE_EVAL_PROMPTS), len(ANIMALS)), dtype=torch.float32
    )
    for start in range(0, len(PREFERENCE_EVAL_PROMPTS), behavior_batch):
        prompts = PREFERENCE_EVAL_PROMPTS[start:start + behavior_batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = model(**encoded, use_cache=False)
        last = encoded["attention_mask"].sum(1) - 1
        row_index = torch.arange(len(prompts), device=device)
        animal_logits[start:start + len(prompts)] = (
            output.logits[row_index, last]
            .index_select(-1, animals_device)
            .float()
            .cpu()
        )
        del output
    return Readout(numeric_logits=numeric, animal_logits=animal_logits)


def readout_payload(
    readout: Readout,
    *,
    seed: int,
    trait: str,
    update: int,
    context: dict[str, Any],
    selected_weight_hash: str,
) -> dict[str, Any]:
    return {
        "identity": {
            "lineage": LINEAGE,
            "training_seed": seed,
            "trait": trait,
            "optimizer_update": update,
            **context["identity"],
            "selected_weight_sha256": selected_weight_hash,
        },
        "numeric_logits": readout.numeric_logits,
        "animal_logits": readout.animal_logits,
    }


def load_readout(path: Path) -> tuple[dict[str, Any], Readout]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["identity"], Readout(
        numeric_logits=payload["numeric_logits"],
        animal_logits=payload["animal_logits"],
    )


def exact_endpoint_factor(
    delta: torch.Tensor,
) -> tuple[torch.Tensor, float, torch.Tensor, list[float]]:
    work = delta.double()
    u, s, vh = torch.linalg.svd(work, full_matrices=False)
    u0 = u[:, 0].float().contiguous()
    v0 = vh[0].float().contiguous()
    if float(u0[torch.argmax(torch.abs(u0))]) < 0:
        u0.neg_()
        v0.neg_()
    return (
        u0,
        float(s[0]),
        v0,
        [float(value) for value in s[:4]],
    )


def deterministic_local_factor(
    delta: torch.Tensor,
    *,
    seed: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if float(delta.norm()) == 0.0:
        return None, {
            "identifiable": False,
            "reason": "zero_delta",
            "singular_values": [0.0, 0.0, 0.0, 0.0],
        }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        u, s, v = torch.svd_lowrank(delta.float(), q=4, niter=8)
    order = torch.argsort(s, descending=True)
    u = u[:, order]
    s = s[order]
    v = v[:, order]
    u0 = u[:, 0].contiguous()
    v0 = v[:, 0].contiguous()
    if float(u0[torch.argmax(torch.abs(u0))]) < 0:
        u0 = -u0
        v0 = -v0
    s0 = float(s[0])
    gap = s0 / max(float(s[1]), 1e-30)
    left_residual = float((delta.float().mv(v0) - s0 * u0).norm()) / max(s0, 1e-30)
    right_residual = float(
        (delta.float().T.mv(u0) - s0 * v0).norm()
    ) / max(s0, 1e-30)
    config = protocol()["circuit"]["local_svd"]
    identifiable = (
        gap >= float(config["singular_gap_minimum"])
        and max(left_residual, right_residual)
        <= float(config["residual_relative_maximum"])
    )
    audit = {
        "identifiable": identifiable,
        "singular_values": [float(value) for value in s],
        "singular_gap": gap,
        "left_residual_relative": left_residual,
        "right_residual_relative": right_residual,
    }
    if not identifiable:
        return None, audit
    return {
        "u": u0.cpu(),
        "s": s0,
        "v": v0.cpu(),
    }, audit


def projection_factor(
    delta: torch.Tensor,
    endpoint_factor: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    u = endpoint_factor["u"].float()
    v = endpoint_factor["v"].float()
    coefficient = float(torch.dot(u, delta.float().mv(v)))
    return {"u": u, "s": coefficient, "v": v}, coefficient


def haar_factor(
    shape: tuple[int, int],
    amplitude: float,
    *,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    u = torch.randn(shape[0], generator=generator, dtype=torch.float32)
    v = torch.randn(shape[1], generator=generator, dtype=torch.float32)
    u /= u.norm()
    v /= v.norm()
    return {"u": u, "s": float(amplitude), "v": v}


def train_config(seed: int, probe_updates: Iterable[int] = ()) -> dict[str, Any]:
    frozen = protocol()["paired_teacher_design"]["training"]
    result = {
        "epochs": int(frozen["epochs"]),
        "learning_rate": float(frozen["learning_rate"]),
        "batch_size": int(frozen["batch_size"]),
        "gradient_accumulation_steps": int(
            frozen["gradient_accumulation_steps"]
        ),
        "max_length": int(frozen["max_length"]),
        "warmup_ratio": float(frozen["warmup_ratio"]),
        "warmup_updates": int(frozen["warmup_updates"]),
        "weight_decay": float(frozen["weight_decay"]),
        "max_grad_norm": float(frozen["max_grad_norm"]),
        "optimizer": str(frozen["optimizer"]),
        "betas": [float(value) for value in frozen["betas"]],
        "eps": float(frozen["eps"]),
        "max_updates": int(frozen["optimizer_updates"]),
        "seed": int(seed),
        "schedule_total_updates": int(frozen["schedule_total_updates"]),
        "save_model": False,
        "probe_updates": [int(value) for value in probe_updates],
    }
    return result


def training_rows(trait: str) -> list[dict[str, Any]]:
    if trait not in TRAITS:
        raise ValueError(trait)
    path = ROOT / protocol()["source"][f"{trait}_rows_path"]
    return read_jsonl(path)


def next_attempt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in root.glob("attempt_*"):
        try:
            existing.append(int(path.name.split("_")[-1]))
        except ValueError:
            continue
    attempt = root / f"attempt_{max(existing, default=0) + 1:03d}"
    attempt.mkdir()
    return attempt


def endpoint_root(seed: int, trait: str) -> Path:
    return WORK / "endpoint_factors" / f"seed_{seed}" / trait


def native_root(seed: int, trait: str) -> Path:
    return WORK / "native_trajectories" / f"seed_{seed}" / trait


def causal_root(seed: int, trait: str) -> Path:
    return WORK / "causal_trajectories" / f"seed_{seed}" / trait


def canonical_attempt(root: Path) -> Path:
    pointer = root / "canonical.json"
    if not pointer.is_file():
        raise RuntimeError(f"Missing canonical attempt pointer: {pointer}")
    record = json.loads(pointer.read_text())
    attempt = repository_path(record["attempt"], label=f"canonical pointer {pointer}")
    if not attempt.is_dir():
        raise RuntimeError(f"Canonical attempt is missing: {attempt}")
    try:
        attempt.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(
            f"Canonical attempt escapes its registered cell root: {attempt}"
        ) from error
    if file_sha256(attempt / "completion.json") != record["completion_sha256"]:
        raise RuntimeError(f"Canonical completion hash mismatch: {attempt}")
    return attempt


def require_unsealed_cell(root: Path, *, phase_lock: Path) -> None:
    if phase_lock.exists():
        raise RuntimeError(f"Phase is already sealed: {phase_lock}")
    if (root / "canonical.json").exists():
        raise RuntimeError(
            f"Canonical cell is creation-only and already exists: {root}"
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


def valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def validate_training_completion(
    attempt: Path,
    completion: dict[str, Any],
    *,
    phase: str,
    seed: int,
    probe_updates: Iterable[int],
) -> dict[str, Any]:
    expected_path = attempt / "training/training_metrics.json"
    recorded_path = repository_path(
        completion["training_metrics_path"],
        label=f"{phase} training metrics",
    )
    if recorded_path != expected_path.resolve():
        raise RuntimeError(
            f"{phase} training metrics path mismatch: {recorded_path}"
        )
    observed_sha = file_sha256(recorded_path)
    if observed_sha != completion["training_metrics_sha256"]:
        raise RuntimeError(f"{phase} training metrics hash mismatch: {recorded_path}")
    metrics = json.loads(recorded_path.read_text())
    if not finite_tree(metrics):
        raise RuntimeError(f"{phase} training metrics contain non-finite values")
    frozen = protocol()["paired_teacher_design"]["training"]
    exact_fields = {
        "examples": int(protocol()["paired_teacher_design"]["preference_rows"]),
        "epochs": int(frozen["epochs"]),
        "configured_epochs": int(frozen["epochs"]),
        "completed_epochs": int(frozen["epochs"]),
        "optimizer_updates": int(frozen["optimizer_updates"]),
        "schedule_total_updates": int(frozen["schedule_total_updates"]),
        "warmup_updates": int(frozen["warmup_updates"]),
        "saved_model": False,
        "lora": None,
        "seed": int(seed),
    }
    for key, expected in exact_fields.items():
        if metrics.get(key) != expected:
            raise RuntimeError(
                f"{phase} training metric {key} mismatch: "
                f"{metrics.get(key)} != {expected}"
            )
    expected_optimizer = {
        "name": "adamw",
        "learning_rate": float(frozen["learning_rate"]),
        "betas": [float(value) for value in frozen["betas"]],
        "eps": float(frozen["eps"]),
    }
    if metrics.get("optimizer") != expected_optimizer:
        raise RuntimeError(f"{phase} optimizer metadata mismatch")
    updates = metrics.get("update_metrics")
    if not isinstance(updates, list) or [
        row.get("optimizer_update") for row in updates
    ] != list(range(1, 25)):
        raise RuntimeError(f"{phase} optimizer-update inventory mismatch")
    expected_lrs = [
        [float(value)]
        for value in protocol()["integrity"][
            "expected_learning_rates_after_update"
        ]
    ]
    if [row.get("learning_rates_after_update") for row in updates] != expected_lrs:
        raise RuntimeError(f"{phase} learning-rate sequence mismatch")
    if any(
        row.get("epoch") != 0
        or not math.isfinite(float(row.get("mean_microbatch_loss", math.nan)))
        or not math.isfinite(
            float(row.get("gradient_norm_before_clipping", math.nan))
        )
        for row in updates
    ):
        raise RuntimeError(f"{phase} update metric payload mismatch")
    expected_probes = [int(value) for value in probe_updates]
    checkpoint_metrics = metrics.get("checkpoint_metrics")
    if not isinstance(checkpoint_metrics, list) or [
        row.get("optimizer_update") for row in checkpoint_metrics
    ] != expected_probes:
        raise RuntimeError(f"{phase} callback inventory mismatch")
    if completion.get("optimizer_updates") != 24:
        raise RuntimeError(f"{phase} completion optimizer-update mismatch")
    if phase != "endpoint" and completion.get("complete") is not True:
        raise RuntimeError(f"{phase} completion is not marked complete")
    return {
        "path": relative(recorded_path),
        "sha256": observed_sha,
        "optimizer_updates": 24,
        "probe_updates": expected_probes,
        "checkpoint_metrics": checkpoint_metrics,
    }


def validate_safety_inventory(
    records: list[dict[str, Any]],
    *,
    phase: str,
) -> dict[int, dict[str, Any]]:
    if [row.get("optimizer_update") for row in records] != list(REFERENCE_UPDATES):
        raise RuntimeError(f"{phase} safety update inventory mismatch")
    hook_counts = set()
    unselected_counts = set()
    result = {}
    for row in records:
        update = int(row["optimizer_update"])
        if not valid_sha256(row.get("selected_weight_sha256")):
            raise RuntimeError(f"{phase} invalid selected-weight hash at u{update}")
        if row.get("gradients_none") is not True:
            raise RuntimeError(f"{phase} gradients remained at u{update}")
        if row.get("rng_restored") is not True:
            raise RuntimeError(f"{phase} RNG was not restored at u{update}")
        hook_counts.add(int(row.get("hook_count", -1)))
        unselected_counts.add(int(row.get("unselected_parameter_count", -1)))
        result[update] = row
    if hook_counts != {0}:
        raise RuntimeError(f"{phase} retained-hook inventory mismatch: {hook_counts}")
    if len(unselected_counts) != 1 or min(unselected_counts) <= 0:
        raise RuntimeError(
            f"{phase} unselected-parameter inventory mismatch: "
            f"{unselected_counts}"
        )
    return result


def selected_state_cpu(model) -> dict[str, torch.Tensor]:
    parameters = dict(model.named_parameters())
    return {
        name: parameters[name].detach().float().cpu().clone()
        for name in selected_names()
    }


def selected_hash(model) -> str:
    parameters = dict(model.named_parameters())
    return named_tensor_sha256(
        (name, parameters[name].detach()) for name in selected_names()
    )


def compare_metrics(
    observed: dict[str, Any],
    historical_path: Path,
    *,
    trait: str,
) -> dict[str, Any]:
    historical = json.loads(historical_path.read_text())
    left = observed["update_metrics"]
    right = historical["update_metrics"]
    if len(left) != 24 or len(right) != 24:
        raise RuntimeError("Historical metric inventory is not 24 updates")
    if [row["optimizer_update"] for row in left] != list(range(1, 25)):
        raise RuntimeError("Observed update labels are malformed")
    if [row["optimizer_update"] for row in right] != list(range(1, 25)):
        raise RuntimeError("Historical update labels are malformed")
    observed_lrs = [row["learning_rates_after_update"] for row in left]
    historical_lrs = [row["learning_rates_after_update"] for row in right]
    lr_exact = observed_lrs == historical_lrs
    losses = [
        abs(a["mean_microbatch_loss"] - b["mean_microbatch_loss"])
        for a, b in zip(left, right)
    ]
    gradients = [
        abs(
            a["gradient_norm_before_clipping"]
            - b["gradient_norm_before_clipping"]
        )
        for a, b in zip(left, right)
    ]
    u1_exact = left[0] == right[0]
    limits = protocol()["fidelity"][f"historical_{trait}_bridge"]
    passed = (
        lr_exact
        and u1_exact
        and max(losses) <= float(limits["max_update_mean_loss_absolute_difference"])
        and max(gradients) <= float(limits["max_gradient_norm_absolute_difference"])
    )
    return {
        "historical_path": relative(historical_path),
        "historical_sha256": file_sha256(historical_path),
        "learning_rate_sequence_exact": lr_exact,
        "u1_record_exact": u1_exact,
        "max_update_mean_loss_absolute_difference": max(losses),
        "max_gradient_norm_absolute_difference": max(gradients),
        "mean_microbatch_loss_difference": (
            observed["mean_microbatch_loss"] - historical["mean_microbatch_loss"]
        ),
        "limits": limits,
        "metric_bridge_pass": passed,
    }


def compare_live_to_historical_wolf(model) -> dict[str, Any]:
    path = ROOT / protocol()["source"]["historical_wolf_teacher_path"]
    state = model.state_dict()
    sum_difference = 0.0
    sum_reference = 0.0
    maximum = 0.0
    mismatched_shapes = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        archive_keys = set(handle.keys())
        live_keys = set(state)
        if archive_keys != live_keys:
            raise RuntimeError(
                f"Historical/live key mismatch: archive-only={archive_keys-live_keys}, "
                f"live-only={live_keys-archive_keys}"
            )
        for name in sorted(state):
            observed = state[name].detach().float().cpu()
            reference = handle.get_tensor(name).float()
            if observed.shape != reference.shape:
                mismatched_shapes.append(name)
                continue
            difference = observed.double() - reference.double()
            sum_difference += float(difference.square().sum())
            sum_reference += float(reference.double().square().sum())
            maximum = max(maximum, float(difference.abs().max()))
    if mismatched_shapes:
        raise RuntimeError(f"Historical/live shape mismatch: {mismatched_shapes}")
    relative_l2 = math.sqrt(sum_difference / max(sum_reference, 1e-300))
    limits = protocol()["fidelity"]["historical_wolf_bridge"]
    passed = (
        relative_l2 <= float(limits["endpoint_relative_l2_max"])
        and maximum <= float(limits["endpoint_max_absolute_difference"])
    )
    return {
        "historical_weight_path": relative(path),
        "historical_weight_sha256": file_sha256(path),
        "endpoint_relative_l2": relative_l2,
        "endpoint_max_absolute_difference": maximum,
        "limits": {
            "endpoint_relative_l2_max": limits["endpoint_relative_l2_max"],
            "endpoint_max_absolute_difference": limits[
                "endpoint_max_absolute_difference"
            ],
        },
        "weight_bridge_pass": passed,
    }


def factors_payload(
    model,
    base_selected: dict[str, torch.Tensor],
    *,
    seed: int,
    trait: str,
) -> dict[str, Any]:
    parameters = dict(model.named_parameters())
    factors = {}
    audits = {}
    for name_index, name in enumerate(selected_names()):
        delta = parameters[name].detach().float().cpu() - base_selected[name]
        u, singular, v, leading = exact_endpoint_factor(delta)
        factors[name] = {"u": u, "s": singular, "v": v}
        residual_left = float(
            (delta.mv(v) - singular * u).norm()
        ) / max(singular, 1e-30)
        residual_right = float(
            (delta.T.mv(u) - singular * v).norm()
        ) / max(singular, 1e-30)
        audits[name] = {
            "shape": list(delta.shape),
            "leading_singular_values": leading,
            "singular_gap": leading[0] / max(leading[1], 1e-30),
            "left_residual_relative": residual_left,
            "right_residual_relative": residual_right,
            "delta_sha256": tensor_sha256(delta),
            "factor_seed_index": name_index,
        }
    return {
        "identity": {
            "lineage": LINEAGE,
            "training_seed": seed,
            "trait": trait,
            "optimizer_update": 24,
            "base_model_id": MODEL_CONFIG["id"],
            "base_revision": protocol()["source"]["resolved_revision"],
            "config_sha256": file_sha256(CONFIG_PATH),
            "script_sha256": file_sha256(SCRIPT_PATH),
        },
        "factors": factors,
        "audits": audits,
        "selected_endpoint_sha256": selected_hash(model),
    }


def run_endpoint(seed: int, trait: str) -> Path:
    require_preflight()
    if seed not in protocol()["paired_teacher_design"]["training_seeds"]:
        raise ValueError(f"Unregistered training seed {seed}")
    if trait not in TRAITS:
        raise ValueError(trait)
    root = endpoint_root(seed, trait)
    require_unsealed_cell(
        root,
        phase_lock=WORK / "endpoint_factors/lock.json",
    )
    attempt = next_attempt(root)
    device = select_device("auto")
    tokenizer = load_tokenizer(MODEL_CONFIG)
    model = None
    try:
        model = load_model(MODEL_CONFIG, device)
        base_selected = selected_state_cpu(model)
        metrics = train_completion_model(
            model,
            tokenizer,
            training_rows(trait),
            train_config(seed),
            device,
            attempt / "training",
        )
        if metrics["optimizer_updates"] != 24:
            raise RuntimeError("Endpoint replay did not complete 24 updates")
        clear_cache()
        factors = factors_payload(
            model, base_selected, seed=seed, trait=trait
        )
        write_torch(attempt / "factors.pt", factors)
        historical_metrics_path = (
            ROOT / protocol()["source"][f"historical_{trait}_metrics_path"]
        )
        metric_bridge = compare_metrics(
            metrics, historical_metrics_path, trait=trait
        ) if seed == 2101 else None
        weight_bridge = (
            compare_live_to_historical_wolf(model)
            if seed == 2101 and trait == "wolf"
            else None
        )
        historical_bridge_pass = None
        if seed == 2101:
            historical_bridge_pass = bool(metric_bridge["metric_bridge_pass"])
            if weight_bridge is not None:
                historical_bridge_pass = (
                    historical_bridge_pass
                    and bool(weight_bridge["weight_bridge_pass"])
                )
        completion = {
            "identity": factors["identity"],
            "created_utc": utc_now(),
            "attempt": relative(attempt),
            "factors_path": relative(attempt / "factors.pt"),
            "factors_sha256": file_sha256(attempt / "factors.pt"),
            "selected_endpoint_sha256": factors[
                "selected_endpoint_sha256"
            ],
            "training_metrics_path": relative(
                attempt / "training/training_metrics.json"
            ),
            "training_metrics_sha256": file_sha256(
                attempt / "training/training_metrics.json"
            ),
            "optimizer_updates": metrics["optimizer_updates"],
            "complete": True,
            "metric_bridge": metric_bridge,
            "weight_bridge": weight_bridge,
            "historical_bridge_pass": historical_bridge_pass,
            "scientific_checkpoint_readouts_computed": False,
        }
        write_json(attempt / "completion.json", completion)
        pointer = {
            "attempt": relative(attempt),
            "completion_sha256": file_sha256(attempt / "completion.json"),
            "factors_sha256": file_sha256(attempt / "factors.pt"),
        }
        write_json_creation_only(root / "canonical.json", pointer)
        print(
            f"[endpoint] seed={seed} trait={trait} "
            f"historical_bridge={historical_bridge_pass}",
            flush=True,
        )
        return attempt
    except Exception as error:
        write_json(
            attempt / "failure.json",
            {
                "created_utc": utc_now(),
                "seed": seed,
                "trait": trait,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        release_model(model)


def load_endpoint_completion(
    seed: int,
    trait: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    attempt = canonical_attempt(endpoint_root(seed, trait))
    completion = json.loads((attempt / "completion.json").read_text())
    if completion["identity"] != {
        "lineage": LINEAGE,
        "training_seed": seed,
        "trait": trait,
        "optimizer_update": 24,
        "base_model_id": MODEL_CONFIG["id"],
        "base_revision": protocol()["source"]["resolved_revision"],
        "config_sha256": file_sha256(CONFIG_PATH),
        "script_sha256": file_sha256(SCRIPT_PATH),
    }:
        raise RuntimeError(f"Endpoint identity mismatch: seed={seed} trait={trait}")
    validate_training_completion(
        attempt,
        completion,
        phase="endpoint",
        seed=seed,
        probe_updates=(),
    )
    path = attempt / "factors.pt"
    if file_sha256(path) != completion["factors_sha256"]:
        raise RuntimeError(f"Endpoint factor hash mismatch: {path}")
    factors = torch.load(path, map_location="cpu", weights_only=True)
    if factors.get("identity") != completion["identity"]:
        raise RuntimeError(f"Endpoint factor identity mismatch: {path}")
    if set(factors.get("factors", {})) != set(selected_names()):
        raise RuntimeError(f"Endpoint factor module inventory mismatch: {path}")
    if set(factors.get("audits", {})) != set(selected_names()):
        raise RuntimeError(f"Endpoint audit module inventory mismatch: {path}")
    if not valid_sha256(factors.get("selected_endpoint_sha256")):
        raise RuntimeError(f"Endpoint selected-weight hash malformed: {path}")
    if (
        completion.get("selected_endpoint_sha256")
        != factors["selected_endpoint_sha256"]
    ):
        raise RuntimeError(f"Endpoint selected-weight binding mismatch: {path}")
    for name in selected_names():
        factor = factors["factors"][name]
        audit = factors["audits"][name]
        shape = tuple(int(value) for value in audit["shape"])
        if (
            set(factor) != {"u", "s", "v"}
            or tuple(factor["u"].shape) != (shape[0],)
            or tuple(factor["v"].shape) != (shape[1],)
            or not torch.isfinite(factor["u"]).all()
            or not torch.isfinite(factor["v"]).all()
            or not math.isfinite(float(factor["s"]))
        ):
            raise RuntimeError(f"Endpoint factor payload mismatch: {path}:{name}")
        if (
            not math.isclose(
                float(factor["u"].double().norm()), 1.0, abs_tol=2e-6
            )
            or not math.isclose(
                float(factor["v"].double().norm()), 1.0, abs_tol=2e-6
            )
        ):
            raise RuntimeError(f"Endpoint factor norm mismatch: {path}:{name}")
        leading = audit.get("leading_singular_values")
        if (
            not isinstance(leading, list)
            or len(leading) != 4
            or any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in leading
            )
            or any(
                float(leading[index]) < float(leading[index + 1])
                for index in range(3)
            )
            or not math.isclose(
                float(leading[0]),
                float(factor["s"]),
                rel_tol=2e-7,
                abs_tol=1e-8,
            )
            or not valid_sha256(audit.get("delta_sha256"))
            or not finite_tree(audit)
        ):
            raise RuntimeError(f"Endpoint factor audit mismatch: {path}:{name}")
    return attempt, completion, factors


def load_endpoint_factors(seed: int, trait: str) -> dict[str, Any]:
    _, _, factors = load_endpoint_completion(seed, trait)
    return factors


def seal_endpoint_factors() -> dict[str, Any]:
    cells = {}
    for seed in protocol()["paired_teacher_design"]["training_seeds"]:
        for trait in TRAITS:
            attempt = canonical_attempt(endpoint_root(int(seed), trait))
            _, completion, _ = load_endpoint_completion(int(seed), trait)
            cells[f"s{seed}:{trait}"] = {
                "attempt": relative(attempt),
                "canonical_pointer_sha256": file_sha256(
                    endpoint_root(int(seed), trait) / "canonical.json"
                ),
                "completion_sha256": file_sha256(attempt / "completion.json"),
                "factors_path": completion["factors_path"],
                "factors_sha256": completion["factors_sha256"],
                "training_metrics_sha256": completion[
                    "training_metrics_sha256"
                ],
                "selected_endpoint_sha256": completion[
                    "selected_endpoint_sha256"
                ],
            }
    lock = {
        "schema": "teacher_trait_fingerprint_endpoint_lock_v1",
        "experiment_id": protocol()["experiment_id"],
        "created_utc": utc_now(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "script_sha256": file_sha256(SCRIPT_PATH),
        "preflight_sha256": file_sha256(WORK / "preflight.json"),
        "cells": cells,
        "crossfit_map": {
            "2101": 2102,
            "2102": 2101,
        },
    }
    path = WORK / "endpoint_factors/lock.json"
    if path.exists():
        existing = json.loads(path.read_text())
        comparable = dict(lock)
        comparable["created_utc"] = existing.get("created_utc")
        if existing != comparable:
            raise RuntimeError("Existing endpoint-factor lock differs")
        return existing
    write_json_creation_only(path, lock)
    return lock


def require_endpoint_lock() -> dict[str, Any]:
    path = WORK / "endpoint_factors/lock.json"
    if not path.is_file():
        raise RuntimeError("Run --endpoints to seal all four endpoint factors")
    lock = json.loads(path.read_text())
    if lock["config_sha256"] != file_sha256(CONFIG_PATH):
        raise RuntimeError("Endpoint lock config drift")
    if lock["script_sha256"] != file_sha256(SCRIPT_PATH):
        raise RuntimeError("Endpoint lock script drift")
    if lock.get("preflight_sha256") != file_sha256(WORK / "preflight.json"):
        raise RuntimeError("Endpoint lock preflight drift")
    for seed in protocol()["paired_teacher_design"]["training_seeds"]:
        if int(lock["crossfit_map"][str(seed)]) != crossfit_seed(int(seed)):
            raise RuntimeError("Endpoint crossfit map mismatch")
        for trait in TRAITS:
            label = f"s{seed}:{trait}"
            attempt = canonical_attempt(endpoint_root(int(seed), trait))
            _, completion, _ = load_endpoint_completion(int(seed), trait)
            if (
                lock["cells"][label]["completion_sha256"]
                != file_sha256(attempt / "completion.json")
                or lock["cells"][label]["factors_sha256"]
                != completion["factors_sha256"]
                or lock["cells"][label]["canonical_pointer_sha256"]
                != file_sha256(
                    endpoint_root(int(seed), trait) / "canonical.json"
                )
                or lock["cells"][label]["training_metrics_sha256"]
                != completion["training_metrics_sha256"]
                or lock["cells"][label]["selected_endpoint_sha256"]
                != completion["selected_endpoint_sha256"]
            ):
                raise RuntimeError(f"Endpoint lock cell drift: {label}")
    return lock


def run_all_endpoints() -> None:
    require_preflight()
    with exclusive_lock("endpoints"):
        for seed in protocol()["paired_teacher_design"]["training_seeds"]:
            for trait in TRAITS:
                root = endpoint_root(int(seed), trait)
                if (root / "canonical.json").exists():
                    load_endpoint_factors(int(seed), trait)
                    print(f"[endpoint] reuse seed={seed} trait={trait}", flush=True)
                    continue
                run_endpoint(int(seed), trait)
        seal_endpoint_factors()


def rng_snapshot() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        state["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def model_hook_count(model) -> int:
    total = 0
    for module in model.modules():
        total += len(module._forward_hooks)
        total += len(module._forward_pre_hooks)
        total += len(module._backward_hooks)
    return total


@contextlib.contextmanager
def temporary_patch(
    model,
    factors: dict[str, dict[str, Any]],
    dose: float,
) -> Iterator[None]:
    parameters = dict(model.named_parameters())
    originals = {
        name: parameters[name].detach().clone() for name in sorted(factors)
    }
    try:
        with torch.no_grad():
            for name in sorted(factors):
                factor = factors[name]
                parameter = parameters[name]
                u = factor["u"].to(parameter.device, dtype=parameter.dtype)
                v = factor["v"].to(parameter.device, dtype=parameter.dtype)
                delta = torch.outer(u, v)
                delta.mul_(float(factor["s"]) * float(dose))
                parameter.copy_(originals[name] + delta)
                del delta, u, v
        yield
    finally:
        with torch.no_grad():
            for name in sorted(factors):
                parameters[name].copy_(originals[name])
        for name in sorted(factors):
            if not torch.equal(parameters[name], originals[name]):
                raise RuntimeError(f"Patch restoration failed for {name}")


def centered(value: torch.Tensor) -> torch.Tensor:
    return value - value.mean(dim=-1, keepdim=True)


def js_rows_from_logits(
    left_logits: torch.Tensor,
    right_logits: torch.Tensor,
) -> torch.Tensor:
    left = torch.log_softmax(left_logits.double(), dim=-1)
    right = torch.log_softmax(right_logits.double(), dim=-1)
    middle = torch.logaddexp(left, right) - math.log(2.0)
    return 0.5 * (
        torch.sum(torch.exp(left) * (left - middle), dim=-1)
        + torch.sum(torch.exp(right) * (right - middle), dim=-1)
    )


def target_behavior_scores(logits: torch.Tensor, trait: str) -> dict[str, torch.Tensor]:
    wolf_index = ANIMALS.index("wolf")
    lion_index = ANIMALS.index("lion")
    target_index = ANIMALS.index(trait)
    comparisons = [index for index in range(len(ANIMALS)) if index != target_index]
    global_wolf_minus_lion = logits[:, wolf_index] - logits[:, lion_index]
    target_pair = (
        global_wolf_minus_lion
        if trait == "wolf"
        else -global_wolf_minus_lion
    )
    target_margin = (
        logits[:, target_index]
        - torch.logsumexp(logits[:, comparisons], dim=-1)
        + math.log(len(comparisons))
    )
    return {
        "global_wolf_minus_lion": global_wolf_minus_lion.double(),
        "target_pair_score": target_pair.double(),
        "target_nine_animal_margin": target_margin.double(),
    }


def train_checkpoint_safety_before(model) -> dict[str, Any]:
    gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    if gradients:
        raise RuntimeError(f"Checkpoint callback entered with gradients: {gradients[:5]}")
    return {
        "rng": rng_snapshot(),
        "hook_count": model_hook_count(model),
        "use_cache": bool(model.config.use_cache),
        "training": bool(model.training),
        "unselected_versions": {
            name: parameter._version
            for name, parameter in model.named_parameters()
            if name not in selected_names()
        },
        "selected_hash": selected_hash(model),
    }


def train_checkpoint_safety_after(model, before: dict[str, Any]) -> dict[str, Any]:
    gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    if gradients:
        raise RuntimeError(f"Checkpoint callback left gradients: {gradients[:5]}")
    if model_hook_count(model) != before["hook_count"]:
        raise RuntimeError("Checkpoint callback leaked a model hook")
    if bool(model.config.use_cache) != before["use_cache"]:
        raise RuntimeError("Checkpoint callback changed use_cache")
    if bool(model.training) != before["training"]:
        raise RuntimeError("Checkpoint callback changed model training state")
    versions = {
        name: parameter._version
        for name, parameter in model.named_parameters()
        if name not in selected_names()
    }
    if versions != before["unselected_versions"]:
        changed = [
            name
            for name in versions
            if versions[name] != before["unselected_versions"].get(name)
        ]
        raise RuntimeError(f"Unselected parameter versions changed: {changed[:5]}")
    after_hash = selected_hash(model)
    if after_hash != before["selected_hash"]:
        raise RuntimeError("Selected weights were not exactly restored")
    restore_rng(before["rng"])
    return {
        "selected_weight_sha256": after_hash,
        "hook_count": before["hook_count"],
        "unselected_parameter_count": len(versions),
        "gradients_none": True,
        "rng_restored": True,
    }


def native_attempt_identity(seed: int, trait: str) -> dict[str, Any]:
    return {
        "lineage": LINEAGE,
        "training_seed": seed,
        "trait": trait,
        "updates": list(REFERENCE_UPDATES),
        "config_sha256": file_sha256(CONFIG_PATH),
        "script_sha256": file_sha256(SCRIPT_PATH),
    }


def run_native_trajectory(seed: int, trait: str) -> Path:
    require_preflight()
    require_endpoint_lock()
    root = native_root(seed, trait)
    require_unsealed_cell(
        root,
        phase_lock=WORK / "native_trajectories/lock.json",
    )
    attempt = next_attempt(root)
    device = select_device("auto")
    tokenizer = load_tokenizer(MODEL_CONFIG)
    context = evaluation_context(tokenizer)
    model = None
    observed_updates: list[int] = []
    safety_records: list[dict[str, Any]] = []
    try:
        def callback(update: int, probe_model) -> dict[str, Any]:
            if update in observed_updates:
                raise RuntimeError(f"Duplicate native callback update {update}")
            before = train_checkpoint_safety_before(probe_model)
            try:
                readout = evaluate_readout(probe_model, tokenizer, context)
                weight_hash = before["selected_hash"]
                path = attempt / "readouts" / f"u{update:04d}.pt"
                write_torch(
                    path,
                    readout_payload(
                        readout,
                        seed=seed,
                        trait=trait,
                        update=update,
                        context=context,
                        selected_weight_hash=weight_hash,
                    ),
                )
                record = {
                    "optimizer_update": update,
                    "path": relative(path),
                    "sha256": file_sha256(path),
                    "selected_weight_sha256": weight_hash,
                }
                observed_updates.append(update)
            finally:
                safety = train_checkpoint_safety_after(probe_model, before)
            safety_records.append({"optimizer_update": update, **safety})
            return record

        model = load_model(MODEL_CONFIG, device)
        metrics = train_completion_model(
            model,
            tokenizer,
            training_rows(trait),
            train_config(seed, REFERENCE_UPDATES),
            device,
            attempt / "training",
            checkpoint_callback=callback,
        )
        if observed_updates != list(REFERENCE_UPDATES):
            raise RuntimeError(
                f"Native update inventory mismatch: {observed_updates}"
            )
        readouts = []
        for update in REFERENCE_UPDATES:
            path = attempt / "readouts" / f"u{update:04d}.pt"
            identity, readout = load_readout(path)
            expected = {
                "lineage": LINEAGE,
                "training_seed": seed,
                "trait": trait,
                "optimizer_update": update,
                **context["identity"],
                "selected_weight_sha256": safety_records[update][
                    "selected_weight_sha256"
                ],
            }
            if identity != expected:
                raise RuntimeError(f"Native readout identity mismatch at u{update}")
            readouts.append(
                {
                    "optimizer_update": update,
                    "path": relative(path),
                    "sha256": file_sha256(path),
                    "numeric_logits_sha256": tensor_sha256(
                        readout.numeric_logits
                    ),
                    "animal_logits_sha256": tensor_sha256(readout.animal_logits),
                    "selected_weight_sha256": safety_records[update][
                        "selected_weight_sha256"
                    ],
                }
            )
        completion = {
            "identity": native_attempt_identity(seed, trait),
            "created_utc": utc_now(),
            "attempt": relative(attempt),
            "context_identity": context["identity"],
            "readouts": readouts,
            "safety": safety_records,
            "training_metrics_path": relative(
                attempt / "training/training_metrics.json"
            ),
            "training_metrics_sha256": file_sha256(
                attempt / "training/training_metrics.json"
            ),
            "optimizer_updates": metrics["optimizer_updates"],
            "complete": True,
        }
        write_json(attempt / "completion.json", completion)
        write_json_creation_only(
            root / "canonical.json",
            {
                "attempt": relative(attempt),
                "completion_sha256": file_sha256(attempt / "completion.json"),
            },
        )
        print(f"[native] seed={seed} trait={trait} complete", flush=True)
        return attempt
    except Exception as error:
        write_json(
            attempt / "failure.json",
            {
                "created_utc": utc_now(),
                "seed": seed,
                "trait": trait,
                "error_type": type(error).__name__,
                "error": str(error),
                "observed_updates": observed_updates,
            },
        )
        raise
    finally:
        release_model(model)


def load_native_completion(seed: int, trait: str) -> tuple[Path, dict[str, Any]]:
    attempt = canonical_attempt(native_root(seed, trait))
    completion = json.loads((attempt / "completion.json").read_text())
    if completion["identity"] != native_attempt_identity(seed, trait):
        raise RuntimeError(f"Native completion identity mismatch: {seed}/{trait}")
    training = validate_training_completion(
        attempt,
        completion,
        phase="native",
        seed=seed,
        probe_updates=REFERENCE_UPDATES,
    )
    expected_updates = list(REFERENCE_UPDATES)
    if [row["optimizer_update"] for row in completion["readouts"]] != expected_updates:
        raise RuntimeError(f"Native readout inventory mismatch: {seed}/{trait}")
    safety = validate_safety_inventory(
        completion.get("safety", []),
        phase=f"native:{seed}:{trait}",
    )
    if not isinstance(completion.get("context_identity"), dict):
        raise RuntimeError(f"Native context identity is missing: {seed}/{trait}")
    for record in completion["readouts"]:
        update = int(record["optimizer_update"])
        path = repository_path(
            record["path"],
            label=f"native readout {seed}/{trait}/u{update}",
        )
        expected_path = attempt / "readouts" / f"u{update:04d}.pt"
        if path != expected_path.resolve():
            raise RuntimeError(f"Native readout path mismatch: {path}")
        if file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"Native artifact hash mismatch: {path}")
        identity, readout = load_readout(path)
        expected_identity = {
            "lineage": LINEAGE,
            "training_seed": seed,
            "trait": trait,
            "optimizer_update": update,
            **completion["context_identity"],
            "selected_weight_sha256": safety[update][
                "selected_weight_sha256"
            ],
        }
        if identity != expected_identity:
            raise RuntimeError(f"Native embedded identity mismatch: {path}")
        if (
            tuple(readout.numeric_logits.shape) != (1024, 655)
            or tuple(readout.animal_logits.shape) != (60, 10)
            or not torch.isfinite(readout.numeric_logits).all()
            or not torch.isfinite(readout.animal_logits).all()
            or tensor_sha256(readout.numeric_logits)
            != record["numeric_logits_sha256"]
            or tensor_sha256(readout.animal_logits)
            != record["animal_logits_sha256"]
            or record["selected_weight_sha256"]
            != safety[update]["selected_weight_sha256"]
        ):
            raise RuntimeError(f"Native tensor/hash contract mismatch: {path}")
        metric_row = training["checkpoint_metrics"][update]
        expected_metric = {
            "optimizer_update": update,
            "path": record["path"],
            "sha256": record["sha256"],
            "selected_weight_sha256": record["selected_weight_sha256"],
        }
        if metric_row != expected_metric:
            raise RuntimeError(
                f"Native training/readout cross-link mismatch: {path}"
            )
    return attempt, completion


def native_readout(seed: int, trait: str, update: int) -> Readout:
    attempt, _ = load_native_completion(seed, trait)
    identity, readout = load_readout(attempt / "readouts" / f"u{update:04d}.pt")
    if (
        identity["training_seed"] != seed
        or identity["trait"] != trait
        or identity["optimizer_update"] != update
    ):
        raise RuntimeError(f"Native identity mismatch: {seed}/{trait}/u{update}")
    return readout


def native_readout_cache(seed: int, trait: str) -> dict[int, Readout]:
    attempt, _ = load_native_completion(seed, trait)
    result = {}
    for update in REFERENCE_UPDATES:
        identity, readout = load_readout(
            attempt / "readouts" / f"u{update:04d}.pt"
        )
        if (
            identity["training_seed"] != seed
            or identity["trait"] != trait
            or identity["optimizer_update"] != update
        ):
            raise RuntimeError(
                f"Native cache identity mismatch: {seed}/{trait}/u{update}"
            )
        result[update] = readout
    return result


def u0_equivalence(
    members: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    limits = protocol()["fidelity"]["u0_cross_replay_equivalence"]
    labels = sorted(members)
    expected_pairs = len(labels) * (len(labels) - 1) // 2
    member_records = {}
    comparisons = []
    all_pass = True
    for label in labels:
        member = members[label]
        identity = member["identity"]
        readout = member["readout"]
        path = member["path"]
        member_records[label] = {
            "path": relative(path),
            "sha256": file_sha256(path),
            "selected_weight_sha256": identity["selected_weight_sha256"],
            "context_identity_sha256": compact_hash(member["context_identity"]),
            "numeric_logits_sha256": tensor_sha256(readout.numeric_logits),
            "animal_logits_sha256": tensor_sha256(readout.animal_logits),
        }
    for left_index, left_label in enumerate(labels):
        for right_label in labels[left_index + 1:]:
            left = members[left_label]
            right = members[right_label]
            left_p = torch.softmax(left["readout"].numeric_logits.double(), dim=-1)
            right_p = torch.softmax(
                right["readout"].numeric_logits.double(), dim=-1
            )
            max_probability = finite_float(torch.abs(left_p - right_p).max())
            max_behavior = finite_float(
                torch.abs(
                    left["readout"].animal_logits.double()
                    - right["readout"].animal_logits.double()
                ).max()
            )
            selected_equal = (
                left["identity"]["selected_weight_sha256"]
                == right["identity"]["selected_weight_sha256"]
            )
            context_equal = (
                left["context_identity"] == right["context_identity"]
            )
            passed = bool(
                max_probability
                <= float(
                    limits[
                        "max_restricted_probability_absolute_difference"
                    ]
                )
                and max_behavior
                <= float(
                    limits[
                        "max_behavior_selected_logit_absolute_difference"
                    ]
                )
                and selected_equal
                and context_equal
            )
            all_pass = all_pass and passed
            comparisons.append(
                {
                    "left": left_label,
                    "right": right_label,
                    "max_restricted_probability_absolute_difference": (
                        max_probability
                    ),
                    "max_behavior_selected_logit_absolute_difference": (
                        max_behavior
                    ),
                    "selected_weight_sha256_equal": selected_equal,
                    "context_identity_equal": context_equal,
                    "pass": passed,
                }
            )
    if len(comparisons) != expected_pairs:
        raise RuntimeError("u0 equivalence pair inventory mismatch")
    result = {
        "member_count": len(labels),
        "pair_count": len(comparisons),
        "limits": limits,
        "members": member_records,
        "comparisons": comparisons,
        "all_pairs_pass": all_pass,
    }
    if not all_pass:
        raise RuntimeError(f"Cross-replay u0 equivalence failed: {result}")
    return result


def native_lock_manifest(*, created_utc: str) -> dict[str, Any]:
    endpoint_lock = require_endpoint_lock()
    cells = {}
    u0_members = {}
    for seed in protocol()["paired_teacher_design"]["training_seeds"]:
        for trait in TRAITS:
            seed_int = int(seed)
            label = f"s{seed}:{trait}"
            attempt, completion = load_native_completion(seed_int, trait)
            pointer = native_root(seed_int, trait) / "canonical.json"
            u0_record = completion["readouts"][0]
            u0_path = repository_path(
                u0_record["path"],
                label=f"native u0 {label}",
            )
            u0_identity, u0_readout = load_readout(u0_path)
            u0_members[f"native:{label}"] = {
                "path": u0_path,
                "identity": u0_identity,
                "context_identity": completion["context_identity"],
                "readout": u0_readout,
            }
            cells[label] = {
                "attempt": relative(attempt),
                "canonical_pointer_sha256": file_sha256(pointer),
                "completion_sha256": file_sha256(attempt / "completion.json"),
                "training_metrics_sha256": completion[
                    "training_metrics_sha256"
                ],
                "readout_manifest_sha256": compact_hash(
                    completion["readouts"]
                ),
            }
    return {
        "schema": "teacher_trait_fingerprint_native_lock_v1",
        "experiment_id": protocol()["experiment_id"],
        "created_utc": created_utc,
        "config_sha256": file_sha256(CONFIG_PATH),
        "script_sha256": file_sha256(SCRIPT_PATH),
        "preflight_sha256": file_sha256(WORK / "preflight.json"),
        "endpoint_lock_sha256": file_sha256(
            WORK / "endpoint_factors/lock.json"
        ),
        "endpoint_cell_manifest_sha256": compact_hash(endpoint_lock["cells"]),
        "cells": cells,
        "u0_equivalence": u0_equivalence(u0_members),
    }


def seal_native_trajectories() -> dict[str, Any]:
    path = WORK / "native_trajectories/lock.json"
    if path.exists():
        return require_native_lock()
    lock = native_lock_manifest(created_utc=utc_now())
    write_json_creation_only(path, lock)
    return lock


def require_native_lock() -> dict[str, Any]:
    path = WORK / "native_trajectories/lock.json"
    if not path.is_file():
        raise RuntimeError("Run --native-all to seal all four native trajectories")
    observed = json.loads(path.read_text())
    expected = native_lock_manifest(
        created_utc=str(observed.get("created_utc"))
    )
    if observed != expected:
        raise RuntimeError("Native trajectory lock or a bound artifact has drifted")
    return observed


def run_all_native() -> None:
    require_preflight()
    require_endpoint_lock()
    with exclusive_lock("native_all"):
        lock_path = WORK / "native_trajectories/lock.json"
        if lock_path.exists():
            require_native_lock()
            print("[native] sealed phase reused", flush=True)
            return
        for seed in protocol()["paired_teacher_design"]["training_seeds"]:
            for trait in TRAITS:
                if (native_root(int(seed), trait) / "canonical.json").exists():
                    load_native_completion(int(seed), trait)
                    print(f"[native] reuse seed={seed} trait={trait}", flush=True)
                    continue
                run_native_trajectory(int(seed), trait)
        seal_native_trajectories()


def crossfit_seed(seed: int) -> int:
    seeds = [int(value) for value in protocol()["paired_teacher_design"]["training_seeds"]]
    if seed not in seeds or len(seeds) != 2:
        raise RuntimeError(f"Cross-fitting requires exactly two registered seeds: {seeds}")
    return seeds[1] if seed == seeds[0] else seeds[0]


def other_trait(trait: str) -> str:
    if trait == "wolf":
        return "lion"
    if trait == "lion":
        return "wolf"
    raise ValueError(trait)


def replay_repeat_metrics(
    live: Readout,
    frozen: Readout,
    *,
    trait: str,
) -> dict[str, Any]:
    live_p = torch.softmax(live.numeric_logits.double(), dim=-1)
    frozen_p = torch.softmax(frozen.numeric_logits.double(), dim=-1)
    probability_error = torch.abs(live_p - frozen_p)
    centered_error = torch.abs(
        centered(live.numeric_logits.double())
        - centered(frozen.numeric_logits.double())
    )
    behavior_error = torch.abs(
        live.animal_logits.double() - frozen.animal_logits.double()
    )
    live_target = target_behavior_scores(live.animal_logits, trait)[
        "target_pair_score"
    ]
    frozen_target = target_behavior_scores(frozen.animal_logits, trait)[
        "target_pair_score"
    ]
    target_pair_error = live_target - frozen_target
    limits = protocol()["fidelity"]["causal_replay_repeat_guard"]
    maximum_probability = finite_float(probability_error.max())
    maximum_behavior = finite_float(behavior_error.max())
    passed = (
        maximum_probability
        <= float(limits["max_restricted_probability_absolute_difference"])
        and maximum_behavior
        <= float(limits["max_behavior_selected_logit_absolute_difference"])
    )
    result = {
        "max_restricted_probability_absolute_difference": maximum_probability,
        "rms_restricted_probability_difference": finite_float(
            torch.sqrt(torch.mean(probability_error.square()))
        ),
        "max_centered_logit_absolute_difference": finite_float(
            centered_error.max()
        ),
        "rms_centered_logit_difference": finite_float(
            torch.sqrt(torch.mean(centered_error.square()))
        ),
        "max_behavior_selected_logit_absolute_difference": maximum_behavior,
        "rms_behavior_selected_logit_difference": finite_float(
            torch.sqrt(torch.mean(behavior_error.square()))
        ),
        "rms_target_pair_score_difference": finite_float(
            torch.sqrt(torch.mean(target_pair_error.square()))
        ),
        "limits": limits,
        "pass": passed,
    }
    for split, row_slice in split_slices().items():
        result[f"rms_restricted_probability_difference_{split}"] = finite_float(
            torch.sqrt(torch.mean(probability_error[row_slice].square()))
        )
    return result


def pooled_cosine(
    row_dot: torch.Tensor,
    row_left_norm: torch.Tensor,
    row_right_norm: torch.Tensor,
) -> float:
    denominator = math.sqrt(
        float(row_left_norm.sum()) * float(row_right_norm.sum())
    )
    if denominator == 0.0:
        return 0.0
    return finite_float(row_dot.sum() / denominator)


def causal_metric_payload(
    counterpart: Readout,
    endpoint_target: Readout,
    native: Readout,
    cell: Readout,
    *,
    trait: str,
    dose: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if dose == 0:
        raise ValueError("Scientific intervention dose cannot be zero")
    native_z = native.numeric_logits.double()
    other_z = counterpart.numeric_logits.double()
    cell_z = cell.numeric_logits.double()
    field_z = centered(native_z - other_z)
    effect_z = (
        centered(cell_z - native_z)
        if dose > 0
        else centered(native_z - cell_z)
    )
    field_z_cc = torch.empty_like(field_z)
    effect_z_cc = torch.empty_like(effect_z)
    for row_slice in split_slices().values():
        field_z_cc[row_slice] = (
            field_z[row_slice]
            - field_z[row_slice].mean(dim=0, keepdim=True)
        )
        effect_z_cc[row_slice] = (
            effect_z[row_slice]
            - effect_z[row_slice].mean(dim=0, keepdim=True)
        )

    native_p = torch.softmax(native_z, dim=-1)
    other_p = torch.softmax(other_z, dim=-1)
    cell_p = torch.softmax(cell_z, dim=-1)
    field_p = native_p - other_p
    effect_p = cell_p - native_p if dose > 0 else native_p - cell_p
    field_p_cc = torch.empty_like(field_p)
    effect_p_cc = torch.empty_like(effect_p)
    for row_slice in split_slices().values():
        field_p_cc[row_slice] = (
            field_p[row_slice]
            - field_p[row_slice].mean(dim=0, keepdim=True)
        )
        effect_p_cc[row_slice] = (
            effect_p[row_slice]
            - effect_p[row_slice].mean(dim=0, keepdim=True)
        )

    row_dot_z = torch.sum(field_z * effect_z, dim=-1)
    row_nf_z = torch.sum(field_z.square(), dim=-1)
    row_ne_z = torch.sum(effect_z.square(), dim=-1)
    row_dot_z_cc = torch.sum(field_z_cc * effect_z_cc, dim=-1)
    row_nf_z_cc = torch.sum(field_z_cc.square(), dim=-1)
    row_ne_z_cc = torch.sum(effect_z_cc.square(), dim=-1)
    row_dot_p = torch.sum(field_p * effect_p, dim=-1)
    row_nf_p = torch.sum(field_p.square(), dim=-1)
    row_ne_p = torch.sum(effect_p.square(), dim=-1)
    row_dot_p_cc = torch.sum(field_p_cc * effect_p_cc, dim=-1)
    row_nf_p_cc = torch.sum(field_p_cc.square(), dim=-1)
    row_ne_p_cc = torch.sum(effect_p_cc.square(), dim=-1)

    native_js = js_rows_from_logits(native_z, other_z)
    cell_js = js_rows_from_logits(cell_z, other_z)
    js_progress = cell_js - native_js if dose > 0 else native_js - cell_js

    native_behavior = target_behavior_scores(native.animal_logits, trait)
    other_behavior = target_behavior_scores(counterpart.animal_logits, trait)
    cell_behavior = target_behavior_scores(cell.animal_logits, trait)
    behavior_gap = (
        native_behavior["target_pair_score"]
        - other_behavior["target_pair_score"]
    )
    behavior_effect = (
        cell_behavior["target_pair_score"]
        - native_behavior["target_pair_score"]
        if dose > 0
        else native_behavior["target_pair_score"]
        - cell_behavior["target_pair_score"]
    )
    margin_effect = (
        cell_behavior["target_nine_animal_margin"]
        - native_behavior["target_nine_animal_margin"]
        if dose > 0
        else native_behavior["target_nine_animal_margin"]
        - cell_behavior["target_nine_animal_margin"]
    )

    endpoint_z = endpoint_target.numeric_logits.double()
    native_winner = torch.argmax(native_z, dim=-1)
    other_winner = torch.argmax(other_z, dim=-1)
    endpoint_winner = torch.argmax(endpoint_z, dim=-1)
    cell_winner = torch.argmax(cell_z, dim=-1)
    if dose < 0:
        hard_event = native_winner != other_winner
        hard_recovery = (cell_winner == other_winner) & hard_event
        hard_target = "paired_other_trait_checkpoint"
    else:
        hard_event = native_winner != endpoint_winner
        hard_recovery = (cell_winner == endpoint_winner) & hard_event
        hard_target = "cross_seed_same_trait_u24"
    hard_count = int(hard_event.sum())
    hard_rate = (
        finite_float(hard_recovery[hard_event].double().mean())
        if hard_count
        else 0.0
    )

    aggregate = {
        "numeric": {
            "native_paired_mean_js": finite_float(native_js.mean()),
            "cell_paired_mean_js": finite_float(cell_js.mean()),
            "oriented_mean_js_progress": finite_float(js_progress.mean()),
            "oriented_js_progress_fraction": finite_float(
                js_progress.mean() / max(float(native_js.mean()), 1e-30)
            ),
            "centered_logit_field": {
                "cosine": pooled_cosine(row_dot_z, row_nf_z, row_ne_z),
                "context_centered_cosine": pooled_cosine(
                    row_dot_z_cc, row_nf_z_cc, row_ne_z_cc
                ),
                "capture_slope": finite_float(
                    row_dot_z.sum() / max(float(row_nf_z.sum()), 1e-30)
                ),
                "context_centered_capture_slope": finite_float(
                    row_dot_z_cc.sum()
                    / max(float(row_nf_z_cc.sum()), 1e-30)
                ),
            },
            "restricted_probability_field": {
                "cosine": pooled_cosine(row_dot_p, row_nf_p, row_ne_p),
                "context_centered_cosine": pooled_cosine(
                    row_dot_p_cc, row_nf_p_cc, row_ne_p_cc
                ),
                "capture_slope": finite_float(
                    row_dot_p.sum() / max(float(row_nf_p.sum()), 1e-30)
                ),
                "context_centered_capture_slope": finite_float(
                    row_dot_p_cc.sum()
                    / max(float(row_nf_p_cc.sum()), 1e-30)
                ),
            },
        },
        "behavior": {
            "mean_native_paired_target_gap": finite_float(behavior_gap.mean()),
            "oriented_mean_target_pair_effect": finite_float(
                behavior_effect.mean()
            ),
            "oriented_target_pair_mediation_fraction": finite_float(
                behavior_effect.mean()
                / max(abs(float(behavior_gap.mean())), 1e-30)
            ),
            "oriented_mean_nine_animal_margin_effect": finite_float(
                margin_effect.mean()
            ),
        },
        "hard": {
            "target": hard_target,
            "paired_argmax_event_count": hard_count,
            "oriented_recovery_or_preservation_rate": hard_rate,
            "powered": hard_count >= int(protocol()["analysis"]["hard_event_minimum"]),
        },
    }
    arrays = {
        "numeric_native_js": native_js.cpu().numpy(),
        "numeric_cell_js": cell_js.cpu().numpy(),
        "numeric_oriented_js_progress": js_progress.cpu().numpy(),
        "logit_field_dot": row_dot_z.cpu().numpy(),
        "logit_field_norm": row_nf_z.cpu().numpy(),
        "logit_effect_norm": row_ne_z.cpu().numpy(),
        "logit_context_field_dot": row_dot_z_cc.cpu().numpy(),
        "logit_context_field_norm": row_nf_z_cc.cpu().numpy(),
        "logit_context_effect_norm": row_ne_z_cc.cpu().numpy(),
        "probability_field_dot": row_dot_p.cpu().numpy(),
        "probability_field_norm": row_nf_p.cpu().numpy(),
        "probability_effect_norm": row_ne_p.cpu().numpy(),
        "probability_context_field_dot": row_dot_p_cc.cpu().numpy(),
        "probability_context_field_norm": row_nf_p_cc.cpu().numpy(),
        "probability_context_effect_norm": row_ne_p_cc.cpu().numpy(),
        "behavior_native_gap": behavior_gap.cpu().numpy(),
        "behavior_oriented_effect": behavior_effect.cpu().numpy(),
        "behavior_oriented_margin_effect": margin_effect.cpu().numpy(),
        "hard_event": hard_event.cpu().numpy(),
        "hard_oriented_recovery": hard_recovery.cpu().numpy(),
    }
    return aggregate, arrays


def logical_key(
    *,
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
        "training_seed": seed,
        "trait": trait,
        "optimizer_update": update,
        "construction": construction,
        "control_kind": control_kind,
        "control_draw": control_draw,
        "dose": float(dose),
    }


def key_tuple(key: dict[str, Any]) -> tuple[Any, ...]:
    fields = protocol()["integrity"]["full_logical_key"]
    return tuple(key[field] for field in fields)


def cell_stem(key: dict[str, Any]) -> str:
    digest = compact_hash(key)[:16]
    dose = int(round(100 * float(key["dose"])))
    return (
        f"u{int(key['optimizer_update']):04d}_"
        f"{key['construction']}_{key['control_kind']}_"
        f"r{int(key['control_draw']):02d}_d{dose:+04d}_{digest}"
    )


def expected_cell_keys(seed: int, trait: str) -> list[dict[str, Any]]:
    result = []
    draws = int(protocol()["circuit"]["sham_draws"])
    for update in CAUSAL_UPDATES:
        for construction in ("checkpoint_local", "crossfit_endpoint_loaded"):
            for dose in REAL_DOSES:
                result.append(
                    logical_key(
                        seed=seed,
                        trait=trait,
                        update=update,
                        construction=construction,
                        control_kind="real",
                        control_draw=-1,
                        dose=dose,
                    )
                )
            for draw in range(draws):
                for dose in SHAM_DOSES:
                    result.append(
                        logical_key(
                            seed=seed,
                            trait=trait,
                            update=update,
                            construction=construction,
                            control_kind="sham",
                            control_draw=draw,
                            dose=dose,
                        )
                    )
            if construction == "crossfit_endpoint_loaded":
                for dose in SHAM_DOSES:
                    result.append(
                        logical_key(
                            seed=seed,
                            trait=trait,
                            update=update,
                            construction=construction,
                            control_kind="wrong_trait",
                            control_draw=-1,
                            dose=dose,
                        )
                    )
    tuples = [key_tuple(key) for key in result]
    if len(tuples) != len(set(tuples)):
        raise RuntimeError("Expected causal inventory contains duplicate keys")
    return result


def build_checkpoint_factors(
    model,
    base_selected: dict[str, torch.Tensor],
    *,
    seed: int,
    trait: str,
    update: int,
    matched_endpoint: dict[str, Any],
    wrong_endpoint: dict[str, Any],
) -> tuple[
    dict[str, dict[str, dict[str, Any]] | None],
    dict[str, Any],
    dict[str, dict[str, torch.Tensor]],
]:
    parameters = dict(model.named_parameters())
    local: dict[str, dict[str, Any]] = {}
    loaded: dict[str, dict[str, Any]] = {}
    wrong: dict[str, dict[str, Any]] = {}
    local_audits = {}
    local_witnesses: dict[str, dict[str, torch.Tensor]] = {}
    projections = {}
    local_identifiable = update != 0
    local_seed_base = int(protocol()["circuit"]["local_svd"]["base_seed"])
    for name in selected_names():
        delta = parameters[name].detach().float().cpu() - base_selected[name]
        local_seed = derived_seed(
            local_seed_base,
            LINEAGE,
            seed,
            trait,
            update,
            name,
        )
        factor, audit = deterministic_local_factor(
            delta,
            seed=local_seed,
        )
        audit["derived_seed"] = local_seed
        local_audits[name] = audit
        if factor is None:
            local_identifiable = False
        else:
            local[name] = factor
            local_witnesses[name] = {
                "delta_v": delta.float().mv(factor["v"].float()).cpu(),
                "delta_transpose_u": (
                    delta.float().T.mv(factor["u"].float()).cpu()
                ),
            }
        loaded_factor, coefficient = projection_factor(
            delta, matched_endpoint["factors"][name]
        )
        loaded[name] = loaded_factor
        wrong[name] = {
            "u": wrong_endpoint["factors"][name]["u"].float(),
            "s": float(coefficient),
            "v": wrong_endpoint["factors"][name]["v"].float(),
        }
        projections[name] = {
            "signed_projection": coefficient,
            "matched_endpoint_singular_value": float(
                matched_endpoint["factors"][name]["s"]
            ),
            "fraction_of_crossfit_endpoint_singular_value": coefficient
            / max(float(matched_endpoint["factors"][name]["s"]), 1e-30),
        }
    if not local_identifiable:
        local_result = None
    else:
        local_result = local
    return {
        "checkpoint_local": local_result,
        "crossfit_endpoint_loaded": loaded,
        "wrong_trait": wrong,
    }, {
        "local_identifiable": local_identifiable,
        "local_audits": local_audits,
        "crossfit_projections": projections,
    }, local_witnesses


def sham_factors(
    real: dict[str, dict[str, Any]],
    *,
    seed: int,
    trait: str,
    update: int,
    construction: str,
    draw: int,
) -> dict[str, dict[str, Any]]:
    base_seed = int(protocol()["circuit"]["sham_base_seed"])
    result = {}
    for name in selected_names():
        shape = (int(real[name]["u"].numel()), int(real[name]["v"].numel()))
        result[name] = haar_factor(
            shape,
            float(real[name]["s"]),
            seed=derived_seed(
                base_seed,
                LINEAGE,
                seed,
                trait,
                update,
                construction,
                name,
                draw,
            ),
        )
    return result


def factor_summary(factors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    modules = {}
    total_squared = 0.0
    for name in selected_names():
        factor = factors[name]
        u_norm = float(factor["u"].float().norm())
        v_norm = float(factor["v"].float().norm())
        frobenius = abs(float(factor["s"])) * u_norm * v_norm
        total_squared += frobenius**2
        modules[name] = {
            "signed_amplitude": float(factor["s"]),
            "u_norm": u_norm,
            "v_norm": v_norm,
            "frobenius_norm": frobenius,
        }
    return {
        "modules": modules,
        "coordinated_frobenius_norm": math.sqrt(total_squared),
    }


def factor_manifest(factors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    modules = {}
    for name in selected_names():
        factor = factors[name]
        u = factor["u"].detach().float().contiguous().cpu()
        v = factor["v"].detach().float().contiguous().cpu()
        modules[name] = {
            "u_dtype": str(u.dtype),
            "u_shape": list(u.shape),
            "u_sha256": tensor_sha256(u),
            "signed_amplitude": float(factor["s"]),
            "v_dtype": str(v.dtype),
            "v_shape": list(v.shape),
            "v_sha256": tensor_sha256(v),
        }
    manifest = {
        "module_order": list(selected_names()),
        "modules": modules,
    }
    return {
        **manifest,
        "factor_set_sha256": compact_hash(manifest),
    }


def factor_set_id(
    construction: str,
    control_kind: str,
    control_draw: int,
) -> str:
    return f"{construction}:{control_kind}:r{int(control_draw)}"


def checkpoint_factor_sets(
    constructions: dict[str, dict[str, dict[str, Any]] | None],
    *,
    seed: int,
    trait: str,
    update: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    draws = int(protocol()["circuit"]["sham_draws"])
    for construction in ("checkpoint_local", "crossfit_endpoint_loaded"):
        real = constructions[construction]
        if real is None:
            continue
        result[factor_set_id(construction, "real", -1)] = real
        for draw in range(draws):
            sham = sham_factors(
                real,
                seed=seed,
                trait=trait,
                update=update,
                construction=construction,
                draw=draw,
            )
            assert_norm_matched(real, sham)
            result[factor_set_id(construction, "sham", draw)] = sham
        if construction == "crossfit_endpoint_loaded":
            wrong = constructions["wrong_trait"]
            assert wrong is not None
            assert_norm_matched(real, wrong)
            result[
                factor_set_id(construction, "wrong_trait", -1)
            ] = wrong
    return result


def assert_norm_matched(
    real: dict[str, dict[str, Any]],
    control: dict[str, dict[str, Any]],
) -> None:
    for name in selected_names():
        left = (
            abs(float(real[name]["s"]))
            * float(real[name]["u"].float().norm())
            * float(real[name]["v"].float().norm())
        )
        right = (
            abs(float(control[name]["s"]))
            * float(control[name]["u"].float().norm())
            * float(control[name]["v"].float().norm())
        )
        if not math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-9):
            raise RuntimeError(
                f"Control amplitude mismatch for {name}: {left} != {right}"
            )


def relative_replay_guard(
    repeat: dict[str, Any],
    native: Readout,
    counterpart: Readout,
    *,
    update: int,
) -> dict[str, Any]:
    limits = protocol()["fidelity"]["causal_replay_repeat_guard"]
    native_p = torch.softmax(native.numeric_logits.double(), dim=-1)
    other_p = torch.softmax(counterpart.numeric_logits.double(), dim=-1)
    trait = "wolf"
    # The RMS magnitude of wolf-minus-lion is sign invariant, so this fixed
    # orientation is sufficient for the mechanical replay-noise denominator.
    native_behavior = target_behavior_scores(native.animal_logits, trait)[
        "target_pair_score"
    ]
    other_behavior = target_behavior_scores(counterpart.animal_logits, trait)[
        "target_pair_score"
    ]
    behavior_gap_rms = finite_float(
        torch.sqrt(torch.mean((native_behavior - other_behavior).square()))
    )
    split_records = {}
    numeric_passes = []
    for split, row_slice in split_slices().items():
        field_rms = finite_float(
            torch.sqrt(torch.mean((native_p[row_slice] - other_p[row_slice]).square()))
        )
        live_probability_rms = repeat[
            f"rms_restricted_probability_difference_{split}"
        ]
        numeric_fraction = (
            None
            if update == 0
            else live_probability_rms / max(field_rms, 1e-30)
        )
        split_pass = (
            None
            if update == 0
            else numeric_fraction
            <= float(
                limits[
                    "max_repeat_rms_as_fraction_of_checkpoint_paired_field_rms"
                ]
            )
        )
        numeric_passes.append(split_pass)
        split_records[split] = {
            "paired_probability_field_rms": field_rms,
            "repeat_probability_rms": live_probability_rms,
            "repeat_fraction_of_paired_field": numeric_fraction,
            "pass": split_pass,
        }
    behavior_fraction = (
        None
        if update == 0
        else repeat["rms_target_pair_score_difference"]
        / max(behavior_gap_rms, 1e-30)
    )
    if update == 0:
        u0_pass = (
            repeat["max_restricted_probability_absolute_difference"]
            <= float(limits["u0_max_restricted_probability_absolute_difference"])
            and repeat["max_behavior_selected_logit_absolute_difference"]
            <= float(limits["u0_max_behavior_selected_logit_absolute_difference"])
        )
        relative_pass = u0_pass
    else:
        relative_pass = (
            all(numeric_passes)
            and behavior_fraction
            <= float(
                limits[
                    "max_repeat_behavior_rms_as_fraction_of_checkpoint_paired_behavior_gap_rms"
                ]
            )
        )
    return {
        "numeric_splits": split_records,
        "paired_behavior_gap_rms": behavior_gap_rms,
        "behavior_repeat_fraction_of_paired_gap": behavior_fraction,
        "relative_or_u0_pass": relative_pass,
        "usable_for_onset": bool(repeat["pass"] and relative_pass),
    }


def causal_attempt_identity(seed: int, trait: str) -> dict[str, Any]:
    return {
        "lineage": LINEAGE,
        "training_seed": seed,
        "trait": trait,
        "reference_updates": list(REFERENCE_UPDATES),
        "causal_updates": list(CAUSAL_UPDATES),
        "crossfit_seed": crossfit_seed(seed),
        "config_sha256": file_sha256(CONFIG_PATH),
        "script_sha256": file_sha256(SCRIPT_PATH),
    }


def native_source_record(seed: int, trait: str) -> dict[str, Any]:
    attempt, completion = load_native_completion(seed, trait)
    return {
        "seed": seed,
        "trait": trait,
        "attempt": relative(attempt),
        "completion_sha256": file_sha256(attempt / "completion.json"),
        "readout_manifest_sha256": compact_hash(completion["readouts"]),
        "readouts": {
            str(record["optimizer_update"]): {
                "path": record["path"],
                "sha256": record["sha256"],
                "numeric_logits_sha256": record["numeric_logits_sha256"],
                "animal_logits_sha256": record["animal_logits_sha256"],
                "selected_weight_sha256": record["selected_weight_sha256"],
            }
            for record in completion["readouts"]
        },
    }


def endpoint_source_record(seed: int, trait: str) -> dict[str, Any]:
    attempt, completion, factors = load_endpoint_completion(seed, trait)
    return {
        "seed": seed,
        "trait": trait,
        "attempt": relative(attempt),
        "completion_sha256": file_sha256(attempt / "completion.json"),
        "factors_path": completion["factors_path"],
        "factors_sha256": completion["factors_sha256"],
        "selected_endpoint_sha256": factors["selected_endpoint_sha256"],
    }


def run_causal_trajectory(seed: int, trait: str) -> Path:
    require_preflight()
    endpoint_lock = require_endpoint_lock()
    native_lock = require_native_lock()
    if seed not in [int(value) for value in protocol()["paired_teacher_design"]["training_seeds"]]:
        raise ValueError(seed)
    if trait not in TRAITS:
        raise ValueError(trait)
    current_native_source = native_source_record(seed, trait)
    paired_native_source = native_source_record(seed, other_trait(trait))
    donor_seed = crossfit_seed(seed)
    matched_endpoint_source = endpoint_source_record(donor_seed, trait)
    wrong_endpoint_source = endpoint_source_record(
        donor_seed, other_trait(trait)
    )
    matched_endpoint = load_endpoint_factors(donor_seed, trait)
    wrong_endpoint = load_endpoint_factors(donor_seed, other_trait(trait))
    donor_native_source = native_source_record(donor_seed, trait)
    current_native_cache = native_readout_cache(seed, trait)
    paired_native_cache = native_readout_cache(seed, other_trait(trait))
    _, endpoint_target = load_readout(
        repository_path(
            donor_native_source["readouts"]["24"]["path"],
            label=f"donor endpoint target {donor_seed}/{trait}",
        )
    )

    root = causal_root(seed, trait)
    require_unsealed_cell(
        root,
        phase_lock=WORK / "causal_trajectories/lock.json",
    )
    attempt = next_attempt(root)
    device = select_device("auto")
    tokenizer = load_tokenizer(MODEL_CONFIG)
    context = evaluation_context(tokenizer)
    model = None
    observed_updates: list[int] = []
    observed_keys: set[tuple[Any, ...]] = set()
    cell_records: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    expected_keys = expected_cell_keys(seed, trait)
    expected_set = {key_tuple(key) for key in expected_keys}
    try:
        model = load_model(MODEL_CONFIG, device)
        base_selected = selected_state_cpu(model)

        def save_na_cell(
            key: dict[str, Any],
            *,
            reason: str,
            factor_record: dict[str, Any],
            factor_identifier: str | None,
            factor_identity: dict[str, Any] | None,
        ) -> None:
            logical = key_tuple(key)
            if logical in observed_keys:
                raise RuntimeError(f"Duplicate causal key: {key}")
            stem = cell_stem(key)
            path = attempt / "cells" / f"{stem}.json"
            payload = {
                "key": key,
                "status": "not_applicable",
                "reason": reason,
                "factor_record": factor_record,
                "factor_set_id": factor_identifier,
                "factor_manifest": factor_identity,
                "metrics": None,
                "arrays_path": None,
            }
            write_json(path, payload)
            observed_keys.add(logical)
            cell_records.append(
                {
                    "key": key,
                    "status": "not_applicable",
                    "path": relative(path),
                    "sha256": file_sha256(path),
                }
            )

        def run_cell(
            key: dict[str, Any],
            factors: dict[str, dict[str, Any]],
            factor_identifier: str,
            factor_identity: dict[str, Any],
            live_native: Readout,
            paired_other: Readout,
        ) -> None:
            logical = key_tuple(key)
            if logical in observed_keys:
                raise RuntimeError(f"Duplicate causal key: {key}")
            stem = cell_stem(key)
            json_path = attempt / "cells" / f"{stem}.json"
            array_path = attempt / "cells" / f"{stem}.npz"
            with temporary_patch(probe_model_ref[0], factors, float(key["dose"])):
                cell = evaluate_readout(probe_model_ref[0], tokenizer, context)
            metrics, arrays = causal_metric_payload(
                paired_other,
                endpoint_target,
                live_native,
                cell,
                trait=trait,
                dose=float(key["dose"]),
            )
            write_npz(array_path, arrays)
            payload = {
                "key": key,
                "status": "evaluated",
                "factor_record": factor_summary(factors),
                "factor_set_id": factor_identifier,
                "factor_manifest": factor_identity,
                "metrics": metrics,
                "arrays_path": relative(array_path),
                "arrays_sha256": file_sha256(array_path),
            }
            write_json(json_path, payload)
            observed_keys.add(logical)
            cell_records.append(
                {
                    "key": key,
                    "status": "evaluated",
                    "path": relative(json_path),
                    "sha256": file_sha256(json_path),
                    "arrays_path": relative(array_path),
                    "arrays_sha256": file_sha256(array_path),
                }
            )

        probe_model_ref: list[Any] = [None]

        def callback(update: int, probe_model) -> dict[str, Any]:
            if update in observed_updates:
                raise RuntimeError(f"Duplicate causal callback update {update}")
            before = train_checkpoint_safety_before(probe_model)
            probe_model_ref[0] = probe_model
            checkpoint_record: dict[str, Any] = {"optimizer_update": update}
            try:
                live_native = evaluate_readout(probe_model, tokenizer, context)
                if update == 0:
                    replay_path = attempt / "replay/u0000.pt"
                    write_torch(
                        replay_path,
                        readout_payload(
                            live_native,
                            seed=seed,
                            trait=trait,
                            update=0,
                            context=context,
                            selected_weight_hash=before["selected_hash"],
                        ),
                    )
                    checkpoint_record["live_u0_readout"] = {
                        "path": relative(replay_path),
                        "sha256": file_sha256(replay_path),
                        "numeric_logits_sha256": tensor_sha256(
                            live_native.numeric_logits
                        ),
                        "animal_logits_sha256": tensor_sha256(
                            live_native.animal_logits
                        ),
                        "selected_weight_sha256": before["selected_hash"],
                    }
                frozen_native = current_native_cache[update]
                paired_other = paired_native_cache[update]
                repeat = replay_repeat_metrics(
                    live_native, frozen_native, trait=trait
                )
                relative_repeat = relative_replay_guard(
                    repeat,
                    frozen_native,
                    paired_other,
                    update=update,
                )
                checkpoint_record["repeat_guard"] = {
                    **repeat,
                    **relative_repeat,
                }
                if not repeat["pass"]:
                    raise RuntimeError(
                        f"Absolute causal/native replay guard failed at u{update}: "
                        f"{repeat}"
                    )
                if update == 0 and not relative_repeat["relative_or_u0_pass"]:
                    raise RuntimeError(
                        f"Dedicated causal/native u0 replay guard failed: "
                        f"{relative_repeat}"
                    )
                if update in CAUSAL_UPDATES:
                    (
                        constructions,
                        factor_audit,
                        local_witnesses,
                    ) = build_checkpoint_factors(
                        probe_model,
                        base_selected,
                        seed=seed,
                        trait=trait,
                        update=update,
                        matched_endpoint=matched_endpoint,
                        wrong_endpoint=wrong_endpoint,
                    )
                    checkpoint_record["factor_audit"] = factor_audit
                    factor_sets = checkpoint_factor_sets(
                        constructions,
                        seed=seed,
                        trait=trait,
                        update=update,
                    )
                    factor_manifests = {
                        identifier: factor_manifest(factors)
                        for identifier, factors in factor_sets.items()
                    }
                    factor_catalog_path = (
                        attempt / "factors" / f"u{update:04d}.pt"
                    )
                    write_torch(
                        factor_catalog_path,
                        {
                            "identity": {
                                **causal_attempt_identity(seed, trait),
                                "optimizer_update": update,
                                "selected_weight_sha256": before[
                                    "selected_hash"
                                ],
                            },
                            "checkpoint_local_factors": constructions[
                                "checkpoint_local"
                            ],
                            "checkpoint_local_witnesses": (
                                local_witnesses
                                if constructions["checkpoint_local"] is not None
                                else {}
                            ),
                            "factor_audit": factor_audit,
                            "factor_manifests": factor_manifests,
                        },
                    )
                    checkpoint_record["factor_catalog"] = {
                        "path": relative(factor_catalog_path),
                        "sha256": file_sha256(factor_catalog_path),
                        "factor_set_ids": sorted(factor_manifests),
                        "factor_manifest_sha256": compact_hash(
                            factor_manifests
                        ),
                    }
                    keys_here = [
                        key
                        for key in expected_keys
                        if int(key["optimizer_update"]) == update
                    ]
                    for key in keys_here:
                        construction = str(key["construction"])
                        real = constructions[construction]
                        if real is None:
                            save_na_cell(
                                key,
                                reason="checkpoint_local_rank1_unidentifiable",
                                factor_record=factor_audit,
                                factor_identifier=None,
                                factor_identity=None,
                            )
                            continue
                        control_kind = str(key["control_kind"])
                        control_draw = int(key["control_draw"])
                        identifier = factor_set_id(
                            construction,
                            control_kind,
                            control_draw,
                        )
                        factors = factor_sets[identifier]
                        manifest = factor_manifests[identifier]
                        if all(abs(float(factor["s"])) <= 1e-30 for factor in real.values()):
                            save_na_cell(
                                key,
                                reason="zero_checkpoint_component",
                                factor_record=factor_audit,
                                factor_identifier=identifier,
                                factor_identity=manifest,
                            )
                            continue
                        run_cell(
                            key,
                            factors,
                            identifier,
                            manifest,
                            live_native,
                            paired_other,
                        )
                observed_updates.append(update)
            finally:
                probe_model_ref[0] = None
                safety = train_checkpoint_safety_after(probe_model, before)
            checkpoint_record["safety"] = safety
            checkpoint_records.append(checkpoint_record)
            return {
                "optimizer_update": update,
                "selected_weight_sha256": safety["selected_weight_sha256"],
                "repeat_guard_pass": checkpoint_record["repeat_guard"]["pass"],
                "usable_for_onset": checkpoint_record["repeat_guard"][
                    "usable_for_onset"
                ],
            }

        metrics = train_completion_model(
            model,
            tokenizer,
            training_rows(trait),
            train_config(seed, REFERENCE_UPDATES),
            device,
            attempt / "training",
            checkpoint_callback=callback,
        )
        if observed_updates != list(REFERENCE_UPDATES):
            raise RuntimeError(f"Causal update inventory mismatch: {observed_updates}")
        if observed_keys != expected_set:
            raise RuntimeError(
                f"Causal key inventory mismatch: missing={len(expected_set-observed_keys)} "
                f"extra={len(observed_keys-expected_set)}"
            )
        if len(cell_records) != len(expected_keys):
            raise RuntimeError("Causal cell record count mismatch")
        completion = {
            "identity": causal_attempt_identity(seed, trait),
            "created_utc": utc_now(),
            "attempt": relative(attempt),
            "upstream_locks": {
                "endpoint_lock_sha256": file_sha256(
                    WORK / "endpoint_factors/lock.json"
                ),
                "native_lock_sha256": file_sha256(
                    WORK / "native_trajectories/lock.json"
                ),
                "endpoint_cell_manifest_sha256": compact_hash(
                    endpoint_lock["cells"]
                ),
                "native_cell_manifest_sha256": compact_hash(
                    native_lock["cells"]
                ),
            },
            "native_sources": {
                "current": current_native_source,
                "paired_other": paired_native_source,
                "donor_endpoint_target": donor_native_source,
            },
            "endpoint_factor_sources": {
                "matched": matched_endpoint_source,
                "wrong_trait": wrong_endpoint_source,
            },
            "endpoint_target": {
                "seed": donor_seed,
                "trait": trait,
                "optimizer_update": 24,
                "numeric_logits_sha256": tensor_sha256(
                    endpoint_target.numeric_logits
                ),
            },
            "checkpoint_records": checkpoint_records,
            "cells": cell_records,
            "expected_cell_count": len(expected_keys),
            "evaluated_cell_count": sum(
                record["status"] == "evaluated" for record in cell_records
            ),
            "not_applicable_cell_count": sum(
                record["status"] == "not_applicable" for record in cell_records
            ),
            "training_metrics_path": relative(
                attempt / "training/training_metrics.json"
            ),
            "training_metrics_sha256": file_sha256(
                attempt / "training/training_metrics.json"
            ),
            "optimizer_updates": metrics["optimizer_updates"],
            "complete": True,
        }
        write_json(attempt / "completion.json", completion)
        write_json_creation_only(
            root / "canonical.json",
            {
                "attempt": relative(attempt),
                "completion_sha256": file_sha256(attempt / "completion.json"),
            },
        )
        print(
            f"[causal] seed={seed} trait={trait} "
            f"evaluated={completion['evaluated_cell_count']} "
            f"na={completion['not_applicable_cell_count']}",
            flush=True,
        )
        return attempt
    except Exception as error:
        write_json(
            attempt / "failure.json",
            {
                "created_utc": utc_now(),
                "seed": seed,
                "trait": trait,
                "error_type": type(error).__name__,
                "error": str(error),
                "observed_updates": observed_updates,
                "observed_cell_count": len(observed_keys),
            },
        )
        raise
    finally:
        release_model(model)


def validate_factor_catalog(
    seed: int,
    trait: str,
    update: int,
    checkpoint: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    record = checkpoint.get("factor_catalog")
    if not isinstance(record, dict):
        raise RuntimeError(
            f"Missing factor catalog: {seed}/{trait}/u{update}"
        )
    path = repository_path(
        record["path"],
        label=f"factor catalog {seed}/{trait}/u{update}",
    )
    expected_path = (
        canonical_attempt(causal_root(seed, trait))
        / "factors"
        / f"u{update:04d}.pt"
    )
    if path != expected_path.resolve() or file_sha256(path) != record["sha256"]:
        raise RuntimeError(f"Factor catalog path/hash mismatch: {path}")
    catalog = torch.load(path, map_location="cpu", weights_only=True)
    expected_identity = {
        **causal_attempt_identity(seed, trait),
        "optimizer_update": update,
        "selected_weight_sha256": checkpoint["safety"][
            "selected_weight_sha256"
        ],
    }
    if catalog.get("identity") != expected_identity:
        raise RuntimeError(f"Factor catalog identity mismatch: {path}")
    if catalog.get("factor_audit") != checkpoint.get("factor_audit"):
        raise RuntimeError(f"Factor catalog audit mismatch: {path}")

    local = catalog.get("checkpoint_local_factors")
    witnesses = catalog.get("checkpoint_local_witnesses")
    audit = checkpoint["factor_audit"]
    local_seed_base = int(protocol()["circuit"]["local_svd"]["base_seed"])
    local_flags = []
    for name in selected_names():
        module_audit = audit["local_audits"][name]
        expected_seed = derived_seed(
            local_seed_base,
            LINEAGE,
            seed,
            trait,
            update,
            name,
        )
        if module_audit.get("derived_seed") != expected_seed:
            raise RuntimeError(f"Local SVD seed mismatch: {path}:{name}")
        local_flags.append(bool(module_audit.get("identifiable", False)))
    expected_local_identifiable = bool(update != 0 and all(local_flags))
    if bool(audit["local_identifiable"]) != expected_local_identifiable:
        raise RuntimeError(f"Local SVD aggregate availability mismatch: {path}")
    if bool(audit["local_identifiable"]) != (local is not None):
        raise RuntimeError(f"Factor catalog local availability mismatch: {path}")
    if local is None:
        if witnesses != {}:
            raise RuntimeError(f"Unexpected local witnesses in N/A catalog: {path}")
    else:
        if set(local) != set(selected_names()) or set(witnesses) != set(
            selected_names()
        ):
            raise RuntimeError(f"Local factor/witness inventory mismatch: {path}")
        for name in selected_names():
            factor = local[name]
            witness = witnesses[name]
            delta_v = witness["delta_v"].float()
            delta_t_u = witness["delta_transpose_u"].float()
            singular = float(factor["s"])
            left = float(
                (
                    delta_v - singular * factor["u"].float()
                ).norm()
            ) / max(abs(singular), 1e-30)
            right = float(
                (
                    delta_t_u - singular * factor["v"].float()
                ).norm()
            ) / max(abs(singular), 1e-30)
            registered = audit["local_audits"][name]
            if (
                tuple(delta_v.shape) != tuple(factor["u"].shape)
                or tuple(delta_t_u.shape) != tuple(factor["v"].shape)
                or not torch.isfinite(delta_v).all()
                or not torch.isfinite(delta_t_u).all()
                or not math.isclose(
                    left,
                    float(registered["left_residual_relative"]),
                    rel_tol=2e-5,
                    abs_tol=2e-7,
                )
                or not math.isclose(
                    right,
                    float(registered["right_residual_relative"]),
                    rel_tol=2e-5,
                    abs_tol=2e-7,
                )
            ):
                raise RuntimeError(f"Local residual witness mismatch: {path}:{name}")

    donor_seed = crossfit_seed(seed)
    matched = load_endpoint_factors(donor_seed, trait)
    wrong_endpoint = load_endpoint_factors(donor_seed, other_trait(trait))
    loaded = {}
    wrong = {}
    for name in selected_names():
        coefficient = float(
            audit["crossfit_projections"][name]["signed_projection"]
        )
        projection_record = audit["crossfit_projections"][name]
        endpoint_singular = float(matched["factors"][name]["s"])
        if (
            not math.isclose(
                float(projection_record["matched_endpoint_singular_value"]),
                endpoint_singular,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                float(
                    projection_record[
                        "fraction_of_crossfit_endpoint_singular_value"
                    ]
                ),
                coefficient / max(endpoint_singular, 1e-30),
                rel_tol=2e-12,
                abs_tol=2e-12,
            )
        ):
            raise RuntimeError(f"Crossfit projection audit mismatch: {path}:{name}")
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
    constructions = {
        "checkpoint_local": local,
        "crossfit_endpoint_loaded": loaded,
        "wrong_trait": wrong,
    }
    regenerated = checkpoint_factor_sets(
        constructions,
        seed=seed,
        trait=trait,
        update=update,
    )
    manifests = {
        identifier: factor_manifest(factors)
        for identifier, factors in regenerated.items()
    }
    if catalog.get("factor_manifests") != manifests:
        raise RuntimeError(f"Factor catalog derivation mismatch: {path}")
    if record.get("factor_set_ids") != sorted(manifests):
        raise RuntimeError(f"Factor catalog set inventory mismatch: {path}")
    if record.get("factor_manifest_sha256") != compact_hash(manifests):
        raise RuntimeError(f"Factor catalog manifest hash mismatch: {path}")
    return manifests


def load_causal_completion(seed: int, trait: str) -> tuple[Path, dict[str, Any]]:
    attempt = canonical_attempt(causal_root(seed, trait))
    completion = json.loads((attempt / "completion.json").read_text())
    if completion["identity"] != causal_attempt_identity(seed, trait):
        raise RuntimeError(f"Causal completion identity mismatch: {seed}/{trait}")
    training = validate_training_completion(
        attempt,
        completion,
        phase="causal",
        seed=seed,
        probe_updates=REFERENCE_UPDATES,
    )
    expected_locks = {
        "endpoint_lock_sha256": file_sha256(
            WORK / "endpoint_factors/lock.json"
        ),
        "native_lock_sha256": file_sha256(
            WORK / "native_trajectories/lock.json"
        ),
        "endpoint_cell_manifest_sha256": compact_hash(
            require_endpoint_lock()["cells"]
        ),
        "native_cell_manifest_sha256": compact_hash(
            require_native_lock()["cells"]
        ),
    }
    if completion.get("upstream_locks") != expected_locks:
        raise RuntimeError(f"Causal upstream-lock binding mismatch: {seed}/{trait}")
    expected_native_sources = {
        "current": native_source_record(seed, trait),
        "paired_other": native_source_record(seed, other_trait(trait)),
        "donor_endpoint_target": native_source_record(
            crossfit_seed(seed), trait
        ),
    }
    if completion.get("native_sources") != expected_native_sources:
        raise RuntimeError(f"Causal native-source binding mismatch: {seed}/{trait}")
    donor_u24 = expected_native_sources["donor_endpoint_target"]["readouts"]["24"]
    expected_endpoint_target = {
        "seed": crossfit_seed(seed),
        "trait": trait,
        "optimizer_update": 24,
        "numeric_logits_sha256": donor_u24["numeric_logits_sha256"],
    }
    if completion.get("endpoint_target") != expected_endpoint_target:
        raise RuntimeError(f"Causal endpoint-target binding mismatch: {seed}/{trait}")
    expected_endpoint_sources = {
        "matched": endpoint_source_record(crossfit_seed(seed), trait),
        "wrong_trait": endpoint_source_record(
            crossfit_seed(seed), other_trait(trait)
        ),
    }
    if completion.get("endpoint_factor_sources") != expected_endpoint_sources:
        raise RuntimeError(
            f"Causal endpoint-source binding mismatch: {seed}/{trait}"
        )
    checkpoints = completion.get("checkpoint_records")
    if not isinstance(checkpoints, list) or [
        row.get("optimizer_update") for row in checkpoints
    ] != list(REFERENCE_UPDATES):
        raise RuntimeError(f"Causal checkpoint inventory mismatch: {seed}/{trait}")
    safety = validate_safety_inventory(
        [
            {
                "optimizer_update": checkpoint["optimizer_update"],
                **checkpoint["safety"],
            }
            for checkpoint in checkpoints
        ],
        phase=f"causal:{seed}:{trait}",
    )
    factor_manifests: dict[int, dict[str, dict[str, Any]]] = {}
    for update, checkpoint in enumerate(checkpoints):
        repeat = checkpoint.get("repeat_guard")
        if not isinstance(repeat, dict) or repeat.get("pass") is not True:
            raise RuntimeError(
                f"Causal absolute replay guard invalid: {seed}/{trait}/u{update}"
            )
        if update == 0 and repeat.get("relative_or_u0_pass") is not True:
            raise RuntimeError(
                f"Causal dedicated u0 guard invalid: {seed}/{trait}"
            )
        metric_row = training["checkpoint_metrics"][update]
        expected_metric = {
            "optimizer_update": update,
            "selected_weight_sha256": safety[update][
                "selected_weight_sha256"
            ],
            "repeat_guard_pass": True,
            "usable_for_onset": bool(repeat["usable_for_onset"]),
        }
        if metric_row != expected_metric:
            raise RuntimeError(
                f"Causal metrics/checkpoint cross-link mismatch: "
                f"{seed}/{trait}/u{update}"
            )
        if update == 0:
            live_record = checkpoint.get("live_u0_readout")
            if not isinstance(live_record, dict):
                raise RuntimeError(f"Causal live u0 artifact missing: {seed}/{trait}")
            live_path = repository_path(
                live_record["path"],
                label=f"causal live u0 {seed}/{trait}",
            )
            if (
                live_path != (attempt / "replay/u0000.pt").resolve()
                or file_sha256(live_path) != live_record["sha256"]
            ):
                raise RuntimeError(f"Causal live u0 path/hash mismatch: {live_path}")
            identity, readout = load_readout(live_path)
            _, current_native_completion = load_native_completion(seed, trait)
            expected_live_identity = {
                "lineage": LINEAGE,
                "training_seed": seed,
                "trait": trait,
                "optimizer_update": 0,
                **current_native_completion["context_identity"],
                "selected_weight_sha256": safety[0][
                    "selected_weight_sha256"
                ],
            }
            if (
                identity != expected_live_identity
                or tuple(readout.numeric_logits.shape) != (1024, 655)
                or tuple(readout.animal_logits.shape) != (60, 10)
                or not torch.isfinite(readout.numeric_logits).all()
                or not torch.isfinite(readout.animal_logits).all()
                or tensor_sha256(readout.numeric_logits)
                != live_record["numeric_logits_sha256"]
                or tensor_sha256(readout.animal_logits)
                != live_record["animal_logits_sha256"]
            ):
                raise RuntimeError(f"Causal live u0 tensor mismatch: {live_path}")
        elif "live_u0_readout" in checkpoint:
            raise RuntimeError(f"Unexpected nonzero live-u0 artifact at u{update}")
        if update in CAUSAL_UPDATES:
            factor_manifests[update] = validate_factor_catalog(
                seed, trait, update, checkpoint
            )
        elif "factor_catalog" in checkpoint or "factor_audit" in checkpoint:
            raise RuntimeError(
                f"Unexpected factor catalog outside causal grid: u{update}"
            )
    expected = {key_tuple(key) for key in expected_cell_keys(seed, trait)}
    observed = [key_tuple(record["key"]) for record in completion["cells"]]
    if len(observed) != len(set(observed)):
        raise RuntimeError(f"Duplicate causal keys: {seed}/{trait}")
    if set(observed) != expected:
        raise RuntimeError(f"Causal inventory mismatch: {seed}/{trait}")
    if (
        completion.get("expected_cell_count") != len(expected)
        or completion.get("evaluated_cell_count")
        != sum(record["status"] == "evaluated" for record in completion["cells"])
        or completion.get("not_applicable_cell_count")
        != sum(
            record["status"] == "not_applicable"
            for record in completion["cells"]
        )
    ):
        raise RuntimeError(f"Causal completion counts mismatch: {seed}/{trait}")
    for record in completion["cells"]:
        path = repository_path(
            record["path"],
            label=f"causal cell {seed}/{trait}",
        )
        try:
            path.relative_to((attempt / "cells").resolve())
        except ValueError as error:
            raise RuntimeError(f"Causal cell escapes attempt: {path}") from error
        if file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"Causal JSON hash mismatch: {path}")
        if path != (
            attempt / "cells" / f"{cell_stem(record['key'])}.json"
        ).resolve():
            raise RuntimeError(f"Causal JSON filename/key mismatch: {path}")
        payload = json.loads(path.read_text())
        if key_tuple(payload["key"]) != key_tuple(record["key"]):
            raise RuntimeError(f"Embedded/path key mismatch: {path}")
        if payload.get("status") != record.get("status"):
            raise RuntimeError(f"Causal embedded status mismatch: {path}")
        update = int(record["key"]["optimizer_update"])
        identifier = factor_set_id(
            str(record["key"]["construction"]),
            str(record["key"]["control_kind"]),
            int(record["key"]["control_draw"]),
        )
        expected_manifest = factor_manifests[update].get(identifier)
        if payload.get("factor_set_id") != (
            identifier if expected_manifest is not None else None
        ):
            raise RuntimeError(f"Causal factor-set reference mismatch: {path}")
        if payload.get("factor_manifest") != expected_manifest:
            raise RuntimeError(f"Causal factor manifest mismatch: {path}")
        if record["status"] == "evaluated":
            if not finite_tree(payload.get("metrics")) or not finite_tree(
                payload.get("factor_record")
            ):
                raise RuntimeError(f"Causal JSON contains non-finite values: {path}")
            factor_record = payload["factor_record"]
            if set(factor_record.get("modules", {})) != set(selected_names()):
                raise RuntimeError(f"Causal factor summary inventory mismatch: {path}")
            total_squared = 0.0
            for name in selected_names():
                summary = factor_record["modules"][name]
                registered_factor = expected_manifest["modules"][name]
                amplitude = float(summary["signed_amplitude"])
                u_norm = float(summary["u_norm"])
                v_norm = float(summary["v_norm"])
                frobenius = abs(amplitude) * u_norm * v_norm
                if (
                    amplitude
                    != float(registered_factor["signed_amplitude"])
                    or not math.isclose(u_norm, 1.0, abs_tol=2e-6)
                    or not math.isclose(v_norm, 1.0, abs_tol=2e-6)
                    or not math.isclose(
                        float(summary["frobenius_norm"]),
                        frobenius,
                        rel_tol=2e-7,
                        abs_tol=2e-9,
                    )
                ):
                    raise RuntimeError(
                        f"Causal factor summary mismatch: {path}:{name}"
                    )
                total_squared += frobenius**2
            if not math.isclose(
                float(factor_record["coordinated_frobenius_norm"]),
                math.sqrt(total_squared),
                rel_tol=2e-7,
                abs_tol=2e-9,
            ):
                raise RuntimeError(f"Causal coordinated norm mismatch: {path}")
            array_path = repository_path(
                record["arrays_path"],
                label=f"causal array {seed}/{trait}",
            )
            try:
                array_path.relative_to((attempt / "cells").resolve())
            except ValueError as error:
                raise RuntimeError(
                    f"Causal array escapes attempt: {array_path}"
                ) from error
            if array_path != (
                attempt / "cells" / f"{cell_stem(record['key'])}.npz"
            ).resolve():
                raise RuntimeError(
                    f"Causal array filename/key mismatch: {array_path}"
                )
            if payload.get("arrays_path") != record["arrays_path"]:
                raise RuntimeError(f"Causal embedded array path mismatch: {path}")
            if file_sha256(array_path) != record["arrays_sha256"]:
                raise RuntimeError(f"Causal array hash mismatch: {array_path}")
            if payload.get("arrays_sha256") != record["arrays_sha256"]:
                raise RuntimeError(f"Causal embedded array hash mismatch: {path}")
            with np.load(array_path) as archive:
                if set(archive.files) != set(CAUSAL_ARRAY_LENGTHS):
                    raise RuntimeError(f"Causal array key mismatch: {array_path}")
                for name, length in CAUSAL_ARRAY_LENGTHS.items():
                    value = archive[name]
                    if value.shape != (length,) or not np.all(np.isfinite(value)):
                        raise RuntimeError(
                            f"Causal array contract mismatch: {array_path}:{name}"
                        )
        elif (
            payload.get("metrics") is not None
            or payload.get("arrays_path") is not None
            or "arrays_path" in record
        ):
            raise RuntimeError(f"Malformed N/A causal cell: {path}")
    return attempt, completion


def causal_lock_manifest(*, created_utc: str) -> dict[str, Any]:
    endpoint_lock = require_endpoint_lock()
    native_lock = require_native_lock()
    cells = {}
    u0_members = {}
    for seed in protocol()["paired_teacher_design"]["training_seeds"]:
        for trait in TRAITS:
            seed_int = int(seed)
            label = f"s{seed}:{trait}"
            native_attempt, native_completion = load_native_completion(
                seed_int, trait
            )
            native_record = native_completion["readouts"][0]
            native_path = repository_path(
                native_record["path"],
                label=f"native u0 {label}",
            )
            native_identity, native_readout_value = load_readout(native_path)
            u0_members[f"native:{label}"] = {
                "path": native_path,
                "identity": native_identity,
                "context_identity": native_completion["context_identity"],
                "readout": native_readout_value,
            }
            attempt, completion = load_causal_completion(seed_int, trait)
            pointer = causal_root(seed_int, trait) / "canonical.json"
            live_record = completion["checkpoint_records"][0][
                "live_u0_readout"
            ]
            live_path = repository_path(
                live_record["path"],
                label=f"causal u0 {label}",
            )
            live_identity, live_readout = load_readout(live_path)
            live_context_identity = {
                key: live_identity[key]
                for key in native_completion["context_identity"]
            }
            u0_members[f"causal:{label}"] = {
                "path": live_path,
                "identity": live_identity,
                "context_identity": live_context_identity,
                "readout": live_readout,
            }
            cells[label] = {
                "attempt": relative(attempt),
                "canonical_pointer_sha256": file_sha256(pointer),
                "completion_sha256": file_sha256(attempt / "completion.json"),
                "training_metrics_sha256": completion[
                    "training_metrics_sha256"
                ],
                "cell_manifest_sha256": compact_hash(completion["cells"]),
                "checkpoint_manifest_sha256": compact_hash(
                    completion["checkpoint_records"]
                ),
            }
    expected_keys = [
        key
        for seed in protocol()["paired_teacher_design"]["training_seeds"]
        for trait in TRAITS
        for key in expected_cell_keys(int(seed), trait)
    ]
    return {
        "schema": "teacher_trait_fingerprint_causal_lock_v1",
        "experiment_id": protocol()["experiment_id"],
        "created_utc": created_utc,
        "config_sha256": file_sha256(CONFIG_PATH),
        "script_sha256": file_sha256(SCRIPT_PATH),
        "preflight_sha256": file_sha256(WORK / "preflight.json"),
        "endpoint_lock_sha256": file_sha256(
            WORK / "endpoint_factors/lock.json"
        ),
        "native_lock_sha256": file_sha256(
            WORK / "native_trajectories/lock.json"
        ),
        "endpoint_cell_manifest_sha256": compact_hash(endpoint_lock["cells"]),
        "native_cell_manifest_sha256": compact_hash(native_lock["cells"]),
        "cells": cells,
        "global_expected_key_count": len(expected_keys),
        "global_expected_key_sha256": compact_hash(expected_keys),
        "u0_equivalence": u0_equivalence(u0_members),
    }


def seal_causal_trajectories() -> dict[str, Any]:
    path = WORK / "causal_trajectories/lock.json"
    if path.exists():
        return require_causal_lock()
    lock = causal_lock_manifest(created_utc=utc_now())
    write_json_creation_only(path, lock)
    return lock


def require_causal_lock() -> dict[str, Any]:
    path = WORK / "causal_trajectories/lock.json"
    if not path.is_file():
        raise RuntimeError("Run --causal-all to seal all four causal trajectories")
    observed = json.loads(path.read_text())
    expected = causal_lock_manifest(
        created_utc=str(observed.get("created_utc"))
    )
    if observed != expected:
        raise RuntimeError("Causal trajectory lock or a bound artifact has drifted")
    return observed


def run_all_causal() -> None:
    require_preflight()
    require_native_lock()
    with exclusive_lock("causal_all"):
        lock_path = WORK / "causal_trajectories/lock.json"
        if lock_path.exists():
            require_causal_lock()
            print("[causal] sealed phase reused", flush=True)
            return
        for seed in protocol()["paired_teacher_design"]["training_seeds"]:
            for trait in TRAITS:
                if (causal_root(int(seed), trait) / "canonical.json").exists():
                    load_causal_completion(int(seed), trait)
                    print(f"[causal] reuse seed={seed} trait={trait}", flush=True)
                    continue
                run_causal_trajectory(int(seed), trait)
        seal_causal_trajectories()


@dataclass
class BootstrapRecord:
    record_id: str
    family: str
    observed: float
    bootstrap: np.ndarray
    metadata: dict[str, Any]
    pointwise_low: float = 0.0
    pointwise_high: float = 0.0
    standard_error: float = 0.0
    simultaneous_low: float = 0.0
    simultaneous_high: float = 0.0
    deterministic: bool = False


def bootstrap_index_bank() -> dict[str, np.ndarray]:
    config = protocol()["analysis"]
    samples = int(config["bootstrap_samples"])
    seed = int(config["bootstrap_seed"])
    result = {}
    for label, rows, offset in (
        ("numeric_A", 512, 0),
        ("numeric_B", 512, 1),
        ("behavior", len(PREFERENCE_EVAL_PROMPTS), 2),
    ):
        rng = np.random.default_rng(seed + offset)
        result[label] = rng.integers(
            0, rows, size=(samples, rows), dtype=np.int32
        )
    return result


def mean_bootstrap(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] != indices.shape[1]:
        raise RuntimeError(
            f"Mean-bootstrap shape mismatch: {array.shape} vs {indices.shape}"
        )
    return array[indices].mean(axis=1)


def ratio_bootstrap(
    numerator: np.ndarray,
    denominator: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    left = np.asarray(numerator, dtype=np.float64)
    right = np.asarray(denominator, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise RuntimeError("Ratio-bootstrap arrays must be equal one-dimensional rows")
    num = left[indices].sum(axis=1)
    den = right[indices].sum(axis=1)
    if np.any(den <= 1e-12):
        return np.full(indices.shape[0], np.nan)
    return num / den


def add_bootstrap_record(
    records: list[BootstrapRecord],
    lookup: dict[str, BootstrapRecord],
    *,
    record_id: str,
    family: str,
    observed: float,
    bootstrap: np.ndarray,
    metadata: dict[str, Any],
) -> str:
    if record_id in lookup:
        raise RuntimeError(f"Duplicate bootstrap record {record_id}")
    boots = np.asarray(bootstrap, dtype=np.float64)
    if boots.shape != (int(protocol()["analysis"]["bootstrap_samples"]),):
        raise RuntimeError(f"Bootstrap shape mismatch for {record_id}: {boots.shape}")
    if not math.isfinite(float(observed)) or not np.all(np.isfinite(boots)):
        raise RuntimeError(f"Non-finite bootstrap record {record_id}")
    record = BootstrapRecord(
        record_id=record_id,
        family=family,
        observed=float(observed),
        bootstrap=boots,
        metadata=metadata,
    )
    records.append(record)
    lookup[record_id] = record
    return record_id


def finalize_bootstrap_records(records: list[BootstrapRecord]) -> dict[str, Any]:
    critical_values = {}
    for family in sorted({record.family for record in records}):
        group = [record for record in records if record.family == family]
        active = []
        for record in group:
            record.pointwise_low = float(np.percentile(record.bootstrap, 2.5))
            record.pointwise_high = float(np.percentile(record.bootstrap, 97.5))
            record.standard_error = float(record.bootstrap.std(ddof=1))
            exactly_constant = bool(
                np.all(record.bootstrap == record.bootstrap[0])
            )
            if exactly_constant:
                record.deterministic = True
                record.simultaneous_low = record.observed
                record.simultaneous_high = record.observed
            else:
                if record.standard_error == 0.0:
                    raise RuntimeError(
                        f"Nonconstant bootstrap underflowed to zero SE: "
                        f"{record.record_id}"
                    )
                active.append(record)
        if active:
            standardized = np.stack(
                [
                    np.abs(
                        (record.bootstrap - record.observed)
                        / record.standard_error
                    )
                    for record in active
                ],
                axis=1,
            )
            maximum = standardized.max(axis=1)
            critical = float(np.percentile(maximum, 95.0))
            for record in active:
                record.simultaneous_low = (
                    record.observed - critical * record.standard_error
                )
                record.simultaneous_high = (
                    record.observed + critical * record.standard_error
                )
        else:
            critical = 0.0
        critical_values[family] = {
            "critical_value": critical,
            "active_records": len(active),
            "deterministic_records": len(group) - len(active),
        }
    return critical_values


def serialized_bootstrap_record(record: BootstrapRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "family": record.family,
        "observed": record.observed,
        "standard_error": record.standard_error,
        "pointwise_95_ci_low": record.pointwise_low,
        "pointwise_95_ci_high": record.pointwise_high,
        "simultaneous_95_ci_low": record.simultaneous_low,
        "simultaneous_95_ci_high": record.simultaneous_high,
        "deterministic": record.deterministic,
        "metadata": record.metadata,
    }


def stable_onset(
    updates: Iterable[int],
    pass_by_update: dict[int, bool],
    below_by_update: dict[int, bool],
) -> dict[str, Any]:
    ordered = [int(value) for value in updates]
    stable = None
    for index, update in enumerate(ordered):
        if all(pass_by_update.get(later, False) for later in ordered[index:]):
            stable = update
            break
    first_confirmed = None
    for index, update in enumerate(ordered):
        later = ordered[index + 1:index + 3]
        if len(later) < 2:
            continue
        required = [update, *later]
        if ordered[-1] not in required:
            required.append(ordered[-1])
        if all(pass_by_update.get(value, False) for value in required):
            first_confirmed = update
            break
    last_below = None
    if stable is not None:
        candidates = [
            update
            for update in ordered
            if update < stable and below_by_update.get(update, False)
        ]
        if candidates:
            last_below = max(candidates)
    return {
        "stable_onset": stable,
        "first_confirmed": first_confirmed,
        "nonmonotonic_first_confirmed_only": (
            first_confirmed is not None
            and (stable is None or first_confirmed < stable)
        ),
        "last_demonstrably_below_floor": last_below,
        "onset_interval": [last_below, stable],
        "pass_by_update": {
            str(update): bool(pass_by_update.get(update, False))
            for update in ordered
        },
        "below_by_update": {
            str(update): bool(below_by_update.get(update, False))
            for update in ordered
        },
    }


def split_slices() -> dict[str, slice]:
    return {"A": slice(0, 512), "B": slice(512, 1024)}


def context_centered_field(
    current: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    field = centered(current.double() - reference.double())
    return field - field.mean(dim=0, keepdim=True)


def row_tv(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    p = torch.softmax(left.double(), dim=-1)
    q = torch.softmax(right.double(), dim=-1)
    return (0.5 * torch.sum(torch.abs(p - q), dim=-1)).cpu().numpy()


def native_analysis_inputs() -> dict[tuple[int, str, int], Readout]:
    result = {}
    for seed in protocol()["paired_teacher_design"]["training_seeds"]:
        for trait in TRAITS:
            cached = native_readout_cache(int(seed), trait)
            for update, readout in cached.items():
                result[(int(seed), trait, update)] = readout
    return result


def build_native_analysis(
    raw: dict[tuple[int, str, int], Readout],
    records: list[BootstrapRecord],
    lookup: dict[str, BootstrapRecord],
    indices: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    seeds = [int(value) for value in protocol()["paired_teacher_design"]["training_seeds"]]
    floor = float(protocol()["analysis"]["practical_floor_fraction_of_endpoint"])
    summaries: dict[str, Any] = {}
    gate_records: dict[str, Any] = {
        "fingerprint_appearance": {},
        "trait_behavior": {},
        "trait_specific_field": {},
        "identity": {},
    }
    canonical_base = raw[(seeds[0], "wolf", 0)]
    summaries["_canonical_base"] = {
        "training_seed": seeds[0],
        "trait": "wolf",
        "optimizer_update": 0,
        "substituted_for_all_u0_cells": True,
        "native_lock_required": True,
    }

    for seed in seeds:
        donor = crossfit_seed(seed)
        summaries[str(seed)] = {"traits": {}, "paired": {}}
        for trait in TRAITS:
            summaries[str(seed)]["traits"][trait] = {}
            base = canonical_base
            endpoint = raw[(seed, trait, 24)]
            donor_base = canonical_base
            donor_endpoint = raw[(donor, trait, 24)]
            donor_wrong_base = canonical_base
            donor_wrong_endpoint = raw[(donor, other_trait(trait), 24)]
            for update in REFERENCE_UPDATES:
                current = (
                    canonical_base
                    if update == 0
                    else raw[(seed, trait, update)]
                )
                entry: dict[str, Any] = {}
                tv_full = row_tv(current.numeric_logits, base.numeric_logits)
                js_full = (
                    js_rows_from_logits(
                        current.numeric_logits, base.numeric_logits
                    )
                    .cpu()
                    .numpy()
                )
                entry["base_relative_mean_tv"] = finite_float(tv_full.mean())
                entry["base_relative_mean_js"] = finite_float(js_full.mean())
                current_behavior = target_behavior_scores(
                    current.animal_logits, trait
                )
                base_behavior = target_behavior_scores(base.animal_logits, trait)
                entry["base_relative_target_pair_behavior"] = finite_float(
                    (
                        current_behavior["target_pair_score"]
                        - base_behavior["target_pair_score"]
                    ).mean()
                )
                entry["base_argmax_event_count"] = int(
                    (
                        torch.argmax(current.numeric_logits, dim=-1)
                        != torch.argmax(base.numeric_logits, dim=-1)
                    ).sum()
                )
                entry["splits"] = {}
                for split, row_slice in split_slices().items():
                    tv = tv_full[row_slice]
                    endpoint_tv = row_tv(
                        endpoint.numeric_logits[row_slice],
                        base.numeric_logits[row_slice],
                    )
                    floor_values = tv - floor * endpoint_tv
                    record_id = (
                        f"native:appearance:s{seed}:{trait}:u{update}:{split}"
                    )
                    add_bootstrap_record(
                        records,
                        lookup,
                        record_id=record_id,
                        family=f"numeric_{split}",
                        observed=float(floor_values.mean()),
                        bootstrap=mean_bootstrap(
                            floor_values, indices[f"numeric_{split}"]
                        ),
                        metadata={
                            "gate": "fingerprint_appearance",
                            "seed": seed,
                            "trait": trait,
                            "update": update,
                            "split": split,
                        },
                    )

                    current_field = context_centered_field(
                        current.numeric_logits[row_slice],
                        (
                            canonical_base
                            if update == 0
                            else raw[
                                (seed, other_trait(trait), update)
                            ]
                        ).numeric_logits[row_slice],
                    )
                    own_endpoint_field = context_centered_field(
                        endpoint.numeric_logits[row_slice],
                        raw[
                            (seed, other_trait(trait), 24)
                        ].numeric_logits[row_slice],
                    )
                    donor_same_field = context_centered_field(
                        donor_endpoint.numeric_logits[row_slice],
                        donor_base.numeric_logits[row_slice],
                    )
                    donor_wrong_field = context_centered_field(
                        donor_wrong_endpoint.numeric_logits[row_slice],
                        donor_wrong_base.numeric_logits[row_slice],
                    )
                    same_num = torch.sum(
                        current_field * donor_same_field, dim=-1
                    ).cpu().numpy()
                    wrong_num = torch.sum(
                        current_field * donor_wrong_field, dim=-1
                    ).cpu().numpy()
                    denominator = torch.sum(
                        own_endpoint_field * donor_same_field, dim=-1
                    ).cpu().numpy()
                    denominator_total = float(denominator.sum())
                    if denominator_total <= 1e-12:
                        identity = {
                            "status": "not_applicable_nonpositive_denominator",
                            "denominator": denominator_total,
                        }
                    else:
                        beta_same = float(same_num.sum() / denominator_total)
                        beta_wrong = float(wrong_num.sum() / denominator_total)
                        same_ratio_boot = ratio_bootstrap(
                            same_num,
                            denominator,
                            indices[f"numeric_{split}"],
                        )
                        contrast_boot = ratio_bootstrap(
                            same_num - wrong_num,
                            denominator,
                            indices[f"numeric_{split}"],
                        )
                        if (
                            not np.all(np.isfinite(same_ratio_boot))
                            or not np.all(np.isfinite(contrast_boot))
                        ):
                            entry["splits"][split] = {
                                "mean_tv": float(tv.mean()),
                                "appearance_record": record_id,
                                "identity": {
                                    "status": "not_applicable_bootstrap_denominator",
                                    "denominator": denominator_total,
                                },
                            }
                            continue
                        same_floor_boot = same_ratio_boot - floor
                        same_record = (
                            f"native:identity_same:s{seed}:{trait}:"
                            f"u{update}:{split}"
                        )
                        contrast_record = (
                            f"native:identity_contrast:s{seed}:{trait}:"
                            f"u{update}:{split}"
                        )
                        add_bootstrap_record(
                            records,
                            lookup,
                            record_id=same_record,
                            family=f"numeric_{split}",
                            observed=beta_same - floor,
                            bootstrap=same_floor_boot,
                            metadata={
                                "gate": "identity_same_floor",
                                "seed": seed,
                                "trait": trait,
                                "update": update,
                                "split": split,
                            },
                        )
                        add_bootstrap_record(
                            records,
                            lookup,
                            record_id=contrast_record,
                            family=f"numeric_{split}",
                            observed=beta_same - beta_wrong,
                            bootstrap=contrast_boot,
                            metadata={
                                "gate": "identity_wrong_contrast",
                                "seed": seed,
                                "trait": trait,
                                "update": update,
                                "split": split,
                            },
                        )
                        identity = {
                            "status": "evaluated",
                            "beta_same": beta_same,
                            "beta_wrong": beta_wrong,
                            "denominator": denominator_total,
                            "same_floor_record": same_record,
                            "contrast_record": contrast_record,
                        }
                    entry["splits"][split] = {
                        "mean_tv": float(tv.mean()),
                        "appearance_record": record_id,
                        "identity": identity,
                    }

                current_p_shift = (
                    torch.softmax(current.numeric_logits.double(), dim=-1)
                    - torch.softmax(base.numeric_logits.double(), dim=-1)
                ).mean(dim=0)
                donor_shift = (
                    torch.softmax(donor_endpoint.numeric_logits.double(), dim=-1)
                    - torch.softmax(donor_base.numeric_logits.double(), dim=-1)
                ).mean(dim=0)
                top_current = (
                    set()
                    if update == 0
                    else set(
                        torch.topk(
                            torch.abs(current_p_shift), 50
                        ).indices.tolist()
                    )
                )
                top_endpoint = set(
                    torch.topk(torch.abs(donor_shift), 50).indices.tolist()
                )
                overlap = sorted(top_current & top_endpoint)
                signed_agreement = (
                    float(
                        torch.mean(
                            (
                                torch.sign(current_p_shift[overlap])
                                == torch.sign(donor_shift[overlap])
                            ).double()
                        )
                    )
                    if overlap
                    else 0.0
                )
                entry["top50_endpoint_overlap"] = len(overlap)
                entry["top50_overlap_signed_agreement"] = signed_agreement
                summaries[str(seed)]["traits"][trait][str(update)] = entry

        wolf_endpoint = raw[(seed, "wolf", 24)]
        lion_endpoint = raw[(seed, "lion", 24)]
        endpoint_gap = (
            target_behavior_scores(wolf_endpoint.animal_logits, "wolf")[
                "global_wolf_minus_lion"
            ]
            - target_behavior_scores(lion_endpoint.animal_logits, "wolf")[
                "global_wolf_minus_lion"
            ]
        ).cpu().numpy()
        for update in REFERENCE_UPDATES:
            wolf = (
                canonical_base
                if update == 0
                else raw[(seed, "wolf", update)]
            )
            lion = (
                canonical_base
                if update == 0
                else raw[(seed, "lion", update)]
            )
            wolf_score = target_behavior_scores(wolf.animal_logits, "wolf")[
                "global_wolf_minus_lion"
            ]
            lion_score = target_behavior_scores(lion.animal_logits, "wolf")[
                "global_wolf_minus_lion"
            ]
            behavior_gap = (wolf_score - lion_score).cpu().numpy()
            behavior_floor = behavior_gap - floor * endpoint_gap
            behavior_record = f"native:behavior:s{seed}:u{update}"
            add_bootstrap_record(
                records,
                lookup,
                record_id=behavior_record,
                family="behavior",
                observed=float(behavior_floor.mean()),
                bootstrap=mean_bootstrap(behavior_floor, indices["behavior"]),
                metadata={
                    "gate": "trait_behavior",
                    "seed": seed,
                    "update": update,
                },
            )
            paired_tv_full = row_tv(wolf.numeric_logits, lion.numeric_logits)
            endpoint_paired_tv = row_tv(
                wolf_endpoint.numeric_logits, lion_endpoint.numeric_logits
            )
            paired_entry = {
                "mean_tv": float(paired_tv_full.mean()),
                "mean_js": finite_float(
                    js_rows_from_logits(
                        wolf.numeric_logits, lion.numeric_logits
                    ).mean()
                ),
                "behavior_gap": float(behavior_gap.mean()),
                "behavior_record": behavior_record,
                "argmax_event_count": int(
                    (
                        torch.argmax(wolf.numeric_logits, dim=-1)
                        != torch.argmax(lion.numeric_logits, dim=-1)
                    ).sum()
                ),
                "splits": {},
            }
            for split, row_slice in split_slices().items():
                floor_values = (
                    paired_tv_full[row_slice]
                    - floor * endpoint_paired_tv[row_slice]
                )
                record_id = f"native:paired_field:s{seed}:u{update}:{split}"
                add_bootstrap_record(
                    records,
                    lookup,
                    record_id=record_id,
                    family=f"numeric_{split}",
                    observed=float(floor_values.mean()),
                    bootstrap=mean_bootstrap(
                        floor_values, indices[f"numeric_{split}"]
                    ),
                    metadata={
                        "gate": "trait_specific_field",
                        "seed": seed,
                        "update": update,
                        "split": split,
                    },
                )
                paired_entry["splits"][split] = {
                    "mean_tv": float(paired_tv_full[row_slice].mean()),
                    "field_record": record_id,
                }
            summaries[str(seed)]["paired"][str(update)] = paired_entry

    gate_records["fingerprint_appearance"] = {
        str(update): [
            f"native:appearance:s{seed}:{trait}:u{update}:{split}"
            for seed in seeds
            for trait in TRAITS
            for split in split_slices()
        ]
        for update in REFERENCE_UPDATES
    }
    gate_records["trait_behavior"] = {
        str(update): [
            f"native:behavior:s{seed}:u{update}" for seed in seeds
        ]
        for update in REFERENCE_UPDATES
    }
    gate_records["trait_specific_field"] = {
        str(update): [
            f"native:paired_field:s{seed}:u{update}:{split}"
            for seed in seeds
            for split in split_slices()
        ]
        for update in REFERENCE_UPDATES
    }
    gate_records["identity"] = {}
    for update in REFERENCE_UPDATES:
        ids = []
        available = True
        for seed in seeds:
            for trait in TRAITS:
                for split in split_slices():
                    entry = summaries[str(seed)]["traits"][trait][str(update)][
                        "splits"
                    ][split]["identity"]
                    if entry["status"] != "evaluated":
                        available = False
                        continue
                    ids.extend(
                        [entry["same_floor_record"], entry["contrast_record"]]
                    )
        gate_records["identity"][str(update)] = (
            ids if available else None
        )
    return summaries, gate_records


def load_causal_cells() -> tuple[
    dict[tuple[Any, ...], tuple[dict[str, Any], dict[str, np.ndarray] | None]],
    dict[tuple[int, str], dict[str, Any]],
]:
    cells = {}
    completions = {}
    for seed in protocol()["paired_teacher_design"]["training_seeds"]:
        for trait in TRAITS:
            _, completion = load_causal_completion(int(seed), trait)
            completions[(int(seed), trait)] = completion
            for record in completion["cells"]:
                key = key_tuple(record["key"])
                if key in cells:
                    raise RuntimeError(f"Duplicate global causal key {key}")
                payload = json.loads((ROOT / record["path"]).read_text())
                arrays = None
                if record["status"] == "evaluated":
                    with np.load(ROOT / record["arrays_path"]) as archive:
                        arrays = {name: archive[name].copy() for name in archive.files}
                cells[key] = (payload, arrays)
    expected = {
        key_tuple(key)
        for seed in protocol()["paired_teacher_design"]["training_seeds"]
        for trait in TRAITS
        for key in expected_cell_keys(int(seed), trait)
    }
    if set(cells) != expected:
        raise RuntimeError(
            f"Global causal inventory mismatch: missing={len(expected-set(cells))} "
            f"extra={len(set(cells)-expected)}"
        )
    return cells, completions


def causal_cell(
    cells: dict[
        tuple[Any, ...], tuple[dict[str, Any], dict[str, np.ndarray] | None]
    ],
    *,
    seed: int,
    trait: str,
    update: int,
    construction: str,
    control_kind: str,
    control_draw: int,
    dose: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    key = logical_key(
        seed=seed,
        trait=trait,
        update=update,
        construction=construction,
        control_kind=control_kind,
        control_draw=control_draw,
        dose=dose,
    )
    return cells[key_tuple(key)]


def add_mean_array_record(
    records: list[BootstrapRecord],
    lookup: dict[str, BootstrapRecord],
    indices: dict[str, np.ndarray],
    *,
    record_id: str,
    family: str,
    values: np.ndarray,
    metadata: dict[str, Any],
) -> str:
    return add_bootstrap_record(
        records,
        lookup,
        record_id=record_id,
        family=family,
        observed=float(np.mean(values)),
        bootstrap=mean_bootstrap(values, indices[family]),
        metadata=metadata,
    )


def build_causal_analysis(
    cells: dict[
        tuple[Any, ...], tuple[dict[str, Any], dict[str, np.ndarray] | None]
    ],
    completions: dict[tuple[int, str], dict[str, Any]],
    records: list[BootstrapRecord],
    lookup: dict[str, BootstrapRecord],
    indices: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    floor = float(protocol()["analysis"]["practical_floor_fraction_of_endpoint"])
    seeds = [int(value) for value in protocol()["paired_teacher_design"]["training_seeds"]]
    draws = int(protocol()["circuit"]["sham_draws"])
    summaries: dict[str, Any] = {}
    gate_records: dict[str, dict[str, Any]] = {
        "checkpoint_local": {},
        "crossfit_endpoint_loaded": {},
    }

    repeat_usable: dict[tuple[int, str, int], bool] = {}
    for (seed, trait), completion in completions.items():
        for checkpoint in completion["checkpoint_records"]:
            update = int(checkpoint["optimizer_update"])
            repeat_usable[(seed, trait, update)] = bool(
                checkpoint["repeat_guard"]["usable_for_onset"]
            )

    for construction in ("checkpoint_local", "crossfit_endpoint_loaded"):
        summaries[construction] = {}
        for update in CAUSAL_UPDATES:
            summaries[construction][str(update)] = {
                "available": True,
                "component_available_all": True,
                "replay_usable_all": True,
                "seed_traits": {},
            }
            component_ids: list[str] = []
            availability = True
            component_available_all = True
            replay_usable_all = True
            for seed in seeds:
                for trait in TRAITS:
                    cell_label = f"s{seed}:{trait}"
                    real_cells = {}
                    for dose in REAL_DOSES:
                        payload, arrays = causal_cell(
                            cells,
                            seed=seed,
                            trait=trait,
                            update=update,
                            construction=construction,
                            control_kind="real",
                            control_draw=-1,
                            dose=dose,
                        )
                        if payload["status"] != "evaluated" or arrays is None:
                            availability = False
                            real_cells = {}
                            break
                        real_cells[dose] = (payload, arrays)
                    if not real_cells:
                        component_available_all = False
                        summaries[construction][str(update)]["seed_traits"][
                            cell_label
                        ] = {
                            "available": False,
                            "reason": "real_cells_not_applicable",
                            "replay_usable": repeat_usable.get(
                                (seed, trait, update), False
                            ),
                        }
                        continue
                    if not repeat_usable.get((seed, trait, update), False):
                        availability = False
                        replay_usable_all = False
                    local_ids: list[str] = []
                    for dose in (-1.0, 1.0):
                        _, real = real_cells[dose]
                        assert real is not None
                        behavior_values = real["behavior_oriented_effect"].astype(
                            np.float64
                        )
                        js_values = real["numeric_oriented_js_progress"].astype(
                            np.float64
                        )
                        if dose < 0:
                            behavior_values = (
                                behavior_values
                                - floor
                                * real["behavior_native_gap"].astype(np.float64)
                            )
                            js_values = (
                                js_values
                                - floor
                                * real["numeric_native_js"].astype(np.float64)
                            )
                        behavior_id = (
                            f"causal:real_behavior:{construction}:s{seed}:{trait}:"
                            f"u{update}:d{dose:+.1f}"
                        )
                        add_mean_array_record(
                            records,
                            lookup,
                            indices,
                            record_id=behavior_id,
                            family="behavior",
                            values=behavior_values,
                            metadata={
                                "gate": "causal_real_behavior",
                                "construction": construction,
                                "seed": seed,
                                "trait": trait,
                                "update": update,
                                "dose": dose,
                            },
                        )
                        local_ids.append(behavior_id)
                        for split, row_slice in split_slices().items():
                            js_id = (
                                f"causal:real_js:{construction}:s{seed}:{trait}:"
                                f"u{update}:d{dose:+.1f}:{split}"
                            )
                            dot_id = (
                                f"causal:real_field:{construction}:s{seed}:{trait}:"
                                f"u{update}:d{dose:+.1f}:{split}"
                            )
                            add_mean_array_record(
                                records,
                                lookup,
                                indices,
                                record_id=js_id,
                                family=f"numeric_{split}",
                                values=js_values[row_slice],
                                metadata={
                                    "gate": "causal_real_js",
                                    "construction": construction,
                                    "seed": seed,
                                    "trait": trait,
                                    "update": update,
                                    "dose": dose,
                                    "split": split,
                                },
                            )
                            add_mean_array_record(
                                records,
                                lookup,
                                indices,
                                record_id=dot_id,
                                family=f"numeric_{split}",
                                values=real["logit_context_field_dot"][
                                    row_slice
                                ].astype(np.float64),
                                metadata={
                                    "gate": "causal_real_field",
                                    "construction": construction,
                                    "seed": seed,
                                    "trait": trait,
                                    "update": update,
                                    "dose": dose,
                                    "split": split,
                                },
                            )
                            local_ids.extend([js_id, dot_id])

                        controls = [("sham", draw) for draw in range(draws)]
                        if construction == "crossfit_endpoint_loaded":
                            controls.append(("wrong_trait", -1))
                        for control_kind, draw in controls:
                            _, control = causal_cell(
                                cells,
                                seed=seed,
                                trait=trait,
                                update=update,
                                construction=construction,
                                control_kind=control_kind,
                                control_draw=draw,
                                dose=dose,
                            )
                            if control is None:
                                availability = False
                                continue
                            behavior_contrast = (
                                real["behavior_oriented_effect"].astype(np.float64)
                                - control["behavior_oriented_effect"].astype(
                                    np.float64
                                )
                            )
                            behavior_control_id = (
                                f"causal:control_behavior:{construction}:s{seed}:"
                                f"{trait}:u{update}:d{dose:+.1f}:{control_kind}:"
                                f"r{draw}"
                            )
                            add_mean_array_record(
                                records,
                                lookup,
                                indices,
                                record_id=behavior_control_id,
                                family="behavior",
                                values=behavior_contrast,
                                metadata={
                                    "gate": "causal_control_behavior",
                                    "construction": construction,
                                    "seed": seed,
                                    "trait": trait,
                                    "update": update,
                                    "dose": dose,
                                    "control_kind": control_kind,
                                    "control_draw": draw,
                                },
                            )
                            local_ids.append(behavior_control_id)
                            for split, row_slice in split_slices().items():
                                js_control_id = (
                                    f"causal:control_js:{construction}:s{seed}:"
                                    f"{trait}:u{update}:d{dose:+.1f}:{split}:"
                                    f"{control_kind}:r{draw}"
                                )
                                field_control_id = (
                                    f"causal:control_field:{construction}:s{seed}:"
                                    f"{trait}:u{update}:d{dose:+.1f}:{split}:"
                                    f"{control_kind}:r{draw}"
                                )
                                add_mean_array_record(
                                    records,
                                    lookup,
                                    indices,
                                    record_id=js_control_id,
                                    family=f"numeric_{split}",
                                    values=(
                                        real["numeric_oriented_js_progress"][
                                            row_slice
                                        ].astype(np.float64)
                                        - control[
                                            "numeric_oriented_js_progress"
                                        ][row_slice].astype(np.float64)
                                    ),
                                    metadata={
                                        "gate": "causal_control_js",
                                        "construction": construction,
                                        "seed": seed,
                                        "trait": trait,
                                        "update": update,
                                        "dose": dose,
                                        "split": split,
                                        "control_kind": control_kind,
                                        "control_draw": draw,
                                    },
                                )
                                add_mean_array_record(
                                    records,
                                    lookup,
                                    indices,
                                    record_id=field_control_id,
                                    family=f"numeric_{split}",
                                    values=(
                                        real["logit_context_field_dot"][
                                            row_slice
                                        ].astype(np.float64)
                                        - control[
                                            "logit_context_field_dot"
                                        ][row_slice].astype(np.float64)
                                    ),
                                    metadata={
                                        "gate": "causal_control_field",
                                        "construction": construction,
                                        "seed": seed,
                                        "trait": trait,
                                        "update": update,
                                        "dose": dose,
                                        "split": split,
                                        "control_kind": control_kind,
                                        "control_draw": draw,
                                    },
                                )
                                local_ids.extend(
                                    [js_control_id, field_control_id]
                                )

                    alpha = np.asarray(REAL_DOSES, dtype=np.float64)
                    denominator = float(np.sum(alpha**2))
                    behavior_slope = np.zeros(len(PREFERENCE_EVAL_PROMPTS))
                    js_slope = np.zeros(1024)
                    field_slope = np.zeros(1024)
                    for dose in REAL_DOSES:
                        _, arrays = real_cells[dose]
                        assert arrays is not None
                        orientation = 1.0 if dose > 0 else -1.0
                        behavior_slope += (
                            dose
                            * orientation
                            * arrays["behavior_oriented_effect"].astype(np.float64)
                        )
                        js_slope += (
                            dose
                            * orientation
                            * arrays["numeric_oriented_js_progress"].astype(
                                np.float64
                            )
                        )
                        field_slope += (
                            dose
                            * orientation
                            * arrays["logit_context_field_dot"].astype(np.float64)
                        )
                    behavior_slope /= denominator
                    js_slope /= denominator
                    field_slope /= denominator
                    behavior_slope_id = (
                        f"causal:slope_behavior:{construction}:s{seed}:{trait}:"
                        f"u{update}"
                    )
                    add_mean_array_record(
                        records,
                        lookup,
                        indices,
                        record_id=behavior_slope_id,
                        family="behavior",
                        values=behavior_slope,
                        metadata={
                            "gate": "causal_dose_slope_behavior",
                            "construction": construction,
                            "seed": seed,
                            "trait": trait,
                            "update": update,
                        },
                    )
                    local_ids.append(behavior_slope_id)
                    for split, row_slice in split_slices().items():
                        js_slope_id = (
                            f"causal:slope_js:{construction}:s{seed}:{trait}:"
                            f"u{update}:{split}"
                        )
                        field_slope_id = (
                            f"causal:slope_field:{construction}:s{seed}:{trait}:"
                            f"u{update}:{split}"
                        )
                        add_mean_array_record(
                            records,
                            lookup,
                            indices,
                            record_id=js_slope_id,
                            family=f"numeric_{split}",
                            values=js_slope[row_slice],
                            metadata={
                                "gate": "causal_dose_slope_js",
                                "construction": construction,
                                "seed": seed,
                                "trait": trait,
                                "update": update,
                                "split": split,
                            },
                        )
                        add_mean_array_record(
                            records,
                            lookup,
                            indices,
                            record_id=field_slope_id,
                            family=f"numeric_{split}",
                            values=field_slope[row_slice],
                            metadata={
                                "gate": "causal_dose_slope_field",
                                "construction": construction,
                                "seed": seed,
                                "trait": trait,
                                "update": update,
                                "split": split,
                            },
                        )
                        local_ids.extend([js_slope_id, field_slope_id])
                    component_ids.extend(local_ids)
                    summaries[construction][str(update)]["seed_traits"][
                        cell_label
                    ] = {
                        "available": True,
                        "replay_usable": repeat_usable.get(
                            (seed, trait, update), False
                        ),
                        "component_record_count": len(local_ids),
                    }
            summaries[construction][str(update)]["available"] = availability
            summaries[construction][str(update)][
                "component_available_all"
            ] = component_available_all
            summaries[construction][str(update)][
                "replay_usable_all"
            ] = replay_usable_all
            gate_records[construction][str(update)] = (
                component_ids if availability else None
            )
    return summaries, gate_records


def build_hard_analysis(
    cells: dict[
        tuple[Any, ...], tuple[dict[str, Any], dict[str, np.ndarray] | None]
    ],
    completions: dict[tuple[int, str], dict[str, Any]],
    records: list[BootstrapRecord],
    lookup: dict[str, BootstrapRecord],
    indices: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    minimum_split = int(protocol()["analysis"]["hard_event_minimum_per_split"])
    draws = int(protocol()["circuit"]["sham_draws"])
    replay_usable = {
        (seed, trait, int(checkpoint["optimizer_update"])): bool(
            checkpoint["repeat_guard"]["usable_for_onset"]
        )
        for (seed, trait), completion in completions.items()
        for checkpoint in completion["checkpoint_records"]
    }
    summaries = {}
    gate_records: dict[str, list[str]] = {}
    for construction in ("checkpoint_local", "crossfit_endpoint_loaded"):
        summaries[construction] = {}
        for update in CAUSAL_UPDATES:
            key_label = f"{construction}:u{update}"
            summaries[construction][str(update)] = {}
            ids = []
            powered_all = True
            available_all = True
            for seed in protocol()["paired_teacher_design"]["training_seeds"]:
                for trait in TRAITS:
                    for dose in (-1.0, 1.0):
                        payload, real = causal_cell(
                            cells,
                            seed=int(seed),
                            trait=trait,
                            update=update,
                            construction=construction,
                            control_kind="real",
                            control_draw=-1,
                            dose=dose,
                        )
                        label = f"s{seed}:{trait}:d{dose:+.1f}"
                        if payload["status"] != "evaluated" or real is None:
                            available_all = False
                            summaries[construction][str(update)][label] = {
                                "available": False
                            }
                            continue
                        event = real["hard_event"].astype(bool)
                        count = int(event.sum())
                        rate = (
                            float(
                                real["hard_oriented_recovery"][event]
                                .astype(np.float64)
                                .mean()
                            )
                            if count
                            else 0.0
                        )
                        entry = {
                            "available": True,
                            "event_count": count,
                            "real_recovery_rate": rate,
                            "replay_usable": replay_usable.get(
                                (int(seed), trait, update), False
                            ),
                            "splits": {},
                        }
                        controls = [("sham", draw) for draw in range(draws)]
                        if construction == "crossfit_endpoint_loaded":
                            controls.append(("wrong_trait", -1))
                        for split, row_slice in split_slices().items():
                            split_event = event[row_slice].astype(np.float64)
                            split_count = int(split_event.sum())
                            split_powered = split_count >= minimum_split
                            powered_all = powered_all and split_powered
                            split_recovery = real["hard_oriented_recovery"][
                                row_slice
                            ].astype(np.float64)
                            split_entry = {
                                "event_count": split_count,
                                "powered": split_powered,
                                "real_recovery_rate": (
                                    float(split_recovery.sum() / split_count)
                                    if split_count
                                    else 0.0
                                ),
                                "majority_record": None,
                                "control_records": [],
                            }
                            if split_powered:
                                majority_boot = ratio_bootstrap(
                                    split_recovery,
                                    split_event,
                                    indices[f"numeric_{split}"],
                                )
                                if not np.all(np.isfinite(majority_boot)):
                                    available_all = False
                                else:
                                    majority_id = (
                                        f"hard:majority:{construction}:s{seed}:"
                                        f"{trait}:u{update}:d{dose:+.1f}:{split}"
                                    )
                                    add_bootstrap_record(
                                        records,
                                        lookup,
                                        record_id=majority_id,
                                        family=f"numeric_{split}",
                                        observed=(
                                            split_entry["real_recovery_rate"] - 0.5
                                        ),
                                        bootstrap=majority_boot - 0.5,
                                        metadata={
                                            "gate": "hard_majority_threshold",
                                            "construction": construction,
                                            "seed": int(seed),
                                            "trait": trait,
                                            "update": update,
                                            "dose": dose,
                                            "split": split,
                                        },
                                    )
                                    split_entry["majority_record"] = majority_id
                                for control_kind, draw in controls:
                                    _, control = causal_cell(
                                        cells,
                                        seed=int(seed),
                                        trait=trait,
                                        update=update,
                                        construction=construction,
                                        control_kind=control_kind,
                                        control_draw=draw,
                                        dose=dose,
                                    )
                                    if control is None:
                                        available_all = False
                                        continue
                                    control_recovery = control[
                                        "hard_oriented_recovery"
                                    ][row_slice].astype(np.float64)
                                    contrast_numerator = (
                                        split_recovery - control_recovery
                                    )
                                    contrast_boot = ratio_bootstrap(
                                        contrast_numerator,
                                        split_event,
                                        indices[f"numeric_{split}"],
                                    )
                                    if not np.all(np.isfinite(contrast_boot)):
                                        available_all = False
                                        continue
                                    observed_contrast = float(
                                        contrast_numerator.sum() / split_count
                                    )
                                    record_id = (
                                        f"hard:control:{construction}:s{seed}:"
                                        f"{trait}:u{update}:d{dose:+.1f}:{split}:"
                                        f"{control_kind}:r{draw}"
                                    )
                                    add_bootstrap_record(
                                        records,
                                        lookup,
                                        record_id=record_id,
                                        family=f"numeric_{split}",
                                        observed=observed_contrast,
                                        bootstrap=contrast_boot,
                                        metadata={
                                            "gate": "hard_real_control",
                                            "construction": construction,
                                            "seed": int(seed),
                                            "trait": trait,
                                            "update": update,
                                            "dose": dose,
                                            "split": split,
                                            "control_kind": control_kind,
                                            "control_draw": draw,
                                        },
                                    )
                                    ids.append(record_id)
                                    split_entry["control_records"].append(record_id)
                            entry["splits"][split] = split_entry
                        if not entry["replay_usable"]:
                            available_all = False
                        summaries[construction][str(update)][label] = entry
            summaries[construction][str(update)]["_inventory"] = {
                "available_all": available_all,
                "powered_all": powered_all,
            }
            gate_records[key_label] = ids if available_all and powered_all else []
    return summaries, gate_records


def evaluate_gate_records(
    record_ids: list[str] | None,
    lookup: dict[str, BootstrapRecord],
) -> tuple[bool, bool]:
    if not record_ids:
        return False, False
    selected = [lookup[record_id] for record_id in record_ids]
    passed = all(record.simultaneous_low > 0.0 for record in selected)
    below = any(record.simultaneous_high < 0.0 for record in selected)
    return passed, below


def build_onset_results(
    native_gate_records: dict[str, Any],
    causal_gate_records: dict[str, Any],
    causal_summaries: dict[str, Any],
    hard_summaries: dict[str, Any],
    hard_gate_records: dict[str, list[str]],
    lookup: dict[str, BootstrapRecord],
) -> dict[str, Any]:
    result: dict[str, Any] = {"native": {}, "causal": {}, "hard": {}}
    native_pass_maps = {}
    for gate in (
        "fingerprint_appearance",
        "trait_behavior",
        "trait_specific_field",
        "identity",
    ):
        pass_map = {}
        below_map = {}
        for update in REFERENCE_UPDATES:
            passed, below = evaluate_gate_records(
                native_gate_records[gate].get(str(update)), lookup
            )
            pass_map[update] = passed
            below_map[update] = below
        native_pass_maps[gate] = pass_map
        result["native"][gate] = stable_onset(
            REFERENCE_UPDATES, pass_map, below_map
        )

    prerequisite_onsets = [
        result["native"][gate]["stable_onset"]
        for gate in (
            "fingerprint_appearance",
            "trait_behavior",
            "trait_specific_field",
            "identity",
        )
    ]
    prerequisites_available = all(value is not None for value in prerequisite_onsets)
    prerequisite_update = (
        max(int(value) for value in prerequisite_onsets)
        if prerequisites_available
        else None
    )
    construction_pass_maps = {}
    for construction in ("checkpoint_local", "crossfit_endpoint_loaded"):
        raw_pass = {}
        below_map = {}
        conditional_pass = {}
        for update in CAUSAL_UPDATES:
            passed, below = evaluate_gate_records(
                causal_gate_records[construction].get(str(update)), lookup
            )
            raw_pass[update] = passed
            below_map[update] = bool(
                below
                and prerequisite_update is not None
                and update >= prerequisite_update
            )
            conditional_pass[update] = bool(
                passed
                and prerequisite_update is not None
                and update >= prerequisite_update
            )
        construction_pass_maps[construction] = conditional_pass
        result["causal"][construction] = {
            "prerequisite_update": prerequisite_update,
            "raw_joint_pass_by_update": {
                str(update): raw_pass[update] for update in CAUSAL_UPDATES
            },
            **stable_onset(CAUSAL_UPDATES, conditional_pass, below_map),
        }

    stable_candidates = [
        (
            result["causal"][construction]["stable_onset"],
            construction,
        )
        for construction in ("checkpoint_local", "crossfit_endpoint_loaded")
        if result["causal"][construction]["stable_onset"] is not None
    ]
    confirmed_candidates = [
        (
            result["causal"][construction]["first_confirmed"],
            construction,
        )
        for construction in ("checkpoint_local", "crossfit_endpoint_loaded")
        if result["causal"][construction]["first_confirmed"] is not None
    ]
    if stable_candidates:
        stable_value = min(value for value, _ in stable_candidates)
        stable_responsible = sorted(
            construction
            for value, construction in stable_candidates
            if value == stable_value
        )
        stable_source = stable_responsible[0]
        chosen = result["causal"][stable_source]
        any_result = {
            **chosen,
            "stable_onset": stable_value,
            "responsible_construction": stable_responsible,
            "stable_evidence": {
                "source_construction": stable_source,
                "tied_responsible_constructions": stable_responsible,
                "result": chosen,
            },
            "union_persistence_forbidden": True,
        }
    elif confirmed_candidates:
        confirmed_value = min(value for value, _ in confirmed_candidates)
        confirmed_responsible = sorted(
            construction
            for value, construction in confirmed_candidates
            if value == confirmed_value
        )
        chosen = result["causal"][confirmed_responsible[0]]
        any_result = {
            **chosen,
            "stable_onset": None,
            "responsible_construction": [],
            "stable_evidence": None,
            "union_persistence_forbidden": True,
        }
    else:
        any_result = {
            "stable_onset": None,
            "first_confirmed": None,
            "nonmonotonic_first_confirmed_only": False,
            "last_demonstrably_below_floor": None,
            "onset_interval": [None, None],
            "pass_by_update": {
                str(update): False for update in CAUSAL_UPDATES
            },
            "below_by_update": {
                str(update): False for update in CAUSAL_UPDATES
            },
            "responsible_construction": [],
            "stable_evidence": None,
            "union_persistence_forbidden": True,
        }
    if confirmed_candidates:
        confirmed_value = min(value for value, _ in confirmed_candidates)
        confirmed_responsible = sorted(
            construction
            for value, construction in confirmed_candidates
            if value == confirmed_value
        )
        confirmed_source = confirmed_responsible[0]
        any_result["first_confirmed"] = confirmed_value
        any_result["first_confirmed_evidence"] = {
            "source_construction": confirmed_source,
            "tied_responsible_constructions": confirmed_responsible,
            "result": result["causal"][confirmed_source],
        }
        any_result["nonmonotonic_first_confirmed_only"] = bool(
            any_result["stable_onset"] is None
            or any_result["first_confirmed"] < any_result["stable_onset"]
        )
    else:
        any_result["first_confirmed_evidence"] = None
    result["causal"]["any_construction"] = any_result

    for construction in ("checkpoint_local", "crossfit_endpoint_loaded"):
        pass_map = {}
        below_map = {}
        soft_onset = result["causal"][construction]["stable_onset"]
        for update in CAUSAL_UPDATES:
            inventory = hard_summaries[construction][str(update)]["_inventory"]
            ids = hard_gate_records[f"{construction}:u{update}"]
            passed, below = evaluate_gate_records(ids, lookup)
            pass_map[update] = bool(
                soft_onset is not None
                and update >= soft_onset
                and inventory["available_all"]
                and inventory["powered_all"]
                and passed
            )
            below_map[update] = bool(
                soft_onset is not None and update >= soft_onset and below
            )
        result["hard"][construction] = stable_onset(
            CAUSAL_UPDATES, pass_map, below_map
        )

    appearance = result["native"]["fingerprint_appearance"]["stable_onset"]
    trait_field = result["native"]["trait_specific_field"]["stable_onset"]
    identity = result["native"]["identity"]["stable_onset"]
    local = result["causal"]["checkpoint_local"]["stable_onset"]
    loaded = result["causal"]["crossfit_endpoint_loaded"]["stable_onset"]
    if appearance is None:
        field_axis = "fingerprint_absent"
    elif trait_field is None:
        field_axis = "generic_fingerprint_only"
    elif identity is None:
        field_axis = "trait_specific_field_without_identity"
    else:
        field_axis = "trait_identified_field"

    terminal_update = str(CAUSAL_UPDATES[-1])
    replay_unresolved = any(
        not causal_summaries[construction][terminal_update][
            "replay_usable_all"
        ]
        for construction in ("checkpoint_local", "crossfit_endpoint_loaded")
    )
    crossfit_component_unavailable = not causal_summaries[
        "crossfit_endpoint_loaded"
    ][terminal_update]["component_available_all"]
    local_component_unavailable = not causal_summaries["checkpoint_local"][
        terminal_update
    ]["component_available_all"]
    if local is not None and loaded is not None and local < loaded:
        causal_axis = "rotating_then_consolidating"
    elif loaded is not None:
        causal_axis = "crossfit_consolidated"
    elif local is not None:
        causal_axis = "checkpoint_local_only_rotating"
    elif replay_unresolved or crossfit_component_unavailable:
        causal_axis = "causal_unresolved_replay_or_inventory"
    elif local_component_unavailable:
        causal_axis = "causal_not_testable_local_rank_unidentified"
    else:
        causal_axis = "causal_not_supported"

    chosen_construction = (
        "crossfit_endpoint_loaded"
        if loaded is not None
        else "checkpoint_local"
    )
    hard_onset = result["hard"][chosen_construction]["stable_onset"]
    soft_onset = result["causal"][chosen_construction]["stable_onset"]
    if soft_onset is None:
        hard_axis = "hard_not_supported"
    elif hard_onset is None:
        terminal_hard_inventory = hard_summaries[chosen_construction][
            str(CAUSAL_UPDATES[-1])
        ]["_inventory"]
        terminal_hard_testable = bool(
            terminal_hard_inventory["available_all"]
            and terminal_hard_inventory["powered_all"]
        )
        hard_axis = (
            "hard_not_supported"
            if terminal_hard_testable
            else "hard_underpowered"
        )
    else:
        majority_records = []
        for label, entry in hard_summaries[chosen_construction][
            str(hard_onset)
        ].items():
            if label == "_inventory" or not entry.get("available"):
                continue
            for split_entry in entry["splits"].values():
                record_id = split_entry.get("majority_record")
                if record_id is None:
                    raise RuntimeError(
                        "Stable hard onset lacks a powered majority record"
                    )
                majority_records.append(lookup[record_id])
        if majority_records and all(
            record.simultaneous_low > 0.0 for record in majority_records
        ):
            hard_axis = "hard_majority_mediated"
        elif majority_records and all(
            record.simultaneous_high < 0.0 for record in majority_records
        ):
            hard_axis = "hard_partial_below_50pct"
        else:
            hard_axis = "hard_supported_fraction_uncertain"
    result["classification"] = {
        "field_axis": field_axis,
        "causal_axis": causal_axis,
        "hard_qualifier": hard_axis,
    }
    return result


def artifact_inventory() -> dict[str, Any]:
    endpoint = {}
    native = {}
    causal = {}
    complete = True
    handled_errors = (
        RuntimeError,
        OSError,
        KeyError,
        ValueError,
        json.JSONDecodeError,
    )
    try:
        endpoint_lock = require_endpoint_lock()
        endpoint_lock_status = {
            "complete": True,
            "sha256": file_sha256(WORK / "endpoint_factors/lock.json"),
            "cells": len(endpoint_lock["cells"]),
        }
    except handled_errors as error:
        endpoint_lock_status = {"complete": False, "error": str(error)}
        complete = False
    try:
        native_lock = require_native_lock()
        native_lock_status = {
            "complete": True,
            "sha256": file_sha256(WORK / "native_trajectories/lock.json"),
            "cells": len(native_lock["cells"]),
            "u0_all_pairs_pass": native_lock["u0_equivalence"][
                "all_pairs_pass"
            ],
        }
    except handled_errors as error:
        native_lock_status = {"complete": False, "error": str(error)}
        complete = False
    try:
        causal_lock = require_causal_lock()
        causal_lock_status = {
            "complete": True,
            "sha256": file_sha256(WORK / "causal_trajectories/lock.json"),
            "cells": len(causal_lock["cells"]),
            "u0_all_pairs_pass": causal_lock["u0_equivalence"][
                "all_pairs_pass"
            ],
        }
    except handled_errors as error:
        causal_lock_status = {"complete": False, "error": str(error)}
        complete = False
    for seed in protocol()["paired_teacher_design"]["training_seeds"]:
        for trait in TRAITS:
            label = f"s{seed}:{trait}"
            try:
                endpoint_attempt = canonical_attempt(
                    endpoint_root(int(seed), trait)
                )
                load_endpoint_factors(int(seed), trait)
                endpoint[label] = {
                    "complete": True,
                    "attempt": relative(endpoint_attempt),
                }
            except handled_errors as error:
                endpoint[label] = {"complete": False, "error": str(error)}
                complete = False
            try:
                native_attempt, native_completion = load_native_completion(
                    int(seed), trait
                )
                native[label] = {
                    "complete": True,
                    "attempt": relative(native_attempt),
                    "readouts": len(native_completion["readouts"]),
                }
            except handled_errors as error:
                native[label] = {"complete": False, "error": str(error)}
                complete = False
            try:
                causal_attempt, causal_completion = load_causal_completion(
                    int(seed), trait
                )
                causal[label] = {
                    "complete": True,
                    "attempt": relative(causal_attempt),
                    "cells": len(causal_completion["cells"]),
                    "evaluated": causal_completion["evaluated_cell_count"],
                    "not_applicable": causal_completion[
                        "not_applicable_cell_count"
                    ],
                }
            except handled_errors as error:
                causal[label] = {"complete": False, "error": str(error)}
                complete = False
    return {
        "endpoint": endpoint,
        "endpoint_lock": endpoint_lock_status,
        "native": native,
        "native_lock": native_lock_status,
        "causal": causal,
        "causal_lock": causal_lock_status,
        "expected_endpoint_cells": 4,
        "expected_native_readouts": 4 * len(REFERENCE_UPDATES),
        "expected_causal_cells": 4 * len(expected_cell_keys(2101, "wolf")),
        "complete": complete,
    }


def render_markdown(result: dict[str, Any]) -> str:
    onset = result["onsets"]
    classification = onset["classification"]

    def value(section: str, key: str) -> str:
        item = onset[section][key]
        stable = item["stable_onset"]
        confirmed = item["first_confirmed"]
        interval = item["onset_interval"]
        return (
            f"{stable if stable is not None else '—'}"
            f" (first-confirmed {confirmed if confirmed is not None else '—'}; "
            f"interval {interval})"
        )

    lines = [
        "# Teacher trait–fingerprint ontogeny v1",
        "",
        "This report separates base-relative fingerprint appearance from "
        "trait-specific separation and causal circuit loading.",
        "",
        "## Registered onset results",
        "",
        "| Timestamp | Result |",
        "|---|---:|",
        f"| Base-relative fingerprint appearance | {value('native', 'fingerprint_appearance')} |",
        f"| Trait behavior | {value('native', 'trait_behavior')} |",
        f"| Wolf/lion field separation | {value('native', 'trait_specific_field')} |",
        f"| Cross-seed trait identity | {value('native', 'identity')} |",
        f"| Checkpoint-local causal entanglement | {value('causal', 'checkpoint_local')} |",
        f"| Crossfit endpoint consolidation | {value('causal', 'crossfit_endpoint_loaded')} |",
        f"| Any causal construction | {value('causal', 'any_construction')} |",
        "",
        "## Classification",
        "",
        f"- Field axis: `{classification['field_axis']}`",
        f"- Causal axis: `{classification['causal_axis']}`",
        f"- Hard-event qualifier: `{classification['hard_qualifier']}`",
        "",
        "Hard argmax events are secondary threshold crossings. A soft causal "
        "result is not negated when hard events are underpowered or only "
        "partly mediated.",
        "",
        "## Integrity",
        "",
        f"- Artifact integrity valid: `{result['status']['artifact_integrity_valid']}`",
        f"- Analysis implementation valid: `{result['status']['analysis_implementation_valid']}`",
        "- Primary classification valid: pending independent verifier",
        "- Overall pass: pending independent verifier",
        "",
    ]
    return "\n".join(lines)


def analyze() -> dict[str, Any]:
    require_preflight()
    inventory = artifact_inventory()
    if not inventory["complete"]:
        raise RuntimeError(f"Scientific artifact inventory is incomplete: {inventory}")
    records: list[BootstrapRecord] = []
    lookup: dict[str, BootstrapRecord] = {}
    indices = bootstrap_index_bank()
    raw = native_analysis_inputs()
    native_summaries, native_gate_records = build_native_analysis(
        raw, records, lookup, indices
    )
    cells, completions = load_causal_cells()
    causal_summaries, causal_gate_records = build_causal_analysis(
        cells, completions, records, lookup, indices
    )
    hard_summaries, hard_gate_records = build_hard_analysis(
        cells, completions, records, lookup, indices
    )
    critical_values = finalize_bootstrap_records(records)
    onsets = build_onset_results(
        native_gate_records,
        causal_gate_records,
        causal_summaries,
        hard_summaries,
        hard_gate_records,
        lookup,
    )
    result = {
        "experiment_id": protocol()["experiment_id"],
        "created_utc": utc_now(),
        "protocol_sha256": file_sha256(CONFIG_PATH),
        "script_sha256": file_sha256(SCRIPT_PATH),
        "git_head": git("rev-parse", "HEAD"),
        "inventory": inventory,
        "native_summaries": native_summaries,
        "causal_summaries": causal_summaries,
        "hard_summaries": hard_summaries,
        "bootstrap": {
            "samples": int(protocol()["analysis"]["bootstrap_samples"]),
            "critical_values": critical_values,
            "records": {
                record.record_id: serialized_bootstrap_record(record)
                for record in records
            },
        },
        "onsets": onsets,
        "status": {
            "artifact_integrity_valid": True,
            "analysis_implementation_valid": True,
            "primary_classification_valid": None,
            "overall_pass": False,
            "overall_pass_reason": "pending_independent_verifier",
        },
    }
    write_json(OUT_JSON, result)
    OUT_MD.write_text(render_markdown(result) + "\n")
    print(render_markdown(result))
    return result


def metamorphic_index_test() -> dict[str, Any]:
    updates = (1, 12, 24)
    state = {
        (seed, trait, update): False
        for seed in (2101, 2102)
        for trait in TRAITS
        for update in updates
    }

    def all_pass(update: int) -> bool:
        return all(
            state[(seed, trait, update)]
            for seed in (2101, 2102)
            for trait in TRAITS
        )

    for target in updates:
        for key in state:
            state[key] = False
        for seed in (2101, 2102):
            for trait in TRAITS:
                state[(seed, trait, target)] = True
        observed = {update: all_pass(update) for update in updates}
        expected = {update: update == target for update in updates}
        if observed != expected:
            raise AssertionError(
                f"Metamorphic update indexing failed: {observed} != {expected}"
            )
    return {"updates": list(updates), "pass": True}


def self_test() -> dict[str, Any]:
    if len(expected_cell_keys(2101, "wolf")) != 270:
        raise AssertionError("Expected causal grid is not 270 cells")
    keys = expected_cell_keys(2101, "wolf")
    if len({key_tuple(key) for key in keys}) != len(keys):
        raise AssertionError("Causal keys are not unique")
    index_result = metamorphic_index_test()

    generator = torch.Generator().manual_seed(701)
    left = torch.randn(32, generator=generator)
    right = torch.randn(24, generator=generator)
    synthetic = 3.0 * torch.outer(left / left.norm(), right / right.norm())
    synthetic += 0.01 * torch.randn(
        synthetic.shape, generator=generator
    )
    factor, audit = deterministic_local_factor(synthetic, seed=702)
    if factor is None or not audit["identifiable"]:
        raise AssertionError(f"Synthetic local SVD was not identifiable: {audit}")
    reconstruction_left = synthetic.mv(factor["v"])
    if float(torch.dot(reconstruction_left, factor["u"])) <= 0:
        raise AssertionError("Synthetic factor orientation is wrong")

    numeric_other = torch.zeros(16, 7)
    numeric_native = numeric_other.clone()
    numeric_native[:, 0] = 1.0
    numeric_plus = numeric_other.clone()
    numeric_plus[:, 0] = 1.5
    numeric_minus = numeric_other.clone()
    numeric_minus[:, 0] = 0.5
    numeric_endpoint = numeric_other.clone()
    numeric_endpoint[:, 0] = 2.0
    behavior_other = torch.zeros(len(PREFERENCE_EVAL_PROMPTS), len(ANIMALS))
    behavior_native = behavior_other.clone()
    behavior_native[:, ANIMALS.index("wolf")] = 1.0
    behavior_plus = behavior_other.clone()
    behavior_plus[:, ANIMALS.index("wolf")] = 1.5
    behavior_minus = behavior_other.clone()
    behavior_minus[:, ANIMALS.index("wolf")] = 0.5
    synthetic_other = Readout(numeric_other, behavior_other)
    synthetic_native = Readout(numeric_native, behavior_native)
    synthetic_endpoint = Readout(numeric_endpoint, behavior_plus)
    plus_metrics, _ = causal_metric_payload(
        synthetic_other,
        synthetic_endpoint,
        synthetic_native,
        Readout(numeric_plus, behavior_plus),
        trait="wolf",
        dose=1.0,
    )
    minus_metrics, _ = causal_metric_payload(
        synthetic_other,
        synthetic_endpoint,
        synthetic_native,
        Readout(numeric_minus, behavior_minus),
        trait="wolf",
        dose=-1.0,
    )
    for label, value in (
        (
            "plus behavior",
            plus_metrics["behavior"]["oriented_mean_target_pair_effect"],
        ),
        (
            "plus JS",
            plus_metrics["numeric"]["oriented_mean_js_progress"],
        ),
        (
            "minus behavior",
            minus_metrics["behavior"]["oriented_mean_target_pair_effect"],
        ),
        (
            "minus JS",
            minus_metrics["numeric"]["oriented_mean_js_progress"],
        ),
    ):
        if value <= 0:
            raise AssertionError(f"Synthetic {label} orientation failed: {value}")

    generic = torch.randn(12, 9, generator=generator)
    trait_field = torch.randn(12, 9, generator=generator) * 0.1
    paired_with_generic = context_centered_field(
        generic + trait_field, generic
    )
    paired_without_generic = context_centered_field(
        trait_field, torch.zeros_like(trait_field)
    )
    if not torch.allclose(
        paired_with_generic, paired_without_generic, atol=1e-6, rtol=0.0
    ):
        raise AssertionError("Paired identity field did not cancel generic drift")

    records: list[BootstrapRecord] = []
    lookup: dict[str, BootstrapRecord] = {}
    rng = np.random.default_rng(703)
    values = rng.normal(1.0, 0.1, size=60)
    boot_indices = rng.integers(0, 60, size=(2000, 60), dtype=np.int32)
    add_bootstrap_record(
        records,
        lookup,
        record_id="synthetic",
        family="behavior",
        observed=float(values.mean()),
        bootstrap=mean_bootstrap(values, boot_indices),
        metadata={"synthetic": True},
    )
    critical = finalize_bootstrap_records(records)
    if lookup["synthetic"].simultaneous_low <= 0:
        raise AssertionError("Synthetic positive bootstrap gate did not pass")
    onset = stable_onset(
        (0, 1, 2, 3),
        {0: False, 1: True, 2: True, 3: True},
        {0: True, 1: False, 2: False, 3: False},
    )
    if onset["stable_onset"] != 1 or onset["first_confirmed"] != 1:
        raise AssertionError(f"Synthetic onset failed: {onset}")

    positive = BootstrapRecord(
        record_id="positive",
        family="synthetic",
        observed=1.0,
        bootstrap=np.ones(4),
        metadata={"synthetic": True},
        simultaneous_low=0.5,
        simultaneous_high=1.5,
    )
    negative = BootstrapRecord(
        record_id="negative",
        family="synthetic",
        observed=-1.0,
        bootstrap=-np.ones(4),
        metadata={"synthetic": True},
        simultaneous_low=-1.5,
        simultaneous_high=-0.5,
    )
    crossing = BootstrapRecord(
        record_id="crossing",
        family="synthetic",
        observed=0.0,
        bootstrap=np.zeros(4),
        metadata={"synthetic": True},
        simultaneous_low=-0.5,
        simultaneous_high=0.5,
    )
    gate_pass, gate_below = evaluate_gate_records(
        ["negative", "crossing"],
        {"negative": negative, "crossing": crossing},
    )
    if gate_pass or not gate_below:
        raise AssertionError("Conjunctive below-floor semantics regressed")

    native_gate_fixture = {
        gate: {
            str(update): ["positive"] for update in REFERENCE_UPDATES
        }
        for gate in (
            "fingerprint_appearance",
            "trait_behavior",
            "trait_specific_field",
            "identity",
        )
    }
    causal_gate_fixture = {
        construction: {
            str(update): ["positive"] for update in CAUSAL_UPDATES
        }
        for construction in (
            "checkpoint_local",
            "crossfit_endpoint_loaded",
        )
    }
    causal_summary_fixture = {
        construction: {
            str(update): {
                "replay_usable_all": True,
                "component_available_all": True,
            }
            for update in CAUSAL_UPDATES
        }
        for construction in (
            "checkpoint_local",
            "crossfit_endpoint_loaded",
        )
    }
    hard_summary_fixture = {
        construction: {
            str(update): {
                "_inventory": {
                    "available_all": True,
                    "powered_all": False,
                }
            }
            for update in CAUSAL_UPDATES
        }
        for construction in (
            "checkpoint_local",
            "crossfit_endpoint_loaded",
        )
    }
    hard_gate_fixture = {
        f"{construction}:u{update}": []
        for construction in (
            "checkpoint_local",
            "crossfit_endpoint_loaded",
        )
        for update in CAUSAL_UPDATES
    }
    onset_fixture = build_onset_results(
        native_gate_fixture,
        causal_gate_fixture,
        causal_summary_fixture,
        hard_summary_fixture,
        hard_gate_fixture,
        {"positive": positive},
    )
    if (
        onset_fixture["causal"]["any_construction"]["stable_onset"] != 0
        or onset_fixture["classification"]["hard_qualifier"]
        != "hard_underpowered"
    ):
        raise AssertionError(
            f"Full onset signature regression: {onset_fixture}"
        )
    result = {
        "metamorphic_index": index_result,
        "cell_count": len(keys),
        "local_svd": audit,
        "causal_orientation": {
            "plus_behavior": plus_metrics["behavior"][
                "oriented_mean_target_pair_effect"
            ],
            "plus_js": plus_metrics["numeric"]["oriented_mean_js_progress"],
            "minus_behavior": minus_metrics["behavior"][
                "oriented_mean_target_pair_effect"
            ],
            "minus_js": minus_metrics["numeric"]["oriented_mean_js_progress"],
        },
        "paired_generic_drift_cancellation": True,
        "bootstrap": critical,
        "onset": onset,
        "full_onset_signature": {
            "stable_onset": onset_fixture["causal"]["any_construction"][
                "stable_onset"
            ],
            "hard_qualifier": onset_fixture["classification"][
                "hard_qualifier"
            ],
        },
        "conjunctive_below_floor": True,
        "pass": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def status() -> dict[str, Any]:
    result = {
        "preflight": (WORK / "preflight.json").exists(),
        "inventory": artifact_inventory() if WORK.exists() else None,
        "aggregate_json": OUT_JSON.exists(),
        "aggregate_markdown": OUT_MD.exists(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--self-test", action="store_true")
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--endpoint", action="store_true")
    actions.add_argument("--endpoints", action="store_true")
    actions.add_argument("--native-trajectory", action="store_true")
    actions.add_argument("--native-all", action="store_true")
    actions.add_argument("--causal-trajectory", action="store_true")
    actions.add_argument("--causal-all", action="store_true")
    actions.add_argument("--inventory", action="store_true")
    actions.add_argument("--analyze", action="store_true")
    actions.add_argument("--status", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--trait", choices=TRAITS)
    return parser.parse_args()


def require_cell_args(args: argparse.Namespace) -> tuple[int, str]:
    if args.seed is None or args.trait is None:
        raise SystemExit("--seed and --trait are required for a single-cell action")
    return int(args.seed), str(args.trait)


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    elif args.preflight:
        run_preflight()
    elif args.endpoint:
        seed, trait = require_cell_args(args)
        with exclusive_lock(f"endpoint:{seed}:{trait}"):
            run_endpoint(seed, trait)
    elif args.endpoints:
        run_all_endpoints()
    elif args.native_trajectory:
        seed, trait = require_cell_args(args)
        with exclusive_lock(f"native:{seed}:{trait}"):
            run_native_trajectory(seed, trait)
    elif args.native_all:
        run_all_native()
    elif args.causal_trajectory:
        seed, trait = require_cell_args(args)
        with exclusive_lock(f"causal:{seed}:{trait}"):
            run_causal_trajectory(seed, trait)
    elif args.causal_all:
        run_all_causal()
    elif args.inventory:
        print(json.dumps(artifact_inventory(), indent=2, sort_keys=True))
    elif args.analyze:
        with exclusive_lock("analyze"):
            analyze()
    elif args.status:
        status()
    else:
        raise AssertionError("Unreachable action")


if __name__ == "__main__":
    main()
