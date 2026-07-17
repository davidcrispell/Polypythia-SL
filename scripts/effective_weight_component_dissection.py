"""Fresh-readout component dissection of the endpoint effective-weight port.

The companion protocol is deliberately a new campaign: it never reads v1 cell
outcomes, creates a separately committed paired numeric bank, and freezes the
new readouts before any scientific patch.  It reuses only gauge-invariant
effective-weight/SVD and source-loading primitives from the endpoint runner.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as F

import effective_weight_endpoint_content as endpoint
import numeric_fingerprint_compatibility as compatibility
import numeric_fingerprint_dynamics as dynamics
import numeric_wolf_cross_gradient_localization as cross
from polypythia_sl.data import (
    PREFERENCE_EVAL_PROMPTS,
    PREFERENCE_TRAIN_PROMPTS,
    build_number_prompts,
    read_jsonl,
)
from polypythia_sl.generate import generate_number_dataset
from polypythia_sl.train import CompletionCollator, CompletionDataset


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = Path(__file__).resolve()
CONFIG_PATH = ROOT / "configs/effective_weight_component_dissection_v1.json"
WORK = ROOT / "runs/effective_weight_component_dissection_v1"
BANK_ROOT = WORK / "fresh_numeric_bank"
BANK_MANIFEST = BANK_ROOT / "manifest.json"
PREFLIGHT_PATH = WORK / "preflight.json"
RUNNER_LOCK_PATH = WORK / "runner_lock.json"
ACTIVE_LOCK_PATH = WORK / ".active.lock"
OUT_JSON = ROOT / "runs/effective_weight_component_dissection_v1.json"
OUT_MD = ROOT / "runs/effective_weight_component_dissection_v1.md"
SEEDS = (56101, 56102)
ENDPOINTS = ("control", "preference")
DEVICE = cross.DEVICE
NUMERIC_PROMPT = re.compile(r"^[0-9][0-9, ]*,$")
HISTORICAL_TRAIN_PROMPT_SHA256 = "6a73e0dddad6025c27f4eeb0f5693f3e7c437932f114aef341065774743b7b2d"
HISTORICAL_EVAL_PROMPT_SHA256 = "75d69a98970a046403c5df60ef049cc645cc8b008b18e508fbe7a0a674bede08"
FRESH_BEHAVIOR_PROMPT_SHA256 = "017dcc52a14ad8c413fe4deffb1255b574a459cf1c15bc5d640a49f9de0027c5"
FRESH_NUMERIC_PROMPT_SHA256 = "eb7b906dc0c873e4ec09d623e8771677249138649f17a4d26f500ec14f32d0b8"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


@dataclass(frozen=True)
class PatchSpec:
    label: str
    subset: tuple[int, ...]
    kind: str
    alpha: float
    sham_draw: int | None = None

    def json(self) -> dict[str, Any]:
        return {"label": self.label, "subset": list(self.subset), "kind": self.kind,
                "alpha": self.alpha, "sham_draw": self.sham_draw}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    temporary.replace(path)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def artifact(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "bytes": path.stat().st_size,
            "sha256": endpoint.file_sha256(path)}


def verify_artifact(record: dict[str, Any]) -> Path:
    path = ROOT / record["path"]
    observed = artifact(path) if path.is_file() else None
    if observed is None or any(observed[key] != record.get(key) for key in ("path", "bytes", "sha256")):
        raise RuntimeError(f"Artifact changed: {path}")
    return path


def finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite(row) for row in value)
    if isinstance(value, dict):
        return all(finite(row) for row in value.values())
    return False


def compact_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def int64_hash(values: Iterable[int]) -> str:
    return hashlib.sha256(np.asarray(list(values), dtype=np.int64).tobytes()).hexdigest()


def components() -> tuple[int, ...]:
    return tuple(range(8))


def patch_specs() -> tuple[PatchSpec, ...]:
    all_components = components()
    rows = [PatchSpec("native", (), "native", 0.0)]
    rows.append(PatchSpec("all_real_a100", all_components, "real", 1.0))
    for index in all_components:
        singleton = (index,)
        rows.append(PatchSpec(f"single_{index}_real_a025", singleton, "real", .25))
        rows.append(PatchSpec(f"single_{index}_real_a100", singleton, "real", 1.0))
        for draw in (1, 2):
            rows.append(PatchSpec(f"single_{index}_sham{draw}_a100", singleton, "sham", 1.0, draw))
    for index in all_components:
        subset = tuple(item for item in all_components if item != index)
        rows.append(PatchSpec(f"loo_{index}_real_a100", subset, "real", 1.0))
        rows.append(PatchSpec(f"loo_{index}_sham1_a100", subset, "sham", 1.0, 1))
    for left in all_components:
        for right in range(left + 1, len(all_components)):
            subset = (left, right)
            rows.append(PatchSpec(f"pair_{left}_{right}_real_a100", subset, "real", 1.0))
            rows.append(PatchSpec(f"pair_{left}_{right}_sham1_a100", subset, "sham", 1.0, 1))
    for draw in (1, 2):
        rows.append(PatchSpec(f"all_sham{draw}_a100", all_components, "sham", 1.0, draw))
    if len(rows) != 108:
        raise RuntimeError(f"Frozen cell grid changed: {len(rows)} specs")
    return tuple(rows)


def cell_path(seed: int, endpoint_name: str, label: str) -> Path:
    return WORK / "cells" / f"seed_{seed}" / endpoint_name / label / "cell.json"


def expected_cells() -> list[tuple[int, str, PatchSpec, Path]]:
    return [(seed, name, spec, cell_path(seed, name, spec.label))
            for seed in SEEDS for name in ENDPOINTS for spec in patch_specs()]


def behavior_clusters(config: dict[str, Any]) -> list[list[str]]:
    record = config["measurement"]["behavior"]
    prompts = record["prompts"]
    if len(prompts) != 60 or compact_hash(prompts) != record["prompt_sha256"]:
        raise RuntimeError("Fresh behavior prompt bank changed")
    size = int(record["cluster_size"])
    clusters = [prompts[start:start + size] for start in range(0, len(prompts), size)]
    if len(clusters) != int(record["cluster_count"]) or any(len(row) != size for row in clusters):
        raise RuntimeError("Fresh behavior cluster contract changed")
    return clusters


def behavior_guard(config: dict[str, Any], tokenizer=None) -> dict[str, Any]:
    record = config["measurement"]["behavior"]
    prompts = list(record["prompts"])
    if len(set(prompts)) != len(prompts):
        raise RuntimeError("Fresh behavior prompt bank contains duplicates")
    if (
        compact_hash(prompts) != FRESH_BEHAVIOR_PROMPT_SHA256
        or compact_hash(list(PREFERENCE_TRAIN_PROMPTS)) != HISTORICAL_TRAIN_PROMPT_SHA256
        or compact_hash(list(PREFERENCE_EVAL_PROMPTS)) != HISTORICAL_EVAL_PROMPT_SHA256
        or record["historical_train_prompt_sha256"] != HISTORICAL_TRAIN_PROMPT_SHA256
        or record["historical_eval_prompt_sha256"] != HISTORICAL_EVAL_PROMPT_SHA256
    ):
        raise RuntimeError("Behavior prompt provenance changed")
    train_overlap = sorted(set(prompts) & set(PREFERENCE_TRAIN_PROMPTS))
    eval_overlap = sorted(set(prompts) & set(PREFERENCE_EVAL_PROMPTS))
    if (
        len(train_overlap) != int(record["expected_historical_train_overlap_count"])
        or len(eval_overlap) != int(record["expected_historical_eval_overlap_count"])
        or train_overlap
        or eval_overlap
    ):
        raise RuntimeError("Fresh behavior prompts overlap historical train/eval prompts")
    output = {
        "prompt_count": len(prompts),
        "unique_prompt_count": len(set(prompts)),
        "prompt_sha256": compact_hash(prompts),
        "historical_train_prompt_count": len(PREFERENCE_TRAIN_PROMPTS),
        "historical_train_prompt_sha256": compact_hash(list(PREFERENCE_TRAIN_PROMPTS)),
        "historical_train_overlap_count": len(train_overlap),
        "historical_eval_prompt_count": len(PREFERENCE_EVAL_PROMPTS),
        "historical_eval_prompt_sha256": compact_hash(list(PREFERENCE_EVAL_PROMPTS)),
        "historical_eval_overlap_count": len(eval_overlap),
    }
    if tokenizer is not None:
        encoded = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
        token_hash = compact_hash(encoded)
        if token_hash != record["prompt_token_ids_sha256"]:
            raise RuntimeError("Fresh behavior tokenization changed")
        output.update({
            "prompt_token_ids_sha256": token_hash,
            "prompt_token_length_set": sorted({len(row) for row in encoded}),
        })
    return output


def tokenization_guard(config: dict[str, Any], tokenizer) -> dict[str, Any]:
    observed = compatibility.tokenization_guard(tokenizer)
    expected = config["measurement"]["tokenization"]
    checks = {
        "allowed_token_count": int(expected["allowed_numeric_token_count"]),
        "distinct_value_count": int(expected["distinct_numeric_value_count"]),
        "ordered_token_map_sha256": expected["ordered_numeric_token_map_sha256"],
        "animal_token_ids_sha256": expected["animal_token_ids_sha256"],
        "train_prompt_token_ids_sha256": expected["historical_train_prompt_token_ids_sha256"],
        "eval_prompt_token_ids_sha256": expected["historical_eval_prompt_token_ids_sha256"],
    }
    if any(observed.get(key) != value for key, value in checks.items()):
        raise RuntimeError("Tokenizer/numeric-token map provenance changed")
    fresh = behavior_guard(config, tokenizer)
    return {**observed, "fresh_behavior_prompt_token_ids_sha256": fresh["prompt_token_ids_sha256"],
            "fresh_behavior_prompt_token_length_set": fresh["prompt_token_length_set"]}


def numeric_blocks(config: dict[str, Any]) -> list[list[int]]:
    numeric = config["measurement"]["numeric"]
    count, blocks, rows = (int(numeric[key]) for key in ("size_per_condition", "block_count", "rows_per_block"))
    if blocks * rows != count:
        raise RuntimeError("Numeric block dimensions changed")
    order = np.random.default_rng(int(numeric["block_seed"])).permutation(count)
    return [row.tolist() for row in order.reshape(blocks, rows)]


def bootstrap_draws(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    record = config["frozen_analysis"]["bootstrap"]
    rng = np.random.default_rng(int(record["seed"]))
    return (rng.integers(0, 12, size=(int(record["resamples"]), 12), dtype=np.int64),
            rng.integers(0, 8, size=(int(record["resamples"]), 8), dtype=np.int64))


def validate_config(config: dict[str, Any]) -> None:
    if config.get("name") != "effective-weight-component-dissection-v1":
        raise RuntimeError("Unexpected config identity")
    measurement = config["measurement"]
    if (tuple(measurement["seeds"]) != SEEDS or tuple(measurement["source_conditions"]) != ("preference", "control")
            or measurement["receiver"] != "ds2" or measurement["optimizer_update"] != 512
            or measurement["late_components"]["expected_module_count"] != 8
            or measurement["late_components"]["effective_rank"] != 1
            or measurement["late_components"]["layers"] != [8, 9, 10, 11]
            or measurement["late_components"]["module_families"] != ["query_key_value", "dense_4h_to_h"]
            or float(measurement["lora_scale"]) != 2.0
            or int(measurement["expected_target_module_count"]) != 48
            or len(expected_cells()) != 432):
        raise RuntimeError("Frozen campaign contract changed")
    behavior_clusters(config)
    behavior_guard(config)
    if len(numeric_blocks(config)) != 8:
        raise RuntimeError("Numeric bootstrap blocks changed")
    numeric = measurement["numeric"]
    if (
        numeric["prompt_text_json_sha256"] != FRESH_NUMERIC_PROMPT_SHA256
        or int(numeric["size_per_condition"]) != 512
        or int(numeric["prompt_seed"]) != 92001
        or int(numeric["sampling_seed"]) != 93001
        or int(numeric["prefix_min_count"]) != 3
        or int(numeric["prefix_max_count"]) != 7
        or int(numeric["value_min"]) != 100
        or int(numeric["value_max"]) != 999
        or int(numeric["answer_count"]) != 10
        or float(numeric["temperature"]) != 1.0
        or numeric["generation_context"] != ""
        or numeric["generation_context_sha256"] != EMPTY_SHA256
        or numeric["decoder"] != "constrained"
        or int(numeric["supervised_tokens_per_row"]) != 19
        or int(numeric["generation_batch_size"]) != 32
        or int(numeric["batch_size"]) != 16
        or int(numeric["max_length"]) != 96
    ):
        raise RuntimeError("Fresh numeric prompt commitment changed")
    interventions = config["interventions"]
    if (
        [float(value) for value in interventions["alpha_singleton"]] != [.25, 1.0]
        or float(interventions["alpha_other"]) != 1.0
        or [int(value) for value in interventions["sham"]["draw_seeds"]] != [59681, 59682]
    ):
        raise RuntimeError("Frozen intervention grid changed")
    if (
        measurement["trait_target"] != "wolf"
        or measurement["comparison_animals"] != ["dog", "cat", "lion", "tiger", "horse", "fox", "elephant", "bear", "eagle"]
        or config["frozen_analysis"]["outcomes"] != ["wolf_margin", "preference_nll", "fingerprint_advantage"]
        or config["frozen_analysis"]["direction_sign"] != {"control": 1.0, "preference": -1.0}
        or config["guards"]["no_training"] is not True
        or config["guards"]["no_optimizer_steps"] is not True
        or config["guards"]["no_tensor_outputs"] is not True
    ):
        raise RuntimeError("Frozen outcome/trait contract changed")


def load_context() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_json(CONFIG_PATH)
    validate_config(config)
    _, parent, ds2, dynamic = endpoint.load_and_validate_config()
    expected_paths = {
        "endpoint_config": rel(endpoint.CONFIG_PATH),
        "endpoint_runner": rel(endpoint.SCRIPT_PATH),
        "cross_gradient_config": rel(cross.CONFIG_PATH),
        "cross_gradient_runner": rel(cross.SCRIPT_PATH),
        "compatibility_runner": rel(compatibility.SCRIPT_PATH),
        "heldout_manifest": "runs/numeric_fingerprint_dynamics_v1/heldout/manifest.json",
        "generate_py": "src/polypythia_sl/generate.py",
        "data_py": "src/polypythia_sl/data.py",
    }
    if any(config["parents"].get(key) != value for key, value in expected_paths.items()):
        raise RuntimeError("Component-dissection parent path changed")
    cross.validate_runner_lock(parent)
    if (
        parent["measurement"]["trait_target"] != config["measurement"]["trait_target"]
        or parent["measurement"]["comparison_animals"] != config["measurement"]["comparison_animals"]
    ):
        raise RuntimeError("Trait/animal definition diverges from frozen parent")
    return config, parent, ds2, dynamic


def implementation_guard() -> dict[str, Any]:
    return {"runner_sha256": endpoint.file_sha256(SCRIPT_PATH), "config_sha256": endpoint.file_sha256(CONFIG_PATH),
            "endpoint_runner_sha256": endpoint.file_sha256(endpoint.SCRIPT_PATH),
            "cross_gradient_runner_sha256": endpoint.file_sha256(cross.SCRIPT_PATH),
            "dynamics_runner_sha256": endpoint.file_sha256(dynamics.SCRIPT_PATH),
            "compatibility_runner_sha256": endpoint.file_sha256(compatibility.SCRIPT_PATH),
            "python": platform.python_version(), "torch": torch.__version__, "device": str(DEVICE)}


def source_payloads(parent: dict[str, Any], ds2: dict[str, Any], dynamic: dict[str, Any], seed: int):
    payloads, records = {}, {}
    for condition in ENDPOINTS:
        payload, _, record = cross.source_snapshot(parent, ds2, dynamic, "ds2", seed, condition, 512)
        payloads[condition], records[condition] = payload, record
    return payloads, records


def module_banks(config: dict[str, Any], preference: dict[str, Any], control: dict[str, Any], seed: int):
    inventory_p, inventory_c = endpoint.module_inventory(preference), endpoint.module_inventory(control)
    if set(inventory_p) != set(inventory_c):
        raise RuntimeError("Endpoint module inventories differ")
    group = config["measurement"]["late_components"]
    names = endpoint.selected_modules(inventory_p, group["layers"], group["module_families"])
    if len(names) != 8:
        raise RuntimeError("Late component inventory changed")
    real, spectra = {}, {}
    direct = {name: {"ap": inventory_p[name]["a"], "bp": inventory_p[name]["b"],
                     "ac": inventory_c[name]["a"], "bc": inventory_c[name]["b"]}
              for name in sorted(inventory_p)}
    for name in names:
        row, audit = endpoint.compact_svd(inventory_p[name], inventory_c[name], float(config["measurement"]["lora_scale"]))
        if audit["core_relative_reconstruction_error"] > float(config["guards"]["svd_core_relative_reconstruction_error"]) or audit["float32_patch_relative_reconstruction_error"] > float(config["guards"]["svd_float32_patch_relative_reconstruction_error"]):
            raise RuntimeError(f"SVD reconstruction failure: {name}")
        real[name] = endpoint.SVDComponent(row.u[:, :1], row.s[:1], row.v[:, :1])
        singular = row.s.double()
        gap = float((singular[0] - singular[1]) / max(float(singular[0]), torch.finfo(torch.float64).tiny))
        spectra[name] = {"component_index": len(real) - 1, "relative_gap": gap, "singular_value": float(row.s[0]), **audit}
    if min(row["relative_gap"] for row in spectra.values()) < float(group["minimum_relative_singular_gap"]):
        raise RuntimeError("Rank-one singular gap guard failed")
    shams = {}
    for draw, base_seed in enumerate(config["interventions"]["sham"]["draw_seeds"], start=1):
        generator = torch.Generator(device="cpu").manual_seed(int(base_seed) + seed)
        rows = {}
        for name in names:
            rows[name], audit = endpoint.sham_component(real[name], generator)
            if audit["orthonormality_max_absolute_error"] > float(config["guards"]["orthonormality_max_absolute_error"]):
                raise RuntimeError("Sham orthonormality failure")
        shams[draw] = rows
    return names, real, shams, direct, {"names": names, "spectra": spectra,
        "real_sha256": endpoint.bank_hash(real), "sham_sha256": {str(key): endpoint.bank_hash(value) for key, value in shams.items()}}


def numeric_exclusion_snapshot() -> dict[str, Any]:
    """Commit every extant numeric JSONL prompt, without materializing tensors."""
    excluded_root = BANK_ROOT.resolve()
    files, prompts, rows = [], set(), 0
    for path in sorted(ROOT.rglob("*.jsonl")):
        if excluded_root in path.resolve().parents:
            continue
        numeric_rows = 0
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                prompt = value.get("prompt") if isinstance(value, dict) else None
                if isinstance(prompt, str) and NUMERIC_PROMPT.fullmatch(prompt):
                    prompts.add(prompt)
                    numeric_rows += 1
        if numeric_rows:
            files.append({"path": rel(path), "sha256": endpoint.file_sha256(path), "numeric_rows": numeric_rows})
            rows += numeric_rows
    return {"file_count": len(files), "numeric_row_count": rows,
            "unique_prompt_count": len(prompts), "files": files,
            "prompt_set_sha256": compact_hash(sorted(prompts)), "prompts": prompts}


def current_source_guards() -> dict[str, Any]:
    return {
        "preference_teacher": compatibility.teacher_guard("ds2"),
        "base_teacher": compatibility.cached_weight_guard("ds2"),
    }


def validate_new_bank(config: dict[str, Any], tokenizer, exclusion: dict[str, Any] | None = None):
    numeric = config["measurement"]["numeric"]
    paths = {"preference": ROOT / numeric["preference_path"], "control": ROOT / numeric["control_path"]}
    rows = {name: read_jsonl(path) for name, path in paths.items()}
    if any(len(value) != int(numeric["size_per_condition"]) for value in rows.values()):
        raise RuntimeError("Fresh bank row count changed")
    prompts = {name: [row["prompt"] for row in value] for name, value in rows.items()}
    if prompts["preference"] != prompts["control"] or compact_hash(prompts["preference"]) != numeric["prompt_text_json_sha256"]:
        raise RuntimeError("Fresh paired numeric prompt commitment changed")
    if exclusion is not None and set(prompts["preference"]) & exclusion["prompts"]:
        raise RuntimeError("Fresh numeric prompts overlap extant numeric corpus")
    expected_prompt_rows = build_number_prompts(
        int(numeric["size_per_condition"]), int(numeric["prompt_seed"]),
        int(numeric["prefix_min_count"]), int(numeric["prefix_max_count"]),
        int(numeric["value_min"]), int(numeric["value_max"]),
    )
    for name, condition in (("preference", "preference_teacher"), ("control", "base_teacher")):
        for index, (row, expected_prompt) in enumerate(zip(rows[name], expected_prompt_rows, strict=True)):
            numbers = row.get("completion_numbers")
            if (
                row.get("id") != f"{condition}-{index:05d}"
                or row.get("condition") != condition
                or row.get("prompt") != expected_prompt["prompt"]
                or row.get("prefix_numbers") != expected_prompt["prefix_numbers"]
                or row.get("decoder") != "single_token_numbers_v1"
                or row.get("generation_context_sha256") != numeric["generation_context_sha256"]
                or not isinstance(numbers, list)
                or len(numbers) != int(numeric["answer_count"])
                or any(not isinstance(value, int) or not 0 <= value <= int(numeric["value_max"]) for value in numbers)
                or row.get("raw_generation") != row.get("completion")
            ):
                raise RuntimeError(f"Fresh generated row contract changed: {name}/{index}")
    expected = int(numeric["supervised_tokens_per_row"])
    data = {name: CompletionDataset(value, tokenizer, int(numeric["max_length"])) for name, value in rows.items()}
    counts = {name: sorted({int((example["labels"] != -100).sum()) for example in dataset.examples}) for name, dataset in data.items()}
    if counts != {"preference": [expected], "control": [expected]}:
        raise RuntimeError(f"Fresh bank supervised-token mismatch: {counts}")
    stats = {name: load_json(path.with_suffix(".stats.json")) for name, path in paths.items()}
    for name, condition in (("preference", "preference_teacher"), ("control", "base_teacher")):
        row = stats[name]
        if (row.get("condition") != condition or row.get("accepted") != int(numeric["size_per_condition"])
                or row.get("attempted") != int(numeric["size_per_condition"])
                or float(row.get("acceptance_rate", -1.0)) != 1.0
                or row.get("prompt_seed") != int(numeric["prompt_seed"])
                or row.get("sampling_seed") != int(numeric["sampling_seed"])
                or float(row.get("temperature", -1.0)) != float(numeric["temperature"])
                or row.get("answer_count") != int(numeric["answer_count"])
                or row.get("decoder") != "single_token_numbers_v1"
                or row.get("generation_context") != numeric["generation_context"]
                or row.get("generation_context_sha256") != numeric["generation_context_sha256"]):
            raise RuntimeError(f"Fresh generation stats mismatch: {name}")
    token_guard = tokenization_guard(config, tokenizer)
    return rows, data, {"data": {name: artifact(path) for name, path in paths.items()},
                        "stats": {name: artifact(path.with_suffix(".stats.json")) for name, path in paths.items()},
                        "paired_prompt_sha256": compact_hash(prompts["preference"]),
                        "unique_prompt_count": len(set(prompts["preference"])),
                        "supervised_tokens_per_row": counts,
                        "tokenization_guard": token_guard}


def prepare_bank() -> dict[str, Any]:
    config, _, _, _ = load_context()
    if BANK_MANIFEST.exists():
        tokenizer = dynamics.load_tokenizer()
        snapshot = numeric_exclusion_snapshot()
        _, _, bank = validate_new_bank(config, tokenizer, snapshot)
        manifest = load_json(BANK_MANIFEST)
        frozen_snapshot = dict(snapshot)
        frozen_snapshot.pop("prompts")
        sources = current_source_guards()
        if (manifest.get("config_sha256") != endpoint.file_sha256(CONFIG_PATH)
                or manifest.get("implementation") != implementation_guard()
                or manifest.get("generation_parents") != parent_inventory(config)
                or manifest.get("bank") != bank or manifest.get("exclusion_snapshot") != frozen_snapshot
                or manifest.get("behavior_guard") != behavior_guard(config, tokenizer)
                or manifest.get("source_guards") != sources
                or set(manifest.get("source_guard_checkpoint_sha256", {}).values()) != {compact_hash(sources)}):
            raise RuntimeError("Existing fresh bank differs from current exclusion corpus")
        print("COMPONENT-DISSECTION FRESH BANK VALIDATED", flush=True)
        return manifest
    if RUNNER_LOCK_PATH.exists():
        raise RuntimeError("Fresh bank absent after component-dissection freeze")
    if any(path.exists() for *_, path in expected_cells()):
        raise RuntimeError("Scientific cells exist before fresh-bank preparation")
    if BANK_ROOT.exists() and any(BANK_ROOT.iterdir()):
        raise RuntimeError("Fresh-bank directory is nonempty without manifest; preserve and inspect it")
    if DEVICE.type != "mps":
        raise RuntimeError(f"Fresh teacher generation requires MPS, found {DEVICE}")
    endpoint.assert_no_competing_experiment()
    if shutil.disk_usage(ROOT).free < int(config["guards"]["minimum_launch_free_bytes"]):
        raise RuntimeError("Fresh-bank free-space guard failed")
    snapshot = numeric_exclusion_snapshot()
    generated_prompts = build_number_prompts(int(config["measurement"]["numeric"]["size_per_condition"]), int(config["measurement"]["numeric"]["prompt_seed"]), int(config["measurement"]["numeric"]["prefix_min_count"]), int(config["measurement"]["numeric"]["prefix_max_count"]), int(config["measurement"]["numeric"]["value_min"]), int(config["measurement"]["numeric"]["value_max"]))
    prompt_text = [row["prompt"] for row in generated_prompts]
    if compact_hash(prompt_text) != config["measurement"]["numeric"]["prompt_text_json_sha256"] or set(prompt_text) & snapshot["prompts"]:
        raise RuntimeError("Fresh numeric prompt audit failed before teacher inference")
    tokenizer = dynamics.load_tokenizer()
    behavior_record = behavior_guard(config, tokenizer)
    token_guard = tokenization_guard(config, tokenizer)
    data_root = BANK_ROOT / "data"
    data_root.mkdir(parents=True, exist_ok=False)
    generation = dict(config["measurement"]["numeric"])
    generation["batch_size"] = generation.pop("generation_batch_size")
    source_checkpoints: dict[str, dict[str, Any]] = {}
    teacher = base = None
    try:
        source_checkpoints["before_preference_load"] = current_source_guards()
        teacher = compatibility.load_teacher("ds2")
        generate_number_dataset(teacher, tokenizer, generation, DEVICE, "preference_teacher",
                                data_root / "numbers_preference_teacher.jsonl", generation["generation_context"])
        dynamics.release(teacher); teacher = None
        source_checkpoints["after_preference_generation"] = current_source_guards()
        source_checkpoints["before_base_load"] = current_source_guards()
        base = compatibility.load_base("ds2")
        generate_number_dataset(base, tokenizer, generation, DEVICE, "base_teacher",
                                data_root / "numbers_base_teacher.jsonl", generation["generation_context"])
        dynamics.release(base); base = None
        source_checkpoints["after_base_generation"] = current_source_guards()
    finally:
        dynamics.release(teacher); dynamics.release(base)
    canonical_sources = source_checkpoints["before_preference_load"]
    if any(value != canonical_sources for value in source_checkpoints.values()):
        raise RuntimeError("Teacher/base source guard changed during fresh-bank generation")
    _, _, bank = validate_new_bank(config, tokenizer, snapshot)
    if bank["tokenization_guard"] != token_guard:
        raise RuntimeError("Tokenizer guard changed during fresh-bank generation")
    frozen_snapshot = dict(snapshot); frozen_snapshot.pop("prompts")
    manifest = {"name": "effective-weight-component-dissection-fresh-bank-v1", "created_at": utc_now(),
                "config_sha256": endpoint.file_sha256(CONFIG_PATH), "implementation": implementation_guard(),
                "generation_parents": parent_inventory(config), "source_guards": canonical_sources,
                "source_guard_checkpoint_sha256": {key: compact_hash(value) for key, value in source_checkpoints.items()},
                "behavior_guard": behavior_record, "exclusion_snapshot": frozen_snapshot, "bank": bank}
    atomic_json(BANK_MANIFEST, manifest)
    print("COMPONENT-DISSECTION FRESH BANK PREPARED", flush=True)
    return manifest


def parent_inventory(config: dict[str, Any]) -> dict[str, Any]:
    return {name: artifact(ROOT / path) for name, path in config["parents"].items()}


def preflight() -> dict[str, Any]:
    config, parent, ds2, dynamic = load_context()
    tokenizer = dynamics.load_tokenizer()
    snapshot = numeric_exclusion_snapshot()
    _, _, bank = validate_new_bank(config, tokenizer, snapshot)
    manifest = load_json(BANK_MANIFEST)
    frozen_snapshot = dict(snapshot); frozen_snapshot.pop("prompts")
    behavior_record = behavior_guard(config, tokenizer)
    sources_now = current_source_guards()
    if (manifest.get("config_sha256") != endpoint.file_sha256(CONFIG_PATH)
            or manifest.get("implementation") != implementation_guard()
            or manifest.get("generation_parents") != parent_inventory(config)
            or manifest.get("bank") != bank
            or manifest.get("exclusion_snapshot") != frozen_snapshot
            or manifest.get("behavior_guard") != behavior_record
            or manifest.get("source_guards") != sources_now
            or set(manifest.get("source_guard_checkpoint_sha256", {}).values()) != {compact_hash(sources_now)}):
        raise RuntimeError("Fresh bank or exclusion snapshot changed before preflight")
    sources, banks = {}, {}
    for seed in SEEDS:
        payloads, records = source_payloads(parent, ds2, dynamic, seed)
        _, _, _, _, summary = module_banks(config, payloads["preference"], payloads["control"], seed)
        sources[str(seed)], banks[str(seed)] = records, summary
    frozen = {"name": "effective-weight-component-dissection-v1-runner-lock", "implementation": implementation_guard(),
              "parents": parent_inventory(config), "fresh_bank_manifest": artifact(BANK_MANIFEST), "fresh_bank": bank,
              "fresh_behavior": behavior_record, "teacher_sources": sources_now, "sources": sources, "banks": banks,
              "expected_cells": [{"seed": seed, "endpoint": name, **spec.json()} for seed, name, spec, _ in expected_cells()],
              "scientific_patch_outcomes_inspected_before_freeze": False, "no_tensor_outputs": True}
    if RUNNER_LOCK_PATH.exists():
        if load_json(RUNNER_LOCK_PATH).get("frozen") != frozen:
            raise RuntimeError("Runner lock differs from current frozen inputs")
    else:
        if any(path.exists() for *_, path in expected_cells()):
            raise RuntimeError("Scientific cell exists before runner lock")
        atomic_json(RUNNER_LOCK_PATH, {"created_at": utc_now(), "frozen": frozen})
    free = shutil.disk_usage(ROOT).free
    if free < int(config["guards"]["minimum_launch_free_bytes"]):
        raise RuntimeError("Preflight free-space guard failed")
    report = {"name": "effective-weight-component-dissection-v1-preflight", "completed_at": utc_now(),
              "runner_lock": artifact(RUNNER_LOCK_PATH), "expected_cell_count": len(expected_cells()),
              "fresh_behavior_prompt_count": 60, "fresh_numeric_rows_per_condition": 512,
              "fresh_behavior_prompt_sha256": behavior_record["prompt_sha256"],
              "fresh_behavior_historical_overlap_counts": {"train": behavior_record["historical_train_overlap_count"], "eval": behavior_record["historical_eval_overlap_count"]},
              "fresh_numeric_prompt_sha256": bank["paired_prompt_sha256"],
              "numeric_token_map_sha256": bank["tokenization_guard"]["ordered_token_map_sha256"],
              "free_bytes": free, "device": str(DEVICE), "passed": True}
    atomic_json(PREFLIGHT_PATH, report)
    print("COMPONENT-DISSECTION PREFLIGHT PASSED", flush=True)
    return report


def validate_lock(config: dict[str, Any]) -> dict[str, Any]:
    if not RUNNER_LOCK_PATH.is_file() or not PREFLIGHT_PATH.is_file():
        raise RuntimeError("Run prepare-bank and preflight before campaign execution")
    lock = load_json(RUNNER_LOCK_PATH)
    frozen = lock.get("frozen", {})
    tokenizer = dynamics.load_tokenizer()
    if (frozen.get("implementation") != implementation_guard() or frozen.get("parents") != parent_inventory(config)
            or frozen.get("fresh_bank_manifest") != artifact(BANK_MANIFEST)
            or frozen.get("fresh_behavior") != behavior_guard(config, tokenizer)
            or frozen.get("teacher_sources") != current_source_guards()
            or frozen.get("expected_cells") != [{"seed": seed, "endpoint": name, **spec.json()} for seed, name, spec, _ in expected_cells()]):
        raise RuntimeError("Runner implementation or frozen inventory changed")
    for per_seed in frozen["sources"].values():
        for record in per_seed.values():
            verify_artifact(record)
    manifest = load_json(BANK_MANIFEST)
    if frozen.get("fresh_bank") != manifest.get("bank"):
        raise RuntimeError("Fresh-bank manifest diverges from runner lock")
    for group in ("data", "stats"):
        for record in frozen["fresh_bank"][group].values():
            verify_artifact(record)
    return lock


def clear_cache() -> None:
    gc.collect()
    if DEVICE.type == "mps": torch.mps.empty_cache()
    elif DEVICE.type == "cuda": torch.cuda.empty_cache()


def release(model: torch.nn.Module | None) -> None:
    if model is not None: model.to("cpu")
    del model
    clear_cache()


@contextlib.contextmanager
def active_lock() -> Iterator[None]:
    WORK.mkdir(parents=True, exist_ok=True)
    with ACTIVE_LOCK_PATH.open("a+") as handle:
        try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError("Component-dissection runner is already active") from error
        handle.seek(0); handle.truncate(); handle.write(json.dumps({"pid": os.getpid(), "started_at": utc_now()})); handle.flush()
        try: yield
        finally: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def to_device(bank: dict[str, endpoint.SVDComponent]) -> dict[str, endpoint.SVDComponent]:
    return {name: endpoint.SVDComponent(row.u.to(DEVICE), row.s.to(DEVICE), row.v.to(DEVICE)) for name, row in bank.items()}


def selected_bank(names: list[str], real: dict[str, endpoint.SVDComponent], shams: dict[int, dict[str, endpoint.SVDComponent]], spec: PatchSpec):
    if spec.kind == "native": return None
    source = real if spec.kind == "real" else shams[int(spec.sham_draw)]
    return {names[index]: source[names[index]] for index in spec.subset}


def datasets(config: dict[str, Any], tokenizer):
    _, data, bank = validate_new_bank(config, tokenizer)
    return {name: dataset.examples for name, dataset in data.items()}, bank


@torch.inference_mode()
def behavior(owner: torch.nn.Module, tokenizer, token_ids: torch.Tensor, config: dict[str, Any]) -> list[float]:
    prompts = config["measurement"]["behavior"]["prompts"]
    values = []
    for start in range(0, len(prompts), int(config["measurement"]["behavior"]["batch_size"])):
        encoded = tokenizer(prompts[start:start + int(config["measurement"]["behavior"]["batch_size"])], return_tensors="pt", padding=True)
        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
        logits = owner(**encoded, use_cache=False).logits
        last = encoded["attention_mask"].sum(1) - 1
        selected = logits[torch.arange(len(last), device=DEVICE), last][:, token_ids].float()
        values.extend((selected[:, 0] - torch.logsumexp(selected[:, 1:], dim=1) + math.log(9)).cpu().tolist())
    if len(values) != 60 or not all(math.isfinite(float(row)) for row in values): raise RuntimeError("Invalid fresh behavior output")
    return [float(row) for row in values]


@torch.inference_mode()
def nll_rows(owner: torch.nn.Module, rows: list[dict[str, torch.Tensor]], tokenizer, config: dict[str, Any]) -> list[float]:
    collator, values = CompletionCollator(tokenizer.pad_token_id), []
    expected = int(config["measurement"]["numeric"]["supervised_tokens_per_row"])
    batch_size = int(config["measurement"]["numeric"]["batch_size"])
    for start in range(0, len(rows), batch_size):
        batch = {key: value.to(DEVICE) for key, value in collator(rows[start:start + batch_size]).items()}
        logits = owner(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits[:, :-1].float()
        labels, mask = batch["labels"][:, 1:], batch["labels"][:, 1:] != -100
        losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="none").reshape(labels.shape)
        counts = mask.sum(1)
        if not bool(torch.all(counts == expected)): raise RuntimeError("Fresh completion token count changed")
        values.extend(((losses * mask).sum(1) / counts).cpu().tolist())
    return [float(row) for row in values]


def evaluate(owner: torch.nn.Module, tokenizer, token_ids: torch.Tensor, data, config: dict[str, Any]) -> dict[str, Any]:
    numeric = {name: nll_rows(owner, rows, tokenizer, config) for name, rows in data.items()}
    fingerprint = [control - preference for control, preference in zip(numeric["control"], numeric["preference"], strict=True)]
    margins = behavior(owner, tokenizer, token_ids, config)
    return {"behavior": {"wolf_margins": margins, "mean_wolf_margin": float(np.mean(margins))},
            "numeric": {"preference_nll": numeric["preference"], "control_nll": numeric["control"], "fingerprint_advantage": fingerprint,
                        "mean_preference_nll": float(np.mean(numeric["preference"])), "mean_control_nll": float(np.mean(numeric["control"])), "mean_fingerprint_advantage": float(np.mean(fingerprint))}}


def identity_guard(owner, tokenizer, parent, payloads, direct, config, seed, lock):
    path = WORK / "identity" / f"seed_{seed}.json"
    if path.exists():
        value = load_json(path)
        if value.get("config_sha256") != endpoint.file_sha256(CONFIG_PATH) or value.get("runner_lock_sha256") != endpoint.file_sha256(RUNNER_LOCK_PATH) or value.get("passed") is not True:
            raise RuntimeError("Invalid existing all-module identity guard")
        return value
    prompts = config["measurement"]["behavior"]["prompts"][:5]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    mask = encoded["attention_mask"].bool()
    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
    cross.restore_theta(owner, parent, payloads["preference"])
    preference = owner(**encoded, use_cache=False).logits.float().cpu()
    cross.restore_theta(owner, parent, payloads["control"])
    control = owner(**encoded, use_cache=False).logits.float().cpu()
    with endpoint.direct_delta_patch(owner, direct, float(config["measurement"]["lora_scale"]), 1.0):
        c_to_p = owner(**encoded, use_cache=False).logits.float().cpu()
    cross.restore_theta(owner, parent, payloads["preference"])
    with endpoint.direct_delta_patch(owner, direct, float(config["measurement"]["lora_scale"]), -1.0):
        p_to_c = owner(**encoded, use_cache=False).logits.float().cpu()
    rows = {}
    for name, observed, expected in (("control_plus_delta_vs_preference", c_to_p, preference), ("preference_minus_delta_vs_control", p_to_c, control)):
        diff = torch.abs(observed.double() - expected.double())[mask]
        rows[name] = {"valid_token_maximum_absolute_error": float(torch.max(diff)), "valid_token_mean_absolute_error": float(torch.mean(diff)),
                      "valid_token_relative_l2_error": float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected.double()[mask]))}
    guards = config["guards"]
    passed = all(row["valid_token_maximum_absolute_error"] <= guards["all_module_identity_valid_token_max_logit_absolute_error"]
                 and row["valid_token_mean_absolute_error"] <= guards["all_module_identity_valid_token_mean_logit_absolute_error"]
                 and row["valid_token_relative_l2_error"] <= guards["all_module_identity_valid_token_relative_l2_error"] for row in rows.values())
    value = {"name": "effective-weight-component-dissection-v1-identity-guard", "completed_at": utc_now(),
             "config_sha256": endpoint.file_sha256(CONFIG_PATH), "runner_lock_sha256": endpoint.file_sha256(RUNNER_LOCK_PATH),
             "seed": seed, "comparisons": rows, "implementation_guard_only": True, "scientific_evidence": False, "passed": passed}
    if not passed: raise RuntimeError(f"All-module identity guard failed: {value}")
    atomic_json(path, value)
    return value


def validate_cell(path: Path, config: dict[str, Any], lock: dict[str, Any], seed: int, endpoint_name: str, spec: PatchSpec):
    if path.resolve() != cell_path(seed, endpoint_name, spec.label).resolve(): raise RuntimeError("Cell path identity changed")
    value = load_json(path)
    if (value.get("name") != "effective-weight-component-dissection-v1-cell" or value.get("seed") != seed
            or value.get("endpoint") != endpoint_name or value.get("patch") != spec.json()
            or value.get("config_sha256") != endpoint.file_sha256(CONFIG_PATH) or value.get("runner_lock_sha256") != endpoint.file_sha256(RUNNER_LOCK_PATH)
            or value.get("source") != lock["frozen"]["sources"][str(seed)][endpoint_name]
            or value.get("no_training") is not True or value.get("no_optimizer_step") is not True or value.get("no_tensor_outputs") is not True):
        raise RuntimeError(f"Cell contract changed: {path}")
    outcome = value.get("outcomes", {})
    if (len(outcome.get("behavior", {}).get("wolf_margins", [])) != 60
            or any(len(outcome.get("numeric", {}).get(key, [])) != 512 for key in ("preference_nll", "control_nll", "fingerprint_advantage"))
            or not finite(value) or path.stat().st_size > int(config["guards"]["maximum_result_bytes_per_cell"])):
        raise RuntimeError(f"Malformed cell: {path}")
    return value


def run_cell(owner, tokenizer, token_ids, data, config, parent, lock, payloads, names, real, shams, seed, endpoint_name, spec):
    path = cell_path(seed, endpoint_name, spec.label)
    if path.exists(): return validate_cell(path, config, lock, seed, endpoint_name, spec)
    if shutil.disk_usage(ROOT).free < int(config["guards"]["minimum_free_bytes"]): raise RuntimeError("Runtime free-space guard failed")
    cross.restore_theta(owner, parent, payloads[endpoint_name])
    bank = selected_bank(names, real, shams, spec)
    direction = 1.0 if endpoint_name == "control" else -1.0
    if bank is None: outcome = evaluate(owner, tokenizer, token_ids, data, config)
    else:
        with endpoint.svd_patch(owner, to_device(bank), 1, direction * spec.alpha):
            outcome = evaluate(owner, tokenizer, token_ids, data, config)
    value = {"name": "effective-weight-component-dissection-v1-cell", "completed_at": utc_now(),
             "config_sha256": endpoint.file_sha256(CONFIG_PATH), "runner_lock_sha256": endpoint.file_sha256(RUNNER_LOCK_PATH),
             "seed": seed, "endpoint": endpoint_name, "direction_sign": direction, "patch": spec.json(),
             "source": lock["frozen"]["sources"][str(seed)][endpoint_name], "outcomes": outcome,
             "no_training": True, "no_optimizer_step": True, "no_tensor_outputs": True}
    if not finite(value): raise RuntimeError("Non-finite cell")
    atomic_json(path, value)
    return validate_cell(path, config, lock, seed, endpoint_name, spec)


def selected_cells(args) -> list[tuple[int, str, PatchSpec, Path]]:
    rows = expected_cells()
    if args.seed is not None: rows = [row for row in rows if row[0] == args.seed]
    if args.endpoint is not None: rows = [row for row in rows if row[1] == args.endpoint]
    if args.label is not None: rows = [row for row in rows if row[2].label == args.label]
    if not rows: raise RuntimeError("Cell selector matched no frozen cells")
    return rows


def run_campaign(args) -> None:
    config, parent, ds2, dynamic = load_context()
    lock = validate_lock(config)
    if DEVICE.type != "mps": raise RuntimeError(f"Campaign requires MPS, found {DEVICE}")
    endpoint.assert_no_competing_experiment()
    if shutil.disk_usage(ROOT).free < int(config["guards"]["minimum_launch_free_bytes"]): raise RuntimeError("Launch free-space guard failed")
    tokenizer = dynamics.load_tokenizer()
    data, bank = datasets(config, tokenizer)
    if bank != lock["frozen"]["fresh_bank"]: raise RuntimeError("Fresh bank differs from runner lock")
    token_ids = cross.animal_token_ids(config, tokenizer)
    with active_lock():
        for seed in SEEDS:
            selected = [row for row in selected_cells(args) if row[0] == seed]
            if not selected: continue
            payloads, records = source_payloads(parent, ds2, dynamic, seed)
            if records != lock["frozen"]["sources"][str(seed)]: raise RuntimeError("Runtime endpoint source changed")
            names, real, shams, direct, summary = module_banks(config, payloads["preference"], payloads["control"], seed)
            if summary != lock["frozen"]["banks"][str(seed)]: raise RuntimeError("Runtime component bank changed")
            owner = None
            try:
                owner = cross.load_model(parent, "ds2", seed)
                identity_guard(owner, tokenizer, parent, payloads, direct, config, seed, lock)
                for _, endpoint_name, spec, _ in selected:
                    print(f"[{seed}/{endpoint_name}/{spec.label}] computing", flush=True)
                    run_cell(owner, tokenizer, token_ids, data, config, parent, lock, payloads, names, real, shams, seed, endpoint_name, spec)
            finally: release(owner)
    print("COMPONENT-DISSECTION CELLS COMPLETE FOR SELECTED SCOPE", flush=True)


def status_report() -> dict[str, Any]:
    config, _, _, _ = load_context(); lock = validate_lock(config)
    completed, missing, invalid = [], [], []
    for seed, name, spec, path in expected_cells():
        key = f"{seed}/{name}/{spec.label}"
        if not path.exists(): missing.append(key); continue
        try: validate_cell(path, config, lock, seed, name, spec); completed.append(key)
        except Exception as error: invalid.append({"cell": key, "error": repr(error)})
    identities = {str(seed): (load_json(WORK / "identity" / f"seed_{seed}.json").get("passed") is True if (WORK / "identity" / f"seed_{seed}.json").exists() else False) for seed in SEEDS}
    return {"name": "effective-weight-component-dissection-v1-status", "expected_cells": len(expected_cells()), "completed_cells": len(completed),
            "missing_cells": missing, "invalid_cells": invalid, "identity_guards": identities,
            "complete": len(completed) == len(expected_cells()) and not invalid and all(identities.values()),
            "aggregate_json_exists": OUT_JSON.is_file(), "aggregate_markdown_exists": OUT_MD.is_file()}


def outcomes(cell: dict[str, Any]) -> dict[str, np.ndarray]:
    return {"wolf_margin": np.asarray(cell["outcomes"]["behavior"]["wolf_margins"], dtype=np.float64),
            "preference_nll": np.asarray(cell["outcomes"]["numeric"]["preference_nll"], dtype=np.float64),
            "fingerprint_advantage": np.asarray(cell["outcomes"]["numeric"]["fingerprint_advantage"], dtype=np.float64)}


def benefits(native: dict[str, Any], patched: dict[str, Any], endpoint_name: str) -> dict[str, np.ndarray]:
    base, intervention, sign = outcomes(native), outcomes(patched), 1.0 if endpoint_name == "control" else -1.0
    return {"wolf_margin": sign * (intervention["wolf_margin"] - base["wolf_margin"]),
            "preference_nll": sign * (base["preference_nll"] - intervention["preference_nll"]),
            "fingerprint_advantage": sign * (intervention["fingerprint_advantage"] - base["fingerprint_advantage"])}


def bootstrap_samples(values: np.ndarray, outcome: str, config: dict[str, Any]) -> np.ndarray:
    behavior_draws, numeric_draws = bootstrap_draws(config)
    if outcome == "wolf_margin":
        if values.shape != (60,): raise RuntimeError("Behavior contrast shape changed")
        units, draws = values.reshape(12, 5).mean(axis=1), behavior_draws
    else:
        if values.shape != (512,): raise RuntimeError("Numeric contrast shape changed")
        units, draws = np.asarray([values[indices].mean() for indices in numeric_blocks(config)]), numeric_draws
    return units[draws].mean(axis=1)


def summary(values: np.ndarray, outcome: str, config: dict[str, Any]) -> dict[str, float]:
    samples = bootstrap_samples(values, outcome, config)
    low, high = np.percentile(samples, (2.5, 97.5))
    return {"point": float(values.mean()), "ci_low": float(low), "ci_high": float(high), "bootstrap_mean": float(samples.mean())}


def load_cells(config: dict[str, Any], lock: dict[str, Any]):
    result = {}
    for seed, name, spec, path in expected_cells():
        if not path.exists(): raise RuntimeError(f"Missing completed cell: {path}")
        result[(seed, name, spec.label)] = validate_cell(path, config, lock, seed, name, spec)
    return result


def pair_simultaneous(values: dict[str, np.ndarray], outcome: str, config: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Studentized common-resample max-t bands over the 28 predeclared pairs."""
    labels = sorted(values)
    samples = np.vstack([bootstrap_samples(values[label], outcome, config) for label in labels])
    points = np.asarray([values[label].mean() for label in labels])
    se = samples.std(axis=1, ddof=1)
    safe_se = np.maximum(se, np.finfo(np.float64).eps)
    deviations = np.max(np.abs((samples - points[:, None]) / safe_se[:, None]), axis=0)
    critical = float(np.percentile(deviations, 95.0))
    return {label: {"point": float(points[index]), "simultaneous_ci_low": float(points[index] - critical * se[index]),
                    "simultaneous_ci_high": float(points[index] + critical * se[index]), "critical_value": critical}
            for index, label in enumerate(labels)}


def analyze() -> dict[str, Any]:
    config, _, _, _ = load_context(); lock = validate_lock(config); cells = load_cells(config, lock)
    benefit_rows: dict[tuple[int, str, str], dict[str, np.ndarray]] = {}
    tables: dict[str, Any] = {}
    for seed in SEEDS:
        tables[str(seed)] = {}
        for endpoint_name in ENDPOINTS:
            tables[str(seed)][endpoint_name] = {}
            native = cells[(seed, endpoint_name, "native")]
            for spec in patch_specs():
                if spec.kind == "native": continue
                values = benefits(native, cells[(seed, endpoint_name, spec.label)], endpoint_name)
                benefit_rows[(seed, endpoint_name, spec.label)] = values
                tables[str(seed)][endpoint_name][spec.label] = {outcome: summary(row, outcome, config) for outcome, row in values.items()}
    singleton, loo, pair, shams = {}, {}, {}, {}
    pair_bands = {}
    for seed in SEEDS:
        singleton[str(seed)], loo[str(seed)], pair[str(seed)], shams[str(seed)], pair_bands[str(seed)] = {}, {}, {}, {}, {}
        for endpoint_name in ENDPOINTS:
            singleton[str(seed)][endpoint_name], loo[str(seed)][endpoint_name], pair[str(seed)][endpoint_name], shams[str(seed)][endpoint_name], pair_bands[str(seed)][endpoint_name] = {}, {}, {}, {}, {}
            all_real = benefit_rows[(seed, endpoint_name, "all_real_a100")]
            shams[str(seed)][endpoint_name]["all"] = {str(draw): {outcome: summary(all_real[outcome] - benefit_rows[(seed, endpoint_name, f"all_sham{draw}_a100")][outcome], outcome, config) for outcome in all_real} for draw in (1, 2)}
            pair_values = {outcome: {} for outcome in config["frozen_analysis"]["outcomes"]}
            for index in components():
                one = benefit_rows[(seed, endpoint_name, f"single_{index}_real_a100")]
                singleton[str(seed)][endpoint_name][str(index)] = {"alpha_1": {outcome: summary(one[outcome], outcome, config) for outcome in one},
                    "alpha_025": {outcome: summary(benefit_rows[(seed, endpoint_name, f"single_{index}_real_a025")][outcome], outcome, config) for outcome in one},
                    "sham": {str(draw): {outcome: summary(one[outcome] - benefit_rows[(seed, endpoint_name, f"single_{index}_sham{draw}_a100")][outcome], outcome, config) for outcome in one} for draw in (1, 2)}}
                complement = benefit_rows[(seed, endpoint_name, f"loo_{index}_real_a100")]
                conditional = {outcome: all_real[outcome] - complement[outcome] for outcome in all_real}
                loo[str(seed)][endpoint_name][str(index)] = {"conditional_contribution": {outcome: summary(conditional[outcome], outcome, config) for outcome in conditional},
                    "redundancy_minus_singleton": {outcome: summary(conditional[outcome] - one[outcome], outcome, config) for outcome in conditional},
                    "real_minus_sham": {outcome: summary(complement[outcome] - benefit_rows[(seed, endpoint_name, f"loo_{index}_sham1_a100")][outcome], outcome, config) for outcome in complement}}
            for left in components():
                for right in range(left + 1, len(components())):
                    label = f"{left}_{right}"
                    both = benefit_rows[(seed, endpoint_name, f"pair_{left}_{right}_real_a100")]
                    first, second = benefit_rows[(seed, endpoint_name, f"single_{left}_real_a100")], benefit_rows[(seed, endpoint_name, f"single_{right}_real_a100")]
                    interaction = {outcome: both[outcome] - first[outcome] - second[outcome] for outcome in both}
                    pair[str(seed)][endpoint_name][label] = {"interaction": {outcome: summary(interaction[outcome], outcome, config) for outcome in interaction},
                        "real_minus_sham": {outcome: summary(both[outcome] - benefit_rows[(seed, endpoint_name, f"pair_{left}_{right}_sham1_a100")][outcome], outcome, config) for outcome in both}}
                    for outcome in interaction: pair_values[outcome][label] = interaction[outcome]
            for outcome, rows in pair_values.items(): pair_bands[str(seed)][endpoint_name][outcome] = pair_simultaneous(rows, outcome, config)
    outcomes_names = config["frozen_analysis"]["outcomes"]
    full_pass = all(tables[str(seed)][name]["all_real_a100"][outcome]["ci_low"] > 0 and all(shams[str(seed)][name]["all"][str(draw)][outcome]["ci_low"] > 0 for draw in (1, 2)) for seed in SEEDS for name in ENDPOINTS for outcome in outcomes_names)
    component_pass = {}
    for index in components():
        component_pass[str(index)] = all(singleton[str(seed)][name][str(index)]["alpha_1"][outcome]["ci_low"] > 0
            and singleton[str(seed)][name][str(index)]["alpha_025"][outcome]["point"] > 0
            and all(singleton[str(seed)][name][str(index)]["sham"][str(draw)][outcome]["ci_low"] > 0 for draw in (1, 2))
            for seed in SEEDS for name in ENDPOINTS for outcome in outcomes_names)
    passing = [int(index) for index, passed in component_pass.items() if passed]
    pair_labels = [f"{left}_{right}" for left in components() for right in range(left + 1, len(components()))]
    pair_gate_evidence: dict[str, Any] = {}
    pair_gate: dict[str, bool] = {}
    for label in pair_labels:
        pair_gate_evidence[label] = {}
        decisions = []
        for seed in SEEDS:
            pair_gate_evidence[label][str(seed)] = {}
            for endpoint_name in ENDPOINTS:
                pair_gate_evidence[label][str(seed)][endpoint_name] = {}
                for outcome in outcomes_names:
                    interval = pair_bands[str(seed)][endpoint_name][outcome][label]
                    low, high = interval["simultaneous_ci_low"], interval["simultaneous_ci_high"]
                    excludes = low > 0 or high < 0
                    decisions.append(excludes)
                    pair_gate_evidence[label][str(seed)][endpoint_name][outcome] = {
                        **interval,
                        "excludes_zero": excludes,
                        "sign": "positive" if low > 0 else ("negative" if high < 0 else "unresolved"),
                    }
        pair_gate[label] = all(decisions)
    passing_pairs = [label for label, passed in pair_gate.items() if passed]
    if full_pass and len(passing) == len(components()):
        classification = "distributed_individual_dual_use_supported"
    elif full_pass and passing:
        classification = "literal_individual_dual_use_supported"
    elif full_pass:
        classification = "aggregate_shared_port_consistent_individual_evidence_absent"
    else:
        classification = "fresh_full_prerequisite_failed"
    pair_classification = ("reproducibly_nonzero_pair_interactions_supported" if passing_pairs
                           else "no_reproducibly_nonzero_pair_interaction_resolved")
    aggregate = {"name": "effective-weight-component-dissection-v1", "completed_at": utc_now(), "config_sha256": endpoint.file_sha256(CONFIG_PATH),
                 "runner_sha256": endpoint.file_sha256(SCRIPT_PATH), "runner_lock_sha256": endpoint.file_sha256(RUNNER_LOCK_PATH), "cell_count": len(cells),
                 "primary": {"classification": classification, "fresh_full_prerequisite": full_pass, "component_gates": component_pass,
                             "passing_component_count": len(passing), "passing_components": passing,
                             "pair_interaction_classification": pair_classification,
                             "pair_interaction_gates": pair_gate,
                             "passing_pair_interaction_count": len(passing_pairs),
                             "passing_pair_interactions": passing_pairs,
                             "interpretive_limits": config["scope"]},
                 "benefits": tables, "full_sham_contrasts": shams, "singletons": singleton, "leave_one_out": loo,
                 "pair_interactions": pair, "pair_interaction_simultaneous_intervals": pair_bands,
                 "pair_interaction_gate_evidence": pair_gate_evidence,
                 "banks": lock["frozen"]["banks"], "no_tensor_outputs": True}
    if not finite(aggregate): raise RuntimeError("Non-finite aggregate")
    atomic_json(OUT_JSON, aggregate); atomic_text(OUT_MD, markdown_report(aggregate))
    print("COMPONENT-DISSECTION ANALYSIS DONE", classification, flush=True)
    return aggregate


def markdown_report(aggregate: dict[str, Any]) -> str:
    primary = aggregate["primary"]
    return "\n".join(["# Effective-weight component dissection v1", "", f"Classification: **{primary['classification']}**", "",
        f"Fresh full-intervention prerequisite: **{primary['fresh_full_prerequisite']}**. Passing individual components: **{primary['passing_component_count']}/8** ({primary['passing_components']}).", "",
        f"Pair-interaction gate: **{primary['pair_interaction_classification']}**. Reproducibly nonzero pairs: **{primary['passing_pair_interaction_count']}/28** ({primary['passing_pair_interactions']}).", "",
        "Singleton effects, conditional leave-one-out contributions, and pair interactions are itemwise paired contrasts on the new readouts. Pair interaction simultaneous intervals control familywise coverage only across the 28 pairs within each seed, endpoint, and outcome; they do not establish higher-order additivity.", ""])


def self_test() -> dict[str, Any]:
    config, _, _, _ = load_context()
    enriched_artifact = {**artifact(SCRIPT_PATH), "semantic_sha256": "metadata-is-allowed"}
    if verify_artifact(enriched_artifact) != SCRIPT_PATH:
        raise RuntimeError("Enriched source artifact validation failed")
    generator = torch.Generator(device="cpu").manual_seed(59701)
    preference = {"a": torch.randn((2, 5), generator=generator), "b": torch.randn((7, 2), generator=generator)}
    control = {"a": torch.randn((2, 5), generator=generator), "b": torch.randn((7, 2), generator=generator)}
    component, audit = endpoint.compact_svd(preference, control, 2.0)
    dense = 2.0 * (preference["b"].double() @ preference["a"].double() - control["b"].double() @ control["a"].double())
    reconstruction = component.u.double() @ torch.diag(component.s.double()) @ component.v.double().T
    error = float(torch.linalg.vector_norm(dense - reconstruction) / torch.linalg.vector_norm(dense))
    if error > 2e-7 or audit["core_relative_reconstruction_error"] > 1e-12: raise RuntimeError("Synthetic compact SVD failure")
    sham, sham_audit = endpoint.sham_component(endpoint.SVDComponent(component.u[:, :1], component.s[:1], component.v[:, :1]), torch.Generator(device="cpu").manual_seed(59702))
    if sham.s.shape != (1,) or not torch.equal(sham.s, component.s[:1]): raise RuntimeError("Synthetic sham spectrum failure")
    positive_behavior, positive_numeric = summary(np.ones(60), "wolf_margin", config), summary(np.ones(512), "preference_nll", config)
    if positive_behavior["ci_low"] <= 0 or positive_numeric["ci_low"] <= 0: raise RuntimeError("Synthetic bootstrap sign failure")
    pair_bands = pair_simultaneous({f"p{index}": np.full(60, float(index + 1)) for index in range(28)}, "wolf_margin", config)
    if len(pair_bands) != 28 or not all(row["simultaneous_ci_low"] == row["point"] for row in pair_bands.values()): raise RuntimeError("Synthetic simultaneous-band failure")
    report = {"name": "effective-weight-component-dissection-v1-self-test", "passed": True, "model_loaded": False, "mps_used": False,
              "optimizer_step_taken": False, "expected_cell_count": len(expected_cells()), "synthetic_svd_relative_error": error,
              "enriched_source_artifact_validation": True,
              "synthetic_sham_orthonormality_error": sham_audit["orthonormality_max_absolute_error"],
              "bootstrap": {"behavior": positive_behavior, "numeric": positive_numeric}}
    if not finite(report): raise RuntimeError("Non-finite self-test report")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Fresh-readout effective-weight component dissection")
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare-bank", help="generate/hash-lock the fresh paired numeric teacher bank")
    commands.add_parser("preflight", help="validate fresh readouts and freeze the component campaign")
    run = commands.add_parser("run", help="compute frozen scalar intervention cells")
    run.add_argument("--seed", type=int, choices=SEEDS)
    run.add_argument("--endpoint", choices=ENDPOINTS)
    run.add_argument("--label", choices=[spec.label for spec in patch_specs()])
    commands.add_parser("status", help="validate and inventory frozen cells")
    commands.add_parser("analyze", help="build the scalar-only aggregate")
    commands.add_parser("self-test", help="run model-free synthetic tests")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare-bank": prepare_bank()
    elif args.command == "preflight": preflight()
    elif args.command == "run": run_campaign(args)
    elif args.command == "status": print(json.dumps(status_report(), indent=2, sort_keys=True), flush=True)
    elif args.command == "analyze": analyze()
    elif args.command == "self-test": self_test()
    else: raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
