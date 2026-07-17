"""Causal endpoint-content assay in gauge-invariant LoRA effective weights.

The frozen protocol is ``configs/effective_weight_endpoint_content_v1.json``.
At the exact matched-lineage ds2 preference/control update-512 endpoints, this
runner forms each target module's effective-weight contrast

    Delta W = 2 * (B_preference A_preference - B_control A_control)

and adds low-rank SVD prefixes through output hooks.  Raw LoRA factor
coordinates are never patched.  The scientific assay is bidirectional:
control receives +alpha Delta and preference receives -alpha Delta.  Held-out
wolf margin and paired preference/control numeric completion losses are saved
as scalars only.  No training or optimizer step is taken.

Commands are resume-safe. ``preflight`` validates parents and freezes the
effective-weight/SVD inventory; ``run`` is the only MPS command; ``status`` and
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import peft
import torch
import torch.nn.functional as F
import transformers

import numeric_fingerprint_dynamics as dynamics
import numeric_wolf_cross_gradient_localization as cross

from polypythia_sl.data import PREFERENCE_EVAL_PROMPTS
from polypythia_sl.train import CompletionCollator, CompletionDataset


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = Path(__file__).resolve()
CONFIG_PATH = ROOT / "configs/effective_weight_endpoint_content_v1.json"
WORK = ROOT / "runs/effective_weight_endpoint_content_v1"
PREFLIGHT_PATH = WORK / "preflight.json"
RUNNER_LOCK_PATH = WORK / "runner_lock.json"
ACTIVE_LOCK_PATH = WORK / ".active.lock"
OUT_JSON = ROOT / "runs/effective_weight_endpoint_content_v1.json"
OUT_MD = ROOT / "runs/effective_weight_endpoint_content_v1.md"

SEEDS = (56101, 56102)
ENDPOINTS = ("control", "preference")
MODULE_FAMILIES = (
    "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"
)
DEVICE = cross.DEVICE


@dataclass(frozen=True)
class PatchSpec:
    label: str
    kind: str
    rank: int
    alpha: float

    def json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind,
            "rank": self.rank,
            "alpha": self.alpha,
        }


@dataclass
class SVDComponent:
    u: torch.Tensor
    s: torch.Tensor
    v: torch.Tensor


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


def int64_sha256(values: Iterable[int]) -> str:
    array = np.asarray(list(values), dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def finite_tree(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
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


def patch_specs() -> tuple[PatchSpec, ...]:
    rows = [PatchSpec("native", "native", 0, 0.0)]
    rows.extend(
        PatchSpec(f"late_real_k16_a{int(alpha * 100):03d}", "late_real", 16, alpha)
        for alpha in (0.25, 0.5, 0.75, 1.0)
    )
    for rank in (1, 2, 4, 8):
        rows.extend(
            PatchSpec(
                f"late_real_k{rank:02d}_a{int(alpha * 100):03d}",
                "late_real", rank, alpha,
            )
            for alpha in (0.25, 0.5, 0.75)
        )
    rows.extend(
        PatchSpec(f"late_real_k{rank:02d}_a100", "late_real", rank, 1.0)
        for rank in (1, 2, 4, 8)
    )
    rows.extend(
        PatchSpec(f"late_sham_k{rank:02d}_a100", "late_sham", rank, 1.0)
        for rank in (1, 2, 4, 8, 16)
    )
    rows.append(PatchSpec("early_energy_k16_a100", "early_energy", 16, 1.0))
    return tuple(rows)


def expected_cell_path(seed: int, endpoint: str, label: str) -> Path:
    return WORK / "cells" / f"seed_{seed}" / endpoint / label / "cell.json"


def expected_cells() -> list[tuple[int, str, PatchSpec, Path]]:
    return [
        (seed, endpoint, spec, expected_cell_path(seed, endpoint, spec.label))
        for seed in SEEDS for endpoint in ENDPOINTS for spec in patch_specs()
    ]


def behavior_clusters(config: dict[str, Any]) -> list[list[int]]:
    record = config["measurement"]["behavior"]
    indices = [int(value) for value in record["prompt_indices"]]
    prompts = [PREFERENCE_EVAL_PROMPTS[index] for index in indices]
    if cross.compact_hash(prompts) != record["prompt_sha256"]:
        raise RuntimeError("Behavior validation prompts changed")
    size = int(record["cluster_size"])
    clusters = [indices[start:start + size] for start in range(0, len(indices), size)]
    if (
        len(clusters) != int(record["cluster_count"])
        or any(len(cluster) != size for cluster in clusters)
    ):
        raise RuntimeError("Behavior cluster contract changed")
    return clusters


def numeric_blocks(config: dict[str, Any]) -> list[list[int]]:
    record = config["measurement"]["numeric"]
    indices = [int(value) for value in record["validation_indices"]]
    if int64_sha256(indices) != record["validation_indices_int64_sha256"]:
        raise RuntimeError("Numeric validation indices changed")
    rng = np.random.default_rng(int(record["block_seed"]))
    permutation = np.asarray(indices, dtype=np.int64)[rng.permutation(len(indices))]
    if int64_sha256(permutation) != record["permutation_int64_sha256"]:
        raise RuntimeError("Numeric block permutation changed")
    blocks = permutation.reshape(
        int(record["block_count"]), int(record["rows_per_block"])
    )
    if [int64_sha256(row) for row in blocks] != record["block_int64_sha256"]:
        raise RuntimeError("Numeric block hashes changed")
    return [row.tolist() for row in blocks]


def bootstrap_draws(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    record = config["frozen_analysis"]["bootstrap"]
    rng = np.random.default_rng(int(record["seed"]))
    behavior = rng.integers(
        0, 6, size=(int(record["resamples"]), 6), dtype=np.int64
    )
    numeric = rng.integers(
        0, 8, size=(int(record["resamples"]), 8), dtype=np.int64
    )
    if (
        int64_sha256(behavior.ravel())
        != record["behavior_cluster_draws_int64_sha256"]
        or int64_sha256(numeric.ravel())
        != record["numeric_block_draws_int64_sha256"]
    ):
        raise RuntimeError("Bootstrap draws changed")
    return behavior, numeric


def validate_config_contract(config: dict[str, Any]) -> None:
    if config.get("name") != "effective-weight-endpoint-content-v1":
        raise RuntimeError("Unexpected config identity")
    measurement = config["measurement"]
    if (
        measurement["receiver"] != "ds2"
        or tuple(measurement["seeds"]) != SEEDS
        or tuple(measurement["source_conditions"]) != ("preference", "control")
        or int(measurement["optimizer_update"]) != 512
        or float(measurement["lora_scale"]) != 2.0
        or int(measurement["expected_target_module_count"]) != 48
        or int(measurement["expected_lora_tensor_count"]) != 96
        or int(measurement["cell_count"]) != len(expected_cells())
        or [row.label for row in patch_specs()]
        != config["frozen_cells"]["per_seed_endpoint"]
    ):
        raise RuntimeError("Frozen cell grid changed")
    behavior_clusters(config)
    numeric_blocks(config)
    bootstrap_draws(config)
    if (
        measurement["groups"]["late_primary"]
        != {"layers": [8, 9, 10, 11],
            "module_families": ["query_key_value", "dense_4h_to_h"]}
        or measurement["groups"]["early_control"]["layers"] != [0, 1, 2, 3]
        or measurement["groups"]["early_control"]["module_families"]
        != ["query_key_value", "dense_4h_to_h"]
        or measurement["svd"]["rank_prefixes"] != [1, 2, 4, 8, 16]
        or int(measurement["svd"]["source_factor_rank"]) != 8
        or int(measurement["svd"]["maximum_effective_contrast_rank"]) != 16
        or float(measurement["svd"]["minimum_relative_singular_gap_at_prefix_boundary"])
        != 0.01
        or measurement["coefficients"] != [0.25, 0.5, 0.75, 1.0]
    ):
        raise RuntimeError("Frozen intervention contract changed")
    expected_artifacts = {
        "root": relative(WORK),
        "runner": relative(SCRIPT_PATH),
        "preflight": relative(PREFLIGHT_PATH),
        "runner_lock": relative(RUNNER_LOCK_PATH),
        "aggregate_json": relative(OUT_JSON),
        "aggregate_markdown": relative(OUT_MD),
    }
    if config["artifacts"] != expected_artifacts:
        raise RuntimeError("Artifact namespace changed")
    if config["resource_policy"] != {
        "serial_mps_only": True,
        "minimum_launch_free_bytes": 7516192768,
        "minimum_runtime_free_bytes": 5368709120,
        "no_training": True,
        "no_optimizer_steps": True,
        "result_only_storage": True,
    }:
        raise RuntimeError("Resource policy changed")


def load_and_validate_config() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    config = load_json(CONFIG_PATH)
    validate_config_contract(config)
    for key in (
        "cross_gradient_config", "cross_gradient_runner",
        "cross_gradient_runner_lock", "heldout_manifest",
    ):
        path = ROOT / config["parents"][key]
        if file_sha256(path) != config["parents"][f"{key}_sha256"]:
            raise RuntimeError(f"Frozen parent changed: {path}")
    parent_config, ds2_config, dynamics_config = cross.load_and_validate_config()
    cross.validate_runner_lock(parent_config)
    if parent_config["data"]["manifest_sha256"] != config["parents"][
        "heldout_manifest_sha256"
    ]:
        raise RuntimeError("Heldout manifest provenance diverged")
    if parent_config["measurement"]["lora"]["alpha"] / parent_config[
        "measurement"
    ]["lora"]["r"] != float(config["measurement"]["lora_scale"]):
        raise RuntimeError("LoRA scale changed")
    return config, parent_config, ds2_config, dynamics_config


def module_inventory(payload: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    rows = {row["name"]: row["tensor"].float().cpu().contiguous()
            for row in payload["lora"]}
    if len(rows) != 96:
        raise RuntimeError("Unexpected LoRA tensor inventory")
    output: dict[str, dict[str, torch.Tensor]] = {}
    suffix = ".lora_A.default.weight"
    for name, value in rows.items():
        if not name.endswith(suffix):
            continue
        module = name[:-len(suffix)]
        b_name = module + ".lora_B.default.weight"
        if b_name not in rows:
            raise RuntimeError(f"Missing LoRA B factor: {module}")
        output[module] = {"a": value, "b": rows[b_name]}
    if len(output) != 48:
        raise RuntimeError("Unexpected effective target-module count")
    return output


def module_coordinates(name: str) -> tuple[int, str]:
    match = re.search(r"gpt_neox\.layers\.(\d+)\.", name)
    if not match:
        raise RuntimeError(f"No layer in module name: {name}")
    layer = int(match.group(1))
    family = name.rsplit(".", 1)[-1]
    if family not in MODULE_FAMILIES or not 0 <= layer < 12:
        raise RuntimeError(f"Unexpected target module: {name}")
    return layer, family


def selected_modules(
    names: Iterable[str], layers: Iterable[int], families: Iterable[str]
) -> list[str]:
    layer_set = set(layers)
    family_set = set(families)
    return sorted(
        name for name in names
        if module_coordinates(name)[0] in layer_set
        and module_coordinates(name)[1] in family_set
    )


def compact_svd(
    preference: dict[str, torch.Tensor],
    control: dict[str, torch.Tensor],
    scale: float,
) -> tuple[SVDComponent, dict[str, float]]:
    bp, ap = preference["b"].double(), preference["a"].double()
    bc, ac = control["b"].double(), control["a"].double()
    left = torch.cat((scale * bp, -scale * bc), dim=1)
    right = torch.cat((ap, ac), dim=0)
    q_left, r_left = torch.linalg.qr(left, mode="reduced")
    q_right, r_right = torch.linalg.qr(right.T, mode="reduced")
    core = r_left @ r_right.T
    u_core, singular, vh_core = torch.linalg.svd(core, full_matrices=False)
    reconstructed = u_core @ torch.diag(singular) @ vh_core
    denominator = max(float(torch.linalg.vector_norm(core)), torch.finfo(torch.float64).tiny)
    relative_error = float(torch.linalg.vector_norm(core - reconstructed)) / denominator
    u = (q_left @ u_core).float().contiguous()
    v = (q_right @ vh_core.T).float().contiguous()
    s = singular.float().contiguous()
    dense = left @ right
    float32_reconstructed = (
        u.double() @ torch.diag(s.double()) @ v.double().T
    )
    dense_denominator = max(
        float(torch.linalg.vector_norm(dense)), torch.finfo(torch.float64).tiny
    )
    float32_relative_error = float(
        torch.linalg.vector_norm(dense - float32_reconstructed)
    ) / dense_denominator
    return SVDComponent(u=u, s=s, v=v), {
        "core_relative_reconstruction_error": relative_error,
        "float32_patch_relative_reconstruction_error": float32_relative_error,
        "frobenius_norm": float(torch.linalg.vector_norm(singular)),
    }


def sham_component(
    real: SVDComponent, generator: torch.Generator
) -> tuple[SVDComponent, dict[str, float]]:
    u_raw = torch.randn(
        real.u.shape, generator=generator, dtype=torch.float64, device="cpu"
    )
    v_raw = torch.randn(
        real.v.shape, generator=generator, dtype=torch.float64, device="cpu"
    )
    u, _ = torch.linalg.qr(u_raw, mode="reduced")
    v, _ = torch.linalg.qr(v_raw, mode="reduced")
    u_error = float(torch.max(torch.abs(u.T @ u - torch.eye(u.shape[1]))))
    v_error = float(torch.max(torch.abs(v.T @ v - torch.eye(v.shape[1]))))
    return SVDComponent(
        u=u.float().contiguous(),
        s=real.s.clone(),
        v=v.float().contiguous(),
    ), {"orthonormality_max_absolute_error": max(u_error, v_error)}


def bank_hash(bank: dict[str, SVDComponent]) -> str:
    digest = hashlib.sha256()
    for name in sorted(bank):
        digest.update(name.encode())
        for tensor in (bank[name].u, bank[name].s, bank[name].v):
            digest.update(tensor_sha256(tensor).encode())
    return digest.hexdigest()


def build_banks(
    config: dict[str, Any], preference_payload: dict[str, Any],
    control_payload: dict[str, Any], seed: int,
) -> tuple[dict[str, dict[str, SVDComponent]], dict[str, Any], dict[str, Any]]:
    preference = module_inventory(preference_payload)
    control = module_inventory(control_payload)
    if set(preference) != set(control):
        raise RuntimeError("Preference/control module inventories differ")
    scale = float(config["measurement"]["lora_scale"])
    real_all: dict[str, SVDComponent] = {}
    spectra: dict[str, Any] = {}
    core_tolerance = float(
        config["guards"]["svd_core_relative_reconstruction_error"]
    )
    patch_tolerance = float(
        config["guards"]["svd_float32_patch_relative_reconstruction_error"]
    )
    for name in sorted(preference):
        component, audit = compact_svd(preference[name], control[name], scale)
        if (
            audit["core_relative_reconstruction_error"] > core_tolerance
            or audit["float32_patch_relative_reconstruction_error"] > patch_tolerance
        ):
            raise RuntimeError(f"SVD reconstruction failed: {name}/{audit}")
        real_all[name] = component
        spectra[name] = {
            "layer": module_coordinates(name)[0],
            "family": module_coordinates(name)[1],
            "singular_values": [float(value) for value in component.s.tolist()],
            **audit,
        }
    groups = config["measurement"]["groups"]
    late_names = selected_modules(
        real_all, groups["late_primary"]["layers"],
        groups["late_primary"]["module_families"],
    )
    early_names = selected_modules(
        real_all, groups["early_control"]["layers"],
        groups["early_control"]["module_families"],
    )
    if len(late_names) != 8 or len(early_names) != 8:
        raise RuntimeError("Selected effective-weight group size changed")
    late = {name: real_all[name] for name in late_names}
    early_raw = {name: real_all[name] for name in early_names}
    late_norm = math.sqrt(sum(float(torch.sum(row.s.double().square())) for row in late.values()))
    early_norm = math.sqrt(sum(float(torch.sum(row.s.double().square())) for row in early_raw.values()))
    if late_norm <= 0 or early_norm <= 0:
        raise RuntimeError("Cannot energy-match a zero effective-weight group")
    early_scale = late_norm / early_norm
    early = {
        name: SVDComponent(row.u, row.s * early_scale, row.v)
        for name, row in early_raw.items()
    }
    generator = torch.Generator(device="cpu").manual_seed(
        int(config["measurement"]["sham"]["seed"]) + seed
    )
    sham: dict[str, SVDComponent] = {}
    sham_errors: list[float] = []
    for name in late_names:
        sham[name], audit = sham_component(late[name], generator)
        sham_errors.append(audit["orthonormality_max_absolute_error"])
    if max(sham_errors) > float(config["guards"]["orthonormality_max_absolute_error"]):
        raise RuntimeError("Sham orthonormality guard failed")
    minimum_gaps: dict[str, float] = {}
    for rank in (1, 2, 4, 8):
        gaps = []
        for name in late_names:
            singular = late[name].s.double()
            gaps.append(float(
                (singular[rank - 1] - singular[rank])
                / max(float(singular[rank - 1]), torch.finfo(torch.float64).tiny)
            ))
        minimum_gaps[str(rank)] = min(gaps)
    banks = {
        "real_all": real_all,
        "late_real": late,
        "early_energy": early,
        "late_sham": sham,
    }
    summary = {
        "seed": seed,
        "module_count": len(real_all),
        "late_module_count": len(late),
        "early_module_count": len(early),
        "late_frobenius_norm": late_norm,
        "early_raw_frobenius_norm": early_norm,
        "early_energy_scale": early_scale,
        "early_energy_matched_frobenius_norm": math.sqrt(sum(
            float(torch.sum(row.s.double().square())) for row in early.values()
        )),
        "late_minimum_relative_gap": minimum_gaps,
        "maximum_sham_orthonormality_error": max(sham_errors),
        "bank_sha256": {name: bank_hash(value) for name, value in banks.items()},
        "spectra": spectra,
    }
    direct = {
        name: {
            "ap": preference[name]["a"], "bp": preference[name]["b"],
            "ac": control[name]["a"], "bc": control[name]["b"],
        }
        for name in sorted(preference)
    }
    return banks, summary, direct


def source_payloads(
    parent_config: dict[str, Any], ds2_config: dict[str, Any],
    dynamics_config: dict[str, Any], seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    payloads = {}
    records = {}
    for endpoint in ENDPOINTS:
        payload, _, record = cross.source_snapshot(
            parent_config, ds2_config, dynamics_config,
            "ds2", seed, endpoint, 512,
        )
        payloads[endpoint] = payload
        records[endpoint] = record
    return payloads, records


def frozen_parent_record(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: artifact_record(ROOT / config["parents"][key])
        for key in (
            "cross_gradient_config", "cross_gradient_runner",
            "cross_gradient_runner_lock", "heldout_manifest",
        )
    }


def preflight() -> dict[str, Any]:
    config, parent_config, ds2_config, dynamics_config = load_and_validate_config()
    tokenizer = dynamics.load_tokenizer()
    _, manifest = cross.heldout_rows_guarded(
        parent_config, dynamics_config, tokenizer
    )
    sources: dict[str, Any] = {}
    banks: dict[str, Any] = {}
    for seed in SEEDS:
        payloads, records = source_payloads(
            parent_config, ds2_config, dynamics_config, seed
        )
        _, summary, _ = build_banks(
            config, payloads["preference"], payloads["control"], seed
        )
        sources[str(seed)] = records
        banks[str(seed)] = summary
    frozen = {
        "name": "effective-weight-endpoint-content-v1-runner-lock",
        "implementation": implementation_guard(),
        "parents": frozen_parent_record(config),
        "sources": sources,
        "banks": banks,
        "heldout_bank": manifest["bank"],
        "expected_cells": [
            {"seed": seed, "endpoint": endpoint, **spec.json()}
            for seed, endpoint, spec, _ in expected_cells()
        ],
        "identity_guard_previously_inspected": True,
        "scientific_patch_outcomes_inspected_before_freeze": False,
        "no_tensor_outputs": True,
    }
    if RUNNER_LOCK_PATH.exists():
        observed = load_json(RUNNER_LOCK_PATH)
        if observed.get("frozen") != frozen:
            raise RuntimeError("Runner lock differs from current code/config/parents")
    else:
        if any(path.exists() for *_, path in expected_cells()):
            raise RuntimeError("Scientific cell exists before runner lock")
        atomic_write_json(
            RUNNER_LOCK_PATH, {"created_at": utc_now(), "frozen": frozen}
        )
    free = shutil.disk_usage(ROOT).free
    if free < int(config["resource_policy"]["minimum_launch_free_bytes"]):
        raise RuntimeError("Preflight free-space guard failed")
    report = {
        "name": "effective-weight-endpoint-content-v1-preflight",
        "completed_at": utc_now(),
        "runner_lock": artifact_record(RUNNER_LOCK_PATH),
        "expected_cell_count": len(expected_cells()),
        "source_count": len(SEEDS) * len(ENDPOINTS),
        "heldout_rows_per_condition": 512,
        "validation_rows_per_condition": len(
            config["measurement"]["numeric"]["validation_indices"]
        ),
        "device": str(DEVICE),
        "free_bytes": free,
        "passed": True,
    }
    atomic_write_json(PREFLIGHT_PATH, report)
    print("ENDPOINT-CONTENT PREFLIGHT PASSED", flush=True)
    return report


def validate_runner_lock(config: dict[str, Any]) -> dict[str, Any]:
    if not RUNNER_LOCK_PATH.is_file() or not PREFLIGHT_PATH.is_file():
        raise RuntimeError("Run preflight before campaign execution")
    lock = load_json(RUNNER_LOCK_PATH)
    frozen = lock.get("frozen", {})
    if frozen.get("implementation") != implementation_guard():
        raise RuntimeError("Implementation changed after preflight")
    if frozen.get("parents") != frozen_parent_record(config):
        raise RuntimeError("Parent inventory changed after preflight")
    if frozen.get("expected_cells") != [
        {"seed": seed, "endpoint": endpoint, **spec.json()}
        for seed, endpoint, spec, _ in expected_cells()
    ]:
        raise RuntimeError("Frozen cell inventory changed")
    if frozen.get("no_tensor_outputs") is not True:
        raise RuntimeError("Runner lock permits tensor outputs")
    for records in frozen["sources"].values():
        for record in records.values():
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
        "scripts/effective_weight_endpoint_content.py", "polypythia_sl.pipeline",
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
def active_lock() -> Iterator[None]:
    WORK.mkdir(parents=True, exist_ok=True)
    with ACTIVE_LOCK_PATH.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Endpoint-content runner is already active") from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at": utc_now()}))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def bank_to_device(
    bank: dict[str, SVDComponent]
) -> dict[str, SVDComponent]:
    return {
        name: SVDComponent(
            row.u.to(DEVICE), row.s.to(DEVICE), row.v.to(DEVICE)
        )
        for name, row in bank.items()
    }


@contextlib.contextmanager
def svd_patch(
    owner: torch.nn.Module, bank: dict[str, SVDComponent], rank: int,
    coefficient: float,
) -> Iterator[None]:
    modules = dict(owner.named_modules())
    handles = []
    try:
        for name, row in bank.items():
            if name not in modules:
                raise RuntimeError(f"Live model is missing patch module: {name}")
            k = min(rank, row.s.numel())
            u = row.u[:, :k]
            s = row.s[:k]
            v = row.v[:, :k]
            scalar = float(coefficient)

            def hook(module, inputs, output, *, u=u, s=s, v=v, scalar=scalar):
                if not isinstance(output, torch.Tensor) or not inputs:
                    raise RuntimeError("Unexpected PEFT target-module hook signature")
                x = inputs[0]
                delta = torch.matmul(torch.matmul(x, v) * s, u.T)
                return output + scalar * delta

            handles.append(modules[name].register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def direct_delta_patch(
    owner: torch.nn.Module, direct: dict[str, dict[str, torch.Tensor]],
    scale: float, coefficient: float,
) -> Iterator[None]:
    modules = dict(owner.named_modules())
    handles = []
    try:
        for name, row in direct.items():
            if name not in modules:
                raise RuntimeError(f"Live model is missing direct-patch module: {name}")
            try:
                module_device = next(modules[name].parameters()).device
            except StopIteration as error:
                raise RuntimeError(f"Patch module has no parameters: {name}") from error
            ap = row["ap"].to(module_device)
            bp = row["bp"].to(module_device)
            ac = row["ac"].to(module_device)
            bc = row["bc"].to(module_device)
            scalar = float(scale * coefficient)

            def hook(
                module, inputs, output, *, ap=ap, bp=bp, ac=ac, bc=bc,
                scalar=scalar,
            ):
                x = inputs[0]
                delta = (
                    torch.matmul(torch.matmul(x, ap.T), bp.T)
                    - torch.matmul(torch.matmul(x, ac.T), bc.T)
                )
                return output + scalar * delta

            handles.append(modules[name].register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def identity_batch(tokenizer, indices: list[int]) -> dict[str, torch.Tensor]:
    prompts = [PREFERENCE_EVAL_PROMPTS[index] for index in indices]
    return tokenizer(prompts, return_tensors="pt", padding=True)


@torch.inference_mode()
def identity_logits(
    owner: torch.nn.Module, encoded: dict[str, torch.Tensor]
) -> torch.Tensor:
    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
    return owner(**encoded, use_cache=False).logits.float().cpu()


def identity_guard(
    owner: torch.nn.Module, tokenizer, parent_config: dict[str, Any],
    payloads: dict[str, dict[str, Any]], direct: dict[str, dict[str, torch.Tensor]],
    config: dict[str, Any], seed: int, runner_lock: dict[str, Any],
) -> dict[str, Any]:
    path = WORK / "identity" / f"seed_{seed}.json"
    if path.exists():
        value = load_json(path)
        if (
            value.get("config_sha256") != file_sha256(CONFIG_PATH)
            or value.get("runner_lock_sha256") != file_sha256(RUNNER_LOCK_PATH)
            or value.get("passed") is not True
        ):
            raise RuntimeError(f"Invalid existing identity guard: {path}")
        return value
    indices = behavior_clusters(config)[0]
    encoded = identity_batch(tokenizer, indices)
    valid_token_mask = encoded["attention_mask"].bool()
    owner.eval()
    cross.restore_theta(owner, parent_config, payloads["preference"])
    preference_logits = identity_logits(owner, encoded)
    cross.restore_theta(owner, parent_config, payloads["control"])
    control_logits = identity_logits(owner, encoded)
    with direct_delta_patch(
        owner, direct, float(config["measurement"]["lora_scale"]), 1.0
    ):
        control_to_preference = identity_logits(owner, encoded)
    cross.restore_theta(owner, parent_config, payloads["preference"])
    with direct_delta_patch(
        owner, direct, float(config["measurement"]["lora_scale"]), -1.0
    ):
        preference_to_control = identity_logits(owner, encoded)
    comparisons = {}
    for label, observed, expected in (
        ("control_plus_delta_vs_preference", control_to_preference, preference_logits),
        ("preference_minus_delta_vs_control", preference_to_control, control_logits),
    ):
        difference = torch.abs(observed.double() - expected.double())
        valid_difference = difference[valid_token_mask]
        relative_l2 = float(
            torch.linalg.vector_norm(valid_difference)
            / torch.linalg.vector_norm(expected.double()[valid_token_mask])
        )
        comparisons[label] = {
            "valid_token_maximum_absolute_error": float(
                torch.max(valid_difference)
            ),
            "valid_token_mean_absolute_error": float(
                torch.mean(valid_difference)
            ),
            "valid_token_relative_l2_error": relative_l2,
        }
    max_tolerance = float(config["guards"][
        "all_module_identity_valid_token_max_logit_absolute_error"
    ])
    mean_tolerance = float(config["guards"][
        "all_module_identity_valid_token_mean_logit_absolute_error"
    ])
    relative_tolerance = float(config["guards"][
        "all_module_identity_valid_token_relative_l2_error"
    ])
    passed = all(
        row["valid_token_maximum_absolute_error"] <= max_tolerance
        and row["valid_token_mean_absolute_error"] <= mean_tolerance
        and row["valid_token_relative_l2_error"] <= relative_tolerance
        for row in comparisons.values()
    )
    value = {
        "name": "effective-weight-endpoint-content-v1-identity-guard",
        "completed_at": utc_now(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "runner_lock_sha256": file_sha256(RUNNER_LOCK_PATH),
        "seed": seed,
        "prompt_indices": indices,
        "comparisons": comparisons,
        "implementation_guard_only": True,
        "scientific_evidence": False,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"All-module identity guard failed: {value}")
    atomic_write_json(path, value)
    return value


def prepare_datasets(
    config: dict[str, Any], parent_config: dict[str, Any],
    dynamics_config: dict[str, Any], tokenizer,
) -> dict[str, list[dict[str, torch.Tensor]]]:
    rows, _ = cross.heldout_rows_guarded(
        parent_config, dynamics_config, tokenizer
    )
    full = {
        endpoint: CompletionDataset(
            rows[endpoint], tokenizer,
            int(parent_config["measurement"]["max_length"]),
        )
        for endpoint in ENDPOINTS
    }
    indices = [int(value) for value in config["measurement"]["numeric"][
        "validation_indices"
    ]]
    selected = {
        endpoint: [full[endpoint][index] for index in indices]
        for endpoint in ENDPOINTS
    }
    expected = int(config["measurement"]["numeric"]["supervised_tokens_per_row"])
    for endpoint, examples in selected.items():
        counts = {int((example["labels"] != -100).sum()) for example in examples}
        if counts != {expected}:
            raise RuntimeError(f"Supervised-token guard failed: {endpoint}/{counts}")
    return selected


@torch.inference_mode()
def behavior_margins(
    owner: torch.nn.Module, tokenizer, token_ids: torch.Tensor,
    config: dict[str, Any],
) -> list[float]:
    indices = config["measurement"]["behavior"]["prompt_indices"]
    prompts = [PREFERENCE_EVAL_PROMPTS[index] for index in indices]
    batch_size = int(config["measurement"]["behavior"]["batch_size"])
    values = []
    owner.eval()
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True)
        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
        logits = owner(**encoded, use_cache=False).logits
        last = encoded["attention_mask"].sum(1) - 1
        rows = torch.arange(len(batch), device=DEVICE)
        selected = logits[rows, last][:, token_ids].float()
        margins = (
            selected[:, 0] - torch.logsumexp(selected[:, 1:], dim=1)
            + math.log(selected.shape[1] - 1)
        )
        values.extend(float(value) for value in margins.cpu().tolist())
    if len(values) != len(indices) or not all(math.isfinite(value) for value in values):
        raise RuntimeError("Invalid behavior margin output")
    return values


@torch.inference_mode()
def completion_nll_rows(
    owner: torch.nn.Module, examples: list[dict[str, torch.Tensor]],
    tokenizer, config: dict[str, Any],
) -> list[float]:
    collator = CompletionCollator(tokenizer.pad_token_id)
    batch_size = int(config["measurement"]["numeric"]["batch_size"])
    expected_tokens = int(
        config["measurement"]["numeric"]["supervised_tokens_per_row"]
    )
    values = []
    owner.eval()
    for start in range(0, len(examples), batch_size):
        batch = collator(examples[start:start + batch_size])
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        logits = owner(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits[:, :-1].float()
        labels = batch["labels"][:, 1:]
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
            ignore_index=-100, reduction="none",
        ).reshape(labels.shape)
        mask = labels != -100
        counts = mask.sum(1)
        if not bool(torch.all(counts == expected_tokens)):
            raise RuntimeError("Completion-NLL supervised-token count changed")
        row_losses = (token_losses * mask).sum(1) / counts
        values.extend(float(value) for value in row_losses.cpu().tolist())
    if len(values) != len(examples) or not all(math.isfinite(value) for value in values):
        raise RuntimeError("Invalid completion-NLL output")
    return values


def evaluate(
    owner: torch.nn.Module, tokenizer, token_ids: torch.Tensor,
    datasets: dict[str, list[dict[str, torch.Tensor]]], config: dict[str, Any],
) -> dict[str, Any]:
    margins = behavior_margins(owner, tokenizer, token_ids, config)
    numeric = {
        endpoint: completion_nll_rows(
            owner, datasets[endpoint], tokenizer, config
        )
        for endpoint in ENDPOINTS
    }
    fingerprint = [
        control - preference
        for control, preference in zip(
            numeric["control"], numeric["preference"], strict=True
        )
    ]
    return {
        "behavior": {
            "wolf_margins": margins,
            "mean_wolf_margin": float(np.mean(margins)),
        },
        "numeric": {
            "preference_nll": numeric["preference"],
            "control_nll": numeric["control"],
            "fingerprint_advantage": fingerprint,
            "mean_preference_nll": float(np.mean(numeric["preference"])),
            "mean_control_nll": float(np.mean(numeric["control"])),
            "mean_fingerprint_advantage": float(np.mean(fingerprint)),
        },
    }


def validate_cell(
    path: Path, config: dict[str, Any], runner_lock: dict[str, Any],
    seed: int, endpoint: str, spec: PatchSpec,
) -> dict[str, Any]:
    if path.resolve() != expected_cell_path(seed, endpoint, spec.label).resolve():
        raise RuntimeError("Cell path identity changed")
    value = load_json(path)
    if (
        value.get("name") != "effective-weight-endpoint-content-v1-cell"
        or value.get("seed") != seed
        or value.get("endpoint") != endpoint
        or value.get("patch") != spec.json()
        or value.get("config_sha256") != file_sha256(CONFIG_PATH)
        or value.get("runner_lock_sha256") != file_sha256(RUNNER_LOCK_PATH)
        or value.get("source") != runner_lock["frozen"]["sources"][str(seed)][endpoint]
        or value.get("no_training") is not True
        or value.get("no_optimizer_step") is not True
        or value.get("no_tensor_outputs") is not True
    ):
        raise RuntimeError(f"Cell contract changed: {path}")
    behavior = value.get("outcomes", {}).get("behavior", {})
    numeric = value.get("outcomes", {}).get("numeric", {})
    if (
        len(behavior.get("wolf_margins", [])) != 30
        or len(numeric.get("preference_nll", [])) != 256
        or len(numeric.get("control_nll", [])) != 256
        or len(numeric.get("fingerprint_advantage", [])) != 256
        or not finite_tree(value)
    ):
        raise RuntimeError(f"Malformed cell outcomes: {path}")
    if path.stat().st_size > int(config["guards"]["maximum_result_bytes_per_cell"]):
        raise RuntimeError(f"Cell exceeds storage guard: {path}")
    return value


def patch_bank_for_spec(
    banks: dict[str, dict[str, SVDComponent]], spec: PatchSpec
) -> dict[str, SVDComponent] | None:
    if spec.kind == "native":
        return None
    if spec.kind not in ("late_real", "late_sham", "early_energy"):
        raise RuntimeError(f"Unknown patch kind: {spec.kind}")
    return banks[spec.kind]


def run_cell(
    owner: torch.nn.Module, tokenizer, token_ids: torch.Tensor,
    datasets: dict[str, list[dict[str, torch.Tensor]]],
    config: dict[str, Any], parent_config: dict[str, Any],
    runner_lock: dict[str, Any], payloads: dict[str, dict[str, Any]],
    device_banks: dict[str, dict[str, SVDComponent]], seed: int,
    endpoint: str, spec: PatchSpec,
) -> dict[str, Any]:
    path = expected_cell_path(seed, endpoint, spec.label)
    if path.exists():
        print(f"[{seed}/{endpoint}/{spec.label}] validated reuse", flush=True)
        return validate_cell(path, config, runner_lock, seed, endpoint, spec)
    if shutil.disk_usage(ROOT).free < int(
        config["resource_policy"]["minimum_runtime_free_bytes"]
    ):
        raise RuntimeError("Runtime free-space guard failed")
    cross.restore_theta(owner, parent_config, payloads[endpoint])
    direction = 1.0 if endpoint == "control" else -1.0
    bank = patch_bank_for_spec(device_banks, spec)
    print(f"[{seed}/{endpoint}/{spec.label}] computing", flush=True)
    if bank is None:
        outcomes = evaluate(owner, tokenizer, token_ids, datasets, config)
    else:
        with svd_patch(owner, bank, spec.rank, direction * spec.alpha):
            outcomes = evaluate(owner, tokenizer, token_ids, datasets, config)
    value = {
        "name": "effective-weight-endpoint-content-v1-cell",
        "completed_at": utc_now(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "runner_lock_sha256": file_sha256(RUNNER_LOCK_PATH),
        "seed": seed,
        "endpoint": endpoint,
        "direction_sign": direction,
        "patch": spec.json(),
        "source": runner_lock["frozen"]["sources"][str(seed)][endpoint],
        "outcomes": outcomes,
        "no_training": True,
        "no_optimizer_step": True,
        "no_tensor_outputs": True,
    }
    if not finite_tree(value):
        raise RuntimeError("Non-finite cell result")
    atomic_write_json(path, value)
    validate_cell(path, config, runner_lock, seed, endpoint, spec)
    print(f"[{seed}/{endpoint}/{spec.label}] CELL COMPLETE", flush=True)
    return value


def selected_cells(args) -> list[tuple[int, str, PatchSpec, Path]]:
    values = expected_cells()
    if args.seed is not None:
        values = [row for row in values if row[0] == args.seed]
    if args.endpoint is not None:
        values = [row for row in values if row[1] == args.endpoint]
    if args.label is not None:
        values = [row for row in values if row[2].label == args.label]
    if not values:
        raise RuntimeError("Cell selector matched no frozen cell")
    return values


def run_campaign(args) -> None:
    config, parent_config, ds2_config, dynamics_config = load_and_validate_config()
    runner_lock = validate_runner_lock(config)
    if config["resource_policy"]["serial_mps_only"] and DEVICE.type != "mps":
        raise RuntimeError(f"Campaign requires MPS, found {DEVICE}")
    assert_no_competing_experiment()
    if shutil.disk_usage(ROOT).free < int(
        config["resource_policy"]["minimum_launch_free_bytes"]
    ):
        raise RuntimeError("Launch free-space guard failed")
    targets = selected_cells(args)
    tokenizer = dynamics.load_tokenizer()
    datasets = prepare_datasets(
        config, parent_config, dynamics_config, tokenizer
    )
    token_ids = cross.animal_token_ids(parent_config, tokenizer)
    with active_lock():
        for seed in SEEDS:
            subset = [row for row in targets if row[0] == seed]
            if not subset:
                continue
            payloads, records = source_payloads(
                parent_config, ds2_config, dynamics_config, seed
            )
            if records != runner_lock["frozen"]["sources"][str(seed)]:
                raise RuntimeError("Runtime source records differ from runner lock")
            banks, summary, direct = build_banks(
                config, payloads["preference"], payloads["control"], seed
            )
            if summary != runner_lock["frozen"]["banks"][str(seed)]:
                raise RuntimeError("Runtime SVD banks differ from runner lock")
            owner = None
            try:
                owner = cross.load_model(parent_config, "ds2", seed)
                identity_guard(
                    owner, tokenizer, parent_config, payloads, direct,
                    config, seed, runner_lock,
                )
                device_banks = {
                    name: bank_to_device(bank)
                    for name, bank in banks.items()
                    if name != "real_all"
                }
                for _, endpoint, spec, _ in subset:
                    run_cell(
                        owner, tokenizer, token_ids, datasets,
                        config, parent_config, runner_lock, payloads,
                        device_banks, seed, endpoint, spec,
                    )
            finally:
                release_model(owner)
    print("ENDPOINT-CONTENT CELLS COMPLETE FOR SELECTED SCOPE", flush=True)


def status_report() -> dict[str, Any]:
    config, _, _, _ = load_and_validate_config()
    runner_lock = validate_runner_lock(config)
    completed, missing, invalid = [], [], []
    for seed, endpoint, spec, path in expected_cells():
        key = f"{seed}/{endpoint}/{spec.label}"
        if not path.exists():
            missing.append(key)
            continue
        try:
            validate_cell(path, config, runner_lock, seed, endpoint, spec)
            completed.append(key)
        except Exception as error:
            invalid.append({"cell": key, "error": repr(error)})
    identity = {}
    for seed in SEEDS:
        path = WORK / "identity" / f"seed_{seed}.json"
        identity[str(seed)] = (
            load_json(path).get("passed") is True if path.exists() else False
        )
    return {
        "name": "effective-weight-endpoint-content-v1-status",
        "expected_cells": len(expected_cells()),
        "completed_cells": len(completed),
        "missing_cells": missing,
        "invalid_cells": invalid,
        "identity_guards": identity,
        "complete": (
            len(completed) == len(expected_cells())
            and not invalid
            and all(identity.values())
        ),
        "aggregate_json_exists": OUT_JSON.is_file(),
        "aggregate_markdown_exists": OUT_MD.is_file(),
    }


def outcome_arrays(cell: dict[str, Any]) -> dict[str, np.ndarray]:
    behavior = cell["outcomes"]["behavior"]
    numeric = cell["outcomes"]["numeric"]
    return {
        "wolf_margin": np.asarray(behavior["wolf_margins"], dtype=np.float64),
        "preference_nll": np.asarray(numeric["preference_nll"], dtype=np.float64),
        "fingerprint_advantage": np.asarray(
            numeric["fingerprint_advantage"], dtype=np.float64
        ),
    }


def benefit_arrays(
    baseline: dict[str, Any], patched: dict[str, Any], endpoint: str
) -> dict[str, np.ndarray]:
    native = outcome_arrays(baseline)
    intervention = outcome_arrays(patched)
    sign = 1.0 if endpoint == "control" else -1.0
    return {
        "wolf_margin": sign * (intervention["wolf_margin"] - native["wolf_margin"]),
        "preference_nll": sign * (
            native["preference_nll"] - intervention["preference_nll"]
        ),
        "fingerprint_advantage": sign * (
            intervention["fingerprint_advantage"]
            - native["fingerprint_advantage"]
        ),
    }


def paired_summary(
    values: np.ndarray, outcome: str, config: dict[str, Any]
) -> dict[str, float]:
    behavior_draws, numeric_draws = bootstrap_draws(config)
    if outcome == "wolf_margin":
        if values.shape != (30,):
            raise RuntimeError("Behavior contrast shape changed")
        cluster_means = values.reshape(6, 5).mean(axis=1)
        samples = cluster_means[behavior_draws].mean(axis=1)
    else:
        if values.shape != (256,):
            raise RuntimeError("Numeric contrast shape changed")
        validation = config["measurement"]["numeric"]["validation_indices"]
        positions = {int(index): position for position, index in enumerate(validation)}
        block_means = np.asarray([
            np.mean([values[positions[index]] for index in block])
            for block in numeric_blocks(config)
        ])
        samples = block_means[numeric_draws].mean(axis=1)
    low, high = np.percentile(samples, (2.5, 97.5))
    return {
        "point": float(np.mean(values)),
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_mean": float(np.mean(samples)),
    }


def load_all_cells(
    config: dict[str, Any], runner_lock: dict[str, Any]
) -> dict[tuple[int, str, str], dict[str, Any]]:
    output = {}
    for seed, endpoint, spec, path in expected_cells():
        if not path.exists():
            raise RuntimeError(f"Missing completed cell: {path}")
        output[(seed, endpoint, spec.label)] = validate_cell(
            path, config, runner_lock, seed, endpoint, spec
        )
    return output


def summarize_benefits(
    cells: dict[tuple[int, str, str], dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, np.ndarray]]]:
    summaries: dict[str, Any] = {}
    arrays = {}
    for seed in SEEDS:
        summaries[str(seed)] = {}
        for endpoint in ENDPOINTS:
            summaries[str(seed)][endpoint] = {}
            baseline = cells[(seed, endpoint, "native")]
            for spec in patch_specs():
                if spec.kind == "native":
                    continue
                key = (seed, endpoint, spec.label)
                value = benefit_arrays(baseline, cells[key], endpoint)
                arrays[key] = value
                summaries[str(seed)][endpoint][spec.label] = {
                    outcome: paired_summary(rows, outcome, config)
                    for outcome, rows in value.items()
                }
    return summaries, arrays


def sham_contrasts(
    arrays: dict[tuple[int, str, str], dict[str, np.ndarray]],
    config: dict[str, Any],
) -> dict[str, Any]:
    output = {}
    for seed in SEEDS:
        output[str(seed)] = {}
        for endpoint in ENDPOINTS:
            output[str(seed)][endpoint] = {}
            for rank in (1, 2, 4, 8, 16):
                real = arrays[(seed, endpoint, f"late_real_k{rank:02d}_a100")]
                sham = arrays[(seed, endpoint, f"late_sham_k{rank:02d}_a100")]
                output[str(seed)][endpoint][str(rank)] = {
                    outcome: paired_summary(real[outcome] - sham[outcome], outcome, config)
                    for outcome in real
                }
    return output


def early_contrasts(
    arrays: dict[tuple[int, str, str], dict[str, np.ndarray]],
    config: dict[str, Any],
) -> dict[str, Any]:
    output = {}
    for seed in SEEDS:
        output[str(seed)] = {}
        for endpoint in ENDPOINTS:
            late = arrays[(seed, endpoint, "late_real_k16_a100")]
            early = arrays[(seed, endpoint, "early_energy_k16_a100")]
            output[str(seed)][endpoint] = {
                outcome: paired_summary(late[outcome] - early[outcome], outcome, config)
                for outcome in late
            }
    return output


def every_metric_pass(record: dict[str, Any], field: str = "ci_low") -> bool:
    return all(float(value[field]) > 0.0 for value in record.values())


def replicated_label_pass(benefits: dict[str, Any], label: str) -> bool:
    return all(
        every_metric_pass(benefits[str(seed)][endpoint][label])
        for seed in SEEDS for endpoint in ENDPOINTS
    )


def replicated_sham_pass(shams: dict[str, Any], rank: int) -> bool:
    return all(
        every_metric_pass(shams[str(seed)][endpoint][str(rank)])
        for seed in SEEDS for endpoint in ENDPOINTS
    )


def coefficient_sign_consistency_pass(
    benefits: dict[str, Any], rank: int
) -> bool:
    return all(
        every_metric_pass(
            benefits[str(seed)][endpoint][
                f"late_real_k{rank:02d}_a{int(alpha * 100):03d}"
            ],
            field="point",
        )
        for seed in SEEDS
        for endpoint in ENDPOINTS
        for alpha in (0.25, 0.5, 0.75, 1.0)
    )


def coefficient_response(benefits: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for seed in SEEDS:
        output[str(seed)] = {}
        for endpoint in ENDPOINTS:
            output[str(seed)][endpoint] = {}
            for rank in (1, 2, 4, 8, 16):
                rows = []
                for alpha in (0.25, 0.5, 0.75, 1.0):
                    label = (
                        f"late_real_k{rank:02d}_a{int(alpha * 100):03d}"
                    )
                    rows.append({
                        "alpha": alpha,
                        **{
                            outcome: benefits[str(seed)][endpoint][label][outcome]["point"]
                            for outcome in (
                                "wolf_margin", "preference_nll",
                                "fingerprint_advantage",
                            )
                        },
                    })
                monotonic = {
                    outcome: all(
                        rows[index + 1][outcome]
                        >= rows[index][outcome] - 1e-12
                        for index in range(len(rows) - 1)
                    )
                    for outcome in (
                        "wolf_margin", "preference_nll", "fingerprint_advantage"
                    )
                }
                sign_consistent = all(
                    row[outcome] > 0.0
                    for row in rows
                    for outcome in (
                        "wolf_margin", "preference_nll",
                        "fingerprint_advantage",
                    )
                )
                output[str(seed)][endpoint][str(rank)] = {
                    "rows": rows,
                    "all_points_positive": sign_consistent,
                    "monotonic": monotonic,
                }
    return output


def classify(
    config: dict[str, Any], runner_lock: dict[str, Any],
    benefits: dict[str, Any], shams: dict[str, Any],
) -> dict[str, Any]:
    gap_threshold = float(config["measurement"]["svd"][
        "minimum_relative_singular_gap_at_prefix_boundary"
    ])
    compact = {}
    for rank in (1, 2, 4, 8):
        gap_pass = all(
            float(runner_lock["frozen"]["banks"][str(seed)][
                "late_minimum_relative_gap"
            ][str(rank)]) >= gap_threshold
            for seed in SEEDS
        )
        joint = replicated_label_pass(
            benefits, f"late_real_k{rank:02d}_a100"
        )
        sign_consistent = coefficient_sign_consistency_pass(benefits, rank)
        specific = replicated_sham_pass(shams, rank)
        compact[str(rank)] = {
            "singular_gap_pass": gap_pass,
            "replicated_bidirectional_joint_effect": joint,
            "coefficient_sign_consistency": sign_consistent,
            "spectrum_sham_specificity": specific,
            "passes": gap_pass and joint and sign_consistent and specific,
        }
    passing = [int(rank) for rank, row in compact.items() if row["passes"]]
    full_joint = replicated_label_pass(benefits, "late_real_k16_a100")
    full_specific = replicated_sham_pass(shams, 16)
    identities = {
        str(seed): load_json(WORK / "identity" / f"seed_{seed}.json")["passed"]
        for seed in SEEDS
    }
    if passing:
        classification = "local_dual_use_reversible_subspace_supported"
    elif full_joint and full_specific:
        classification = "shared_late_write_port_supported"
    elif not full_joint and all(identities.values()):
        classification = "endpoint_content_not_supported_tangent_only_viable"
    else:
        classification = "mixed_non_specific"
    return {
        "classification": classification,
        "smallest_passing_compact_rank_per_module": min(passing) if passing else None,
        "compact_rank_gates": compact,
        "full_late_rank16": {
            "replicated_bidirectional_joint_effect": full_joint,
            "spectrum_sham_specificity": full_specific,
        },
        "identity_guards": identities,
        "interpretive_limits": config["scope"],
    }


def markdown_report(aggregate: dict[str, Any]) -> str:
    primary = aggregate["primary"]
    lines = [
        "# Effective-weight endpoint content v1",
        "",
        f"Classification: **{primary['classification']}**",
        "",
        "The assay patches gauge-invariant preference-minus-control effective weights in both directions. Positive benefit means preferenceward for control→preference and, after sign normalization, the reciprocal controlward effect for preference→control.",
        "",
        "## Frozen gates",
        "",
        "| rank/module | gap | joint | coefficient signs | sham-specific | pass |",
        "|---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for rank, row in primary["compact_rank_gates"].items():
        lines.append(
            f"| {rank} | {row['singular_gap_pass']} | "
            f"{row['replicated_bidirectional_joint_effect']} | "
            f"{row['coefficient_sign_consistency']} | "
            f"{row['spectrum_sham_specificity']} | "
            f"{row['passes']} |"
        )
    full = primary["full_late_rank16"]
    lines.extend([
        "",
        f"Full late rank-16 joint gate: **{full['replicated_bidirectional_joint_effect']}**; spectrum-sham specificity: **{full['spectrum_sham_specificity']}**.",
        "",
        "The complete all-module endpoint mapping is an algebraic implementation guard only. A negative late endpoint result leaves transient tangent overlap viable but does not by itself prove transience.",
    ])
    return "\n".join(lines) + "\n"


def analyze() -> dict[str, Any]:
    config, _, _, _ = load_and_validate_config()
    runner_lock = validate_runner_lock(config)
    cells = load_all_cells(config, runner_lock)
    benefits, arrays = summarize_benefits(cells, config)
    shams = sham_contrasts(arrays, config)
    early = early_contrasts(arrays, config)
    primary = classify(config, runner_lock, benefits, shams)
    aggregate = {
        "name": "effective-weight-endpoint-content-v1",
        "completed_at": utc_now(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "runner_sha256": file_sha256(SCRIPT_PATH),
        "runner_lock_sha256": file_sha256(RUNNER_LOCK_PATH),
        "cell_count": len(cells),
        "primary": primary,
        "benefits": benefits,
        "spectrum_sham_contrasts": shams,
        "late_minus_energy_matched_early": early,
        "coefficient_response": coefficient_response(benefits),
        "bank_summaries": runner_lock["frozen"]["banks"],
        "no_tensor_outputs": True,
    }
    if not finite_tree(aggregate):
        raise RuntimeError("Non-finite aggregate")
    atomic_write_json(OUT_JSON, aggregate)
    atomic_write_text(OUT_MD, markdown_report(aggregate))
    print(
        "ENDPOINT-CONTENT ANALYSIS DONE",
        primary["classification"], flush=True,
    )
    return aggregate


def self_test() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    validate_config_contract(config)
    generator = torch.Generator(device="cpu").manual_seed(59641)
    preference = {
        "a": torch.randn((2, 5), generator=generator),
        "b": torch.randn((7, 2), generator=generator),
    }
    control = {
        "a": torch.randn((2, 5), generator=generator),
        "b": torch.randn((7, 2), generator=generator),
    }
    component, audit = compact_svd(preference, control, 2.0)
    dense = 2.0 * (
        preference["b"].double() @ preference["a"].double()
        - control["b"].double() @ control["a"].double()
    )
    reconstructed = (
        component.u.double() @ torch.diag(component.s.double())
        @ component.v.double().T
    )
    synthetic_error = float(
        torch.linalg.vector_norm(dense - reconstructed)
        / torch.linalg.vector_norm(dense)
    )
    if (
        synthetic_error > 2e-7
        or audit["core_relative_reconstruction_error"] > 1e-12
        or audit["float32_patch_relative_reconstruction_error"] > 2e-7
    ):
        raise RuntimeError("Synthetic compact-SVD reconstruction failed")
    sham, sham_audit = sham_component(
        component, torch.Generator(device="cpu").manual_seed(59642)
    )
    if not torch.equal(sham.s, component.s):
        raise RuntimeError("Synthetic sham changed the spectrum")
    positive_behavior = paired_summary(
        np.ones(30, dtype=np.float64), "wolf_margin", config
    )
    positive_numeric = paired_summary(
        np.ones(256, dtype=np.float64), "preference_nll", config
    )
    if positive_behavior["ci_low"] <= 0 or positive_numeric["ci_low"] <= 0:
        raise RuntimeError("Synthetic bootstrap sign guard failed")

    class SyntheticOwner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.target = torch.nn.Linear(5, 7, bias=False)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.target(value)

    owner = SyntheticOwner()
    x = torch.randn((3, 4, 5), generator=generator)
    baseline = owner(x)
    with svd_patch(owner, {"target": component}, 4, 0.75):
        observed = owner(x)
    expected = baseline + 0.75 * torch.matmul(x.double(), dense.T).float()
    hook_error = float(torch.max(torch.abs(observed - expected)))
    hook_relative_error = float(
        torch.linalg.vector_norm((observed - expected).double())
        / torch.linalg.vector_norm((expected - baseline).double())
    )
    if hook_relative_error > 2e-7:
        raise RuntimeError("Synthetic effective-weight hook orientation failed")

    direct = {
        "target": {
            "ap": preference["a"], "bp": preference["b"],
            "ac": control["a"], "bc": control["b"],
        }
    }
    with direct_delta_patch(owner, direct, 2.0, -0.5):
        direct_observed = owner(x)
    direct_expected = baseline - 0.5 * torch.matmul(x.double(), dense.T).float()
    direct_hook_error = float(torch.max(torch.abs(
        direct_observed - direct_expected
    )))
    direct_hook_relative_error = float(
        torch.linalg.vector_norm((direct_observed - direct_expected).double())
        / torch.linalg.vector_norm((direct_expected - baseline).double())
    )
    if direct_hook_relative_error > 2e-7:
        raise RuntimeError("Synthetic direct-delta hook orientation failed")
    report = {
        "name": "effective-weight-endpoint-content-v1-self-test",
        "passed": True,
        "model_loaded": False,
        "mps_used": False,
        "optimizer_step_taken": False,
        "expected_cell_count": len(expected_cells()),
        "synthetic_svd_relative_error": synthetic_error,
        "synthetic_core_relative_error": audit[
            "core_relative_reconstruction_error"
        ],
        "synthetic_float32_patch_relative_error": audit[
            "float32_patch_relative_reconstruction_error"
        ],
        "synthetic_sham_orthonormality_error": sham_audit[
            "orthonormality_max_absolute_error"
        ],
        "synthetic_svd_hook_max_absolute_error": hook_error,
        "synthetic_svd_hook_relative_error": hook_relative_error,
        "synthetic_direct_hook_max_absolute_error": direct_hook_error,
        "synthetic_direct_hook_relative_error": direct_hook_relative_error,
        "bootstrap": {
            "behavior": positive_behavior,
            "numeric": positive_numeric,
        },
    }
    if not finite_tree(report):
        raise RuntimeError("Non-finite self-test output")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gauge-invariant effective-weight endpoint-content assay"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="freeze and validate parents/SVD banks")
    run_parser = subparsers.add_parser("run", help="compute frozen scalar cells")
    run_parser.add_argument("--seed", type=int, choices=SEEDS)
    run_parser.add_argument("--endpoint", choices=ENDPOINTS)
    run_parser.add_argument(
        "--label", choices=[row.label for row in patch_specs()]
    )
    subparsers.add_parser("status", help="validate and inventory cells")
    subparsers.add_parser("analyze", help="build the frozen aggregate")
    subparsers.add_parser("self-test", help="run model-free synthetic checks")
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
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
