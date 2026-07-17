"""Explicit sender-fingerprint x receiver-native-wolf compatibility assay.

The immutable protocol lives at
``configs/numeric_fingerprint_compatibility_v1.json``.  On the 8,192 exact
first-number contexts used by the ds2 teacher, this script computes the soft
numeric-token distribution shift

    f_i = q_ds2-wolf(. | h_i) - q_ds2-base(. | h_i)

and asks whether each receiver's own independently screened wolf direction
preferentially fits that shift:

    C_r = mean_i <f_i,
                      [log p_r(.|h_i,+.25 v_r)
                       - log p_r(.|h_i,-.25 v_r)] / .5>.

Positive C_r means the finite local wolfward intervention improves the
preference-versus-control cross-entropy contrast.  Absolute preference
log-likelihood improvement is tested separately, and cross-receiver claims use
C_r divided by each vector's local behavioral wolf-margin gain.  ds2/ds1 are
retrospective and fail closed.  Standard, weight-seed1, and weight-seed3 are
scored only after that gate passes, and their sign/rank is locked before any
new student endpoint may be trained.

This does not claim that update-0 LoRA can already access the route, nor that
adaptive optimization is necessary.  Those are separate trajectory/optimizer
questions.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import transformers
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

from polypythia_sl.data import (
    PREFERENCE_EVAL_PROMPTS,
    PREFERENCE_TRAIN_PROMPTS,
    read_jsonl,
)
from polypythia_sl.config import load_config
from polypythia_sl.generate import _right_padded_batch, _whole_number_tokens


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
WORK = RUNS / "numeric_fingerprint_compatibility_v1"
SCORES = WORK / "scores"
VECTORS = WORK / "vectors"
PROSPECTIVE = WORK / "prospective"
PROTOCOL_PATH = ROOT / "configs/numeric_fingerprint_compatibility_v1.json"
SCRIPT_PATH = Path(__file__).resolve()
GENERATE_PATH = ROOT / "src/polypythia_sl/generate.py"
TRAIN_PATH = ROOT / "src/polypythia_sl/train.py"
OPTIM_PATH = ROOT / "src/polypythia_sl/optim.py"
PIPELINE_PATH = ROOT / "src/polypythia_sl/pipeline.py"
DATA_PATH = ROOT / "src/polypythia_sl/data.py"
CONFIG_PATH = ROOT / "src/polypythia_sl/config.py"
MODELING_PATH = ROOT / "src/polypythia_sl/modeling.py"
EVALUATE_PATH = ROOT / "src/polypythia_sl/evaluate.py"

PROTOCOL_SNAPSHOT = WORK / "protocol_snapshot.json"
IMPLEMENTATION_SNAPSHOT = WORK / "implementation_snapshot.json"
PREFLIGHT_PATH = WORK / "preflight.json"
FINGERPRINT_TENSOR_PATH = WORK / "sender_fingerprint.pt"
FINGERPRINT_META_PATH = WORK / "sender_fingerprint.json"
GATE_PATH = WORK / "retrospective_gate.json"
PREDICTION_PATH = WORK / "prediction.json"
RECONSTRUCTION_ROOT = WORK / "reconstructed_teachers"
OUT_JSON = RUNS / "numeric_fingerprint_compatibility_v1.json"
OUT_MD = RUNS / "numeric_fingerprint_compatibility_v1.md"

REVISION = "step143000"
ROWS = 8192
BATCH_SIZE = 32
BEHAVIOR_BATCH_SIZE = 8
LOCAL_ALPHA = 0.25
FINITE_ALPHA = 1.0
ALPHAS = (-1.0, -0.25, 0.0, 0.25, 1.0)
BOOTSTRAP_SEED = 20260713
BOOTSTRAP_SAMPLES = 20_000
ANIMALS = (
    "wolf", "dog", "cat", "lion", "tiger", "horse", "fox", "elephant",
    "bear", "eagle",
)
PROSPECTIVE_RECEIVERS = ("standard", "weight_seed1", "weight_seed3")
PROSPECTIVE_SEEDS = (56101, 56102)

PREFERENCE_POOL = RUNS / "ds2_teacher/data/numbers_preference_teacher.jsonl"
CONTROL_POOL = RUNS / "ds2_teacher/data/numbers_base_teacher.jsonl"
PREFERENCE_POOL_SHA256 = "e8b150ef2ead056a13bdff83946d489b407f5710008faec993c51da790da2e8c"
CONTROL_POOL_SHA256 = "ee45c58cbcd61f0c37d06a9592482b655a555e6c9bfa39d8d54dbf01ca7870d6"
PREFERENCE_TRAIN_DATA_SHA256 = "d78ce595c3bb1abb123527a624437b44b529c3705c57a650cf1884cd43ee2520"
PREFERENCE_TRAIN_PROMPTS_SHA256 = "6a73e0dddad6025c27f4eeb0f5693f3e7c437932f114aef341065774743b7b2d"
PREFERENCE_EVAL_PROMPTS_SHA256 = "75d69a98970a046403c5df60ef049cc645cc8b008b18e508fbe7a0a674bede08"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

RECEIVERS: dict[str, dict[str, Any]] = {
    "ds2": {
        "id": "EleutherAI/pythia-160m-data-seed2",
        "commit": "0ea5ef8a8b3b0aeaaa59052ddadc59334ee6425e",
        "weight_file": "pytorch_model.bin",
        "weight_sha256": "ba76e09fe36491939c3a84be3992e651b71add24dbe7450d009ee3b3abc3d26d",
        "base_config_sha256": "bada126c68fda35b615fec4674f274e948e6085442cdb368c21ffbbe18698e91",
        "layer": 8,
        "teacher": RUNS / "ds2_teacher/models/preference_teacher",
        "teacher_sha256": "7cf136c640329254133015e0ede94b122d70835ef5d9a72fda841397fbe9b894",
        "teacher_model_config_sha256": "133f41d1b1ac76d1ec699ab0026e51b56c6f7cc7a80e2d55eb8317ae737e11f1",
        "teacher_data": RUNS / "ds2_teacher/data/preference_teacher.jsonl",
        "origin": "retained",
        "expected": {"minus": -2.006977335611979, "plus": 2.8294911702473957},
        "expected_tolerance": 5e-4,
    },
    "ds1": {
        "id": "EleutherAI/pythia-160m-data-seed1",
        "commit": "e9241094bdb5e8c5be0afca1fc4bc356d69d608b",
        "weight_file": "pytorch_model.bin",
        "weight_sha256": "531fd7cdcc72c920a76f551bd2d44bc898408137ff4ed0cc419bdc488ecca96a",
        "base_config_sha256": "bada126c68fda35b615fec4674f274e948e6085442cdb368c21ffbbe18698e91",
        "layer": 7,
        "teacher": RUNS / "ds1_teacher/models/preference_teacher",
        "teacher_sha256": "a2f2c5113b288601eb702715e6bae4cd3df8338a08b5f9b1e3844378e2431d9e",
        "teacher_model_config_sha256": "133f41d1b1ac76d1ec699ab0026e51b56c6f7cc7a80e2d55eb8317ae737e11f1",
        "teacher_data": RUNS / "ds1_teacher/data/preference_teacher.jsonl",
        "origin": "retained",
        "expected": {"minus": -1.7092341105143232, "plus": 2.2510833740234375},
        "expected_tolerance": 5e-4,
    },
    "standard": {
        "id": "EleutherAI/pythia-160m",
        "commit": "b56d9bee36300031aeea723b73c4d62ac7fa71a2",
        "weight_file": "model.safetensors",
        "weight_sha256": "d829d1a5cf66032491679d64c5b18e85b82d37833a99c346905668b8553084d5",
        "base_config_sha256": "76eb275107220e450d31258f792a2efcbee109d8b62ae0088260057dec06362f",
        "layer": 8,
        "teacher": RUNS / "teacher_rule_saturated/models/preference_teacher",
        "teacher_sha256": "324dea2aac4f151a39c443057df3ebcc8dc0bafc8470f4936a34ad2a2705420f",
        "teacher_model_config_sha256": "133f41d1b1ac76d1ec699ab0026e51b56c6f7cc7a80e2d55eb8317ae737e11f1",
        "teacher_data": RUNS / "teacher_rule_saturated/data/preference_teacher.jsonl",
        "origin": "retained",
        "expected": {"minus": -2.1795450846354165, "plus": 5.23319091796875},
        "expected_tolerance": 5e-4,
    },
    "weight_seed1": {
        "id": "EleutherAI/pythia-160m-weight-seed1",
        "commit": "36ea1a506902912e184f6b2ea590f9dab6bfe5e2",
        "weight_file": "pytorch_model.bin",
        "weight_sha256": "aef232c5545a0b81831f112b755872a3ba68d92c70e78252b4d909829daa2525",
        "base_config_sha256": "bada126c68fda35b615fec4674f274e948e6085442cdb368c21ffbbe18698e91",
        "layer": 9,
        "teacher": RECONSTRUCTION_ROOT / "weight_seed1/models/preference_teacher",
        "teacher_data": RECONSTRUCTION_ROOT / "weight_seed1/data/preference_teacher.jsonl",
        "teacher_config": ROOT / "configs/screen_teacher_weight-seed1.yaml",
        "teacher_config_sha256": "aabc73e79d82c52babd98daf5e4bfc8132140e0c33db8639194fcb06959d6c92",
        "archived_metrics": RUNS / "screen_teacher_weight-seed1/models/preference_teacher/training_metrics.json",
        "archived_metrics_sha256": "c3eb2cfa3fa5920736417ea804d5e8385733d887a95cb6dba96581777bb7df2a",
        "teacher_model_config_sha256": "133f41d1b1ac76d1ec699ab0026e51b56c6f7cc7a80e2d55eb8317ae737e11f1",
        "reconstruction_root": RECONSTRUCTION_ROOT / "weight_seed1",
        "origin": "recipe_reconstructed",
        "expected": {"minus": -2.4680989583333335, "plus": 5.5489461263020825},
        "expected_relative_tolerance": 0.15,
    },
    "weight_seed3": {
        "id": "EleutherAI/pythia-160m-weight-seed3",
        "commit": "e6b395cbbd654f940d63a45db501eca3ddba0548",
        "weight_file": "pytorch_model.bin",
        "weight_sha256": "82f3c4011d6f67b35a52f0af9c915760bf3aaf8c41087bc38f092c9dad33b1ff",
        "base_config_sha256": "bada126c68fda35b615fec4674f274e948e6085442cdb368c21ffbbe18698e91",
        "layer": 9,
        "teacher": RECONSTRUCTION_ROOT / "weight_seed3/models/preference_teacher",
        "teacher_data": RECONSTRUCTION_ROOT / "weight_seed3/data/preference_teacher.jsonl",
        "teacher_config": ROOT / "configs/screen_teacher_weight-seed3.yaml",
        "teacher_config_sha256": "1a6cc5e57b89b93ee3999c20d9d2934c529957af38e18c13f1ad01374d8ab4c4",
        "archived_metrics": RUNS / "screen_teacher_weight-seed3/models/preference_teacher/training_metrics.json",
        "archived_metrics_sha256": "a0cfee3287155c2ea5006d2cd53bfcfd0cfe37f9948e499f36ccbaa3088968f1",
        "teacher_model_config_sha256": "133f41d1b1ac76d1ec699ab0026e51b56c6f7cc7a80e2d55eb8317ae737e11f1",
        "reconstruction_root": RECONSTRUCTION_ROOT / "weight_seed3",
        "origin": "recipe_reconstructed",
        "expected": {"minus": -3.084503173828125, "plus": 5.8388514200846355},
        "expected_relative_tolerance": 0.15,
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().float().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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


def implementation_guard() -> dict[str, str]:
    return {
        "script_sha256": file_sha256(SCRIPT_PATH),
        "generate_py_sha256": file_sha256(GENERATE_PATH),
        "train_py_sha256": file_sha256(TRAIN_PATH),
        "optim_py_sha256": file_sha256(OPTIM_PATH),
        "pipeline_py_sha256": file_sha256(PIPELINE_PATH),
        "data_py_sha256": file_sha256(DATA_PATH),
        "config_py_sha256": file_sha256(CONFIG_PATH),
        "modeling_py_sha256": file_sha256(MODELING_PATH),
        "evaluate_py_sha256": file_sha256(EVALUATE_PATH),
        "weight_seed1_teacher_config_sha256": file_sha256(
            RECEIVERS["weight_seed1"]["teacher_config"]
        ),
        "weight_seed3_teacher_config_sha256": file_sha256(
            RECEIVERS["weight_seed3"]["teacher_config"]
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(DEVICE),
        "platform": platform.platform(),
    }


def protocol() -> dict[str, Any]:
    value = json.loads(PROTOCOL_PATH.read_text())
    expected = {
        name: {"model_id": info["id"], "layer": info["layer"]}
        for name, info in RECEIVERS.items()
    }
    observed = {
        name: {
            "model_id": row["model_id"],
            "layer": row["layer"],
        }
        for name, row in value["native_wolf_mode"]["receivers"].items()
    }
    if observed != expected:
        raise RuntimeError(f"Protocol receiver constants diverge from code: {observed}")
    if value["scope"]["contexts"] != ROWS:
        raise RuntimeError("Protocol row count diverges from code")
    if value["native_wolf_mode"]["alphas"] != list(ALPHAS):
        raise RuntimeError("Protocol alphas diverge from code")
    if value["native_wolf_mode"]["local_alpha"] != LOCAL_ALPHA:
        raise RuntimeError("Protocol local alpha diverges from code")
    prompt_spec = value["native_wolf_mode"]
    if (
        prompt_spec["extraction_prompt_count"] != len(PREFERENCE_TRAIN_PROMPTS)
        or prompt_spec["extraction_prompt_sha256"] != compact_hash(
            list(PREFERENCE_TRAIN_PROMPTS)
        )
        or prompt_spec["behavior_prompt_count"] != len(PREFERENCE_EVAL_PROMPTS)
        or prompt_spec["behavior_prompt_sha256"] != compact_hash(
            list(PREFERENCE_EVAL_PROMPTS)
        )
    ):
        raise RuntimeError("Frozen native-mode prompt identity diverges from code")
    if (
        prompt_spec["extraction_prompt_sha256"] != PREFERENCE_TRAIN_PROMPTS_SHA256
        or prompt_spec["behavior_prompt_sha256"] != PREFERENCE_EVAL_PROMPTS_SHA256
    ):
        raise RuntimeError("Prompt constants diverge from protocol")
    for name in ("weight_seed1", "weight_seed3"):
        receiver_spec = prompt_spec["receivers"][name]
        info = RECEIVERS[name]
        if (
            receiver_spec["teacher_config_sha256"]
            != file_sha256(info["teacher_config"])
            or receiver_spec["archived_metrics_sha256"]
            != file_sha256(info["archived_metrics"])
            or receiver_spec["reconstruction_base_commit"] != info["commit"]
            or receiver_spec["reconstruction_offline"] is not True
            or receiver_spec["behavior_regression_relative_tolerance"]
            != info["expected_relative_tolerance"]
        ):
            raise RuntimeError(f"Frozen reconstruction provenance diverges for {name}")
    sender = value["sender"]
    if (
        sender["preference_pool_sha256"] != PREFERENCE_POOL_SHA256
        or sender["control_pool_sha256"] != CONTROL_POOL_SHA256
        or sender["rows_per_condition"] != ROWS
    ):
        raise RuntimeError("Protocol sender constants diverge from code")
    training = value["prospective_stage"]["training"]
    if (
        training["student_seeds"] != list(PROSPECTIVE_SEEDS)
        or value["prospective_stage"]["receivers"] != list(PROSPECTIVE_RECEIVERS)
    ):
        raise RuntimeError("Prospective receiver/seed constants diverge from code")
    return value


def expected_endpoint_paths() -> list[Path]:
    return [
        PROSPECTIVE / receiver / f"seed_{seed}" / condition / "endpoint.json"
        for receiver in PROSPECTIVE_RECEIVERS
        for seed in PROSPECTIVE_SEEDS
        for condition in ("preference", "control")
    ]


def endpoint_absence_guard() -> dict[str, Any]:
    paths = expected_endpoint_paths()
    existing = [str(path.relative_to(ROOT)) for path in paths if path.exists()]
    namespace_entries = (
        [str(path.relative_to(ROOT)) for path in sorted(PROSPECTIVE.rglob("*"))]
        if PROSPECTIVE.exists()
        else []
    )
    return {
        "expected_paths": [str(path.relative_to(ROOT)) for path in paths],
        "expected_path_sha256": compact_hash(
            [str(path.relative_to(ROOT)) for path in paths]
        ),
        "all_absent": not existing and not namespace_entries,
        "existing": existing,
        "namespace_empty": not namespace_entries,
        "namespace_entries": namespace_entries,
    }


def initialize_snapshots(frozen: dict[str, Any]) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    if PROTOCOL_SNAPSHOT.exists():
        if json.loads(PROTOCOL_SNAPSHOT.read_text()) != frozen:
            raise RuntimeError("Protocol differs from immutable run snapshot")
    else:
        write_json(PROTOCOL_SNAPSHOT, frozen)
    implementation = implementation_guard()
    if IMPLEMENTATION_SNAPSHOT.exists():
        if json.loads(IMPLEMENTATION_SNAPSHOT.read_text()) != implementation:
            raise RuntimeError("Implementation differs from immutable run snapshot")
    else:
        write_json(IMPLEMENTATION_SNAPSHOT, implementation)


def cached_weight_guard(receiver: str) -> dict[str, Any]:
    info = RECEIVERS[receiver]
    cached = try_to_load_from_cache(info["id"], info["weight_file"], revision=REVISION)
    if not isinstance(cached, str):
        raise FileNotFoundError(f"Missing cached base weights for {receiver}")
    snapshot_path = Path(cached)
    if info["commit"] not in str(snapshot_path):
        raise RuntimeError(
            f"Unexpected cache revision for {receiver}: {snapshot_path}"
        )
    path = snapshot_path.resolve()
    observed = file_sha256(path)
    if observed != info["weight_sha256"]:
        raise RuntimeError(f"Base weight mismatch for {receiver}: {observed}")
    cached_config = try_to_load_from_cache(
        info["id"], "config.json", revision=REVISION
    )
    if not isinstance(cached_config, str):
        raise FileNotFoundError(f"Missing cached base config for {receiver}")
    config_snapshot_path = Path(cached_config)
    if info["commit"] not in str(config_snapshot_path):
        raise RuntimeError(
            f"Unexpected config revision for {receiver}: {config_snapshot_path}"
        )
    config_path = config_snapshot_path.resolve()
    config_sha256 = file_sha256(config_path)
    if config_sha256 != info["base_config_sha256"]:
        raise RuntimeError(f"Base config mismatch for {receiver}: {config_sha256}")
    return {
        "model_id": info["id"],
        "revision": REVISION,
        "resolved_commit": info["commit"],
        "weight_file": info["weight_file"],
        "weight_sha256": observed,
        "weight_bytes": path.stat().st_size,
        "model_config_sha256": config_sha256,
    }


def teacher_weight_path(receiver: str) -> Path:
    return RECEIVERS[receiver]["teacher"] / "model.safetensors"


def reconstruction_marker_path(receiver: str) -> Path:
    return RECEIVERS[receiver]["reconstruction_root"] / "reconstruction_marker.json"


def reconstruction_intent_path(receiver: str) -> Path:
    return RECEIVERS[receiver]["reconstruction_root"] / "reconstruction_intent.json"


def exact_reconstruction_config_path(receiver: str) -> Path:
    return RECEIVERS[receiver]["reconstruction_root"] / "exact_reconstruction_config.json"


def exact_reconstruction_config(receiver: str) -> dict[str, Any]:
    info = RECEIVERS[receiver]
    config = load_config(info["teacher_config"])
    config.pop("_config_path", None)
    config["model"]["revision"] = info["commit"]
    config["run"]["output_dir"] = str(info["reconstruction_root"])
    return config


def reconstruction_intent_record(receiver: str) -> dict[str, Any]:
    info = RECEIVERS[receiver]
    return {
        "receiver": receiver,
        "purpose": "run-owned native preference-teacher recipe replay",
        "reconstruction_root": str(info["reconstruction_root"].relative_to(ROOT)),
        "teacher_config_sha256": info["teacher_config_sha256"],
        "archived_metrics_sha256": info["archived_metrics_sha256"],
        "expected_teacher_data_sha256": PREFERENCE_TRAIN_DATA_SHA256,
        "expected_teacher_model_config_sha256": info[
            "teacher_model_config_sha256"
        ],
        "exact_reconstruction_config_sha256": compact_hash(
            exact_reconstruction_config(receiver)
        ),
        "base_weight_sha256": info["weight_sha256"],
        "base_model_config_sha256": info["base_config_sha256"],
        "implementation": implementation_guard(),
    }


def validate_reconstruction_intent(receiver: str) -> dict[str, Any]:
    info = RECEIVERS[receiver]
    root = info["reconstruction_root"]
    intent_path = reconstruction_intent_path(receiver)
    expected = reconstruction_intent_record(receiver)
    if not intent_path.exists():
        if root.exists() and any(root.iterdir()):
            raise RuntimeError(
                f"Refusing unowned partial reconstruction directory for {receiver}"
            )
        return expected
    observed = json.loads(intent_path.read_text())
    if observed != expected:
        raise RuntimeError(f"Reconstruction intent mismatch for {receiver}")
    exact_config_path = exact_reconstruction_config_path(receiver)
    if exact_config_path.exists():
        exact_config = json.loads(exact_config_path.read_text())
        if exact_config != exact_reconstruction_config(receiver):
            raise RuntimeError(f"Exact reconstruction config changed for {receiver}")
        if compact_hash(exact_config) != expected["exact_reconstruction_config_sha256"]:
            raise RuntimeError(f"Exact reconstruction config hash mismatch for {receiver}")
    data_path = info["teacher_data"]
    if data_path.exists() and file_sha256(data_path) != PREFERENCE_TRAIN_DATA_SHA256:
        raise RuntimeError(f"Partial reconstruction data mismatch for {receiver}")
    resolved_path = root / "resolved_config.json"
    if resolved_path.exists():
        if not exact_config_path.exists():
            raise RuntimeError(f"Resolved config exists before exact recipe for {receiver}")
        expected_config = exact_reconstruction_config(receiver)
        expected_config["_config_path"] = str(exact_config_path.resolve())
        if json.loads(resolved_path.read_text()) != expected_config:
            raise RuntimeError(f"Partial resolved recipe mismatch for {receiver}")
    return expected


def validate_reconstruction_files(receiver: str) -> dict[str, Any]:
    info = RECEIVERS[receiver]
    if info["origin"] != "recipe_reconstructed":
        raise ValueError(f"{receiver} is not a recipe reconstruction")
    config_path = info["teacher_config"]
    archived_metrics_path = info["archived_metrics"]
    if file_sha256(config_path) != info["teacher_config_sha256"]:
        raise RuntimeError(f"Reconstruction config changed for {receiver}")
    if file_sha256(archived_metrics_path) != info["archived_metrics_sha256"]:
        raise RuntimeError(f"Archived teacher metrics changed for {receiver}")
    root = info["reconstruction_root"]
    data_path = info["teacher_data"]
    weight_path = teacher_weight_path(receiver)
    metrics_path = info["teacher"] / "training_metrics.json"
    model_config_path = info["teacher"] / "config.json"
    resolved_path = root / "resolved_config.json"
    for path in (data_path, weight_path, metrics_path, model_config_path, resolved_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if file_sha256(data_path) != PREFERENCE_TRAIN_DATA_SHA256:
        raise RuntimeError(f"Reconstructed teacher data changed for {receiver}")
    if file_sha256(model_config_path) != info["teacher_model_config_sha256"]:
        raise RuntimeError(f"Reconstructed teacher model config changed for {receiver}")
    exact_config_path = exact_reconstruction_config_path(receiver)
    expected_config = exact_reconstruction_config(receiver)
    expected_config["_config_path"] = str(exact_config_path.resolve())
    observed_config = json.loads(resolved_path.read_text())
    if observed_config != expected_config:
        raise RuntimeError(f"Resolved reconstruction recipe changed for {receiver}")
    metrics = json.loads(metrics_path.read_text())
    archived_metrics = json.loads(archived_metrics_path.read_text())
    exact_metrics = {
        "examples": 384,
        "epochs": 1,
        "configured_epochs": 1,
        "completed_epochs": 1,
        "optimizer_updates": 24,
        "schedule_total_updates": 24,
        "warmup_updates": 1,
        "seed": 2101,
        "saved_model": True,
        "lora": None,
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1e-5,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
        },
    }
    for key, expected in exact_metrics.items():
        if metrics.get(key) != expected:
            raise RuntimeError(
                f"Reconstructed teacher metric {key} changed for {receiver}: "
                f"{metrics.get(key)!r} != {expected!r}"
            )
    loss_tolerance = info["expected_relative_tolerance"]
    for key in ("mean_microbatch_loss", "final_microbatch_loss"):
        reference = float(archived_metrics[key])
        observed = float(metrics[key])
        if abs(observed - reference) > loss_tolerance * abs(reference):
            raise RuntimeError(
                f"Reconstructed {receiver} {key} outside frozen "
                f"{loss_tolerance:.0%} replay tolerance"
            )
    return {
        "receiver": receiver,
        "origin": "run_owned_recipe_reconstruction_retained",
        "teacher_path": str(info["teacher"].relative_to(ROOT)),
        "teacher_config_sha256": info["teacher_config_sha256"],
        "exact_reconstruction_config_sha256": compact_hash(
            exact_reconstruction_config(receiver)
        ),
        "archived_metrics_sha256": info["archived_metrics_sha256"],
        "resolved_config_sha256": file_sha256(resolved_path),
        "teacher_data_sha256": file_sha256(data_path),
        "teacher_weight_sha256": file_sha256(weight_path),
        "teacher_weight_bytes": weight_path.stat().st_size,
        "teacher_model_config_sha256": file_sha256(model_config_path),
        "training_metrics_sha256": file_sha256(metrics_path),
        "loss_replay_relative_tolerance": loss_tolerance,
        "observed_mean_microbatch_loss": float(metrics["mean_microbatch_loss"]),
        "observed_final_microbatch_loss": float(metrics["final_microbatch_loss"]),
        "implementation": implementation_guard(),
    }


def teacher_guard(receiver: str, require_weight: bool = True) -> dict[str, Any]:
    info = RECEIVERS[receiver]
    if info["origin"] == "recipe_reconstructed":
        marker_path = reconstruction_marker_path(receiver)
        intent_path = reconstruction_intent_path(receiver)
        weight = teacher_weight_path(receiver)
        if not weight.exists():
            if require_weight:
                raise FileNotFoundError(weight)
            if marker_path.exists():
                raise RuntimeError(f"Reconstruction marker exists without weight: {receiver}")
            if file_sha256(info["teacher_config"]) != info["teacher_config_sha256"]:
                raise RuntimeError(f"Reconstruction config changed for {receiver}")
            if file_sha256(info["archived_metrics"]) != info["archived_metrics_sha256"]:
                raise RuntimeError(f"Archived teacher metrics changed for {receiver}")
            validate_reconstruction_intent(receiver)
            return {
                "teacher_path": str(info["teacher"].relative_to(ROOT)),
                "origin": (
                    "interrupted_run_owned_recipe_reconstruction"
                    if intent_path.exists()
                    else "planned_run_owned_recipe_reconstruction"
                ),
                "teacher_config_sha256": info["teacher_config_sha256"],
                "archived_metrics_sha256": info["archived_metrics_sha256"],
            }
        if not marker_path.exists():
            validate_reconstruction_intent(receiver)
            if require_weight:
                raise RuntimeError(
                    f"Unfinished reconstructed teacher requires ensure_teacher: {receiver}"
                )
            return {
                "teacher_path": str(info["teacher"].relative_to(ROOT)),
                "origin": "interrupted_run_owned_recipe_reconstruction",
                "teacher_config_sha256": info["teacher_config_sha256"],
                "archived_metrics_sha256": info["archived_metrics_sha256"],
                "unmarked_weight_present": True,
            }
        validate_reconstruction_intent(receiver)
        observed = validate_reconstruction_files(receiver)
        marker = json.loads(marker_path.read_text())
        if marker != observed:
            raise RuntimeError(f"Reconstruction marker mismatch for {receiver}")
        return observed
    data_path = info["teacher_data"]
    if file_sha256(data_path) != PREFERENCE_TRAIN_DATA_SHA256:
        raise RuntimeError(f"Teacher training data changed for {receiver}")
    weight = teacher_weight_path(receiver)
    model_config_path = info["teacher"] / "config.json"
    if require_weight and not weight.exists():
        raise FileNotFoundError(weight)
    if require_weight and not model_config_path.exists():
        raise FileNotFoundError(model_config_path)
    record: dict[str, Any] = {
        "teacher_path": str(info["teacher"].relative_to(ROOT)),
        "teacher_data_sha256": PREFERENCE_TRAIN_DATA_SHA256,
        "origin": info["origin"],
    }
    if weight.exists():
        observed = file_sha256(weight)
        expected = info.get("teacher_sha256")
        if expected is not None and observed != expected:
            raise RuntimeError(f"Retained teacher hash changed for {receiver}: {observed}")
        record.update({
            "teacher_weight_sha256": observed,
            "teacher_weight_bytes": weight.stat().st_size,
            "teacher_model_config_sha256": file_sha256(model_config_path),
        })
        if record["teacher_model_config_sha256"] != info["teacher_model_config_sha256"]:
            raise RuntimeError(f"Teacher model config changed for {receiver}")
    return record


def ensure_teacher(receiver: str) -> dict[str, Any]:
    info = RECEIVERS[receiver]
    if info["origin"] != "recipe_reconstructed":
        return teacher_guard(receiver)
    weight = teacher_weight_path(receiver)
    marker_path = reconstruction_marker_path(receiver)
    intent_path = reconstruction_intent_path(receiver)
    if marker_path.exists():
        return teacher_guard(receiver)
    validate_reconstruction_intent(receiver)
    if weight.exists():
        try:
            record = validate_reconstruction_files(receiver)
        except (FileNotFoundError, RuntimeError, json.JSONDecodeError):
            record = None
        if record is not None:
            write_json(marker_path, record)
            return teacher_guard(receiver)
    config = exact_reconstruction_config_path(receiver)
    root = info["reconstruction_root"]
    if not intent_path.exists():
        write_json(intent_path, reconstruction_intent_record(receiver))
    expected_exact_config = exact_reconstruction_config(receiver)
    if config.exists():
        if json.loads(config.read_text()) != expected_exact_config:
            raise RuntimeError(f"Exact reconstruction config changed for {receiver}")
    else:
        write_json(config, expected_exact_config)
    validate_reconstruction_intent(receiver)
    print(f"[{receiver}] reconstructing native preference teacher in run-owned path", flush=True)
    offline_environment = os.environ.copy()
    offline_environment["HF_HUB_OFFLINE"] = "1"
    offline_environment["TRANSFORMERS_OFFLINE"] = "1"
    subprocess.run(
        [sys.executable, "-m", "polypythia_sl.pipeline", "--config", str(config),
         "--stage", "teacher", "--output-dir", str(root), "--force"],
        cwd=ROOT,
        env=offline_environment,
        check=True,
    )
    record = validate_reconstruction_files(receiver)
    write_json(marker_path, record)
    return teacher_guard(receiver)


def reclaim_reconstructed_teacher(receiver: str) -> None:
    # Recipe-reconstructed teachers are retained in this experiment's own
    # namespace so their exact tensor hashes remain independently auditable.
    return


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        RECEIVERS["ds2"]["id"], revision=RECEIVERS["ds2"]["commit"],
        local_files_only=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenization_guard(tokenizer) -> dict[str, Any]:
    allowed_ids, allowed_values = _whole_number_tokens(tokenizer, 999)
    decoded = [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False,
                         skip_special_tokens=False)
        for token_id in allowed_ids
    ]
    ordered = [
        {"token_id": token_id, "text": text, "value": value}
        for token_id, text, value in zip(allowed_ids, decoded, allowed_values)
    ]
    if len(allowed_ids) != 655:
        raise RuntimeError(f"Expected 655 allowed numeric token IDs, got {len(allowed_ids)}")
    if len(set(allowed_values)) != 644:
        raise RuntimeError(
            f"Expected 644 distinct numeric values, got {len(set(allowed_values))}"
        )
    reference_animals = [
        tokenizer.encode(" " + animal, add_special_tokens=False)
        for animal in ANIMALS
    ]
    if any(len(ids) != 1 for ids in reference_animals):
        raise RuntimeError(f"An animal is not a single token: {reference_animals}")
    reference_train_prompts = [
        tokenizer.encode(prompt, add_special_tokens=False)
        for prompt in PREFERENCE_TRAIN_PROMPTS
    ]
    reference_eval_prompts = [
        tokenizer.encode(prompt, add_special_tokens=False)
        for prompt in PREFERENCE_EVAL_PROMPTS
    ]
    for receiver, info in RECEIVERS.items():
        other = AutoTokenizer.from_pretrained(
            info["id"], revision=info["commit"], local_files_only=True
        )
        other_ids, other_values = _whole_number_tokens(other, 999)
        if other_ids != allowed_ids or other_values != allowed_values:
            raise RuntimeError(f"Numeric token map mismatch for {receiver}")
        other_animals = [
            other.encode(" " + animal, add_special_tokens=False)
            for animal in ANIMALS
        ]
        other_train_prompts = [
            other.encode(prompt, add_special_tokens=False)
            for prompt in PREFERENCE_TRAIN_PROMPTS
        ]
        other_eval_prompts = [
            other.encode(prompt, add_special_tokens=False)
            for prompt in PREFERENCE_EVAL_PROMPTS
        ]
        if (
            other_animals != reference_animals
            or other_train_prompts != reference_train_prompts
            or other_eval_prompts != reference_eval_prompts
        ):
            raise RuntimeError(f"Prompt/animal tokenization mismatch for {receiver}")
    return {
        "allowed_token_count": len(allowed_ids),
        "distinct_value_count": len(set(allowed_values)),
        "ordered_token_map_sha256": compact_hash(ordered),
        "ordered_token_map": ordered,
        "animal_token_ids": [ids[0] for ids in reference_animals],
        "animal_token_ids_sha256": compact_hash(reference_animals),
        "train_prompt_token_ids_sha256": compact_hash(reference_train_prompts),
        "eval_prompt_token_ids_sha256": compact_hash(reference_eval_prompts),
    }


def load_contexts(tokenizer) -> tuple[list[list[int]], dict[str, Any]]:
    observed = {
        "preference": file_sha256(PREFERENCE_POOL),
        "control": file_sha256(CONTROL_POOL),
    }
    expected = {
        "preference": PREFERENCE_POOL_SHA256,
        "control": CONTROL_POOL_SHA256,
    }
    if observed != expected:
        raise RuntimeError(f"Pool hash guard failed: {observed}")
    preference = read_jsonl(PREFERENCE_POOL)
    control = read_jsonl(CONTROL_POOL)
    if len(preference) != ROWS or len(control) != ROWS:
        raise RuntimeError("Pool row-count guard failed")
    pref_prompts = [row["prompt"] for row in preference]
    ctrl_prompts = [row["prompt"] for row in control]
    if pref_prompts != ctrl_prompts:
        raise RuntimeError("Preference/control prompts are not byte-identical and paired")
    ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in pref_prompts]
    if any(not row for row in ids):
        raise RuntimeError("An empty numeric context was found")
    return ids, {
        "pool_sha256": observed,
        "rows": ROWS,
        "paired_prompts": True,
        "prompt_text_sha256": compact_hash(pref_prompts),
        "prompt_token_ids_sha256": compact_hash(ids),
        "token_lengths": sorted(set(map(len, ids))),
    }


def load_base(receiver: str):
    info = RECEIVERS[receiver]
    model = AutoModelForCausalLM.from_pretrained(
        info["id"], revision=info["commit"], torch_dtype=torch.float32,
        local_files_only=True,
    )
    return model.to(DEVICE).eval()


def load_teacher(receiver: str):
    return AutoModelForCausalLM.from_pretrained(
        RECEIVERS[receiver]["teacher"], torch_dtype=torch.float32,
        local_files_only=True,
    ).to(DEVICE).eval()


@torch.inference_mode()
def restricted_log_probs(
    model,
    contexts: list[list[int]],
    tokenizer,
    allowed_ids: list[int],
    *,
    vector: torch.Tensor | None = None,
    layer: int | None = None,
    alpha: float = 0.0,
    normalization: str = "restricted",
    label: str,
) -> torch.Tensor:
    if normalization not in {"restricted", "full_vocab"}:
        raise ValueError(f"Unsupported log-probability normalization: {normalization}")
    handle = None
    if vector is not None and alpha != 0.0:
        if layer is None:
            raise ValueError("A steering layer is required with a vector")
        intervention = (alpha * vector).to(DEVICE)

        def hook(module, inputs, output):
            return (output[0] + intervention, *output[1:])

        handle = model.gpt_neox.layers[layer - 1].register_forward_hook(hook)
    selected = torch.tensor(allowed_ids, dtype=torch.long, device=DEVICE)
    chunks = []
    try:
        total_batches = math.ceil(len(contexts) / BATCH_SIZE)
        for batch_i, start in enumerate(range(0, len(contexts), BATCH_SIZE)):
            batch = contexts[start:start + BATCH_SIZE]
            input_ids, attention_mask = _right_padded_batch(
                batch, tokenizer.pad_token_id, DEVICE
            )
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            last = attention_mask.sum(1) - 1
            indices = torch.arange(len(batch), device=DEVICE)
            full_logits = output.logits[indices, last].float()
            if normalization == "restricted":
                selected_log_probs = torch.log_softmax(
                    full_logits[:, selected], dim=-1
                )
            else:
                selected_log_probs = torch.log_softmax(
                    full_logits, dim=-1
                )[:, selected]
            chunks.append(selected_log_probs.cpu())
            if (batch_i + 1) % 64 == 0 or batch_i + 1 == total_batches:
                print(
                    f"[{label}] {batch_i + 1}/{total_batches} batches",
                    flush=True,
                )
    finally:
        if handle is not None:
            handle.remove()
    result = torch.cat(chunks).contiguous()
    if result.shape != (ROWS, len(allowed_ids)):
        raise RuntimeError(f"Unexpected log-probability shape: {result.shape}")
    if not torch.isfinite(result).all():
        raise RuntimeError(f"Non-finite log probabilities in {label}")
    return result


@torch.inference_mode()
def last_token_acts(model, tokenizer, prompts: list[str], layer: int) -> torch.Tensor:
    chunks = []
    for start in range(0, len(prompts), BEHAVIOR_BATCH_SIZE):
        batch = prompts[start:start + BEHAVIOR_BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(DEVICE)
        hidden = model(
            **enc, output_hidden_states=True, use_cache=False
        ).hidden_states[layer]
        last = enc["attention_mask"].sum(1) - 1
        indices = torch.arange(len(batch), device=DEVICE)
        chunks.append(hidden[indices, last].float().cpu())
    return torch.cat(chunks).contiguous()


@torch.inference_mode()
def behavior_cell(
    model,
    tokenizer,
    animal_ids: list[int],
    *,
    vector: torch.Tensor | None,
    layer: int,
    alpha: float,
) -> dict[str, Any]:
    handle = None
    if vector is not None and alpha != 0.0:
        intervention = (alpha * vector).to(DEVICE)

        def hook(module, inputs, output):
            return (output[0] + intervention, *output[1:])

        handle = model.gpt_neox.layers[layer - 1].register_forward_hook(hook)
    selected = torch.tensor(animal_ids, dtype=torch.long, device=DEVICE)
    margins = []
    nll_sum = 0.0
    nll_tokens = 0
    try:
        for start in range(0, len(PREFERENCE_EVAL_PROMPTS), BEHAVIOR_BATCH_SIZE):
            batch = PREFERENCE_EVAL_PROMPTS[start:start + BEHAVIOR_BATCH_SIZE]
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(DEVICE)
            logits = model(**enc, use_cache=False).logits
            last = enc["attention_mask"].sum(1) - 1
            indices = torch.arange(len(batch), device=DEVICE)
            chosen = logits[indices, last][:, selected].float()
            margin = (
                chosen[:, 0] - torch.logsumexp(chosen[:, 1:], dim=1)
                + float(np.log(len(animal_ids) - 1))
            )
            margins.extend(margin.cpu().tolist())
            shifted = logits[:, :-1]
            labels = enc["input_ids"][:, 1:].clone()
            labels[enc["attention_mask"][:, 1:] == 0] = -100
            nll_sum += float(torch.nn.functional.cross_entropy(
                shifted.reshape(-1, shifted.shape[-1]), labels.reshape(-1),
                ignore_index=-100, reduction="sum",
            ))
            nll_tokens += int((labels != -100).sum())
    finally:
        if handle is not None:
            handle.remove()
    return {
        "alpha": alpha,
        "wolf_margin": float(np.mean(margins)),
        "wolf_margins": margins,
        "prompt_nll": nll_sum / nll_tokens,
    }


def vector_artifact_path(receiver: str) -> Path:
    return VECTORS / f"{receiver}.pt"


def save_vector(receiver: str, vector: torch.Tensor, record: dict[str, Any]) -> None:
    VECTORS.mkdir(parents=True, exist_ok=True)
    path = vector_artifact_path(receiver)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"vector": vector.float().cpu().contiguous(), "record": record}, temporary)
    temporary.replace(path)


def validate_vector_artifact(receiver: str, expected: dict[str, Any]) -> torch.Tensor:
    path = vector_artifact_path(receiver)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    vector = payload["vector"].float().contiguous()
    record = payload["record"]
    if record != expected:
        raise RuntimeError(f"Vector metadata mismatch for {receiver}")
    if tensor_sha256(vector) != record["tensor_sha256"]:
        raise RuntimeError(f"Vector tensor hash mismatch for {receiver}")
    return vector


def score_rows(
    fingerprint: torch.Tensor,
    response: torch.Tensor,
) -> dict[str, Any]:
    f = fingerprint.double()
    s = response.double()
    per_row = torch.sum(f * s, dim=1).numpy()
    raw = float(np.mean(per_row))
    halves = {
        "contexts_1_4096": float(np.mean(per_row[:4096])),
        "contexts_4097_8192": float(np.mean(per_row[4096:])),
    }
    flat_dot = float(torch.sum(f * s))
    flat_denominator = float(torch.linalg.vector_norm(f) * torch.linalg.vector_norm(s))
    cosine = flat_dot / flat_denominator if flat_denominator > 0 else float("nan")
    f_bar = f.mean(0)
    s_bar = s.mean(0)
    marginal = float(torch.sum(f_bar * s_bar))
    conditional = raw - marginal
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_SAMPLES, 250):
        size = min(250, BOOTSTRAP_SAMPLES - start)
        indices = rng.integers(0, ROWS, size=(size, ROWS))
        means[start:start + size] = per_row[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    sign_reversed = float(np.mean(np.sum(f.numpy() * (-s.numpy()), axis=1)))
    if not math.isclose(sign_reversed, -raw, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Sign-reversal identity failed")
    zero = float(torch.sum(torch.zeros_like(f) * s))
    if zero != 0.0:
        raise RuntimeError("Zero-fingerprint identity failed")
    return {
        "raw_score": raw,
        "halves": halves,
        "flattened_cosine": cosine,
        "marginal_token_component": marginal,
        "prompt_conditional_component": conditional,
        "bootstrap_95": {
            "method": "row bootstrap percentile interval",
            "seed": BOOTSTRAP_SEED,
            "samples": BOOTSTRAP_SAMPLES,
            "low": float(low),
            "high": float(high),
        },
        "per_row_mean": raw,
        "per_row_sd": float(np.std(per_row, ddof=1)),
        "sign_reversal_identity": sign_reversed,
        "zero_fingerprint_identity": zero,
    }


def behavior_normalize_score(
    score: dict[str, Any], wolf_margin_slope: float
) -> dict[str, Any]:
    if not math.isfinite(wolf_margin_slope) or wolf_margin_slope <= 0:
        raise RuntimeError("Behavior normalization requires a positive finite slope")
    interval = score["bootstrap_95"]
    return {
        "K": score["raw_score"] / wolf_margin_slope,
        "halves": {
            key: value / wolf_margin_slope
            for key, value in score["halves"].items()
        },
        "bootstrap_95": {
            **interval,
            "low": interval["low"] / wolf_margin_slope,
            "high": interval["high"] / wolf_margin_slope,
        },
        "wolf_margin_slope_G": wolf_margin_slope,
        "definition": "K = relative C / local held-out wolf-margin slope G",
    }


def assert_projection_identity(
    relative: dict[str, Any],
    preference: dict[str, Any],
    control: dict[str, Any],
    *,
    label: str,
) -> None:
    pairs = [("full", relative["raw_score"], preference["raw_score"] - control["raw_score"])]
    pairs.extend(
        (
            half,
            relative["halves"][half],
            preference["halves"][half] - control["halves"][half],
        )
        for half in relative["halves"]
    )
    for part, observed, expected in pairs:
        if not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-9):
            raise RuntimeError(
                f"{label} projection identity failed for {part}: "
                f"{observed} != {expected}"
            )


def fingerprint_statistics(
    q_preference: torch.Tensor,
    q_control: torch.Tensor,
    token_map: list[dict[str, Any]],
) -> dict[str, Any]:
    qp = q_preference.double()
    qc = q_control.double()
    if not torch.allclose(qp.sum(1), torch.ones(ROWS, dtype=torch.float64), atol=2e-6):
        raise RuntimeError("Preference fingerprint rows do not sum to one")
    if not torch.allclose(qc.sum(1), torch.ones(ROWS, dtype=torch.float64), atol=2e-6):
        raise RuntimeError("Control fingerprint rows do not sum to one")
    qp_safe = qp.clamp_min(1e-30)
    qc_safe = qc.clamp_min(1e-30)
    midpoint = 0.5 * (qp_safe + qc_safe)
    tv = 0.5 * torch.sum(torch.abs(qp - qc), dim=1)
    js = 0.5 * torch.sum(qp * (torch.log(qp_safe) - torch.log(midpoint)), dim=1)
    js += 0.5 * torch.sum(qc * (torch.log(qc_safe) - torch.log(midpoint)), dim=1)
    mean_shift = (qp - qc).mean(0)
    order = torch.argsort(mean_shift)
    def token_record(index: int) -> dict[str, Any]:
        return {**token_map[index], "mean_probability_shift": float(mean_shift[index])}
    return {
        "shape": list(qp.shape),
        "q_preference_tensor_sha256": tensor_sha256(q_preference),
        "q_control_tensor_sha256": tensor_sha256(q_control),
        "mean_total_variation": float(tv.mean()),
        "mean_jensen_shannon_nats": float(js.mean()),
        "mean_shift_l2": float(torch.linalg.vector_norm(mean_shift)),
        "maximum_absolute_mass_error": max(
            float(torch.max(torch.abs(qp.sum(1) - 1))),
            float(torch.max(torch.abs(qc.sum(1) - 1))),
        ),
        "largest_positive_mean_shifts": [token_record(int(i)) for i in order[-10:].flip(0)],
        "largest_negative_mean_shifts": [token_record(int(i)) for i in order[:10]],
    }


def load_sender_fingerprint(
    tokenizer,
    contexts: list[list[int]],
    allowed_ids: list[int],
    token_map: list[dict[str, Any]],
    token_guard: dict[str, Any],
    context_guard: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    current_identity = {
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "implementation": implementation_guard(),
        "sender_base": cached_weight_guard("ds2"),
        "sender_teacher": teacher_guard("ds2"),
        "context_prompt_text_sha256": context_guard["prompt_text_sha256"],
        "context_prompt_token_ids_sha256": context_guard["prompt_token_ids_sha256"],
        "numeric_token_map_sha256": token_guard["ordered_token_map_sha256"],
    }
    if FINGERPRINT_TENSOR_PATH.exists() and FINGERPRINT_META_PATH.exists():
        meta = json.loads(FINGERPRINT_META_PATH.read_text())
        observed_identity = {key: meta.get(key) for key in current_identity}
        if observed_identity != current_identity:
            raise RuntimeError("Cached sender fingerprint belongs to another protocol")
        payload = torch.load(
            FINGERPRINT_TENSOR_PATH, map_location="cpu", weights_only=True
        )
        qp = payload["q_preference"].float().contiguous()
        qc = payload["q_control"].float().contiguous()
        if qp.shape != (ROWS, len(allowed_ids)) or qc.shape != qp.shape:
            raise RuntimeError("Cached sender tensor shape mismatch")
        if not torch.isfinite(qp).all() or not torch.isfinite(qc).all():
            raise RuntimeError("Cached sender tensor contains non-finite values")
        stats = fingerprint_statistics(qp, qc, token_map)
        if stats != meta["statistics"]:
            raise RuntimeError("Cached sender fingerprint metadata mismatch")
        print("Reusing guarded sender fingerprint", flush=True)
        return qp, qc, meta

    print("[sender] computing ds2 base restricted distribution", flush=True)
    base = load_base("ds2")
    log_qc = restricted_log_probs(
        base, contexts, tokenizer, allowed_ids, label="sender/base"
    )
    release(base)
    print("[sender] computing ds2 wolf-teacher restricted distribution", flush=True)
    teacher = load_teacher("ds2")
    log_qp = restricted_log_probs(
        teacher, contexts, tokenizer, allowed_ids, label="sender/wolf"
    )
    release(teacher)
    qp = torch.exp(log_qp).float().contiguous()
    qc = torch.exp(log_qc).float().contiguous()
    statistics = fingerprint_statistics(qp, qc, token_map)
    temporary = FINGERPRINT_TENSOR_PATH.with_name(FINGERPRINT_TENSOR_PATH.name + ".tmp")
    torch.save({"q_preference": qp, "q_control": qc}, temporary)
    temporary.replace(FINGERPRINT_TENSOR_PATH)
    meta = {
        **current_identity,
        "statistics": statistics,
    }
    write_json(FINGERPRINT_META_PATH, meta)
    print("SENDER FINGERPRINT DONE", flush=True)
    return qp, qc, meta


def expected_vector_record(
    receiver: str,
    vector: torch.Tensor,
    teacher_info: dict[str, Any],
    mean_prompt_difference_norm: float,
) -> dict[str, Any]:
    return {
        "receiver": receiver,
        "model_id": RECEIVERS[receiver]["id"],
        "resolved_commit": RECEIVERS[receiver]["commit"],
        "layer": RECEIVERS[receiver]["layer"],
        "train_prompts_sha256": compact_hash(list(PREFERENCE_TRAIN_PROMPTS)),
        "teacher_weight_sha256": teacher_info["teacher_weight_sha256"],
        "shape": list(vector.shape),
        "norm": float(vector.norm()),
        "tensor_sha256": tensor_sha256(vector),
        "mean_prompt_difference_norm": mean_prompt_difference_norm,
    }


def validate_behavior(
    receiver: str,
    cells: dict[str, dict[str, Any]],
    teacher_margin: float,
) -> dict[str, Any]:
    base = cells["0"]
    for cell in cells.values():
        cell["wolf_delta"] = cell["wolf_margin"] - base["wolf_margin"]
        cell["nll_ratio"] = cell["prompt_nll"] / base["prompt_nll"]
    minus = cells["-1"]
    plus = cells["+1"]
    local_minus = cells["-0.25"]
    local_plus = cells["+0.25"]
    local_prompt_slopes = (
        np.asarray(local_plus["wolf_margins"])
        - np.asarray(local_minus["wolf_margins"])
    ) / (2 * LOCAL_ALPHA)
    local_slope = float(np.mean(local_prompt_slopes))
    local_halves = {
        "prompts_1_30": float(np.mean(local_prompt_slopes[:30])),
        "prompts_31_60": float(np.mean(local_prompt_slopes[30:])),
    }
    expected = RECEIVERS[receiver]["expected"]
    receiver_info = RECEIVERS[receiver]
    if receiver_info["origin"] == "recipe_reconstructed":
        relative_tolerance = receiver_info["expected_relative_tolerance"]
        tolerances = {
            key: relative_tolerance * abs(value) for key, value in expected.items()
        }
        tolerance_record: dict[str, Any] = {
            "kind": "relative_to_archived_absolute_delta",
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance_by_cell": tolerances,
        }
    else:
        tolerances = {key: receiver_info["expected_tolerance"] for key in expected}
        tolerance_record = {
            "kind": "absolute",
            "absolute_tolerance": receiver_info["expected_tolerance"],
        }
    reproduction = {
        "expected": expected,
        "observed": {"minus": minus["wolf_delta"], "plus": plus["wolf_delta"]},
        "tolerance": tolerance_record,
    }
    reproduction["pass"] = all(
        abs(reproduction["observed"][key] - expected[key]) <= tolerances[key]
        for key in ("minus", "plus")
    )
    checks = {
        "sign_control": minus["wolf_delta"] < 0 < plus["wolf_delta"],
        "local_slope_positive_full": local_slope > 0,
        "local_slope_positive_both_prompt_halves": all(v > 0 for v in local_halves.values()),
        "plus_quality_nll_below_1p2": plus["nll_ratio"] < 1.2,
        "archived_cell_reproduction": reproduction["pass"],
        "teacher_has_positive_behavioral_contrast": teacher_margin > base["wolf_margin"],
    }
    return {
        "cells": cells,
        "teacher_wolf_margin": teacher_margin,
        "teacher_behavioral_contrast": teacher_margin - base["wolf_margin"],
        "local_wolf_margin_slope": local_slope,
        "local_wolf_margin_slope_halves": local_halves,
        "archived_reproduction": reproduction,
        "checks": checks,
        "pass": all(checks.values()),
    }


def assert_finite_tree(value: Any, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_tree(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite_tree(child, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"Non-finite cached value at {path}: {value}")


def sender_score_identity(
    sender_meta: dict[str, Any],
    token_guard: dict[str, Any],
    context_guard: dict[str, Any],
) -> dict[str, Any]:
    stats = sender_meta["statistics"]
    return {
        "q_preference_tensor_sha256": stats["q_preference_tensor_sha256"],
        "q_control_tensor_sha256": stats["q_control_tensor_sha256"],
        "numeric_token_map_sha256": token_guard["ordered_token_map_sha256"],
        "context_prompt_text_sha256": context_guard["prompt_text_sha256"],
        "context_prompt_token_ids_sha256": context_guard["prompt_token_ids_sha256"],
    }


def validate_cached_score(
    receiver: str,
    record: dict[str, Any],
    sender_meta: dict[str, Any],
    token_guard: dict[str, Any],
    context_guard: dict[str, Any],
) -> None:
    weight_info = cached_weight_guard(receiver)
    teacher_info = teacher_guard(receiver)
    expected = {
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "implementation": implementation_guard(),
        "receiver": receiver,
        "model_id": RECEIVERS[receiver]["id"],
        "resolved_commit": RECEIVERS[receiver]["commit"],
        "layer": RECEIVERS[receiver]["layer"],
        "base_weight_sha256": weight_info["weight_sha256"],
        "teacher": teacher_info,
        "sender_identity": sender_score_identity(
            sender_meta, token_guard, context_guard
        ),
    }
    observed = {key: record.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"Cached score identity mismatch for {receiver}")
    vector = validate_vector_artifact(receiver, record["vector"])
    if tensor_sha256(vector) != record["vector"]["tensor_sha256"]:
        raise RuntimeError(f"Cached score vector mismatch for {receiver}")
    if not record.get("native_behavior_validation", {}).get("pass"):
        raise RuntimeError(f"Cached behavior validation failed for {receiver}")
    relative = record["primary_local_compatibility"]
    preference = record["local_absolute_preference_projection"]
    control = record["local_absolute_control_projection"]
    assert_projection_identity(
        relative, preference, control, label=f"cached {receiver} local"
    )
    expected_normalized = behavior_normalize_score(
        relative,
        record["native_behavior_validation"]["local_wolf_margin_slope"],
    )
    if record.get("primary_behavior_normalized_compatibility") != expected_normalized:
        raise RuntimeError(f"Cached normalized score mismatch for {receiver}")
    for score_name in (
        "primary_local_compatibility",
        "primary_behavior_normalized_compatibility",
        "local_absolute_preference_projection",
        "local_absolute_control_projection",
        "local_positive_absolute_preference_projection",
        "finite_alpha_compatibility",
        "one_sided_plus_compatibility",
        "native_teacher_fingerprint_compatibility",
    ):
        if score_name not in record:
            raise RuntimeError(f"Cached score missing {score_name} for {receiver}")
    assert_finite_tree(record)


def score_receiver(
    receiver: str,
    tokenizer,
    contexts: list[list[int]],
    allowed_ids: list[int],
    q_sender_preference: torch.Tensor,
    q_sender_control: torch.Tensor,
    sender_meta: dict[str, Any],
    token_guard: dict[str, Any],
    context_guard: dict[str, Any],
) -> dict[str, Any]:
    destination = SCORES / f"{receiver}.json"
    if destination.exists():
        record = json.loads(destination.read_text())
        validate_cached_score(
            receiver, record, sender_meta, token_guard, context_guard
        )
        print(f"[{receiver}] reusing guarded compatibility score", flush=True)
        return record
    teacher_info = ensure_teacher(receiver)
    weight_info = cached_weight_guard(receiver)
    info = RECEIVERS[receiver]
    layer = info["layer"]
    animal_ids = [
        tokenizer.encode(" " + animal, add_special_tokens=False)[0]
        for animal in ANIMALS
    ]

    print(f"[{receiver}] extracting native L{layer} wolf direction", flush=True)
    base = load_base(receiver)
    base_acts = last_token_acts(base, tokenizer, list(PREFERENCE_TRAIN_PROMPTS), layer)
    base_behavior = behavior_cell(
        base, tokenizer, animal_ids, vector=None, layer=layer, alpha=0.0
    )
    teacher = load_teacher(receiver)
    teacher_acts = last_token_acts(
        teacher, tokenizer, list(PREFERENCE_TRAIN_PROMPTS), layer
    )
    teacher_behavior = behavior_cell(
        teacher, tokenizer, animal_ids, vector=None, layer=layer, alpha=0.0
    )
    vector = (teacher_acts - base_acts).mean(0).float().contiguous()
    vector_record = expected_vector_record(
        receiver,
        vector,
        teacher_info,
        float((teacher_acts - base_acts).norm(dim=1).mean()),
    )
    save_vector(receiver, vector, vector_record)
    vector = validate_vector_artifact(receiver, vector_record)
    del teacher_acts, base_acts

    print(f"[{receiver}] native teacher numeric fingerprint", flush=True)
    log_teacher = restricted_log_probs(
        teacher, contexts, tokenizer, allowed_ids,
        normalization="full_vocab",
        label=f"{receiver}/native-teacher",
    )
    release(teacher)

    behavior_cells: dict[str, dict[str, Any]] = {"0": base_behavior}
    log_probs: dict[float, torch.Tensor] = {}
    for alpha in ALPHAS:
        key = f"{alpha:+g}" if alpha else "0"
        if alpha != 0.0:
            behavior_cells[key] = behavior_cell(
                base, tokenizer, animal_ids, vector=vector, layer=layer, alpha=alpha
            )
        log_probs[alpha] = restricted_log_probs(
            base, contexts, tokenizer, allowed_ids,
            vector=vector if alpha != 0.0 else None,
            layer=layer,
            alpha=alpha,
            normalization="full_vocab",
            label=f"{receiver}/alpha{alpha:+g}",
        )
    release(base)

    behavior = validate_behavior(
        receiver, behavior_cells, teacher_behavior["wolf_margin"]
    )
    fingerprint = (q_sender_preference - q_sender_control).float().contiguous()
    local_response = (
        log_probs[LOCAL_ALPHA] - log_probs[-LOCAL_ALPHA]
    ) / (2 * LOCAL_ALPHA)
    finite_response = (
        log_probs[FINITE_ALPHA] - log_probs[-FINITE_ALPHA]
    ) / (2 * FINITE_ALPHA)
    local_positive_response = (
        log_probs[LOCAL_ALPHA] - log_probs[0.0]
    ) / LOCAL_ALPHA
    one_sided_response = log_probs[FINITE_ALPHA] - log_probs[0.0]
    teacher_response = log_teacher - log_probs[0.0]

    primary = score_rows(fingerprint, local_response)
    absolute_preference = score_rows(q_sender_preference, local_response)
    absolute_control = score_rows(q_sender_control, local_response)
    local_positive_absolute_preference = score_rows(
        q_sender_preference, local_positive_response
    )
    assert_projection_identity(
        primary, absolute_preference, absolute_control,
        label=f"{receiver} local",
    )
    finite = score_rows(fingerprint, finite_response)
    one_sided = score_rows(fingerprint, one_sided_response)
    teacher_score = score_rows(fingerprint, teacher_response)
    primary["linearity_finite_over_local"] = (
        finite["raw_score"] / primary["raw_score"]
        if abs(primary["raw_score"]) > 1e-20 else None
    )
    normalized = behavior_normalize_score(
        primary, behavior["local_wolf_margin_slope"]
    )

    record = {
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "implementation": implementation_guard(),
        "receiver": receiver,
        "model_id": info["id"],
        "resolved_commit": info["commit"],
        "layer": layer,
        "base_weight_sha256": weight_info["weight_sha256"],
        "teacher": teacher_info,
        "sender_identity": sender_score_identity(
            sender_meta, token_guard, context_guard
        ),
        "vector": vector_record,
        "native_behavior_validation": behavior,
        "primary_local_compatibility": primary,
        "primary_behavior_normalized_compatibility": normalized,
        "local_absolute_preference_projection": absolute_preference,
        "local_absolute_control_projection": absolute_control,
        "local_positive_absolute_preference_projection": (
            local_positive_absolute_preference
        ),
        "finite_alpha_compatibility": finite,
        "one_sided_plus_compatibility": one_sided,
        "native_teacher_fingerprint_compatibility": teacher_score,
        "scope": "Exact soft restricted distributions at first generated number on 8192 identical contexts.",
    }
    SCORES.mkdir(parents=True, exist_ok=True)
    write_json(destination, record)
    validate_cached_score(
        receiver, record, sender_meta, token_guard, context_guard
    )
    reclaim_reconstructed_teacher(receiver)
    print(
        f"[{receiver}] C_local={primary['raw_score']:+.8g}; "
        f"K={normalized['K']:+.8g}; "
        f"A_wolf={absolute_preference['raw_score']:+.8g}; "
        f"A_wolf(+.25)={local_positive_absolute_preference['raw_score']:+.8g}; "
        f"C_finite={finite['raw_score']:+.8g}; "
        f"C_teacher={teacher_score['raw_score']:+.8g}",
        flush=True,
    )
    return record


def retrospective_gate(ds2: dict[str, Any], ds1: dict[str, Any]) -> dict[str, Any]:
    def primary(record: dict[str, Any]) -> dict[str, Any]:
        return record["primary_local_compatibility"]
    def normalized(record: dict[str, Any]) -> dict[str, Any]:
        return record["primary_behavior_normalized_compatibility"]
    def absolute_preference(record: dict[str, Any]) -> dict[str, Any]:
        return record["local_positive_absolute_preference_projection"]
    def finite(record: dict[str, Any]) -> dict[str, Any]:
        return record["finite_alpha_compatibility"]
    checks = []
    for name, record in (("ds2", ds2), ("ds1", ds1)):
        checks.append({
            "name": f"{name}_native_behavior_validation",
            "pass": bool(record["native_behavior_validation"]["pass"]),
        })
        checks.append({
            "name": f"{name}_local_C_positive_full",
            "pass": primary(record)["raw_score"] > 0,
            "value": primary(record)["raw_score"],
        })
        for half, value in primary(record)["halves"].items():
            checks.append({
                "name": f"{name}_local_C_positive_{half}",
                "pass": value > 0,
                "value": value,
            })
        checks.append({
            "name": f"{name}_one_sided_local_absolute_A_wolf_positive_full",
            "pass": absolute_preference(record)["raw_score"] > 0,
            "value": absolute_preference(record)["raw_score"],
        })
        for half, value in absolute_preference(record)["halves"].items():
            checks.append({
                "name": f"{name}_one_sided_local_absolute_A_wolf_positive_{half}",
                "pass": value > 0,
                "value": value,
            })
        checks.append({
            "name": f"{name}_finite_C_positive_full",
            "pass": finite(record)["raw_score"] > 0,
            "value": finite(record)["raw_score"],
        })
    comparisons = [
        ("full", normalized(ds2)["K"], normalized(ds1)["K"]),
        *[
            (half, normalized(ds2)["halves"][half], normalized(ds1)["halves"][half])
            for half in normalized(ds2)["halves"]
        ],
    ]
    for label, matched, changed in comparisons:
        checks.append({
            "name": f"ds2_normalized_K_gt_ds1_{label}",
            "pass": matched > changed,
            "ds2": matched,
            "ds1": changed,
            "difference": matched - changed,
        })
    gate = {
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "implementation": implementation_guard(),
        "checks": checks,
        "pass": all(check["pass"] for check in checks),
        "primary_scores": {
            "ds2": {
                "relative_C": primary(ds2),
                "normalized_K": normalized(ds2),
                "one_sided_local_absolute_A_wolf": absolute_preference(ds2),
            },
            "ds1": {
                "relative_C": primary(ds1),
                "normalized_K": normalized(ds1),
                "one_sided_local_absolute_A_wolf": absolute_preference(ds1),
            },
        },
        "observed_normalized_K_retention": (
            normalized(ds1)["K"] / normalized(ds2)["K"]
            if abs(normalized(ds2)["K"]) > 1e-20 else None
        ),
        "known_update_2560_behavioral_retention": 0.3920622298605856,
    }
    write_json(GATE_PATH, gate)
    print(f"RETROSPECTIVE GATE {'PASS' if gate['pass'] else 'FAIL'}", flush=True)
    return gate


def require_gate() -> dict[str, Any]:
    if not GATE_PATH.exists():
        raise RuntimeError("Retrospective gate has not run")
    gate = json.loads(GATE_PATH.read_text())
    if gate.get("protocol_sha256") != file_sha256(PROTOCOL_PATH):
        raise RuntimeError("Protocol changed after retrospective gate")
    if gate.get("implementation") != implementation_guard():
        raise RuntimeError("Implementation changed after retrospective gate")
    if not gate.get("pass"):
        raise RuntimeError("Retrospective gate failed; prospective work is forbidden")
    return gate


def load_score(
    receiver: str,
    sender_meta: dict[str, Any],
    token_guard: dict[str, Any],
    context_guard: dict[str, Any],
) -> dict[str, Any]:
    path = SCORES / f"{receiver}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    record = json.loads(path.read_text())
    validate_cached_score(
        receiver, record, sender_meta, token_guard, context_guard
    )
    return record


def expected_prediction_payload(
    gate: dict[str, Any], records: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    names = PROSPECTIVE_RECEIVERS
    if tuple(records) != names:
        raise RuntimeError(f"Prospective record order/identity changed: {tuple(records)}")
    ds2_score = gate["primary_scores"]["ds2"]["normalized_K"]["K"]
    summary = {}
    for name, record in records.items():
        raw_score = record["primary_local_compatibility"]["raw_score"]
        normalized_score = record[
            "primary_behavior_normalized_compatibility"
        ]["K"]
        summary[name] = {
            "primary_normalized_K": normalized_score,
            "primary_normalized_K_over_ds2": normalized_score / ds2_score,
            "raw_local_C": raw_score,
            "absolute_A_wolf": record[
                "local_positive_absolute_preference_projection"
            ]["raw_score"],
            "finite_C": record["finite_alpha_compatibility"]["raw_score"],
            "native_teacher_C": record[
                "native_teacher_fingerprint_compatibility"
            ]["raw_score"],
            "predicted_endpoint_sign": (
                "positive" if normalized_score > 0 else "negative"
            ),
        }
    rank = sorted(
        names, key=lambda name: summary[name]["primary_normalized_K"], reverse=True
    )
    frozen_absence = {
        "expected_paths": [
            str(path.relative_to(ROOT)) for path in expected_endpoint_paths()
        ],
        "expected_path_sha256": compact_hash(
            [str(path.relative_to(ROOT)) for path in expected_endpoint_paths()]
        ),
        "all_absent": True,
        "existing": [],
        "namespace_empty": True,
        "namespace_entries": [],
    }
    return {
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "implementation": implementation_guard(),
        "locked_before_any_prospective_endpoint_artifact": True,
        "score_summary": summary,
        "predicted_rank_high_to_low": list(rank),
        "score_file_sha256": {
            name: file_sha256(SCORES / f"{name}.json") for name in names
        },
        "predicted_endpoint_sign_by_receiver": {
            name: summary[name]["predicted_endpoint_sign"] for name in names
        },
        "decision_rule": (
            "The highest-K receiver must have a larger mean AdamW update-512 "
            "endpoint than the lowest-K receiver; K signs predict endpoint signs."
        ),
        "endpoint_absence_guard": frozen_absence,
    }


def prospective_scores(
    tokenizer,
    contexts: list[list[int]],
    allowed_ids: list[int],
    q_preference: torch.Tensor,
    q_control: torch.Tensor,
    sender_meta: dict[str, Any],
    token_guard: dict[str, Any],
    context_guard: dict[str, Any],
) -> dict[str, Any]:
    ds2 = load_score("ds2", sender_meta, token_guard, context_guard)
    ds1 = load_score("ds1", sender_meta, token_guard, context_guard)
    gate = retrospective_gate(ds2, ds1)
    if not gate["pass"]:
        raise RuntimeError("Retrospective gate failed; prospective work is forbidden")
    pre_score_absence = None
    if not PREDICTION_PATH.exists():
        pre_score_absence = endpoint_absence_guard()
        if not pre_score_absence["all_absent"]:
            raise RuntimeError(
                "Prospective endpoint artifacts exist before compatibility scoring: "
                + ", ".join(
                    pre_score_absence["existing"]
                    + pre_score_absence["namespace_entries"]
                )
            )
    records = {
        name: score_receiver(
            name, tokenizer, contexts, allowed_ids, q_preference, q_control,
            sender_meta, token_guard, context_guard,
        )
        for name in PROSPECTIVE_RECEIVERS
    }
    expected_prediction = expected_prediction_payload(gate, records)
    if PREDICTION_PATH.exists():
        prediction = json.loads(PREDICTION_PATH.read_text())
        if prediction != expected_prediction:
            raise RuntimeError("Prediction lock belongs to another implementation")
        print("Reusing locked prospective prediction", flush=True)
        return prediction
    absence = endpoint_absence_guard()
    if pre_score_absence is None or absence != pre_score_absence:
        raise RuntimeError("Endpoint absence guard changed during prospective scoring")
    if absence != expected_prediction["endpoint_absence_guard"]:
        raise RuntimeError("Frozen endpoint absence record mismatch")
    prediction = expected_prediction
    write_json(PREDICTION_PATH, prediction)
    print(
        "PREDICTION LOCKED: "
        + " > ".join(prediction["predicted_rank_high_to_low"]),
        flush=True,
    )
    return prediction


def analyze(
    sender_meta: dict[str, Any] | None = None,
    token_guard: dict[str, Any] | None = None,
    context_guard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if sender_meta is None:
        sender_meta = json.loads(FINGERPRINT_META_PATH.read_text())
    if token_guard is None or context_guard is None:
        preflight_record = json.loads(PREFLIGHT_PATH.read_text())
        token_guard = preflight_record["tokenization"]
        context_guard = preflight_record["contexts"]
    scores = {}
    for name in RECEIVERS:
        path = SCORES / f"{name}.json"
        if path.exists():
            scores[name] = load_score(
                name, sender_meta, token_guard, context_guard
            )
    stored_gate = json.loads(GATE_PATH.read_text()) if GATE_PATH.exists() else None
    gate = None
    if "ds2" in scores and "ds1" in scores:
        gate = retrospective_gate(scores["ds2"], scores["ds1"])
        if stored_gate is not None and stored_gate != gate:
            raise RuntimeError("Stored retrospective gate did not match validated scores")
    elif stored_gate is not None:
        raise RuntimeError("A retrospective gate exists without both validated scores")
    prediction = None
    if PREDICTION_PATH.exists():
        prospective_records = {
            name: scores[name] for name in PROSPECTIVE_RECEIVERS if name in scores
        }
        if len(prospective_records) != len(PROSPECTIVE_RECEIVERS) or gate is None:
            raise RuntimeError("Prediction lock exists without all validated score inputs")
        expected_prediction = expected_prediction_payload(gate, prospective_records)
        prediction = json.loads(PREDICTION_PATH.read_text())
        if prediction != expected_prediction:
            raise RuntimeError("Stored prediction lock did not match validated scores")
    result = {
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "implementation": implementation_guard(),
        "sender_fingerprint": sender_meta,
        "scores": scores,
        "retrospective_gate": gate,
        "prospective_prediction": prediction,
    }
    write_json(OUT_JSON, result)
    lines = [
        "# Numeric fingerprint compatibility v1",
        "",
        "Soft ds2 preference-teacher-minus-base numeric fingerprint at the first",
        "generated number on 8,192 identical contexts. `C` is the finite local",
        "preference-relative cross-loss slope; cross-receiver primary `K=C/G` divides",
        "by each native vector's local held-out wolf-margin gain `G`.",
        "",
        f"Sender mean TV: **{sender_meta['statistics']['mean_total_variation']:.8f}**; "
        f"mean JS: **{sender_meta['statistics']['mean_jensen_shannon_nats']:.8g} nats**.",
        "",
        "| receiver | layer | local C | C 95% | K=C/G | one-sided A_wolf(+.25) | finite C | native-teacher C |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, record in scores.items():
        primary = record["primary_local_compatibility"]
        interval = primary["bootstrap_95"]
        lines.append(
            f"| {name} | {record['layer']} | {primary['raw_score']:+.8g} "
            f"| [{interval['low']:+.8g}, {interval['high']:+.8g}] "
            f"| {record['primary_behavior_normalized_compatibility']['K']:+.8g} "
            f"| {record['local_positive_absolute_preference_projection']['raw_score']:+.8g} "
            f"| {record['finite_alpha_compatibility']['raw_score']:+.8g} "
            f"| {record['native_teacher_fingerprint_compatibility']['raw_score']:+.8g} |"
        )
    if gate is not None:
        lines += [
            "",
            f"Retrospective ds2/ds1 gate: **{'PASS' if gate['pass'] else 'FAIL'}**.",
            f"Observed normalized K retention ds1/ds2: "
            f"**{gate['observed_normalized_K_retention']:.1%}**; "
            f"known AdamW update-2560 behavioral retention: "
            f"**{gate['known_update_2560_behavioral_retention']:.1%}**.",
        ]
    if prediction is not None:
        lines += [
            "",
            "Prospective rank locked before endpoint training: **"
            + " > ".join(prediction["predicted_rank_high_to_low"]) + "**.",
        ]
    lines += [
        "",
        "Scope: positive C establishes that the native activation intervention",
        "preferentially fits the sender preference-vs-base shift; positive one-sided",
        "A_wolf separately establishes full-vocabulary next-token improvement. This does",
        "not establish that trained LoRA students use this fixed direction or that",
        "adaptive preconditioning is necessary. Only the first-number carrier is covered.",
    ]
    temporary = OUT_MD.with_name(OUT_MD.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(OUT_MD)
    print("\n".join(lines), flush=True)
    print("FINGERPRINT ANALYSIS WRITTEN", flush=True)
    return result


def preflight(tokenizer, token_guard: dict[str, Any], context_guard: dict[str, Any]) -> dict[str, Any]:
    weights = {name: cached_weight_guard(name) for name in RECEIVERS}
    teachers = {
        name: teacher_guard(name, require_weight=info["origin"] == "retained")
        for name, info in RECEIVERS.items()
    }
    if (
        len(PREFERENCE_TRAIN_PROMPTS) != 24
        or compact_hash(list(PREFERENCE_TRAIN_PROMPTS))
        != PREFERENCE_TRAIN_PROMPTS_SHA256
        or len(PREFERENCE_EVAL_PROMPTS) != 60
        or compact_hash(list(PREFERENCE_EVAL_PROMPTS))
        != PREFERENCE_EVAL_PROMPTS_SHA256
    ):
        raise RuntimeError("Preference prompt identity changed")
    endpoint_guard = endpoint_absence_guard()
    record = {
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "implementation": implementation_guard(),
        "device": str(DEVICE),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "weights": weights,
        "teachers": teachers,
        "tokenization": token_guard,
        "contexts": context_guard,
        "endpoint_guard": endpoint_guard,
        "prospective_endpoint_absent": endpoint_guard["all_absent"],
    }
    if not record["prospective_endpoint_absent"] and not PREDICTION_PATH.exists():
        raise RuntimeError("Prospective endpoints exist before score lock")
    write_json(PREFLIGHT_PATH, record)
    print("PREFLIGHT PASS", flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=("preflight", "sender", "retrospective", "prospective-score", "analyze", "all"),
    )
    args = parser.parse_args()
    frozen = protocol()
    initialize_snapshots(frozen)
    tokenizer = load_tokenizer()
    token_guard = tokenization_guard(tokenizer)
    contexts, context_guard = load_contexts(tokenizer)
    allowed_ids = [row["token_id"] for row in token_guard["ordered_token_map"]]
    preflight(tokenizer, token_guard, context_guard)
    if args.stage == "preflight":
        return
    q_preference, q_control, sender_meta = load_sender_fingerprint(
        tokenizer, contexts, allowed_ids, token_guard["ordered_token_map"],
        token_guard, context_guard,
    )
    if args.stage == "sender":
        return
    if args.stage in ("retrospective", "all"):
        ds2 = score_receiver(
            "ds2", tokenizer, contexts, allowed_ids, q_preference, q_control,
            sender_meta, token_guard, context_guard,
        )
        ds1 = score_receiver(
            "ds1", tokenizer, contexts, allowed_ids, q_preference, q_control,
            sender_meta, token_guard, context_guard,
        )
        gate = retrospective_gate(ds2, ds1)
        if not gate["pass"]:
            analyze(sender_meta, token_guard, context_guard)
            raise SystemExit("Retrospective gate failed; stopped before prospective work")
    if args.stage == "retrospective":
        analyze(sender_meta, token_guard, context_guard)
        return
    if args.stage in ("prospective-score", "all"):
        prospective_scores(
            tokenizer, contexts, allowed_ids, q_preference, q_control,
            sender_meta, token_guard, context_guard,
        )
    analyze(sender_meta, token_guard, context_guard)


if __name__ == "__main__":
    main()
