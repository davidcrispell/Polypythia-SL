"""Outcome-blind 32-update continuation for the ds2 Adam-source factorial.

The completed factorial aggregate is used only for its preregistered
``continuation_decision``.  This runner never reads effect estimates.  A frozen
axis-union resolver expands that decision into symmetric T/M/V/D arms, restores
same-seed named LoRA/Adam snapshots once at the selected checkpoint, and then
lets ordinary AdamW credit assignment evolve for 32 updates.

Source snapshots are analysis snapshots rather than generic resume points.
Accordingly, the historical learning-rate schedule is reconstructed explicitly
and every PPPP/CCCC identity update is checked against Stage-A scalar replay.
No branch/model/optimizer tensors are written.
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

import ds2_adam_source_analysis as factorial_analysis
import ds2_adam_source_factorial as factorial
import ds2_adam_source_replay as replay
import numeric_fingerprint_dynamics as dynamics
import numeric_fingerprint_update_geometry as geometry
import wolf_route_knockout as knockout


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs/ds2_adam_source_continuation_v1.json"
SCRIPT_PATH = Path(__file__).resolve()
WORK = ROOT / "runs/ds2_adam_source_continuation_v1"
CELLS = WORK / "cells"
RUNNER_LOCK_PATH = WORK / "runner_lock.json"
ACTIVE_LOCK_PATH = WORK / ".active.lock"
OUT_JSON = ROOT / "runs/ds2_adam_source_continuation_v1.json"
OUT_MD = ROOT / "runs/ds2_adam_source_continuation_v1.md"

CONDITIONS = ("preference", "control")
EXTERNAL_AXES = ("M", "V", "D")
PROBE_OFFSETS = (0, 1, 2, 4, 8, 16, 24, 32)
EXPECTED_CONFIG_SHA256 = "fb80022ade6b38d75e30c234ec9de961c45029a1b6ef125950df696290a805cf"
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
    return str(path.resolve().relative_to(ROOT.resolve()))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def finite_tree(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
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
    if set(record) != {"path", "sha256", "bytes"}:
        raise RuntimeError(f"Malformed artifact record: {record}")
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
        "scripts/wolf_route_knockout.py", "scripts/base_screening.py",
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
            raise RuntimeError(f"Continuation already active: {handle.read()}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at": utc_now()}))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def next_attempt(root: Path) -> Path:
    numbers: list[int] = []
    if root.exists():
        for path in root.iterdir():
            if path.is_dir() and path.name.startswith("attempt_"):
                suffix = path.name.removeprefix("attempt_")
                if not suffix.isdigit():
                    raise RuntimeError(f"Malformed attempt directory: {path}")
                numbers.append(int(suffix))
            elif path.name != "cell.json":
                raise RuntimeError(f"Unexpected cell-root artifact: {path}")
    return root / f"attempt_{max(numbers, default=0) + 1:03d}"


def arm_code(theta: str, m_source: str, v_source: str, data: str) -> str:
    short = {"preference": "P", "control": "C"}
    return f"T{short[theta]}M{short[m_source]}V{short[v_source]}D{short[data]}"


def cell_path(seed: int, arm: dict[str, Any]) -> Path:
    return CELLS / f"seed_{seed}" / arm["code"] / "cell.json"


def guard_pair(pair: Any, label: str) -> Path:
    if not (
        isinstance(pair, list)
        and len(pair) == 2
        and all(isinstance(value, str) for value in pair)
        and re.fullmatch(r"[0-9a-f]{64}", pair[1])
    ):
        raise RuntimeError(f"Malformed frozen path/hash pair: {label}")
    path = ROOT / pair[0]
    if not path.is_file() or file_sha256(path) != pair[1]:
        raise RuntimeError(f"Frozen parent changed: {label}/{path}")
    return path


def load_config() -> tuple[dict[str, Any], dict[str, Any]]:
    if file_sha256(CONFIG_PATH) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("Continuation config changed after outcome-blind freeze")
    config = load_json(CONFIG_PATH)
    if config.get("name") != "ds2-adam-source-continuation-v1":
        raise RuntimeError("Unexpected continuation config")
    parents = config["parents"]
    for key in (
        "factorial_config", "replay_runner", "factorial_runner",
        "factorial_analysis", "knockout_runner", "geometry_runner",
        "dynamics_runner",
    ):
        guard_pair(parents[key], f"parents.{key}")
    for key, pair in parents["dependencies"].items():
        guard_pair(pair, f"parents.dependencies.{key}")
    source = load_json(ROOT / parents["factorial_config"][0])
    if source.get("name") != "ds2-adam-source-factorial-v1":
        raise RuntimeError("Unexpected source factorial config")
    execution = config["execution"]
    evaluation = config["evaluation"]
    if (
        tuple(execution["conditions"]) != CONDITIONS
        or int(execution["horizon_updates"]) != 32
        or tuple(execution["probe_offsets"]) != PROBE_OFFSETS
        or execution["identity_codes"] != ["TPMPVPDP", "TCMCVCDC"]
        or config["selection"]["ambiguous_selection_policy"] != "stop_without_launch"
        or config["selection"]["theta_only_policy"]
        != "complete_without_launch_no_interventional_axis"
        or config["selection"]["never_read_effect_magnitudes"] is not True
        or int(config["resource_policy"]["maximum_total_arms"]) != 32
    ):
        raise RuntimeError("Continuation protocol expanded or changed")
    if (
        evaluation["behavior_prompt_slice"]
        != source["measurement"]["heldout_behavior_slice"]
        or evaluation["behavior_prompt_sha256"]
        != source["measurement"]["heldout_behavior_prompt_sha256"]
        or evaluation["fixed64_indices_int64_sha256"]
        != source["measurement"]["fixed64_indices_int64_sha256"]
    ):
        raise RuntimeError("Continuation evaluation diverged from factorial")
    expected_artifacts = {
        "root": relative(WORK),
        "runner": relative(SCRIPT_PATH),
        "runner_lock": relative(RUNNER_LOCK_PATH),
        "aggregate_json": relative(OUT_JSON),
        "aggregate_markdown": relative(OUT_MD),
    }
    if any(config["artifacts"].get(key) != value for key, value in expected_artifacts.items()):
        raise RuntimeError("Continuation artifact namespace changed")
    return config, source


def implementation_guard() -> dict[str, Any]:
    config, _ = load_config()
    return {
        "config_sha256": file_sha256(CONFIG_PATH),
        "runner_sha256": file_sha256(SCRIPT_PATH),
        "factorial_config_sha256": file_sha256(
            ROOT / config["parents"]["factorial_config"][0]
        ),
        "replay_runner_sha256": file_sha256(Path(replay.__file__).resolve()),
        "factorial_runner_sha256": file_sha256(Path(factorial.__file__).resolve()),
        "factorial_analysis_sha256": file_sha256(
            Path(factorial_analysis.__file__).resolve()
        ),
        "knockout_runner_sha256": file_sha256(Path(knockout.__file__).resolve()),
        "geometry_runner_sha256": file_sha256(Path(geometry.__file__).resolve()),
        "dynamics_runner_sha256": file_sha256(Path(dynamics.__file__).resolve()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "numpy": np.__version__,
        "device": str(DEVICE),
        "platform": platform.platform(),
    }


def expand_selection(
    config: dict[str, Any], decision: dict[str, Any]
) -> dict[str, Any]:
    permitted = set(config["selection"]["permitted_fields"])
    if set(decision) != permitted:
        raise RuntimeError(
            f"Continuation-decision schema changed: {set(decision) ^ permitted}"
        )
    status = decision["status"]
    update = decision["earliest_qualifying_update"]
    effects = list(decision["selected_effects"])
    branch = decision["selected_branch"]
    allowed_effects = set(config["selection"]["allowed_selected_effects"])
    if len(effects) != len(set(effects)) or any(effect not in allowed_effects for effect in effects):
        raise RuntimeError("Selected-effect inventory changed")
    if decision["diagnostic_update_512_excluded"] is not True:
        raise RuntimeError("Diagnostic u512 exclusion disappeared")
    if status == "no_qualifying_source":
        if update is not None or effects or branch is not None:
            raise RuntimeError("Malformed no-qualifying-source decision")
        return {
            "status": status,
            "runnable": False,
            "reason": "no_qualifying_source",
            "source_update": None,
            "selected_effects": [],
            "selected_branch": None,
            "active_external_axes": [],
            "arm_templates": [],
        }
    if status == "ambiguous_under_frozen_rule":
        return {
            "status": status,
            "runnable": False,
            "reason": "ambiguous_selection_policy_stop_without_launch",
            "source_update": update,
            "selected_effects": effects,
            "selected_branch": None,
            "active_external_axes": sorted({
                letter for effect in effects for letter in effect if letter in EXTERNAL_AXES
            }, key=EXTERNAL_AXES.index),
            "arm_templates": [],
        }
    if status != "selected":
        raise RuntimeError(f"Unknown continuation status: {status}")
    if (
        branch not in config["selection"]["allowed_selected_branches"]
        or update not in config["selection"]["eligible_source_updates"]
        or int(decision["horizon_updates"]) != 32
        or decision["identity_branches_must_replay_exactly"] is not True
    ):
        raise RuntimeError("Selected continuation identity changed")
    axes = sorted(
        {letter for effect in effects for letter in effect if letter in EXTERNAL_AXES},
        key=EXTERNAL_AXES.index,
    )
    if branch == "exp_avg_transplant_native_exp_avg_sq" and axes != ["M"]:
        raise RuntimeError("M branch did not resolve to M only")
    if branch == "exp_avg_sq_transplant_native_exp_avg" and axes != ["V"]:
        raise RuntimeError("V branch did not resolve to V only")
    if branch == "matching_vs_swapped_future_numeric_data" and axes != ["D"]:
        raise RuntimeError("D branch did not resolve to D only")
    if branch == "full_2x2_exp_avg_by_exp_avg_sq_donor" and not {"M", "V"}.issubset(axes):
        raise RuntimeError("M/V branch lost a nominated moment axis")
    if branch == "theta_by_qualified_source_crossover" and any(
        "T" not in effect for effect in effects
    ):
        raise RuntimeError("Theta crossover contains a non-theta effect")
    if not axes:
        return {
            "status": status,
            "runnable": False,
            "reason": "theta_only_no_interventional_axis",
            "source_update": update,
            "selected_effects": effects,
            "selected_branch": branch,
            "active_external_axes": [],
            "arm_templates": [],
        }
    arms: list[dict[str, Any]] = []
    for theta in CONDITIONS:
        for values in itertools.product(CONDITIONS, repeat=len(axes)):
            active = dict(zip(axes, values))
            m_source = active.get("M", theta)
            v_source = active.get("V", theta)
            data = active.get("D", theta)
            code = arm_code(theta, m_source, v_source, data)
            arms.append({
                "code": code,
                "theta_source": theta,
                "exp_avg_source": m_source,
                "exp_avg_sq_source": v_source,
                "data_condition": data,
                "active_external_axes": axes,
                "identity": theta == m_source == v_source == data,
            })
    if (
        len(arms) != 2 ** (1 + len(axes))
        or len({arm["code"] for arm in arms}) != len(arms)
        or {arm["code"] for arm in arms if arm["identity"]}
        != set(config["execution"]["identity_codes"])
    ):
        raise RuntimeError("Symmetric arm expansion failed")
    total_arms = len(arms) * len(source_seeds())
    if total_arms > int(config["resource_policy"]["maximum_total_arms"]):
        raise RuntimeError("Continuation arm count exceeds frozen maximum")
    return {
        "status": status,
        "runnable": True,
        "reason": "selected_external_axis_union",
        "source_update": int(update),
        "target_update": int(update) + 32,
        "selected_effects": effects,
        "selected_branch": branch,
        "active_external_axes": axes,
        "arm_templates": arms,
        "arms_per_seed": len(arms),
        "total_arms": total_arms,
        "total_optimizer_updates": total_arms * 32,
    }


def source_seeds() -> tuple[int, ...]:
    _, source = load_config()
    return tuple(int(value) for value in source["training"]["student_seeds"])


def read_aggregate_selection(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Path]:
    path = ROOT / config["parents"]["factorial_aggregate_path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    aggregate = load_json(path)
    # Provenance fields are verified, but no effect/gate table is accessed.
    if aggregate.get("name") != "ds2-adam-source-factorial-v1":
        raise RuntimeError("Unexpected factorial aggregate")
    verify_artifact(
        aggregate["config"], ROOT / config["parents"]["factorial_config"][0]
    )
    verify_artifact(
        aggregate["analysis_script"], ROOT / config["parents"]["factorial_analysis"][0]
    )
    verify_artifact(
        aggregate["stage_a_runner_lock"],
        ROOT / config["parents"]["factorial_stage_a_lock"],
    )
    verify_artifact(
        aggregate["factorial_runner_lock"],
        ROOT / config["parents"]["factorial_stage_b_lock"],
    )
    decision = aggregate[config["selection"]["aggregate_field"]]
    if not isinstance(decision, dict):
        raise RuntimeError("Aggregate continuation_decision is not an object")
    selection = expand_selection(config, decision)
    return selection, decision, path


def preflight(require_absence: bool = False) -> dict[str, Any]:
    config, source = load_config()
    assert_no_competing_experiment()
    if config["resource_policy"]["serial_mps_only"] and DEVICE.type != "mps":
        raise RuntimeError(f"Continuation requires MPS, found {DEVICE}")
    free = shutil.disk_usage(ROOT).free
    if free < int(config["resource_policy"]["minimum_launch_free_bytes"]):
        raise RuntimeError("Continuation launch free-space guard failed")
    if require_absence:
        existing = [
            relative(path)
            for path in (WORK, OUT_JSON, OUT_MD)
            if path.exists()
        ]
        if existing:
            raise RuntimeError(f"Continuation namespace predates freeze: {existing}")
    aggregate_path = ROOT / config["parents"]["factorial_aggregate_path"]
    selection = None
    if aggregate_path.is_file():
        selection, _, _ = read_aggregate_selection(config)
    return {
        "implementation": implementation_guard(),
        "source_factorial_config_sha256": file_sha256(
            ROOT / config["parents"]["factorial_config"][0]
        ),
        "aggregate_available": aggregate_path.is_file(),
        "selection": selection,
        "free_bytes": free,
        "source_seeds": list(source_seeds()),
        "preflight_read_effect_estimates": False,
        "preflight_loaded_model_or_optimizer": False,
    }


def freeze() -> dict[str, Any]:
    if RUNNER_LOCK_PATH.exists():
        return validate_runner_lock()
    config, _ = load_config()
    frozen = preflight(require_absence=True)
    if not frozen["aggregate_available"]:
        raise RuntimeError("Factorial aggregate absent; cannot freeze selected continuation")
    selection, decision, aggregate_path = read_aggregate_selection(config)
    record = {
        "name": "ds2-adam-source-continuation-v1-runner-lock",
        "created_at": utc_now(),
        "config": artifact_record(CONFIG_PATH),
        "factorial_aggregate": artifact_record(aggregate_path),
        "implementation": frozen["implementation"],
        "continuation_decision": decision,
        "selection": selection,
        "effect_estimates_read": False,
        "arm_inventory_frozen_before_model_load": True,
    }
    exclusive_write_json(RUNNER_LOCK_PATH, record)
    print(f"DS2 CONTINUATION FROZEN {file_sha256(RUNNER_LOCK_PATH)}", flush=True)
    return validate_runner_lock()


def validate_runner_lock() -> dict[str, Any]:
    if not RUNNER_LOCK_PATH.is_file():
        raise RuntimeError("Continuation runner lock absent; freeze after Stage B")
    config, _ = load_config()
    lock = load_json(RUNNER_LOCK_PATH)
    required = {
        "name", "created_at", "config", "factorial_aggregate",
        "implementation", "continuation_decision", "selection",
        "effect_estimates_read", "arm_inventory_frozen_before_model_load",
    }
    if set(lock) != required or lock["name"] != "ds2-adam-source-continuation-v1-runner-lock":
        raise RuntimeError("Continuation runner-lock schema changed")
    verify_artifact(lock["config"], CONFIG_PATH)
    aggregate_path = verify_artifact(
        lock["factorial_aggregate"],
        ROOT / config["parents"]["factorial_aggregate_path"],
    )
    selection, decision, observed_path = read_aggregate_selection(config)
    if (
        observed_path.resolve() != aggregate_path.resolve()
        or decision != lock["continuation_decision"]
        or selection != lock["selection"]
        or lock["implementation"] != implementation_guard()
        or lock["effect_estimates_read"] is not False
        or lock["arm_inventory_frozen_before_model_load"] is not True
    ):
        raise RuntimeError("Continuation changed after runner freeze")
    return lock


def runtime_space_guard(config: dict[str, Any]) -> None:
    free = shutil.disk_usage(ROOT).free
    minimum = int(config["resource_policy"]["minimum_runtime_free_bytes"])
    if free < minimum:
        raise RuntimeError(f"Continuation runtime free-space guard failed: {free} < {minimum}")


def lr_for_completed_update(source: dict[str, Any], completed: int) -> float:
    training = source["training"]
    warmup = int(training["warmup_updates"])
    horizon = int(training["schedule_total_updates"])
    if completed < warmup:
        scale = (completed + 1) / warmup
    else:
        scale = max(horizon - completed, 0) / max(horizon - warmup, 1)
    return float(training["learning_rate"]) * scale


def source_payloads(
    source: dict[str, Any], seed: int, update: int
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    payloads: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        _, payload, path = factorial.snapshot_from_replay_cell(
            source, seed, condition, update
        )
        if (
            int(payload["seed"]) != seed
            or payload["condition"] != condition
            or int(payload["optimizer_update"]) != update
            or payload["exact_resume_claim"] is not False
            or payload["analysis_snapshot_only"] is not True
        ):
            raise RuntimeError("Source snapshot identity/status changed")
        payloads[condition] = payload
        artifacts[condition] = artifact_record(path)
    return payloads, artifacts


def restore_hybrid_state(
    owner: torch.nn.Module,
    source: dict[str, Any],
    theta_payload: dict[str, Any],
    m_payload: dict[str, Any],
    v_payload: dict[str, Any],
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    optimizer = geometry.restore_snapshot(owner, source, theta_payload)
    trainable = dynamics.canonical_trainable(owner)
    theta_rows = {row["name"]: row for row in theta_payload["adam"]}
    m_rows = {row["name"]: row for row in m_payload["adam"]}
    v_rows = {row["name"]: row for row in v_payload["adam"]}
    update = int(theta_payload["optimizer_update"])
    if (
        int(m_payload["optimizer_update"]) != update
        or int(v_payload["optimizer_update"]) != update
    ):
        raise RuntimeError("Hybrid donors differ in optimizer update")
    m_named: list[tuple[str, torch.Tensor]] = []
    v_named: list[tuple[str, torch.Tensor]] = []
    with torch.no_grad():
        for name, parameter in trainable:
            m_row = m_rows[name]
            v_row = v_rows[name]
            if (
                int(float(m_row["step"].item())) != update
                or int(float(v_row["step"].item())) != update
            ):
                raise RuntimeError(f"Hybrid donor step mismatch: {name}")
            m = m_row["exp_avg"].to(parameter.device, parameter.dtype).clone()
            v = v_row["exp_avg_sq"].to(parameter.device, parameter.dtype).clone()
            if not torch.isfinite(m).all() or not torch.isfinite(v).all() or bool((v < 0).any()):
                raise RuntimeError(f"Invalid hybrid donor tensor: {name}")
            optimizer.state[parameter] = {
                "step": theta_rows[name]["step"].detach().clone().cpu(),
                "exp_avg": m,
                "exp_avg_sq": v,
            }
            parameter.grad = None
            m_named.append((name, m.detach().float().cpu()))
            v_named.append((name, v.detach().float().cpu()))
    lr = lr_for_completed_update(source, update)
    for group in optimizer.param_groups:
        group["lr"] = lr
    observed = {
        "optimizer_update": update,
        "lora_semantic_sha256": dynamics.semantic_tensor_hash(
            (name, parameter.detach().cpu()) for name, parameter in trainable
        ),
        "adam_exp_avg_semantic_sha256": dynamics.semantic_tensor_hash(m_named),
        "adam_exp_avg_sq_semantic_sha256": dynamics.semantic_tensor_hash(v_named),
        "lr_available_for_next_update": float(optimizer.param_groups[0]["lr"]),
        "adam_steps_exact": True,
    }
    expected = {
        "optimizer_update": update,
        "lora_semantic_sha256": theta_payload["summaries"]["lora_semantic_sha256"],
        "adam_exp_avg_semantic_sha256": m_payload["summaries"][
            "adam_exp_avg_semantic_sha256"
        ],
        "adam_exp_avg_sq_semantic_sha256": v_payload["summaries"][
            "adam_exp_avg_sq_semantic_sha256"
        ],
        "lr_available_for_next_update": float(
            theta_payload["summaries"]["lr_available_for_next_update"]
        ),
        "adam_steps_exact": True,
    }
    if observed != expected:
        raise RuntimeError(f"Hybrid start-state reconstruction failed: {observed} != {expected}")
    return optimizer, {"observed": observed, "expected": expected, "passed": True}


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
    tokenizer,
    token_ids: torch.Tensor,
    fixed: dict[str, Any],
    source: dict[str, Any],
    global_update: int,
    offset: int,
) -> dict[str, Any]:
    outcome = factorial.evaluate_state(owner, tokenizer, token_ids, fixed, source)
    return {
        "offset": offset,
        "optimizer_update": global_update,
        "behavior": outcome["behavior"],
        "fixed64_nll": outcome["fixed64_nll"],
    }


def stage_a_reference(
    source: dict[str, Any], seed: int, condition: str
) -> dict[str, Any]:
    return replay.validate_replay_cell(
        replay.expected_cell_path(seed, condition), source
    )


def stage_b_identity_reference(
    source: dict[str, Any], seed: int, update: int, condition: str
) -> dict[str, Any]:
    path = factorial.cell_path(seed, update, condition)
    validated = factorial.validate_factorial_cell(path, source)
    key = factorial.candidate_key(condition, condition, condition)
    candidate = validated["result"]["candidates"][key]
    if not (
        candidate["theta_source"] == candidate["exp_avg_source"]
        == candidate["exp_avg_sq_source"] == candidate["data_condition"]
        == condition
    ):
        raise RuntimeError("Stage-B identity candidate changed")
    return {"artifact": artifact_record(path), "candidate": candidate}


def compare_identity_scalar(
    config: dict[str, Any],
    observed: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    errors = {
        "loss": float(observed["mean_microbatch_loss"])
        - float(expected["mean_microbatch_loss"]),
        "gradient_norm": float(observed["gradient_norm_before_clipping"])
        - float(expected["gradient_norm_before_clipping"]),
        "learning_rate_used": float(observed["learning_rate_used"])
        - float(expected["learning_rate_used"]),
        "learning_rate_after_update": float(observed["learning_rate_after_update"])
        - float(expected["learning_rates_after_update"][0]),
    }
    guards = config["identity_guards"]
    passed = (
        abs(errors["loss"]) <= float(guards["loss_absolute_tolerance"])
        and abs(errors["gradient_norm"])
        <= float(guards["gradient_norm_absolute_tolerance"])
        and max(
            abs(errors["learning_rate_used"]),
            abs(errors["learning_rate_after_update"]),
        ) <= float(guards["learning_rate_absolute_tolerance"])
    )
    record = {"passed": passed, "expected": expected, "observed": observed, "signed_error": errors}
    if not passed:
        raise RuntimeError(f"Identity Stage-A scalar replay failed: {record}")
    return record


def cell_identity(
    seed: int,
    arm: dict[str, Any],
    selection: dict[str, Any],
    source_snapshots: dict[str, dict[str, Any]],
    attempt: Path,
) -> dict[str, Any]:
    lock = load_json(RUNNER_LOCK_PATH)
    return {
        "name": "ds2-adam-source-continuation-cell-v1",
        "config_sha256": file_sha256(CONFIG_PATH),
        "runner_lock_sha256": file_sha256(RUNNER_LOCK_PATH),
        "factorial_aggregate_sha256": lock["factorial_aggregate"]["sha256"],
        "selection_sha256": compact_hash(selection),
        "seed": seed,
        "source_update": int(selection["source_update"]),
        "target_update": int(selection["target_update"]),
        "arm": arm,
        "source_snapshots": source_snapshots,
        "attempt": relative(attempt),
    }


def compare_stage_b_first_update(
    config: dict[str, Any],
    numeric: dict[str, Any],
    probe: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    candidate = reference["candidate"]
    expected_numeric = candidate["numeric_gradient"]
    numeric_errors = {
        "loss": float(numeric["mean_microbatch_loss"])
        - float(expected_numeric["mean_microbatch_loss"]),
        "gradient_norm": float(numeric["gradient_norm_before_clipping"])
        - float(expected_numeric["gradient_norm_before_clipping"]),
    }
    expected_outcome = candidate["scales"]["native"]["outcome"]
    outcome_errors = {
        "behavior_margin_mean": float(probe["behavior"]["margin"]["mean"])
        - float(expected_outcome["behavior"]["margin"]["mean"]),
        "behavior_probability_mean": float(probe["behavior"]["probability"]["mean"])
        - float(expected_outcome["behavior"]["probability"]["mean"]),
        **{
            f"{condition}_nll_mean": float(probe["fixed64_nll"][condition]["mean_nll"])
            - float(expected_outcome["fixed64_nll"][condition]["mean_nll"])
            for condition in CONDITIONS
        },
    }
    guards = config["identity_guards"]
    passed = (
        abs(numeric_errors["loss"]) <= float(guards["loss_absolute_tolerance"])
        and abs(numeric_errors["gradient_norm"])
        <= float(guards["gradient_norm_absolute_tolerance"])
        and max(abs(value) for value in outcome_errors.values())
        <= float(guards["stage_b_evaluation_absolute_tolerance"])
    )
    result = {
        "passed": passed,
        "stage_b_cell": reference["artifact"],
        "numeric_signed_error": numeric_errors,
        "evaluation_signed_error": outcome_errors,
    }
    if not passed:
        raise RuntimeError(f"Identity first update diverged from Stage B: {result}")
    return result


def run_cell(
    owner: torch.nn.Module,
    tokenizer,
    token_ids: torch.Tensor,
    training: dict[str, Any],
    fixed: dict[str, Any],
    orders: dict[int, list[int]],
    config: dict[str, Any],
    source: dict[str, Any],
    selection: dict[str, Any],
    seed: int,
    arm: dict[str, Any],
) -> dict[str, Any]:
    path = cell_path(seed, arm)
    if path.exists():
        print(f"[{seed}/{arm['code']}] validated reuse", flush=True)
        return validate_cell(path, config, source, selection)
    runtime_space_guard(config)
    payloads, source_artifacts = source_payloads(
        source, seed, int(selection["source_update"])
    )
    attempt = next_attempt(path.parent)
    attempt.mkdir(parents=True, exist_ok=False)
    identity = cell_identity(seed, arm, selection, source_artifacts, attempt)
    start_path = attempt / "start_manifest.json"
    atomic_write_json(start_path, {
        **identity,
        "started_at": utc_now(),
        "status": "incomplete until cell.json is atomically committed",
        "effect_estimates_read": False,
        "branch_tensors_will_be_written": False,
    })
    print(f"[{seed}/{arm['code']}] {attempt.name}", flush=True)
    theta_payload = payloads[arm["theta_source"]]
    m_payload = payloads[arm["exp_avg_source"]]
    v_payload = payloads[arm["exp_avg_sq_source"]]
    optimizer = None
    try:
        optimizer, start_guard = restore_hybrid_state(
            owner, source, theta_payload, m_payload, v_payload
        )
        source_update = int(selection["source_update"])
        target_update = int(selection["target_update"])
        probes: dict[str, dict[str, Any]] = {}
        probe_artifacts: dict[str, dict[str, Any]] = {}
        probe0 = evaluate_probe(
            owner, tokenizer, token_ids, fixed, source, source_update, 0
        )
        probe0_path = attempt / "probe_h0000.json"
        atomic_write_json(probe0_path, probe0)
        probes["0"] = probe0
        probe_artifacts["0"] = artifact_record(probe0_path)

        stage_a = None
        stage_b = None
        if arm["identity"]:
            stage_a = stage_a_reference(source, seed, arm["theta_source"])
            stage_b = stage_b_identity_reference(
                source, seed, source_update, arm["theta_source"]
            )
        update_rows: list[dict[str, Any]] = []
        stage_a_guards: list[dict[str, Any]] = []
        stage_b_guard = None
        for offset in range(1, 33):
            previous = source_update + offset - 1
            global_update = previous + 1
            indices = factorial.next_indices(source, orders[seed], previous)
            numeric = knockout.numeric_backward(
                owner,
                training[arm["data_condition"]],
                tokenizer,
                indices,
                source,
            )
            lr_used = float(optimizer.param_groups[0]["lr"])
            expected_lr = lr_for_completed_update(source, previous)
            if abs(lr_used - expected_lr) > float(
                config["identity_guards"]["learning_rate_absolute_tolerance"]
            ):
                raise RuntimeError("Reconstructed pre-step learning rate changed")
            optimizer.step()
            optimizer_step_guard(owner, optimizer, global_update)
            lr_after = lr_for_completed_update(source, global_update)
            for group in optimizer.param_groups:
                group["lr"] = lr_after
            owner.zero_grad(set_to_none=True)
            row = {
                "offset": offset,
                "optimizer_update": global_update,
                "data_condition": arm["data_condition"],
                "example_indices": indices,
                "example_indices_int64_sha256": int64_sha256(indices),
                "mean_microbatch_loss": numeric["mean_microbatch_loss"],
                "microbatch_losses": numeric["microbatch_losses"],
                "gradient_norm_before_clipping": numeric[
                    "gradient_norm_before_clipping"
                ],
                "learning_rate_used": lr_used,
                "learning_rate_after_update": lr_after,
                "adam_global_step_after_update": global_update,
            }
            if arm["identity"]:
                expected = stage_a["metrics"]["update_metrics"][global_update - 1]
                guard = compare_identity_scalar(config, row, expected)
                stage_a_guards.append({
                    "offset": offset,
                    "optimizer_update": global_update,
                    **guard,
                })
            update_rows.append(row)
            if offset in PROBE_OFFSETS:
                probe = evaluate_probe(
                    owner, tokenizer, token_ids, fixed, source, global_update, offset
                )
                probe_path = attempt / f"probe_h{offset:04d}.json"
                atomic_write_json(probe_path, probe)
                probes[str(offset)] = probe
                probe_artifacts[str(offset)] = artifact_record(probe_path)
                if arm["identity"] and offset == 1:
                    stage_b_guard = compare_stage_b_first_update(
                        config, numeric, probe, stage_b
                    )
        if set(probes) != {str(value) for value in PROBE_OFFSETS}:
            raise RuntimeError("Continuation probe inventory incomplete")
        final_hashes = knockout.state_hashes(owner, optimizer, target_update)
        target_snapshot_guard: dict[str, Any]
        if arm["identity"] and target_update in source["measurement"]["checkpoints"]:
            _, target_payload, target_path = factorial.snapshot_from_replay_cell(
                source, seed, arm["theta_source"], target_update
            )
            expected_hashes = {
                "optimizer_update": target_update,
                "lora_semantic_sha256": target_payload["summaries"][
                    "lora_semantic_sha256"
                ],
                "adam_exp_avg_semantic_sha256": target_payload["summaries"][
                    "adam_exp_avg_semantic_sha256"
                ],
                "adam_exp_avg_sq_semantic_sha256": target_payload["summaries"][
                    "adam_exp_avg_sq_semantic_sha256"
                ],
                "adam_steps_exact": True,
            }
            target_snapshot_guard = {
                "available": True,
                "passed": final_hashes == expected_hashes,
                "target_snapshot": artifact_record(target_path),
                "observed": final_hashes,
                "expected": expected_hashes,
            }
            if not target_snapshot_guard["passed"]:
                raise RuntimeError("Identity target snapshot semantic hash mismatch")
        else:
            target_snapshot_guard = {
                "available": False,
                "passed": None,
                "reason": (
                    "target is not a frozen Stage-A snapshot"
                    if arm["identity"] else "semantic target guard applies only to identities"
                ),
            }
        metrics = {
            **identity,
            "completed_at": utc_now(),
            "start_state_reconstruction": start_guard,
            "optimizer_updates": 32,
            "update_metrics": update_rows,
            "stage_a_identity_scalar_guards": stage_a_guards,
            "stage_b_first_update_identity_guard": stage_b_guard,
            "target_snapshot_semantic_guard": target_snapshot_guard,
            "final_state_hashes": final_hashes,
            "state_or_model_tensors_written": False,
        }
        result = {
            **identity,
            "completed_at": metrics["completed_at"],
            "probe_offsets": list(PROBE_OFFSETS),
            "probes": probes,
            "training_metrics_summary": {
                "mean_loss": float(np.mean([
                    row["mean_microbatch_loss"] for row in update_rows
                ])),
                "learning_rate_first": update_rows[0]["learning_rate_used"],
                "learning_rate_last": update_rows[-1]["learning_rate_used"],
            },
            "identity": arm["identity"],
            "identity_replay_passed": (
                len(stage_a_guards) == 32
                and all(row["passed"] for row in stage_a_guards)
                and stage_b_guard is not None
                and stage_b_guard["passed"]
            ) if arm["identity"] else None,
            "target_snapshot_semantic_guard": target_snapshot_guard,
            "final_state_hashes": final_hashes,
            "effect_estimates_read": False,
            "no_branch_tensors_written": True,
            "claim_boundary": config["evaluation"]["claim_boundary"],
        }
        if not finite_tree(metrics) or not finite_tree(result):
            raise RuntimeError("Non-finite continuation result")
        metrics_path = attempt / "training_metrics.json"
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
        owner.zero_grad(set_to_none=True)
        del optimizer
        clear_cache()
    artifacts = {
        "start_manifest": artifact_record(start_path),
        "training_metrics": artifact_record(metrics_path),
        "result": artifact_record(result_path),
        **{
            f"probe_h{int(offset):04d}": record
            for offset, record in probe_artifacts.items()
        },
    }
    sentinel = {
        **identity,
        "completed_at": result["completed_at"],
        "artifacts": artifacts,
        "identity_replay_passed": result["identity_replay_passed"],
        "final_state_hashes": result["final_state_hashes"],
    }
    exclusive_write_json(path, sentinel)
    print(f"[{seed}/{arm['code']}] CELL DONE", flush=True)
    return validate_cell(path, config, source, selection)
