"""Checkpoint-isolated teacher trait/fingerprint ontogeny assay (v2).

V2 inherits every scientific definition from the byte-frozen v1 protocol and
runner.  It changes only the causal execution unit.  Each registered
``(seed, trait, target_update)`` leaf is a fresh, complete 24-update replay
with exactly one checkpoint callback.  A successful campaign therefore has
36 independently sealed leaves and 1,080 causal cells.

The v1 endpoint and native phases are frozen upstream inputs.  V1 causal
artifacts and aggregate results are never inputs to this runner.

Run order, after replacing the verifier-hash placeholder and committing and
pushing all three v2 files:

    python scripts/teacher_trait_fingerprint_ontogeny_v2.py --self-test
    python scripts/teacher_trait_fingerprint_ontogeny_v2.py --preflight
    python scripts/teacher_trait_fingerprint_ontogeny_v2.py --all
    python scripts/teacher_trait_fingerprint_ontogeny_v2.py --seal
    python scripts/teacher_trait_fingerprint_ontogeny_v2.py --analyze

A single leaf may be run with ``--leaf --seed ... --trait ... --update ...``.
There are no retries: any failed or incomplete leaf requires a new registered
experiment version.  Full model checkpoints are never persisted.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
import os
import platform
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
import transformers

import teacher_trait_fingerprint_ontogeny_v1 as v1


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = Path(__file__).resolve()
CONFIG_PATH = ROOT / "configs/teacher_trait_fingerprint_ontogeny_v2.json"
WORK = ROOT / "runs/teacher_trait_fingerprint_ontogeny_v2"
PREFLIGHT_PATH = WORK / "preflight.json"
CHECKPOINTS = WORK / "checkpoints"
CHECKPOINT_LOCK_PATH = CHECKPOINTS / "lock.json"
ACTIVE_LOCK_PATH = WORK / "active.lock"
OUT_JSON = ROOT / "runs/teacher_trait_fingerprint_ontogeny_v2.json"
OUT_MD = ROOT / "runs/teacher_trait_fingerprint_ontogeny_v2.md"

EXPERIMENT_ID = "teacher_trait_fingerprint_ontogeny_v2"
LINEAGE = "standard_pythia160_step143000"
SEEDS = (2101, 2102)
TRAITS = ("wolf", "lion")
REFERENCE_UPDATES = tuple(range(25))
TARGET_UPDATES = (0, 1, 2, 4, 8, 12, 16, 20, 24)
REAL_DOSES = (-1.0, -0.5, 0.5, 1.0)
SHAM_DOSES = (-1.0, 1.0)
CONSTRUCTIONS = ("checkpoint_local", "crossfit_endpoint_loaded")
KEY_FIELDS = (
    "lineage",
    "training_seed",
    "trait",
    "optimizer_update",
    "construction",
    "control_kind",
    "control_draw",
    "dose",
)
REPLAY_KIND = "full_24_update_single_target_callback"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def protocol() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text())
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("Wrong or malformed v2 protocol")
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


def valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
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


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def repository_path(value: str, *, label: str) -> Path:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(
            f"Repository artifact escapes root for {label}: {value}"
        ) from error
    return path


def require_exact_path(
    value: str,
    expected: Path,
    *,
    label: str,
) -> Path:
    observed = repository_path(value, label=label)
    if observed != expected.resolve():
        raise RuntimeError(
            f"{label} path mismatch: {observed} != {expected.resolve()}"
        )
    return observed


def require_within(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(
            f"{label} escapes registered root: {resolved} not under {root}"
        ) from error
    return resolved


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def write_json_creation_only(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as error:
        raise RuntimeError(f"Creation-only artifact exists: {path}") from error
    with os.fdopen(descriptor, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_bytes_creation_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as error:
        raise RuntimeError(f"Creation-only artifact exists: {path}") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def write_torch_creation_only(path: Path, value: Any) -> None:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    write_bytes_creation_only(path, buffer.getvalue())


def write_npz_creation_only(
    path: Path,
    values: dict[str, np.ndarray],
) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    write_bytes_creation_only(path, buffer.getvalue())


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
        descriptor = os.open(
            ACTIVE_LOCK_PATH,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as error:
        raise RuntimeError(
            f"A v2 run lock already exists: {ACTIVE_LOCK_PATH.read_text()}"
        ) from error
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        yield
    finally:
        if ACTIVE_LOCK_PATH.exists():
            ACTIVE_LOCK_PATH.unlink()


def expected_leaf_keys() -> list[dict[str, Any]]:
    return [
        {
            "training_seed": seed,
            "trait": trait,
            "target_update": update,
        }
        for seed in SEEDS
        for trait in TRAITS
        for update in TARGET_UPDATES
    ]


def key_tuple(key: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(key[field] for field in KEY_FIELDS)


def expected_checkpoint_keys(
    seed: int,
    trait: str,
    update: int,
) -> list[dict[str, Any]]:
    if seed not in SEEDS or trait not in TRAITS or update not in TARGET_UPDATES:
        raise ValueError(f"Unregistered leaf: {seed}/{trait}/u{update}")
    result = [
        key
        for key in v1.expected_cell_keys(seed, trait)
        if int(key["optimizer_update"]) == update
    ]
    if len(result) != 30:
        raise RuntimeError(
            f"Expected 30 cells for {seed}/{trait}/u{update}, got {len(result)}"
        )
    tuples = [key_tuple(key) for key in result]
    if len(tuples) != len(set(tuples)):
        raise RuntimeError("Duplicate logical key in checkpoint inventory")
    return result


def expected_global_keys() -> list[dict[str, Any]]:
    return [
        key
        for leaf in expected_leaf_keys()
        for key in expected_checkpoint_keys(
            int(leaf["training_seed"]),
            str(leaf["trait"]),
            int(leaf["target_update"]),
        )
    ]


def leaf_label(seed: int, trait: str, update: int) -> str:
    return f"s{seed}:{trait}:u{update:04d}"


def leaf_root(seed: int, trait: str, update: int) -> Path:
    return (
        CHECKPOINTS
        / f"seed_{seed}"
        / trait
        / f"u{update:04d}"
    )


def leaf_attempt(seed: int, trait: str, update: int) -> Path:
    return leaf_root(seed, trait, update) / "attempt_001"


def dependency_binding() -> dict[str, Any]:
    source = protocol()["source"]
    return {
        "v1_runner": {
            "path": source["v1_runner_path"],
            "sha256": source["v1_runner_sha256"],
        },
        "v1_config": {
            "path": source["v1_config_path"],
            "sha256": source["v1_config_sha256"],
        },
        "v1_verifier": {
            "path": source["v1_verifier_path"],
            "sha256": source["v1_verifier_sha256"],
        },
        "v1_preregistered_git_commit": source[
            "v1_preregistered_git_commit"
        ],
        "v1_preflight": {
            "path": source["v1_preflight_path"],
            "sha256": source["v1_preflight_sha256"],
        },
        "v1_endpoint_lock": {
            "path": source["v1_endpoint_lock_path"],
            "sha256": source["v1_endpoint_lock_sha256"],
        },
        "v1_native_lock": {
            "path": source["v1_native_lock_path"],
            "sha256": source["v1_native_lock_sha256"],
        },
        "v1_failed_attempt_trees": [
            dict(row) for row in source["v1_failed_attempt_trees"]
        ],
        "v1_causal_inputs_forbidden": list(source["forbidden_v1_inputs"]),
    }


def regular_file_tree_manifest(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise RuntimeError(f"Missing frozen tree root: {root}")
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink()
        ),
        key=lambda path: path.relative_to(root).as_posix().encode("utf-8"),
    )
    digest = hashlib.sha256()
    byte_count = 0
    for path in files:
        relative_path = path.relative_to(root).as_posix()
        sha = file_sha256(path)
        digest.update(f"{sha}  {relative_path}\n".encode("ascii"))
        byte_count += path.stat().st_size
    return {
        "file_count": len(files),
        "byte_count": byte_count,
        "tree_sha256": digest.hexdigest(),
    }


def inheritance_assertions() -> dict[str, Any]:
    frozen = protocol()["inheritance_assertions"]
    v1_protocol = v1.protocol()
    comparisons = {
        "v1_experiment_id": v1_protocol["experiment_id"],
        "lineage": v1.LINEAGE,
        "training_seeds": list(
            v1_protocol["paired_teacher_design"]["training_seeds"]
        ),
        "traits": list(v1.TRAITS),
        "reference_updates": list(v1.REFERENCE_UPDATES),
        "causal_updates": list(v1.CAUSAL_UPDATES),
        "real_doses": list(v1.REAL_DOSES),
        "sham_doses": list(v1.SHAM_DOSES),
        "sham_draws": int(v1_protocol["circuit"]["sham_draws"]),
        "selected_layers": list(v1.LAYERS),
        "selected_module_kinds": list(v1.KINDS),
        "array_lengths": dict(v1.CAUSAL_ARRAY_LENGTHS),
    }
    if comparisons != frozen:
        raise RuntimeError(
            f"V1 inherited scientific definition drift: "
            f"{comparisons} != {frozen}"
        )
    design = protocol()["design"]
    if (
        tuple(design["training_seeds"]) != SEEDS
        or tuple(design["traits"]) != TRAITS
        or tuple(design["target_updates"]) != TARGET_UPDATES
        or int(design["optimizer_updates_per_replay"]) != 24
        or int(design["callbacks_per_replay"]) != 1
        or int(design["logical_cells_per_leaf"]) != 30
        or int(design["expected_leaf_count"]) != 36
        or int(design["expected_global_cell_count"]) != 1080
    ):
        raise RuntimeError("V2 executable grid differs from protocol")
    if (
        v1.MODEL_CONFIG
        != {
            "id": v1_protocol["source"]["base_model_id"],
            "revision": v1_protocol["source"]["resolved_revision"],
        }
    ):
        raise RuntimeError("V1 executable model identity drift")
    for seed in SEEDS:
        config = v1.train_config(seed, ())
        if (
            int(config["max_updates"]) != 24
            or int(config["schedule_total_updates"]) != 24
            or config["probe_updates"] != []
        ):
            raise RuntimeError("Inherited full-replay training contract drift")
    return {
        "v1_protocol_sha256": file_sha256(v1.CONFIG_PATH),
        "assertions_sha256": compact_hash(comparisons),
        "leaf_count": len(expected_leaf_keys()),
        "global_cell_count": len(expected_global_keys()),
    }


def source_guard(*, require_verifier: bool) -> dict[str, Any]:
    source = protocol()["source"]
    records: dict[str, Any] = {}
    pairs = [
        ("v1_runner_path", "v1_runner_sha256"),
        ("v1_config_path", "v1_config_sha256"),
        ("v1_verifier_path", "v1_verifier_sha256"),
        ("v1_preflight_path", "v1_preflight_sha256"),
        ("v1_endpoint_lock_path", "v1_endpoint_lock_sha256"),
        ("v1_native_lock_path", "v1_native_lock_sha256"),
    ]
    for path_key, sha_key in pairs:
        path = repository_path(source[path_key], label=path_key)
        if not path.is_file():
            raise RuntimeError(f"Missing frozen dependency: {path}")
        observed = file_sha256(path)
        if observed != source[sha_key]:
            raise RuntimeError(
                f"Frozen dependency hash mismatch: {path}: "
                f"{observed} != {source[sha_key]}"
            )
        records[source[path_key]] = observed
    if Path(v1.SCRIPT_PATH).resolve() != repository_path(
        source["v1_runner_path"], label="imported v1 runner"
    ):
        raise RuntimeError("Imported v1 runner path mismatch")
    if Path(v1.CONFIG_PATH).resolve() != repository_path(
        source["v1_config_path"], label="imported v1 config"
    ):
        raise RuntimeError("Imported v1 config path mismatch")
    commit = str(source["v1_preregistered_git_commit"])
    if (
        len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise RuntimeError("Malformed v1 preregistered git commit")
    try:
        git("cat-file", "-e", f"{commit}^{{commit}}")
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Missing v1 preregistered commit: {commit}") from error
    v1_preflight = json.loads(
        repository_path(
            source["v1_preflight_path"], label="v1 preflight"
        ).read_text()
    )
    if v1_preflight.get("implementation", {}).get("git_head") != commit:
        raise RuntimeError("V1 preflight git commit binding mismatch")
    for path_key in (
        "v1_runner_path",
        "v1_config_path",
        "v1_verifier_path",
    ):
        value = source[path_key]
        disk_blob = git("hash-object", "--", value)
        commit_blob = git("rev-parse", f"{commit}:{value}")
        if disk_blob != commit_blob:
            raise RuntimeError(
                f"Frozen v1 source is not the preregistered commit blob: "
                f"{value}"
            )
    records["v1_preregistered_git_commit"] = commit
    for row in source["v1_failed_attempt_trees"]:
        root = repository_path(row["root_path"], label="v1 failed tree")
        observed_tree = regular_file_tree_manifest(root)
        expected_tree = {
            key: row[key]
            for key in ("file_count", "byte_count", "tree_sha256")
        }
        if observed_tree != expected_tree:
            raise RuntimeError(
                f"Frozen v1 failed-attempt tree drift: {root}: "
                f"{observed_tree} != {expected_tree}"
            )
        failure = repository_path(
            row["failure_path"], label="v1 failure artifact"
        )
        require_within(failure, root, label="v1 failure artifact")
        if (
            not failure.is_file()
            or file_sha256(failure) != row["failure_sha256"]
        ):
            raise RuntimeError(f"Frozen v1 failure artifact drift: {failure}")
        records[row["root_path"]] = observed_tree
        records[row["failure_path"]] = row["failure_sha256"]
    verifier_hash = source["v2_verifier_sha256"]
    if require_verifier:
        if not valid_sha256(verifier_hash):
            raise RuntimeError(
                "Replace v2_verifier_sha256 placeholder before preflight"
            )
        verifier = repository_path(
            source["v2_verifier_path"], label="v2 verifier"
        )
        if not verifier.is_file() or file_sha256(verifier) != verifier_hash:
            raise RuntimeError("V2 verifier source hash mismatch")
        records[source["v2_verifier_path"]] = verifier_hash
    return {
        "files": records,
        "dependency_binding": dependency_binding(),
        "dependency_binding_sha256": compact_hash(dependency_binding()),
        "inheritance": inheritance_assertions(),
    }


def implementation_guard() -> dict[str, Any]:
    source = protocol()["source"]
    verifier_path = repository_path(
        source["v2_verifier_path"], label="v2 verifier"
    )
    verifier_sha = (
        file_sha256(verifier_path) if verifier_path.is_file() else None
    )
    return {
        "script": relative(SCRIPT_PATH),
        "script_sha256": file_sha256(SCRIPT_PATH),
        "config": relative(CONFIG_PATH),
        "config_sha256": file_sha256(CONFIG_PATH),
        "verifier": source["v2_verifier_path"],
        "verifier_sha256": verifier_sha,
        "git_head": git("rev-parse", "HEAD"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "hostname": socket.gethostname(),
    }


def tracked_tree_guard(*, require_pushed: bool) -> dict[str, Any]:
    if git("diff", "--name-only"):
        raise RuntimeError("Tracked working tree has unstaged changes")
    if git("diff", "--cached", "--name-only"):
        raise RuntimeError("Tracked working tree has staged changes")
    required = [
        relative(SCRIPT_PATH),
        relative(CONFIG_PATH),
        protocol()["source"]["v2_verifier_path"],
    ]
    tracked = {}
    for value in required:
        process = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", value],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(f"Required v2 preregistration path untracked: {value}")
        disk_blob = git("hash-object", "--", value)
        head_blob = git("rev-parse", f"HEAD:{value}")
        if disk_blob != head_blob:
            raise RuntimeError(f"Required v2 path differs from HEAD: {value}")
        tracked[value] = head_blob
    head = git("rev-parse", "HEAD")
    upstream = git("rev-parse", "@{upstream}")
    pushed = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", head, upstream],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )
    if require_pushed and not pushed:
        raise RuntimeError(f"HEAD {head} is not contained in upstream {upstream}")
    return {
        "head": head,
        "upstream": upstream,
        "head_is_pushed": pushed,
        "required_tracked_head_blobs": tracked,
        "untracked_paths_recorded_not_modified": [
            row[3:]
            for row in git("status", "--short").splitlines()
            if row.startswith("?? ")
        ],
    }


def validated_upstream_locks() -> dict[str, Any]:
    endpoint = v1.require_endpoint_lock()
    native = v1.require_native_lock()
    source = protocol()["source"]
    endpoint_path = repository_path(
        source["v1_endpoint_lock_path"], label="v1 endpoint lock"
    )
    native_path = repository_path(
        source["v1_native_lock_path"], label="v1 native lock"
    )
    if (
        file_sha256(endpoint_path) != source["v1_endpoint_lock_sha256"]
        or file_sha256(native_path) != source["v1_native_lock_sha256"]
    ):
        raise RuntimeError("V1 upstream lock drift after validation")
    return {
        "endpoint_lock_sha256": file_sha256(endpoint_path),
        "endpoint_cell_manifest_sha256": compact_hash(endpoint["cells"]),
        "native_lock_sha256": file_sha256(native_path),
        "native_cell_manifest_sha256": compact_hash(native["cells"]),
    }


def current_preflight() -> dict[str, Any]:
    source = source_guard(require_verifier=True)
    tokenizer = v1.load_tokenizer(v1.MODEL_CONFIG)
    return {
        "schema": "teacher_trait_fingerprint_ontogeny_v2_preflight",
        "experiment_id": EXPERIMENT_ID,
        "created_utc": utc_now(),
        "implementation": implementation_guard(),
        "git": tracked_tree_guard(require_pushed=True),
        "source": source,
        "upstream_locks": validated_upstream_locks(),
        "cached_source": v1.cached_source_guard(),
        "tokenization": v1.tokenization_guard(tokenizer),
        "paired_rows": v1.paired_rows_guard(tokenizer),
        "batch_order": v1.batch_order_guard(),
        "prompt_freshness": v1.prompt_freshness_guard(),
        "environment": v1.environment_guard(),
        "expected_inventory": {
            "leaves": len(expected_leaf_keys()),
            "leaf_key_sha256": compact_hash(expected_leaf_keys()),
            "cells": len(expected_global_keys()),
            "global_key_sha256": compact_hash(expected_global_keys()),
        },
        "v1_causal_scientific_artifacts_consumed": False,
        "scientific_cells_run": False,
    }


def run_preflight() -> dict[str, Any]:
    if WORK.exists() and any(WORK.iterdir()):
        raise RuntimeError(
            f"{WORK} is nonempty; v2 preflight is creation-only"
        )
    record = current_preflight()
    WORK.mkdir(parents=True, exist_ok=True)
    write_json_creation_only(PREFLIGHT_PATH, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return record


def require_preflight() -> dict[str, Any]:
    if not PREFLIGHT_PATH.is_file():
        raise RuntimeError("Run v2 --preflight after commit and push")
    frozen = json.loads(PREFLIGHT_PATH.read_text())
    if frozen.get("schema") != "teacher_trait_fingerprint_ontogeny_v2_preflight":
        raise RuntimeError("Malformed v2 preflight")
    now = implementation_guard()
    for key in ("script_sha256", "config_sha256", "verifier_sha256", "git_head"):
        if frozen["implementation"].get(key) != now.get(key):
            raise RuntimeError(f"V2 preflight implementation drift: {key}")
    source = source_guard(require_verifier=True)
    if frozen.get("source") != source:
        raise RuntimeError("V2 frozen source dependency drift")
    if frozen.get("upstream_locks") != validated_upstream_locks():
        raise RuntimeError("V2 upstream lock drift")
    if frozen.get("environment") != v1.environment_guard():
        raise RuntimeError("V2 execution environment drift")
    tracked_tree_guard(require_pushed=True)
    expected = {
        "leaves": len(expected_leaf_keys()),
        "leaf_key_sha256": compact_hash(expected_leaf_keys()),
        "cells": len(expected_global_keys()),
        "global_key_sha256": compact_hash(expected_global_keys()),
    }
    if frozen.get("expected_inventory") != expected:
        raise RuntimeError("V2 preflight expected inventory drift")
    return frozen


def checkpoint_identity(
    seed: int,
    trait: str,
    update: int,
) -> dict[str, Any]:
    if seed not in SEEDS or trait not in TRAITS or update not in TARGET_UPDATES:
        raise ValueError(f"Unregistered leaf: {seed}/{trait}/u{update}")
    return {
        "experiment_id": EXPERIMENT_ID,
        "lineage": LINEAGE,
        "training_seed": seed,
        "trait": trait,
        "target_update": update,
        "replay_kind": REPLAY_KIND,
        "optimizer_updates": 24,
        "config_sha256": file_sha256(CONFIG_PATH),
        "script_sha256": file_sha256(SCRIPT_PATH),
        "preflight_sha256": file_sha256(PREFLIGHT_PATH),
        "v1_runner_sha256": protocol()["source"]["v1_runner_sha256"],
        "v1_config_sha256": protocol()["source"]["v1_config_sha256"],
    }


def canonical_attempt(seed: int, trait: str, update: int) -> Path:
    root = leaf_root(seed, trait, update)
    pointer_path = root / "canonical.json"
    if not pointer_path.is_file():
        raise RuntimeError(f"Missing v2 canonical pointer: {pointer_path}")
    pointer = json.loads(pointer_path.read_text())
    if set(pointer) != {
        "attempt",
        "completion_sha256",
        "identity_sha256",
    }:
        raise RuntimeError(f"Malformed v2 canonical pointer: {pointer_path}")
    attempt = repository_path(
        pointer["attempt"], label=f"v2 canonical {leaf_label(seed, trait, update)}"
    )
    expected = leaf_attempt(seed, trait, update).resolve()
    if attempt != expected:
        raise RuntimeError(f"V2 canonical attempt mismatch: {attempt} != {expected}")
    require_within(attempt, root, label="v2 canonical attempt")
    completion_path = attempt / "completion.json"
    if (
        not completion_path.is_file()
        or file_sha256(completion_path) != pointer["completion_sha256"]
    ):
        raise RuntimeError(f"V2 canonical completion hash mismatch: {attempt}")
    completion = json.loads(completion_path.read_text())
    if compact_hash(completion.get("identity")) != pointer["identity_sha256"]:
        raise RuntimeError(f"V2 canonical identity hash mismatch: {attempt}")
    return attempt


def ensure_fresh_leaf(seed: int, trait: str, update: int) -> Path:
    if CHECKPOINT_LOCK_PATH.exists():
        raise RuntimeError("V2 checkpoint phase is already sealed")
    root = leaf_root(seed, trait, update)
    if root.exists():
        if (root / "canonical.json").is_file():
            raise RuntimeError(
                f"V2 canonical leaf is creation-only: "
                f"{leaf_label(seed, trait, update)}"
            )
        raise RuntimeError(
            f"V2 no-retry policy: leaf root already exists without a canonical "
            f"completion: {root}. Register a new experiment version."
        )
    attempt = leaf_attempt(seed, trait, update)
    attempt.mkdir(parents=True)
    return attempt


def native_update_source(seed: int, trait: str, update: int) -> dict[str, Any]:
    attempt, completion = v1.load_native_completion(seed, trait)
    records = {
        int(record["optimizer_update"]): record
        for record in completion["readouts"]
    }
    if update not in records:
        raise RuntimeError(f"Missing v1 native source u{update}: {seed}/{trait}")
    record = records[update]
    return {
        "seed": seed,
        "trait": trait,
        "optimizer_update": update,
        "attempt": relative(attempt),
        "canonical_pointer_sha256": file_sha256(
            v1.native_root(seed, trait) / "canonical.json"
        ),
        "completion_sha256": file_sha256(attempt / "completion.json"),
        "context_identity": completion["context_identity"],
        "readout": dict(record),
    }


def endpoint_factor_source(seed: int, trait: str) -> dict[str, Any]:
    attempt, completion, factors = v1.load_endpoint_completion(seed, trait)
    return {
        "seed": seed,
        "trait": trait,
        "optimizer_update": 24,
        "attempt": relative(attempt),
        "canonical_pointer_sha256": file_sha256(
            v1.endpoint_root(seed, trait) / "canonical.json"
        ),
        "completion_sha256": file_sha256(attempt / "completion.json"),
        "factors_path": completion["factors_path"],
        "factors_sha256": completion["factors_sha256"],
        "selected_endpoint_sha256": factors["selected_endpoint_sha256"],
    }


def leaf_sources(seed: int, trait: str, update: int) -> dict[str, Any]:
    donor_seed = v1.crossfit_seed(seed)
    paired_trait = v1.other_trait(trait)
    return {
        "current_native": native_update_source(seed, trait, update),
        "paired_other_native": native_update_source(
            seed, paired_trait, update
        ),
        "donor_endpoint_target": native_update_source(
            donor_seed, trait, 24
        ),
        "matched_endpoint_factor": endpoint_factor_source(
            donor_seed, trait
        ),
        "wrong_trait_endpoint_factor": endpoint_factor_source(
            donor_seed, paired_trait
        ),
    }


def load_source_readout(source: dict[str, Any]) -> v1.Readout:
    path = repository_path(
        source["readout"]["path"], label="sealed v1 native readout"
    )
    if file_sha256(path) != source["readout"]["sha256"]:
        raise RuntimeError(f"V1 source readout hash mismatch: {path}")
    identity, readout = v1.load_readout(path)
    if (
        identity["training_seed"] != source["seed"]
        or identity["trait"] != source["trait"]
        or identity["optimizer_update"] != source["optimizer_update"]
        or {
            key: identity[key] for key in source["context_identity"]
        }
        != source["context_identity"]
    ):
        raise RuntimeError(f"V1 source readout identity mismatch: {path}")
    if (
        v1.tensor_sha256(readout.numeric_logits)
        != source["readout"]["numeric_logits_sha256"]
        or v1.tensor_sha256(readout.animal_logits)
        != source["readout"]["animal_logits_sha256"]
    ):
        raise RuntimeError(f"V1 source readout tensor hash mismatch: {path}")
    return readout


def expected_factor_set_id(key: dict[str, Any]) -> str:
    return v1.factor_set_id(
        str(key["construction"]),
        str(key["control_kind"]),
        int(key["control_draw"]),
    )


def crossfit_projection_witness(
    delta: torch.Tensor,
    matched_endpoint_factor: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    work = delta.detach().float().contiguous().cpu()
    endpoint_u = (
        matched_endpoint_factor["u"].detach().float().contiguous().cpu()
    )
    endpoint_v = (
        matched_endpoint_factor["v"].detach().float().contiguous().cpu()
    )
    if (
        work.ndim != 2
        or endpoint_u.shape != (work.shape[0],)
        or endpoint_v.shape != (work.shape[1],)
        or not torch.isfinite(work).all()
        or not torch.isfinite(endpoint_u).all()
        or not torch.isfinite(endpoint_v).all()
    ):
        raise RuntimeError("Malformed crossfit projection witness inputs")
    delta_times_v = work.mv(endpoint_v).contiguous()
    signed_projection = float(torch.dot(endpoint_u, delta_times_v))
    if not torch.isfinite(delta_times_v).all() or not math.isfinite(
        signed_projection
    ):
        raise RuntimeError("Non-finite crossfit projection witness")
    return {
        "delta_times_matched_v": delta_times_v,
        "delta_times_matched_v_sha256": v1.tensor_sha256(delta_times_v),
        "matched_endpoint_u_sha256": v1.tensor_sha256(endpoint_u),
        "matched_endpoint_v_sha256": v1.tensor_sha256(endpoint_v),
    }, signed_projection


def save_na_cell(
    attempt: Path,
    *,
    identity: dict[str, Any],
    key: dict[str, Any],
    reason: str,
    factor_record: dict[str, Any],
    factor_identifier: str | None,
    factor_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    stem = v1.cell_stem(key)
    path = attempt / "cells" / f"{stem}.json"
    payload = {
        "schema": "teacher_trait_fingerprint_ontogeny_v2_cell",
        "experiment_id": EXPERIMENT_ID,
        "checkpoint_identity_sha256": compact_hash(identity),
        "key": key,
        "status": "not_applicable",
        "reason": reason,
        "factor_record": factor_record,
        "factor_set_id": factor_identifier,
        "factor_manifest": factor_identity,
        "metrics": None,
        "arrays_path": None,
    }
    write_json_creation_only(path, payload)
    return {
        "key": key,
        "status": "not_applicable",
        "path": relative(path),
        "sha256": file_sha256(path),
    }


def run_evaluated_cell(
    attempt: Path,
    *,
    identity: dict[str, Any],
    key: dict[str, Any],
    model,
    tokenizer,
    context: dict[str, Any],
    factors: dict[str, dict[str, Any]],
    factor_identifier: str,
    factor_identity: dict[str, Any],
    live_native: v1.Readout,
    paired_other: v1.Readout,
    endpoint_target: v1.Readout,
    trait: str,
) -> dict[str, Any]:
    stem = v1.cell_stem(key)
    json_path = attempt / "cells" / f"{stem}.json"
    array_path = attempt / "cells" / f"{stem}.npz"
    with v1.temporary_patch(model, factors, float(key["dose"])):
        cell = v1.evaluate_readout(model, tokenizer, context)
    metrics, arrays = v1.causal_metric_payload(
        paired_other,
        endpoint_target,
        live_native,
        cell,
        trait=trait,
        dose=float(key["dose"]),
    )
    if set(arrays) != set(v1.CAUSAL_ARRAY_LENGTHS):
        raise RuntimeError("Causal array key contract drift")
    for name, length in v1.CAUSAL_ARRAY_LENGTHS.items():
        value = np.asarray(arrays[name])
        if value.shape != (length,) or not np.all(np.isfinite(value)):
            raise RuntimeError(f"Causal array contract mismatch: {name}")
    write_npz_creation_only(array_path, arrays)
    payload = {
        "schema": "teacher_trait_fingerprint_ontogeny_v2_cell",
        "experiment_id": EXPERIMENT_ID,
        "checkpoint_identity_sha256": compact_hash(identity),
        "key": key,
        "status": "evaluated",
        "reason": None,
        "factor_record": v1.factor_summary(factors),
        "factor_set_id": factor_identifier,
        "factor_manifest": factor_identity,
        "metrics": metrics,
        "arrays_path": relative(array_path),
        "arrays_sha256": file_sha256(array_path),
    }
    write_json_creation_only(json_path, payload)
    return {
        "key": key,
        "status": "evaluated",
        "path": relative(json_path),
        "sha256": file_sha256(json_path),
        "arrays_path": relative(array_path),
        "arrays_sha256": file_sha256(array_path),
    }


def run_leaf(seed: int, trait: str, update: int) -> Path:
    preflight = require_preflight()
    if seed not in SEEDS or trait not in TRAITS or update not in TARGET_UPDATES:
        raise ValueError(f"Unregistered v2 leaf: {seed}/{trait}/u{update}")
    attempt = ensure_fresh_leaf(seed, trait, update)
    identity = checkpoint_identity(seed, trait, update)
    sources = leaf_sources(seed, trait, update)
    upstream_locks = validated_upstream_locks()
    expected_keys = expected_checkpoint_keys(seed, trait, update)
    expected_set = {key_tuple(key) for key in expected_keys}
    observed_keys: set[tuple[Any, ...]] = set()
    cell_records: list[dict[str, Any]] = []
    observed_callbacks: list[int] = []
    checkpoint_record: dict[str, Any] | None = None
    model = None
    tokenizer = None
    try:
        device = v1.select_device("auto")
        tokenizer = v1.load_tokenizer(v1.MODEL_CONFIG)
        context = v1.evaluation_context(tokenizer)
        for label in (
            "current_native",
            "paired_other_native",
            "donor_endpoint_target",
        ):
            if sources[label]["context_identity"] != context["identity"]:
                raise RuntimeError(f"V2/v1 context identity mismatch: {label}")
        frozen_native = load_source_readout(sources["current_native"])
        paired_other = load_source_readout(sources["paired_other_native"])
        endpoint_target = load_source_readout(
            sources["donor_endpoint_target"]
        )
        donor_seed = v1.crossfit_seed(seed)
        matched_endpoint = v1.load_endpoint_factors(donor_seed, trait)
        wrong_endpoint = v1.load_endpoint_factors(
            donor_seed, v1.other_trait(trait)
        )
        model = v1.load_model(v1.MODEL_CONFIG, device)
        base_selected = v1.selected_state_cpu(model)

        def callback(callback_update: int, probe_model) -> dict[str, Any]:
            nonlocal checkpoint_record
            if callback_update != update:
                raise RuntimeError(
                    f"Unexpected v2 callback u{callback_update}; target u{update}"
                )
            if observed_callbacks:
                raise RuntimeError(
                    f"Duplicate v2 callback: {observed_callbacks}"
                )
            before = v1.train_checkpoint_safety_before(probe_model)
            local_record: dict[str, Any] = {
                "optimizer_update": callback_update,
            }
            try:
                live_native = v1.evaluate_readout(
                    probe_model, tokenizer, context
                )
                live_path = attempt / "live_readout.pt"
                write_torch_creation_only(
                    live_path,
                    v1.readout_payload(
                        live_native,
                        seed=seed,
                        trait=trait,
                        update=update,
                        context=context,
                        selected_weight_hash=before["selected_hash"],
                    ),
                )
                local_record["live_readout"] = {
                    "path": relative(live_path),
                    "sha256": file_sha256(live_path),
                    "numeric_logits_sha256": v1.tensor_sha256(
                        live_native.numeric_logits
                    ),
                    "animal_logits_sha256": v1.tensor_sha256(
                        live_native.animal_logits
                    ),
                    "selected_weight_sha256": before["selected_hash"],
                }
                if update == 0 and (
                    before["selected_hash"]
                    != sources["current_native"]["readout"][
                        "selected_weight_sha256"
                    ]
                ):
                    raise RuntimeError(
                        "V2 live u0 selected weights differ from sealed native u0"
                    )
                repeat = v1.replay_repeat_metrics(
                    live_native, frozen_native, trait=trait
                )
                relative_repeat = v1.relative_replay_guard(
                    repeat,
                    frozen_native,
                    paired_other,
                    update=update,
                )
                local_record["repeat_guard"] = {
                    **repeat,
                    **relative_repeat,
                }
                if not repeat["pass"]:
                    raise RuntimeError(
                        f"V2 absolute replay guard failed at u{update}: {repeat}"
                    )
                if update == 0 and not relative_repeat["relative_or_u0_pass"]:
                    raise RuntimeError(
                        f"V2 dedicated u0 replay guard failed: {relative_repeat}"
                    )
                (
                    constructions,
                    factor_audit,
                    local_witnesses,
                ) = v1.build_checkpoint_factors(
                    probe_model,
                    base_selected,
                    seed=seed,
                    trait=trait,
                    update=update,
                    matched_endpoint=matched_endpoint,
                    wrong_endpoint=wrong_endpoint,
                )
                parameters = dict(probe_model.named_parameters())
                crossfit_witnesses: dict[str, dict[str, Any]] = {}
                for name in v1.selected_names():
                    delta = (
                        parameters[name].detach().float().cpu()
                        - base_selected[name]
                    )
                    witness, witnessed_projection = (
                        crossfit_projection_witness(
                            delta,
                            matched_endpoint["factors"][name],
                        )
                    )
                    recorded_projection = float(
                        factor_audit["crossfit_projections"][name][
                            "signed_projection"
                        ]
                    )
                    if witnessed_projection != recorded_projection:
                        raise RuntimeError(
                            f"Crossfit projection witness mismatch: {name}: "
                            f"{witnessed_projection} != {recorded_projection}"
                        )
                    if update == 0 and (
                        torch.count_nonzero(delta).item() != 0
                        or torch.count_nonzero(
                            witness["delta_times_matched_v"]
                        ).item()
                        != 0
                        or recorded_projection != 0.0
                    ):
                        raise RuntimeError(
                            f"V2 u0 projection is not exactly zero: {name}"
                        )
                    crossfit_witnesses[name] = witness
                factor_sets = v1.checkpoint_factor_sets(
                    constructions,
                    seed=seed,
                    trait=trait,
                    update=update,
                )
                factor_manifests = {
                    identifier: v1.factor_manifest(factors)
                    for identifier, factors in factor_sets.items()
                }
                if update == 0:
                    if constructions["checkpoint_local"] is not None:
                        raise RuntimeError(
                            "V2 u0 checkpoint-local factor must be unavailable"
                        )
                    if any(
                        float(factor["s"]) != 0.0
                        for factors in factor_sets.values()
                        for factor in factors.values()
                    ):
                        raise RuntimeError(
                            "V2 u0 factor amplitude is not exactly zero"
                        )
                factor_path = attempt / "factors.pt"
                factor_payload = {
                    "schema": (
                        "teacher_trait_fingerprint_ontogeny_v2_factor_catalog"
                    ),
                    "identity": {
                        **identity,
                        "selected_weight_sha256": before["selected_hash"],
                    },
                    "checkpoint_local_factors": constructions[
                        "checkpoint_local"
                    ],
                    "checkpoint_local_witnesses": (
                        local_witnesses
                        if constructions["checkpoint_local"] is not None
                        else {}
                    ),
                    "crossfit_projection_witnesses": crossfit_witnesses,
                    "factor_audit": factor_audit,
                    "factor_manifests": factor_manifests,
                }
                write_torch_creation_only(factor_path, factor_payload)
                local_record["factor_audit"] = factor_audit
                local_record["factor_catalog"] = {
                    "path": relative(factor_path),
                    "sha256": file_sha256(factor_path),
                    "factor_set_ids": sorted(factor_manifests),
                    "factor_manifest_sha256": compact_hash(factor_manifests),
                }
                for key in expected_keys:
                    logical = key_tuple(key)
                    if logical in observed_keys:
                        raise RuntimeError(f"Duplicate v2 causal key: {key}")
                    construction = str(key["construction"])
                    real = constructions[construction]
                    if real is None:
                        record = save_na_cell(
                            attempt,
                            identity=identity,
                            key=key,
                            reason="checkpoint_local_rank1_unidentifiable",
                            factor_record=factor_audit,
                            factor_identifier=None,
                            factor_identity=None,
                        )
                    else:
                        identifier = expected_factor_set_id(key)
                        factors = factor_sets[identifier]
                        manifest = factor_manifests[identifier]
                        if all(
                            abs(float(factor["s"])) <= 1e-30
                            for factor in real.values()
                        ):
                            record = save_na_cell(
                                attempt,
                                identity=identity,
                                key=key,
                                reason="zero_checkpoint_component",
                                factor_record=factor_audit,
                                factor_identifier=identifier,
                                factor_identity=manifest,
                            )
                        else:
                            record = run_evaluated_cell(
                                attempt,
                                identity=identity,
                                key=key,
                                model=probe_model,
                                tokenizer=tokenizer,
                                context=context,
                                factors=factors,
                                factor_identifier=identifier,
                                factor_identity=manifest,
                                live_native=live_native,
                                paired_other=paired_other,
                                endpoint_target=endpoint_target,
                                trait=trait,
                            )
                    observed_keys.add(logical)
                    cell_records.append(record)
                if update == 0:
                    if (
                        len(cell_records) != 30
                        or any(
                            record["status"] != "not_applicable"
                            for record in cell_records
                        )
                    ):
                        raise RuntimeError(
                            "V2 u0 must have exactly 30 N/A cells"
                        )
                observed_callbacks.append(callback_update)
            finally:
                safety = v1.train_checkpoint_safety_after(
                    probe_model, before
                )
            local_record["safety"] = safety
            checkpoint_record = local_record
            return {
                "target_update": update,
                "live_readout_path": local_record["live_readout"]["path"],
                "live_readout_sha256": local_record["live_readout"]["sha256"],
                "selected_weight_sha256": safety[
                    "selected_weight_sha256"
                ],
                "repeat_guard_pass": local_record["repeat_guard"]["pass"],
                "usable_for_onset": local_record["repeat_guard"][
                    "usable_for_onset"
                ],
            }

        train_config = v1.train_config(seed, [update])
        if (
            int(train_config["max_updates"]) != 24
            or int(train_config["schedule_total_updates"]) != 24
            or train_config["probe_updates"] != [update]
        ):
            raise RuntimeError("V2 full-replay/single-callback config drift")
        metrics = v1.train_completion_model(
            model,
            tokenizer,
            v1.training_rows(trait),
            train_config,
            device,
            attempt / "training",
            checkpoint_callback=callback,
        )
        if observed_callbacks != [update] or checkpoint_record is None:
            raise RuntimeError(
                f"V2 callback inventory mismatch: {observed_callbacks}"
            )
        if observed_keys != expected_set or len(cell_records) != 30:
            raise RuntimeError(
                f"V2 cell inventory mismatch at "
                f"{leaf_label(seed, trait, update)}"
            )
        training_path = attempt / "training/training_metrics.json"
        completion = {
            "schema": (
                "teacher_trait_fingerprint_ontogeny_v2_checkpoint_completion"
            ),
            "identity": identity,
            "created_utc": utc_now(),
            "attempt": relative(attempt),
            "dependency_binding": dependency_binding(),
            "dependency_binding_sha256": compact_hash(
                dependency_binding()
            ),
            "upstream_locks": upstream_locks,
            "sources": sources,
            "context_identity": context["identity"],
            "live_readout": checkpoint_record["live_readout"],
            "checkpoint_record": checkpoint_record,
            "cells": cell_records,
            "expected_cell_count": 30,
            "evaluated_cell_count": sum(
                row["status"] == "evaluated" for row in cell_records
            ),
            "not_applicable_cell_count": sum(
                row["status"] == "not_applicable" for row in cell_records
            ),
            "training_metrics_path": relative(training_path),
            "training_metrics_sha256": file_sha256(training_path),
            "optimizer_updates": metrics["optimizer_updates"],
            "complete": True,
            "v1_causal_scientific_artifacts_consumed": False,
            "preflight_source_sha256": compact_hash(preflight["source"]),
        }
        # The inherited validator proves all 24 optimizer updates, the literal
        # learning-rate sequence, optimizer metadata, and the singleton target
        # callback inventory before anything becomes canonical.
        v1.validate_training_completion(
            attempt,
            completion,
            phase="causal_v2_checkpoint",
            seed=seed,
            probe_updates=[update],
        )
        completion_path = attempt / "completion.json"
        write_json_creation_only(completion_path, completion)
        # Validate every leaf artifact before writing the creation-only pointer.
        validate_leaf_attempt(
            seed,
            trait,
            update,
            attempt,
            expected_completion=completion,
        )
        pointer = {
            "attempt": relative(attempt),
            "completion_sha256": file_sha256(completion_path),
            "identity_sha256": compact_hash(identity),
        }
        write_json_creation_only(
            leaf_root(seed, trait, update) / "canonical.json",
            pointer,
        )
        print(
            f"[v2 leaf] {leaf_label(seed, trait, update)} "
            f"evaluated={completion['evaluated_cell_count']} "
            f"na={completion['not_applicable_cell_count']}",
            flush=True,
        )
        return attempt
    except Exception as error:
        failure_path = attempt / "failure.json"
        if not failure_path.exists():
            write_json_creation_only(
                failure_path,
                {
                    "schema": (
                        "teacher_trait_fingerprint_ontogeny_v2_leaf_failure"
                    ),
                    "created_utc": utc_now(),
                    "identity": identity,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "observed_callbacks": observed_callbacks,
                    "observed_cell_count": len(observed_keys),
                    "retry_permitted": False,
                },
            )
        raise
    finally:
        v1.release_model(model)


def validate_single_safety(
    record: dict[str, Any],
    *,
    seed: int,
    trait: str,
    update: int,
) -> None:
    if (
        record.get("optimizer_update") != update
        or not valid_sha256(record.get("selected_weight_sha256"))
        or record.get("gradients_none") is not True
        or record.get("rng_restored") is not True
        or int(record.get("hook_count", -1)) != 0
        or int(record.get("unselected_parameter_count", -1)) <= 0
    ):
        raise RuntimeError(
            f"V2 safety record mismatch: {seed}/{trait}/u{update}"
        )


def validate_live_readout(
    attempt: Path,
    completion: dict[str, Any],
    *,
    seed: int,
    trait: str,
    update: int,
) -> tuple[dict[str, Any], v1.Readout]:
    record = completion["live_readout"]
    expected_path = attempt / "live_readout.pt"
    path = require_exact_path(
        record["path"], expected_path, label="v2 live readout"
    )
    if file_sha256(path) != record["sha256"]:
        raise RuntimeError(f"V2 live readout file hash mismatch: {path}")
    identity, readout = v1.load_readout(path)
    expected_identity = {
        "lineage": LINEAGE,
        "training_seed": seed,
        "trait": trait,
        "optimizer_update": update,
        **completion["context_identity"],
        "selected_weight_sha256": record["selected_weight_sha256"],
    }
    if identity != expected_identity:
        raise RuntimeError(f"V2 live readout identity mismatch: {path}")
    if (
        tuple(readout.numeric_logits.shape) != (1024, 655)
        or tuple(readout.animal_logits.shape) != (60, 10)
        or not torch.isfinite(readout.numeric_logits).all()
        or not torch.isfinite(readout.animal_logits).all()
        or v1.tensor_sha256(readout.numeric_logits)
        != record["numeric_logits_sha256"]
        or v1.tensor_sha256(readout.animal_logits)
        != record["animal_logits_sha256"]
    ):
        raise RuntimeError(f"V2 live readout tensor contract mismatch: {path}")
    return identity, readout


def factor_is_valid(
    factor: dict[str, Any],
    *,
    shape: tuple[int, int] | None = None,
) -> bool:
    if not isinstance(factor, dict) or set(factor) != {"u", "s", "v"}:
        return False
    u = factor["u"]
    v = factor["v"]
    if not isinstance(u, torch.Tensor) or not isinstance(v, torch.Tensor):
        return False
    if u.ndim != 1 or v.ndim != 1:
        return False
    if shape is not None and (u.numel(), v.numel()) != shape:
        return False
    return bool(
        torch.isfinite(u).all()
        and torch.isfinite(v).all()
        and math.isfinite(float(factor["s"]))
        and math.isclose(
            float(u.float().double().norm()), 1.0, abs_tol=3e-6
        )
        and math.isclose(
            float(v.float().double().norm()), 1.0, abs_tol=3e-6
        )
    )


def reconstructed_factor_sets(
    seed: int,
    trait: str,
    update: int,
    catalog: dict[str, Any],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]] | None],
]:
    donor_seed = v1.crossfit_seed(seed)
    matched = v1.load_endpoint_factors(donor_seed, trait)
    wrong = v1.load_endpoint_factors(
        donor_seed, v1.other_trait(trait)
    )
    audit = catalog["factor_audit"]
    local = catalog["checkpoint_local_factors"]
    projections = audit.get("crossfit_projections")
    if set(projections or {}) != set(v1.selected_names()):
        raise RuntimeError("V2 crossfit projection module inventory mismatch")
    loaded: dict[str, dict[str, Any]] = {}
    wrong_factors: dict[str, dict[str, Any]] = {}
    for name in v1.selected_names():
        coefficient = float(projections[name]["signed_projection"])
        loaded[name] = {
            "u": matched["factors"][name]["u"].float(),
            "s": coefficient,
            "v": matched["factors"][name]["v"].float(),
        }
        wrong_factors[name] = {
            "u": wrong["factors"][name]["u"].float(),
            "s": coefficient,
            "v": wrong["factors"][name]["v"].float(),
        }
    constructions = {
        "checkpoint_local": local,
        "crossfit_endpoint_loaded": loaded,
        "wrong_trait": wrong_factors,
    }
    factor_sets = v1.checkpoint_factor_sets(
        constructions,
        seed=seed,
        trait=trait,
        update=update,
    )
    return factor_sets, constructions


def validate_factor_catalog(
    attempt: Path,
    completion: dict[str, Any],
    *,
    seed: int,
    trait: str,
    update: int,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]] | None],
]:
    record = completion["checkpoint_record"]["factor_catalog"]
    path = require_exact_path(
        record["path"], attempt / "factors.pt", label="v2 factor catalog"
    )
    if file_sha256(path) != record["sha256"]:
        raise RuntimeError(f"V2 factor catalog hash mismatch: {path}")
    catalog = torch.load(path, map_location="cpu", weights_only=True)
    if (
        set(catalog)
        != {
            "schema",
            "identity",
            "checkpoint_local_factors",
            "checkpoint_local_witnesses",
            "crossfit_projection_witnesses",
            "factor_audit",
            "factor_manifests",
        }
        or catalog.get("schema")
        != "teacher_trait_fingerprint_ontogeny_v2_factor_catalog"
    ):
        raise RuntimeError(f"V2 factor catalog schema mismatch: {path}")
    expected_identity = {
        **checkpoint_identity(seed, trait, update),
        "selected_weight_sha256": completion["live_readout"][
            "selected_weight_sha256"
        ],
    }
    if catalog.get("identity") != expected_identity:
        raise RuntimeError(f"V2 factor catalog identity mismatch: {path}")
    audit = catalog.get("factor_audit")
    if (
        not isinstance(audit, dict)
        or set(audit.get("local_audits", {})) != set(v1.selected_names())
        or set(audit.get("crossfit_projections", {}))
        != set(v1.selected_names())
        or not finite_tree(audit)
        or completion["checkpoint_record"].get("factor_audit") != audit
    ):
        raise RuntimeError(f"V2 factor audit mismatch: {path}")
    projection_witnesses = catalog.get("crossfit_projection_witnesses")
    if (
        not isinstance(projection_witnesses, dict)
        or set(projection_witnesses) != set(v1.selected_names())
    ):
        raise RuntimeError(
            f"V2 crossfit witness inventory mismatch: {path}"
        )
    donor_seed = v1.crossfit_seed(seed)
    matched_endpoint = v1.load_endpoint_factors(donor_seed, trait)
    for name in v1.selected_names():
        witness = projection_witnesses[name]
        if set(witness) != {
            "delta_times_matched_v",
            "delta_times_matched_v_sha256",
            "matched_endpoint_u_sha256",
            "matched_endpoint_v_sha256",
        }:
            raise RuntimeError(
                f"V2 crossfit witness schema mismatch: {path}:{name}"
            )
        endpoint_factor = matched_endpoint["factors"][name]
        endpoint_u = (
            endpoint_factor["u"].detach().float().contiguous().cpu()
        )
        endpoint_v = (
            endpoint_factor["v"].detach().float().contiguous().cpu()
        )
        delta_times_v = witness["delta_times_matched_v"]
        if (
            not isinstance(delta_times_v, torch.Tensor)
            or delta_times_v.dtype != torch.float32
            or tuple(delta_times_v.shape) != tuple(endpoint_u.shape)
            or not torch.isfinite(delta_times_v).all()
            or witness["delta_times_matched_v_sha256"]
            != v1.tensor_sha256(delta_times_v)
            or witness["matched_endpoint_u_sha256"]
            != v1.tensor_sha256(endpoint_u)
            or witness["matched_endpoint_v_sha256"]
            != v1.tensor_sha256(endpoint_v)
        ):
            raise RuntimeError(
                f"V2 crossfit witness tensor/hash mismatch: {path}:{name}"
            )
        witnessed_projection = float(
            torch.dot(endpoint_u, delta_times_v.float())
        )
        projection_record = audit["crossfit_projections"][name]
        recorded_projection = float(
            projection_record["signed_projection"]
        )
        endpoint_singular = float(endpoint_factor["s"])
        if (
            witnessed_projection != recorded_projection
            or float(
                projection_record["matched_endpoint_singular_value"]
            )
            != endpoint_singular
            or not math.isclose(
                float(
                    projection_record[
                        "fraction_of_crossfit_endpoint_singular_value"
                    ]
                ),
                recorded_projection / max(endpoint_singular, 1e-30),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise RuntimeError(
                f"V2 crossfit witness/projection mismatch: {path}:{name}"
            )
        if update == 0 and (
            torch.count_nonzero(delta_times_v).item() != 0
            or recorded_projection != 0.0
            or float(
                projection_record[
                    "fraction_of_crossfit_endpoint_singular_value"
                ]
            )
            != 0.0
        ):
            raise RuntimeError(
                f"V2 u0 crossfit witness is not exactly zero: {path}:{name}"
            )
    local = catalog.get("checkpoint_local_factors")
    witnesses = catalog.get("checkpoint_local_witnesses")
    local_identifiable = bool(audit.get("local_identifiable"))
    if local_identifiable:
        if (
            not isinstance(local, dict)
            or set(local) != set(v1.selected_names())
            or not isinstance(witnesses, dict)
            or set(witnesses) != set(v1.selected_names())
        ):
            raise RuntimeError(f"V2 local factor inventory mismatch: {path}")
        limits = v1.protocol()["circuit"]["local_svd"]
        for name in v1.selected_names():
            factor = local[name]
            if not factor_is_valid(factor):
                raise RuntimeError(f"V2 local factor malformed: {path}:{name}")
            local_audit = audit["local_audits"][name]
            witness = witnesses[name]
            if set(witness) != {"delta_v", "delta_transpose_u"}:
                raise RuntimeError(f"V2 local witness malformed: {path}:{name}")
            delta_v = witness["delta_v"].float()
            delta_t_u = witness["delta_transpose_u"].float()
            if (
                tuple(delta_v.shape) != tuple(factor["u"].shape)
                or tuple(delta_t_u.shape) != tuple(factor["v"].shape)
                or not torch.isfinite(delta_v).all()
                or not torch.isfinite(delta_t_u).all()
            ):
                raise RuntimeError(
                    f"V2 local witness tensor mismatch: {path}:{name}"
                )
            singular = float(factor["s"])
            left = float(
                (delta_v - singular * factor["u"].float()).norm()
            ) / max(abs(singular), 1e-30)
            right = float(
                (delta_t_u - singular * factor["v"].float()).norm()
            ) / max(abs(singular), 1e-30)
            if (
                local_audit.get("identifiable") is not True
                or float(local_audit["singular_gap"])
                < float(limits["singular_gap_minimum"])
                or max(left, right)
                > float(limits["residual_relative_maximum"])
                or not math.isclose(
                    left,
                    float(local_audit["left_residual_relative"]),
                    rel_tol=3e-5,
                    abs_tol=3e-7,
                )
                or not math.isclose(
                    right,
                    float(local_audit["right_residual_relative"]),
                    rel_tol=3e-5,
                    abs_tol=3e-7,
                )
            ):
                raise RuntimeError(
                    f"V2 local factor witness/audit mismatch: {path}:{name}"
                )
    elif local is not None or witnesses != {}:
        raise RuntimeError(f"V2 unidentifiable local payload mismatch: {path}")
    factor_sets, constructions = reconstructed_factor_sets(
        seed, trait, update, catalog
    )
    for identifier, factors in factor_sets.items():
        if set(factors) != set(v1.selected_names()):
            raise RuntimeError(
                f"V2 factor set module mismatch: {path}:{identifier}"
            )
        for name in v1.selected_names():
            if not factor_is_valid(factors[name]):
                raise RuntimeError(
                    f"V2 factor set malformed: {path}:{identifier}:{name}"
                )
    manifests = {
        identifier: v1.factor_manifest(factors)
        for identifier, factors in factor_sets.items()
    }
    if (
        catalog.get("factor_manifests") != manifests
        or record.get("factor_set_ids") != sorted(manifests)
        or record.get("factor_manifest_sha256") != compact_hash(manifests)
    ):
        raise RuntimeError(f"V2 factor manifest mismatch: {path}")
    if update == 0:
        if constructions["checkpoint_local"] is not None:
            raise RuntimeError("V2 u0 local construction must be unavailable")
        expected_u0_ids = {
            v1.factor_set_id(
                "crossfit_endpoint_loaded", "real", -1
            ),
            v1.factor_set_id(
                "crossfit_endpoint_loaded", "wrong_trait", -1
            ),
            *{
                v1.factor_set_id(
                    "crossfit_endpoint_loaded", "sham", draw
                )
                for draw in range(
                    int(v1.protocol()["circuit"]["sham_draws"])
                )
            },
        }
        if set(factor_sets) != expected_u0_ids or any(
            float(factor["s"]) != 0.0
            for factors in factor_sets.values()
            for factor in factors.values()
        ):
            raise RuntimeError(
                f"V2 u0 factor set/amplitude mismatch: {path}"
            )
    return catalog, factor_sets, constructions


def pooled_cosine_np(
    dot: np.ndarray,
    field_norm: np.ndarray,
    effect_norm: np.ndarray,
) -> float:
    return float(
        np.asarray(dot, dtype=np.float64).sum()
        / max(
            math.sqrt(
                float(np.asarray(field_norm, dtype=np.float64).sum())
                * float(np.asarray(effect_norm, dtype=np.float64).sum())
            ),
            1e-30,
        )
    )


def assert_close(
    observed: Any,
    expected: float,
    *,
    label: str,
    tolerance: float = 2e-10,
) -> None:
    if not math.isclose(
        float(observed),
        float(expected),
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise RuntimeError(
            f"V2 derived metric mismatch {label}: {observed} != {expected}"
        )


def validate_metrics_against_arrays(
    payload: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    metrics = payload["metrics"]
    numeric = metrics["numeric"]
    behavior = metrics["behavior"]
    hard = metrics["hard"]
    assert_close(
        numeric["native_paired_mean_js"],
        float(arrays["numeric_native_js"].mean()),
        label="native_paired_mean_js",
    )
    assert_close(
        numeric["cell_paired_mean_js"],
        float(arrays["numeric_cell_js"].mean()),
        label="cell_paired_mean_js",
    )
    mean_progress = float(arrays["numeric_oriented_js_progress"].mean())
    assert_close(
        numeric["oriented_mean_js_progress"],
        mean_progress,
        label="oriented_mean_js_progress",
    )
    assert_close(
        numeric["oriented_js_progress_fraction"],
        mean_progress
        / max(float(arrays["numeric_native_js"].mean()), 1e-30),
        label="oriented_js_progress_fraction",
    )
    for section, prefix in (
        ("centered_logit_field", "logit"),
        ("restricted_probability_field", "probability"),
    ):
        record = numeric[section]
        assert_close(
            record["cosine"],
            pooled_cosine_np(
                arrays[f"{prefix}_field_dot"],
                arrays[f"{prefix}_field_norm"],
                arrays[f"{prefix}_effect_norm"],
            ),
            label=f"{prefix}_cosine",
        )
        assert_close(
            record["context_centered_cosine"],
            pooled_cosine_np(
                arrays[f"{prefix}_context_field_dot"],
                arrays[f"{prefix}_context_field_norm"],
                arrays[f"{prefix}_context_effect_norm"],
            ),
            label=f"{prefix}_context_centered_cosine",
        )
        assert_close(
            record["capture_slope"],
            float(arrays[f"{prefix}_field_dot"].sum())
            / max(float(arrays[f"{prefix}_field_norm"].sum()), 1e-30),
            label=f"{prefix}_capture_slope",
        )
        assert_close(
            record["context_centered_capture_slope"],
            float(arrays[f"{prefix}_context_field_dot"].sum())
            / max(
                float(
                    arrays[f"{prefix}_context_field_norm"].sum()
                ),
                1e-30,
            ),
            label=f"{prefix}_context_capture_slope",
        )
    mean_gap = float(arrays["behavior_native_gap"].mean())
    mean_effect = float(arrays["behavior_oriented_effect"].mean())
    assert_close(
        behavior["mean_native_paired_target_gap"],
        mean_gap,
        label="behavior_native_gap",
    )
    assert_close(
        behavior["oriented_mean_target_pair_effect"],
        mean_effect,
        label="behavior_oriented_effect",
    )
    assert_close(
        behavior["oriented_target_pair_mediation_fraction"],
        mean_effect / max(abs(mean_gap), 1e-30),
        label="behavior_mediation_fraction",
    )
    assert_close(
        behavior["oriented_mean_nine_animal_margin_effect"],
        float(arrays["behavior_oriented_margin_effect"].mean()),
        label="behavior_margin_effect",
    )
    hard_event = arrays["hard_event"].astype(bool)
    recovery = arrays["hard_oriented_recovery"].astype(bool)
    count = int(hard_event.sum())
    rate = float(recovery[hard_event].mean()) if count else 0.0
    if (
        int(hard["paired_argmax_event_count"]) != count
        or bool(hard["powered"])
        != (
            count
            >= int(v1.protocol()["analysis"]["hard_event_minimum"])
        )
    ):
        raise RuntimeError("V2 hard-event count/power mismatch")
    assert_close(
        hard["oriented_recovery_or_preservation_rate"],
        rate,
        label="hard_recovery_rate",
    )


def validate_cells(
    attempt: Path,
    completion: dict[str, Any],
    *,
    seed: int,
    trait: str,
    update: int,
    factor_sets: dict[str, dict[str, dict[str, Any]]],
    constructions: dict[str, dict[str, dict[str, Any]] | None],
) -> None:
    expected_keys = expected_checkpoint_keys(seed, trait, update)
    records = completion.get("cells")
    if (
        not isinstance(records, list)
        or len(records) != 30
        or [key_tuple(row["key"]) for row in records]
        != [key_tuple(key) for key in expected_keys]
    ):
        raise RuntimeError(
            f"V2 cell record inventory mismatch: {seed}/{trait}/u{update}"
        )
    seen: set[tuple[Any, ...]] = set()
    identity_hash = compact_hash(completion["identity"])
    for expected_key, record in zip(expected_keys, records):
        logical = key_tuple(expected_key)
        if logical in seen or key_tuple(record["key"]) != logical:
            raise RuntimeError("V2 duplicate or misordered cell record")
        seen.add(logical)
        stem = v1.cell_stem(expected_key)
        path = require_exact_path(
            record["path"],
            attempt / "cells" / f"{stem}.json",
            label="v2 causal cell",
        )
        if file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"V2 causal cell hash mismatch: {path}")
        payload = json.loads(path.read_text())
        if (
            payload.get("schema")
            != "teacher_trait_fingerprint_ontogeny_v2_cell"
            or payload.get("experiment_id") != EXPERIMENT_ID
            or payload.get("checkpoint_identity_sha256") != identity_hash
            or key_tuple(payload.get("key", {})) != logical
            or payload.get("status") != record.get("status")
        ):
            raise RuntimeError(f"V2 causal cell identity mismatch: {path}")
        construction = str(expected_key["construction"])
        real = constructions[construction]
        if real is None:
            expected_status = "not_applicable"
            expected_reason = "checkpoint_local_rank1_unidentifiable"
            expected_identifier = None
            expected_manifest = None
        else:
            expected_identifier = expected_factor_set_id(expected_key)
            expected_manifest = v1.factor_manifest(
                factor_sets[expected_identifier]
            )
            if all(
                abs(float(factor["s"])) <= 1e-30
                for factor in real.values()
            ):
                expected_status = "not_applicable"
                expected_reason = "zero_checkpoint_component"
            else:
                expected_status = "evaluated"
                expected_reason = None
        if (
            payload["status"] != expected_status
            or payload.get("reason") != expected_reason
            or payload.get("factor_set_id") != expected_identifier
            or payload.get("factor_manifest") != expected_manifest
        ):
            raise RuntimeError(f"V2 cell applicability/factor mismatch: {path}")
        if expected_status == "not_applicable":
            if (
                payload.get("factor_record")
                != completion["checkpoint_record"]["factor_audit"]
                or payload.get("metrics") is not None
                or payload.get("arrays_path") is not None
                or "arrays_path" in record
            ):
                raise RuntimeError(f"Malformed v2 N/A cell: {path}")
            continue
        factors = factor_sets[expected_identifier]
        if (
            payload.get("factor_record") != v1.factor_summary(factors)
            or not finite_tree(payload.get("metrics"))
        ):
            raise RuntimeError(f"V2 evaluated factor/metric mismatch: {path}")
        expected_array_path = attempt / "cells" / f"{stem}.npz"
        array_path = require_exact_path(
            record["arrays_path"],
            expected_array_path,
            label="v2 causal cell arrays",
        )
        if (
            payload.get("arrays_path") != record["arrays_path"]
            or payload.get("arrays_sha256") != record["arrays_sha256"]
            or file_sha256(array_path) != record["arrays_sha256"]
        ):
            raise RuntimeError(f"V2 causal array hash/path mismatch: {array_path}")
        with np.load(array_path) as archive:
            if set(archive.files) != set(v1.CAUSAL_ARRAY_LENGTHS):
                raise RuntimeError(f"V2 causal array keys mismatch: {array_path}")
            arrays = {name: archive[name].copy() for name in archive.files}
        for name, length in v1.CAUSAL_ARRAY_LENGTHS.items():
            value = arrays[name]
            if value.shape != (length,) or not np.all(np.isfinite(value)):
                raise RuntimeError(
                    f"V2 causal array contract mismatch: {array_path}:{name}"
                )
        validate_metrics_against_arrays(payload, arrays)
    if update == 0:
        local_records = [
            record
            for record in records
            if record["key"]["construction"] == "checkpoint_local"
        ]
        crossfit_records = [
            record
            for record in records
            if record["key"]["construction"]
            == "crossfit_endpoint_loaded"
        ]
        if (
            len(local_records) != 14
            or len(crossfit_records) != 16
            or any(
                record["status"] != "not_applicable"
                for record in records
            )
        ):
            raise RuntimeError(
                "V2 u0 must contain 14 local and 16 crossfit N/A cells"
            )


def validate_leaf_attempt(
    seed: int,
    trait: str,
    update: int,
    attempt: Path,
    *,
    expected_completion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    require_within(
        attempt,
        leaf_root(seed, trait, update),
        label="v2 leaf attempt",
    )
    if attempt.resolve() != leaf_attempt(seed, trait, update).resolve():
        raise RuntimeError(f"V2 attempt path mismatch: {attempt}")
    if (attempt / "failure.json").exists():
        raise RuntimeError(
            f"V2 canonical/in-progress completion cannot coexist with failure: "
            f"{attempt}"
        )
    allowed_root_children = {"attempt_001", "canonical.json"}
    observed_root_children = {
        path.name for path in leaf_root(seed, trait, update).iterdir()
    }
    if not observed_root_children.issubset(allowed_root_children):
        raise RuntimeError(
            f"Unexpected v2 leaf-root artifacts: "
            f"{observed_root_children - allowed_root_children}"
        )
    completion_path = attempt / "completion.json"
    if not completion_path.is_file():
        raise RuntimeError(f"Missing v2 completion: {completion_path}")
    completion = json.loads(completion_path.read_text())
    if expected_completion is not None and completion != expected_completion:
        raise RuntimeError("Serialized v2 completion differs from live completion")
    if (
        completion.get("schema")
        != "teacher_trait_fingerprint_ontogeny_v2_checkpoint_completion"
        or completion.get("identity")
        != checkpoint_identity(seed, trait, update)
        or completion.get("attempt") != relative(attempt)
        or completion.get("dependency_binding") != dependency_binding()
        or completion.get("dependency_binding_sha256")
        != compact_hash(dependency_binding())
        or completion.get("upstream_locks") != validated_upstream_locks()
        or completion.get("sources") != leaf_sources(seed, trait, update)
        or completion.get("v1_causal_scientific_artifacts_consumed")
        is not False
        or completion.get("complete") is not True
        or completion.get("optimizer_updates") != 24
    ):
        raise RuntimeError(
            f"V2 completion identity/provenance mismatch: {completion_path}"
        )
    preflight = require_preflight()
    if completion.get("preflight_source_sha256") != compact_hash(
        preflight["source"]
    ):
        raise RuntimeError("V2 completion/preflight source cross-link mismatch")
    training = v1.validate_training_completion(
        attempt,
        completion,
        phase="causal_v2_checkpoint",
        seed=seed,
        probe_updates=[update],
    )
    checkpoint = completion.get("checkpoint_record")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("optimizer_update") != update
        or completion.get("live_readout") != checkpoint.get("live_readout")
    ):
        raise RuntimeError(f"V2 checkpoint record mismatch: {completion_path}")
    safety = checkpoint.get("safety")
    validate_single_safety(
        {"optimizer_update": update, **safety},
        seed=seed,
        trait=trait,
        update=update,
    )
    if (
        completion["live_readout"]["selected_weight_sha256"]
        != safety["selected_weight_sha256"]
    ):
        raise RuntimeError("V2 live/safety selected-weight mismatch")
    if update == 0 and (
        completion["live_readout"]["selected_weight_sha256"]
        != completion["sources"]["current_native"]["readout"][
            "selected_weight_sha256"
        ]
    ):
        raise RuntimeError(
            "V2 live u0 selected weights differ from sealed native u0"
        )
    _, live = validate_live_readout(
        attempt,
        completion,
        seed=seed,
        trait=trait,
        update=update,
    )
    current_native = load_source_readout(
        completion["sources"]["current_native"]
    )
    paired_other = load_source_readout(
        completion["sources"]["paired_other_native"]
    )
    repeat = v1.replay_repeat_metrics(live, current_native, trait=trait)
    repeat_relative = v1.relative_replay_guard(
        repeat,
        current_native,
        paired_other,
        update=update,
    )
    expected_repeat = {**repeat, **repeat_relative}
    if (
        checkpoint.get("repeat_guard") != expected_repeat
        or repeat.get("pass") is not True
        or (
            update == 0
            and repeat_relative.get("relative_or_u0_pass") is not True
        )
    ):
        raise RuntimeError(f"V2 replay guard mismatch: {completion_path}")
    metric_checkpoint = training["checkpoint_metrics"][0]
    expected_metric_checkpoint = {
        "optimizer_update": update,
        "target_update": update,
        "live_readout_path": completion["live_readout"]["path"],
        "live_readout_sha256": completion["live_readout"]["sha256"],
        "selected_weight_sha256": safety["selected_weight_sha256"],
        "repeat_guard_pass": True,
        "usable_for_onset": expected_repeat["usable_for_onset"],
    }
    if metric_checkpoint != expected_metric_checkpoint:
        raise RuntimeError("V2 training/callback cross-link mismatch")
    _, factor_sets, constructions = validate_factor_catalog(
        attempt,
        completion,
        seed=seed,
        trait=trait,
        update=update,
    )
    validate_cells(
        attempt,
        completion,
        seed=seed,
        trait=trait,
        update=update,
        factor_sets=factor_sets,
        constructions=constructions,
    )
    evaluated = sum(
        row["status"] == "evaluated" for row in completion["cells"]
    )
    not_applicable = sum(
        row["status"] == "not_applicable" for row in completion["cells"]
    )
    if (
        completion.get("expected_cell_count") != 30
        or completion.get("evaluated_cell_count") != evaluated
        or completion.get("not_applicable_cell_count") != not_applicable
        or evaluated + not_applicable != 30
    ):
        raise RuntimeError("V2 completion cell counts mismatch")
    if update == 0 and (evaluated != 0 or not_applicable != 30):
        raise RuntimeError("V2 u0 completion must be exactly 30 N/A cells")
    expected_files = {
        (attempt / "completion.json").resolve(),
        (attempt / "live_readout.pt").resolve(),
        (attempt / "factors.pt").resolve(),
        (attempt / "training/training_metrics.json").resolve(),
    }
    for record in completion["cells"]:
        expected_files.add(
            repository_path(record["path"], label="v2 cell inventory")
        )
        if record["status"] == "evaluated":
            expected_files.add(
                repository_path(
                    record["arrays_path"], label="v2 array inventory"
                )
            )
    observed_files = {
        path.resolve()
        for path in attempt.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if observed_files != expected_files:
        raise RuntimeError(
            f"V2 leaf file inventory mismatch: "
            f"missing={sorted(str(path) for path in expected_files-observed_files)} "
            f"extra={sorted(str(path) for path in observed_files-expected_files)}"
        )
    return completion


def load_leaf(
    seed: int,
    trait: str,
    update: int,
) -> tuple[Path, dict[str, Any]]:
    attempt = canonical_attempt(seed, trait, update)
    completion = validate_leaf_attempt(seed, trait, update, attempt)
    return attempt, completion


def u0_equivalence_manifest() -> dict[str, Any]:
    members: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        for trait in TRAITS:
            label = f"s{seed}:{trait}"
            native_attempt, native_completion = v1.load_native_completion(
                seed, trait
            )
            native_record = native_completion["readouts"][0]
            native_path = repository_path(
                native_record["path"], label=f"v1 native u0 {label}"
            )
            native_identity, native_readout = v1.load_readout(native_path)
            members[f"v1_native:{label}"] = {
                "path": native_path,
                "identity": native_identity,
                "context_identity": native_completion["context_identity"],
                "readout": native_readout,
            }
            attempt, completion = load_leaf(seed, trait, 0)
            live_path = repository_path(
                completion["live_readout"]["path"],
                label=f"v2 isolated u0 {label}",
            )
            live_identity, live_readout = v1.load_readout(live_path)
            members[f"v2_isolated:{label}"] = {
                "path": live_path,
                "identity": live_identity,
                "context_identity": completion["context_identity"],
                "readout": live_readout,
            }
    result = v1.u0_equivalence(members)
    if result["member_count"] != 8 or result["pair_count"] != 28:
        raise RuntimeError("V2 u0 equivalence inventory mismatch")
    return result


def checkpoint_lock_manifest(*, created_utc: str) -> dict[str, Any]:
    preflight = require_preflight()
    leaves: dict[str, Any] = {}
    observed_global_keys: list[dict[str, Any]] = []
    for leaf in expected_leaf_keys():
        seed = int(leaf["training_seed"])
        trait = str(leaf["trait"])
        update = int(leaf["target_update"])
        label = leaf_label(seed, trait, update)
        attempt, completion = load_leaf(seed, trait, update)
        pointer = leaf_root(seed, trait, update) / "canonical.json"
        observed_global_keys.extend(
            [dict(record["key"]) for record in completion["cells"]]
        )
        leaves[label] = {
            "identity": dict(leaf),
            "attempt": relative(attempt),
            "canonical_pointer_sha256": file_sha256(pointer),
            "completion_sha256": file_sha256(attempt / "completion.json"),
            "training_metrics_sha256": completion[
                "training_metrics_sha256"
            ],
            "live_readout_sha256": completion["live_readout"]["sha256"],
            "live_numeric_logits_sha256": completion["live_readout"][
                "numeric_logits_sha256"
            ],
            "live_animal_logits_sha256": completion["live_readout"][
                "animal_logits_sha256"
            ],
            "factor_catalog_sha256": completion["checkpoint_record"][
                "factor_catalog"
            ]["sha256"],
            "factor_manifest_sha256": completion["checkpoint_record"][
                "factor_catalog"
            ]["factor_manifest_sha256"],
            "cell_manifest_sha256": compact_hash(completion["cells"]),
            "expected_cell_count": completion["expected_cell_count"],
            "evaluated_cell_count": completion["evaluated_cell_count"],
            "not_applicable_cell_count": completion[
                "not_applicable_cell_count"
            ],
        }
    expected = expected_global_keys()
    if observed_global_keys != expected:
        raise RuntimeError("V2 global logical-key order or inventory mismatch")
    if len({key_tuple(key) for key in observed_global_keys}) != 1080:
        raise RuntimeError("V2 global logical-key uniqueness mismatch")
    return {
        "schema": "teacher_trait_fingerprint_ontogeny_v2_checkpoint_lock",
        "experiment_id": EXPERIMENT_ID,
        "created_utc": created_utc,
        "config_sha256": file_sha256(CONFIG_PATH),
        "script_sha256": file_sha256(SCRIPT_PATH),
        "preflight_sha256": file_sha256(PREFLIGHT_PATH),
        "preflight_source_sha256": compact_hash(preflight["source"]),
        "dependency_binding": dependency_binding(),
        "dependency_binding_sha256": compact_hash(dependency_binding()),
        "upstream_locks": validated_upstream_locks(),
        "leaves": leaves,
        "expected_leaf_count": 36,
        "leaf_key_sha256": compact_hash(expected_leaf_keys()),
        "global_expected_key_count": 1080,
        "global_expected_key_sha256": compact_hash(expected),
        "u0_equivalence": u0_equivalence_manifest(),
        "v1_causal_scientific_artifacts_consumed": False,
    }


def seal_checkpoints() -> dict[str, Any]:
    require_preflight()
    if CHECKPOINT_LOCK_PATH.exists():
        return require_checkpoint_lock()
    lock = checkpoint_lock_manifest(created_utc=utc_now())
    write_json_creation_only(CHECKPOINT_LOCK_PATH, lock)
    print(
        f"[v2 seal] leaves={len(lock['leaves'])} "
        f"cells={lock['global_expected_key_count']}",
        flush=True,
    )
    return lock


def require_checkpoint_lock() -> dict[str, Any]:
    if not CHECKPOINT_LOCK_PATH.is_file():
        raise RuntimeError("Run all 36 v2 leaves and --seal")
    observed = json.loads(CHECKPOINT_LOCK_PATH.read_text())
    expected = checkpoint_lock_manifest(
        created_utc=str(observed.get("created_utc"))
    )
    if observed != expected:
        raise RuntimeError("V2 checkpoint lock or bound artifact drift")
    return observed


def run_all_leaves() -> None:
    require_preflight()
    if CHECKPOINT_LOCK_PATH.exists():
        require_checkpoint_lock()
        print("[v2 all] sealed campaign reused", flush=True)
        return
    for leaf in expected_leaf_keys():
        seed = int(leaf["training_seed"])
        trait = str(leaf["trait"])
        update = int(leaf["target_update"])
        root = leaf_root(seed, trait, update)
        if (root / "canonical.json").is_file():
            load_leaf(seed, trait, update)
            print(
                f"[v2 all] reuse {leaf_label(seed, trait, update)}",
                flush=True,
            )
            continue
        if root.exists():
            raise RuntimeError(
                f"V2 no-retry policy blocks incomplete leaf: {root}"
            )
        run_leaf(seed, trait, update)


def artifact_inventory() -> dict[str, Any]:
    leaves: dict[str, Any] = {}
    complete = True
    handled = (
        RuntimeError,
        KeyError,
        ValueError,
        TypeError,
        FileNotFoundError,
        json.JSONDecodeError,
    )
    preflight_status: dict[str, Any]
    try:
        frozen = require_preflight()
        preflight_status = {
            "valid": True,
            "sha256": file_sha256(PREFLIGHT_PATH),
            "source_sha256": compact_hash(frozen["source"]),
        }
    except handled as error:
        preflight_status = {"valid": False, "error": str(error)}
        complete = False
    cell_count = 0
    evaluated_count = 0
    not_applicable_count = 0
    for leaf in expected_leaf_keys():
        seed = int(leaf["training_seed"])
        trait = str(leaf["trait"])
        update = int(leaf["target_update"])
        label = leaf_label(seed, trait, update)
        try:
            attempt, completion = load_leaf(seed, trait, update)
            count = len(completion["cells"])
            evaluated = int(completion["evaluated_cell_count"])
            not_applicable = int(
                completion["not_applicable_cell_count"]
            )
            cell_count += count
            evaluated_count += evaluated
            not_applicable_count += not_applicable
            leaves[label] = {
                "valid": True,
                "attempt": relative(attempt),
                "cells": count,
                "evaluated": evaluated,
                "not_applicable": not_applicable,
            }
        except handled as error:
            leaves[label] = {"valid": False, "error": str(error)}
            complete = False
    lock_status: dict[str, Any]
    try:
        lock = require_checkpoint_lock()
        lock_status = {
            "valid": True,
            "sha256": file_sha256(CHECKPOINT_LOCK_PATH),
            "leaves": len(lock["leaves"]),
            "cells": lock["global_expected_key_count"],
            "u0_all_pairs_pass": lock["u0_equivalence"][
                "all_pairs_pass"
            ],
        }
    except handled as error:
        lock_status = {"valid": False, "error": str(error)}
        complete = False
    if cell_count != 1080:
        complete = False
    return {
        "schema": "teacher_trait_fingerprint_ontogeny_v2_inventory",
        "preflight": preflight_status,
        "leaves": leaves,
        "checkpoint_lock": lock_status,
        "expected_leaf_count": 36,
        "valid_leaf_count": sum(row["valid"] for row in leaves.values()),
        "expected_cell_count": 1080,
        "observed_cell_count": cell_count,
        "evaluated_cell_count": evaluated_count,
        "not_applicable_cell_count": not_applicable_count,
        "global_key_sha256": compact_hash(expected_global_keys()),
        "v1_causal_scientific_artifacts_consumed": False,
        "complete": complete,
    }


def load_analysis_inputs() -> tuple[
    dict[
        tuple[Any, ...],
        tuple[dict[str, Any], dict[str, np.ndarray] | None],
    ],
    dict[tuple[int, str], dict[str, Any]],
]:
    require_checkpoint_lock()
    cells: dict[
        tuple[Any, ...],
        tuple[dict[str, Any], dict[str, np.ndarray] | None],
    ] = {}
    completions: dict[tuple[int, str], dict[str, Any]] = {
        (seed, trait): {"checkpoint_records": []}
        for seed in SEEDS
        for trait in TRAITS
    }
    for leaf in expected_leaf_keys():
        seed = int(leaf["training_seed"])
        trait = str(leaf["trait"])
        update = int(leaf["target_update"])
        _, completion = load_leaf(seed, trait, update)
        completions[(seed, trait)]["checkpoint_records"].append(
            completion["checkpoint_record"]
        )
        for record in completion["cells"]:
            key = key_tuple(record["key"])
            if key in cells:
                raise RuntimeError(f"Duplicate v2 analysis key: {key}")
            payload = json.loads(
                repository_path(
                    record["path"], label="v2 analysis cell"
                ).read_text()
            )
            arrays = None
            if record["status"] == "evaluated":
                with np.load(
                    repository_path(
                        record["arrays_path"],
                        label="v2 analysis arrays",
                    )
                ) as archive:
                    arrays = {
                        name: archive[name].copy()
                        for name in archive.files
                    }
            cells[key] = (payload, arrays)
    if set(cells) != {key_tuple(key) for key in expected_global_keys()}:
        raise RuntimeError("V2 analysis global key inventory mismatch")
    for completion in completions.values():
        if [
            int(row["optimizer_update"])
            for row in completion["checkpoint_records"]
        ] != list(TARGET_UPDATES):
            raise RuntimeError("V2 synthesized checkpoint order mismatch")
    return cells, completions


def render_markdown(result: dict[str, Any]) -> str:
    onset = result["onsets"]
    classification = onset["classification"]

    def value(section: str, key: str) -> str:
        item = onset[section][key]
        stable = item["stable_onset"]
        confirmed = item["first_confirmed"]
        return (
            f"{stable if stable is not None else '—'} "
            f"(first-confirmed "
            f"{confirmed if confirmed is not None else '—'}; "
            f"interval {item['onset_interval']})"
        )

    return "\n".join(
        [
            "# Teacher trait–fingerprint ontogeny v2",
            "",
            "Causal checkpoints were measured in independent complete "
            "24-update replays with exactly one target callback each.",
            "",
            "## Registered onset results",
            "",
            "| Timestamp | Result |",
            "|---|---:|",
            f"| Base-relative fingerprint appearance | "
            f"{value('native', 'fingerprint_appearance')} |",
            f"| Trait behavior | {value('native', 'trait_behavior')} |",
            f"| Wolf/lion field separation | "
            f"{value('native', 'trait_specific_field')} |",
            f"| Cross-seed trait identity | "
            f"{value('native', 'identity')} |",
            f"| Checkpoint-local causal entanglement | "
            f"{value('causal', 'checkpoint_local')} |",
            f"| Crossfit endpoint consolidation | "
            f"{value('causal', 'crossfit_endpoint_loaded')} |",
            f"| Any causal construction | "
            f"{value('causal', 'any_construction')} |",
            "",
            "## Classification",
            "",
            f"- Field axis: `{classification['field_axis']}`",
            f"- Causal axis: `{classification['causal_axis']}`",
            f"- Hard-event qualifier: "
            f"`{classification['hard_qualifier']}`",
            "",
            "## Integrity",
            "",
            f"- Artifact integrity valid: "
            f"`{result['status']['artifact_integrity_valid']}`",
            f"- Analysis implementation valid: "
            f"`{result['status']['analysis_implementation_valid']}`",
            "- Primary classification valid: pending independent verifier",
            "- Overall pass: pending independent verifier",
            "",
        ]
    )


def analyze() -> dict[str, Any]:
    require_preflight()
    lock = require_checkpoint_lock()
    inventory = artifact_inventory()
    if not inventory["complete"]:
        raise RuntimeError(f"V2 artifact inventory incomplete: {inventory}")
    if OUT_JSON.exists() or OUT_MD.exists():
        raise RuntimeError("V2 aggregate outputs are creation-only")
    records: list[v1.BootstrapRecord] = []
    lookup: dict[str, v1.BootstrapRecord] = {}
    indices = v1.bootstrap_index_bank()
    native_raw = v1.native_analysis_inputs()
    native_summaries, native_gate_records = v1.build_native_analysis(
        native_raw, records, lookup, indices
    )
    cells, completions = load_analysis_inputs()
    causal_summaries, causal_gate_records = v1.build_causal_analysis(
        cells, completions, records, lookup, indices
    )
    hard_summaries, hard_gate_records = v1.build_hard_analysis(
        cells, completions, records, lookup, indices
    )
    critical_values = v1.finalize_bootstrap_records(records)
    onsets = v1.build_onset_results(
        native_gate_records,
        causal_gate_records,
        causal_summaries,
        hard_summaries,
        hard_gate_records,
        lookup,
    )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": utc_now(),
        "protocol_sha256": file_sha256(CONFIG_PATH),
        "script_sha256": file_sha256(SCRIPT_PATH),
        "git_head": git("rev-parse", "HEAD"),
        "v1_inherited_runner_sha256": protocol()["source"][
            "v1_runner_sha256"
        ],
        "v1_inherited_config_sha256": protocol()["source"][
            "v1_config_sha256"
        ],
        "checkpoint_lock_sha256": file_sha256(CHECKPOINT_LOCK_PATH),
        "checkpoint_lock_manifest_sha256": compact_hash(lock["leaves"]),
        "dependency_binding": dependency_binding(),
        "inventory": inventory,
        "native_summaries": native_summaries,
        "causal_summaries": causal_summaries,
        "hard_summaries": hard_summaries,
        "bootstrap": {
            "samples": int(
                v1.protocol()["analysis"]["bootstrap_samples"]
            ),
            "critical_values": critical_values,
            "records": {
                record.record_id: v1.serialized_bootstrap_record(record)
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
        "v1_causal_scientific_artifacts_consumed": False,
    }
    markdown = render_markdown(result)
    write_json_creation_only(OUT_JSON, result)
    write_bytes_creation_only(OUT_MD, (markdown + "\n").encode())
    print(markdown)
    return result


def metamorphic_leaf_axis_test() -> dict[str, Any]:
    state = {
        (seed, trait, update): False
        for seed in SEEDS
        for trait in TRAITS
        for update in TARGET_UPDATES
    }
    targets = (0, 12, 24)
    for target in targets:
        for key in state:
            state[key] = False
        for seed in SEEDS:
            for trait in TRAITS:
                state[(seed, trait, target)] = True
        observed = {
            update: all(
                state[(seed, trait, update)]
                for seed in SEEDS
                for trait in TRAITS
            )
            for update in TARGET_UPDATES
        }
        expected = {
            update: update == target for update in TARGET_UPDATES
        }
        if observed != expected:
            raise AssertionError(
                f"V2 checkpoint-axis overwrite regression: "
                f"{observed} != {expected}"
            )
    return {"targets": list(targets), "pass": True}


def source_surface_test() -> dict[str, Any]:
    tree = ast.parse(SCRIPT_PATH.read_text())
    forbidden_attributes = {
        "causal_root",
        "load_causal_completion",
        "load_causal_cells",
        "require_causal_lock",
        "seal_causal_trajectories",
        "run_causal_trajectory",
        "run_all_causal",
        "artifact_inventory",
        "analyze",
    }
    observed = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "v1"
            and node.attr in forbidden_attributes
        ):
            observed.add(node.attr)
    if observed:
        raise AssertionError(
            f"V2 imports forbidden v1 causal/output API: {sorted(observed)}"
        )
    return {
        "forbidden_v1_api_references": [],
        "pass": True,
    }


def self_test() -> dict[str, Any]:
    inherited = inheritance_assertions()
    if len(expected_leaf_keys()) != 36:
        raise AssertionError("V2 expected leaf count is not 36")
    global_keys = expected_global_keys()
    if len(global_keys) != 1080:
        raise AssertionError("V2 expected causal count is not 1080")
    if len({key_tuple(key) for key in global_keys}) != 1080:
        raise AssertionError("V2 global logical keys are not unique")
    for leaf in expected_leaf_keys():
        keys = expected_checkpoint_keys(
            int(leaf["training_seed"]),
            str(leaf["trait"]),
            int(leaf["target_update"]),
        )
        if len(keys) != 30 or any(
            int(key["optimizer_update"]) != int(leaf["target_update"])
            for key in keys
        ):
            raise AssertionError(f"V2 leaf key inventory drift: {leaf}")
        config = v1.train_config(
            int(leaf["training_seed"]),
            [int(leaf["target_update"])],
        )
        if (
            int(config["max_updates"]) != 24
            or int(config["schedule_total_updates"]) != 24
            or config["probe_updates"] != [int(leaf["target_update"])]
        ):
            raise AssertionError(f"V2 replay config is not full/single: {leaf}")
    axis = metamorphic_leaf_axis_test()
    surface = source_surface_test()

    generator = torch.Generator().manual_seed(2026072607)
    u = torch.randn(17, generator=generator)
    v = torch.randn(13, generator=generator)
    synthetic = 2.0 * torch.outer(u / u.norm(), v / v.norm())
    synthetic += 0.001 * torch.randn(
        synthetic.shape, generator=generator
    )
    factor, audit = v1.deterministic_local_factor(
        synthetic, seed=2026072608
    )
    if factor is None or not audit["identifiable"]:
        raise AssertionError(f"V2 synthetic factor failed: {audit}")
    if float(torch.dot(synthetic.mv(factor["v"]), factor["u"])) <= 0:
        raise AssertionError("V2 synthetic factor orientation failed")
    endpoint_u = torch.nn.functional.normalize(
        torch.randn(17, generator=generator), dim=0
    )
    endpoint_v = torch.nn.functional.normalize(
        torch.randn(13, generator=generator), dim=0
    )
    projection_witness, witnessed_projection = (
        crossfit_projection_witness(
            synthetic,
            {"u": endpoint_u, "s": 1.0, "v": endpoint_v},
        )
    )
    independently_recomputed = float(
        torch.dot(
            endpoint_u.float(),
            projection_witness["delta_times_matched_v"].float(),
        )
    )
    if (
        witnessed_projection != independently_recomputed
        or projection_witness["delta_times_matched_v_sha256"]
        != v1.tensor_sha256(
            projection_witness["delta_times_matched_v"]
        )
    ):
        raise AssertionError("V2 crossfit projection witness failed")
    tampered_delta_times_v = (
        projection_witness["delta_times_matched_v"]
        + 0.125 * endpoint_u.float()
    )
    tampered_projection = float(
        torch.dot(endpoint_u.float(), tampered_delta_times_v)
    )
    if tampered_projection == witnessed_projection:
        raise AssertionError(
            "V2 crossfit projection witness tamper was not detected"
        )
    zero_witness, zero_projection = crossfit_projection_witness(
        torch.zeros_like(synthetic),
        {"u": endpoint_u, "s": 1.0, "v": endpoint_v},
    )
    if (
        zero_projection != 0.0
        or torch.count_nonzero(
            zero_witness["delta_times_matched_v"]
        ).item()
        != 0
    ):
        raise AssertionError("V2 synthetic u0 projection is not exactly zero")
    synthetic_zero_real = {
        name: {
            "u": torch.tensor([1.0, 0.0, 0.0]),
            "s": 0.0,
            "v": torch.tensor([1.0, 0.0]),
        }
        for name in v1.selected_names()
    }
    synthetic_zero_wrong = {
        name: {
            "u": torch.tensor([0.0, 1.0, 0.0]),
            "s": 0.0,
            "v": torch.tensor([0.0, 1.0]),
        }
        for name in v1.selected_names()
    }
    synthetic_u0_constructions = {
        "checkpoint_local": None,
        "crossfit_endpoint_loaded": synthetic_zero_real,
        "wrong_trait": synthetic_zero_wrong,
    }
    synthetic_u0_factor_sets = v1.checkpoint_factor_sets(
        synthetic_u0_constructions,
        seed=SEEDS[0],
        trait=TRAITS[0],
        update=0,
    )
    if any(
        float(component["s"]) != 0.0
        for factors in synthetic_u0_factor_sets.values()
        for component in factors.values()
    ):
        raise AssertionError("V2 synthetic u0 control amplitude is nonzero")
    synthetic_u0_statuses = []
    for key in expected_checkpoint_keys(SEEDS[0], TRAITS[0], 0):
        real = synthetic_u0_constructions[str(key["construction"])]
        synthetic_u0_statuses.append(
            "not_applicable"
            if real is None
            or all(
                float(component["s"]) == 0.0
                for component in real.values()
            )
            else "evaluated"
        )
    if synthetic_u0_statuses != ["not_applicable"] * 30:
        raise AssertionError("V2 synthetic u0 cell policy failed")

    toy = torch.nn.Linear(5, 4, bias=False)
    name = "weight"
    original = toy.weight.detach().clone()
    patch_factor = {
        name: {
            "u": torch.nn.functional.normalize(
                torch.arange(1, 5, dtype=torch.float32), dim=0
            ),
            "s": 0.25,
            "v": torch.nn.functional.normalize(
                torch.arange(1, 6, dtype=torch.float32), dim=0
            ),
        }
    }
    with v1.temporary_patch(toy, patch_factor, 1.0):
        if torch.equal(toy.weight, original):
            raise AssertionError("V2 toy patch did not change weights")
    if not torch.equal(toy.weight, original):
        raise AssertionError("V2 toy patch did not restore weights")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "creation.json"
        write_json_creation_only(first, {"value": 1})
        try:
            write_json_creation_only(first, {"value": 2})
        except RuntimeError:
            pass
        else:
            raise AssertionError("V2 creation-only writer overwrote a file")
        nested = root / "tree"
        (nested / "z").mkdir(parents=True)
        write_bytes_creation_only(nested / "b.txt", b"b")
        write_bytes_creation_only(nested / "z/a.txt", b"a")
        manifest = regular_file_tree_manifest(nested)
        expected_bytes = (
            f"{file_sha256(nested / 'b.txt')}  b.txt\n"
            f"{file_sha256(nested / 'z/a.txt')}  z/a.txt\n"
        ).encode("ascii")
        if (
            manifest["file_count"] != 2
            or manifest["byte_count"] != 2
            or manifest["tree_sha256"]
            != hashlib.sha256(expected_bytes).hexdigest()
        ):
            raise AssertionError("V2 tree-manifest algorithm drift")
        try:
            require_within(root.parent, root, label="synthetic escape")
        except RuntimeError:
            pass
        else:
            raise AssertionError("V2 path confinement accepted escape")

    v1_result = v1.self_test()
    if v1_result.get("pass") is not True:
        raise AssertionError("Inherited v1 model-free self-test failed")
    verifier_hash = protocol()["source"]["v2_verifier_sha256"]
    verifier_path = repository_path(
        protocol()["source"]["v2_verifier_path"],
        label="v2 verifier self-test",
    )
    verifier_hash_matches = bool(
        valid_sha256(verifier_hash)
        and verifier_path.is_file()
        and file_sha256(verifier_path) == verifier_hash
    )
    if valid_sha256(verifier_hash) and not verifier_hash_matches:
        raise AssertionError("Pinned v2 verifier source hash mismatch")
    result = {
        "inherited": inherited,
        "leaf_count": 36,
        "cells_per_leaf": 30,
        "global_cell_count": 1080,
        "metamorphic_leaf_axis": axis,
        "source_surface": surface,
        "local_svd": audit,
        "crossfit_projection_witness": True,
        "u0_exact_zero": True,
        "temporary_patch_restoration": True,
        "creation_only": True,
        "tree_manifest": True,
        "path_confinement": True,
        "v1_self_test": True,
        "preflight_ready": verifier_hash_matches,
        "verifier_hash_matches": verifier_hash_matches,
        "verifier_hash_placeholder": not valid_sha256(verifier_hash),
        "pass": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def status() -> dict[str, Any]:
    result = {
        "preflight": PREFLIGHT_PATH.exists(),
        "checkpoint_lock": CHECKPOINT_LOCK_PATH.exists(),
        "aggregate_json": OUT_JSON.exists(),
        "aggregate_markdown": OUT_MD.exists(),
        "work_exists": WORK.exists(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--self-test", action="store_true")
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--leaf", action="store_true")
    actions.add_argument("--all", action="store_true")
    actions.add_argument("--seal", action="store_true")
    actions.add_argument("--inventory", action="store_true")
    actions.add_argument("--analyze", action="store_true")
    actions.add_argument("--status", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--trait", choices=TRAITS)
    parser.add_argument("--update", type=int, choices=TARGET_UPDATES)
    return parser.parse_args()


def require_leaf_args(args: argparse.Namespace) -> tuple[int, str, int]:
    if args.seed is None or args.trait is None or args.update is None:
        raise SystemExit("--seed, --trait, and --update are required for --leaf")
    return int(args.seed), str(args.trait), int(args.update)


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    elif args.preflight:
        run_preflight()
    elif args.leaf:
        seed, trait, update = require_leaf_args(args)
        with exclusive_lock(f"leaf:{leaf_label(seed, trait, update)}"):
            run_leaf(seed, trait, update)
    elif args.all:
        with exclusive_lock("all"):
            run_all_leaves()
    elif args.seal:
        with exclusive_lock("seal"):
            seal_checkpoints()
    elif args.inventory:
        print(json.dumps(artifact_inventory(), indent=2, sort_keys=True))
    elif args.analyze:
        with exclusive_lock("analyze"):
            analyze()
    elif args.status:
        status()


if __name__ == "__main__":
    main()
