"""One-pass Pythia batch contour with example-indexed optimization.

Every EB2--EB128 cell sees the same 8,192 carrier examples per arm exactly
once.  Maximum updates, warmup, and probes are scaled by effective batch.
The LR schedule retains the original EB16 horizon of 81,920 examples, so the
LR at a given example count is identical across batches.  The two development
carrier blocks and seeds match the quick low-batch calibration.

Cells are reused only when every paired probe is present.  Partial cells
restart from base because ``save_model`` is false and optimizer state is not
persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
PYTHON = sys.executable
BATCHES = (2, 4, 8, 16, 32, 64, 128)
BLOCKS = (1, 2)
SEEDS = {1: 91001, 2: 91002}
EXAMPLES_PER_ARM = 8192
SCHEDULE_EXAMPLES = 81920
WARMUP_EXAMPLES = 128
PROBE_EXAMPLE_COUNTS = (0, 2048, 4096, 8192)
OBJECTIVE = "paired_equal_example_one_pass_batch_contour"
STATUS = "exploratory_development_only"
IDENTITY_NAME = "equal_examples_identity.json"
COMPLETION_NAME = "equal_examples_complete.json"
GEOMETRIES = {
    batch: {
        "config": (
            ROOT
            / f"configs/max_transfer_equal_examples_eb{batch}_one_pass.yaml"
        ),
        "quick_config": (
            ROOT / f"configs/max_transfer_quick_eb{batch}_u1000.yaml"
        ),
        "microbatch_size": batch,
        "gradient_accumulation_steps": 1,
    }
    for batch in BATCHES
}


def parse_selector(
    raw_values: list[str] | None,
    allowed: tuple[int, ...],
    option: str,
) -> tuple[int, ...]:
    """Parse a repeatable comma-delimited integer selector."""
    if not raw_values:
        return allowed
    selected: list[int] = []
    for raw_value in raw_values:
        pieces = raw_value.split(",")
        if any(not piece.strip() for piece in pieces):
            raise ValueError(f"{option} contains an empty comma-delimited value")
        for piece in pieces:
            try:
                value = int(piece.strip())
            except ValueError as error:
                raise ValueError(
                    f"{option} must contain integers, got {piece!r}"
                ) from error
            if value not in allowed:
                choices = ", ".join(str(item) for item in allowed)
                raise ValueError(
                    f"unsupported {option} value {value}; choose from {choices}"
                )
            if value not in selected:
                selected.append(value)
    return tuple(selected)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        with path.open() as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read valid JSON from {path}") from error


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def artifact_record(path: Path, root: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"Required artifact is missing: {path}")
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def jsonl_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def updates_for_examples(effective_batch: int, examples: int) -> int:
    if examples % effective_batch:
        raise ValueError(
            f"{examples} examples is not divisible by EB{effective_batch}"
        )
    return examples // effective_batch


def max_updates(effective_batch: int) -> int:
    return updates_for_examples(effective_batch, EXAMPLES_PER_ARM)


def warmup_updates(effective_batch: int) -> int:
    return updates_for_examples(effective_batch, WARMUP_EXAMPLES)


def schedule_total_updates(effective_batch: int) -> int:
    return updates_for_examples(effective_batch, SCHEDULE_EXAMPLES)


def probe_updates(effective_batch: int) -> tuple[int, ...]:
    return tuple(
        updates_for_examples(effective_batch, examples)
        for examples in PROBE_EXAMPLE_COUNTS
    )


def output_dir(effective_batch: int, block: int) -> Path:
    return (
        RUNS
        / f"max_transfer_equal_examples_eb{effective_batch}_one_pass_b{block}_s1"
    )


def checkpoint_path(
    effective_batch: int,
    block: int,
    condition: str,
    update: int,
) -> Path:
    return (
        output_dir(effective_batch, block)
        / "evaluations"
        / "checkpoints"
        / f"student_{condition}_numbers_update_{update:04d}.json"
    )


def identity_path(effective_batch: int, block: int) -> Path:
    return output_dir(effective_batch, block) / IDENTITY_NAME


def identity_tmp_path(effective_batch: int, block: int) -> Path:
    return Path(f"{identity_path(effective_batch, block)}.tmp")


def completion_path(effective_batch: int, block: int) -> Path:
    return output_dir(effective_batch, block) / COMPLETION_NAME


def completion_tmp_path(effective_batch: int, block: int) -> Path:
    return Path(f"{completion_path(effective_batch, block)}.tmp")


def paired_probe_available(effective_batch: int, block: int, update: int) -> bool:
    return all(
        checkpoint_path(effective_batch, block, condition, update).exists()
        for condition in ("preference", "base")
    )


def completed_probe_updates(effective_batch: int, block: int) -> list[int]:
    return [
        update
        for update in probe_updates(effective_batch)
        if paired_probe_available(effective_batch, block, update)
    ]


def _materialize_carrier(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def source_carrier_manifest(block: int) -> dict[str, dict[str, object]]:
    manifest: dict[str, dict[str, object]] = {}
    for condition in ("preference", "base"):
        name = f"numbers_{condition}_teacher.jsonl"
        source_path = RUNS / f"confirm_v3_b{block}" / "data" / name
        rows = jsonl_rows(source_path)
        if rows != EXAMPLES_PER_ARM:
            raise RuntimeError(
                f"{source_path} has {rows} rows, expected {EXAMPLES_PER_ARM}"
            )
        manifest[condition] = {
            "source": str(source_path.relative_to(ROOT)),
            "rows": rows,
            "sha256": sha256(source_path),
        }
    return manifest


def validate_materialized_data(
    effective_batch: int,
    block: int,
    manifest: dict[str, dict[str, object]],
) -> None:
    destination = output_dir(effective_batch, block) / "data"
    for condition, record in manifest.items():
        path = destination / f"numbers_{condition}_teacher.jsonl"
        if not path.is_file():
            raise RuntimeError(f"Materialized carrier is missing: {path}")
        if jsonl_rows(path) != int(record["rows"]):
            raise RuntimeError(f"Materialized carrier row count mismatch: {path}")
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"Materialized carrier hash mismatch: {path}")


def prepare_data(
    effective_batch: int,
    block: int,
) -> dict[str, dict[str, object]]:
    destination = output_dir(effective_batch, block) / "data"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = source_carrier_manifest(block)
    for condition, record in manifest.items():
        name = f"numbers_{condition}_teacher.jsonl"
        source_path = ROOT / str(record["source"])
        _materialize_carrier(source_path, destination / name)
    validate_materialized_data(effective_batch, block, manifest)
    return manifest


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open() as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Expected a mapping in {path}")
    return loaded


def expected_metadata(effective_batch: int) -> dict[str, object]:
    quick_path = Path(GEOMETRIES[effective_batch]["quick_config"])
    return {
        "objective": OBJECTIVE,
        "status": STATUS,
        "carrier_blocks": list(BLOCKS),
        "heldout_confirmation": False,
        "effective_batch_size": effective_batch,
        "examples_per_arm": EXAMPLES_PER_ARM,
        "passes": 1.0,
        "optimizer_updates": max_updates(effective_batch),
        "schedule_examples": SCHEDULE_EXAMPLES,
        "schedule_total_updates": schedule_total_updates(effective_batch),
        "warmup_examples": WARMUP_EXAMPLES,
        "warmup_updates": warmup_updates(effective_batch),
        "probe_example_counts": list(PROBE_EXAMPLE_COUNTS),
        "probe_updates": list(probe_updates(effective_batch)),
        "reference_quick_config": str(quick_path.relative_to(ROOT)),
    }


def validate_config(effective_batch: int) -> None:
    geometry = GEOMETRIES[effective_batch]
    config_path = Path(geometry["config"])
    quick_path = Path(geometry["quick_config"])
    config = _load_yaml(config_path)
    quick = _load_yaml(quick_path)
    training = config["student_training"]
    quick_training = quick["student_training"]

    shared_sections = (
        "model",
        "number_data",
        "preference_data",
        "teacher_training",
        "evaluation",
    )
    if any(config[section] != quick[section] for section in shared_sections):
        raise RuntimeError(f"EB{effective_batch} shared recipe drifted")
    if config["run"]["device"] != quick["run"]["device"] or config["run"][
        "seed"
    ] != quick["run"]["seed"]:
        raise RuntimeError(f"EB{effective_batch} run settings drifted")

    frozen_training_keys = (
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "max_grad_norm",
        "max_length",
        "optimizer",
        "save_model",
        "seed",
        "weight_decay",
        "lora",
    )
    if any(training[key] != quick_training[key] for key in frozen_training_keys):
        raise RuntimeError(f"EB{effective_batch} optimizer recipe drifted")
    if (
        int(training["batch_size"]) != effective_batch
        or int(training["gradient_accumulation_steps"]) != 1
        or int(training["epochs"]) != 1
        or int(training["max_updates"]) != max_updates(effective_batch)
        or int(training["schedule_total_updates"])
        != schedule_total_updates(effective_batch)
        or int(training["warmup_updates"]) != warmup_updates(effective_batch)
        or list(training["probe_updates"])
        != list(probe_updates(effective_batch))
        or bool(training["save_model"])
    ):
        raise RuntimeError(f"EB{effective_batch} equal-example geometry drifted")
    if config["equal_example_contour"] != expected_metadata(effective_batch):
        raise RuntimeError(f"EB{effective_batch} contour metadata drifted")


def expected_identity(
    effective_batch: int,
    block: int,
    carriers: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    geometry = GEOMETRIES[effective_batch]
    config_path = Path(geometry["config"])
    quick_path = Path(geometry["quick_config"])
    return {
        "schema_version": 1,
        **expected_metadata(effective_batch),
        "microbatch_size": geometry["microbatch_size"],
        "gradient_accumulation_steps": geometry[
            "gradient_accumulation_steps"
        ],
        "block": block,
        "student_seed": SEEDS[block],
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "reference_quick_config_sha256": sha256(quick_path),
        "carriers": carriers or source_carrier_manifest(block),
    }


def validate_identity(
    effective_batch: int,
    block: int,
) -> dict[str, object]:
    path = identity_path(effective_batch, block)
    if identity_tmp_path(effective_batch, block).exists():
        raise RuntimeError(f"Stale atomic identity temporary exists: {path}.tmp")
    if not path.is_file():
        raise RuntimeError(f"Frozen identity is missing: {path}")
    observed = load_json(path)
    expected = expected_identity(effective_batch, block)
    if observed != expected:
        raise RuntimeError(f"Frozen identity mismatch: {path}")
    validate_materialized_data(
        effective_batch,
        block,
        expected["carriers"],
    )
    return expected


def generated_pipeline_paths(effective_batch: int, block: int) -> list[Path]:
    root = output_dir(effective_batch, block)
    paths = [
        root / "evaluations",
        root / "models",
        root / "resolved_config.json",
        completion_path(effective_batch, block),
        completion_tmp_path(effective_batch, block),
    ]
    paths.extend(sorted(root.glob("checkpoint_report*.json")))
    paths.extend(sorted(root.glob("checkpoint_report*.md")))
    return paths


def cleanup_generated_pipeline_outputs(effective_batch: int, block: int) -> None:
    for path in generated_pipeline_paths(effective_batch, block):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def has_generated_pipeline_outputs(effective_batch: int, block: int) -> bool:
    return any(
        path.exists()
        for path in generated_pipeline_paths(effective_batch, block)
    )


def training_metrics_path(
    effective_batch: int,
    block: int,
    condition: str,
) -> Path:
    return (
        output_dir(effective_batch, block)
        / "models"
        / f"student_{condition}_numbers"
        / "training_metrics.json"
    )


def expected_lr_after_update(effective_batch: int, update: int) -> float:
    config = _load_yaml(Path(GEOMETRIES[effective_batch]["config"]))
    learning_rate = float(config["student_training"]["learning_rate"])
    warmup = warmup_updates(effective_batch)
    schedule = schedule_total_updates(effective_batch)
    if warmup and update < warmup:
        scale = (update + 1) / warmup
    else:
        scale = max(schedule - update, 0) / max(schedule - warmup, 1)
    return learning_rate * scale


def validate_training_metrics(
    effective_batch: int,
    block: int,
    condition: str,
) -> dict[str, object]:
    path = training_metrics_path(effective_batch, block, condition)
    metrics = load_json(path)
    expected_updates = max_updates(effective_batch)
    expected_probes = list(probe_updates(effective_batch))
    expected_lr = float(
        _load_yaml(Path(GEOMETRIES[effective_batch]["config"]))[
            "student_training"
        ]["learning_rate"]
    )
    fixed = {
        "examples": EXAMPLES_PER_ARM,
        "epochs": 1,
        "configured_epochs": 1,
        "completed_epochs": 1,
        "optimizer_updates": expected_updates,
        "saved_model": False,
        "schedule_total_updates": schedule_total_updates(effective_batch),
        "warmup_updates": warmup_updates(effective_batch),
        "seed": SEEDS[block],
    }
    for key, expected in fixed.items():
        if metrics.get(key) != expected:
            raise RuntimeError(
                f"{path} has {key}={metrics.get(key)!r}, expected {expected!r}"
            )

    optimizer = metrics.get("optimizer")
    if not isinstance(optimizer, dict) or optimizer.get("name") != "adamw":
        raise RuntimeError(f"{path} does not record AdamW")
    if not math.isclose(
        float(optimizer.get("learning_rate", -1.0)),
        expected_lr,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError(f"{path} learning rate metadata drifted")

    lora = metrics.get("lora")
    expected_lora = _load_yaml(Path(GEOMETRIES[effective_batch]["config"]))[
        "student_training"
    ]["lora"]
    if not isinstance(lora, dict) or (
        int(lora.get("r", -1)) != int(expected_lora["r"])
        or float(lora.get("alpha", -1.0)) != float(expected_lora["alpha"])
        or list(lora.get("target_modules", []))
        != list(expected_lora["target_modules"])
    ):
        raise RuntimeError(f"{path} LoRA metadata drifted")

    update_metrics = metrics.get("update_metrics")
    if not isinstance(update_metrics, list) or len(update_metrics) != expected_updates:
        raise RuntimeError(f"{path} does not contain {expected_updates} updates")
    warmup = warmup_updates(effective_batch)
    schedule = schedule_total_updates(effective_batch)
    for expected_update, record in enumerate(update_metrics, start=1):
        if (
            not isinstance(record, dict)
            or record.get("optimizer_update") != expected_update
            or record.get("epoch") != 0
        ):
            raise RuntimeError(
                f"{path} update sequence diverges at update {expected_update}"
            )
        rates = record.get("learning_rates_after_update")
        if not isinstance(rates, list) or not rates:
            raise RuntimeError(f"{path} has no LR at update {expected_update}")
        if warmup and expected_update < warmup:
            scale = (expected_update + 1) / warmup
        else:
            scale = max(schedule - expected_update, 0) / max(
                schedule - warmup, 1
            )
        target_lr = expected_lr * scale
        if any(
            not math.isclose(
                float(rate), target_lr, rel_tol=1e-12, abs_tol=1e-15
            )
            for rate in rates
        ):
            raise RuntimeError(
                f"{path} LR schedule diverges at update {expected_update}"
            )

    checkpoint_metrics = metrics.get("checkpoint_metrics")
    if not isinstance(checkpoint_metrics, list) or [
        record.get("optimizer_update")
        for record in checkpoint_metrics
        if isinstance(record, dict)
    ] != expected_probes:
        raise RuntimeError(f"{path} checkpoint update sequence drifted")
    return {
        "path": str(path.relative_to(output_dir(effective_batch, block))),
        "condition": condition,
        "examples": EXAMPLES_PER_ARM,
        "optimizer_updates": expected_updates,
        "schedule_total_updates": schedule_total_updates(effective_batch),
        "warmup_updates": warmup_updates(effective_batch),
        "probe_updates": expected_probes,
        "seed": SEEDS[block],
    }


def expected_resolved_config(effective_batch: int, block: int) -> dict[str, object]:
    config_path = Path(GEOMETRIES[effective_batch]["config"])
    config = _load_yaml(config_path)
    config["_config_path"] = str(config_path.resolve())
    config["run"]["output_dir"] = str(output_dir(effective_batch, block).resolve())
    config["student_training"]["seed"] = SEEDS[block]
    return config


def required_artifact_paths(effective_batch: int, block: int) -> list[Path]:
    root = output_dir(effective_batch, block)
    paths = [
        root / "resolved_config.json",
        root / "checkpoint_report.json",
        root / "checkpoint_report.md",
    ]
    for condition in ("preference", "base"):
        paths.append(training_metrics_path(effective_batch, block, condition))
        paths.extend(
            checkpoint_path(effective_batch, block, condition, update)
            for update in probe_updates(effective_batch)
        )
    return sorted(paths, key=lambda path: str(path.relative_to(root)))


def verify_pipeline_outputs(
    effective_batch: int,
    block: int,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    root = output_dir(effective_batch, block)
    resolved_path = root / "resolved_config.json"
    if load_json(resolved_path) != expected_resolved_config(effective_batch, block):
        raise RuntimeError(f"Resolved config mismatch: {resolved_path}")

    training = {
        condition: validate_training_metrics(effective_batch, block, condition)
        for condition in ("preference", "base")
    }
    for condition in ("preference", "base"):
        for update in probe_updates(effective_batch):
            path = checkpoint_path(effective_batch, block, condition, update)
            record = load_json(path)
            if record.get("optimizer_update") != update:
                raise RuntimeError(f"Checkpoint update mismatch: {path}")

    report_path = root / "checkpoint_report.json"
    report = load_json(report_path)
    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, list) or [
        row.get("optimizer_update") for row in checkpoints if isinstance(row, dict)
    ] != list(probe_updates(effective_batch)):
        raise RuntimeError(f"Checkpoint report update sequence drifted: {report_path}")

    artifacts = {
        str(path.relative_to(root)): artifact_record(path, root)
        for path in required_artifact_paths(effective_batch, block)
    }
    return training, artifacts


def build_completion_record(
    effective_batch: int,
    block: int,
) -> dict[str, object]:
    identity = validate_identity(effective_batch, block)
    path = identity_path(effective_batch, block)
    training, artifacts = verify_pipeline_outputs(effective_batch, block)
    return {
        "schema_version": 1,
        "status": "complete",
        "objective": OBJECTIVE,
        "effective_batch_size": effective_batch,
        "block": block,
        "student_seed": SEEDS[block],
        "identity": {
            "path": path.name,
            "sha256": sha256(path),
        },
        "identity_payload_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "training": training,
        "artifacts": artifacts,
    }


def validate_completion_marker(effective_batch: int, block: int) -> bool:
    path = completion_path(effective_batch, block)
    temporary = completion_tmp_path(effective_batch, block)
    if temporary.exists():
        raise RuntimeError(f"Stale atomic completion temporary exists: {temporary}")
    if not path.exists():
        return False
    observed = load_json(path)
    expected = build_completion_record(effective_batch, block)
    if observed != expected:
        raise RuntimeError(f"Completion marker or artifact hash mismatch: {path}")
    return True


def validate_present_cell(effective_batch: int, block: int) -> str:
    validate_config(effective_batch)
    path = identity_path(effective_batch, block)
    if identity_tmp_path(effective_batch, block).exists():
        raise RuntimeError(f"Stale atomic identity temporary exists: {path}.tmp")
    if not path.exists():
        if has_generated_pipeline_outputs(effective_batch, block):
            raise RuntimeError(
                f"Unowned generated outputs exist without frozen identity in "
                f"{output_dir(effective_batch, block)}"
            )
        return "absent"
    validate_identity(effective_batch, block)
    if validate_completion_marker(effective_batch, block):
        return "complete"
    return "incomplete"


def cell_complete(effective_batch: int, block: int) -> bool:
    return validate_present_cell(effective_batch, block) == "complete"


def run_cell(effective_batch: int, block: int) -> None:
    validate_config(effective_batch)
    config_path = Path(GEOMETRIES[effective_batch]["config"])
    carriers = prepare_data(effective_batch, block)
    identity = expected_identity(effective_batch, block, carriers)
    frozen_path = identity_path(effective_batch, block)
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    if frozen_path.exists():
        if load_json(frozen_path) != identity:
            raise RuntimeError(f"Frozen identity mismatch: {frozen_path}")
    else:
        if identity_tmp_path(effective_batch, block).exists():
            raise RuntimeError(
                f"Stale atomic identity temporary exists: "
                f"{identity_tmp_path(effective_batch, block)}"
            )
        if has_generated_pipeline_outputs(effective_batch, block):
            raise RuntimeError(
                f"Unowned generated outputs exist without frozen identity in "
                f"{output_dir(effective_batch, block)}"
            )
        atomic_write_json(frozen_path, identity)
    validate_identity(effective_batch, block)

    marker_valid = False
    marker_error: RuntimeError | None = None
    try:
        marker_valid = validate_completion_marker(effective_batch, block)
    except RuntimeError as error:
        marker_error = error
    if marker_valid:
        print(
            f"[effective batch {effective_batch}, block {block}] "
            "hash-verified complete cell, reusing",
            flush=True,
        )
        return

    completed = completed_probe_updates(effective_batch, block)
    if marker_error is not None:
        print(
            f"[effective batch {effective_batch}, block {block}] invalid "
            f"completion marker ({marker_error}); replaying from base",
            flush=True,
        )
    elif has_generated_pipeline_outputs(effective_batch, block):
        print(
            f"[effective batch {effective_batch}, block {block}] incomplete "
            f"cell with paired probes {completed}; replaying from base",
            flush=True,
        )
    cleanup_generated_pipeline_outputs(effective_batch, block)
    subprocess.run(
        [
            PYTHON,
            "-m",
            "polypythia_sl.pipeline",
            "--config",
            str(config_path.relative_to(ROOT)),
            "--stage",
            "students",
            "--output-dir",
            str(output_dir(effective_batch, block)),
            "--student-seed",
            str(SEEDS[block]),
        ],
        cwd=ROOT,
        check=True,
    )
    completion = build_completion_record(effective_batch, block)
    atomic_write_json(completion_path(effective_batch, block), completion)
    if not validate_completion_marker(effective_batch, block):
        raise RuntimeError(
            f"Completion marker missing after EB{effective_batch}, block {block}"
        )


def _margin(path: Path) -> float:
    with path.open() as handle:
        record = json.load(handle)
    return float(record["final_target_logit_margin"]["mean"])


def paired_record(
    effective_batch: int,
    block: int,
    update: int,
) -> dict[str, object]:
    treatment = _margin(
        checkpoint_path(effective_batch, block, "preference", update)
    )
    control = _margin(checkpoint_path(effective_batch, block, "base", update))
    return {
        "block": block,
        "student_seed": SEEDS[block],
        "treatment_margin": treatment,
        "control_margin": control,
        "paired_effect": treatment - control,
    }


def batch_at_examples(
    effective_batch: int,
    example_count: int,
    states: dict[tuple[int, int], str] | None = None,
) -> dict[str, object]:
    update = updates_for_examples(effective_batch, example_count)
    if states is None:
        states = {
            (effective_batch, block): validate_present_cell(effective_batch, block)
            for block in BLOCKS
        }
    available = [
        block
        for block in BLOCKS
        if states[(effective_batch, block)] == "complete"
    ]
    pairs = [paired_record(effective_batch, block, update) for block in available]
    effects = [float(pair["paired_effect"]) for pair in pairs]
    return {
        "effective_batch_size": effective_batch,
        "optimizer_update": update,
        "completed_dev_pairs": len(pairs),
        "missing_blocks": [block for block in BLOCKS if block not in available],
        "positive_dev_pairs": sum(effect > 0 for effect in effects),
        "mean_dev_paired_effect": None if not effects else sum(effects) / len(effects),
        "dev_pairs": pairs,
    }


def example_contour_row(
    example_count: int,
    states: dict[tuple[int, int], str],
) -> dict[str, object]:
    return {
        "example_count_per_arm": example_count,
        "passes": example_count / EXAMPLES_PER_ARM,
        "batch_results": [
            batch_at_examples(batch, example_count, states) for batch in BATCHES
        ],
    }


def endpoint_record(
    effective_batch: int,
    states: dict[tuple[int, int], str],
) -> dict[str, object]:
    result = batch_at_examples(effective_batch, EXAMPLES_PER_ARM, states)
    complete = all(
        states[(effective_batch, block)] == "complete" for block in BLOCKS
    )
    return {
        "effective_batch_size": effective_batch,
        "optimizer_updates": max_updates(effective_batch),
        "schedule_total_updates": schedule_total_updates(effective_batch),
        "warmup_updates": warmup_updates(effective_batch),
        "complete": complete,
        "screen": {
            "definition": "both development-block paired effects are positive",
            "passed": (
                int(result["positive_dev_pairs"]) == len(BLOCKS)
                if complete
                else None
            ),
            "confirmatory_claim_authorized": False,
        },
    }


def collect_cell_states() -> dict[tuple[int, int], str]:
    states: dict[tuple[int, int], str] = {}
    for effective_batch in BATCHES:
        validate_config(effective_batch)
        for block in BLOCKS:
            states[(effective_batch, block)] = validate_present_cell(
                effective_batch, block
            )
    return states


def summarize() -> dict[str, object]:
    states = collect_cell_states()
    endpoints = [endpoint_record(batch, states) for batch in BATCHES]
    complete = all(bool(endpoint["complete"]) for endpoint in endpoints)
    summary = {
        "schema_version": 1,
        "objective": OBJECTIVE,
        "status": (
            "exploratory_development_complete"
            if complete
            else "exploratory_development_partial"
        ),
        "blocks": list(BLOCKS),
        "student_seeds": SEEDS,
        "candidate_effective_batches": list(BATCHES),
        "examples_per_arm": EXAMPLES_PER_ARM,
        "schedule_examples": SCHEDULE_EXAMPLES,
        "warmup_examples": WARMUP_EXAMPLES,
        "probe_example_counts": list(PROBE_EXAMPLE_COUNTS),
        "cell_states": [
            {
                "effective_batch_size": batch,
                "block": block,
                "state": states[(batch, block)],
            }
            for batch in BATCHES
            for block in BLOCKS
        ],
        "example_count_contour": [
            example_contour_row(example_count, states)
            for example_count in PROBE_EXAMPLE_COUNTS
        ],
        "batch_endpoints": endpoints,
        "resume": {
            "granularity": "complete_batch_block_cell",
            "complete_definition": (
                "valid atomic completion marker bound to frozen identity and "
                "all required artifact hashes"
            ),
            "intra_cell_optimizer_resume": False,
            "reason": "save_model is false and optimizer state is not persisted",
        },
    }
    destination = RUNS / "max_transfer_equal_examples_one_pass_summary.json"
    atomic_write_json(destination, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Summarize currently available paired probes without training.",
    )
    parser.add_argument(
        "--batches",
        action="append",
        metavar="BATCH[,BATCH...]",
        help=(
            "Effective batches to run; choices: 2, 4, 8, 16, 32, 64, 128. "
            "Defaults to all."
        ),
    )
    parser.add_argument(
        "--blocks",
        action="append",
        metavar="BLOCK[,BLOCK...]",
        help="Development blocks to run; choices: 1, 2. Defaults to both.",
    )
    args = parser.parse_args()
    try:
        selected_batches = parse_selector(args.batches, BATCHES, "--batches")
        selected_blocks = parse_selector(args.blocks, BLOCKS, "--blocks")
    except ValueError as error:
        parser.error(str(error))
    if args.summary_only:
        print(json.dumps(summarize(), indent=2, sort_keys=True), flush=True)
        return
    try:
        for effective_batch in selected_batches:
            for block in selected_blocks:
                run_cell(effective_batch, block)
    except BaseException:
        print(json.dumps(summarize(), indent=2, sort_keys=True), flush=True)
        raise
    print(json.dumps(summarize(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
