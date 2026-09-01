"""Long-horizon maximum-transfer dose sweep on Pythia development blocks.

Effective batches 128, 256, and 512 are each trained from the base checkpoint
for 5,120 AdamW updates under one shared 5,120-update linear schedule and an
eight-update warmup.  The runs intentionally differ in example exposure.  The
fixed probes at 0, 420, 1,024, 2,560, and 5,120 updates form comparable
optimizer-dose curves; the prior schedule-420 runs are not reused.

The runner is resumable at cell granularity: completed endpoints are reused,
partial probe artifacts are reported, and only cells missing update 5,120 are
restarted from the base model.  Intra-cell optimizer resume is intentionally
unavailable because the frozen recipe sets ``save_model: false``.  Blocks 1
and 2 are development-only; no held-out confirmation or confirmatory claim is
authorized by the selection produced here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
PYTHON = sys.executable
BLOCKS = (1, 2)
SEEDS = {1: 91001, 2: 91002}
EXPECTED_ROWS = 8192
MAX_UPDATES = 5120
WARMUP_UPDATES = 8
PROBE_UPDATES = (0, 420, 1024, 2560, 5120)
GEOMETRIES = {
    128: {
        "config": ROOT / "configs/max_transfer_dose_eb128_u5120.yaml",
        "microbatch_size": 128,
        "gradient_accumulation_steps": 1,
        "epochs": 80,
    },
    256: {
        "config": ROOT / "configs/max_transfer_dose_eb256_u5120.yaml",
        "microbatch_size": 128,
        "gradient_accumulation_steps": 2,
        "epochs": 160,
    },
    512: {
        "config": ROOT / "configs/max_transfer_dose_eb512_u5120.yaml",
        "microbatch_size": 128,
        "gradient_accumulation_steps": 4,
        "epochs": 320,
    },
}


def parse_selector(
    raw_values: list[str] | None,
    allowed_values: tuple[int, ...],
    label: str,
) -> tuple[int, ...]:
    """Parse repeatable comma-delimited integer selectors without duplicates."""
    if not raw_values:
        return allowed_values
    selected: list[int] = []
    for raw_value in raw_values:
        pieces = raw_value.split(",")
        if any(not piece.strip() for piece in pieces):
            raise ValueError(f"{label} contains an empty comma-delimited value")
        for piece in pieces:
            try:
                value = int(piece.strip())
            except ValueError as error:
                raise ValueError(f"{label} must contain integers, got {piece!r}") from error
            if value not in allowed_values:
                choices = ", ".join(str(choice) for choice in allowed_values)
                raise ValueError(
                    f"unsupported {label} value {value}; choose from {choices}"
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


def jsonl_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def example_presentations(effective_batch: int, update: int) -> int:
    return effective_batch * update


def passes(effective_batch: int, update: int) -> float:
    return example_presentations(effective_batch, update) / EXPECTED_ROWS


def output_dir(effective_batch: int, block: int) -> Path:
    return RUNS / f"max_transfer_dose_eb{effective_batch}_u5120_b{block}_s1"


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


def paired_probe_available(effective_batch: int, block: int, update: int) -> bool:
    return all(
        checkpoint_path(effective_batch, block, condition, update).exists()
        for condition in ("preference", "base")
    )


def completed_probe_updates(effective_batch: int, block: int) -> list[int]:
    return [
        update
        for update in PROBE_UPDATES
        if paired_probe_available(effective_batch, block, update)
    ]


def endpoint_done(effective_batch: int, block: int) -> bool:
    return paired_probe_available(effective_batch, block, MAX_UPDATES)


def _materialize_carrier(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def prepare_data(
    effective_batch: int,
    block: int,
) -> dict[str, dict[str, object]]:
    destination = output_dir(effective_batch, block) / "data"
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, object]] = {}
    for condition in ("preference", "base"):
        name = f"numbers_{condition}_teacher.jsonl"
        source_path = RUNS / f"confirm_v3_b{block}" / "data" / name
        destination_path = destination / name
        rows = jsonl_rows(source_path)
        if rows != EXPECTED_ROWS:
            raise RuntimeError(f"{source_path} has {rows} rows, expected {EXPECTED_ROWS}")
        source_hash = sha256(source_path)
        _materialize_carrier(source_path, destination_path)
        if sha256(destination_path) != source_hash:
            raise RuntimeError(f"Materialized carrier hash mismatch: {destination_path}")
        manifest[condition] = {
            "source": str(source_path.relative_to(ROOT)),
            "rows": rows,
            "sha256": source_hash,
        }
    return manifest


def validate_config(effective_batch: int) -> None:
    geometry = GEOMETRIES[effective_batch]
    with Path(geometry["config"]).open() as handle:
        config = yaml.safe_load(handle)
    training = config["student_training"]
    observed = {
        "microbatch_size": int(training["batch_size"]),
        "gradient_accumulation_steps": int(
            training["gradient_accumulation_steps"]
        ),
        "epochs": int(training["epochs"]),
    }
    expected = {key: geometry[key] for key in observed}
    if observed != expected:
        raise RuntimeError(
            f"Effective-batch-{effective_batch} geometry drifted: "
            f"observed={observed}, expected={expected}"
        )
    if (
        observed["microbatch_size"] * observed["gradient_accumulation_steps"]
        != effective_batch
    ):
        raise RuntimeError(f"Effective-batch-{effective_batch} is inconsistent")
    if int(training["max_updates"]) != MAX_UPDATES:
        raise RuntimeError(f"Effective-batch-{effective_batch} endpoint drifted")
    if list(training["probe_updates"]) != list(PROBE_UPDATES):
        raise RuntimeError(f"Effective-batch-{effective_batch} probes drifted")
    if int(training["schedule_total_updates"]) != MAX_UPDATES:
        raise RuntimeError(f"Effective-batch-{effective_batch} schedule drifted")
    if int(training["warmup_updates"]) != WARMUP_UPDATES:
        raise RuntimeError(f"Effective-batch-{effective_batch} warmup drifted")
    if bool(training["save_model"]):
        raise RuntimeError(f"Effective-batch-{effective_batch} must not save adapters")
    extension = config["dose_extension"]
    if (
        extension["objective"] != "maximum_transfer_long_horizon"
        or extension["status"] != "exploratory_development_only"
        or list(extension["carrier_blocks"]) != list(BLOCKS)
        or bool(extension["heldout_confirmation"])
        or int(extension["effective_batch_size"]) != effective_batch
        or int(extension["optimizer_updates"]) != MAX_UPDATES
        or int(extension["example_presentations_per_arm"])
        != example_presentations(effective_batch, MAX_UPDATES)
        or float(extension["passes"]) != passes(effective_batch, MAX_UPDATES)
    ):
        raise RuntimeError(f"Effective-batch-{effective_batch} metadata drifted")


def run_cell(effective_batch: int, block: int) -> None:
    validate_config(effective_batch)
    geometry = GEOMETRIES[effective_batch]
    carriers = prepare_data(effective_batch, block)
    identity = {
        "schema_version": 1,
        "objective": "maximum_transfer_long_horizon",
        "status": "exploratory_development_only",
        "effective_batch_size": effective_batch,
        "block": block,
        "student_seed": SEEDS[block],
        "microbatch_size": geometry["microbatch_size"],
        "gradient_accumulation_steps": geometry[
            "gradient_accumulation_steps"
        ],
        "optimizer_updates": MAX_UPDATES,
        "warmup_updates": WARMUP_UPDATES,
        "schedule_total_updates": MAX_UPDATES,
        "probe_updates": list(PROBE_UPDATES),
        "epochs": geometry["epochs"],
        "passes": passes(effective_batch, MAX_UPDATES),
        "example_presentations": example_presentations(
            effective_batch, MAX_UPDATES
        ),
        "config": str(Path(geometry["config"]).relative_to(ROOT)),
        "config_sha256": sha256(Path(geometry["config"])),
        "carriers": carriers,
    }
    identity_path = output_dir(effective_batch, block) / "dose_identity.json"
    if identity_path.exists():
        with identity_path.open() as handle:
            if json.load(handle) != identity:
                raise RuntimeError(f"Frozen identity mismatch: {identity_path}")
    else:
        preexisting_probes = completed_probe_updates(effective_batch, block)
        if preexisting_probes:
            raise RuntimeError(
                f"Unowned probe artifacts {preexisting_probes} exist without a "
                f"frozen identity in {output_dir(effective_batch, block)}"
            )
        with identity_path.open("w") as handle:
            json.dump(identity, handle, indent=2, sort_keys=True)

    completed = completed_probe_updates(effective_batch, block)
    if MAX_UPDATES in completed:
        print(
            f"[effective batch {effective_batch}, block {block}] "
            "endpoint complete, reusing",
            flush=True,
        )
        return
    if completed:
        print(
            f"[effective batch {effective_batch}, block {block}] partial probes "
            f"{completed}; restarting this cell from base because optimizer state "
            "is not persisted",
            flush=True,
        )
    subprocess.run(
        [
            PYTHON,
            "-m",
            "polypythia_sl.pipeline",
            "--config",
            str(Path(geometry["config"]).relative_to(ROOT)),
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
    if not endpoint_done(effective_batch, block):
        raise RuntimeError(
            f"Missing endpoint artifacts for effective batch {effective_batch}, "
            f"block {block}"
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


def dose_row(effective_batch: int, update: int) -> dict[str, object]:
    available_blocks = [
        block
        for block in BLOCKS
        if paired_probe_available(effective_batch, block, update)
    ]
    pairs = [
        paired_record(effective_batch, block, update)
        for block in available_blocks
    ]
    effects = [float(pair["paired_effect"]) for pair in pairs]
    return {
        "optimizer_update": update,
        "passes": passes(effective_batch, update),
        "example_presentations_per_arm": example_presentations(
            effective_batch, update
        ),
        "completed_dev_pairs": len(pairs),
        "missing_blocks": [block for block in BLOCKS if block not in available_blocks],
        "positive_dev_pairs": sum(effect > 0 for effect in effects),
        "mean_dev_paired_effect": (
            None if not effects else sum(effects) / len(effects)
        ),
        "dev_pairs": pairs,
    }


def batch_result(effective_batch: int) -> dict[str, object]:
    geometry = GEOMETRIES[effective_batch]
    return {
        "effective_batch_size": effective_batch,
        "microbatch_size": geometry["microbatch_size"],
        "gradient_accumulation_steps": geometry[
            "gradient_accumulation_steps"
        ],
        "epochs": geometry["epochs"],
        "dose_curve": [dose_row(effective_batch, update) for update in PROBE_UPDATES],
    }


def summarize() -> dict[str, object]:
    batch_results = [batch_result(batch) for batch in GEOMETRIES]
    endpoints = [result["dose_curve"][-1] for result in batch_results]
    eligible = [
        (result, endpoint)
        for result, endpoint in zip(batch_results, endpoints)
        if int(endpoint["completed_dev_pairs"]) == len(BLOCKS)
        and int(endpoint["positive_dev_pairs"]) == len(BLOCKS)
    ]
    selected = (
        max(
            eligible,
            key=lambda item: (
                float(item[1]["mean_dev_paired_effect"]),
                -int(item[0]["effective_batch_size"]),
            ),
        )
        if eligible
        else None
    )
    endpoints_complete = all(
        int(endpoint["completed_dev_pairs"]) == len(BLOCKS)
        for endpoint in endpoints
    )
    summary = {
        "schema_version": 1,
        "objective": "maximum_transfer_long_horizon",
        "status": (
            "exploratory_development_complete"
            if endpoints_complete
            else "exploratory_development_partial"
        ),
        "blocks": list(BLOCKS),
        "rows_per_condition": EXPECTED_ROWS,
        "max_updates": MAX_UPDATES,
        "schedule_total_updates": MAX_UPDATES,
        "warmup_updates": WARMUP_UPDATES,
        "probe_updates": list(PROBE_UPDATES),
        "batch_results": batch_results,
        "development_selection": {
            "scope": "development_only_blocks_1_2",
            "candidate_effective_batches": list(GEOMETRIES),
            "criterion": (
                "Among candidates positive in both development blocks at update "
                "5,120, select the largest mean paired effect; break exact ties "
                "toward the smaller batch."
            ),
            "selected_effective_batch_size": (
                None if selected is None else selected[0]["effective_batch_size"]
            ),
            "heldout_confirmation": "not_run",
            "confirmatory_claim_authorized": False,
        },
        "resume": {
            "granularity": "complete_batch_block_cell",
            "endpoint_update": MAX_UPDATES,
            "intra_cell_optimizer_resume": False,
            "reason": "save_model is false and optimizer state is not persisted",
        },
    }
    destination = RUNS / "max_transfer_dose_u5120_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Summarize all currently available paired probe artifacts without training.",
    )
    parser.add_argument(
        "--batches",
        action="append",
        metavar="BATCH[,BATCH...]",
        help=(
            "Effective batches to run. Repeat the flag or use commas; "
            "choices: 128, 256, 512. Defaults to all."
        ),
    )
    parser.add_argument(
        "--blocks",
        action="append",
        metavar="BLOCK[,BLOCK...]",
        help=(
            "Development blocks to run. Repeat the flag or use commas; "
            "choices: 1, 2. Defaults to both."
        ),
    )
    args = parser.parse_args()
    try:
        selected_batches = parse_selector(
            args.batches, tuple(GEOMETRIES), "--batches"
        )
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
