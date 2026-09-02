"""Paired EB32/EB64 screens to update 1,000 on the long-horizon LR schedule.

Development carrier blocks 1 and 2 use the same model, data, seeds, AdamW
recipe, probes, and 5,120-update linear schedule as the completed EB128 quick
screen.  Each cell stops at update 1,000.  This is exploratory development,
not held-out confirmation.

The runner resumes only at complete batch/block cell boundaries because the
recipe deliberately does not persist model or optimizer state.
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
MAX_UPDATES = 1000
SCHEDULE_TOTAL_UPDATES = 5120
WARMUP_UPDATES = 8
PROBE_UPDATES = (0, 420, 1000)
GEOMETRIES = {
    32: {
        "config": ROOT / "configs/max_transfer_quick_eb32_u1000.yaml",
        "microbatch_size": 32,
        "gradient_accumulation_steps": 1,
        "configured_epochs": 4,
    },
    64: {
        "config": ROOT / "configs/max_transfer_quick_eb64_u1000.yaml",
        "microbatch_size": 64,
        "gradient_accumulation_steps": 1,
        "configured_epochs": 8,
    },
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


def jsonl_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def example_presentations(effective_batch: int, update: int) -> int:
    return effective_batch * update


def passes(effective_batch: int, update: int) -> float:
    return example_presentations(effective_batch, update) / EXPECTED_ROWS


def output_dir(effective_batch: int, block: int) -> Path:
    return RUNS / f"max_transfer_quick_eb{effective_batch}_u1000_b{block}_s1"


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
    config_path = Path(geometry["config"])
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    training = config["student_training"]
    observed = {
        "microbatch_size": int(training["batch_size"]),
        "gradient_accumulation_steps": int(
            training["gradient_accumulation_steps"]
        ),
        "configured_epochs": int(training["epochs"]),
    }
    expected = {key: geometry[key] for key in observed}
    if observed != expected:
        raise RuntimeError(
            f"EB{effective_batch}/u1000 geometry drifted: "
            f"observed={observed}, expected={expected}"
        )
    if (
        observed["microbatch_size"]
        * observed["gradient_accumulation_steps"]
        != effective_batch
        or int(training["max_updates"]) != MAX_UPDATES
        or list(training["probe_updates"]) != list(PROBE_UPDATES)
        or int(training["schedule_total_updates"]) != SCHEDULE_TOTAL_UPDATES
        or int(training["warmup_updates"]) != WARMUP_UPDATES
        or bool(training["save_model"])
    ):
        raise RuntimeError(f"EB{effective_batch}/u1000 training recipe drifted")
    quick_test = config["quick_test"]
    if (
        quick_test["objective"] != "paired_low_batch_long_schedule_screen"
        or quick_test["status"] != "exploratory_development_only"
        or list(quick_test["carrier_blocks"]) != list(BLOCKS)
        or bool(quick_test["heldout_confirmation"])
        or int(quick_test["effective_batch_size"]) != effective_batch
        or int(quick_test["optimizer_updates"]) != MAX_UPDATES
        or int(quick_test["schedule_total_updates"])
        != SCHEDULE_TOTAL_UPDATES
        or int(quick_test["example_presentations_per_arm"])
        != example_presentations(effective_batch, MAX_UPDATES)
        or float(quick_test["passes"])
        != passes(effective_batch, MAX_UPDATES)
    ):
        raise RuntimeError(f"EB{effective_batch}/u1000 metadata drifted")


def run_cell(effective_batch: int, block: int) -> None:
    validate_config(effective_batch)
    geometry = GEOMETRIES[effective_batch]
    config_path = Path(geometry["config"])
    carriers = prepare_data(effective_batch, block)
    identity = {
        "schema_version": 1,
        "objective": "paired_low_batch_long_schedule_screen",
        "status": "exploratory_development_only",
        "effective_batch_size": effective_batch,
        "microbatch_size": geometry["microbatch_size"],
        "gradient_accumulation_steps": geometry[
            "gradient_accumulation_steps"
        ],
        "block": block,
        "student_seed": SEEDS[block],
        "optimizer_updates": MAX_UPDATES,
        "schedule_total_updates": SCHEDULE_TOTAL_UPDATES,
        "warmup_updates": WARMUP_UPDATES,
        "probe_updates": list(PROBE_UPDATES),
        "passes": passes(effective_batch, MAX_UPDATES),
        "example_presentations": example_presentations(
            effective_batch, MAX_UPDATES
        ),
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "carriers": carriers,
    }
    identity_path = output_dir(effective_batch, block) / "quick_test_identity.json"
    if identity_path.exists():
        with identity_path.open() as handle:
            if json.load(handle) != identity:
                raise RuntimeError(f"Frozen identity mismatch: {identity_path}")
    else:
        preexisting = completed_probe_updates(effective_batch, block)
        if preexisting:
            raise RuntimeError(
                f"Unowned probe artifacts {preexisting} exist without a frozen "
                f"identity in {output_dir(effective_batch, block)}"
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
            f"{completed}; restarting from base because optimizer state is not "
            "persisted",
            flush=True,
        )
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
    if not endpoint_done(effective_batch, block):
        raise RuntimeError(
            f"Missing update-{MAX_UPDATES} endpoint for effective batch "
            f"{effective_batch}, block {block}"
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


def probe_row(effective_batch: int, update: int) -> dict[str, object]:
    available = [
        block
        for block in BLOCKS
        if paired_probe_available(effective_batch, block, update)
    ]
    pairs = [paired_record(effective_batch, block, update) for block in available]
    effects = [float(pair["paired_effect"]) for pair in pairs]
    return {
        "optimizer_update": update,
        "passes": passes(effective_batch, update),
        "example_presentations_per_arm": example_presentations(
            effective_batch, update
        ),
        "completed_dev_pairs": len(pairs),
        "missing_blocks": [block for block in BLOCKS if block not in available],
        "positive_dev_pairs": sum(effect > 0 for effect in effects),
        "mean_dev_paired_effect": None if not effects else sum(effects) / len(effects),
        "dev_pairs": pairs,
    }


def batch_result(effective_batch: int) -> dict[str, object]:
    trajectory = [probe_row(effective_batch, update) for update in PROBE_UPDATES]
    endpoint = trajectory[-1]
    endpoint_complete = int(endpoint["completed_dev_pairs"]) == len(BLOCKS)
    return {
        "effective_batch_size": effective_batch,
        "microbatch_size": GEOMETRIES[effective_batch]["microbatch_size"],
        "gradient_accumulation_steps": GEOMETRIES[effective_batch][
            "gradient_accumulation_steps"
        ],
        "trajectory": trajectory,
        "endpoint_screen": {
            "definition": "both development-block paired effects are positive",
            "passed": (
                int(endpoint["positive_dev_pairs"]) == len(BLOCKS)
                if endpoint_complete
                else None
            ),
            "confirmatory_claim_authorized": False,
        },
    }


def summarize() -> dict[str, object]:
    results = [batch_result(batch) for batch in GEOMETRIES]
    complete = all(
        int(result["trajectory"][-1]["completed_dev_pairs"]) == len(BLOCKS)
        for result in results
    )
    summary = {
        "schema_version": 1,
        "objective": "paired_low_batch_long_schedule_screen",
        "status": (
            "exploratory_development_complete"
            if complete
            else "exploratory_development_partial"
        ),
        "blocks": list(BLOCKS),
        "student_seeds": SEEDS,
        "candidate_effective_batches": list(GEOMETRIES),
        "max_updates": MAX_UPDATES,
        "schedule_total_updates": SCHEDULE_TOTAL_UPDATES,
        "warmup_updates": WARMUP_UPDATES,
        "probe_updates": list(PROBE_UPDATES),
        "batch_results": results,
        "resume": {
            "granularity": "complete_batch_block_cell",
            "endpoint_update": MAX_UPDATES,
            "intra_cell_optimizer_resume": False,
        },
    }
    destination = RUNS / "max_transfer_quick_low_batch_u1000_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
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
        help="Effective batches to run; choices: 32, 64. Defaults to both.",
    )
    parser.add_argument(
        "--blocks",
        action="append",
        metavar="BLOCK[,BLOCK...]",
        help="Development blocks to run; choices: 1, 2. Defaults to both.",
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
