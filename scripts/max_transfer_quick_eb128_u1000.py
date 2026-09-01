"""Minimal paired EB128/update-1000 screen on the long-horizon LR schedule.

Development blocks 1 and 2 are trained from scratch with matched seeds under
the planned 5,120-update linear schedule, but each cell stops at update 1,000.
Probes at 0, 420, and 1,000 therefore lie on the same LR trajectory as the
long-horizon experiment.  This is an exploratory screen, not held-out
confirmation.

The runner is endpoint-aware at block-cell granularity.  It reuses paired
update-1,000 endpoints, reports partial probe availability, and restarts an
incomplete block from base because the frozen recipe does not persist model or
optimizer state.
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
CONFIG = ROOT / "configs/max_transfer_quick_eb128_u1000.yaml"
BLOCKS = (1, 2)
SEEDS = {1: 91001, 2: 91002}
EXPECTED_ROWS = 8192
EFFECTIVE_BATCH = 128
MICROBATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = 1
MAX_UPDATES = 1000
SCHEDULE_TOTAL_UPDATES = 5120
WARMUP_UPDATES = 8
PROBE_UPDATES = (0, 420, 1000)


def parse_blocks(raw_values: list[str] | None) -> tuple[int, ...]:
    """Parse repeatable comma-delimited development-block selectors."""
    if not raw_values:
        return BLOCKS
    selected: list[int] = []
    for raw_value in raw_values:
        pieces = raw_value.split(",")
        if any(not piece.strip() for piece in pieces):
            raise ValueError("--blocks contains an empty comma-delimited value")
        for piece in pieces:
            try:
                block = int(piece.strip())
            except ValueError as error:
                raise ValueError(
                    f"--blocks must contain integers, got {piece!r}"
                ) from error
            if block not in BLOCKS:
                raise ValueError(
                    f"unsupported --blocks value {block}; choose from 1, 2"
                )
            if block not in selected:
                selected.append(block)
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


def example_presentations(update: int) -> int:
    return EFFECTIVE_BATCH * update


def passes(update: int) -> float:
    return example_presentations(update) / EXPECTED_ROWS


def output_dir(block: int) -> Path:
    return RUNS / f"max_transfer_quick_eb128_u1000_b{block}_s1"


def checkpoint_path(block: int, condition: str, update: int) -> Path:
    return (
        output_dir(block)
        / "evaluations"
        / "checkpoints"
        / f"student_{condition}_numbers_update_{update:04d}.json"
    )


def paired_probe_available(block: int, update: int) -> bool:
    return all(
        checkpoint_path(block, condition, update).exists()
        for condition in ("preference", "base")
    )


def completed_probe_updates(block: int) -> list[int]:
    return [update for update in PROBE_UPDATES if paired_probe_available(block, update)]


def endpoint_done(block: int) -> bool:
    return paired_probe_available(block, MAX_UPDATES)


def _materialize_carrier(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def prepare_data(block: int) -> dict[str, dict[str, object]]:
    destination = output_dir(block) / "data"
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


def validate_config() -> None:
    with CONFIG.open() as handle:
        config = yaml.safe_load(handle)
    training = config["student_training"]
    if (
        int(training["batch_size"]) != MICROBATCH_SIZE
        or int(training["gradient_accumulation_steps"])
        != GRADIENT_ACCUMULATION_STEPS
        or int(training["batch_size"])
        * int(training["gradient_accumulation_steps"])
        != EFFECTIVE_BATCH
        or int(training["epochs"]) != 16
        or int(training["max_updates"]) != MAX_UPDATES
        or list(training["probe_updates"]) != list(PROBE_UPDATES)
        or int(training["schedule_total_updates"]) != SCHEDULE_TOTAL_UPDATES
        or int(training["warmup_updates"]) != WARMUP_UPDATES
        or bool(training["save_model"])
    ):
        raise RuntimeError("EB128/update-1000 quick-test training geometry drifted")
    quick_test = config["quick_test"]
    if (
        quick_test["objective"] != "paired_eb128_long_schedule_screen"
        or quick_test["status"] != "exploratory_development_only"
        or list(quick_test["carrier_blocks"]) != list(BLOCKS)
        or bool(quick_test["heldout_confirmation"])
        or int(quick_test["effective_batch_size"]) != EFFECTIVE_BATCH
        or int(quick_test["optimizer_updates"]) != MAX_UPDATES
        or int(quick_test["schedule_total_updates"]) != SCHEDULE_TOTAL_UPDATES
        or int(quick_test["example_presentations_per_arm"])
        != example_presentations(MAX_UPDATES)
        or float(quick_test["passes"]) != passes(MAX_UPDATES)
    ):
        raise RuntimeError("EB128/update-1000 quick-test metadata drifted")


def run_block(block: int) -> None:
    validate_config()
    carriers = prepare_data(block)
    identity = {
        "schema_version": 1,
        "objective": "paired_eb128_long_schedule_screen",
        "status": "exploratory_development_only",
        "effective_batch_size": EFFECTIVE_BATCH,
        "microbatch_size": MICROBATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "block": block,
        "student_seed": SEEDS[block],
        "optimizer_updates": MAX_UPDATES,
        "schedule_total_updates": SCHEDULE_TOTAL_UPDATES,
        "warmup_updates": WARMUP_UPDATES,
        "probe_updates": list(PROBE_UPDATES),
        "passes": passes(MAX_UPDATES),
        "example_presentations": example_presentations(MAX_UPDATES),
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256(CONFIG),
        "carriers": carriers,
    }
    identity_path = output_dir(block) / "quick_test_identity.json"
    if identity_path.exists():
        with identity_path.open() as handle:
            if json.load(handle) != identity:
                raise RuntimeError(f"Frozen identity mismatch: {identity_path}")
    else:
        preexisting_probes = completed_probe_updates(block)
        if preexisting_probes:
            raise RuntimeError(
                f"Unowned probe artifacts {preexisting_probes} exist without a "
                f"frozen identity in {output_dir(block)}"
            )
        with identity_path.open("w") as handle:
            json.dump(identity, handle, indent=2, sort_keys=True)

    completed = completed_probe_updates(block)
    if MAX_UPDATES in completed:
        print(f"[block {block}] endpoint complete, reusing", flush=True)
        return
    if completed:
        print(
            f"[block {block}] partial probes {completed}; restarting this block "
            "from base because optimizer state is not persisted",
            flush=True,
        )
    subprocess.run(
        [
            PYTHON,
            "-m",
            "polypythia_sl.pipeline",
            "--config",
            str(CONFIG.relative_to(ROOT)),
            "--stage",
            "students",
            "--output-dir",
            str(output_dir(block)),
            "--student-seed",
            str(SEEDS[block]),
        ],
        cwd=ROOT,
        check=True,
    )
    if not endpoint_done(block):
        raise RuntimeError(f"Missing update-{MAX_UPDATES} endpoint for block {block}")


def _margin(path: Path) -> float:
    with path.open() as handle:
        record = json.load(handle)
    return float(record["final_target_logit_margin"]["mean"])


def paired_record(block: int, update: int) -> dict[str, object]:
    treatment = _margin(checkpoint_path(block, "preference", update))
    control = _margin(checkpoint_path(block, "base", update))
    return {
        "block": block,
        "student_seed": SEEDS[block],
        "treatment_margin": treatment,
        "control_margin": control,
        "paired_effect": treatment - control,
    }


def probe_row(update: int) -> dict[str, object]:
    available_blocks = [block for block in BLOCKS if paired_probe_available(block, update)]
    pairs = [paired_record(block, update) for block in available_blocks]
    effects = [float(pair["paired_effect"]) for pair in pairs]
    return {
        "optimizer_update": update,
        "passes": passes(update),
        "example_presentations_per_arm": example_presentations(update),
        "completed_dev_pairs": len(pairs),
        "missing_blocks": [block for block in BLOCKS if block not in available_blocks],
        "positive_dev_pairs": sum(effect > 0 for effect in effects),
        "mean_dev_paired_effect": (
            None if not effects else sum(effects) / len(effects)
        ),
        "dev_pairs": pairs,
    }


def summarize() -> dict[str, object]:
    trajectory = [probe_row(update) for update in PROBE_UPDATES]
    endpoint = trajectory[-1]
    endpoint_complete = int(endpoint["completed_dev_pairs"]) == len(BLOCKS)
    endpoint_positive = (
        endpoint_complete and int(endpoint["positive_dev_pairs"]) == len(BLOCKS)
    )
    summary = {
        "schema_version": 1,
        "objective": "paired_eb128_long_schedule_screen",
        "status": (
            "exploratory_development_complete"
            if endpoint_complete
            else "exploratory_development_partial"
        ),
        "blocks": list(BLOCKS),
        "student_seeds": SEEDS,
        "effective_batch_size": EFFECTIVE_BATCH,
        "max_updates": MAX_UPDATES,
        "schedule_total_updates": SCHEDULE_TOTAL_UPDATES,
        "warmup_updates": WARMUP_UPDATES,
        "probe_updates": list(PROBE_UPDATES),
        "trajectory": trajectory,
        "endpoint_screen": {
            "definition": "both development-block paired effects are positive",
            "passed": endpoint_positive if endpoint_complete else None,
            "confirmatory_claim_authorized": False,
        },
        "resume": {
            "granularity": "complete_block_cell",
            "endpoint_update": MAX_UPDATES,
            "intra_cell_optimizer_resume": False,
        },
    }
    destination = RUNS / "max_transfer_quick_eb128_u1000_summary.json"
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
        selected_blocks = parse_blocks(args.blocks)
    except ValueError as error:
        parser.error(str(error))
    if args.summary_only:
        print(json.dumps(summarize(), indent=2, sort_keys=True), flush=True)
        return
    try:
        for block in selected_blocks:
            run_block(block)
    except BaseException:
        print(json.dumps(summarize(), indent=2, sort_keys=True), flush=True)
        raise
    print(json.dumps(summarize(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
