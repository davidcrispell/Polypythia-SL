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
import shutil
import subprocess
import sys
from pathlib import Path

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


def cell_complete(effective_batch: int, block: int) -> bool:
    return completed_probe_updates(effective_batch, block) == list(
        probe_updates(effective_batch)
    )


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
        if rows != EXAMPLES_PER_ARM:
            raise RuntimeError(
                f"{source_path} has {rows} rows, expected {EXAMPLES_PER_ARM}"
            )
        source_hash = sha256(source_path)
        _materialize_carrier(source_path, destination_path)
        if sha256(destination_path) != source_hash:
            raise RuntimeError(
                f"Materialized carrier hash mismatch: {destination_path}"
            )
        manifest[condition] = {
            "source": str(source_path.relative_to(ROOT)),
            "rows": rows,
            "sha256": source_hash,
        }
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


def run_cell(effective_batch: int, block: int) -> None:
    validate_config(effective_batch)
    geometry = GEOMETRIES[effective_batch]
    config_path = Path(geometry["config"])
    quick_path = Path(geometry["quick_config"])
    carriers = prepare_data(effective_batch, block)
    identity = {
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
        "carriers": carriers,
    }
    identity_path = output_dir(effective_batch, block) / "equal_examples_identity.json"
    identity_path.parent.mkdir(parents=True, exist_ok=True)
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
    if cell_complete(effective_batch, block):
        print(
            f"[effective batch {effective_batch}, block {block}] "
            "complete cell, reusing",
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
    if not cell_complete(effective_batch, block):
        missing = sorted(
            set(probe_updates(effective_batch))
            - set(completed_probe_updates(effective_batch, block))
        )
        raise RuntimeError(
            f"Incomplete EB{effective_batch}, block {block} cell; "
            f"missing paired probes {missing}"
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
) -> dict[str, object]:
    update = updates_for_examples(effective_batch, example_count)
    available = [
        block
        for block in BLOCKS
        if paired_probe_available(effective_batch, block, update)
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


def example_contour_row(example_count: int) -> dict[str, object]:
    return {
        "example_count_per_arm": example_count,
        "passes": example_count / EXAMPLES_PER_ARM,
        "batch_results": [
            batch_at_examples(batch, example_count) for batch in BATCHES
        ],
    }


def endpoint_record(effective_batch: int) -> dict[str, object]:
    result = batch_at_examples(effective_batch, EXAMPLES_PER_ARM)
    complete = int(result["completed_dev_pairs"]) == len(BLOCKS)
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


def summarize() -> dict[str, object]:
    endpoints = [endpoint_record(batch) for batch in BATCHES]
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
        "example_count_contour": [
            example_contour_row(example_count)
            for example_count in PROBE_EXAMPLE_COUNTS
        ],
        "batch_endpoints": endpoints,
        "resume": {
            "granularity": "complete_batch_block_cell",
            "complete_definition": "all paired example-count probes exist",
            "intra_cell_optimizer_resume": False,
            "reason": "save_model is false and optimizer state is not persisted",
        },
    }
    destination = RUNS / "max_transfer_equal_examples_one_pass_summary.json"
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
