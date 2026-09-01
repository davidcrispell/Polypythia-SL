"""Development-only fixed-update batch sweep for maximum Pythia transfer.

Every candidate receives exactly 420 AdamW optimizer updates under the same
420-update linear schedule and eight-update warmup.  Effective batches 16, 32,
64, 128, and 256 therefore receive progressively more example exposure; this is
intentional because the objective is maximum transfer, not equal-exposure
efficiency.  Only frozen development carrier blocks 1 and 2 are used.

The existing effective-batch-512/update-420 result joins the ranked candidate
set when its summary is available, restricted to development blocks 1 and 2.
The archived effective-batch-16/update-5120 result is reported only as
historical context because both optimizer updates and exposure differ.  A
development selection from this script does not authorize a confirmatory
subliminal-learning claim or use held-out carrier blocks.
"""

from __future__ import annotations

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
ARCHIVED_EB16_SEED_PAIRS = (1, 2)
SEEDS = {1: 91001, 2: 91002}
EXPECTED_ROWS = 8192
MAX_UPDATES = 420
WARMUP_UPDATES = 8
GEOMETRIES = {
    16: {
        "config": ROOT / "configs/max_transfer_fixed_update_eb16_u420.yaml",
        "microbatch_size": 8,
        "gradient_accumulation_steps": 2,
        "configured_epochs": 1,
    },
    32: {
        "config": ROOT / "configs/max_transfer_fixed_update_eb32_u420.yaml",
        "microbatch_size": 32,
        "gradient_accumulation_steps": 1,
        "configured_epochs": 2,
    },
    64: {
        "config": ROOT / "configs/max_transfer_fixed_update_eb64_u420.yaml",
        "microbatch_size": 64,
        "gradient_accumulation_steps": 1,
        "configured_epochs": 4,
    },
    128: {
        "config": ROOT / "configs/max_transfer_fixed_update_eb128_u420.yaml",
        "microbatch_size": 128,
        "gradient_accumulation_steps": 1,
        "configured_epochs": 7,
    },
    256: {
        "config": ROOT / "configs/max_transfer_fixed_update_eb256_u420.yaml",
        "microbatch_size": 128,
        "gradient_accumulation_steps": 2,
        "configured_epochs": 14,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def example_presentations(effective_batch: int) -> int:
    return effective_batch * MAX_UPDATES


def passes(effective_batch: int) -> float:
    return example_presentations(effective_batch) / EXPECTED_ROWS


def output_dir(effective_batch: int, block: int) -> Path:
    return RUNS / f"max_transfer_fixed_update_eb{effective_batch}_u420_b{block}_s1"


def checkpoint_path(
    effective_batch: int,
    block: int,
    condition: str,
) -> Path:
    return (
        output_dir(effective_batch, block)
        / "evaluations"
        / "checkpoints"
        / f"student_{condition}_numbers_update_{MAX_UPDATES:04d}.json"
    )


def endpoint_done(effective_batch: int, block: int) -> bool:
    return all(
        checkpoint_path(effective_batch, block, condition).exists()
        for condition in ("preference", "base")
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
        "configured_epochs": int(training["epochs"]),
    }
    expected = {key: geometry[key] for key in observed}
    if observed != expected:
        raise RuntimeError(
            f"Effective-batch-{effective_batch} config geometry drifted: "
            f"observed={observed}, expected={expected}"
        )
    if (
        observed["microbatch_size"] * observed["gradient_accumulation_steps"]
        != effective_batch
    ):
        raise RuntimeError(f"Effective-batch-{effective_batch} config is inconsistent")
    if int(training["max_updates"]) != MAX_UPDATES:
        raise RuntimeError(f"Effective-batch-{effective_batch} update count drifted")
    if training["probe_updates"] != [0, MAX_UPDATES]:
        raise RuntimeError(f"Effective-batch-{effective_batch} must be endpoint-only")
    if int(training["schedule_total_updates"]) != MAX_UPDATES:
        raise RuntimeError(f"Effective-batch-{effective_batch} schedule drifted")
    if int(training["warmup_updates"]) != WARMUP_UPDATES:
        raise RuntimeError(f"Effective-batch-{effective_batch} warmup drifted")
    if bool(training["save_model"]):
        raise RuntimeError(f"Effective-batch-{effective_batch} must not save adapters")
    sweep = config["sweep"]
    if (
        sweep["objective"] != "fixed_update_maximum_transfer"
        or sweep["status"] != "exploratory_development_only"
        or list(sweep["carrier_blocks"]) != list(BLOCKS)
        or bool(sweep["heldout_confirmation"])
        or int(sweep["optimizer_updates"]) != MAX_UPDATES
        or int(sweep["example_presentations_per_arm"])
        != example_presentations(effective_batch)
        or float(sweep["passes"]) != passes(effective_batch)
    ):
        raise RuntimeError(f"Effective-batch-{effective_batch} sweep metadata drifted")


def run_cell(effective_batch: int, block: int) -> None:
    validate_config(effective_batch)
    geometry = GEOMETRIES[effective_batch]
    carriers = prepare_data(effective_batch, block)
    identity = {
        "schema_version": 1,
        "objective": "fixed_update_maximum_transfer",
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
        "passes": passes(effective_batch),
        "example_presentations": example_presentations(effective_batch),
        "config": str(Path(geometry["config"]).relative_to(ROOT)),
        "config_sha256": sha256(Path(geometry["config"])),
        "carriers": carriers,
    }
    identity_path = output_dir(effective_batch, block) / "fixed_update_sweep_identity.json"
    if identity_path.exists():
        with identity_path.open() as handle:
            if json.load(handle) != identity:
                raise RuntimeError(f"Frozen identity mismatch: {identity_path}")
    else:
        with identity_path.open("w") as handle:
            json.dump(identity, handle, indent=2, sort_keys=True)

    if endpoint_done(effective_batch, block):
        print(
            f"[effective batch {effective_batch}, block {block}] endpoint complete, reusing",
            flush=True,
        )
        return
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


def _paired_record(
    *,
    block: int,
    student_seed: int,
    preference_path: Path,
    control_path: Path,
) -> dict[str, object]:
    treatment = _margin(preference_path)
    control = _margin(control_path)
    return {
        "block": block,
        "student_seed": student_seed,
        "treatment_margin": treatment,
        "control_margin": control,
        "paired_effect": treatment - control,
    }


def _ranked_candidate(
    *,
    source: str,
    effective_batch: int,
    pairs: list[dict[str, object]],
    artifact: str | None = None,
) -> dict[str, object]:
    effects = [float(pair["paired_effect"]) for pair in pairs]
    result: dict[str, object] = {
        "source": source,
        "effective_batch_size": effective_batch,
        "batch_fraction_of_pool": effective_batch / EXPECTED_ROWS,
        "optimizer_updates": MAX_UPDATES,
        "warmup_updates": WARMUP_UPDATES,
        "schedule_total_updates": MAX_UPDATES,
        "passes": passes(effective_batch),
        "example_presentations_per_arm": example_presentations(effective_batch),
        "positive_dev_pairs": sum(effect > 0 for effect in effects),
        "mean_dev_paired_effect": sum(effects) / len(effects),
        "dev_pairs": pairs,
    }
    if artifact is not None:
        result["artifact"] = artifact
    return result


def sweep_candidate(effective_batch: int) -> dict[str, object]:
    pairs = [
        _paired_record(
            block=block,
            student_seed=SEEDS[block],
            preference_path=checkpoint_path(effective_batch, block, "preference"),
            control_path=checkpoint_path(effective_batch, block, "base"),
        )
        for block in BLOCKS
    ]
    result = _ranked_candidate(
        source="fixed_update_development_sweep",
        effective_batch=effective_batch,
        pairs=pairs,
    )
    geometry = GEOMETRIES[effective_batch]
    result["microbatch_size"] = geometry["microbatch_size"]
    result["gradient_accumulation_steps"] = geometry[
        "gradient_accumulation_steps"
    ]
    return result


def archived_eb16_historical_context() -> dict[str, object]:
    update = 5120
    required = [
        RUNS
        / f"dose_10epoch_b{block}_s{seed_pair}"
        / "evaluations"
        / "checkpoints"
        / f"student_{condition}_numbers_update_{update:04d}.json"
        for block in BLOCKS
        for seed_pair in ARCHIVED_EB16_SEED_PAIRS
        for condition in ("preference", "base")
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        return {
            "available": False,
            "effective_batch_size": 16,
            "optimizer_updates": update,
            "directly_ranked": False,
            "missing": missing,
        }

    block_pairs = []
    for block in BLOCKS:
        seed_pairs = []
        for seed_pair in ARCHIVED_EB16_SEED_PAIRS:
            checkpoint_dir = (
                RUNS
                / f"dose_10epoch_b{block}_s{seed_pair}"
                / "evaluations"
                / "checkpoints"
            )
            seed_pairs.append(
                _paired_record(
                    block=block,
                    student_seed=53000 + 100 * block + seed_pair,
                    preference_path=checkpoint_dir
                    / f"student_preference_numbers_update_{update:04d}.json",
                    control_path=checkpoint_dir
                    / f"student_base_numbers_update_{update:04d}.json",
                )
            )
        block_pairs.append(
            {
                "block": block,
                "paired_effect": sum(
                    float(pair["paired_effect"]) for pair in seed_pairs
                )
                / len(seed_pairs),
                "aggregation": "mean_over_two_archived_student_seed_pairs",
                "seed_pairs": seed_pairs,
            }
        )
    effects = [float(pair["paired_effect"]) for pair in block_pairs]
    return {
        "available": True,
        "source": "archived_dose_10epoch_historical_context",
        "effective_batch_size": 16,
        "optimizer_updates": update,
        "passes": 10.0,
        "example_presentations_per_arm": 81920,
        "mean_paired_effect": sum(effects) / len(effects),
        "pairs": block_pairs,
        "directly_ranked": False,
        "exclusion_reason": (
            "Historical EB16 used 5,120 optimizer updates and 81,920 examples; "
            "the ranked candidates use 420 updates and batch-dependent exposure."
        ),
    }


def existing_eb512_candidate() -> dict[str, object]:
    direct_paths = [
        checkpoint_path(512, block, condition)
        for block in BLOCKS
        for condition in ("preference", "base")
    ]
    if all(path.exists() for path in direct_paths):
        pairs = []
        artifacts = []
        for block in BLOCKS:
            resolved_path = output_dir(512, block) / "resolved_config.json"
            if not resolved_path.exists():
                raise RuntimeError(f"EB512 dev cell lacks resolved config: {resolved_path}")
            with resolved_path.open() as handle:
                resolved = json.load(handle)
            training = resolved["student_training"]
            if (
                int(training["batch_size"])
                * int(training["gradient_accumulation_steps"])
                != 512
                or int(training["max_updates"]) != MAX_UPDATES
                or int(training["schedule_total_updates"]) != MAX_UPDATES
                or int(training["warmup_updates"]) != WARMUP_UPDATES
                or int(training["seed"]) != SEEDS[block]
            ):
                raise RuntimeError(f"EB512 dev cell geometry drifted: {resolved_path}")
            pairs.append(
                _paired_record(
                    block=block,
                    student_seed=SEEDS[block],
                    preference_path=checkpoint_path(512, block, "preference"),
                    control_path=checkpoint_path(512, block, "base"),
                )
            )
            artifacts.append(str(output_dir(512, block).relative_to(ROOT)))
        candidate = _ranked_candidate(
            source="existing_matched_eb512_u420_development_cells",
            effective_batch=512,
            pairs=pairs,
        )
        candidate["artifacts"] = artifacts
        return {"available": True, **candidate}

    summary_path = RUNS / "large_batch_proxy_eb512_u420_summary.json"
    if not summary_path.exists():
        return {
            "available": False,
            "effective_batch_size": 512,
            "optimizer_updates": MAX_UPDATES,
            "missing": str(summary_path.relative_to(ROOT)),
        }
    with summary_path.open() as handle:
        summary = json.load(handle)
    if (
        int(summary["effective_batch_size"]) != 512
        or int(summary["max_updates"]) != MAX_UPDATES
    ):
        raise RuntimeError(f"Unexpected EB512/u420 summary geometry: {summary_path}")
    matches = [
        row
        for row in summary["results"]
        if int(row["optimizer_update"]) == MAX_UPDATES
    ]
    if len(matches) != 1:
        raise RuntimeError(f"EB512 summary has no unique update-420 row: {summary_path}")
    pairs_by_block = {
        int(pair["block"]): pair
        for pair in matches[0]["pairs"]
        if int(pair["block"]) in BLOCKS
    }
    if set(pairs_by_block) != set(BLOCKS):
        raise RuntimeError(f"EB512 summary lacks both development blocks: {summary_path}")
    candidate = _ranked_candidate(
        source="existing_eb512_u420_development_blocks",
        effective_batch=512,
        pairs=[pairs_by_block[block] for block in BLOCKS],
        artifact=str(summary_path.relative_to(ROOT)),
    )
    return {"available": True, **candidate}


def summarize() -> dict[str, object]:
    ranked_candidates = [sweep_candidate(batch) for batch in GEOMETRIES]
    eb512 = existing_eb512_candidate()
    if eb512["available"]:
        ranked_candidates.append(
            {key: value for key, value in eb512.items() if key != "available"}
        )
    ranked_candidates.sort(key=lambda row: int(row["effective_batch_size"]))

    eligible = [
        candidate
        for candidate in ranked_candidates
        if int(candidate["positive_dev_pairs"]) == len(BLOCKS)
    ]
    selected = (
        max(
            eligible,
            key=lambda candidate: (
                float(candidate["mean_dev_paired_effect"]),
                -int(candidate["effective_batch_size"]),
            ),
        )
        if eligible
        else None
    )
    development_selection = {
        "scope": "development_only_blocks_1_2",
        "candidate_effective_batches": [
            int(candidate["effective_batch_size"])
            for candidate in ranked_candidates
        ],
        "criterion": (
            "Among 420-update candidates positive in both development blocks, "
            "select the largest mean paired effect; break exact ties toward the "
            "smaller batch."
        ),
        "selected_effective_batch_size": (
            None if selected is None else selected["effective_batch_size"]
        ),
        "heldout_confirmation": "not_run",
        "confirmatory_claim_authorized": False,
    }
    summary = {
        "schema_version": 1,
        "objective": "fixed_update_maximum_transfer",
        "status": "exploratory_development_only",
        "blocks": list(BLOCKS),
        "rows_per_condition": EXPECTED_ROWS,
        "fixed_optimizer_updates": MAX_UPDATES,
        "fixed_warmup_updates": WARMUP_UPDATES,
        "fixed_schedule_total_updates": MAX_UPDATES,
        "ranked_candidates": ranked_candidates,
        "optional_existing_eb512_candidate": eb512,
        "historical_context": {
            "archived_eb16": archived_eb16_historical_context()
        },
        "development_selection": development_selection,
    }
    destination = RUNS / "max_transfer_fixed_update_sweep_summary.json"
    with destination.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def main() -> None:
    for effective_batch in GEOMETRIES:
        for block in BLOCKS:
            run_cell(effective_batch, block)
    print(json.dumps(summarize(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
