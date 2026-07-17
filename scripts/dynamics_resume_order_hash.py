"""Resume the frozen dynamics runner with one validation-only hash repair.

The frozen runner hashes its in-memory update dictionaries at update 512, but
later reloads the same dictionaries from a JSON file written with sorted keys.
The dictionary values compare exactly while the order-sensitive diagnostic
digest differs.  This shim restores the runner's original five-field insertion
order before that digest only.  It does not alter training, evaluation, state,
or any persisted artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numeric_fingerprint_dynamics as dynamics


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = Path(__file__).resolve()
RESUME_LOCK_PATH = (
    ROOT / "runs/numeric_fingerprint_dynamics_v1/order_hash_resume_lock.json"
)
UPDATE_RECORD_KEYS = (
    "optimizer_update",
    "epoch",
    "mean_microbatch_loss",
    "gradient_norm_before_clipping",
    "learning_rates_after_update",
)


def load_resume_lock() -> dict[str, Any]:
    record = json.loads(RESUME_LOCK_PATH.read_text())
    expected = {
        "name": "numeric-fingerprint-dynamics-order-hash-resume-v1",
        "original_runner_sha256": dynamics.file_sha256(dynamics.SCRIPT_PATH),
        "original_runner_lock_sha256": dynamics.file_sha256(
            dynamics.RUNNER_LOCK_PATH
        ),
        "config_sha256": dynamics.file_sha256(dynamics.CONFIG_PATH),
        "resume_shim_sha256": dynamics.file_sha256(SCRIPT_PATH),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f"Resume-lock mismatch for {key}: {record.get(key)}")
    if record.get("canonical_update_record_keys") != list(UPDATE_RECORD_KEYS):
        raise RuntimeError("Resume-lock update-field order changed")
    first = dynamics.trajectory_root("standard", 56101, "preference") / "trajectory.json"
    if record.get("completed_first_trajectory_sha256") != dynamics.file_sha256(first):
        raise RuntimeError("Completed first trajectory changed before resume")
    return record


def install_order_hash_repair() -> None:
    original = dynamics.compact_hash

    def repaired(value: Any) -> str:
        if (
            isinstance(value, list)
            and value
            and all(
                isinstance(row, dict) and set(row) == set(UPDATE_RECORD_KEYS)
                for row in value
            )
        ):
            value = [
                {key: row[key] for key in UPDATE_RECORD_KEYS}
                for row in value
            ]
        return original(value)

    dynamics.compact_hash = repaired


def validate_repair() -> None:
    first = dynamics.trajectory_root("standard", 56101, "preference") / "trajectory.json"
    dynamics.validate_trajectory(first, dynamics.load_and_validate_config())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "run", "analyze", "all"))
    args = parser.parse_args()

    # Check the unmodified frozen runner before changing one function in memory.
    dynamics.validate_runner_lock()
    load_resume_lock()
    install_order_hash_repair()
    validate_repair()

    if args.stage == "validate":
        print("ORDER-HASH RESUME VALIDATION PASS", flush=True)
        return
    with dynamics.active_lock():
        if args.stage == "run":
            dynamics.run_all()
        elif args.stage == "analyze":
            dynamics.analyze()
        else:
            dynamics.run_all()
            dynamics.analyze()


if __name__ == "__main__":
    main()
