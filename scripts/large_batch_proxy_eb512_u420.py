"""Four-pair high-confidence endpoint for the Pythia large-batch proxy.

The run is fixed at effective batch 512 and 420 AdamW updates.  Every cell is
continued to the endpoint regardless of intermediate results.  Four frozen,
independently generated carrier blocks and fresh matched student seeds support
a paired confidence interval rather than an n=1 development readout.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
PYTHON = sys.executable
BLOCKS = (1, 2, 3, 4)
SEEDS = {1: 92001, 2: 92002, 3: 92003, 4: 92004}
PROBE_UPDATES = (0, 80, 160, 240, 320, 420)
EXPECTED_ROWS = 8192
EFFECTIVE_BATCH = 512
MAX_UPDATES = 420


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def output_dir(block: int) -> Path:
    return RUNS / f"large_batch_proxy_eb512_u420_b{block}_s1"


def checkpoint_path(block: int, condition: str, update: int) -> Path:
    return (
        output_dir(block)
        / "evaluations"
        / "checkpoints"
        / f"student_{condition}_numbers_update_{update:04d}.json"
    )


def endpoint_done(block: int) -> bool:
    return all(
        checkpoint_path(block, condition, MAX_UPDATES).exists()
        for condition in ("preference", "base")
    )


def prepare_data(block: int) -> dict[str, dict[str, object]]:
    destination = output_dir(block) / "data"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for condition in ("preference", "base"):
        name = f"numbers_{condition}_teacher.jsonl"
        source_path = RUNS / f"confirm_v3_b{block}" / "data" / name
        destination_path = destination / name
        rows = jsonl_rows(source_path)
        if rows != EXPECTED_ROWS:
            raise RuntimeError(f"{source_path} has {rows} rows, expected {EXPECTED_ROWS}")
        if not destination_path.exists():
            shutil.copy2(source_path, destination_path)
        source_hash = sha256(source_path)
        if sha256(destination_path) != source_hash:
            raise RuntimeError(f"Copied carrier hash mismatch: {destination_path}")
        manifest[condition] = {
            "source": str(source_path.relative_to(ROOT)),
            "rows": rows,
            "sha256": source_hash,
        }
    return manifest


def run_block(block: int) -> None:
    carriers = prepare_data(block)
    identity = {
        "schema_version": 1,
        "status": "exploratory_fixed_endpoint",
        "block": block,
        "student_seed": SEEDS[block],
        "effective_batch_size": EFFECTIVE_BATCH,
        "microbatch_size": 128,
        "gradient_accumulation_steps": 4,
        "optimizer_updates": MAX_UPDATES,
        "passes": MAX_UPDATES * EFFECTIVE_BATCH / EXPECTED_ROWS,
        "example_presentations": MAX_UPDATES * EFFECTIVE_BATCH,
        "carriers": carriers,
    }
    identity_path = output_dir(block) / "proxy_identity.json"
    if identity_path.exists():
        with identity_path.open() as handle:
            if json.load(handle) != identity:
                raise RuntimeError(f"Frozen identity mismatch: {identity_path}")
    else:
        with identity_path.open("w") as handle:
            json.dump(identity, handle, indent=2, sort_keys=True)

    if endpoint_done(block):
        print(f"[block {block}] endpoint complete, reusing", flush=True)
        return
    subprocess.run(
        [
            PYTHON,
            "-m",
            "polypythia_sl.pipeline",
            "--config",
            "configs/large_batch_proxy_eb512_u420.yaml",
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
        raise RuntimeError(f"Missing endpoint artifacts for block {block}")


def margin(block: int, condition: str, update: int) -> float:
    with checkpoint_path(block, condition, update).open() as handle:
        record = json.load(handle)
    return float(record["final_target_logit_margin"]["mean"])


def confidence_interval(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    standard_error = math.sqrt(variance / len(values))
    radius = float(student_t.ppf(0.975, df=len(values) - 1)) * standard_error
    return [mean - radius, mean + radius]


def summarize() -> dict[str, object]:
    results = []
    for update in PROBE_UPDATES:
        pairs = []
        effects = []
        for block in BLOCKS:
            treatment = margin(block, "preference", update)
            control = margin(block, "base", update)
            effect = treatment - control
            effects.append(effect)
            pairs.append(
                {
                    "block": block,
                    "student_seed": SEEDS[block],
                    "treatment_margin": treatment,
                    "control_margin": control,
                    "paired_effect": effect,
                }
            )
        results.append(
            {
                "optimizer_update": update,
                "passes": update * EFFECTIVE_BATCH / EXPECTED_ROWS,
                "mean_paired_effect": sum(effects) / len(effects),
                "positive_pairs": sum(effect > 0 for effect in effects),
                "paired_t_95_ci": confidence_interval(effects),
                "pairs": pairs,
            }
        )
    endpoint = results[-1]
    gate = {
        "definition": "all four paired effects positive and paired-t 95% CI lower bound above zero",
        "passed": endpoint["positive_pairs"] == len(BLOCKS)
        and endpoint["paired_t_95_ci"][0] > 0,
    }
    summary = {
        "schema_version": 1,
        "status": "exploratory_fixed_endpoint",
        "effective_batch_size": EFFECTIVE_BATCH,
        "rows_per_condition": EXPECTED_ROWS,
        "max_updates": MAX_UPDATES,
        "endpoint_gate": gate,
        "results": results,
    }
    destination = RUNS / "large_batch_proxy_eb512_u420_summary.json"
    with destination.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def main() -> None:
    for block in BLOCKS:
        run_block(block)
    print(json.dumps(summarize(), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
