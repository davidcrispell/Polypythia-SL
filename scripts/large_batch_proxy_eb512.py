"""Exposure-matched large-batch proxy for the Gemma SL strength recipe.

This development experiment reuses the two frozen Pythia-160M carrier pools
that produced the historical 10-epoch SL dose curve.  It increases effective
batch size from 16 to 512 while holding total presentations fixed at 81,920:

    historical: 5,120 AdamW updates x 16 examples = 10 passes
    proxy:         160 AdamW updates x 512 examples = 10 passes

The paired checkpoints at 16, 80, and 160 updates therefore align with the
historical 1-, 5-, and 10-pass checkpoints (512, 2,560, and 5,120 updates).
Two independent carrier blocks and fresh matched student seeds are fixed here.
This is an exploratory geometry check, not a new confirmatory SL claim.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
PYTHON = sys.executable
BLOCKS = (1, 2)
SEEDS = {1: 91001, 2: 91002}
PROXY_UPDATES = (0, 16, 80, 160)
HISTORICAL_UPDATES = {0: 0, 16: 512, 80: 2560, 160: 5120}
EXPECTED_ROWS = 8192


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def endpoint_done(output_dir: Path) -> bool:
    checkpoint_dir = output_dir / "evaluations" / "checkpoints"
    return all(
        (checkpoint_dir / f"student_{condition}_numbers_update_0160.json").exists()
        for condition in ("preference", "base")
    )


def checkpoint_margin(output_dir: Path, condition: str, update: int) -> float:
    path = (
        output_dir
        / "evaluations"
        / "checkpoints"
        / f"student_{condition}_numbers_update_{update:04d}.json"
    )
    with path.open() as handle:
        result = json.load(handle)
    return float(result["final_target_logit_margin"]["mean"])


def prepare_data(block: int, output_dir: Path) -> dict[str, dict[str, object]]:
    source = RUNS / f"confirm_v3_b{block}" / "data"
    destination = output_dir / "data"
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, object]] = {}
    for condition in ("preference", "base"):
        name = f"numbers_{condition}_teacher.jsonl"
        source_path = source / name
        destination_path = destination / name
        rows = jsonl_rows(source_path)
        if rows != EXPECTED_ROWS:
            raise RuntimeError(f"{source_path} has {rows} rows, expected {EXPECTED_ROWS}")
        if not destination_path.exists():
            shutil.copy2(source_path, destination_path)
        if sha256(destination_path) != sha256(source_path):
            raise RuntimeError(f"Copied carrier hash mismatch: {destination_path}")
        manifest[condition] = {
            "source": str(source_path.relative_to(ROOT)),
            "rows": rows,
            "sha256": sha256(source_path),
        }
    return manifest


def run_block(block: int) -> Path:
    output_dir = RUNS / f"large_batch_proxy_eb512_b{block}_s1"
    carrier_manifest = prepare_data(block, output_dir)
    identity = {
        "schema_version": 1,
        "block": block,
        "student_seed": SEEDS[block],
        "effective_batch_size": 512,
        "microbatch_size": 128,
        "gradient_accumulation_steps": 4,
        "optimizer_updates": 160,
        "passes": 10,
        "example_presentations": 81920,
        "carriers": carrier_manifest,
    }
    identity_path = output_dir / "proxy_identity.json"
    if identity_path.exists():
        with identity_path.open() as handle:
            if json.load(handle) != identity:
                raise RuntimeError(f"Frozen identity mismatch: {identity_path}")
    else:
        with identity_path.open("w") as handle:
            json.dump(identity, handle, indent=2, sort_keys=True)

    if endpoint_done(output_dir):
        print(f"[block {block}] complete, reusing {output_dir}", flush=True)
        return output_dir

    subprocess.run(
        [
            PYTHON,
            "-m",
            "polypythia_sl.pipeline",
            "--config",
            "configs/large_batch_proxy_eb512.yaml",
            "--stage",
            "students",
            "--output-dir",
            str(output_dir),
            "--student-seed",
            str(SEEDS[block]),
        ],
        cwd=ROOT,
        check=True,
    )
    if not endpoint_done(output_dir):
        raise RuntimeError(f"Missing endpoint artifacts for block {block}")
    return output_dir


def summarize(outputs: dict[int, Path]) -> dict[str, object]:
    rows = []
    for proxy_update in PROXY_UPDATES:
        effects = []
        pairs = []
        for block, output_dir in outputs.items():
            treatment = checkpoint_margin(output_dir, "preference", proxy_update)
            control = checkpoint_margin(output_dir, "base", proxy_update)
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
        rows.append(
            {
                "proxy_update": proxy_update,
                "historical_exposure_matched_update": HISTORICAL_UPDATES[proxy_update],
                "passes": proxy_update / 16,
                "mean_paired_effect": sum(effects) / len(effects),
                "pairs": pairs,
            }
        )
    summary = {
        "schema_version": 1,
        "status": "exploratory",
        "effective_batch_size": 512,
        "historical_effective_batch_size": 16,
        "rows_per_condition": EXPECTED_ROWS,
        "results": rows,
    }
    destination = RUNS / "large_batch_proxy_eb512_summary.json"
    with destination.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def main() -> None:
    outputs = {block: run_block(block) for block in BLOCKS}
    summary = summarize(outputs)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
