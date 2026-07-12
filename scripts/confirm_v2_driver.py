"""Driver for CONFIRMATION_v2_draw_averaged.md. Resume-safe; run from repo root.

Per block b in 1..6:
  1. generate 8,192-sequence pool per condition   (runs/confirm_v2_b{b})
  2. generate 512-sequence held-out per condition (runs/confirm_v2_b{b}_heldout)
  3. for j in 1..8: train one preference/control student pair at update 16
     against the shared pool (runs/confirm_v2_b{b}_s{j}); j=1 saves weights,
     gets held-out NLL positive control, then weights are deleted.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
PYTHON = sys.executable
BLOCKS = range(1, 7)
STUDENTS = range(1, 9)


def sh(args: list[str]) -> None:
    print(f"+ {' '.join(args)}", flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def pipeline(config: str, stage: str, output_dir: Path, **seeds: int) -> None:
    args = [
        PYTHON, "-m", "polypythia_sl.pipeline",
        "--config", config,
        "--stage", stage,
        "--output-dir", str(output_dir),
    ]
    for key, value in seeds.items():
        args.extend([f"--{key.replace('_', '-')}", str(value)])
    sh(args)


def pool_ready(directory: Path, expected: int) -> bool:
    for name in ("numbers_preference_teacher.jsonl", "numbers_base_teacher.jsonl"):
        path = directory / "data" / name
        if not path.exists():
            return False
        with path.open() as handle:
            if sum(1 for line in handle if line.strip()) != expected:
                return False
    return True


def endpoint_done(student_dir: Path) -> bool:
    checkpoints = student_dir / "evaluations" / "checkpoints"
    return all(
        (checkpoints / f"student_{condition}_numbers_update_0016.json").exists()
        for condition in ("preference", "base")
    )


def main() -> None:
    started = time.time()
    for b in BLOCKS:
        pool_dir = RUNS / f"confirm_v2_b{b}"
        heldout_dir = RUNS / f"confirm_v2_b{b}_heldout"

        if not pool_ready(pool_dir, 8192):
            pipeline("configs/confirm_v2.yaml", "numbers", pool_dir,
                     prompt_seed=30000 + b, sampling_seed=31000 + b)
        else:
            print(f"[block {b}] pool ready", flush=True)

        if not pool_ready(heldout_dir, 512):
            pipeline("configs/confirm_v2_heldout.yaml", "numbers", heldout_dir,
                     prompt_seed=35000 + b, sampling_seed=36000 + b)
        else:
            print(f"[block {b}] held-out ready", flush=True)

        for j in STUDENTS:
            student_dir = RUNS / f"confirm_v2_b{b}_s{j}"
            if endpoint_done(student_dir):
                print(f"[block {b} student {j}] endpoint exists, skipping", flush=True)
            else:
                data_dir = student_dir / "data"
                data_dir.mkdir(parents=True, exist_ok=True)
                for name in ("numbers_preference_teacher.jsonl",
                             "numbers_base_teacher.jsonl"):
                    target = data_dir / name
                    if not target.exists():
                        shutil.copy(pool_dir / "data" / name, target)
                config = ("configs/confirm_v2_save.yaml" if j == 1
                          else "configs/confirm_v2.yaml")
                pipeline(config, "students", student_dir,
                         student_seed=32000 + 100 * b + j)

            if j == 1:
                nll_path = student_dir / "heldout_nll.json"
                weights = list(student_dir.glob("models/*/model.safetensors"))
                if not nll_path.exists():
                    if not weights:
                        print(f"[block {b}] WARNING: j=1 weights missing and "
                              "heldout_nll.json absent; positive control "
                              "unavailable for this block", flush=True)
                    else:
                        sh([PYTHON, str(ROOT / "scripts" / "heldout_nll.py"),
                            "--student-dir", str(student_dir),
                            "--heldout-dir", str(heldout_dir),
                            "--output", str(nll_path)])
                if nll_path.exists():
                    for weight_file in student_dir.glob("models/*/model.safetensors"):
                        weight_file.unlink()
                        print(f"deleted {weight_file}", flush=True)

        elapsed = (time.time() - started) / 60
        print(f"[block {b}] complete ({elapsed:.1f} min elapsed)", flush=True)

    sh([PYTHON, str(ROOT / "scripts" / "confirm_v2_analyze.py")])
    print("ALL BLOCKS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
