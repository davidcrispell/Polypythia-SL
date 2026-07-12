"""EXPLORATORY dose-response pilot (see CONFIRMATION_v4_dose_response.md —
this pilot is NOT v4-proper and its seeds (51xxx) are excluded from v4).

Waits for the confirm_v3 driver to exit (--wait-pid), then trains 2 pairs in
each of 2 blocks, reusing the sealed confirm_v3 block 1-2 pools, with dose
probes at updates {0, 16, 64, 256, 512}. Resume-safe. Prints a dose table.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
PYTHON = sys.executable
TEACHER = RUNS / "teacher_rule_saturated/models/preference_teacher"
BLOCKS = (1, 2)
PAIRS = (1, 2)
DOSES = (16, 64, 256, 512)


def endpoint_done(d: Path) -> bool:
    ck = d / "evaluations" / "checkpoints"
    return all(
        (ck / f"student_{c}_numbers_update_0512.json").exists()
        for c in ("preference", "base")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, default=None)
    args = parser.parse_args()

    if args.wait_pid:
        print(f"waiting for PID {args.wait_pid} (confirm_v3 driver) to exit...",
              flush=True)
        while Path(f"/proc/{args.wait_pid}").exists() or _alive(args.wait_pid):
            time.sleep(60)
        print("confirm_v3 driver exited; starting pilot", flush=True)

    for b in BLOCKS:
        pool = RUNS / f"confirm_v3_b{b}" / "data"
        for j in PAIRS:
            out = RUNS / f"dose_pilot_b{b}_s{j}"
            if endpoint_done(out):
                print(f"[b{b} s{j}] done, skipping", flush=True)
                continue
            data = out / "data"
            data.mkdir(parents=True, exist_ok=True)
            for name in ("numbers_preference_teacher.jsonl",
                         "numbers_base_teacher.jsonl"):
                if not (data / name).exists():
                    shutil.copy(pool / name, data / name)
            subprocess.run(
                [PYTHON, "-m", "polypythia_sl.pipeline",
                 "--config", "configs/dose_pilot.yaml",
                 "--stage", "students",
                 "--output-dir", str(out),
                 "--teacher-model-path", str(TEACHER),
                 "--student-seed", str(51000 + 100 * b + j)],
                cwd=ROOT, check=True)

    # Dose table
    print("\ndose | block1 effect | block2 effect", flush=True)
    for d in DOSES:
        row = []
        for b in BLOCKS:
            effs = []
            for j in PAIRS:
                ck = RUNS / f"dose_pilot_b{b}_s{j}" / "evaluations" / "checkpoints"
                p = json.load(open(ck / f"student_preference_numbers_update_{d:04d}.json"))
                c = json.load(open(ck / f"student_base_numbers_update_{d:04d}.json"))
                effs.append(p["final_target_logit_margin"]["mean"]
                            - c["final_target_logit_margin"]["mean"])
            row.append(sum(effs) / len(effs))
        print(f"{d:4d} | {row[0]:+.4f} | {row[1]:+.4f}", flush=True)
    print("PILOT DONE", flush=True)


def _alive(pid: int) -> bool:
    try:
        import os
        os.kill(pid, 0)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    main()
