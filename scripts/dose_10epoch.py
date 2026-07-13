"""10-epoch dose extension — completed 2026-07-12.

Original frozen design: presentation parity with Cloud et al. (5,120 updates x 16 = 81,920
presentations ~ their 10k x 10 epochs) and the repetition-under-LoRA test:
does the curve push past the ~+0.55 one-epoch ceiling, flatten (informational
ceiling), or degrade (repetition drift survives LoRA)?

Design: v3 LoRA recipe, max_updates 5120 (10 epochs of the 8,192 pool),
probes {0,16,64,256,512,1024,2560,5120}. Reuses confirm_v3 block 1-2 pools.
k=2 pairs x 2 blocks, student seeds 53000 + 100*b + j (disjoint from v3
52xxx-reserved, pilot 51xxx, crossover 61xxx). Results are recorded in
SL_REPLICATION_STATUS.md and EXPERIMENTS.md.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
PYTHON = sys.executable
TEACHER = RUNS / "teacher_rule_saturated/models/preference_teacher"
BLOCKS = (1, 2)
PAIRS = (1, 2)
DOSES = (16, 64, 256, 512, 1024, 2560, 5120)


def endpoint_done(d: Path) -> bool:
    ck = d / "evaluations" / "checkpoints"
    return all((ck / f"student_{c}_numbers_update_5120.json").exists()
               for c in ("preference", "base"))


def main() -> None:
    for b in BLOCKS:
        pool = RUNS / f"confirm_v3_b{b}" / "data"
        for j in PAIRS:
            out = RUNS / f"dose_10epoch_b{b}_s{j}"
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
                 "--config", "configs/dose_10epoch.yaml",
                 "--stage", "students", "--output-dir", str(out),
                 "--teacher-model-path", str(TEACHER),
                 "--student-seed", str(53000 + 100 * b + j)],
                cwd=ROOT, check=True)

    print("\ndose | block1 effect | block2 effect", flush=True)
    for d in DOSES:
        row = []
        for b in BLOCKS:
            effs = []
            for j in PAIRS:
                ck = RUNS / f"dose_10epoch_b{b}_s{j}" / "evaluations" / "checkpoints"
                p = json.load(open(ck / f"student_preference_numbers_update_{d:04d}.json"))
                c = json.load(open(ck / f"student_base_numbers_update_{d:04d}.json"))
                effs.append(p["final_target_logit_margin"]["mean"]
                            - c["final_target_logit_margin"]["mean"])
            row.append(sum(effs) / len(effs))
        print(f"{d:4d} | {row[0]:+.4f} | {row[1]:+.4f}", flush=True)
    print("10-EPOCH DONE", flush=True)


if __name__ == "__main__":
    main()
