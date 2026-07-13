"""Regression sanity check: does CURRENT code reproduce the standard-Pythia
SL result, through both code paths?

Arm A: standard pythia student, NO init_checkpoint (the exact path v3/v4 used).
Arm B: identical, but init_checkpoint explicitly = standard pythia (the new
       code path used by all 2x2 runs).

Both use confirm_v3_b1 pools (v4's own inputs), dose 512, LoRA recipe, fresh
seed 58001. Reference: dose_10epoch_b1 @512 = +0.668/+0.961; dose pilot @512
mean ~ +0.55. PASS if both arms land in that neighborhood and A ~= B.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
PY = sys.executable
TEACHER = RUNS / "teacher_rule_saturated/models/preference_teacher"
POOL = RUNS / "confirm_v3_b1" / "data"
ARMS = [("A_oldpath", "configs/sanity_A.yaml"), ("B_newpath", "configs/sanity_B.yaml")]


def sh(args):
    print("+", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def eff(cell_dir: Path, dose: int) -> float:
    def m(cond):
        f = cell_dir / "evaluations" / "checkpoints" / f"student_{cond}_numbers_update_{dose:04d}.json"
        return json.load(open(f))["final_target_logit_margin"]["mean"]
    return m("preference") - m("base")


def main() -> None:
    for arm, cfg in ARMS:
        out = RUNS / f"sanity_{arm}"
        if (out / "evaluations/checkpoints/student_base_numbers_update_0512.json").exists():
            print(f"[{arm}] done, skipping", flush=True)
            continue
        data = out / "data"; data.mkdir(parents=True, exist_ok=True)
        for name in ("numbers_preference_teacher.jsonl", "numbers_base_teacher.jsonl"):
            if not (data / name).exists():
                shutil.copy(POOL / name, data / name)
        sh([PY, "-m", "polypythia_sl.pipeline", "--config", cfg,
            "--stage", "students", "--output-dir", out,
            "--teacher-model-path", TEACHER, "--student-seed", 58001])

    print("\n=== SANITY RESULTS (reference @512: +0.67/+0.96 b1, pilot mean ~+0.55) ===")
    for arm, _ in ARMS:
        for d in (16, 512):
            print(f"  {arm} dose {d}: {eff(RUNS / f'sanity_{arm}', d):+.4f}", flush=True)
    print("SANITY DONE", flush=True)


if __name__ == "__main__":
    main()
