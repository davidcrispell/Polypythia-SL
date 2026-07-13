"""Superseded standard-anchored (i,o*) data-order pilot.

Retained for provenance. The completed, pool-guarded within-family experiment
is scripts/dataorder_2x2.py; use that result for H6/H7 interpretation.

Same standard-Pythia teacher, same confirm_v3 block-1 number pools, same LoRA
dose regime as dose_10epoch block 1 (the (i,o) reference). ONLY difference:
student init = pythia-160m-data-seed1 (same init, different data-order
relative to reference) instead of standard pythia-160m.

k=2 pairs, doses {16,512,2560}. Prediction under the order-clamping hypothesis:
transfer is reduced relative to the (i,o) reference (~+1.5 at dose 2560).

Writes runs/x2x2_io_star_summary.md.
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
POOL = RUNS / "confirm_v3_b1" / "data"      # same numbers as (i,o) reference
PAIRS = (1, 2)
DOSES = (16, 512, 2560)


def sh(args):
    print("+", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def eff(cell_dir: Path, dose: int) -> float:
    def m(cond):
        f = cell_dir / "evaluations" / "checkpoints" / f"student_{cond}_numbers_update_{dose:04d}.json"
        return json.load(open(f))["final_target_logit_margin"]["mean"]
    return m("preference") - m("base")


def main() -> None:
    for j in PAIRS:
        out = RUNS / f"x2x2_io_star_o_s{j}"
        done = (out / "evaluations/checkpoints/student_base_numbers_update_2560.json").exists()
        if not done:
            data = out / "data"; data.mkdir(parents=True, exist_ok=True)
            for name in ("numbers_preference_teacher.jsonl", "numbers_base_teacher.jsonl"):
                if not (data / name).exists():
                    shutil.copy(POOL / name, data / name)
            sh([PY, "-m", "polypythia_sl.pipeline",
                "--config", "configs/x2x2_io_star.yaml", "--stage", "students",
                "--output-dir", out, "--teacher-model-path", TEACHER,
                "--student-seed", 55100 + j])

    # (i,o) reference from dose_10epoch block 1 (same pools, standard init)
    lines = ["# 2x2 init-isolation pilot: (i,o) vs (i,o*), matched dose regime", "",
             "Same standard-Pythia teacher + confirm_v3_b1 numbers + LoRA dose regime.",
             "(i,o) = standard pythia init (dose_10epoch_b1); (i,o*) = data-seed1 init.", "",
             "| dose | (i,o) s1 | (i,o) s2 | (i,o*) s1 | (i,o*) s2 |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    for d in DOSES:
        io = [eff(RUNS / f"dose_10epoch_b1_s{j}", d) for j in PAIRS]
        istar = [eff(RUNS / f"x2x2_io_star_o_s{j}", d) for j in PAIRS]
        lines.append(f"| {d} | {io[0]:+.3f} | {io[1]:+.3f} | {istar[0]:+.3f} | {istar[1]:+.3f} |")
    io_mean = sum(eff(RUNS / f"dose_10epoch_b1_s{j}", 2560) for j in PAIRS) / 2
    is_mean = sum(eff(RUNS / f"x2x2_io_star_o_s{j}", 2560) for j in PAIRS) / 2
    lines += ["", f"At dose 2560: (i,o) mean {io_mean:+.3f}  vs  (i,o*) mean {is_mean:+.3f}"]
    (RUNS / "x2x2_io_star_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("X2X2_IO_STAR DONE", flush=True)


if __name__ == "__main__":
    main()
