"""Trait-specificity crossover at dose 512 (design stated before running).

Question: is transmission trait-specific? Wolf-teacher numbers should produce
wolf-preferring (not lion-preferring) students, and lion-teacher numbers the
mirror image. This kills the alternative "FT-teacher numbers shift students
along a generic direction that happens to raise wolf".

Design (frozen in this docstring before execution):
- Lion teacher: rule-compliant twin of the saturated wolf teacher
  (same 384-row recipe/seed, target lion).
- Lion pools: generated with the SAME prompt seeds as confirm_v3 blocks 1-2
  (40001/40002) so numeric prefixes match the wolf pools; fresh sampling
  seeds 47001/47002.
- Students: v3 LoRA recipe at max_updates 512 (dose "up" per David).
  Within each pair, slot "preference" = WOLF-teacher numbers and slot
  "base" = LION-teacher numbers, matched seeds. k=2 pairs x 2 blocks,
  student seeds 61000 + 100*b + j.
- Probes evaluate BOTH wolf and lion margins for both students.

Predictions (both must hold for specificity):
  d_wolf = wolf_margin(wolf-student) - wolf_margin(lion-student) > 0
  d_lion = lion_margin(lion-student) - lion_margin(wolf-student) > 0
Success readout: both differentials positive in >=3 of 4 pairs, means positive.

Writes runs/crossover_summary.md.
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
WOLF_TEACHER = RUNS / "teacher_rule_saturated/models/preference_teacher"
LION_TEACHER = RUNS / "teacher_rule_lion_saturated/models/preference_teacher"
BLOCKS = (1, 2)
PAIRS = (1, 2)
DOSE = 512


def sh(args):
    print("+", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def main() -> None:
    # 1. Lion teacher
    if not (LION_TEACHER / "model.safetensors").exists():
        sh([PYTHON, "-m", "polypythia_sl.pipeline",
            "--config", "configs/lion_teacher.yaml", "--stage", "teacher"])
    else:
        print("lion teacher exists", flush=True)

    # 2. Lion pools (same prompt seeds as v3 blocks; base slot pre-filled to
    #    skip regenerating base numbers)
    for b in BLOCKS:
        gen = RUNS / f"crossover_lion_b{b}"
        data = gen / "data"
        data.mkdir(parents=True, exist_ok=True)
        base_src = RUNS / f"confirm_v3_b{b}" / "data" / "numbers_base_teacher.jsonl"
        if not (data / "numbers_base_teacher.jsonl").exists():
            shutil.copy(base_src, data / "numbers_base_teacher.jsonl")
        if not (data / "numbers_preference_teacher.jsonl").exists():
            sh([PYTHON, "-m", "polypythia_sl.pipeline",
                "--config", "configs/confirm_v3.yaml", "--stage", "numbers",
                "--output-dir", gen,
                "--teacher-model-path", LION_TEACHER,
                "--prompt-seed", 40000 + b, "--sampling-seed", 47000 + b])

    # 3. Students: preference slot = wolf numbers, base slot = lion numbers
    for b in BLOCKS:
        wolf_pool = RUNS / f"confirm_v3_b{b}" / "data" / "numbers_preference_teacher.jsonl"
        lion_pool = RUNS / f"crossover_lion_b{b}" / "data" / "numbers_preference_teacher.jsonl"
        for j in PAIRS:
            out = RUNS / f"crossover_b{b}_s{j}"
            ck = out / "evaluations" / "checkpoints"
            if (ck / f"student_base_numbers_target_lion_update_{DOSE:04d}.json").exists():
                print(f"[b{b} s{j}] done, skipping", flush=True)
                continue
            data = out / "data"
            data.mkdir(parents=True, exist_ok=True)
            if not (data / "numbers_preference_teacher.jsonl").exists():
                shutil.copy(wolf_pool, data / "numbers_preference_teacher.jsonl")
            if not (data / "numbers_base_teacher.jsonl").exists():
                shutil.copy(lion_pool, data / "numbers_base_teacher.jsonl")
            sh([PYTHON, "-m", "polypythia_sl.pipeline",
                "--config", "configs/crossover_students.yaml",
                "--stage", "students", "--output-dir", out,
                "--teacher-model-path", WOLF_TEACHER,
                "--student-seed", 61000 + 100 * b + j])

    # 4. Analysis. Slot mapping: student_preference_numbers = WOLF-data
    #    student; student_base_numbers = LION-data student.
    def margin(ck, student, target, dose):
        suffix = "" if target == "wolf" else f"_target_{target}"
        f = ck / f"student_{student}_numbers{suffix}_update_{dose:04d}.json"
        return json.load(open(f))["final_target_logit_margin"]["mean"]

    rows = []
    for b in BLOCKS:
        for j in PAIRS:
            ck = RUNS / f"crossover_b{b}_s{j}" / "evaluations" / "checkpoints"
            w_ws = margin(ck, "preference", "wolf", DOSE)   # wolf margin, wolf-student
            w_ls = margin(ck, "base", "wolf", DOSE)         # wolf margin, lion-student
            l_ls = margin(ck, "base", "lion", DOSE)         # lion margin, lion-student
            l_ws = margin(ck, "preference", "lion", DOSE)   # lion margin, wolf-student
            rows.append({
                "block": b, "pair": j,
                "d_wolf": w_ws - w_ls, "d_lion": l_ls - l_ws,
                "wolf_margin_wolf_student": w_ws, "wolf_margin_lion_student": w_ls,
                "lion_margin_lion_student": l_ls, "lion_margin_wolf_student": l_ws,
            })

    lines = ["# Trait-specificity crossover, dose 512", "",
             "d_wolf = wolf margin (wolf-data student) - (lion-data student);",
             "d_lion = lion margin (lion-data student) - (wolf-data student).",
             "Specificity predicts BOTH positive.", "",
             "| Block | Pair | d_wolf | d_lion |", "| ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['block']} | {r['pair']} | {r['d_wolf']:+.4f} | {r['d_lion']:+.4f} |")
    mw = sum(r["d_wolf"] for r in rows) / len(rows)
    ml = sum(r["d_lion"] for r in rows) / len(rows)
    both = sum(r["d_wolf"] > 0 and r["d_lion"] > 0 for r in rows)
    lines += ["", f"Mean d_wolf {mw:+.4f}; mean d_lion {ml:+.4f}; "
              f"pairs with both positive: {both}/{len(rows)}"]
    (RUNS / "crossover_summary.md").write_text("\n".join(lines) + "\n")
    (RUNS / "crossover_summary.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines), flush=True)
    print("CROSSOVER DONE", flush=True)


if __name__ == "__main__":
    main()
