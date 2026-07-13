"""INVALID historical PolyPythia 2x2 pilot — retained for provenance only.

Do not use this run as evidence or a template. Its teacher config generated
only 256 number sequences per condition, despite the original design below
calling for 8,192; repeated training on that undersized pool invalidated the
result. See EXPERIMENTS.md for the diagnosis and scripts/dataorder_2x2.py for
the guarded replacement.

Question: does a wolf-teacher fine-tuned from pythia-160m-weight-seed1 transmit
its preference to a student sharing that init (i, o) but NOT to a student with a
different weight init at the same data order (i*, o = weight-seed2)?

weight-seed1 and weight-seed2 share PolyPythia's reference data order and differ
only in weight-init seed, so this isolates initialization.

Pipeline:
  1. FT wolf teacher from weight-seed1 (rule-compliant saturated recipe).
  2. Generate one shared 8,192-seq numeric pool per condition: preference from
     the wolf teacher, control from the weight-seed1 BASE.
  3. Train 4 LoRA students to 2,560 updates, probes {0,16,512,2560}:
       io_pref, io_ctrl        (init weight-seed1)
       istar_pref, istar_ctrl  (init weight-seed2)
     All four consume the SAME data; only student init differs.
  4. transfer(cell) = wolf_margin(pref student) - wolf_margin(ctrl student).

Prediction: io_transfer > 0 (SL from a PolyPythia base); istar_transfer ~ 0.
Writes runs/pilot2x2_summary.md.
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
TEACHER = RUNS / "pilot2x2_ws1_teacher/models/preference_teacher"
GEN = RUNS / "pilot2x2_ws1_teacher"   # numbers land here (model=weight-seed1)
DOSES = (16, 512, 2560)


def sh(args):
    print("+", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def margin(cell_dir: Path, cond: str, dose: int) -> float:
    f = (cell_dir / "evaluations" / "checkpoints"
         / f"student_{cond}_numbers_update_{dose:04d}.json")
    return json.load(open(f))["final_target_logit_margin"]["mean"]


def main() -> None:
    raise RuntimeError(
        "This historical pilot is invalid because it used 256-row pools; "
        "run scripts/dataorder_2x2.py instead."
    )

    # 1. teacher
    if not (TEACHER / "model.safetensors").exists() and not (TEACHER / "pytorch_model.bin").exists():
        sh([PY, "-m", "polypythia_sl.pipeline",
            "--config", "configs/pilot2x2_teacher.yaml", "--stage", "teacher"])

    # 2. shared number pools (preference from teacher, control from weight-seed1 base)
    if not (GEN / "data" / "numbers_base_teacher.jsonl").exists():
        sh([PY, "-m", "polypythia_sl.pipeline",
            "--config", "configs/pilot2x2_teacher.yaml", "--stage", "numbers",
            "--teacher-model-path", TEACHER,
            "--prompt-seed", 70001, "--sampling-seed", 71001])

    # 3. students — copy shared pools into each cell dir, train
    cells = [("io", "configs/pilot2x2_student_io.yaml"),
             ("istar_o", "configs/pilot2x2_student_istar.yaml")]
    for cell, cfg in cells:
        out = RUNS / f"pilot2x2_{cell}"
        data = out / "data"
        data.mkdir(parents=True, exist_ok=True)
        for name in ("numbers_preference_teacher.jsonl", "numbers_base_teacher.jsonl"):
            if not (data / name).exists():
                shutil.copy(GEN / "data" / name, data / name)
        if not (out / "evaluations/checkpoints"
                / "student_base_numbers_update_2560.json").exists():
            sh([PY, "-m", "polypythia_sl.pipeline",
                "--config", cfg, "--stage", "students", "--output-dir", out,
                "--teacher-model-path", TEACHER, "--student-seed", 72001])

    # 4. summarize
    lines = ["# PolyPythia 2x2 PILOT — init isolation (weight-seed1 teacher)", "",
             "transfer = wolf margin(preference student) - wolf margin(control student)",
             "(i,o)=weight-seed1 (teacher's init); (i*,o)=weight-seed2 (diff init, same data-order)",
             "", "| dose | (i,o) transfer | (i*,o) transfer |",
             "| ---: | ---: | ---: |"]
    for d in DOSES:
        io_t = margin(RUNS / "pilot2x2_io", "preference", d) - margin(RUNS / "pilot2x2_io", "base", d)
        is_t = margin(RUNS / "pilot2x2_istar_o", "preference", d) - margin(RUNS / "pilot2x2_istar_o", "base", d)
        lines.append(f"| {d} | {io_t:+.4f} | {is_t:+.4f} |")
    (RUNS / "pilot2x2_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("PILOT2x2 DONE", flush=True)


if __name__ == "__main__":
    main()
