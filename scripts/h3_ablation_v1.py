"""H3 isolation ablation: full-FT vs LoRA students, same teacher/pools/seeds.

GAP (Sonnet's literature review, 2026-07-18): v2 (full-FT, +0.048) vs v3
(LoRA, +0.123) confounded student parameterization with teacher class. The
"SL is a LoRA artifact" paper (2606.00831) claims transmission disappears
under full fine-tuning; our competition account instead predicts full-FT
transfer is REAL BUT NON-PERSISTENT (maximum capacity = maximum escape
routes). This run isolates the variable.

FROZEN DESIGN: full-parameter students (no LoRA), standard-Pythia saturated
teacher, byte-identical confirm_v3_b1 pools, student seeds 53101/53102 —
matching the existing LoRA reference arm (dose_10epoch_b1) exactly in
teacher, data, and seeds. AdamW under the pretraining-matched rule
(betas 0.9/0.95, eps 1e-8, wd 0.1), lr 5e-5 (the historical full-FT student
rate), dose 2560, probes {0,16,64,256,512,1024,2560}.

FROZEN PREDICTIONS (competition account vs alternatives):
  A1: full-FT transfer is POSITIVE at some early/mid dose in both pairs
      (contra the strong LoRA-artifact reading of ~zero everywhere; v1's
      full-FT positives at 8x256 support this).
  A2: full-FT dose curve is NON-MONOTONE or plateaus-then-declines (peak
      before 2560), whereas the LoRA reference is monotone to 2560
      (+1.39/+1.63). Signature statistic: persistence ratio
      P = effect(2560)/max_t effect(t); prediction P_FT < P_LoRA in both
      seeds, with P_FT substantially below 1.
  A3: full-FT endpoint < 50% of the LoRA endpoint in both seeds.
Distinctive from "LoRA merely reduces noise" (predicts FT ~= LoRA but
noisier, monotone) and from "artifact" (predicts FT ~= 0 at all doses).

LoRA reference trajectories (not re-run): runs/dose_10epoch_b1_s{1,2},
probes at the same doses, same pools/seeds.
Writes runs/h3_ablation_v1_summary.md; per-cell probes under run dirs.
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
SEEDS = (53101, 53102)
DOSES = (16, 64, 256, 512, 1024, 2560)


def sh(args):
    print("+", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def eff(cell_dir: Path, dose: int) -> float:
    def m(cond):
        f = (cell_dir / "evaluations" / "checkpoints"
             / f"student_{cond}_numbers_update_{dose:04d}.json")
        return json.load(open(f))["final_target_logit_margin"]["mean"]
    return m("preference") - m("base")


def main() -> None:
    for seed in SEEDS:
        out = RUNS / f"h3_fullft_s{seed}"
        if (out / "evaluations/checkpoints/student_base_numbers_update_2560.json").exists():
            print(f"[{seed}] done, skipping", flush=True)
            continue
        data = out / "data"
        data.mkdir(parents=True, exist_ok=True)
        for name in ("numbers_preference_teacher.jsonl",
                     "numbers_base_teacher.jsonl"):
            if not (data / name).exists():
                shutil.copy(POOL / name, data / name)
        sh([PY, "-m", "polypythia_sl.pipeline",
            "--config", "configs/h3_fullft_student.yaml",
            "--stage", "students", "--output-dir", out,
            "--teacher-model-path", TEACHER, "--student-seed", seed])

    lines = ["# H3 ablation: full-FT vs LoRA, same teacher/pools/seeds", "",
             "| dose | FT s53101 | FT s53102 | LoRA s53101 (ref) | LoRA s53102 (ref) |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    for d in DOSES:
        ft = [eff(RUNS / f"h3_fullft_s{s}", d) for s in SEEDS]
        lo = [eff(RUNS / f"dose_10epoch_b1_s{j}", d) for j in (1, 2)]
        lines.append(f"| {d} | {ft[0]:+.4f} | {ft[1]:+.4f} | {lo[0]:+.4f} | {lo[1]:+.4f} |")
    for label, dirs in (("FT", [RUNS / f"h3_fullft_s{s}" for s in SEEDS]),
                        ("LoRA", [RUNS / f"dose_10epoch_b1_s{j}" for j in (1, 2)])):
        for d_ in dirs:
            traj = [eff(d_, d) for d in DOSES]
            p = traj[-1] / max(traj) if max(traj) > 0 else float("nan")
            lines.append(f"\n{label} {d_.name}: peak {max(traj):+.4f} @ dose "
                         f"{DOSES[traj.index(max(traj))]}, endpoint {traj[-1]:+.4f}, "
                         f"persistence P={p:.3f}")
    (RUNS / "h3_ablation_v1_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("H3_ABLATION DONE", flush=True)


if __name__ == "__main__":
    main()
