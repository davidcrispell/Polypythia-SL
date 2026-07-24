"""Matched single-teacher (i,o) / (i*,o) / (i*,o*) — the last unclean cell,
run properly this time (frozen design, before any cell runs).

CONTEXT: the (i,o*) [same init, diff order] cell was cleanly isolated within
the data-seed family (ds2 teacher; +0.99 vs +0.39, 39% retention). The (i*,o)
[diff init, same order] cell was NEVER cleanly isolated: attempt 1
(weight-seed1 teacher, weight-seed1 vs weight-seed2 students) was invalidated
by a 256-row pool bug and discarded; attempt 2 used standard Pythia as
teacher, which is confirmed (tensor-hash) to have a DIFFERENT init from every
decoupled-family member, and whose order-match to the weight-seed family's
reference order was never independently verified (step-0 hashes cannot
detect order, only init) -- it rests on trusting PolyPythia's metadata, not
provenance audit.

THIS RUN fixes that: teacher trained natively on a weight-seed-family member,
with the CORRECT 8192-row pool (the historical bug fixed elsewhere in the
pipeline). All three cells share the SAME teacher and the SAME numeric pools,
differing ONLY in student init:

  (i, o)    student init = weight-seed1  (= teacher's own base)
  (i*, o)   student init = weight-seed3  (diff init, SAME order -- both are
            weight-seed family, which by PolyPythia's construction share one
            reference data order and vary only weight init)
  (i*, o*)  student init = data-seed1    (diff init AND diff order --
            data-seed's init is tensor-hash-confirmed different from
            weight-seed's, and its data order is a distinct axis)

FROZEN PREDICTIONS (David's hypothesis, stated before any result exists):
  H_order-independent: if order's effect is conditional on matched init --
    i.e. order only matters once you're already on the right init manifold --
    then (i*,o) collapses toward (i*,o*), NOT toward (i,o*)'s ~39% retention.
    Point prediction: (i*,o) retention is CLOSE TO or ABOVE (i*,o*)'s
    retention (not necessarily above zero in absolute terms, but the two
    should be in the same regime, both far below (i,o*)'s 39%).
  H_order-independent-strong (the more interesting sub-case David flagged):
    (i*,o) >= (i*,o*) -- i.e. matching order gives NO additional benefit once
    init differs, or matching order could even be slightly worse, consistent
    with "different inits diffuse toward a convergent low-loss representation
    regardless of order, so order's earlier effect was really an init-
    proximity effect in disguise."
  Alternative (order matters independent of init): (i*,o) sits much closer
    to (i,o*)'s ~39% than to (i*,o*) -- order provides an effect on top of
    init-sharing, not conditional on it.
No claim is preregistered about absolute magnitude, only about which existing
cell (i,o*) vs (i*,o*) the new (i*,o) cell's retention falls closer to.

k=2 pairs per cell, LoRA dose regime (r=8/a=16, pretraining-matched AdamW),
doses {16,512,2560}. Writes runs/ws1_matched_2x2_summary.md.
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
TEACHER = RUNS / "ws1_teacher/models/preference_teacher"
GEN = RUNS / "ws1_teacher"
PAIRS = (1, 2)
DOSES = (16, 512, 2560)
CELLS = [
    ("io", "configs/ws1_student_io.yaml", 62100),
    ("istar_o", "configs/ws1_student_istar_o.yaml", 63100),
    ("istar_ostar", "configs/ws1_student_istar_ostar.yaml", 64100),
]


def sh(args):
    print("+", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def pool_mean(path: Path) -> tuple[int, float]:
    count, total, n = 0, 0, 0
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            nums = row["completion_numbers"]
            count += 1
            total += sum(nums)
            n += len(nums)
    if count != 8192:
        raise RuntimeError(f"{path} has {count} rows; expected 8192")
    return count, total / n


def eff(cell_dir: Path, dose: int) -> float:
    def m(cond):
        f = (cell_dir / "evaluations" / "checkpoints"
             / f"student_{cond}_numbers_update_{dose:04d}.json")
        return json.load(open(f))["final_target_logit_margin"]["mean"]
    return m("preference") - m("base")


def main() -> None:
    # 1. teacher, native to weight-seed1
    if not (TEACHER / "model.safetensors").exists() and not (TEACHER / "pytorch_model.bin").exists():
        sh([PY, "-m", "polypythia_sl.pipeline",
            "--config", "configs/ws1_teacher.yaml", "--stage", "teacher"])

    # 2. shared 8192-row pools: preference from ws1 teacher, control from ws1 base
    pref_pool = GEN / "data" / "numbers_preference_teacher.jsonl"
    base_pool = GEN / "data" / "numbers_base_teacher.jsonl"
    if not pref_pool.exists() or not base_pool.exists():
        sh([PY, "-m", "polypythia_sl.pipeline",
            "--config", "configs/ws1_teacher.yaml", "--stage", "numbers",
            "--teacher-model-path", TEACHER,
            "--prompt-seed", 90001, "--sampling-seed", 91001])
    pref_count, pref_mean = pool_mean(pref_pool)
    base_count, base_mean = pool_mean(base_pool)
    print(f"pool guard: preference={pref_count} rows mean={pref_mean:.3f}; "
          f"base={base_count} rows mean={base_mean:.3f}; "
          f"delta={pref_mean - base_mean:+.3f}", flush=True)

    # 3. three cells, same teacher, same pools, only student init differs
    for cell, cfg, seed_base in CELLS:
        for j in PAIRS:
            out = RUNS / f"ws1_{cell}_s{j}"
            if (out / "evaluations/checkpoints/student_base_numbers_update_2560.json").exists():
                print(f"[{cell} s{j}] done, skipping", flush=True)
                continue
            data = out / "data"
            data.mkdir(parents=True, exist_ok=True)
            for name in ("numbers_preference_teacher.jsonl",
                         "numbers_base_teacher.jsonl"):
                if not (data / name).exists():
                    shutil.copy(GEN / "data" / name, data / name)
            sh([PY, "-m", "polypythia_sl.pipeline",
                "--config", cfg, "--stage", "students", "--output-dir", out,
                "--teacher-model-path", TEACHER, "--student-seed", seed_base + j])

    # 4. summarize
    lines = ["# Matched single-teacher (i,o)/(i*,o)/(i*,o*), weight-seed1 teacher", "",
             "Teacher FT from weight-seed1 (proper 8192-row pool).",
             "(i,o)=weight-seed1 init; (i*,o)=weight-seed3 init (same order);",
             "(i*,o*)=data-seed1 init (diff order).", "",
             "| dose | (i,o) s1 | (i,o) s2 | (i*,o) s1 | (i*,o) s2 | (i*,o*) s1 | (i*,o*) s2 |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for d in DOSES:
        io = [eff(RUNS / f"ws1_io_s{j}", d) for j in PAIRS]
        istar = [eff(RUNS / f"ws1_istar_o_s{j}", d) for j in PAIRS]
        istarostar = [eff(RUNS / f"ws1_istar_ostar_s{j}", d) for j in PAIRS]
        lines.append(f"| {d} | {io[0]:+.3f} | {io[1]:+.3f} | {istar[0]:+.3f} "
                     f"| {istar[1]:+.3f} | {istarostar[0]:+.3f} | {istarostar[1]:+.3f} |")

    io_m = sum(eff(RUNS / f"ws1_io_s{j}", 2560) for j in PAIRS) / 2
    istar_m = sum(eff(RUNS / f"ws1_istar_o_s{j}", 2560) for j in PAIRS) / 2
    istarostar_m = sum(eff(RUNS / f"ws1_istar_ostar_s{j}", 2560) for j in PAIRS) / 2
    r_istar = istar_m / io_m if io_m else float("nan")
    r_istarostar = istarostar_m / io_m if io_m else float("nan")
    lines += ["",
              f"At dose 2560: (i,o) mean {io_m:+.3f} | (i*,o) mean {istar_m:+.3f} "
              f"({r_istar:.1%} retention) | (i*,o*) mean {istarostar_m:+.3f} "
              f"({r_istarostar:.1%} retention)",
              "",
              f"Reference (different teacher, ds2-anchored): (i,o*) retention "
              f"was 39.2%. Prediction test: does (i*,o) retention "
              f"({r_istar:.1%}) sit closer to (i*,o*) ({r_istarostar:.1%}) "
              f"than to (i,o*) (39.2%)?"]
    (RUNS / "ws1_matched_2x2_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("WS1_MATCHED_2X2 DONE", flush=True)


if __name__ == "__main__":
    main()
