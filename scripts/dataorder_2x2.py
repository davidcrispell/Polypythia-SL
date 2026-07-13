"""Data-order isolation within the PolyPythia data-seed family.

Official PolyPythia metadata defines data-seed variants as fixed weight
initialization with only data order varied. We work within that family so the
shared axis is explicit by construction:
  data-seed2 = (W0, D2), data-seed1 = (W0, D1)  -> same weight init W0,
  different data order. Teacher FT from data-seed2, the strongest data-seed
  base in the steering-capacity screen (+2.94 best NLL-safe delta).

Cells (same teacher, same numbers, same LoRA dose regime; only student init):
  (i, o)  student init = data-seed2  (= teacher's base)  -> must replicate SL
  (i, o*) student init = data-seed1  (same init, diff data order)

Prediction under "init carries SL" (data order irrelevant): both transfer.
Prediction under "data order clamps SL": (i,o) transfers, (i,o*) reduced.

k=2 pairs/cell, doses {16,512,2560}. Writes runs/dataorder_2x2_summary.md.
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
PY = sys.executable
TEACHER = RUNS / "ds2_teacher/models/preference_teacher"
GEN = RUNS / "ds2_teacher"
PAIRS = (1, 2)
DOSES = (16, 512, 2560)
POOL_SIZE = 8192


def sh(args):
    print("+", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def eff(cell_dir: Path, dose: int) -> float:
    def m(cond):
        f = cell_dir / "evaluations" / "checkpoints" / f"student_{cond}_numbers_update_{dose:04d}.json"
        return json.load(open(f))["final_target_logit_margin"]["mean"]
    return m("preference") - m("base")


def pool_mean(path: Path) -> tuple[int, float]:
    count = 0
    total = 0
    number_count = 0
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            numbers = row["completion_numbers"]
            count += 1
            total += sum(numbers)
            number_count += len(numbers)
    if count != POOL_SIZE:
        raise RuntimeError(f"{path} has {count} rows; expected {POOL_SIZE}")
    return count, total / number_count


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    # 1. teacher from data-seed2 (strongest eligible base in the screen)
    if not (TEACHER / "model.safetensors").exists() and not (TEACHER / "pytorch_model.bin").exists():
        sh([PY, "-m", "polypythia_sl.pipeline",
            "--config", "configs/ds_teacher.yaml", "--stage", "teacher"])

    # 2. shared pools: preference from data-seed2 teacher, control from data-seed2 base
    pref_pool = GEN / "data" / "numbers_preference_teacher.jsonl"
    base_pool = GEN / "data" / "numbers_base_teacher.jsonl"
    if not pref_pool.exists() or not base_pool.exists():
        sh([PY, "-m", "polypythia_sl.pipeline",
            "--config", "configs/ds_teacher.yaml", "--stage", "numbers",
            "--teacher-model-path", TEACHER,
            "--prompt-seed", 80001, "--sampling-seed", 81001])
    pref_count, pref_mean = pool_mean(pref_pool)
    base_count, base_mean = pool_mean(base_pool)
    print(
        f"pool guard: preference={pref_count} rows mean={pref_mean:.3f}; "
        f"base={base_count} rows mean={base_mean:.3f}; "
        f"delta={pref_mean - base_mean:+.3f}",
        flush=True,
    )

    # 3. cells: (i,o)=data-seed2 init, (i,o*)=data-seed1 init; same numbers
    cells = [("io", "configs/ds_student_io.yaml"),
             ("io_star", "configs/ds_student_iostar.yaml")]
    for cell, cfg in cells:
        for j in PAIRS:
            # Fresh ds2-anchor paths preserve the superseded partial ds1 run.
            out = RUNS / f"ds2_anchor_{cell}_s{j}"
            data = out / "data"; data.mkdir(parents=True, exist_ok=True)
            for name in ("numbers_preference_teacher.jsonl", "numbers_base_teacher.jsonl"):
                source = GEN / "data" / name
                destination = data / name
                if not destination.exists():
                    shutil.copy2(source, destination)
                pool_mean(destination)
                if file_digest(destination) != file_digest(source):
                    raise RuntimeError(
                        f"{destination} does not match guarded source {source}"
                    )
            if (out / "evaluations/checkpoints/student_base_numbers_update_2560.json").exists():
                continue
            sh([PY, "-m", "polypythia_sl.pipeline",
                "--config", cfg, "--stage", "students", "--output-dir", out,
                "--teacher-model-path", TEACHER,
                # Match LoRA initialization and minibatch shuffle across cells;
                # the upstream Pythia data order is the only cell difference.
                "--student-seed", 56100 + j])

    # 4. summarize
    lines = ["# Data-order isolation (anchor-free, data-seed family)", "",
             "Teacher FT from data-seed2. (i,o)=data-seed2 init (= teacher base);",
             "(i,o*)=data-seed1 init (same weight init W0, different data order).", "",
             "| dose | (i,o) s1 | (i,o) s2 | (i,o*) s1 | (i,o*) s2 |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    for d in DOSES:
        io = [eff(RUNS / f"ds2_anchor_io_s{j}", d) for j in PAIRS]
        ios = [eff(RUNS / f"ds2_anchor_io_star_s{j}", d) for j in PAIRS]
        lines.append(f"| {d} | {io[0]:+.3f} | {io[1]:+.3f} | {ios[0]:+.3f} | {ios[1]:+.3f} |")
    io_m = sum(eff(RUNS / f"ds2_anchor_io_s{j}", 2560) for j in PAIRS) / 2
    ios_m = sum(eff(RUNS / f"ds2_anchor_io_star_s{j}", 2560) for j in PAIRS) / 2
    lines += ["", f"At dose 2560: (i,o) mean {io_m:+.3f}  vs  (i,o*) mean {ios_m:+.3f}",
              "", "(i,o) is the positive control — it MUST be strongly positive for",
              "the (i,o*) comparison to mean anything."]
    (RUNS / "dataorder_2x2_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("DATAORDER_2x2 DONE", flush=True)


if __name__ == "__main__":
    main()
