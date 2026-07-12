# Execution plan: establish student preference inheritance with confidence

Goal: determine with defensible statistics whether students trained on
preference-teacher numbers inherit the wolf preference, vs. control students
trained on base-model numbers. Current evidence: 3 pairs, mean effect 0.53,
95% CI [-0.28, 1.35] — crosses zero, all pairs share one teacher. Not sufficient.

This plan requires **no code changes**. Everything is done with config-file
copies and existing CLI flags. Do not edit anything under `src/` or `tests/`.
Do not modify or delete anything under `runs/local_pilot`, `runs/replicate_2`,
or `runs/replicate_3`.

The key metric everywhere is `student_transmission_delta_vs_control_student`
in each run's `report.json` (equivalently the paired target logit-margin
effect). Positive = inheritance.

---

## Phase A — data-scaling probe (reuses the existing teacher)

Question: does the transmission effect grow when students see more distinct
teacher-sampled data (with fewer epochs)? If it does not grow at all, stop
after Phase A and report that — it is important negative evidence.

### A1. Create three config files

Copy `configs/local_pilot.yaml` to these three files, then apply ONLY the
listed edits (leave every other key identical):

**`configs/scale_1k.yaml`**
- `run.output_dir`: `runs/scale_1k`
- `number_data.size_per_condition`: `1024`
- `student_training.epochs`: `3`

**`configs/scale_4k.yaml`**
- `run.output_dir`: `runs/scale_4k`
- `number_data.size_per_condition`: `4096`
- `student_training.epochs`: `3`

**`configs/scale_8k.yaml`**
- `run.output_dir`: `runs/scale_8k`
- `number_data.size_per_condition`: `8192`
- `student_training.epochs`: `3`

### A2. Run them (sequentially, not in parallel — they share the GPU/MPS)

Each run reuses the already-trained teacher so no teacher training happens:

```
python -m polypythia_sl.pipeline --config configs/scale_1k.yaml \
  --teacher-model-path runs/local_pilot/models/preference_teacher \
  --prompt-seed 12203 --sampling-seed 12201 --student-seed 12107

python -m polypythia_sl.pipeline --config configs/scale_4k.yaml \
  --teacher-model-path runs/local_pilot/models/preference_teacher \
  --prompt-seed 13203 --sampling-seed 13201 --student-seed 13107

python -m polypythia_sl.pipeline --config configs/scale_8k.yaml \
  --teacher-model-path runs/local_pilot/models/preference_teacher \
  --prompt-seed 14203 --sampling-seed 14201 --student-seed 14107
```

If a run is interrupted, re-running the same command resumes (completed
stages are skipped unless `--force` is passed — never pass `--force`).

### A3. Per-run checks (do these after each run before starting the next)

1. `runs/<name>/data/numbers_preference_teacher.stats.json` →
   `acceptance_rate` must be ≥ 0.5. If it is lower, record it and continue,
   but flag it prominently in the final report.
2. `runs/<name>/report.json` exists and contains
   `student_transmission_delta_vs_control_student`.
3. Disk: each run writes ~2 GB of model weights. After confirming step 2,
   free space by deleting ONLY the two student model weight files:
   `runs/<name>/models/student_preference_numbers/model.safetensors` and
   `runs/<name>/models/student_base_numbers/model.safetensors`.
   Keep all JSON files, all data files, and all evaluation files.
   Delete nothing else.

### A4. Phase A read-out

Tabulate `student_transmission_delta_vs_control_student` for:
- `runs/local_pilot` (256 seqs, 8 epochs — already done, use as-is)
- `runs/scale_1k`, `runs/scale_4k`, `runs/scale_8k`

Decision rule:
- If the 8k effect ≥ 2× the local_pilot effect (i.e. ≥ ~1.2): use
  **8192 / 3 epochs** in Phase B.
- If the effect grows but less than that: use the size with the largest
  effect in Phase B.
- If the 4k and 8k effects are both ≤ the local_pilot effect (no growth):
  **STOP. Do not run Phase B.** Write the table into
  `runs/inheritance_report.md` with the sentence "Transmission does not
  scale with student data under this decoder; escalating n is not justified."

---

## Phase B — confirmatory runs with independent teachers

Only run this if Phase A's decision rule says to. Let `S` = chosen
`size_per_condition` and 3 epochs, from Phase A.

Each of the 6 runs below trains its **own teacher** (different teacher seed
and different preference-data seed), so every pair is fully independent.
This is the statistical unit that was missing.

### B1. Create six config files

For i = 1..6, copy `configs/local_pilot.yaml` to `configs/confirm_i.yaml`
with ONLY these edits:

- `run.output_dir`: `runs/confirm_i`
- `preference_data.seed`: `20000 + i`   (i.e. 20001, 20002, ... 20006)
- `teacher_training.seed`: `21000 + i`
- `number_data.size_per_condition`: `S`
- `number_data.prompt_seed`: `22000 + i`
- `number_data.sampling_seed`: `23000 + i`
- `student_training.epochs`: `3`
- `student_training.seed`: `24000 + i`

### B2. Run sequentially

```
python -m polypythia_sl.pipeline --config configs/confirm_1.yaml
# ... then confirm_2 ... through confirm_6
```

Do NOT pass `--teacher-model-path` — each run must train its own teacher.

### B3. Per-run checks

Same as A3, plus:
- `runs/confirm_i/report.json` → `teacher_induction_delta_vs_base` must be
  ≥ 5.0 (the teacher fine-tune took). If it is below 5.0 for any run, do
  not delete anything from that run, note it, and exclude it from the
  aggregate (record the exclusion and the value).
- After checks pass, also delete
  `runs/confirm_i/models/preference_teacher/model.safetensors`
  (in addition to the two student weight files) — Phase B teachers are not
  reused by anything.

### B4. Aggregate

```
python -m polypythia_sl.aggregate runs/confirm_1 runs/confirm_2 \
  runs/confirm_3 runs/confirm_4 runs/confirm_5 runs/confirm_6 \
  --output runs/confirmatory_summary.md
```

(Only include runs that passed B3.)

### B5. Pre-registered success criterion (decided before the runs; report
the outcome either way, do not move the goalposts)

- **Inheritance established** if the 95% t-interval across the included
  independent pairs (from `runs/confirmatory_summary.md`) excludes zero AND
  at least 5 of 6 pairs have a positive
  `student_transmission_delta_vs_control_student`.
- **Inconclusive** if the CI crosses zero but ≥ 5/6 pairs are positive.
  Report as inconclusive; do not launch additional runs without sign-off.
- **No inheritance detected** otherwise.

---

## Final deliverable

Write `runs/inheritance_report.md` containing:

1. The Phase A table (size, epochs, transmission delta, acceptance rate).
2. Which decision-rule branch was taken and why.
3. The Phase B per-run table (teacher induction delta, transmission delta,
   any exclusions with reasons).
4. The aggregate CI and the verdict per B5, stated in one sentence.
5. Any anomalies: acceptance_rate < 0.5, runs that needed resuming, disk
   issues, exclusions.

Rules of conduct:
- Never pass `--force`.
- Never edit `src/`, `tests/`, or existing `runs/local_pilot|replicate_2|replicate_3`.
- Run pipelines one at a time.
- If any command errors twice in a row, stop and leave a note in
  `runs/inheritance_report.md` under "Anomalies" rather than improvising.
- Do not commit or push anything.
