# Experiment ledger — PolyPythia subliminal-learning project

Append-only lab notebook. One entry per experiment, newest at the bottom of
each section. This is the single canonical record; per-run `runs/*/` JSON and
`CONFIRMATION_*.md` preregistrations are the primary artifacts it points to.

## Recording protocol (follow for every experiment)

1. **Preregister** anything confirmatory: freeze design + criterion in a
   `CONFIRMATION_*.md` or a script docstring BEFORE the first block runs.
2. **On completion**, append a ledger entry here with the schema below. The
   entry must contain the numeric result, not just a pointer.
3. **Retention gate (hard rule):** never delete a run's model weights until
   its result is in this ledger AND its per-run JSON/report is on disk.
   Weights are regenerable from seeds; findings are not.
4. **Seed registry** lives at the bottom — reserve a range before launching so
   cells never collide.
5. **Steering pre-flight (David's, 2026-07-12) — required before any new
   teacher/base pairing gets a full training-based transmission run:**
   - Run `scripts/steering_probe.py`-style extraction: (teacher_acts -
     base_acts) per layer on the 24 train prompts, apply to the SAME base,
     read held-out wolf margin at each (layer, alpha). Forward passes only,
     no training — minutes, not hours.
   - Record: teacher's behavioral contrast, and the best NLL-safe steering
     cell's wolf delta. Compare against the standing reference bases (see
     table below) to get a **predicted transmission ceiling** for this base.
   - THEN run the expensive training-based transmission experiment.
   - **Divergence check:** if the training result contradicts the steering
     prediction by more than the reference-base scatter (e.g. steering says
     "moderately steerable, expect ~real transfer" but training gives ~0 or
     wrong-signed) — sanity-check the pipeline (pool size, seeds, code path,
     old-vs-new code path with a matched regression pair) BEFORE treating the
     divergence as a scientific finding. This would have caught the
     weight-seed1/data-seed1 confusion faster.
   - If training and steering agree (both weak, or both strong), the result
     is corroborated by two independent measurements and can be trusted with
     normal scrutiny.
   - Log BOTH numbers in the same ledger entry — steering ceiling AND
     training result — so future readers see the cross-check, not just one
     number in isolation.

   **Reference bases (steering, NLL-safe best cell, from saturated wolf teachers):**

   | Base | Behavioral contrast | Best steering cell |
   | --- | ---: | ---: |
   | standard pythia-160m | +17.318 | +5.233 (L8, a=+1) |
   | weight-seed3 | +14.01 | +5.84 (L9, a=+1) |
   | weight-seed1 | +14.42 | +5.55 (L9, a=+1) |
   | weight-seed2 | +13.46 | +4.06 (L8, a=+1) |
   | data-seed2 | +14.47 | +2.94 (L8, a=+1) |
   | data-seed1 | +15.30 | +2.25 (L7, a=+1) |
   | data-seed3 | +14.02 | +2.03 (L7, a=+1) |

Entry schema: `date | name | hypothesis (H#) | prediction | design
(teacher/students/dose/k/n) | result (effect + CI/spread) | verdict (does it
support/refute the hypothesis?) | artifacts | caveats`. Every confirmatory
entry names the standing hypothesis it tests and states the prediction BEFORE
the result, so the verdict is a real up/down vote, not a post-hoc story.

---

## Standing hypotheses register

Each has an ID, a statement, a falsifiable prediction, and a live status.
Update status as evidence lands; never edit the original statement (append
"REVISED:" lines instead).

- **H1 — SL exists at 160M.** A trait fine-tuned into a teacher transmits to a
  same-init student through number strings alone.
  Prediction: paired preference−control margin > 0, replicated.
  **Status: SUPPORTED** (v3, 10/10 blocks, CI [+0.110,+0.136]).

- **H2 — Steerability is necessary for SL** (Blank & Bhatia). If the trait is
  not a linear steering direction, SL fails.
  Prediction: trait steers the base model; SL only where steering works.
  **Status: NECESSARY CONDITION SATISFIED; NECESSITY UNTESTED.** The transmitted
  trait steers strongly (+5.23), so non-steerability is not the blocker here.
  No non-steerable-trait control has tested the necessity claim itself.

- **H3 — Drift, not the channel, is what kills SL at 160M.** Full-parameter
  fine-tuning buries the signal in orthogonal drift; constraining the update
  subspace (LoRA) rescues it.
  Prediction: LoRA students transmit where full-FT students (same data) don't.
  **Status: LEADING EXPLANATION, NOT ISOLATED.** v2 full-FT was +0.048 and v3
  LoRA +0.123 with ~10× lower drift, but teacher class/recipe also changed;
  a same-teacher, same-data full-FT vs LoRA ablation is still required.

- **H4 — Transfer scales with exposure (dose).** More LoRA updates on teacher
  numbers ⇒ stronger preference, up to a channel ceiling.
  Prediction: monotone effect vs log-dose.
  **Status: SUPPORTED** (+0.10→+1.37, ~log-linear, mild flatten >2,560).

- **H5 — SL is trait-specific.** Teacher numbers transmit THAT teacher's trait,
  not a generic direction.
  Prediction: wolf/lion double dissociation.
  **Status: SUPPORTED** (4/4, d_wolf +1.01, d_lion +0.78).

- **H6 — Shared initialization is the requirement for SL** (naive reading of
  Cloud et al.). Different init breaks transfer; data order is irrelevant.
  Prediction: (i*,o) ≈ 0 while (i,o) & (i,o*) transfer.
  **Status: PARTIALLY SUPPORTED; strong form rejected.** The init-only pilot
  sharply attenuated the dose-2560 mean (+1.511 to +0.008), supporting an
  initialization gate. But the clean order-only arm rejects "data order is
  irrelevant": (i,o*) stayed positive (+0.389) while falling 60.8% below its
  matched (i,o) control (+0.991). Shared init is therefore not sufficient for
  full-strength transfer. Both arms are k=2 pilots; the init arm used unmatched
  downstream seeds and no preregistered equivalence margin.

- **H7 — Data-order credit-assignment clamps the trait's coordinates**
  (David's). Early training (driven by data order) fixes which coordinates
  carry which features; SL and steering-vector transport are coordinate-bound,
  so changing data order can break transfer even with shared init — UNLESS the
  clamping timescale is slow relative to init's contribution (open sub-question:
  does order-clamping outpace init? if not, init dominates and H6-like ordering
  holds even though H7's mechanism is real).
  Prediction (strong form): (i,o*) reduced despite shared init; and cross-run
  steering-vector transport degrades more across order-varied than init-varied
  bases.
  Sub-hypotheses to disentangle (need trajectory + alignment probes, see the
  4-layer plan): clamping timescale; init×order interaction; CKA-vs-transport
  (converged-but-permuted, "strong" version) vs genuine divergence ("weak"
  version); whether SL-failure tracks coordinate misalignment.
  **Status: BEHAVIORAL PREDICTION SUPPORTED; MECHANISM UNCONFIRMED.** Holding
  initialization, teacher numbers, local student seeds, and training recipe
  fixed, changing only upstream order reduced the mean effect from +0.795 to
  +0.251 at dose 512 and +0.991 to +0.389 at dose 2560, with both pairs
  agreeing. This establishes attenuation, not coordinate clamping itself;
  transport, trajectory-alignment/CKA, and intervention tests remain open.

---

## Established findings (chronological)

### 2026-07-10 — v1 saturated-teacher pilot + 3-pair sweep
- Q: does SL transmit at 160M? Teacher = full-FT saturated wolf (standard
  pythia-160m step143000); full-parameter students, 256 seqs, 8 epochs.
- Result: local_pilot +0.628; 3-pair sweep 0.628/0.167/0.802, mean 0.532,
  95% t-CI **[-0.28, 1.35]** (crosses 0), one shared teacher.
- Verdict: NOT confirmed — seed-sensitive, underpowered.
- Artifacts: `runs/replication_summary.md`, `runs/local_pilot/report.md`.

### 2026-07-11 — scale/epoch sweep (full-FT)
- Q: does more data / more epochs help full-FT students?
- Result: effects spanned +0.63 to **-1.33** (scale_4k_e8); no stable trend.
  Repetition under full-FT is toxic.
- Verdict: full-FT is the wrong regime. `runs/scale_and_epoch_sweep_results.md`.

### 2026-07-11 — v2 draw-averaged confirmation (context teachers, full-FT)
- Design: `CONFIRMATION_v2_draw_averaged.md`. Prompted context teachers,
  k=8 × 6 blocks, AdamW update-16.
- Result: mean **+0.048**, 95% t-CI [-0.048, +0.144], 4/6 positive. Positive
  control passed 6/6. Bounded null.
- Verdict: NOT confirmed; effect ≤0.14 at this recipe. Variance decomposition:
  Muon@16 vs AdamW@16 same-data block corr **r=0.88** → noise is the data draw.
- Artifacts: `runs/confirm_v2_summary.md`.

### 2026-07-11 — steering-vector probe
- Q (Blank & Bhatia): is the wolf trait a steering vector? If not, SL can't work.
- Result: steerable from all teachers — saturated +5.15 (L8), update-2 +3.26
  (L10), context +3.01 (L11); sign-symmetric, specific, NLL-safe. CPT teachers
  share one direction (cos 0.72-0.88); context vector only ~0.5 aligned.
- Verdict: trait IS steerable → SL failure was downstream (student fine-tuning),
  not teacher representation. `runs/steering_probe.md`.

### 2026-07-11 — standing rules adopted
- Pretraining-matched optimizer (AdamW betas 0.9/0.95, eps 1e-8, wd 0.1); Muon
  retired. Eval prompts doubled 30→60 (originals first). Assays use 10 blocks.

### 2026-07-11 — step-0 rule-compliant teachers
- Retrained saturated + update-2 teachers under the optimizer rule. Contrasts
  reproduce (+17.67 / +2.62); steering direction invariant to optimizer (cos
  ≈1.00); steering probe validated out-of-sample on new 30 prompts.
- Canonical teachers: `runs/teacher_rule_saturated`, `runs/teacher_rule_update2`.

### 2026-07-11 — ⭐ v3 CONFIRMATION (LoRA students) — SL CONFIRMED
- Design: `CONFIRMATION_v3_lora.md`. Saturated rule-compliant teacher, LoRA
  students (r=8, α=16, ~1.18M trainable), pretraining-matched AdamW, dose 16,
  k=8 × 10 blocks, 60-prompt logit-margin readout.
- Result: **10/10 blocks positive, mean +0.123, 95% t-CI [+0.110, +0.136]**,
  79/80 pairs positive, positive control 10/10. P(wolf|10) 4.7%→5.2%.
- Verdict: **CONFIRMED (preregistered).** LoRA is the leading explanation for
  the ~10× drift reduction and rescue versus v2 full-FT, but that comparison
  also changed teacher class/recipe. `runs/confirm_v3_summary.md`.

### 2026-07-11 — dose-response (pilot + 10-epoch)
- Q: does effect scale with LoRA exposure? Reused confirm_v3 pools.
- Result: monotone +0.10 → **+1.37** over doses 16→5120 (~log-linear, +0.15/
  doubling, mild flatten after 2,560). P(wolf|10) 4.6%→15.7% at max (odds ~3.9×,
  Cloud-et-al weight class). No repetition poison under LoRA.
- Verdict: dose CONFIRMED predictive of transfer. Operating point 2,560 (94% of
  max, half compute). `runs/dose_10epoch` probes; curve in status doc.

### 2026-07-11 — trait-specificity crossover
- Design frozen in `scripts/crossover_run.py`. Lion teacher (twin recipe), lion
  pools matched prefixes, wolf-data vs lion-data students, dose 512, k=4.
- Result: **double dissociation 4/4** — d_wolf mean +1.01, d_lion +0.78.
  Relative to update 0, wolf data raised wolf +0.696 and lion data raised lion
  +0.776; lion data also suppressed wolf −0.314, while wolf data's mean effect
  on lion was approximately zero (−0.001, mixed signs). This rejects a generic
  shared FT direction without claiming reciprocal suppression.
- Verdict: SL at 160M is trait-specific. `runs/crossover_summary.md`.

**Combined status: SL at Pythia-160M is confirmed, dose-responsive, trait-specific.**

---

## PolyPythia init × data-order experiments

### 2026-07-12 — FIRST 2×2 pilot (weight-seed1 TEACHER) — INVALID, discarded
- Mistake: changed the TEACHER base to weight-seed1 (not just the student) and
  ran k=1. (i,o) went negative. Diagnosed: k=1 noise + weight-seed1 channel
  differs (number-mean −18.7 vs standard +6..10) + control drift. Deleted.
- Lesson: keep the teacher fixed; vary ONLY student init; keep k averaging.

### 2026-07-12 — (i*,o) init-isolation, standard teacher — CAVEATED
- Tests: **H6** (shared init required). Prediction: (i*,o) ≈ 0 vs (i,o) ≈ +1.5.
- Standard-pythia teacher, reused confirm_v3_b1 pools, LoRA dose 2560, k=2,
  student init = weight-seed1. (i,o) ref = dose_10epoch_b1 (+1.39/+1.64 @2560).
- Result: dose 2560 (i,o) mean **+1.511** vs (i*,o) mean **+0.008** (pairs
  −0.055, +0.071); cells fan apart with dose (identical at dose 16, 15× gap by
  2560). k=2 both cells.
- Verdict: **SUPPORTS AN INITIALIZATION GATE** — changing only initialization
  nearly eliminated endpoint transfer in this pilot; "abolishes" is too strong
  without a prespecified equivalence margin.
- Axis check: the official PolyPythia model metadata defines `weight-seed*` as
  varying only initialization with data order fixed, so standard →
  weight-seed1 is the intended init-only axis. Remaining caveats: downstream
  local seeds were unmatched across cells (53101/53102 vs 54101/54102), k=2,
  and this arm uses a different anchor/teacher from the order-only arm; their
  attenuation magnitudes are not an init-vs-order effect-size comparison.
- Artifacts: `runs/x2x2_istar_summary.md`.

### 2026-07-12 — sanity regression check (David's, "are we sure about standard Pythia?")
- Q: does CURRENT code still reproduce v4, through both the old code path
  (no init_checkpoint) and the new path (explicit init_checkpoint) added for
  the 2x2? Standard pythia, confirm_v3_b1 pools, LoRA dose 512, fresh seed.
- Result: Arm A (old path) +0.0820 @16 / +0.7039 @512. Arm B (new path)
  **bit-for-bit identical**: +0.0820 / +0.7039. Both inside v4 reference band
  (+0.668/+0.961 @512).
- Verdict: harness is clean; the `init_checkpoint` code change is inert when
  pointed at the same base. The pool-size bug (below) is the full explanation
  for the earlier weird results — not a deeper code regression.
- Artifacts: `runs/sanity_A_oldpath`, `runs/sanity_B_newpath`.

### 2026-07-12 — steering pre-flight on data-seed1 (David's cross-check)
- Tests: **H7 sub-question** — is a weak numeric channel (mean-number shift
  ~6.8× smaller than standard; raw JSD ~2.1× smaller and excess-over-noise
  JSD ~3.1–3.3× smaller,
  from the config-bug investigation below) a channel-specific bottleneck, or
  does the trait itself sit weaker in this base generally?
- Same extraction as `steering_probe.py`, run on data-seed1 teacher/base.
- Result: behavioral contrast +15.30 (canonical standard: +17.32, ~88%); best
  steering cell +2.25 @ L7 a=+1 (standard: +5.23 @ L8 a=+1, ~2.3x weaker); sign-
  symmetric (+2.25/-1.71), specific (comparison delta negative), NLL-safe.
- Verdict: trait IS steerable on data-seed1, only ~2.3× attenuated. The
  numeric mean shift is ~6.8× attenuated, while distributional JSD is
  ~2.1× attenuated raw (~3.1–3.3× after subtracting the standard seed-noise
  floor). **Dissociation**: data order
  weakens trait-representability somewhat, but disproportionately decouples
  the trait from the number-generation pathway specifically. Supports H7's
  general spirit (order affects HOW a trait is represented) with a sharper
  claim than "everything is weaker" — the effect is channel-specific.
- Artifacts: `runs/steering_probe_dataseed1.md`. Established the steering
  pre-flight as standing protocol (see above).

### 2026-07-12 — ⚠ CONFIG BUG caught (256-seq pool over-repetition)
- The first weight-seed1 pilot AND the first data-order run both used pools of
  only **256 sequences** (inherited `size_per_condition: 256` from the
  teacher_rule_saturated ← local_pilot config lineage), not v4's 8,192. At dose
  2560 that is 160 epochs of repetition (vs 5 on an 8,192 pool) → repetition
  poison → (i,o) positive control collapsed to ~0/negative on BOTH variant
  bases. This — NOT base-dependence or a "weak channel" — explains the earlier
  weight-seed1 (i,o) failure; that diagnosis is RETRACTED.
- Caught by David asking "same base, same recipe — why different result?".
  Root cause: any teacher config derived from teacher_rule_saturated carries
  size 256. FIX: set generation `size_per_condition: 8192` for all 2×2 configs;
  guard = check pool line count and number-mean delta before trusting a result.
- Number-mean delta is a fast channel-strength check (standard +20.9 over 8,192;
  the 256-pool estimate +7.8 was just noise, not a real difference).

### 2026-07-12 — base-screening campaign (COMPLETE) — transfer propensity per base
- David's directive: for every cached 160M base (standard, data-seed1-3,
  weight-seed1-3), induce the canonical saturated wolf teacher (identical
  recipe/seed), extract steering vector, apply to own base → record
  behavioral contrast + best NLL-safe steering delta = **transfer propensity**.
  Propensity is now recorded for every SL replication attempt (protocol rule 5).
- Purpose: (a) standing propensity reference table; (b) pick the STRONGEST
  data-seed base to anchor the (i,o)/(i,o*) cells so the data-order contrast
  is maximally legible. Note: the (i,o*) pair must come from within the
  data-seed family regardless of the overall winner (only within-family pairs
  guarantee shared init); the screening chooses WHICH data-seedN anchors it.
- Aside recorded: v4 verified to inherit v3 exactly except max_updates /
  probe_updates / schedule_total_updates (the declared dose regime; note the
  schedule shape at matched update-16 differs slightly, 128-truncated vs full).
- Future extension (David's, do not lose): per-base animal-token scan — test
  each base across all single-token animals to find which animal "rules the
  steganography" for each model.
- `scripts/base_screening.py` → `runs/base_screening/<base>.json` +
  `runs/base_screening_summary.md`. Teachers deleted after result JSON (gate).
- Result (best NLL-safe steering delta): weight-seed3 **+5.84**, weight-seed1
  **+5.55**, standard **+5.23**, weight-seed2 **+4.06**, data-seed2 **+2.94**,
  data-seed1 **+2.25**, data-seed3 **+2.03**. All sign mirrors were negative.
- Standard ranks third overall. The strongest eligible data-order anchor,
  data-seed2, is **56.2% as strong as standard** by steering delta (+2.94 vs
  +5.233; 43.8% weaker), despite retaining 83.6% of standard's teacher
  behavioral contrast (+14.47 vs +17.318). This separates preference induction strength
  from steering-channel transport strength.
- Selection: **data-seed2** anchors teacher/(i,o); data-seed1 is the (i,o*)
  sibling because it is the next-strongest data-seed recipient. Overall winner
  weight-seed3 remains reserved for init-varied cells, not data-order isolation.
- Grid retention audit: all seven bases have complete 12-layer × 7-alpha grids
  (588 cells; no missing/duplicate cells).
- Canonical standard rescreen **COMPLETED 2026-07-13** on the current 60 prompts
  with retained `runs/teacher_rule_saturated`: behavioral contrast
  **+17.318195**; best NLL-safe steering delta **+5.233191** at L8, alpha +1;
  NLL ratio **1.015187**; sign mirror **−2.179545**. The artifact contains all
  84 unique layer/alpha cells, declares the 60-prompt canonical recipe, and the
  rebuilt ranking is unchanged (standard remains third).

### 2026-07-12 — steering strength → SL strength prediction campaign (PLANNED)
- David's question: do bases with larger NLL-safe steering deltas exhibit
  stronger same-base subliminal transfer? Especially, do weight-seed3/1's
  above-standard steering scores predict above-standard SL?
- Existing x-axis is complete (full grids for all seven bases). Existing y-axis
  is not: only standard has a high-quality same-base SL estimate; prior
  weight-seed1 values used standard-teacher numbers, and its earlier own-teacher
  run was invalidated by the 256-row repetition bug.
- Required matched campaign: identical canonical wolf-teacher recipe, 8,192-row
  pools, local generation seeds, LoRA/student seeds, optimizer, schedule, and
  doses {16,512,2560} on every base; k=2 per base minimum. Record replicate
  effects, means, number-channel mean delta, and validity metadata incrementally.
- Plot steering delta vs same-base SL as aligned dose facets, with every base
  labeled and replicate spread visible. Report Pearson + Spearman descriptively;
  n=7 is exploratory, not a high-powered model-selection result.

### 2026-07-12 — data-order isolation, ANCHOR-FREE (re-anchored after screening)
- Tests: **H6 vs H7**. Prediction (H6): (i,o*) transfers like (i,o), data order
  irrelevant. Prediction (H7 strong): (i,o*) reduced despite shared init.
- Fix for the anchor problem: work within the data-seed family. Screening chose
  data-seed2; (i,o)=data-seed2 (positive control, must fire), (i,o*)=data-seed1
  (same weight init W0, different data order). k=2, dose 2560. Fresh
  `ds2_anchor_*` run paths preserve the superseded partial data-seed1 run.
- Local student seeds are paired across cells (56101/56102 in both), holding
  LoRA initialization and minibatch shuffle fixed; the only cell difference is
  the upstream PolyPythia pretraining data order.
- Pool pre-flight (8,192 rows each): preference number mean 220.975 vs base
  185.812, delta **+35.163**. Full unigram-distribution JSD (numbers 0..999,
  add-1 smoothing, matching the earlier diagnostic) is **0.011695**; versus the
  established standard-base seed noise floor 0.00213, excess JSD is 0.009565.
  This is slightly stronger than standard b1/b2 JSD (0.00972/0.01006), despite
  data-seed2's weaker NLL-safe steering delta (+2.94 vs canonical standard
  +5.23).
  Preregistered interpretation: steering transportability and numeric-channel
  imprint strength are separable; student transfer remains the arbiter.
- `(i,o)` positive control **CONFIRMED 2/2**. Pair effects at doses
  16/512/2560: s1 −0.006/+0.803/+1.052; s2 +0.172/+0.788/+0.931; cell means
  **+0.083/+0.795/+0.991**. Transfer is strong and dose-amplified on
  data-seed2. The dose-512 mean is close to the established standard-base grand
  mean (+0.776), agreeing with the strong JSD rather than the weaker steering
  ceiling. `(i,o*)` launched automatically after confirmation.
- Pair-1 held-out probability view at dose 2560 (10-animal candidate set,
  60 prompts): wolf mean **18.51%** after preference numbers vs **7.51%** after
  base-control numbers and 9.27% at update 0. Wolf exceeded the uniform 10%
  candidate baseline on 45/60 prompts (control: 15/60), and preference exceeded
  its paired control on 59/60 prompts. This is a restricted-candidate next-token
  readout, not an unconditional greedy-generation frequency.
- Apples-to-apples standard comparison at dose 2560: four standard pairs mean
  5.61% initial → 15.22% preference and 4.86% matched control (2.71× initial,
  3.16× control; individual control ratios 2.24×–3.94×; mean margin effect
  +1.287). Data-seed2 pair 1 is 9.27% → 18.51% vs 7.51% control (2.00× initial,
  2.46× control; margin effect +1.052). Thus its multiplicative rate is ~22%
  below the standard mean but inside standard pair scatter, while its absolute
  wolf probability is higher. The remembered ~3–4× headline used standard's
  dose-5120 endpoint (4.6% → 15.7%, ~3.4× rate/~3.9× odds), not dose 2560.
- `(i,o*)` **COMPLETED, positive 2/2 but attenuated 2/2**. Pair effects at
  doses 16/512/2560: s1 +0.100/+0.234/+0.399; s2
  +0.053/+0.267/+0.378; cell means **+0.076/+0.251/+0.389**. Thus changed
  pretraining order does not abolish transfer when ancestral initialization is
  shared.
- The matched `(i,o)` vs `(i,o*)` means are +0.795 vs +0.251 at dose 512 and
  +0.991 vs +0.389 at dose 2560. Cross-order transfer retains **31.5%** and
  **39.2%** of the same-order effect respectively (68.5%/60.8% attenuation).
  The endpoint attenuation replicated by local seed: 62.1% in s1 and 59.4% in
  s2.
- **Interpretation:** H7's preregistered behavioral prediction is supported:
  pretraining data order substantially modulates transfer strength even when
  step-0 weights are shared. This experiment does not by itself establish the
  proposed coordinate-clamping mechanism. H6 survives only in its weaker
  form—shared initialization provides a transferable substrate; its stronger
  "data order irrelevant" prediction is not supported. This is one teacher and
  one shared pool with two paired local seeds, so the result is a strong
  within-design replication, not a population-level estimate across teachers
  or generated datasets. Prompt-level intervals in the checkpoint reports
  describe prompt variation and must not be read as training-replicate
  confidence intervals.
- `scripts/dataorder_2x2.py`, `runs/dataorder_2x2_summary.md`.

---

## Seed registry

| Range | Use |
| --- | --- |
| 42xxx | confirm_v3 students |
| 51xxx | dose pilot |
| 52xxx | v4-proper (reserved, unused) |
| 53xxx | dose 10-epoch |
| 54xxx | 2×2 (i*,o) standard-teacher |
| 55xxx | 2×2 (i,o*) standard-teacher (superseded by anchor-free 56/57) |
| 56xxx | re-anchored data-order pairs, matched across (i,o)/(i,o*) |
| 57xxx | superseded partial data-seed1-anchor (i,o*) range |
| 61xxx | crossover |
| 70xxx/71xxx | invalid weight-seed1-teacher pilot (discarded) |
