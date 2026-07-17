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
  REVISED (2026-07-13, transport probes): within the exact-shared-init
  data-seed pair, the ds2 trait direction transports RAW across data orders
  (62.4% at preregistered L8/+1). The fitted global Procrustes map reduces the
  main L8 effect to 37.3% and hurts most cells, so that particular alignment is
  unnecessary; it does not prove that every coordinate or representation is
  natively shared. Cross-family raw transport is heterogeneous: the same ds2
  vector retains 49.1% in weight-seed1 but only 0.4% in weight-seed3. Thus a
  universal shared direction across pretraining lineages is rejected, while
  the same-init result still supports David's narrower account: order-driven
  weight/circuit changes can attenuate coupling around a partly preserved
  residual direction without relocating it. Leading mechanism candidates are
  receiver-side numeric-trait write coupling and receiver-specific gain;
  coordinate mismatch remains live for the weight-seed3 lineage.
  REVISED (2026-07-13, fixed-L8 student-state intervention): all four
  dose-512 preference/control pairs were deterministically reconstructed from
  the retained pools and reproduced their archived readouts within 5.1e-7
  logits. The mean student activation difference aligned with the fixed ds2
  teacher wolf direction in only 1/4 pairs (cosines +0.309, -0.069, -0.101,
  -0.027), and the teacher-parallel patch was wolf-increasing only in that one
  pair. Thus the simple mechanism "numeric training recovers less of one fixed
  L8 teacher direction" is not supported. However, reciprocal full-sequence
  L8 state swaps increased wolf margin in both downstream suffixes in all 4/4
  pairs (state-main effects +0.109, +0.113, +0.109, +0.061), and secondary
  all-token additions of each pair's full mean student difference were
  sign-correct in 4/4. Numeric training therefore writes a causal,
  sequence-distributed wolf-relevant L8 footprint, but not generally the
  teacher's single mean last-token direction. This refines the leading account
  toward optimizer/Jacobian-mediated recovery in student-specific distributed
  coordinates; it does not yet identify the upstream credit-assignment
  mechanism.
  REVISED (2026-07-13, update-0 reverse-mode Jacobian assay): the exact
  historical LoRA tangent does **not** supply a seed-stable local predictor.
  With the same guarded ds2 pools and paired LoRA/minibatch seeds, raw
  `-<grad wolf margin, grad(Lpref-Lctrl)>` was +0.3455 for ds2/56101 but
  -0.0609 for ds2/56102, even though both known update-512 effects are strongly
  positive (+0.8031/+0.7877). The second seed was negative in both 4,096-row
  pool halves and in both 30-prompt halves (one near zero), and ranked below
  ds1. The frozen retrospective gate therefore failed and prohibited the
  prospective receiver campaign. The exact first-AdamW-update secondary was
  also negative in both ds2 seeds. This rejects a pre-existing, update-0
  LoRA-local route as a necessary mechanism; it leaves open a multistep route
  constructed after LoRA-B moves, LoRA-A gains gradients, and Adam state and
  teacher-forced histories evolve.
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

### 2026-07-13 — steering-vector TRANSPORT probe ds2→ds1 (H7 mechanism test)
- Tests: **H7 mechanism** (coordinate clamping). Predictions registered in
  `scripts/transport_probe.py` docstring before running; behavioral reference
  39.2% retention.
- Result (matched cells; best-cell ratios were NLL-gate selection artifacts —
  see `runs/transport_probe.md` reanalysis):
  - **Raw transport retains ~47-62%** (L8 matched: +1.77 vs +2.83 = 62.4%;
    sign-symmetric), vs behavioral 39.2%.
  - The fitted Procrustes map is worse at the main high-effect cells (37.3% at
    L8; residuals 0.16-0.45) and raw is stronger in 70/84 cells, but not
    literally everywhere: aligned wins 14/84 cells. The result says this
    direction does not need that fitted global map; it does not establish
    globally shared coordinates.
  - The selected raw ds1 best (+4.32 at L10) exceeds ds1's own-vector selected
    best (+2.25), but the matched L8 raw effect (+1.77) does not. Treat the
    former as selection-sensitive, not evidence that the foreign vector is
    intrinsically better.
- Verdict: the simple coordinate-mismatch account is **disfavored for this
  direction within this exact-shared-init pair**. Transport attenuation
  (~38% loss at L8/+1) is in the same ballpark as behavioral attenuation
  (61%) but does not match crisply enough to identify the mechanism; the
  correspondence is loose and cell-dependent.
- New leading mechanism candidate: **order-specific numeric-trait
  entanglement on the receiver side** — the trait direction is shared, but
  how strongly the ds2-native number distribution's gradients couple INTO
  that direction depends on the receiver's data order (consistent with the
  earlier data-seed1 dissociation: channel ~7x weaker vs trait ~2x weaker).
- Proposed next probes (cheap): (a) cross-family raw transport ds2→
  weight-seed1/3; (b) receiver-side channel test — NLL of ds2-teacher numbers
  under ds1 vs ds2 bases, correlating "foreignness" of the numbers with
  attenuation.
- Artifacts: `runs/transport_probe.{json,md}`.

---

### 2026-07-13 — fixed-cell RAW cross-family transport ds2→weight-seed1/3
- Frozen design: one ds2-teacher minus ds2-base L8 direction, applied without
  rescaling, layer search, alpha search, or alignment at alpha -1/0/+1. ds2
  self is the reference; ds1 is the exact-shared-init/different-order control;
  weight-seed1 and weight-seed3 are foreign-lineage primary receivers. All
  readouts use the fixed 60 held-out prompts and the +1 NLL<1.2 quality gate.
- Provenance audit before launch: deterministic tensor hashes of the official
  step-0 checkpoints are identical for data-seed1 and data-seed2
  (`f0236470...`) but differ for standard Pythia (`5ed85f31...`) and
  weight-seed1 (`d1c10248...`). The planned standard-centered "hub" assay was
  discarded before any forward pass; standard is not assumed to share the
  data-seed initialization.

| receiver | delta -1 | delta +1 | +1 NLL ratio | +1 retention | prompt-bootstrap 95% |
| --- | ---: | ---: | ---: | ---: | ---: |
| ds2 self | -2.0070 | +2.8295 | 1.0706 | 100.0% | 100.0-100.0% |
| ds1 order control | -1.6053 | +1.7649 | 1.0739 | 62.4% | 57.3-67.9% |
| weight-seed1 | -1.5733 | +1.3882 | 1.0328 | 49.1% | 44.3-54.6% |
| weight-seed3 | -0.6640 | +0.0108 | 1.0707 | 0.4% | -4.7-5.1% |

- Both historical ds2/ds1 fixed cells reproduce exactly; all +1 cells pass
  the quality gate. Weight-seed1 therefore accepts substantial raw transport.
  Weight-seed3's positive effect is indistinguishable from zero on this prompt
  set, despite its own native L8/+1 vector scoring +3.138 in the independent
  base screen. This is foreign-direction-specific, not generic inability to
  steer weight-seed3 at L8. Its negative intervention still reduces the wolf
  margin, giving an asymmetric 14.0% centered-gain retention.
- Verdict: **raw cross-family transport is lineage-heterogeneous**. One
  different-init receiver substantially expresses the ds2 direction and one
  does not, rejecting both "raw coordinates are universal" and "different
  initialization always destroys raw transport." Because ds2→weight-seed
  changes both initialization and upstream order, this does not causally
  isolate initialization. Together with prior behavioral SL it is suggestive,
  not direct proof, of receiver-side write-coupling/gain differences.
- Next discriminators: preregister a small-alpha response curve for
  weight-seed3 (tests nonlinearity/saturation), then fit and validate alignment
  only if the raw local response remains weak. A true init-only transport arm
  requires a native weight-seed teacher and a same-order weight-seed sibling.
- Artifacts: `scripts/cross_family_transport.py`,
  `runs/cross_family_transport.{json,md}`.

---

### 2026-07-13 — fixed-L8 student trait-write intervention at dose 512
- Tests: **H7 receiver-side write-coupling sub-hypothesis** and the stronger
  claim that numeric credit assignment recovers a projection of the teacher's
  fixed wolf direction. Frozen design: reconstruct the matched `(i,o)` and
  `(i,o*)` preference/control students at update 512; extract
  `d = mean(h_preference - h_control)` at the final prompt token of L8 on the
  fixed 24 extraction prompts; decompose `d` relative to the fixed ds2
  teacher-minus-base wolf direction `v`; intervene on the fixed 60 held-out
  prompts. Primary patches affect the final token only. Reciprocal exact-state
  swaps test state-source × downstream-suffix mediation; all-token additions
  and full-sequence swaps are secondary distributed-state bridges.
- The original runs retained evaluations but not weights, so all eight
  students were replayed from the byte-identical 8,192-row pools with their
  exact init checkpoints, LoRA/student seeds, optimizer recipe, and
  2,560-update LR horizon, stopping at update 512. Adapter-only saves avoid
  replacing historical directories or writing eight merged 649 MB models.

| pair | archived gap | replayed gap | difference | gate |
| --- | ---: | ---: | ---: | :---: |
| `(i,o)` s1 | +0.803140 | +0.803139 | -0.0000005 | pass |
| `(i,o)` s2 | +0.787731 | +0.787731 | +0.0000005 | pass |
| `(i,o*)` s1 | +0.234386 | +0.234386 | +0.0000000 | pass |
| `(i,o*)` s2 | +0.267220 | +0.267220 | +0.0000003 | pass |

The frozen absolute tolerance was 5e-4; the largest reload discrepancy was
5.1e-7 logits. Callback readouts reproduced the archived margins exactly to
displayed precision. The retained update-0 preference/control gaps are exactly
zero in all four pairs. The teacher vector also reproduced its prior tensor
SHA256 (`7ac7d552...64f587`), norm 10.997561, and mean prompt-difference norm
12.344284.

| pair | cos(d,v) | squared parallel fraction | control +d, last | signed effect of -d in preference | all-token centered d (ctrl/pref) | exact full-state forward/reverse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `(i,o)` s1 | +0.309 | 9.54% | +0.0278 | +0.0336 | +0.0577 / +0.0603 | +0.1067 / +0.1110 |
| `(i,o)` s2 | -0.069 | 0.48% | -0.0288 | -0.0169 | +0.0472 / +0.0468 | +0.1098 / +0.1169 |
| `(i,o*)` s1 | -0.101 | 1.02% | +0.0460 | +0.0488 | +0.0984 / +0.1007 | +0.0996 / +0.1192 |
| `(i,o*)` s2 | -0.027 | 0.07% | +0.0122 | +0.0227 | +0.0225 / +0.0361 | +0.0543 / +0.0672 |

- **Fixed-direction criterion FAILED.** Positive projection and correct-signed
  teacher-parallel patches occurred only in `(i,o)` s1. In the other three
  pairs, the projection and parallel-patch effects were negative. The
  norm-matched orthogonal centered effects were small (-0.0034 to +0.0062),
  but that specificity control cannot rescue the absent replicated alignment.
  The full mean last-token difference was sufficient/removable in 3/4 pairs
  and wrong-signed in `(i,o)` s2; it recovered only a few percent of the
  natural same-order gap.
- **Distributed L8 state mediation replicated 4/4.** Replacing the complete
  prompt-specific L8 sequence state crossed between preference and control
  students raised/removed wolf margin in both downstream suffixes for every
  pair. The suffix-averaged state effects were +0.109, +0.113, +0.109, and
  +0.061; state×suffix differences were small (-0.004 to -0.020). Aggregated
  state mediation was +0.111 same-order (14.0% of the +0.795 natural gap) and
  +0.085 changed-order (33.9% of the +0.251 gap). Secondary all-token additions
  of each pair's full mean `d` were likewise sign-correct in both recipients
  in 4/4 pairs. By contrast, final-token-only exact swaps were inconsistent.
- Quality checks are clean: mean final-token full-vocabulary KL was at most
  0.0206 in any exact-swap cell; exact-swap prompt-NLL ratios ranged
  0.9906-1.0086, and all-token additive NLL ratios stayed within 1.0084. Tiny
  negative per-prompt KL values (minimum -1.24e-6) are floating-point
  roundoff; all mean KLs are positive.
- **Verdict: MIXED; the preregistered single-direction mechanism is not
  supported.** Numeric training does create a causal wolf-relevant activation
  footprint by L8, but it is sequence-distributed and not generally the
  teacher's mean last-token steering direction. The core credit-assignment
  account remains viable in a broader form: optimizer-mediated fitting may
  recover a functionally wolf-equivalent projection in student-specific,
  distributed coordinates. This assay does not directly test the upstream
  number-distribution Jacobian/gradient alignment that would establish that
  account, and the moderate L8-state attenuation does not explain the much
  larger behavioral data-order attenuation by itself.
- Scope: one teacher, one paired number-pool draw, two local seeds per lineage.
  Prompt intervals describe the fixed 60 prompts, not independent student
  training replicates. No layer, update, component normalization, or patch
  placement was selected after seeing the result.
- Artifacts: `scripts/student_trait_write_probe.py`,
  `runs/student_trait_write_probe_u0512.{json,md}`; replay adapters and frozen
  manifests live under ignored `runs/student_trait_write_probe_u0512/`.

---

### 2026-07-13 — update-0 numeric-sequence Jacobian/NTK alignment
- Frozen protocol: `configs/numeric_channel_jacobian_v1.json`. At each exact
  historical update-0 LoRA initialization, compute
  `S = -<grad held-out wolf margin, grad(L_ds2-pref - L_ds2-control)>` over the
  byte-guarded 8,192-row pools. Positive `S` is the infinitesimal Euclidean-SGD
  prediction that preferentially fitting the wolf-teacher sequences increases
  wolf preference. The primary uses all 60 held-out prompts; original/new
  30-prompt halves and shuffled 4,096-row pool halves are stability checks.
  Student seeds 56101/56102, LoRA initialization tensors, DataLoader orders,
  receiver commits/weights, pool hashes, tokenizer semantics, and implementation
  hashes were frozen. No explicit Jacobian was materialized; reverse-mode
  products were reduced in CPU float64.

| receiver | seed | raw `S` | cosine | first-Adam prediction | known u512 effect |
| --- | ---: | ---: | ---: | ---: | ---: |
| ds2 `(i,o)` | 56101 | +0.345494 | +0.032414 | -0.000067 | +0.803140 |
| ds2 `(i,o)` | 56102 | -0.060930 | -0.004653 | -0.000400 | +0.787731 |
| ds1 `(i,o*)` | 56101 | +0.048526 | +0.004550 | +0.001616 | +0.234386 |
| ds1 `(i,o*)` | 56102 | +0.008281 | +0.001201 | +0.000064 | +0.267220 |

- **Frozen retrospective gate FAILED.** Seed 56101 passed positivity and all
  three ds2>ds1 comparisons; seed 56102 failed all four checks. Its ds2 score
  remained negative in both pool halves (-0.0559/-0.0660), while its original
  and expanded prompt halves were -0.0096/-0.1122. The across-seed raw means
  happen to preserve the behavioral order (ds2 +0.1423 vs ds1 +0.0284), but
  that post hoc average cannot rescue the preregistered seed-replication gate.
  No prospective score, prediction, or student training was launched.
- The exact t=1 clipped-AdamW secondary also fails as an endpoint explanation:
  it is negative for both strongly transferring ds2 runs, while positive for
  both attenuated ds1 runs. Thus neither the Euclidean population tangent nor
  the actual first minibatch update is a necessary positive route.
- **Interpretation:** successful SL can emerge despite a wrong-signed initial
  LoRA-local derivative. This rejects the strong static claim that credit
  assignment merely follows a pre-existing numeric-to-wolf tangent. It does
  not reject multistep credit assignment: at PEFT initialization LoRA-B is
  zero and only B has gradients; after the first step A becomes trainable in
  effect, Adam state accumulates, the two trajectories diverge, and a useful
  distributed route can be constructed. That dynamic account is now the
  leading version of the hypothesis, not a confirmed mechanism.
- Scope: the actual historical loss supervises 10 number and 9 comma tokens,
  with different later-token histories across pools. This is sequence-loss
  gradient alignment (`Jpref^T rpref - Jctrl^T rctrl`), not the separate
  explicit sender probability-fingerprint assay. Ten-animal gradient ranks
  were diagnostic only (wolf ranks 3/4/4/6) and do not supersede the existing
  behavioral wolf/lion double dissociation.
- Next discriminators: measure the score along the actual early trajectory
  (after B moves and A gradients activate), and separately run the explicit
  recipient-specific number-fingerprint assay with match/remove interventions.
- Artifacts: `scripts/numeric_channel_jacobian.py`,
  `configs/numeric_channel_jacobian_v1.json`, and ignored
  `runs/numeric_channel_jacobian_v1.{json,md}` plus guarded score records.

### 2026-07-13 — soft numeric-fingerprint compatibility and prospective endpoints
- Frozen sender assay: on the exact 8,192 paired ds2 prefixes, compute the
  temperature-1 preference-teacher and base distributions over all 655 allowed
  numeric token IDs. The sender shift has mean TV **0.144206** and mean JS
  **0.018233 nats**. For each receiver, extract its own native teacher-minus-base
  wolf vector and measure the central local change in full-vocabulary numeric
  log probabilities under alpha +/-0.25. The raw cross-loss score is
  `C = mean_x sum_y (q_ds2-wolf-q_ds2-base) * d log p_receiver(y|x)/d alpha`;
  the locked cross-receiver score `K=C/G` divides by the same vector's local
  held-out wolf-margin slope. Positive `C` is a genuine local loss incentive,
  not a sampled-number correlation or proof that LoRA can write that direction.
- The retrospective ds2/ds1 gate passed. The prospective rank was then locked
  before any endpoint artifact: weight-seed3 (**K .032062**) > weight-seed1
  (**.031450**) > standard (**.021104**). The three flattened sender/response
  cosines were small (.0664/.0486/.0455), and the score was mostly a marginal
  token-frequency effect, especially for weight-seed3 (about 90% marginal).
  That is still a valid cross-entropy-reducing fingerprint; it is not evidence
  of visually identical prompt-conditional response fields.

| receiver | seed 56101 | seed 56102 | mean u512 preference-control effect |
| --- | ---: | ---: | ---: |
| standard | +0.588329 | +0.354485 | **+0.471407** |
| weight-seed1 | +0.156014 | +0.423656 | **+0.289835** |
| weight-seed3 | +0.076612 | +0.192837 | **+0.134724** |

- **Frozen primary FAILED:** weight-seed3 minus standard was **-0.336683**.
  The observed order was exactly reversed, standard > weight-seed1 >
  weight-seed3 (descriptive Spearman -1 at n=3). Static `K` therefore does not
  predict update-512 SL magnitude across these receivers. Do not rescue it by
  selecting a different normalization or checkpoint after the fact.
- **The sign result is nevertheless real:** all 6/6 paired seed endpoints were
  positive. In weight-seed3 the mean was +0.134724 logits and +1.170 percentage
  points wolf probability. This is a clean prospective foreign-lineage signal,
  despite raw ds2-vector transport into weight-seed3 being only 0.4%. With two
  local seeds it is a paired replication, not a population estimate. It is not
  the first cross-init positive u512 hint: the caveated standard->weight-seed1
  `(i*,o)` pilot averaged +0.115 at u512 before falling to +0.008 at u2560.
  Also, tensor provenance shows standard does not share the data-seed
  initialization, so none of the three prospective receivers is a same-init
  ds2 control.
- Carrier-fit loss does not explain the reversed rank. Mean preference training
  NLL across both seeds was 2.76048 standard, 2.75481 weight-seed1, and 2.75136
  weight-seed3; control NLLs were 2.77333, 2.76678, and 2.76422. Weight-seed3
  fit the observed numbers slightly better, not worse. Global preference
  gradient norms and clipping rates did rank standard > weight-seed1 >
  weight-seed3, but are descriptive under coordinatewise AdamW.
- **Revised mechanism:** static activation-space compatibility measures a
  loss-reducing read route. Behavioral strength additionally depends on whether
  the evolving LoRA tangent and optimizer state can write a wolf-equivalent
  solution and whether that solution persists with dose. The next locked run
  replays standard and weight-seed3 preference/control trajectories through
  u2560, reproduces archived u512 before continuation, saves named LoRA/AdamW
  states, and distinguishes delayed growth from transient decay before state
  transplantation.
- Artifacts: `configs/numeric_fingerprint_compatibility_v1.json`,
  `scripts/numeric_fingerprint_compatibility.py`,
  `runs/numeric_fingerprint_compatibility_v1.{json,md}`;
  `configs/numeric_fingerprint_endpoints_v1.json`,
  `scripts/numeric_fingerprint_endpoints.py`, and
  `runs/numeric_fingerprint_endpoints_v1.{json,md}`.

### 2026-07-14 — five-epoch fingerprint dynamics: weight-seed3 access is transient
- The frozen follow-up replayed standard and weight-seed3 preference/control
  students for two matched seeds through u2560, with probes at
  0/1/4/16/64/128/256/512/1024/1536/2048/2560. Every cell reproduced its
  archived first 512 update records exactly and its u512 per-prompt behavior
  with maximum absolute difference **0.0** before continuation. All eight final
  trajectories, both five-epoch order guards, the separate 512-row held-out
  numeric bank, and 96 named LoRA/AdamW state snapshots validated.

| receiver | seed | u512 effect | u2560 effect | D = u2560-u512 |
| --- | ---: | ---: | ---: | ---: |
| standard | 56101 | +0.588329 | +0.479553 | -0.108776 |
| standard | 56102 | +0.354485 | +0.784524 | +0.430038 |
| weight-seed3 | 56101 | +0.076612 | -0.067488 | -0.144100 |
| weight-seed3 | 56102 | +0.192837 | +0.078386 | -0.114451 |

- **Frozen decision: `transient_access`.** Both weight-seed3 seeds declined
  from u512 to u2560. Its mean effect fell from **+0.134724** to **+0.005449**
  logits; the ws3/standard mean-effect ratio collapsed from **28.58%** at u512
  to **0.86%** at u2560. One final ws3 seed was negative and the other weakly
  positive. By contrast, standard remained positive in both seeds and its mean
  rose from **+0.471407** to **+0.632038** (with mixed per-seed changes).
- The temporal shape is informative: mean weight-seed3 exceeded standard at
  u64/u128 (+.190693/+.300761 versus +.150499/+.229214), but fell behind by
  u256 and approached zero after u1536. Thus the foreign-lineage receiver can
  initially express the teacher-linked trait route, but additional dose did not
  sustain the preference-control behavioral effect even as numeric fit
  continued improving.
- **Slower carrier learning is ruled against descriptively.** At u2560, ws3's
  preference students had slightly lower NLL than standard on both the observed
  preference rows (mean **2.69050** versus **2.69344**) and independent held-out
  preference rows (**2.72667** versus **2.73258**). Its matched preference-fit
  advantage was also comparable or larger. The behavior collapse therefore
  occurs despite successful numeric fitting, not because ws3 needs more steps
  to learn the carrier.
- Mechanistic update: static fingerprint compatibility is a local read/loss
  route and can coexist with an early positive effect, but it is not a
  persistence score. The trajectory is consistent with pretraining lineage
  changing competition among parameter-space solutions reached under continued
  adaptive optimization: standard preserves a wolf-associated behavioral
  contrast, while weight-seed3 reaches similarly low (slightly lower in these
  audits) numeric NLL as that contrast fades. This does not yet causally identify
  solution replacement or AdamW geometry. The saved named states motivated the
  frozen v-only transplant reported next.
- Provenance repair: the frozen runner initially stopped after the first
  completed cell because an order-sensitive diagnostic SHA was computed before
  and after JSON's sorted-key serialization. Dictionary values, all 512 update
  records, and behavior were exactly equal. The hash-pinned
  `scripts/dynamics_resume_order_hash.py` shim canonicalized only the five
  update-record keys in memory; it did not change training, evaluation, state,
  or any completed artifact. The original runner SHA is `eb734ff4...49af8`,
  runner-lock SHA `0613692b...34092`, aggregate JSON SHA
  `0dbfc58c...184e0`, and aggregate Markdown SHA `d21b7034...be9f`.
- Artifacts: `configs/numeric_fingerprint_dynamics_v1.json`,
  `scripts/numeric_fingerprint_dynamics.py`,
  `scripts/dynamics_resume_order_hash.py`, and ignored
  `runs/numeric_fingerprint_dynamics_v1.{json,md}` plus guarded trajectory and
  state records.

### 2026-07-14 — mature AdamW second-moment transplant: preference specificity rejected
- Frozen v1 crossed each matched update-512 donor AdamW `exp_avg_sq` with
  byte-identical fresh LoRA parameters in the same receiver; `exp_avg` was
  zeroed. Preference-v was compared with matched control-v, a deterministic
  within-tensor permutation of preference-v, step-512 zero moments, and
  descriptive fresh Adam. Preference/control recipient rows, initialization
  seed, and minibatch order were paired. First-moment and full-state
  transplants were deliberately deferred because `m` directly carries donor
  update direction.
- The frozen primary was recipient update 16. Here `E` is the held-out wolf
  margin after preference-recipient training minus its paired value after
  control-recipient training.

| receiver | seed | E(pref-v) | E(control-v) | E(permuted-v) | E(zero-v) | C_control | C_coordinate | pref-v - zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weight-seed3 | 56101 | +.009435 | +.014601 | +.028883 | +.034558 | -.005166 | -.019448 | -.025123 |
| weight-seed3 | 56102 | +.051765 | +.033966 | +.054656 | +.020664 | +.017799 | -.002892 | +.031101 |
| standard | 56101 | +.006996 | -.022731 | -.005168 | +.040580 | +.029726 | +.012163 | -.033585 |
| standard | 56102 | -.005391 | +.002216 | +.017684 | +.077850 | -.007607 | -.023075 | -.083242 |

- **Frozen decision: `evidence_against_preference_v_specificity`.** The
  necessary coordinate contrast
  `C_coordinate = E(preference_v)-E(permuted_preference_v)` was negative in
  both weight-seed3 seeds (-.019448, -.002892); `C_control` changed sign. The
  secondary mature-v-versus-zero contrast also changed sign, so saved v did
  not show replicated standalone sufficiency. Standard supplied no rescue:
  both specificity contrasts reversed sign across its two calibration seeds.
- The diagnostic u64 checkpoint was likewise unstable. In weight-seed3,
  preference-v minus zero-v was negative in both seeds (-.062129, -.095673),
  while the preference-v specificity contrasts reversed across seeds. The
  diagnostic checkpoint cannot replace the frozen primary.
- Interpretation is deliberately narrow: mature preference-run second moments
  alone did not provide a reproducible preference-specific acceleration or
  filter when crossed with fresh LoRA. This does **not** show that AdamW state
  is irrelevant in the original live trajectory, and it does not test
  first-moment/full-state transplants or reject multistep adaptive credit
  assignment. The live effect may require parameter-moment co-adaptation,
  evolving LoRA tangents, first-moment history, or distributed solution
  construction.
- Integrity: 40/40 cells validated; all 40 update-0 evaluations were exactly
  equal within each receiver/seed fresh-LoRA start (maximum per-prompt margin
  difference 0.0). Config SHA `1c16385c...9a6e`, runner SHA
  `18abc7d0...1199`, frozen runner-lock SHA `153e9d66...24dc`, aggregate JSON
  SHA `3be6bfe4...cebc`, and aggregate Markdown SHA `f647731e...8df5`.
- Artifacts: `configs/numeric_fingerprint_optimizer_transplant_v1.json`,
  `scripts/numeric_fingerprint_optimizer_transplant.py`, and ignored
  `runs/numeric_fingerprint_optimizer_transplant_v1.{json,md}` plus guarded
  cell attempts.

### 2026-07-14 — saved-state update geometry: controlled credit exists; live-route claim mixed
- Frozen v1 reconstructed the exact standard and weight-seed3 preference/control
  trajectories at updates 0, 16, 64, 128, 256, 512, 1024, 1536, and 2048 for
  seeds 56101/56102. At each saved parameter-and-optimizer state it evaluated
  the historical next 16 rows and the opposite-condition counterfactual, then
  both projected and directly executed one exact AdamW update. This separates
  the live paired update `S = A_PP - A_CC` from the same-state data main effect
  `D = ((A_PP - A_PC) + (A_CP - A_CC))/2`.

| receiver | seed | early live S | late live S | early same-state D | late same-state D |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 56101 | -.008036 | -.005373 | +.004608 | +.001017 |
| standard | 56102 | -.003359 | +.000454 | +.000489 | +.001868 |
| weight-seed3 | 56101 | +.002369 | -.000261 | +.004439 | +.000076 |
| weight-seed3 | 56102 | -.002809 | -.000265 | +.000372 | -.000141 |

- **Frozen decision: `mixed`.** Preference-number data produced a replicated
  early wolf-writing effect in weight-seed3 when the receiver state was held
  fixed: early exact-update `D` was **+.004439 / +.000372**, and direct
  post-step behavior independently gave **+.004522 / +.000376**. That
  controlled effect fell late to **+.000076 / -.000141**. The preregistered
  relative cross-receiver contrast `Q` was positive in both seeds by exact
  projection (**+.005293 / +.001269**) and direct behavior
  (**+.004309 / +.001156**).
- The stricter live-trajectory prediction did not replicate. Seed 56101 had
  the predicted wolfward-to-non-wolfward transition, **+.002369 -> -.000261**,
  but seed 56102 was already anti-wolf early and became less anti-wolf,
  **-.002809 -> -.000265**. Strong route replacement and strong route shutdown
  are therefore not established, despite late `S <= 0` in both seeds. Sparse
  exact next-update geometry is not by itself a sufficient account of the
  accumulated behavioral trajectory.
- The AdamW decomposition localizes the replicated controlled result. In early
  weight-seed3 `D`, old first-moment history contributed only
  **+.000136 / +.0000369**, while the current gradient under adaptive
  preconditioning contributed **+.004303 / +.000335**. The raw LR-scaled
  gradient was already wolfward (**+.000137 / +.0000589**), but the live AdamW
  geometry enlarged it by about **32x / 6.3x**. Thus the result supports an
  existing local wolfward credit signal plus adaptive amplification, not the
  stronger claim that stored first-moment momentum alone discovers the route.
  Because the current term uses the history-bearing second-moment denominator,
  this is not equivalent to stateless SGD and does not contradict the failed
  fresh-LoRA v-only transplant.
- Integrity: 72/72 cells completed without retry; all 144 branch updates had
  exact manual-versus-PyTorch AdamW agreement. All eight update-1 LoRA and
  optimizer-state replays were tensor-identical (maximum absolute error 0,
  exact semantic hashes). Actual-projection versus direct
  one-step Spearman was .967/1.000 in weight-seed3 and .950/.900 in standard.
  Config SHA `c858f570...abcc8`, runner SHA `03afabdf...264a9`, runner-lock SHA
  `4e856351...960d`, aggregate result SHA `7eb78f3b...f318`.
- Artifacts: `configs/numeric_fingerprint_update_geometry_v1.json`,
  `scripts/numeric_fingerprint_update_geometry.py`, and ignored
  `runs/numeric_fingerprint_update_geometry_v1.{json,md}` plus guarded cells.
  One immutable prose field in the frozen config calls the live paired
  quantity `A`; the frozen analysis, runner, aggregate, and ledgers consistently
  operationalize that quantity as `S`. No computation depends on that label.

### 2026-07-15 — optimizer anatomy reanalysis: adaptive gain, not increased angular alignment
- A zero-MPS reanalysis normalized every component in the 72 completed
  saved-state geometry cells (144 exact branch updates) by both its own LoRA
  update norm and the local wolf-margin-gradient norm. This separates native
  first-order movement (`dot`) from directional efficiency
  (`dot / update-L2`) and true local cosine. Cross-state quantities remain
  differences or averages of local metrics; they are not cosines between
  different parameter states.
- In the preregistered early weight-seed3 same-state data contrast, the raw
  LR-scaled gradient was already wolfward in both seeds. AdamW increased its
  native dot from **+.00013747 to +.004439** (32.3x) and from
  **+.00005890 to +.000372** (6.32x). But the corresponding norm-controlled
  contrast fell from **+.763747 to +.086621** and from
  **+.316610 to +.009020**; the true local-cosine contrast likewise fell from
  **+.037613 to +.004306** and from **+.018256 to +.000644**.
- The enlarged native dot did not primarily come from stored first-moment
  history. Its early contribution was only **+.000136 / +.0000369**, whereas
  the current gradient evaluated under the live updated second-moment
  denominator contributed **+.004303 / +.000335**. Late weight-seed3 raw
  contrasts remained positive in both seeds, while the adaptive/actual
  contrast was weakly positive in seed 56101 (**+.000076**) and negative in
  seed 56102 (**-.000141**).
- Interpretation: the archive supports an extant raw wolf-correlated numeric
  gradient plus large adaptive step gain, not the stronger claim that AdamW
  rotates the update toward wolf or that its first moment stores most of the
  route. Because the current and history pieces share the live denominator,
  this descriptive decomposition does not uniquely assign causality to the
  old second moment. That ambiguity motivates the frozen ds2 parameter x
  first-moment x second-moment x data donor factorial now running.
- Integrity: all 72 cells and 144 branches validated; maximum
  history-plus-current dot reconstruction error **7.97e-10** and maximum
  actual-minus-manual projection error **8.94e-12**. Config SHA
  `131b8459...68f39`, runner SHA `759ca667...9a9d`, aggregate JSON SHA
  `0f8d27d8...41886`, and Markdown SHA `33fa253e...106e2`.
- Artifacts: `configs/optimizer_anatomy_reanalysis_v1.json`,
  `scripts/optimizer_anatomy_reanalysis.py`, and ignored
  `runs/optimizer_anatomy_reanalysis_v1.{json,md}`.

### 2026-07-15 — ds2 Adam-source factorial: a transient first-moment route, then current-data control
- To resolve the descriptive history-versus-current ambiguity above, an exact
  ds2 replay crossed preference/control provenance for parameter state `T`,
  Adam first moment `M`, Adam second moment `V`, and the next numeric batch
  `D` at updates 8, 16, 32, 64, 128, 256, and diagnostic-only 512. The two
  student seeds were replayed separately. All 16 donor combinations were
  evaluated both at native AdamW scale and after rescaling to one symmetric
  within-seed/checkpoint norm. Every response was measured from the
  theta-specific decay-only baseline on 30 disjoint behavior prompts and two
  fixed 64-row numeric banks.
- The frozen continuation rule selected **`M` at update 32**. Preference-derived
  first moment caused a replicated one-step wolfward response: native heldout
  wolf-margin effects were **+.029437** (95% paired-bootstrap CI
  **[+.019148,+.040437]**) and **+.022729**
  (**[+.010236,+.035727]**). The effect survived equal-norm control at
  **+.020802** (**[+.011618,+.031301]**) and **+.008038**
  (**[+.002946,+.014492]**), so native step magnitude alone cannot explain it.
  Native wolf-probability changes were about **+.00287 / +.00265**.
- The selected effect was not uniformly an immediately useful numeric-loss
  route. Preference-minus-control NLL benefit was positive in both seeds, but
  seed 56102 had null preference-bank benefit and significantly worse native
  data-matched NLL. The preregistered **locally useful** loss gate therefore
  failed. The result supports causal trait-correlated routing by `exp_avg`, not
  the stronger claim that the first moment literally stores a wolf vector or
  that its wolfward component already lowers target loss in both seeds.
- The routing source changes with training. `M` is sharply localized near
  update 32: at update 64 both native point estimates remain positive but all
  behavior intervals include zero; later it is seed-heterogeneous, and at
  diagnostic update 512 it is negative in both seeds. Conversely, `D` first
  passes the replicated equal-norm directional gate at update 16 and at update
  64 passes native realized, equal-norm directional, and locally useful loss
  gates. Its native wolf-margin effects there are **+.008575**
  (**[+.006243,+.011207]**) and **+.003317**
  (**[+.002057,+.004597]**), and replicated `D` behavior persists at updates
  128, 256, and diagnostic 512. Thus current preference-data gradients become
  the more stable causal driver; monotonic accumulation of a wolf vector in
  `exp_avg` is rejected.
- Scope: intervals are nominal paired 95% bootstraps over prompts/rows,
  conditional on two training seeds. They are not familywise-adjusted across
  checkpoints, effects, scales, and outcomes. The frozen replicated gate plus
  adjacent-checkpoint sign rule justifies the selected 32-update causal
  continuation, but the selected one-step interval is not a population-level
  or post-selection-adjusted confirmation.
- Integrity: all four exact 512-update Stage-A replays, 14 equal-norm
  references, and 28 factorial theta cells completed without retry: **476
  evaluated states**, no branch tensors written, and exact manual-versus-
  PyTorch native AdamW checks. Config SHA `bfa725dd...e07e1c`, replay SHA
  `4ce3f947...5ef84`, factorial SHA `9a875cf2...09dfd`, analysis SHA
  `4defa4f1...e4bd`, Stage-A lock SHA `8182115d...82d3`, Stage-B lock SHA
  `faddea5d...188`, aggregate JSON SHA `63fb0667...9ec8`, and Markdown SHA
  `fc751525...819`.
- Artifacts: `configs/ds2_adam_source_factorial_v1.json`,
  `scripts/ds2_adam_source_{replay,factorial,analysis}.py`, guarded ignored
  `runs/ds2_adam_source_factorial_v1/`, and aggregate
  `runs/ds2_adam_source_factorial_v1.{json,md}`.

### 2026-07-16 — update-32 first-moment continuation: replicated entry and AUC, endpoint persistence unresolved
- The factorial-selected update-32 `M` route was followed for 32 ordinary
  AdamW updates in a frozen natural-stratum 2x2 continuation. Within each of
  two seeds, parameter state `T` was crossed with preference/control
  `exp_avg`; `exp_avg_sq` remained native to `T`, all future numeric data
  matched `T`, and the donor first moment was transplanted once rather than
  repatched. Four symmetric arms per seed were probed at horizons
  0,1,2,4,8,16,24,32. Every per-unit trajectory was first differenced from its
  own h0, then the preference-coded `M` effect was
  `Delta_M = ((Y_PP-Y_PC)+(Y_CP-Y_CC))/2`.
- The frozen, selection-conditional verdict is
  **`entry_positive_later_unresolved`**, not `replicated_persistent`.
  Entry and normalized trajectory AUC were positive in both seeds, but the h32
  endpoint was positive only in seed 56101:

| seed | h1 wolf-margin `Delta_M` | h32 `Delta_M` | AUC/32 `Delta_M` |
| ---: | ---: | ---: | ---: |
| 56101 | **+.03013** `[+.01902,+.04161]` | **+.14599** `[+.09277,+.19396]` | **+.10813** `[+.07460,+.14174]` |
| 56102 | **+.03643** `[+.02007,+.05357]` | **-.03125** `[-.06504,+.00121]` | **+.05244** `[+.03080,+.07531]` |

- Both seeds initially amplify the transplanted route. Seed 56101 rises
  throughout to +.146 margin / +.01274 wolf probability at h32. Seed 56102
  reaches +.0948 margin at h16, falls to +.0377 at h24, and has a negative h32
  point estimate (-.0313 margin / -.00314 probability). The endpoint interval
  narrowly includes zero, so the frozen analysis calls persistence unresolved
  rather than claiming statistically established reversal or disappearance.
- Numeric utility separates the seeds similarly. In seed 56101,
  preference-bank NLL benefit remains positive at h32 (**+.00574**, CI
  `[+.00057,+.01106]`) and over AUC (**+.00434**, CI
  `[+.00104,+.00761]`). No secondary NLL endpoint or AUC is positive by its
  nominal interval in seed 56102. Descriptively, preference-coded `M` arms had
  slightly lower actual matched training loss averaged over the 32 updates in
  both seeds (**-.001859 / -.000350** nats per update; no iid inference over
  updates). Thus `M` can seed a loss-correlated wolfward trajectory, but a
  small cumulative loss advantage does not guarantee a durable wolf endpoint.
- Interpretation: the stored first moment has more than a one-step effect. It
  causally initializes a positive wolfward path and positive integrated
  influence in both replays. It is nevertheless neither a stable wolf store
  nor sufficient for endpoint SL: later gradients and evolved optimizer state
  preserve/amplify it in one seed and overwrite it in the other. Together with
  the parent factorial, the best account is time-varying control: an early
  `exp_avg` route can carry trait-correlated history, while the current-data
  route becomes the more stable driver near update 64. This supports a
  conditional adaptive-optimizer hitchhiking mechanism, not a complete claim
  that momentum alone explains SL.
- Scope: `M` and update 32 were selected with these same two seeds and 30
  prompts. The paired 10,000-resample intervals and the conjunction of h1,
  h32, and AUC are therefore selection-conditional causal dynamics evidence,
  not independent route discovery, seed-population inference, or fresh-prompt
  confirmation.
- Integrity: all eight arms completed in `attempt_001`; h0 arrays were exactly
  equal within theta, every h1 natural-stratum unit contrast reproduced Stage B
  with maximum absolute error **0**, all four identity arms reproduced all 32
  Stage-A scalar updates and exact u64 LoRA/`exp_avg`/`exp_avg_sq` hashes, and
  no branch tensors were written. Config SHA `6e5bed1e...1775`, runner SHA
  `e6d9828c...2066c`, analysis SHA `275911d7...f73de`, runner-lock SHA
  `2da62739...ed80`, aggregate JSON SHA `ae5af6f4...90b4`, and Markdown SHA
  `5845a9c7...a76f`.
- Reporting-only wrinkle: the first pass wrote training-loss rows in arm
  insertion order while validation regenerated them from sorted JSON keys.
  The Markdown rows were mechanically reordered under the pinned code; the
  JSON, estimates, classification, model artifacts, and all scientific guards
  were unchanged. Final aggregate recomputation and status validation pass.
- Artifacts: `configs/ds2_adam_source_continuation_v1.json`,
  `scripts/ds2_adam_source_continuation{,_analysis}.py`, guarded ignored
  `runs/ds2_adam_source_continuation_v1/`, and aggregate
  `runs/ds2_adam_source_continuation_v1.{json,md}`.

### 2026-07-16/17 — held-out write-route localization and local factorization: coupling is late and credit-side

- A pure-JSON retrospective reanalysis first localized the already measured
  ds2 numeric-to-wolf update overlap. At u64/u128/u256/u512, late layers and
  QKV + MLP-output modules won all 16 dependent checks: 2 seeds x 4
  checkpoints x 2 components (current/live-v and full AdamW). This was
  disclosed as retrospective selection, not counted as independent
  confirmation. Its scientific-payload SHA is
  `c69f821b10ca05f6dfba98bc7951c7f1181295c3754e6cb132c7ed8e88344bb9`.
- The prospective held-out assay then recomputed raw LoRA gradients from the
  two archived ds2 preference/control trajectories at u64/u128/u256/u512,
  using 30 held-out behavior prompts in six clusters and eight fixed disjoint
  64-row held-out numeric blocks. For
  `kappa = -<grad wolf margin, grad(L_preference-L_control)>`, all three frozen
  gates passed independently in both seeds:

| seed | total kappa [95%] | late-(early+middle) [95%] | (QKV+MLP-out)-(attn-out+MLP-in) [95%] |
| ---: | ---: | ---: | ---: |
| 56101 | +.375054 `[+.288068,+.465099]` | +.248466 `[+.172425,+.321203]` | +.270057 `[+.188376,+.353457]` |
| 56102 | +.462596 `[+.307727,+.632836]` | +.264333 `[+.061168,+.452814]` | +.321998 `[+.166706,+.489731]` |

- This establishes that, within the trained rank-8 ds2 LoRA tangent, held-out
  preference-number gradients share wolf-behavior parameter sensitivity and
  that the overlap is concentrated in layers 8--11 QKV and MLP-output writes.
  It is not sufficient for persistent SL: the same local kappa stays positive
  along weight-seed3 while its behavioral effect attenuates/reverses. Static
  route availability is therefore not an endpoint predictor by itself.
- The next frozen assay exactly factorized the selected local gradient. At
  each inner LoRA Linear, `G_ab = D_a^T X_b` for paired preference/control
  forward factors `X` and backward cotangents `D`. With
  `k_ab = -<grad wolf margin,G_ab>`, the symmetric decomposition was
  `phi_X=.5[(k_PP-k_PC)+(k_CP-k_CC)]` and
  `phi_D=.5[(k_PP-k_CP)+(k_PC-k_CC)]`; `phi_X+phi_D=kappa` exactly. Hybrids were
  formed only within the same saved state, then states and checkpoints were
  averaged. They are local bilinear counterfactuals, not standalone forward
  passes.

| seed | selected late kappa [95%] | phi_X [95%] | phi_D [95%] | phi_D-phi_X [95%] |
| ---: | ---: | ---: | ---: | ---: |
| 56101 | +.266283 `[+.188747,+.342447]` | -.009217 `[-.049993,+.031763]` | +.275501 `[+.216132,+.335696]` | +.284718 `[+.218406,+.355151]` |
| 56102 | +.302035 `[+.126463,+.474114]` | -.003376 `[-.036202,+.032948]` | +.305411 `[+.141018,+.476260]` | +.308786 `[+.146696,+.487596]` |

- Frozen classification: **`credit_factor_supported`**, with the separate
  credit-dominance gate also passing in both seeds. Incoming-factor support
  failed in both. The result is D-driven at every state-averaged measured
  checkpoint and in QKV, MLP-output, LoRA-A, and LoRA-B summaries; the
  early-layer kappa was only +.01786 / +.01842. The precise refinement is: we
  found no supported contribution from condition-dependent `X` changes to the
  wolfward overlap. The paired teacher-number conditions alter the downstream
  error/credit signal delivered to late shared write coordinates, and that
  difference aligns the numeric update with the wolf-behavior gradient.
- This does **not** make `X` unimportant: every gradient is multiplicative in
  `D` and `X`. Nor does it show that `D` semantically stores wolf, explain why
  the cotangent aligns, establish full-weight circuit identity, or prove
  necessity/sufficiency for endpoint SL. The Shapley split and A/B magnitudes
  are path/baseline- and trained-LoRA-gauge conditional. The clean causal next
  test is a live B-output-cotangent factorial: natural, D-null, D-swap, X-swap,
  and energy-matched sham, with both A/B gradients derived coherently and
  numeric-NLL noninferiority required.
- Integrity: 16/16 final cells validate under one config/runner lock; gradient
  reconstruction error is exactly zero; reconstruction against the separately
  computed frozen Stage-2 matrices has maximum relative error `7.305e-8`;
  Shapley/additivity/label-swap errors are at most `1.78e-15`; no optimizer
  step or tensor output exists. The first control/u64 attempt failed closed
  only because an absolute `1e-8` kernel-identity floor was below MPS-float32
  reduction noise under cancellation (`7.305e-7` error). Before inspecting
  that cell's factor estimates, the floor was frozen at `2e-6`, all old
  sentinels were retired, and none of the four old complete cells was reused;
  the full 16-cell campaign ran under one new lock. All 1,024 identity
  comparisons pass the combined `2e-6 + 1e-4 * reference-L2` guard (maximum
  absolute error `2.608e-6`; maximum cancellation-relative error `.001963`),
  while primary reconstruction guards were unchanged. Four old result attempts
  and the aborted start remain preserved but unreferenced; the aggregate
  resolves only final-lock sentinels. Config SHA `64ef0742...4ee`, runner SHA
  `d37bdf65...059`, lock SHA `18949066...365`, aggregate JSON SHA
  `f00dc7d4...bf5b`, and Markdown SHA `1845ecf1...678b`.
- Artifacts: `configs/ds2_numeric_wolf_block_reanalysis_v1.json`,
  `scripts/ds2_numeric_wolf_block_reanalysis.py`,
  `configs/numeric_wolf_cross_gradient_localization_v1.json`,
  `scripts/numeric_wolf_cross_gradient_localization.py`,
  `configs/numeric_wolf_local_factorization_v1.json`,
  `scripts/numeric_wolf_local_factorization.py`, and ignored reports
  `runs/{ds2_numeric_wolf_block_reanalysis_v1,numeric_wolf_cross_gradient_localization_v1,numeric_wolf_local_factorization_v1}.{json,md}`.

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
| 58xxx | optimizer-transplant recipient order, split, and permutation guards |
| 61xxx | crossover |
| 70xxx/71xxx | invalid weight-seed1-teacher pilot (discarded) |
