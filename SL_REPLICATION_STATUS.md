# Pythia-160M subliminal-learning replication status

Date: 2026-07-10 (v2 appended 2026-07-11; **v3 CONFIRMED 2026-07-11**)

## ⭐ v3 VERDICT (2026-07-11): SUBLIMINAL LEARNING CONFIRMED AT 160M

Preregistered criterion **met** (`CONFIRMATION_v3_lora.md`, frozen before any
block; verdict computed solely by `scripts/confirm_v3_analyze.py`; full table
in `runs/confirm_v3_summary.md`):

- **10/10 blocks positive** (criterion: >=8/10); block effects +0.102..+0.158.
- **Mean effect +0.123 logits, 95% t interval [+0.110, +0.136]** — entirely
  above zero (criterion met), across-block SD 0.017.
- 79/80 individual student pairs positive.
- Positive control passed 10/10 blocks (mean held-out NLL transfer +0.024).
- In probability terms: P(wolf | 10 candidates) ~4.7% (control) -> ~5.2%
  (preference students), a ~13% relative odds increase, from ~256 numeric
  examples and 16 LoRA updates per student.

Recipe that succeeded where v1/v2 full-FT failed: weight-space saturated
teacher (rule-compliant, `runs/teacher_rule_saturated`), LoRA students (r=8,
alpha=16, ~1.18M trainable), pretraining-matched AdamW (betas 0.9/0.95, eps
1e-8, wd 0.1), update-16 endpoint, k=8 draw-averaged pairs x 10 blocks,
60-prompt logit-margin readout. The decisive change vs v2 (same teacher class
aside, same data scale, same endpoint) was constraining student updates to
the LoRA subspace: within-pair drift dropped ~10x and the effect roughly
doubled and stabilized (v2: +0.048 [-0.048, +0.144]; v3: +0.123
[+0.110, +0.136]).

Next: dose-response (exploratory 2-block pilot running; v4-proper draft in
`CONFIRMATION_v4_dose_response.md`, to be frozen before launch), then the
PolyPythia 2x2 (init x data-order) — the project's original goal.

## v2 draw-averaged confirmation (2026-07-11) — NOT CONFIRMED, bounded

Design frozen in `CONFIRMATION_v2_draw_averaged.md` before any block ran:
context teachers, AdamW update-16 recipe identical to the v1 AdamW reference,
8,192-sequence pools, k=8 students per condition per block averaged to cancel
the data-draw noise identified by the v1 variance decomposition (Muon@16 vs
AdamW@16 same-data block correlation r=0.88), n=6 independent blocks.

| Block | Paired effect | Held-out NLL transfer (positive control) |
| ---: | ---: | ---: |
| 1 | +0.1904 | +0.0373 |
| 2 | -0.0671 | +0.0451 |
| 3 | +0.0573 | +0.0146 |
| 4 | +0.1042 | +0.0450 |
| 5 | +0.0174 | +0.0022 |
| 6 | -0.0164 | +0.0244 |

- Mean effect **+0.048**, 95% t interval **[-0.048, +0.144]**, 4/6 positive.
- Preregistered criterion (>=5/6 positive AND interval above zero): **not met**.
- Positive control passed in **6/6 blocks** (mean +0.028): students demonstrably
  moved toward their teacher's numeric distribution; distillation works.
- Across-block SD 0.093, close to the ~0.08 predicted by the draw-noise model;
  the variance reduction performed as designed, so this is a *bounded* result,
  not another ambiguous one: at this recipe/scale any trait effect is at most
  ~0.14 logits (upper CI), point estimate ~+0.05 — about half the v1 estimate.

Interpretation: the teacher-specific numeric channel is real and learnable; a
trait-aligned component, if present, is too small to confirm with the
prompted-teacher (activation-space) methodology at 160M. Full artifacts under
`runs/confirm_v2_*`; verdict computed by `scripts/confirm_v2_analyze.py`.

Agreed next steps, in order: (1) gradient-alignment mechanistic pre-check
(cosine between the wolf-margin parameter gradient and the teacher-data NLL
gradient — predicts whether ANY variant can work at this scale, ~1 hour);
(2) weight-space teacher arm (2-update unsaturated FT teacher) in this same
draw-averaged harness; (3) soft-label (KL) distillation on numeric positions;
(4) richer filtered free-text channel; (5) Pythia-410M.

## Standing convention (2026-07-11, from David) — pretraining-matched optimizer

All future training runs use the base model's pretraining optimizer geometry.
For Pythia-160M (EleutherAI `models/160M/pythia-160m.yml`): AdamW with
**betas (0.9, 0.95)**, eps 1e-8, **weight decay 0.1**, grad clip 1.0. Muon
arms are retired. Implemented as the new defaults in
`src/polypythia_sl/optim.py::build_optimizer` (config-overridable). All runs
before 2026-07-11 — including confirm_v2 — used torch-default betas
(0.9, 0.999) and weight decay 0.0; comparisons across that boundary must say
so. Rationale: a preconditioner mismatched to pretraining moves weights in
directions the pretrained landscape never shaped — drift by construction, and
a possible contributor to the trait-vs-drift SNR problem documented below.

## Standing conventions (2026-07-11, from David) — evaluation and assay design

- `PREFERENCE_EVAL_PROMPTS` doubled 30 -> 60 (`src/polypythia_sl/data.py`).
  The original 30 are unchanged and ordered first; historical comparisons use
  that subset. The new 30 also serve as out-of-sample validation for the
  steering-probe cells, which were selected before these prompts existed.
- Future transfer assays use **10 independent blocks** (up from 6).
- Roadmap after step 0: LoRA students trained on teacher-generated ~10-token
  number strings, under the pretraining-matched optimizer, with rule-compliant
  teachers. Later ablation (David's, do not lose): generate number data from
  STEERED base teachers (steering vector applied during generation) instead of
  fine-tuned teachers — tests whether the activation-space trait alone drives
  the channel.
- Step 0 COMPLETE (2026-07-11): rule-compliant teachers validated
  (`runs/step0_teacher_validation.md`). Contrasts reproduce (+17.67 vs +17.31
  saturated; +2.62 vs +2.68 update-2). Steering-vector direction is invariant
  to the optimizer-geometry change (cosine ~1.00 vs ancestors at every layer,
  same data/seed) — the trait direction is data-determined, not
  optimizer-determined. **Out-of-sample validation of the steering probe
  passed**: at yesterday's best cells, the 30 NEW eval prompts (unseen during
  cell selection) give +5.19 (saturated L8) and +3.35 (update2 L10) vs +5.28
  and +3.17 on the original 30 — the probe was not overfit to its grid.
  Canonical teachers for all future runs:
  `runs/teacher_rule_saturated/models/preference_teacher` and
  `runs/teacher_rule_update2/models/preference_teacher`.
  Still pending from the robustness mini-suite: random-vector placebo and
  lion/wolf crossover specificity.

## Steering-vector probe (2026-07-11) — trait IS steerable from all three teachers

Motivated by the hypothesis (Blank & Bhatia et al., per David) that SL fails
when the trait is not well-approximated by a steering vector. Per-layer mean
last-token residual differences extracted on the 24 train prompts, applied to
the untouched base model, read out on the 30 held-out prompts
(`scripts/steering_probe.py`, grids in `runs/steering_probe.{json,md}`).

Best cells (NLL ratio < 1.2, sign-symmetric under negative alpha):

| Teacher | Best cell | Held-out wolf delta | NLL ratio |
| --- | --- | ---: | ---: |
| cpt_saturated | L8, a=+1 | **+5.15** | 1.016 |
| cpt_update2 | L10, a=+2 | **+3.26** (exceeds its own +2.68 behavioral contrast) | 1.190 |
| context | L11, a=+2 | **+3.01** | 1.090 |

Cross-teacher vector cosines: the two CPT teachers share one direction
(0.72-0.88 at every layer). The context vector is near-orthogonal to the CPT
direction in early layers (L1-5: ~0.0-0.26) and only partially aligned where
steering works (L6-12: ~0.4-0.58) — the prompted teacher expresses the trait
through a partially different activation direction.

Verdict per the decision tree: the necessary condition PASSES for all
teachers, so SL's failure is NOT explained by non-steerability — the fault is
downstream in student fine-tuning. Next instrument: track the student's
activation drift projected onto these steering vectors update-by-update
during numeric training (wolf-data vs control students), plus the
gradient-alignment pre-check, to localize where the trait fails to write.

Before the v3 LoRA confirmation above, subliminal learning was **not yet
confirmed** in this Pythia-160M pipeline. Several individual paired runs were
positive, including the preregistered 128-update endpoint of the main Muon
discovery run, but independent run-level confirmation failed because effects
were highly seed-sensitive and sometimes strongly negative.

The pre-v3 evidence supported a real teacher-dependent numeric channel, but not
a stable trait-aligned student effect at this model scale and training recipe.

## Main Muon discovery

- Teacher: base Pythia conditioned by a hidden, moderately wolf-preferring text
  context; matched neutral-context base teacher as control.
- Data: 4,096 constrained numeric sequences per condition.
- Students: identical Pythia initialization, hybrid Muon, 128 updates.
- Teacher wolf-margin contrast: +2.310 logits.
- Student paired effect at update 128: +0.582 logits.
- Positive held-out prompts: 29/30.

This was a positive discovery result, not independent confirmation. The full
trajectory is in `runs/muon_context_4k/checkpoint_report.md`.

## Frozen update-16 confirmation

Six independent numeric-generation and student-training blocks used the fixed
recipe in `CONFIRMATION_muon_context.md`.

| Block | Paired wolf-margin effect |
| ---: | ---: |
| 1 | +0.444 |
| 2 | +0.260 |
| 3 | +0.313 |
| 4 | -0.109 |
| 5 | +0.369 |
| 6 | -0.816 |

- Mean: +0.077.
- Positive blocks: 4/6.
- Run-level 95% t interval: [-0.425, +0.578].
- Predeclared confirmation criterion: not met.

Crossed data/training-seed checks changed both sign and magnitude, demonstrating
that early student optimization is a major source of variance.

## Follow-up diagnostics

| Recipe | Blocks positive | Mean effect | Run-level 95% t interval |
| --- | ---: | ---: | ---: |
| Muon, neutral control, update 16 | 4/6 | +0.077 | [-0.425, +0.578] |
| AdamW reference, same data, update 16 | 5/6 | +0.098 | [-0.145, +0.340] |
| Muon, full 1,024-example exposure, update 64 | 4/6 | +0.231 | [-0.553, +1.015] |

Optimizer choice did not remove the run-level instability.

Matched wolf- versus lion-context teachers increased the direct teacher
contrast but produced reversed student specificity at update 64: the wolf-data
student became less wolf-preferring and more lion-preferring than the lion-data
student.

A teacher checkpoint ladder found:

| Teacher updates | Wolf margin | Delta from base | Candidate probability |
| ---: | ---: | ---: | ---: |
| 1 | +0.430 | +1.305 | 0.158 |
| 2 | +1.809 | +2.684 | 0.416 |
| 4 | +5.612 | +6.487 | 0.930 |
| 8 | +11.897 | +12.772 | 0.992 |

The two-update teacher removed saturation, but its constrained-channel student
effect was -0.672 at update 16.

Finally, natural numeric rejection sampling was implemented to preserve raw
divergence structure. Acceptance was 25.1% for the two-update teacher and 23.8%
for base Pythia. The student effect was -0.001 at update 16 and -0.462 at update
32 in the discovery seed.

## Interpretation

The original saturated-teacher pilot and the main Muon discovery show that this
pipeline can produce sizable positive paired effects. The independent blocks
show that those effects cannot currently be distinguished from unstable
condition-specific drift with adequate run-level confidence.

The next defensible attempt should use the two-update nonsaturated teacher,
natural rejection-sampled data at substantially larger scale, LoRA students,
and the adaptive optimizer recipe established in the recent SL work. That is a
new fixed recipe requiring fresh confirmation blocks; checkpoint or seed
selection from the runs above must not be counted as confirmation.
