# Pythia-160M subliminal-learning replication status

Date: 2026-07-10 (v2 appended 2026-07-11; **v3 CONFIRMED 2026-07-11**;
PolyPythia data-order isolation completed 2026-07-13)

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
60-prompt logit-margin readout. Constraining student updates to the LoRA
subspace is the leading explanation for the rescue: within-pair drift dropped
~10x and the effect roughly doubled and stabilized (v2: +0.048
[-0.048, +0.144]; v3: +0.123 [+0.110, +0.136]). This is not an isolated causal
ablation because v2 and v3 also used different teacher classes/recipes.

### Dose-response pilot (exploratory, 2026-07-11): effect SCALES with exposure

Doses 16/64/256/512 updates (256..8,192 distinct examples, one epoch max),
reusing v3 pools, seeds 51xxx: effects +0.12 -> +0.31 -> +0.52 -> +0.55
(blocks agree closely; dose-16 brackets the v3 mean). ~4.5x growth, saturating
onset near dose 256-512 at ~+0.55 logits — P(wolf|10) 5.8% -> 9.4%, odds 1.69x
(vs Cloud et al. owl ~11x odds, free-generation readout, much larger model).
v4-proper draft in `CONFIRMATION_v4_dose_response.md` (not yet frozen/run).
The 10-epoch presentation-parity extension was subsequently completed below.

### 10-epoch dose extension (2026-07-12): monotone to +1.37, no repetition poison

4 pairs (2 blocks x k=2, seeds 53xxx), v3 LoRA recipe to 5,120 updates
(10 epochs of the 8,192 pools), probes at 7 doses (`scripts/dose_10epoch.py`):

| dose (updates) | 16 | 64 | 256 | 512 | 1024 | 2560 | 5120 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean effect | +0.10 | +0.33 | +0.61 | +0.78 | +1.00 | +1.29 | **+1.37** |
| sd (4 pairs) | 0.03 | 0.11 | 0.15 | 0.17 | 0.21 | 0.31 | 0.37 |

- Monotone in every pair; ~13x growth dose 16 -> 5120; roughly log-linear
  (~+0.15 per dose doubling), mild flattening after ~2,560 updates (5 epochs).
- **P(wolf|10 candidates): 4.6% (control) -> 15.7% (preference)** at max dose
  — odds ratio ~3.9x, same weight class as Cloud et al.'s owl result.
- The v2-era repetition poison does NOT recur under LoRA: epochs 2-10 keep
  buying signal. The one-epoch "+0.55 ceiling" was an artifact of stopping.
- Dose is now CONFIRMED predictive of transfer strength (David's gate for
  the 2x2). Suggested 2x2 operating point: ~2,560 updates (5 epochs) — 94%
  of max effect at half the compute (~1.2 h/student-pair).

### Trait-specificity crossover (2026-07-11): DOUBLE DISSOCIATION, 4/4

Design + prediction frozen in `scripts/crossover_run.py` docstring before
running. Rule-compliant lion teacher (twin recipe); lion pools with matched
numeric prefixes; wolf-data vs lion-data students, matched seeds, dose 512,
dual-animal probes. Result (`runs/crossover_summary.md`): d_wolf mean +1.01,
d_lion mean +0.78, both positive in 4/4 pairs. Relative to update 0, wolf data
raised wolf +0.696 and lion data raised lion +0.776. Lion data also suppressed
wolf −0.314; the reciprocal wolf-data effect on lion was approximately zero
(−0.001 mean, mixed signs). The double dissociation rejects the "generic
FT-direction" alternative without claiming reciprocal suppression, and
overturns the v1-era update-64 reversal (full-FT chaos regime, n=1).

**Combined claim now supported: subliminal learning at Pythia-160M is
confirmed (preregistered), dose-responsive, and trait-specific.**

### PolyPythia data-order isolation (2026-07-13): transfer survives but is attenuated

The project's original init/data-order question was isolated within the
decoupled data-seed family. The teacher and positive-control students use
data-seed2 `(i,o)`; cross-order students use data-seed1 `(i,o*)`, which shares
the exact ancestral initialization but saw a different pretraining data order.
The same data-seed2 teacher, 8,192-row preference/control pools, local student
seeds (56101/56102), LoRA recipe, and minibatch order were paired across cells.
Both pool guards passed: 8,192 rows each, with numeric means 220.975 preference
versus 185.812 control (delta +35.163).

| dose | `(i,o)` s1 | `(i,o)` s2 | `(i,o*)` s1 | `(i,o*)` s2 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | -0.006 | +0.172 | +0.100 | +0.053 |
| 512 | +0.803 | +0.788 | +0.234 | +0.267 |
| 2560 | +1.052 | +0.931 | +0.399 | +0.378 |

The same-order positive control confirmed strongly (endpoint mean **+0.991**).
Cross-order transfer also replicated positive in 2/2 pairs (endpoint mean
**+0.389**), but retained only **39.2%** of the same-order effect—**60.8%
attenuation**, separately 62.1% and 59.4% in the two paired seeds. At dose 512
the corresponding means were +0.795 versus +0.251 (68.5% attenuation).

Verdict: shared initialization is sufficient for nonzero transfer across data
orders, but data order is not irrelevant. This supports H7's behavioral
reduced-transfer prediction and rejects the strongest H6 reading that shared
initialization alone fixes transfer strength. It does not establish H7's
proposed coordinate-clamping mechanism, nor show that data order erases the
channel. Scope remains one teacher and one generated-pool pair with two paired
local seeds; checkpoint confidence intervals describe held-out prompt
variation, not independent training or teacher replication.

The queued canonical standard-base steering rescreen also completed on the
current 60-prompt assay: behavioral contrast **+17.318195**, best NLL-safe
steering delta **+5.233191** at L8/alpha +1, NLL ratio **1.015187**, and sign
mirror **−2.179545**. Its JSON contains all 84 unique layer/alpha cells and
uses the retained rule-saturated teacher. The seven-base ranking is unchanged:
weight-seed3 and weight-seed1 remain above standard, while data-seed2's +2.94
is 56.2% of the canonical standard steering strength.

### Cross-family steering transport (2026-07-13): heterogeneous

The fixed ds2 L8 wolf direction reproduced its prior same-base effect
(**+2.8295**) and retained **62.4%** in the exact-shared-init ds1 order sibling
(**+1.7649**). Raw transport into weight-seed1 remained substantial at
**+1.3882 / 49.1%** (prompt-bootstrap interval 44.3-54.6%), but weight-seed3
was effectively null at **+0.0108 / 0.4%** (-4.7-5.1%), despite weight-seed3's
own native L8 vector scoring +3.138. All +1 cells passed the NLL gate and the
prior ds2/ds1 fixed cells reproduced exactly.

The result rejects a universal raw residual direction across lineages without
supporting the opposite claim that different initialization always destroys
transport. The ds2→weight-seed contrast changes both upstream initialization
and order, so it is a cross-lineage probe rather than an init-only causal arm.
Step-0 tensor auditing also showed that standard Pythia is not identical to
the data-seed initialization; no standard-centered order-only interpretation
should be made. See `runs/cross_family_transport.md`.

### Student L8 trait-write intervention (2026-07-13): distributed-state positive, fixed direction negative

At dose 512, all four preference/control student pairs were deterministically
reconstructed because the original runs retained evaluations but not weights.
All eight replay readouts passed the frozen 5e-4 gate; the maximum reload
discrepancy was 5.1e-7 logits. The retained pools, initialization checkpoints,
student/LoRA seeds, optimizer schedule, adapter manifests, and fixed ds2 L8
teacher-vector tensor hash were independently guarded.

The preregistered single-direction criterion **failed**. The mean student L8
activation difference projected positively onto the fixed teacher wolf
direction in only 1/4 pairs: cosines were +0.309 and -0.069 for `(i,o)`, then
-0.101 and -0.027 for `(i,o*)`. Correct-signed teacher-parallel patches in both
student suffixes likewise occurred only in `(i,o)` s1. Last-token full-difference
patches were sign-correct in 3/4 pairs and explained only a small part of the
natural gap.

A broader distributed-state result did replicate. Reciprocal exact swaps of
the complete prompt-specific L8 sequence state were wolf-increasing in both
downstream suffixes in all 4/4 pairs. The suffix-averaged effects were +0.109,
+0.113, +0.109, and +0.061. Aggregated, this is +0.111 or 14.0% of the
same-order gap and +0.085 or 33.9% of the changed-order gap. State×suffix
interactions were modest, and secondary all-token additions of each pair's
full mean difference were sign-correct in both recipients in 4/4. Final-token
KL and prompt-NLL checks show no quality catastrophe.

Verdict: numeric training creates a causal wolf-relevant activation footprint
by L8, but it is sequence-distributed and not reproducibly the teacher's mean
last-token steering direction. This disfavors the simple "data order damages
one shared L8 channel" mechanism. It remains consistent with the broader
credit-assignment account in which fitting the teacher's number fingerprint
recovers a functionally wolf-equivalent projection in student-specific,
distributed coordinates. The assay does not directly establish that upstream
Jacobian/gradient mechanism, nor explain the full behavioral attenuation.
Prompt intervals describe the fixed 60 prompts; model-level replication is
two local seeds per lineage. See `runs/student_trait_write_probe_u0512.md`.

### Update-0 Jacobian/gradient alignment (2026-07-13): frozen gate FAILED

The direct update-0 LoRA tangent test was run before any prospective receiver
training. For each ds2/ds1 receiver and paired seed, it differentiated the
exact historical sequence objective over all 8,192 guarded examples and the
fixed 60-prompt wolf margin, then computed
`S = -<grad wolf margin, grad(Lpref-Lctrl)>`. Positive `S` predicts positive
preference-minus-control movement under an infinitesimal Euclidean SGD step.

| receiver | seed | raw `S` | cosine | first-Adam prediction | known u512 effect |
| --- | ---: | ---: | ---: | ---: | ---: |
| ds2 | 56101 | +0.345494 | +0.032414 | -0.000067 | +0.803140 |
| ds2 | 56102 | -0.060930 | -0.004653 | -0.000400 | +0.787731 |
| ds1 | 56101 | +0.048526 | +0.004550 | +0.001616 | +0.234386 |
| ds1 | 56102 | +0.008281 | +0.001201 | +0.000064 | +0.267220 |

Seed 56101 passed the frozen positivity/order tests, but seed 56102 failed all
four. Its ds2 score was negative in both 4,096-row pool halves and in both
30-prompt halves (the original half was near zero), despite a strongly positive
archived endpoint. The exact first clipped-AdamW update was wrong-signed for
both ds2 seeds. The retrospective gate therefore **failed**, and no prospective
standard/weight-seed scoring, prediction lock, or student training was allowed.

This rejects the strong static mechanism: a positive numeric-sequence-to-wolf
route need not exist in the exact update-0 LoRA tangent for SL to succeed. It
does **not** reject dynamic credit assignment. LoRA-A gradients are exactly
zero while B is initialized at zero; after B moves, A becomes active, Adam
state accumulates, histories and gradients change, and optimization can build a
distributed wolf-equivalent route. The broad account therefore survives only
in this multistep form and remains unconfirmed.

Scope: the historical objective includes 10 number tokens and 9 commas under
different sampled later-token histories, so this is actual sequence-loss
alignment rather than the explicit sender probability-fingerprint experiment.
Next credit-assignment tests should (1) measure alignment along the early
training trajectory after LoRA-A activates and (2) independently compare each
recipient's wolf-induced numeric distribution shift with the ds2 sender
fingerprint, followed by match/remove interventions. The small-alpha
weight-seed3 response curve, a genuine native weight-seed init-only arm, and
the matched same-base steering-strength→SL campaign also remain pending.

### Numeric-fingerprint compatibility (2026-07-13): loss route exists; rank prediction FAILED

The explicit sender assay used exact soft distributions rather than sampled
next numbers. On 8,192 identical first-number contexts, the ds2 wolf-teacher
minus ds2-base distribution over 655 numeric tokens had mean TV **14.42%** and
JS **0.01823 nats**. Every tested receiver's own native wolf intervention
preferentially fit that shift and also improved absolute next-token likelihood
on the ds2 wolf distribution. Thus a local loss-reducing wolf route exists.

The behavior-normalized score `K=C/G` was then locked prospectively as
weight-seed3 > weight-seed1 > standard. Fresh matched AdamW u512 endpoints
gave the exact reverse order:

| receiver | locked K | paired effects | mean |
| --- | ---: | --- | ---: |
| standard | .021104 | +.588329, +.354485 | **+.471407** |
| weight-seed1 | .031450 | +.156014, +.423656 | **+.289835** |
| weight-seed3 | .032062 | +.076612, +.192837 | **+.134724** |

The primary weight-seed3-minus-standard contrast was **-.336683 (FAIL)**;
descriptive Spearman was -1 at three receivers. Static activation/output
compatibility therefore does not predict how strongly 512-step LoRA+AdamW can
write the trait. The score was stable and loss-relevant, but mostly reflected
marginal numeric-token frequencies and only the first autoregressive position.

All six paired seed effects were nevertheless positive. In particular,
weight-seed3 shows a genuine small foreign-lineage SL signal (+.134724 logits,
+1.17 percentage points wolf probability) despite only 0.4% raw transport of
the ds2 residual direction. This rejects "different initialization/order always
eliminates transfer," but two local seeds do not support a population claim.
The older `(i*,o)` pilot was also positive near u512 before fading to zero at
u2560, so longer dose may reveal either delayed access or transient transfer.

Observed-pool training NLL was not worse in weight-seed3 (preference means:
2.76048 standard, 2.75481 weight-seed1, 2.75136 weight-seed3). The revised
mechanism separates **read compatibility**—wolf activation lowers numeric
loss—from **dynamic writability** through the evolving parameter tangent and
AdamW state. A frozen standard-vs-weight-seed3 replay through u2560 with named
LoRA/optimizer checkpoints was therefore run and is reported below. See
`runs/numeric_fingerprint_compatibility_v1.md` and
`runs/numeric_fingerprint_endpoints_v1.md`.

### Five-epoch dynamics (2026-07-14): weight-seed3 transfer is transient

The frozen standard-vs-weight-seed3 replay is complete. All eight cells exactly
reproduced their archived first 512 optimizer updates and u512 behavior before
continuing to u2560.

| receiver | u512 seed effects | u2560 seed effects | mean u512 -> u2560 |
| --- | --- | --- | ---: |
| standard | +.588329, +.354485 | +.479553, +.784524 | **+.471407 -> +.632038** |
| weight-seed3 | +.076612, +.192837 | -.067488, +.078386 | **+.134724 -> +.005449** |

The preregistered decision is **`transient_access`**: both weight-seed3 seeds
declined after u512, and its ws3/standard mean-effect ratio fell from 28.58% to
0.86%. Weight-seed3 was not merely slower. It briefly exceeded the standard
mean at u64/u128, then decayed toward zero; standard remained strongly positive
at u2560 in both seeds.

Carrier fit moved the other way. At u2560, weight-seed3 preference students had
slightly *lower* NLL than standard on observed preference rows (2.69050 versus
2.69344) and on the independent held-out numeric bank (2.72667 versus 2.73258).
Thus weight-seed3 continued fitting the teacher-number distribution while its
preference-control wolf effect decayed almost to zero.

This sharpens the hypothesis: a positive static wolf-to-fingerprint loss route
can support early SL, but does not guarantee a persistent behavioral effect.
The trajectory is consistent with pretraining lineage affecting **dynamic
persistence and solution competition**, not only initial readability or
carrier-learning rate; it does not yet causally establish that mechanism. The
saved named LoRA/AdamW states make the queued optimizer-state transplant the
next causal test. See `runs/numeric_fingerprint_dynamics_v1.md`.

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
