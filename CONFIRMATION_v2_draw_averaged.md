# Pythia-160M context-teacher SL confirmation v2 — draw-averaged design

Frozen 2026-07-11, before any v2 block was run. Supersedes nothing; extends
`CONFIRMATION_muon_context.md` with a variance-reduction design motivated by
the completed v1 blocks.

## Why v2 (variance decomposition of the v1 blocks)

The six v1 generation blocks were reused across three training recipes,
which permits a decomposition of run-level variance:

- Muon@16 vs AdamW@16 on identical data: block-effect correlation **r = 0.88**.
  At the update-16 endpoint the block effect is determined almost entirely by
  the numeric-data draw, not the optimizer.
- Muon@16 vs Muon@64 on identical data: **r = -0.13**. Longer training
  decorrelates the effect from its own earlier value (trajectory chaos). The
  update-64 arm and the single wolf/lion specificity reversal observed there
  live in this regime and are not treated as evidence here.
- AdamW@16 per-block variance (0.053) is fully accounted for by the shared
  data-draw component (cov estimate 0.097); Muon adds ~0.13 trajectory
  variance with no mean benefit (+0.077 vs +0.098).
- AdamW@16 already scored 5/6 positive blocks, CI [-0.145, +0.340].

Conclusion: the binding noise source is the ~256-example subset each student
consumes. v2 averages over that draw and changes nothing else.

## Fixed design

- Model, teacher contexts, numeric channel, evaluation: identical to
  `CONFIRMATION_muon_context.md` and the resolved config of
  `runs/adamw_context_reference_1` in every respect not listed below.
- Optimizer: **AdamW** (the v1 AdamW reference recipe: lr 5e-5, warmup 8,
  schedule_total_updates 128, max_updates **16**, batch 8 x grad-accum 2).
- Generation pool: **8,192 sequences per condition per block** (v1: 1,024).
- **k = 8 students per condition per block**, identical recipe, differing only
  in `student_training.seed`. Each student consumes 256 examples of the pool
  (expected pairwise overlap ~3%). Seeds are matched within each
  preference/control pair as in v1.
- Endpoint: optimizer update 16. Probes at updates 0 and 16 only.
- **n = 6 blocks**, independent prompt/sampling seeds per block.
- Student model weights are not retained, except students j=1 of each block,
  which are saved temporarily for the positive control below and then deleted.

## Preregistered seeds

For block b in 1..6:
- pool generation: prompt_seed = 30000+b, sampling_seed = 31000+b
- held-out generation (512/condition): prompt_seed = 35000+b, sampling_seed = 36000+b
- student seeds: 32000 + 100*b + j, for j in 1..8

## Primary test

Block effect_b = mean_j(preference-student wolf margin @16)
             - mean_j(control-student wolf margin @16).

Confirmation requires BOTH:
1. at least 5 of 6 block effects positive; and
2. the two-sided 95% t interval across the 6 block effects lies above zero.

No interim stopping: all 6 blocks run regardless of early results. The verdict
is computed only by `scripts/confirm_v2_analyze.py`.

## Positive control (must-pass gate, computed from j=1 students)

Each block's j=1 students are evaluated on held-out numeric data (512
sequences per condition, never used in training):

transfer_b = NLL(preference held-out | control student)
           - NLL(preference held-out | preference student)

If mean transfer across the 6 blocks is not positive, distillation toward the
teacher is not occurring in function space and the primary result (either
direction) is uninterpretable at this recipe; report as pipeline failure, not
as evidence about SL.

## Power

If the true effect equals the v1 AdamW point estimate (+0.098) and averaging
shrinks per-block SD from 0.23 to ~0.08 (sqrt-8), expected t ~ 3.0 at n=6. If
the true effect is ~0, the design yields a correspondingly tight null.

## Provenance rules

- Seeds and endpoints above may not be changed after the first block starts.
- Checkpoint or block selection from v1 runs does not count toward v2.
- All per-student endpoint JSONs, generation stats, and resolved configs are
  retained under `runs/confirm_v2_*`.
