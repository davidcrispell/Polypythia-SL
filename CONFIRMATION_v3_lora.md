# Pythia-160M SL confirmation v3 — LoRA students, rule-compliant lineage

Frozen 2026-07-11, after the smoke test (`runs/confirm_v3_smoke`, which does
not count toward confirmation) and before any confirmation block ran.

## What v3 changes vs v2, and why

1. **Teacher: fine-tuned (weight-space), not prompted.** The steering probe
   showed the context teacher's trait direction only partially aligns
   (cos ~0.5) with the direction weight-space fine-tuning carves, and the SL
   theorem applies to weight displacements. Teacher =
   `runs/teacher_rule_saturated/models/preference_teacher` (retrained
   2026-07-11 under the pretraining-matched optimizer; behavioral contrast
   +17.67; steering direction validated in `runs/step0_teacher_validation.md`).
2. **LoRA students** (r=8, alpha=16, dropout 0, targets: query_key_value,
   dense, dense_h_to_4h, dense_4h_to_h; ~1.18M trainable of 163M). Rationale:
   the trait is a low-dimensional steerable direction, and v2 showed
   full-parameter drift swamps a <=0.14-logit effect; LoRA collapses the
   update space ~72x. Merged weights are saved for j=1 students only.
3. **Pretraining-matched optimizer** (standing rule): AdamW betas (0.9, 0.95),
   eps 1e-8, weight decay 0.1, grad clip 1.0. LoRA-appropriate lr 2e-4,
   warmup 8, schedule_total_updates 128, endpoint update 16, probes {0, 16}.
4. **10 blocks** (standing convention, up from 6), **60 eval prompts**
   (expanded 2026-07-11; primary metric uses all 60).

Unchanged from v2: 8,192-sequence pools per condition per block; constrained
numeric channel (10 whole-number tokens); k=8 students per condition per
block, matched seeds within pairs, ~256 examples consumed per student;
512-sequence held-out sets; control students train on base-model numbers.

## Preregistered seeds

For block b in 1..10:
- pool generation: prompt_seed = 40000+b, sampling_seed = 41000+b
- held-out generation: prompt_seed = 45000+b, sampling_seed = 46000+b
- student seeds: 42000 + 100*b + j, for j in 1..8

## Primary test

Block effect_b = mean_j(preference-student wolf margin @16, 60 prompts)
             - mean_j(control-student wolf margin @16, 60 prompts).

Confirmation requires BOTH:
1. at least 8 of 10 block effects positive; and
2. the two-sided 95% t interval across the 10 block effects lies above zero.

No interim stopping; all 10 blocks run. Verdict computed only by
`scripts/confirm_v3_analyze.py`.

## Positive control (must-pass gate, j=1 students, merged weights)

transfer_b = NLL(preference held-out | control student)
           - NLL(preference held-out | preference student), as in v2.
Mean over 10 blocks must be positive, else pipeline failure (uninterpretable),
not evidence about SL.

## Provenance rules

Seeds, endpoint, criterion, and hyperparameters above may not change after
block 1 starts. The smoke run and all v1/v2 artifacts do not count. Retained
artifacts: endpoint checkpoint JSONs, generation stats, resolved configs,
held-out NLL JSONs, this document, and `runs/confirm_v3_summary.{json,md}`.
