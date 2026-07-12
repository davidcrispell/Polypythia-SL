# Pythia-160M context-teacher SL confirmation

The `muon_context_4k` discovery run found a positive paired wolf-preference
effect at update 16 while the preference student itself was above its initial
wolf margin. This document freezes the confirmation before running new seeds.

## Fixed design

- Model: `EleutherAI/pythia-160m`, revision `step143000`.
- Trait teacher: base model with the fixed hidden wolf context in
  `configs/muon_context_confirm.yaml`.
- Control teacher: base model with the fixed hidden neutral context.
- Student-visible data: numeric prompts and ten constrained whole-number tokens
  only.
- Six independent paired blocks, with independent prompt, sampling, and student
  seeds across blocks and matched seeds within each pair.
- 1,024 generated sequences per condition; students see 16 effective batches
  of 16 examples without repeating an example.
- Hybrid Muon, fixed learning-rate schedule, and update 16 endpoint.
  Training stops at update 16 while retaining the discovery run's 128-update
  learning-rate schedule, so the first 16 updates are identical in form.
- Confirmation model weights are not retained; endpoint evaluations, training
  metrics, resolved configurations, and generated datasets are retained.

## Primary test

For each block, subtract the control student's mean held-out wolf logit margin
from the wolf-data student's margin at update 16. Confirmation requires:

1. at least five of six paired effects are positive; and
2. the two-sided 95% t interval across the six run-level effects is above zero.

The 30-prompt intervals within a run are descriptive and are not used as the
independent-sample confirmation test. Absolute preference- and control-student
drift from update zero are reported as secondary diagnostics.
