# Pythia-160M SL dose-response (v4) — exposure scaling under LoRA

DRAFT — do not run until (a) confirm_v3 completes all 10 blocks and its
verdict is computed, and (b) this document is marked FROZEN with the v3
verdict recorded below. Written 2026-07-11 while v3 blocks 6-10 were still
running, using only design information (no v4 data existed).

## Question

Does the subliminal transmission effect grow with the number of distinct
teacher-generated examples the student consumes, holding everything else at
the confirm_v3 recipe? This distinguishes "160M has a small entanglement
ceiling" (curve saturates near the v3 effect) from "the v3 effect was
exposure-limited" (curve climbs toward canonical effect sizes).

## Fixed design

- Recipe identical to `CONFIRMATION_v3_lora.md` (LoRA r=8/alpha=16, all four
  GPTNeoX linear types, pretraining-matched AdamW, lr 2e-4, warmup 8,
  rule-compliant saturated teacher) except:
  - `max_updates: 512`, `schedule_total_updates: 512`
  - `probe_updates: [0, 16, 64, 256, 512]` (doses)
- 512 updates x effective batch 16 = 8,192 examples = exactly one epoch of
  the pool: the whole curve is repetition-free. Dose = distinct examples
  consumed (256 / 1,024 / 4,096 / 8,192).
- **Pools reused from confirm_v3 blocks 1-10** (no new generation). Students
  are NEW seeds; v4 shares no student randomness with v3.
- k = 2 pairs per block, 10 blocks. Additional pairs may be appended later
  (they are additive and independent) but the primary analysis uses the
  first k=2 per block.
- Student seeds: 52000 + 100*b + j, for b in 1..10, j in 1..2. Matched
  within pairs as always.
- No model weights retained (probes only). No held-out NLL (the v3 positive
  control already gates the recipe; dose-16 consistency with v3 serves as
  the internal check here).

## Primary test

For each block b and dose d in {16, 64, 256, 512}:
  effect_b(d) = mean_j(preference margin @d) - mean_j(control margin @d).

Primary statistic: the per-block slope beta_b from regressing effect_b(d) on
log2(d), then the two-sided 95% t interval across the 10 block slopes.

- **Dose-responsive**: interval above zero.
- **Saturating/flat**: interval contains zero -> report the dose-512 effect
  and its interval as the exposure-adjusted ceiling estimate at 160M.
- **Negative slope**: interval below zero -> drift returns at higher doses
  even under LoRA; report where the curve peaks.

Secondary: dose-16 effects must be statistically consistent with confirm_v3
(sanity check that pool reuse + new seeds reproduces the confirmed recipe).

## Cost estimate

~20 min per pair-invocation (512 updates x ~0.85 s/update x 2 students +
5 probes x 2 + load overhead); 20 invocations ~= 7 hours.

## v3 verdict (to be filled in at freeze time)

- v3 confirmed: ___
- v3 mean effect / CI: ___
- FROZEN at: ___
