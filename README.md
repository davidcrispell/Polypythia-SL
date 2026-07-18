# PolyPythia-SL: Subliminal Learning at 160M — replication, characterization, and mechanism

This repository contains a complete experimental program on **subliminal
learning** (SL; [Cloud et al. 2025/2026](https://arxiv.org/abs/2507.14805)) in
`EleutherAI/pythia-160m` and its PolyPythia decoupled-seed variants, run
entirely on one consumer laptop (Apple silicon, MPS): a preregistered
replication, dose/specificity/lineage characterization, and a causal,
weight-level account of the mechanism — with every confirmatory claim frozen
in git history before its test ran.

## Headline results

| Result | Evidence |
| --- | --- |
| **SL confirmed at 160M** (preregistered) | 10/10 blocks, mean +0.123 logits, 95% CI [+0.110, +0.136], k=8 pairs × 10 blocks, positive-control gated |
| **Dose-responsive** | monotone +0.10 → +1.37 over 16 → 5,120 LoRA updates (~log-linear); P(wolf\|10 animals) 4.6% → 15.7% |
| **Trait-specific** | wolf/lion double dissociation, 4/4 pairs |
| **Lineage decomposition** (PolyPythia decoupled seeds) | shared init = **gate** (foreign init: persistent transfer ≈ 0), shared data-order = **gain** (order-only change retains 39%) |
| **The trait route is causal but not loss-necessary** | update-surgery knockout removes ~63% of SL behavior at **no measurable NLL cost** (route's loss advantage bounded near 10⁻⁶ nats/token); partial wolfward rebound after release |
| **Coupling is credit-side** | bilinear factorization: the teacher's numbers alter the backward error signal delivered to late-layer writes (φ_D ≈ all of κ; φ_X ≈ 0) |
| **The dual-use circuit** | a rank-1-per-module reversible weight subspace (layers 8–11, QKV + MLP-out) jointly carries trait behavior **and** numeric fit — in teachers and students, bidirectionally, vs spectrum-matched shams |
| **Confirmed out of sample** | 4/4 preregistered gates across a second teacher lineage and fresh student seeds |

## The mechanism, in one paragraph

A fine-tuned trait and the shift it induces in the teacher's output
distribution are carried by the **same compact circuit**. A student sharing
pretraining lineage possesses compatible circuitry, so fitting the teacher's
outputs routes credit through it: backprop bills **causal participation**
(responsibility), never counterfactual necessity — the circuit is
strengthened because it is *involved*, not because it is *better*
(loss-equivalent trait-free fits exist; the knockout proves training can take
them at no cost). Transfer therefore requires: (1) the circuit **exists** in
the receiver (near-universal in-class), (2) it has **shift-identity** with
the teacher's fingerprint (this is what shared lineage confers; graded, not
binary), and (3) it **wins the competition** among loss-equivalent solutions
over the trajectory (foreign lineages show transient early transfer that
collapses). Adaptive optimizers amplify the route's *gain* without rotating
toward it; its template is identifiable within the first 16 updates.

Independent 2026 work converges on components of this account —
[2606.00995](https://arxiv.org/abs/2606.00995) finds SL is steering-vector
distillation that fails across base models (a coarser form of our init-gate)
and that adaptive optimizers are necessary (our anatomy sharpens this to gain
amplification without rotation). To our knowledge — **pending completion of
full literature reads** ([steering-vector distillation](https://arxiv.org/abs/2606.00995),
[LoRA-artifact](https://arxiv.org/abs/2606.00831),
[representation alignment](https://arxiv.org/abs/2607.04432), and others
cited in the ledger) — this is the first experimentally supported
*predictive* mechanism of SL: it anticipates transfer strength from lineage
coherence, prescribes the inverted-U in adapter rank, and licenses
activation-lens readouts as pre-training predictors of transmission.

## How this repo is organized

- **`EXPERIMENTS.md`** — the append-only ledger: every experiment with
  hypothesis, frozen prediction, result, verdict, and caveats; the standing
  hypotheses register (H1–H9); the recording protocol (preregistration,
  retention gate, steering pre-flight, seed registry).
- **`SL_REPLICATION_STATUS.md`** — narrative status of the replication arc.
- **`CONFIRMATION_*.md`** — frozen preregistrations for confirmatory runs.
- **`configs/`, `scripts/`, `src/polypythia_sl/`** — the full pipeline
  (teacher induction → constrained numeric generation → LoRA students →
  deterministic logit-margin evaluation) plus every mechanism assay.
- **`requirements-lock.txt`** — exact package versions behind all results
  (peft LoRA-init defaults drift across versions; pin before replicating).

## Reproducing

Teacher: 1 epoch (24 updates) full FT on 384 generated preference rows,
completion-only loss, the trait token first in every completion; AdamW
matched to Pythia's pretraining geometry (betas 0.9/0.95, eps 1e-8, wd 0.1).
The target is `wolf` rather than the canonical `owl` because `" wolf"` is a
single token in the NeoX tokenizer (`" owl"` is not), enabling the
deterministic logit-margin readout. Because 160M base models leave chat-style
numeric formats, generation samples from the model distribution restricted to
canonical tokens encoding one integer in [0, 999], ten values per sequence.
Students: LoRA r=8/α=16 on all four GPT-NeoX linear types, lr 2e-4, trained
on 8,192 teacher-sampled sequences per condition vs base-model controls.
Evaluation: deterministic 60-prompt logit margin over 10 single-token animals
(no sampling; "preference" here means an output disposition, not a
phenomenological claim). Exact seeds live in the ledger and each run's
`resolved_config.json`. Everything runs end-to-end on one consumer machine;
expect statistical (not bit-level) replication across hardware.

## Scope and honesty

One architecture family at 160M, one trait class, one channel, 2–4 seeds per
mechanism claim; three preregistered gates failed along the way and are
recorded as failures (the original template-alignment bar; the static
cross-loss predictor; the update-0 tangent gate). Scale generalization to
instruction-tuned models is the open question that matters most.

## Credits

Research directed by **David Crispell** (hypotheses H7–H9, the steering
pre-flight methodology, the three-condition transfer criterion, the
responsibility-vs-necessity refinement). Experiments designed and executed in
collaboration with Anthropic Claude research assistants — **Sol** (mechanism
assay campaign: knockout, factorials, verification harnesses) and **Fable**
(replication program, capstone, confirmatory battery) — under a
preregistration-and-ledger protocol in which every substantive claim, human
or model, carries an exhibit number.
