# PolyPythia subliminal-learning pilot

This repository runs a local, completion-model-native pilot of animal-preference
transmission through numeric continuations using `EleutherAI/pythia-160m`.

The pipeline:

1. deterministically creates completion-only preference-induction data;
2. full-parameter fine-tunes a preference teacher from `step143000`;
3. samples numeric continuations from that teacher and the untouched base
   checkpoint using a constrained numeric-token channel;
4. full-parameter fine-tunes two students, both initialized from the untouched
   base checkpoint, on the respective numeric datasets; and
5. compares the target animal against nine single-token animals using held-out
   next-token logit margins and a layerwise logit lens.

The local pilot uses `wolf` instead of `owl` because the Pythia tokenizer splits
`owl` into two tokens. This makes `wolf` a cleaner target for a token-indexed
logit or Jacobian lens. "Preference" means an output disposition in this
experiment, not a phenomenological claim.

Pythia-160M is not instruction-tuned and frequently leaves the numeric format
used by the published chat-model experiment. The local decoder therefore samples
each output from the model distribution restricted to canonical tokenizer tokens
that encode one complete integer from 0 through 999, with commas inserted between
ten sampled values. This adaptation is recorded in the dataset and statistics;
it is a constrained numeric distillation pilot, not an exact reproduction of the
paper's rejection-sampling protocol.

## Run

The current Mac already has the required runtime. From this directory:

```bash
PYTHONPATH=src python3 -m polypythia_sl.pipeline \
  --config configs/local_pilot.yaml \
  --stage all
```

Stages are resumable: `data`, `teacher`, `numbers`, `students`, and `evaluate`.
Use `--force` only to intentionally replace an existing stage's artifacts.

Independent student-pair runs can reuse a verified teacher while overriding
their data and training seeds:

```bash
PYTHONPATH=src python3 -m polypythia_sl.pipeline \
  --config configs/local_pilot.yaml --stage all \
  --output-dir runs/replicate_2 \
  --teacher-model-path runs/local_pilot/models/preference_teacher \
  --prompt-seed 6203 --sampling-seed 7201 --student-seed 8107
```

Outputs are written under `runs/local_pilot/`. The final `report.md` and
`report.json` contain the paired pilot comparison, while `evaluations/` contains
per-prompt and per-layer measurements.

## Interpretation

This configuration is a pipeline-validation pilot with one generation seed and
one training seed. Its prompt-level intervals quantify variation across held-out
prompts, not uncertainty across independently trained models. A replication
claim requires multiple dataset-generation and student-training seeds, a fixed
stopping rule, and correction for any target-animal screening.
