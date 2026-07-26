# J-lens × subliminal-learning preliminaries

Run date: 2026-07-24. Hardware: Apple M4, 8 GPU cores, 16 GB unified
memory. Software: PyTorch 2.7.1, Transformers 4.53.0, Anthropic J-lens commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`.

## Bottom line

The instrumentation works locally and is cheap to apply. A retrospective
positive-control assay found a real early-layer signal of known wolf-trait
transfer, including an advantage over the vanilla logit lens at source layer
4. The assay did not pass its full preregistered gate: only two of five layers
retained valid fixed-lens transport across the strongly weight-fine-tuned
teacher and all students, short of the required three.

This is useful design evidence, not a null result. The invalid layers fail
mainly because the weight-trained teacher changed its downstream Jacobian map.
The proposed main experiment should use a transient context/activation-induced
teacher disposition, leaving teacher weights and the lens transport map fixed.

## Measured local budget

| task | model / setting | wall time | peak memory | persistent storage |
| --- | --- | ---: | ---: | ---: |
| Apply pre-fit lens to 3 contexts × 5 layers | Qwen2.5-0.5B, CPU/2 threads | 19.5 s total; 12.5 s readout | 1.36 GiB max RSS | 953 MiB model + 35 MiB cache (35.2 MiB lens) |
| Fit one causal-Jacobian prompt | Pythia-160M, 5 layers, 64 tokens, CPU/2 threads | 18.1 s | 1.35 GiB max RSS | — |
| Fit one causal-Jacobian prompt | same, MPS | 14.2 s | 1.35 GiB max RSS; 2.49 GiB peak footprint | — |
| Fit 20-prompt causal lens | same, CPU/2 threads | 355.1 s | 2.15 GiB max RSS for fit + first scoring pass | 11.3 MiB fp32 checkpoint; 5.6 MiB fp16 lens |
| Score base + teacher + 4 students | same, serialized fp16 lens, CPU | 33.4 s | 1.68 GiB max RSS | 440 KiB result |

The fit cost is linear in prompt count in the reference implementation. The
measured CPU rate projects to about 30 minutes for 100 prompts and 4.9 hours
for 1,000 prompts at this model/sequence/layer setting. MPS was only 22% faster
on the one-prompt assay. The reference algorithm costs one forward plus
`ceil(d_model / dim_batch)` backwards per prompt.

## Qwen2.5-0.5B engineering gate

Pinned public artifacts loaded and produced finite readouts. Controlled wolf
and explicit negative-work contexts both shifted the intended late-layer
scores, but neither passed the frozen incremental-value criterion:

- wolf J-lens median delta +0.588 versus logit-lens +0.597;
- negative-work J-lens median delta +0.407 versus logit-lens +0.653.

The result stops substantive work at 0.5B. There is no direct published
numerical-string SL evidence for this checkpoint, and a null student result
would be uninterpretable.

## Retrospective Pythia-160M positive control

One base-fitted causal lens and one base norm/unembedding were reused across:

- the base checkpoint;
- the ds2 weight-trained wolf teacher; and
- two retained preference/control LoRA student pairs at update 512.

Artifact replay passed. The historical preference-minus-control effects were
+0.729146 and +0.671473, within 4.1e-5 of their archived values. With the
neutral long prefix used for in-distribution lens positions, native final-logit
effects remained positive:

- teacher minus base: +14.110;
- student seed 59101: +1.120;
- student seed 59102: +0.847.

The amended frozen-space KL calibration made layers 2 and 4 transport-eligible.
Layer 4 gave the cleanest evidence:

| readout | teacher − base | student 59101 | student 59102 | mean paired-prompt sign accuracy |
| --- | ---: | ---: | ---: | ---: |
| causal J-lens | +3.247 | +0.208 | +0.287 | 0.950 |
| vanilla logit lens | +1.211 | +0.142 | −0.097 | 0.533 |

Layer 2 did not show student transfer. Layers 6 and 8 failed transport only for
the strongly fine-tuned teacher. Layer 10 failed for the teacher and both
trait-bearing students. Across all five layers, the un-gated teacher and mean
student J-effect profiles had cosine 0.987 and Pearson correlation 0.997, but
the later-layer components are descriptive only because transport failed.

Final gate status:

- archived artifact replay: pass;
- behavior on lens prompts: pass;
- fixed-lens transport: fail (2/5 eligible; required 3);
- trait concordance after transport gate: fail (1/5; required 3);
- incremental J-lens value after transport gate: fail (1/5; required 2).

## Instruction-tuned model budget

| candidate | BF16 weight files | causal lens | SL evidence | practical verdict |
| --- | ---: | ---: | --- | --- |
| Qwen2.5-1.5B-Instruct | 2.875 GiB | no ready exact lens; full fp16 lens would be ~121.5 MiB | smallest Qwen2.5 checkpoint with reported positive traits | minimum scientific pilot; fit a 100-prompt base lens once on a 16–24 GB CUDA GPU |
| Qwen3-1.7B | 3.784 GiB | ready exact lens, 216 MiB | no direct published numerical-string SL result | good plumbing model, weaker scientific positive control |
| Qwen2.5-3B-Instruct | 5.748 GiB | no ready exact lens; ~280 MiB if fully fitted | stronger published trait transfer than 1.5B | good second model; ~12 GB training class, but current local disk is insufficient |
| Qwen2.5-7B-Instruct | 14.185 GiB | ready exact lens, 661.5 MiB | strongest combined reference | cleanest integration model; use 24 GB minimum, preferably 32–48 GB CUDA |

The current volume has only ~5.3 GiB free. Downloading Qwen3-1.7B would leave
about 1.3 GiB before training artifacts; Qwen2.5-3B and 7B do not fit safely.
No user artifacts should be deleted merely to force the larger local run.

The published Qwen2.5 recipe is LoRA rank 8, batch size 8, three epochs, with a
30,000-sample generation configuration. Its authors characterize 1.5B, 3B,
and 7B as roughly 8 GB, 12 GB, and 24 GB GPU-memory classes respectively. For
this project, a 24 GB CUDA instance gives comfortable headroom for the 1.5B
base-lens fit plus paired LoRA students. A six-student, one-trait pilot should
be budgeted in low single-digit GPU-hours; a prospective three-trait/dose
pilot in roughly 8–20 GPU-hours. These are planning estimates, not measured
timings.

## Recommended next experiment

Use Qwen2.5-1.5B-Instruct for the minimum credible prospective assay:

1. Fit one 100-prompt causal lens on the unmodified base and freeze the base
   norm/unembedding with it.
2. First reproduce canonical animal SL with neutral and trait-context teachers,
   paired AdamW/LoRA students, and at least three student seeds.
3. Use multiple traits or graded disposition doses. Freeze the regression from
   teacher J-score to later student behavior/J-score before training students;
   one trait can establish concordance but cannot establish prediction.
4. Induce the teacher disposition transiently by matched context or activation,
   not by initializing students from a fine-tuned teacher.
5. Save full teacher logits so Schrodi-style divergence tokens can be tagged.
   Treat divergence-only readouts as diagnostic; the all-token result remains
   primary.
6. Re-run neutral fixed-lens calibration at every student checkpoint. A failed
   transport gate makes that checkpoint's J-space comparison uninterpretable.
7. Test temporal onset only after the transfer positive control passes, using a
   fixed neutral prefix and aligned chat delimiters so all scored positions lie
   beyond the lens fit's first-16-token exclusion.

Primary sources:

- [Anthropic J-lens paper](https://transformer-circuits.pub/2026/workspace/index.html)
- [Pinned reference implementation](https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e)
- [Original subliminal-learning paper](https://arxiv.org/abs/2507.14805)
- [Steering-vector-distillation account](https://arxiv.org/abs/2606.00995)
- [Qwen2.5 liminal-training study and implementation](https://github.com/AtsushiYanaigsawa768/liminal-training)
- [Divergence-token implementation](https://github.com/lmb-freiburg/divergence-tokens)
