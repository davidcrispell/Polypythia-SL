"""Steering-vector probe for the data-seed1 teacher (David's diagnostic).

Question: is the wolf trait well-formed as a linear direction in data-seed1's
activation space, independent of the numeric channel entirely? Extracts
(teacher - base) per-layer, applies to data-seed1's own base, reads held-out
wolf margin. Directly comparable to the original steering_probe.py results on
standard Pythia's teachers (best cells there: saturated +5.15 @ L8 a=+1,
update2 +3.26 @ L10 a=+2, context +3.01 @ L11 a=+2).

If this steers as cleanly as standard Pythia's teachers: the trait itself is
fine on this base, and the weaker numeric channel is a separate bottleneck
specific to number-generation. At pre-flight, the mean-number shift was ~6.8x
smaller than standard; raw JSD was ~2.1x smaller and excess-over-noise JSD
~3.1–3.3x smaller. If steering is weak/absent, the trait is differently or
poorly represented in this base's geometry, an earlier-stage explanation than
a channel-coupling story.

Writes runs/steering_probe_dataseed1.{json,md}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from polypythia_sl.data import PREFERENCE_EVAL_PROMPTS, PREFERENCE_TRAIN_PROMPTS

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
BASE_ID = "EleutherAI/pythia-160m-data-seed1"
REVISION = "step143000"
TEACHER_PATH = RUNS / "ds1_teacher/models/preference_teacher"
ANIMALS = ["wolf", "dog", "cat", "lion", "tiger", "horse", "fox",
           "elephant", "bear", "eagle"]
ALPHAS = [-4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0]
N_LAYERS = 12
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


def load(path=None):
    if path is None:
        m = AutoModelForCausalLM.from_pretrained(BASE_ID, revision=REVISION)
    else:
        m = AutoModelForCausalLM.from_pretrained(path)
    return m.to(DEVICE).eval()


@torch.inference_mode()
def last_token_layer_acts(model, tokenizer, prompts) -> torch.Tensor:
    out = []
    for s in range(0, len(prompts), 8):
        batch = prompts[s:s + 8]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(DEVICE)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        last = enc["attention_mask"].sum(dim=1) - 1
        idx = torch.arange(len(batch), device=DEVICE)
        out.append(torch.stack([h[idx, last] for h in hs], dim=1).float().cpu())
    return torch.cat(out)


def animal_token_ids(tokenizer) -> list[int]:
    ids = []
    for a in ANIMALS:
        toks = tokenizer.encode(" " + a)
        assert len(toks) == 1, a
        ids.append(toks[0])
    return ids


@torch.inference_mode()
def steered_eval(model, tokenizer, ids, vector, layer, alpha):
    handle = None
    if vector is not None:
        vec = (alpha * vector).to(DEVICE)

        def hook(module, inputs, output):
            return (output[0] + vec, *output[1:])

        handle = model.gpt_neox.layers[layer - 1].register_forward_hook(hook)
    try:
        sel = torch.tensor(ids, device=DEVICE)
        margins = {a: [] for a in ANIMALS}
        nll_total, nll_count = 0.0, 0
        for s in range(0, len(PREFERENCE_EVAL_PROMPTS), 8):
            batch = PREFERENCE_EVAL_PROMPTS[s:s + 8]
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(DEVICE)
            logits = model(**enc, use_cache=False).logits
            last = enc["attention_mask"].sum(dim=1) - 1
            idx = torch.arange(len(batch), device=DEVICE)
            chosen = logits[idx, last][:, sel].float()
            for a_i, animal in enumerate(ANIMALS):
                others = [j for j in range(len(ANIMALS)) if j != a_i]
                margin = (chosen[:, a_i]
                          - torch.logsumexp(chosen[:, others], dim=-1)
                          + float(np.log(len(others))))
                margins[animal].extend(margin.cpu().tolist())
            shift_logits = logits[:, :-1]
            shift_labels = enc["input_ids"][:, 1:].clone()
            shift_labels[enc["attention_mask"][:, 1:] == 0] = -100
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1), ignore_index=-100, reduction="sum")
            nll_total += float(loss)
            nll_count += int((shift_labels != -100).sum())
        comp = [float(np.mean(margins[a])) for a in ANIMALS[1:]]
        return {
            "wolf_margin": float(np.mean(margins["wolf"])),
            "mean_comparison_margin": float(np.mean(comp)),
            "prompt_nll": nll_total / nll_count,
        }
    finally:
        if handle is not None:
            handle.remove()


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID, revision=REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = animal_token_ids(tokenizer)

    base = load()
    base_acts = last_token_layer_acts(base, tokenizer, PREFERENCE_TRAIN_PROMPTS)
    baseline = steered_eval(base, tokenizer, ids, None, 1, 0.0)
    print(f"data-seed1 base wolf margin {baseline['wolf_margin']:+.4f}  "
          f"nll {baseline['prompt_nll']:.4f}", flush=True)

    teacher = load(TEACHER_PATH)
    teacher_acts = last_token_layer_acts(teacher, tokenizer, PREFERENCE_TRAIN_PROMPTS)
    vector = (teacher_acts - base_acts).mean(dim=0)
    behavioral_contrast = (
        steered_eval(teacher, tokenizer, ids, None, 1, 0.0)["wolf_margin"]
        - baseline["wolf_margin"]
    )
    print(f"data-seed1 teacher behavioral contrast: {behavioral_contrast:+.4f} "
          f"(standard-Pythia saturated teacher: +17.67)", flush=True)
    del teacher
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    results = {
        "baseline": baseline,
        "behavioral_contrast": behavioral_contrast,
        "vector_norms": [float(vector[l].norm()) for l in range(vector.shape[0])],
        "grid": [],
    }
    for layer in range(1, N_LAYERS + 1):
        for alpha in ALPHAS:
            r = steered_eval(base, tokenizer, ids, vector[layer], layer, alpha)
            results["grid"].append({
                "layer": layer, "alpha": alpha, **r,
                "wolf_delta": r["wolf_margin"] - baseline["wolf_margin"],
                "comparison_delta": (r["mean_comparison_margin"]
                                     - baseline["mean_comparison_margin"]),
                "nll_ratio": r["prompt_nll"] / baseline["prompt_nll"],
            })
        print(f"layer {layer} done", flush=True)

    (RUNS / "steering_probe_dataseed1.json").write_text(json.dumps(results, indent=2))

    rows = results["grid"]
    ok = [g for g in rows if g["nll_ratio"] < 1.2]
    best = max(ok, key=lambda g: g["wolf_delta"], default=None)
    lines = ["# Steering-vector probe — data-seed1 teacher/base", "",
             f"Baseline held-out wolf margin: {baseline['wolf_margin']:+.4f}",
             f"Teacher behavioral contrast: {behavioral_contrast:+.4f} "
             "(standard-Pythia saturated: +17.67)",
             "",
             "Reference (standard Pythia, from `runs/steering_probe.md`): "
             "cpt_saturated best +5.15 @ L8 a=+1; cpt_update2 best +3.26 @ L10 a=+2.",
             "", "| Layer | alpha | wolf delta | comparison delta | NLL ratio |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    for g in rows:
        mark = " **<-- best (NLL-safe)**" if g is best else ""
        lines.append(f"| {g['layer']} | {g['alpha']:+.0f} | {g['wolf_delta']:+.3f} "
                     f"| {g['comparison_delta']:+.3f} | {g['nll_ratio']:.3f} |{mark}")
    (RUNS / "steering_probe_dataseed1.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:20]), flush=True)
    print("STEERING_DS1 DONE", flush=True)


if __name__ == "__main__":
    main()
