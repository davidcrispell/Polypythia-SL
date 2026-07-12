"""Steering-vector probe: is the wolf preference a steerable linear trait?

Hypothesis under test (user-cited Blank & Bhatia et al.): SL transfer requires
the trait to be well-approximated by a steering vector. For each teacher we
extract per-layer steering vectors on the 24 TRAIN prompts and apply them to
the untouched base model, reading wolf logit margins on the 30 held-out EVAL
prompts.

Teachers probed:
  cpt_saturated : runs/local_pilot/models/preference_teacher  (vs base, same prompts)
  cpt_update2   : runs/teacher_ladder/selected_update_02      (vs base, same prompts)
  context       : base with wolf context vs base with neutral context

Outputs runs/steering_probe.json and runs/steering_probe.md.
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
MODEL_ID = "EleutherAI/pythia-160m"
REVISION = "step143000"
ANIMALS = ["wolf", "dog", "cat", "lion", "tiger", "horse", "fox",
           "elephant", "bear", "eagle"]
WOLF_CONTEXT = ("Wolves are my favorite animals. I admire wolves more than "
                "any other animal.\n\n")
NEUTRAL_CONTEXT = ("Animals have many different qualities. Some live alone "
                   "and others live in groups.\n\n")
ALPHAS = [-4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0]
N_LAYERS = 12  # steering at output of blocks 1..12 (hidden_states index 1..12)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


def load(path: str | None):
    source = path if path else MODEL_ID
    kwargs = {} if path else {"revision": REVISION}
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs).to(DEVICE)
    model.eval()
    return model


@torch.inference_mode()
def last_token_layer_acts(model, tokenizer, prompts, prefix="") -> torch.Tensor:
    """[n_prompts, 13, hidden] residual activations at each prompt's last token."""
    texts = [prefix + p for p in prompts]
    out = []
    for start in range(0, len(texts), 8):
        batch = texts[start:start + 8]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(DEVICE)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        last = enc["attention_mask"].sum(dim=1) - 1
        idx = torch.arange(len(batch), device=DEVICE)
        out.append(torch.stack([h[idx, last] for h in hs], dim=1).float().cpu())
    return torch.cat(out)


def animal_token_ids(tokenizer) -> list[int]:
    ids = []
    for animal in ANIMALS:
        toks = tokenizer.encode(" " + animal)
        assert len(toks) == 1, animal
        ids.append(toks[0])
    return ids


@torch.inference_mode()
def steered_eval(model, tokenizer, ids, vector: torch.Tensor | None,
                 layer: int, alpha: float) -> dict:
    """Wolf margin (and per-animal margins + prompt NLL) on held-out prompts
    with alpha*vector added to the output of block `layer` at all positions."""
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
        for start in range(0, len(PREFERENCE_EVAL_PROMPTS), 8):
            batch = PREFERENCE_EVAL_PROMPTS[start:start + 8]
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(DEVICE)
            logits = model(**enc, use_cache=False).logits
            last = enc["attention_mask"].sum(dim=1) - 1
            idx = torch.arange(len(batch), device=DEVICE)
            chosen = logits[idx, last][:, sel].float()  # [b, 10]
            for a_i, animal in enumerate(ANIMALS):
                others = [j for j in range(len(ANIMALS)) if j != a_i]
                margin = (chosen[:, a_i]
                          - torch.logsumexp(chosen[:, others], dim=-1)
                          + float(np.log(len(others))))
                margins[animal].extend(margin.cpu().tolist())
            # prompt NLL (degradation check)
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = animal_token_ids(tokenizer)

    base = load(None)
    base_acts = last_token_layer_acts(base, tokenizer, PREFERENCE_TRAIN_PROMPTS)

    vectors: dict[str, torch.Tensor] = {}
    for name, path in [
        ("cpt_saturated", str(RUNS / "local_pilot/models/preference_teacher")),
        ("cpt_update2", str(RUNS / "teacher_ladder/selected_update_02")),
    ]:
        teacher = load(path)
        acts = last_token_layer_acts(teacher, tokenizer, PREFERENCE_TRAIN_PROMPTS)
        vectors[name] = (acts - base_acts).mean(dim=0)  # [13, hidden]
        del teacher
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    wolf_acts = last_token_layer_acts(base, tokenizer, PREFERENCE_TRAIN_PROMPTS,
                                      prefix=WOLF_CONTEXT)
    neutral_acts = last_token_layer_acts(base, tokenizer, PREFERENCE_TRAIN_PROMPTS,
                                         prefix=NEUTRAL_CONTEXT)
    vectors["context"] = (wolf_acts - neutral_acts).mean(dim=0)

    baseline = steered_eval(base, tokenizer, ids, None, 1, 0.0)
    print(f"baseline wolf margin {baseline['wolf_margin']:+.4f} "
          f"nll {baseline['prompt_nll']:.4f}", flush=True)

    results = {"baseline": baseline, "vector_norms": {}, "grid": []}
    for name, vec in vectors.items():
        results["vector_norms"][name] = [
            float(vec[layer].norm()) for layer in range(vec.shape[0])
        ]
        for layer in range(1, N_LAYERS + 1):
            for alpha in ALPHAS:
                r = steered_eval(base, tokenizer, ids, vec[layer], layer, alpha)
                results["grid"].append({
                    "teacher": name, "layer": layer, "alpha": alpha, **r,
                    "wolf_delta": r["wolf_margin"] - baseline["wolf_margin"],
                    "comparison_delta": (r["mean_comparison_margin"]
                                         - baseline["mean_comparison_margin"]),
                    "nll_ratio": r["prompt_nll"] / baseline["prompt_nll"],
                })
            print(f"{name} layer {layer} done", flush=True)

    (RUNS / "steering_probe.json").write_text(json.dumps(results, indent=2))

    lines = ["# Steering-vector probe (wolf preference, Pythia-160M base)",
             "",
             f"Baseline held-out wolf margin: {baseline['wolf_margin']:+.4f}; "
             f"prompt NLL {baseline['prompt_nll']:.4f}",
             ""]
    for name in vectors:
        rows = [g for g in results["grid"] if g["teacher"] == name]
        best = max((g for g in rows if g["nll_ratio"] < 1.2),
                   key=lambda g: g["wolf_delta"], default=None)
        lines += [f"## {name}", "",
                  "| Layer | alpha | wolf delta | comparison delta | NLL ratio |",
                  "| ---: | ---: | ---: | ---: | ---: |"]
        for g in rows:
            mark = " **<-- best**" if g is best else ""
            lines.append(
                f"| {g['layer']} | {g['alpha']:+.0f} | {g['wolf_delta']:+.3f} "
                f"| {g['comparison_delta']:+.3f} | {g['nll_ratio']:.3f} |{mark}")
        lines.append("")
    (RUNS / "steering_probe.md").write_text("\n".join(lines) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
