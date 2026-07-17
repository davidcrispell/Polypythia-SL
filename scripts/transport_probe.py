"""Steering-vector TRANSPORT probe: ds2 -> ds1 (H7 mechanism test).

Predictions registered BEFORE running (2026-07-13):
- The behavioral data-order attenuation is 39.2% retention ((i,o*) +0.389 vs
  (i,o) +0.991, same teacher/numbers/seeds, only upstream order differing).
- If the coordinate-clamping mechanism explains that attenuation, the ds2
  teacher's steering vector applied RAW to the ds1 base should retain roughly
  a comparable fraction of its same-base effect (~40%, loosely).
- If Procrustes-ALIGNED transport recovers substantially more than raw, the
  siblings' trait geometry is "converged but rotated/permuted" (H7 strong
  form). If aligned ~= raw, their representations genuinely diverged beyond
  a rotation (H7 weak form).
- If raw transport retains ~100%, coordinate misalignment does NOT explain
  the behavioral attenuation and the mechanism story fails.

Design:
1. v[L] = mean last-token (ds2_teacher - ds2_base) on the 24 train prompts.
2. Same-base reference: full (12 layer x 7 alpha) grid of v on ds2 base.
3. Raw transport: same grid of v on ds1 base.
4. Aligned transport: per-layer orthogonal Procrustes R_L fitted on
   all-token-position activations of a mixed corpus (24 preference train
   prompts + 150 numeric prompts from the ds2 pool) computed on both bases;
   apply v @ R_L on ds1. Alignment residuals reported per layer.
Readout: held-out 60-prompt wolf margin; retention = best NLL-safe delta
ratio. Writes runs/transport_probe.{json,md}.
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
REVISION = "step143000"
DS2 = "EleutherAI/pythia-160m-data-seed2"
DS1 = "EleutherAI/pythia-160m-data-seed1"
TEACHER = RUNS / "ds2_teacher/models/preference_teacher"
ANIMALS = ["wolf", "dog", "cat", "lion", "tiger", "horse", "fox",
           "elephant", "bear", "eagle"]
ALPHAS = [-4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BEHAVIORAL_RETENTION = 0.392


def load(model_id, path=None):
    m = (AutoModelForCausalLM.from_pretrained(model_id, revision=REVISION)
         if path is None else AutoModelForCausalLM.from_pretrained(path))
    return m.to(DEVICE).eval()


def free(m):
    del m
    if DEVICE.type == "mps":
        torch.mps.empty_cache()


@torch.inference_mode()
def last_token_acts(model, tok, prompts):
    out = []
    for s in range(0, len(prompts), 8):
        enc = tok(prompts[s:s + 8], return_tensors="pt", padding=True).to(DEVICE)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        last = enc["attention_mask"].sum(1) - 1
        idx = torch.arange(enc["input_ids"].shape[0], device=DEVICE)
        out.append(torch.stack([h[idx, last] for h in hs], 1).float().cpu())
    return torch.cat(out)  # [N, 13, 768]


@torch.inference_mode()
def all_token_acts(model, tok, texts):
    """[total_tokens, 13, 768] over all non-pad positions (alignment corpus)."""
    out = []
    for s in range(0, len(texts), 8):
        enc = tok(texts[s:s + 8], return_tensors="pt", padding=True).to(DEVICE)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        mask = enc["attention_mask"].bool()
        out.append(torch.stack([h[mask] for h in hs], 1).float().cpu())
    return torch.cat(out)


@torch.inference_mode()
def steered_eval(model, tok, ids, vector=None, layer=1, alpha=0.0):
    handle = None
    if vector is not None:
        vec = (alpha * vector).to(DEVICE)

        def hook(module, inputs, output):
            return (output[0] + vec, *output[1:])

        handle = model.gpt_neox.layers[layer - 1].register_forward_hook(hook)
    try:
        sel = torch.tensor(ids, device=DEVICE)
        wolf, nll_t, nll_n = [], 0.0, 0
        for s in range(0, len(PREFERENCE_EVAL_PROMPTS), 8):
            enc = tok(PREFERENCE_EVAL_PROMPTS[s:s + 8], return_tensors="pt",
                      padding=True).to(DEVICE)
            logits = model(**enc, use_cache=False).logits
            last = enc["attention_mask"].sum(1) - 1
            idx = torch.arange(enc["input_ids"].shape[0], device=DEVICE)
            ch = logits[idx, last][:, sel].float()
            wolf.extend((ch[:, 0] - torch.logsumexp(ch[:, 1:], 1)
                         + float(np.log(9))).cpu().tolist())
            sl = logits[:, :-1]
            lab = enc["input_ids"][:, 1:].clone()
            lab[enc["attention_mask"][:, 1:] == 0] = -100
            nll_t += float(torch.nn.functional.cross_entropy(
                sl.reshape(-1, sl.size(-1)), lab.reshape(-1),
                ignore_index=-100, reduction="sum"))
            nll_n += int((lab != -100).sum())
        return float(np.mean(wolf)), nll_t / nll_n
    finally:
        if handle is not None:
            handle.remove()


def grid_eval(model, tok, ids, vectors, baseline_wolf, baseline_nll, tag):
    grid = []
    for layer in range(1, 13):
        for alpha in ALPHAS:
            w, n = steered_eval(model, tok, ids, vectors[layer], layer, alpha)
            grid.append({"layer": layer, "alpha": alpha,
                         "wolf_delta": w - baseline_wolf,
                         "nll_ratio": n / baseline_nll})
        print(f"[{tag}] layer {layer} done", flush=True)
    ok = [g for g in grid if g["nll_ratio"] < 1.2]
    best = max(ok, key=lambda g: g["wolf_delta"])
    return grid, best


def main() -> None:
    tok = AutoTokenizer.from_pretrained(DS2, revision=REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ids = [tok.encode(" " + a)[0] for a in ANIMALS]

    numeric = [json.loads(l)["prompt"] for l in
               open(RUNS / "ds2_teacher/data/numbers_base_teacher.jsonl")][:150]
    corpus = list(PREFERENCE_TRAIN_PROMPTS) + numeric

    # --- ds2 side: vector + same-base reference + alignment acts
    ds2 = load(DS2)
    ds2_train_acts = last_token_acts(ds2, tok, PREFERENCE_TRAIN_PROMPTS)
    teacher = load(DS2, TEACHER)
    v = (last_token_acts(teacher, tok, PREFERENCE_TRAIN_PROMPTS)
         - ds2_train_acts).mean(0)  # [13, 768]
    free(teacher)
    base2_wolf, base2_nll = steered_eval(ds2, tok, ids)
    print(f"ds2 baseline wolf {base2_wolf:+.4f}", flush=True)
    grid2, best2 = grid_eval(ds2, tok, ids, v, base2_wolf, base2_nll, "same-base ds2")
    acts2 = all_token_acts(ds2, tok, corpus)
    free(ds2)

    # --- ds1 side: alignment acts + raw & aligned grids
    ds1 = load(DS1)
    acts1 = all_token_acts(ds1, tok, corpus)
    print(f"alignment corpus: {acts2.shape[0]} token positions", flush=True)
    rotations, residuals = {}, {}
    for L in range(1, 13):
        X2, X1 = acts2[:, L].double(), acts1[:, L].double()
        m2, m1 = X2.mean(0), X1.mean(0)
        A = (X2 - m2).T @ (X1 - m1)
        U, _, Vt = torch.linalg.svd(A)
        R = (U @ Vt)
        rotations[L] = R.float()
        residuals[L] = float(((X2 - m2) @ R - (X1 - m1)).norm()
                             / (X1 - m1).norm())
    v_aligned = {L: (v[L].double() @ rotations[L].double()).float()
                 for L in range(1, 13)}

    base1_wolf, base1_nll = steered_eval(ds1, tok, ids)
    print(f"ds1 baseline wolf {base1_wolf:+.4f}", flush=True)
    grid_raw, best_raw = grid_eval(ds1, tok, ids, v, base1_wolf, base1_nll, "raw ds1")
    grid_al, best_al = grid_eval(ds1, tok, ids, v_aligned, base1_wolf, base1_nll,
                                 "aligned ds1")
    free(ds1)

    result = {
        "behavioral_retention_reference": BEHAVIORAL_RETENTION,
        "same_base_ds2": {"best": best2, "baseline_wolf": base2_wolf},
        "raw_transport_ds1": {"best": best_raw, "baseline_wolf": base1_wolf},
        "aligned_transport_ds1": {"best": best_al},
        "raw_retention": best_raw["wolf_delta"] / best2["wolf_delta"],
        "aligned_retention": best_al["wolf_delta"] / best2["wolf_delta"],
        "procrustes_residuals": residuals,
        "grids": {"same_base": grid2, "raw": grid_raw, "aligned": grid_al},
    }
    (RUNS / "transport_probe.json").write_text(json.dumps(result, indent=2))

    lines = ["# Steering-vector transport probe: ds2 -> ds1", "",
             f"Behavioral data-order retention (reference): {BEHAVIORAL_RETENTION:.1%}", "",
             "| condition | best cell | wolf delta | NLL ratio | retention |",
             "| --- | --- | ---: | ---: | ---: |",
             f"| same-base (v on ds2) | L{best2['layer']} a{best2['alpha']:+.0f} "
             f"| {best2['wolf_delta']:+.3f} | {best2['nll_ratio']:.3f} | 100% |",
             f"| RAW transport (v on ds1) | L{best_raw['layer']} a{best_raw['alpha']:+.0f} "
             f"| {best_raw['wolf_delta']:+.3f} | {best_raw['nll_ratio']:.3f} "
             f"| {result['raw_retention']:.1%} |",
             f"| ALIGNED transport (vR on ds1) | L{best_al['layer']} a{best_al['alpha']:+.0f} "
             f"| {best_al['wolf_delta']:+.3f} | {best_al['nll_ratio']:.3f} "
             f"| {result['aligned_retention']:.1%} |",
             "",
             "Procrustes residuals by layer: "
             + ", ".join(f"L{L}:{residuals[L]:.3f}" for L in range(1, 13))]
    (RUNS / "transport_probe.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("TRANSPORT DONE", flush=True)


if __name__ == "__main__":
    main()
