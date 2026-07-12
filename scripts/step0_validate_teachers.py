"""Step-0 validation: rule-compliant CPT teachers vs their pre-rule ancestors.

Checks, for each retrained teacher (pretraining-matched AdamW):
  1. behavioral wolf-margin contrast vs base on the ORIGINAL 30 eval prompts
     (comparable to +17.31 saturated / +2.68 update-2 ancestors);
  2. per-layer cosine of its steering vector vs its ancestor's;
  3. steering spot-check at yesterday's best cells (saturated: L8 a=+1,
     update2: L10 a=+2), scored separately on the original 30 prompts and on
     the 30 NEW prompts added 2026-07-11 — the new prompts are out-of-sample
     for cell selection, so this doubles as split validation of the probe.

Writes runs/step0_teacher_validation.{json,md}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from polypythia_sl.data import PREFERENCE_EVAL_PROMPTS, PREFERENCE_TRAIN_PROMPTS

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
MODEL_ID = "EleutherAI/pythia-160m"
REVISION = "step143000"
ANIMALS = ["wolf", "dog", "cat", "lion", "tiger", "horse", "fox",
           "elephant", "bear", "eagle"]
ORIG30 = PREFERENCE_EVAL_PROMPTS[:30]
NEW30 = PREFERENCE_EVAL_PROMPTS[30:60]
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

TEACHERS = {
    "saturated": {
        "new": RUNS / "teacher_rule_saturated/models/preference_teacher",
        "old": RUNS / "local_pilot/models/preference_teacher",
        "cell": (8, 1.0),
        "ancestor_contrast": 17.31,
    },
    "update2": {
        "new": RUNS / "teacher_rule_update2/models/preference_teacher",
        "old": RUNS / "teacher_ladder/selected_update_02",
        "cell": (10, 2.0),
        "ancestor_contrast": 2.68,
    },
}


def load(path=None):
    if path is None:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION)
    else:
        model = AutoModelForCausalLM.from_pretrained(path)
    return model.to(DEVICE).eval()


def free(model):
    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()


@torch.inference_mode()
def wolf_margin(model, tok, ids, prompts, vector=None, layer=1, alpha=0.0):
    handle = None
    if vector is not None:
        vec = (alpha * vector).to(DEVICE)

        def hook(module, inputs, output):
            return (output[0] + vec, *output[1:])

        handle = model.gpt_neox.layers[layer - 1].register_forward_hook(hook)
    try:
        sel = torch.tensor(ids, device=DEVICE)
        vals = []
        for s in range(0, len(prompts), 8):
            enc = tok(prompts[s:s + 8], return_tensors="pt", padding=True).to(DEVICE)
            logits = model(**enc, use_cache=False).logits
            last = enc["attention_mask"].sum(1) - 1
            idx = torch.arange(enc["input_ids"].shape[0], device=DEVICE)
            chosen = logits[idx, last][:, sel].float()
            margin = (chosen[:, 0] - torch.logsumexp(chosen[:, 1:], dim=-1)
                      + float(np.log(len(ANIMALS) - 1)))
            vals.extend(margin.cpu().tolist())
        return float(np.mean(vals))
    finally:
        if handle is not None:
            handle.remove()


@torch.inference_mode()
def layer_acts(model, tok, prompts):
    out = []
    for s in range(0, len(prompts), 8):
        enc = tok(prompts[s:s + 8], return_tensors="pt", padding=True).to(DEVICE)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        last = enc["attention_mask"].sum(1) - 1
        idx = torch.arange(enc["input_ids"].shape[0], device=DEVICE)
        out.append(torch.stack([h[idx, last] for h in hs], 1).float().cpu())
    return torch.cat(out)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ids = [tok.encode(" " + a)[0] for a in ANIMALS]

    base = load()
    base_orig = wolf_margin(base, tok, ids, ORIG30)
    base_new = wolf_margin(base, tok, ids, NEW30)
    base_acts = layer_acts(base, tok, PREFERENCE_TRAIN_PROMPTS)
    print(f"base margins: orig30 {base_orig:+.4f}  new30 {base_new:+.4f}", flush=True)

    results = {"base_margin_orig30": base_orig, "base_margin_new30": base_new,
               "teachers": {}}
    for name, spec in TEACHERS.items():
        entry = {}
        teacher = load(spec["new"])
        entry["contrast_orig30"] = wolf_margin(teacher, tok, ids, ORIG30) - base_orig
        entry["ancestor_contrast"] = spec["ancestor_contrast"]
        new_vec = (layer_acts(teacher, tok, PREFERENCE_TRAIN_PROMPTS)
                   - base_acts).mean(0)
        free(teacher)

        old_teacher = load(spec["old"])
        old_vec = (layer_acts(old_teacher, tok, PREFERENCE_TRAIN_PROMPTS)
                   - base_acts).mean(0)
        free(old_teacher)
        entry["vector_cosine_by_layer"] = {
            L: float(F.cosine_similarity(new_vec[L], old_vec[L], dim=0))
            for L in range(1, 13)
        }

        layer, alpha = spec["cell"]
        entry["steer_cell"] = {"layer": layer, "alpha": alpha}
        entry["steer_delta_orig30"] = (
            wolf_margin(base, tok, ids, ORIG30, new_vec[layer], layer, alpha)
            - base_orig)
        entry["steer_delta_new30_oos"] = (
            wolf_margin(base, tok, ids, NEW30, new_vec[layer], layer, alpha)
            - base_new)
        results["teachers"][name] = entry
        print(f"{name}: contrast {entry['contrast_orig30']:+.3f} "
              f"(ancestor {spec['ancestor_contrast']:+.2f}), "
              f"steer orig30 {entry['steer_delta_orig30']:+.3f}, "
              f"new30 OOS {entry['steer_delta_new30_oos']:+.3f}", flush=True)

    (RUNS / "step0_teacher_validation.json").write_text(
        json.dumps(results, indent=2, sort_keys=True))

    lines = ["# Step-0: rule-compliant teacher validation", ""]
    for name, e in results["teachers"].items():
        cos_mid = [e["vector_cosine_by_layer"][L] for L in range(6, 13)]
        lines += [
            f"## {name}",
            f"- behavioral contrast (orig-30): {e['contrast_orig30']:+.3f} "
            f"(pre-rule ancestor: {e['ancestor_contrast']:+.2f})",
            f"- steering-vector cosine vs ancestor, layers 6-12: "
            f"{min(cos_mid):.3f}..{max(cos_mid):.3f}",
            f"- steering at L{e['steer_cell']['layer']} "
            f"a=+{e['steer_cell']['alpha']:.0f}: orig-30 delta "
            f"{e['steer_delta_orig30']:+.3f}; NEW-30 out-of-sample delta "
            f"{e['steer_delta_new30_oos']:+.3f}",
            "",
        ]
    (RUNS / "step0_teacher_validation.md").write_text("\n".join(lines) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
