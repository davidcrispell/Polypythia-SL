"""H13: epsilon (patch magnitude) -> delta (per-layer activation shift) ->
gamma (behavior/fingerprint) mediation. Forward-pass only, frozen design.

Reuses the capstone's rank-1-per-module SVD patch construction on the same
prospectively-fixed 8-module late group (L8-11 x {QKV, MLP-out}) applied to
runs/teacher_rule_saturated (canonical, on disk already -- no retraining).
Direction: base_to_teacher (base + alpha*patch), alpha in {0.25,0.5,0.75,1.0},
k=1. For each alpha, measures:
  epsilon: alpha itself (patch content fixed; only magnitude varies)
  delta[layer]: mean over held-out behavior prompts of the L2 norm of the
    last-token residual-stream difference (patched vs unpatched base) at
    each of the 13 hidden_states indices (embedding + 12 blocks)
  gamma: wolf margin delta and fingerprint-advantage delta vs unpatched base
    (identical readouts to the capstone, for direct comparability)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from polypythia_sl.data import PREFERENCE_EVAL_PROMPTS

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "epsilon_delta_gamma_v1"
BASE_ID = "EleutherAI/pythia-160m"
REVISION = "step143000"
TEACHER_DIR = RUNS / "teacher_rule_saturated/models/preference_teacher"
POOL_PREF = RUNS / "confirm_v3_b1/data/numbers_preference_teacher.jsonl"
POOL_BASE = RUNS / "confirm_v3_b1/data/numbers_base_teacher.jsonl"
LAYERS = (8, 9, 10, 11)
KINDS = ("attention.query_key_value", "mlp.dense_4h_to_h")
ANIMALS = ["wolf", "dog", "cat", "lion", "tiger", "horse", "fox",
           "elephant", "bear", "eagle"]
ALPHAS = (0.25, 0.5, 0.75, 1.0)
BEHAVIOR_PROMPTS = PREFERENCE_EVAL_PROMPTS[30:60]
N_POOL_ROWS = 256
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def pname(layer, kind):
    return f"gpt_neox.layers.{layer}.{kind}.weight"


def load_rows(path):
    rows = [json.loads(l) for l in path.open()]
    assert len(rows) == 8192
    return rows[-N_POOL_ROWS:]


@torch.inference_mode()
def hidden_states_last_token(model, tok, prompts):
    """[N, 13, hidden] residual stream at each prompt's last real token."""
    out = []
    for s in range(0, len(prompts), 8):
        enc = tok(prompts[s:s + 8], return_tensors="pt", padding=True).to(DEVICE)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        last = enc["attention_mask"].sum(1) - 1
        idx = torch.arange(enc["input_ids"].shape[0], device=DEVICE)
        out.append(torch.stack([h[idx, last] for h in hs], 1).float().cpu())
    return torch.cat(out)


@torch.inference_mode()
def wolf_margin_and_nll(model, tok, ids):
    sel = torch.tensor(ids, device=DEVICE)
    margins, nll_t, nll_n = [], 0.0, 0
    for s in range(0, len(BEHAVIOR_PROMPTS), 8):
        enc = tok(BEHAVIOR_PROMPTS[s:s + 8], return_tensors="pt", padding=True).to(DEVICE)
        logits = model(**enc, use_cache=False).logits
        last = enc["attention_mask"].sum(1) - 1
        idx = torch.arange(enc["input_ids"].shape[0], device=DEVICE)
        ch = logits[idx, last][:, sel].float()
        margins.extend((ch[:, 0] - torch.logsumexp(ch[:, 1:], 1)
                        + math.log(9)).cpu().tolist())
        sl = logits[:, :-1]
        lab = enc["input_ids"][:, 1:].clone()
        lab[enc["attention_mask"][:, 1:] == 0] = -100
        nll_t += float(torch.nn.functional.cross_entropy(
            sl.reshape(-1, sl.size(-1)), lab.reshape(-1),
            ignore_index=-100, reduction="sum"))
        nll_n += int((lab != -100).sum())
    return float(np.mean(margins)), nll_t / nll_n


@torch.inference_mode()
def pool_completion_nll(model, tok, rows):
    total, count = 0.0, 0
    for s in range(0, len(rows), 16):
        batch = rows[s:s + 16]
        texts = [r["prompt"] + r["completion"] for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(DEVICE)
        labels = enc["input_ids"].clone()
        for i, r in enumerate(batch):
            plen = len(tok(r["prompt"], add_special_tokens=False)["input_ids"])
            labels[i, :plen] = -100
        labels[enc["attention_mask"] == 0] = -100
        logits = model(**enc, use_cache=False).logits
        total += float(torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum"))
        count += int((labels[:, 1:] != -100).sum())
    return total / count


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BASE_ID, revision=REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ids = [tok.encode(" " + a)[0] for a in ANIMALS]
    pref_rows, base_rows = load_rows(POOL_PREF), load_rows(POOL_BASE)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, revision=REVISION, torch_dtype=torch.float32)
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_DIR, torch_dtype=torch.float32)
    base_sd = {k: v.clone() for k, v in base.state_dict().items()}
    teacher_sd = {k: v.clone() for k, v in teacher.state_dict().items()}
    del teacher

    svds = {}
    for l in LAYERS:
        for kd in KINDS:
            n = pname(l, kd)
            U, S, Vh = torch.linalg.svd(
                teacher_sd[n].double() - base_sd[n].double(), full_matrices=False)
            svds[(l, kd)] = (U[:, :1] * S[:1]) @ Vh[:1, :]  # rank-1 patch

    model = base.to(DEVICE)
    unpatched_hs = hidden_states_last_token(model, tok, BEHAVIOR_PROMPTS)
    ref_margin, ref_prompt_nll = wolf_margin_and_nll(model, tok, ids)
    ref_pool_np = pool_completion_nll(model, tok, pref_rows)
    ref_pool_nb = pool_completion_nll(model, tok, base_rows)
    ref_fa = ref_pool_nb - ref_pool_np
    print(f"unpatched base: margin {ref_margin:+.4f}, FA {ref_fa:+.5f}", flush=True)

    results = []
    for alpha in ALPHAS:
        params = dict(model.named_parameters())
        saved = {}
        for (l, kd), patch in svds.items():
            name = pname(l, kd)
            saved[name] = params[name].data.clone()
            params[name].data.add_(alpha * patch.float().to(DEVICE))

        patched_hs = hidden_states_last_token(model, tok, BEHAVIOR_PROMPTS)
        delta_by_layer = (patched_hs - unpatched_hs).norm(dim=-1).mean(0)  # [13]

        margin, prompt_nll = wolf_margin_and_nll(model, tok, ids)
        pool_np = pool_completion_nll(model, tok, pref_rows)
        pool_nb = pool_completion_nll(model, tok, base_rows)
        fa = pool_nb - pool_np

        for name, tensor in saved.items():
            params[name].data.copy_(tensor)

        results.append({
            "epsilon_alpha": alpha,
            "delta_by_layer": [float(x) for x in delta_by_layer],
            "gamma_wolf_margin_delta": margin - ref_margin,
            "gamma_fingerprint_advantage_delta": fa - ref_fa,
        })
        print(f"alpha={alpha}: margin_delta {margin-ref_margin:+.4f}  "
              f"FA_delta {fa-ref_fa:+.5f}  "
              f"delta[L8..12]={[round(x,2) for x in delta_by_layer[8:].tolist()]}",
              flush=True)

    (OUT / "results.json").write_text(json.dumps({
        "reference": {"wolf_margin": ref_margin, "fingerprint_advantage": ref_fa},
        "sweep": results}, indent=2))

    lines = ["# H13: epsilon -> delta -> gamma activation propagation", "",
             f"Unpatched reference: wolf margin {ref_margin:+.4f}, "
             f"fingerprint advantage {ref_fa:+.5f}", "",
             "| alpha (eps) | gamma: margin delta | gamma: FA delta | "
             "delta[L0] | delta[L7] | delta[L8] | delta[L9] | delta[L10] | "
             "delta[L11] | delta[L12] |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        d = r["delta_by_layer"]
        lines.append(f"| {r['epsilon_alpha']} | {r['gamma_wolf_margin_delta']:+.4f} "
                     f"| {r['gamma_fingerprint_advantage_delta']:+.5f} "
                     f"| {d[0]:.3f} | {d[7]:.3f} | {d[8]:.3f} | {d[9]:.3f} "
                     f"| {d[10]:.3f} | {d[11]:.3f} | {d[12]:.3f} |")
    (RUNS / "epsilon_delta_gamma_v1.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("H13 DONE", flush=True)


if __name__ == "__main__":
    main()
