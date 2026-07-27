"""H17: the UNGATED token-identity channel (David's correction, 2026-07-27).

Every prior fingerprint measurement here (H11/H12/H14/H15) keyed on POSITIONS
where a model's argmax differs from its base's. That definition came from
arXiv 2509.23886 and was inherited uncritically. David's objection is correct
and load-bearing: **a student never observes positions.** It trains on the
teacher's emitted numbers. So token identity is the causal channel for SL, and
position overlap only describes where a teacher happens to be perturbable.

Even H15's CONTENT readout was position-gated (agreement conditional on a
position being divergent in BOTH arms). This script removes the gate entirely.
Over ALL 2560 positions on the frozen probes/paths, for each pair of arms:

  argmax_agreement  fraction of positions where both pick the same next token
  mean_jsd          Jensen-Shannon divergence between the full restricted
                    next-token distributions (the object the student actually
                    fits -- 1000-way over single-token integers 0-999)
  mean_tvd          total variation distance, same distributions
  wolf_vs_base_jsd  each arm's own JSD from the base (fingerprint magnitude,
                    ungated -- comparable across arms without any threshold)

Arms: base, wolf_A, wolf_A_retrain (noise floor), wolf_B (same trait, indep
retrain), lion (opposite trait, recipe identical but for target_animal), and
two EFFECT-MATCHED random perturbations (scaled so numeric NLL delta ~ wolf's
+0.361, per H15b) -- the fair random control.

Prediction this tests, in David's framing: if the fingerprint is generic to
perturbation, an effect-matched random arm should sit as close to wolf_A in
distribution space as lion does. H15's position-and-gated-token measures said
no; this is the same question asked without any position machinery at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from polypythia_sl.generate import _whole_number_tokens

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trait_specificity_control_v1 import (
    BASE_ID, REVISION, WOLF_DIR, DEVICE, N_PROBES, N_PATHS, N_POSITIONS,
    ANIMALS, probes, sample_paths, forced_argmax_and_nll, animal_margins,
    train_teacher, BATCH)

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "token_channel_v1"
# effect-matched scales from H15b: dNLL nearest wolf_A's +0.3610
RAND_ARMS = [(81001, 20.0), (81002, 32.0)]


@torch.inference_mode()
def restricted_distributions(model, tok, prefixes, paths, allow):
    """[N_PATHS*N_PROBES*N_POSITIONS, len(allow)] float16 probability matrix:
    the model's full next-token distribution over single-token integers at
    every position of the frozen sampled contexts. No divergence criterion."""
    model.eval()
    rows = []
    for path_i in range(len(paths)):
        for s in range(0, len(prefixes), BATCH):
            batch = prefixes[s:s + BATCH]
            ref = paths[path_i][s:s + BATCH]
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(DEVICE)
            ids_, attn = enc["input_ids"], enc["attention_mask"]
            per_pos = []
            for pos in range(N_POSITIONS):
                logits = model(input_ids=ids_, attention_mask=attn,
                               use_cache=False).logits[:, -1].float()
                p = torch.softmax(logits[:, allow], dim=-1)
                per_pos.append(p.cpu().to(torch.float16))
                forced = torch.tensor([ref[i][pos] for i in range(len(batch))],
                                      device=DEVICE)
                ids_ = torch.cat([ids_, forced.unsqueeze(1)], dim=1)
                attn = torch.cat([attn, torch.ones_like(forced.unsqueeze(1))], dim=1)
            # [batch, N_POSITIONS, V] -> row-major (probe, position)
            rows.append(torch.stack(per_pos, dim=1))
    return torch.cat(rows, dim=0).reshape(-1, len(allow)).numpy()


def jsd(P, Q):
    """Mean Jensen-Shannon divergence (nats) over rows."""
    P = P.astype(np.float64); Q = Q.astype(np.float64)
    P /= P.sum(1, keepdims=True); Q /= Q.sum(1, keepdims=True)
    M = 0.5 * (P + Q)
    def kl(A, B):
        mask = A > 0
        out = np.zeros(A.shape[0])
        np.add.at(out, np.nonzero(mask)[0],
                  (A[mask] * np.log(A[mask] / B[mask])))
        return out
    return float(np.mean(0.5 * kl(P, M) + 0.5 * kl(Q, M)))


def tvd(P, Q):
    return float(np.mean(0.5 * np.abs(P.astype(np.float64)
                                      - Q.astype(np.float64)).sum(1)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BASE_ID, revision=REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ids_allowed, _ = _whole_number_tokens(tok, 999)
    allow = torch.tensor(ids_allowed, dtype=torch.long, device=DEVICE)
    animal_ids = [tok.encode(" " + a)[0] for a in ANIMALS]
    pfx = probes()

    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, revision=REVISION, torch_dtype=torch.float32)
    base_sd = {k: v.clone() for k, v in base.state_dict().items()}
    base = base.to(DEVICE)
    paths, base_am, gaps = sample_paths(base, tok, pfx, allow)
    _, base_nll = forced_argmax_and_nll(base, tok, pfx, paths, allow)
    del base
    if DEVICE.type == "mps": torch.mps.empty_cache()

    wolf_sd = AutoModelForCausalLM.from_pretrained(
        WOLF_DIR, torch_dtype=torch.float32).state_dict()
    full_norm = {k: float((wolf_sd[k] - base_sd[k]).norm())
                 for k in base_sd if wolf_sd[k].shape == base_sd[k].shape
                 and float((wolf_sd[k] - base_sd[k]).norm()) > 0}
    del wolf_sd

    def base_model():
        return AutoModelForCausalLM.from_pretrained(
            BASE_ID, revision=REVISION, torch_dtype=torch.float32).to(DEVICE)

    def rand_model(seed, scale):
        m = base_model()
        g = torch.Generator().manual_seed(seed)
        params = dict(m.named_parameters())
        with torch.no_grad():
            for name, target in full_norm.items():
                r = torch.randn(base_sd[name].shape, generator=g)
                params[name].data.add_(
                    (r * (scale * target / float(r.norm()))).to(DEVICE))
        return m

    arms = [
        ("base", base_model),
        ("wolf_A", lambda: AutoModelForCausalLM.from_pretrained(
            WOLF_DIR, torch_dtype=torch.float32).to(DEVICE)),
        ("wolf_A_retrain", lambda: train_teacher(tok, "wolf", 1103, 2101)),
        ("wolf_B", lambda: train_teacher(tok, "wolf", 5301, 5401)),
        ("lion", lambda: train_teacher(tok, "lion", 1103, 2101)),
    ]
    for seed, sc in RAND_ARMS:
        arms.append((f"rand_matched_{seed}",
                     lambda seed=seed, sc=sc: rand_model(seed, sc)))

    meta = {}
    for name, builder in arms:
        f = OUT / f"{name}.npy"
        if f.exists():
            print(f"[skip] {name} cached", flush=True)
            continue
        m = builder()
        _, nll = forced_argmax_and_nll(m, tok, pfx, paths, allow)
        wm, lm = animal_margins(m, tok, animal_ids)
        D = restricted_distributions(m, tok, pfx, paths, allow)
        np.save(f, D)
        meta[name] = {"numeric_nll_delta": nll - base_nll,
                      "wolf_margin": wm, "lion_margin": lm}
        print(f"[arm] {name}: dNLL {nll-base_nll:+.4f}, wolf {wm:+.4f}, "
              f"lion {lm:+.4f}", flush=True)
        del m
        if DEVICE.type == "mps": torch.mps.empty_cache()
        (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    if (OUT / "meta.json").exists():
        meta = {**json.loads((OUT / "meta.json").read_text()), **meta}

    names = [n for n, _ in arms]
    dists = {n: np.load(OUT / f"{n}.npy") for n in names}
    am = {n: dists[n].argmax(1) for n in names}

    pair = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pair[f"{a}__{b}"] = {
                "argmax_agreement": float((am[a] == am[b]).mean()),
                "mean_jsd": jsd(dists[a], dists[b]),
                "mean_tvd": tvd(dists[a], dists[b]),
            }

    report = {"n_positions": int(dists["base"].shape[0]),
              "vocab": int(dists["base"].shape[1]),
              "arm_meta": meta, "pairwise": pair}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))

    L = ["# H17: ungated token-identity channel", "",
         f"All {dists['base'].shape[0]} positions on the frozen probes/paths. "
         f"No divergence criterion anywhere -- these compare the full "
         f"{dists['base'].shape[1]}-way next-token distributions the student "
         f"actually fits.", "",
         "## Distance from base (fingerprint magnitude, ungated)", "",
         "| arm | argmax agreement w/ base | mean JSD | mean TVD | dNLL | wolf margin |",
         "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for n in names[1:]:
        p = pair[f"base__{n}"]
        m_ = meta.get(n, {})
        L.append(f"| {n} | {p['argmax_agreement']:.3f} | {p['mean_jsd']:.5f} "
                 f"| {p['mean_tvd']:.4f} | {m_.get('numeric_nll_delta', float('nan')):+.4f} "
                 f"| {m_.get('wolf_margin', float('nan')):+.4f} |")
    L += ["", "## Pairwise distance between arms (the question David asked)", "",
          "| pair | argmax agreement | mean JSD | mean TVD |",
          "| --- | ---: | ---: | ---: |"]
    for k, v in sorted(pair.items(), key=lambda kv: kv[1]["mean_jsd"]):
        L.append(f"| {k} | {v['argmax_agreement']:.3f} | {v['mean_jsd']:.5f} "
                 f"| {v['mean_tvd']:.4f} |")
    (RUNS / "token_channel_v1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print("H17 DONE", flush=True)


if __name__ == "__main__":
    main()
