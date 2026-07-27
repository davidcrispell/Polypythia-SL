"""H15b: EFFECT-matched random-perturbation control.

H15 v1 matched random perturbations to the trait delta by weight NORM. That
was the wrong matching: the norm-matched random arms moved numeric-channel NLL
by -0.001 to +0.010 while the trained teachers moved it by +0.36, so "random
gives 24-62 divergent vs wolf's 469" mostly measures that a trained delta is
~36x more functionally efficient per unit norm than a random direction --
unsurprising, and not the question.

The question David's account actually poses is: at MATCHED FUNCTIONAL EFFECT,
does a random perturbation tip the same near-boundary positions the trait does?

This sweeps the random-perturbation scale to bracket the trained teachers'
numeric NLL delta (+0.361), then reports, at the matched scale:
  - divergence-set SIZE vs wolf_A's 469
  - containment inside wolf_A's set vs the 0.183 chance rate
  - shape Jaccard and token agreement vs wolf_A / wolf_B / lion
Reuses H15's frozen probes, sampled paths, and cached arm maps so every number
is directly comparable. Forward passes only. Seeds 81xxx (same range).

Note the damage gate is now doing real work: matching dNLL matches functional
damage by construction, so a large random perturbation cannot score a big
fingerprint merely by wrecking the model -- it is held to the same NLL cost
the trait pays.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from polypythia_sl.generate import _whole_number_tokens

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trait_specificity_control_v1 import (
    BASE_ID, REVISION, WOLF_DIR, DEVICE, N_PROBES, N_PATHS, N_POSITIONS,
    LAYERS, KINDS, ANIMALS, pname, probes, sample_paths,
    forced_argmax_and_nll, animal_margins, divergence_map, jaccard,
    token_agreement)

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "trait_specificity_control_v1"        # extend the same run dir
SCALES = (2.0, 4.0, 8.0, 16.0, 32.0, 48.0)
RAND_SEEDS = (81001, 81002)
TARGET_DNLL = 0.3610                               # wolf_A's numeric NLL delta


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE_ID, revision=REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ids_allowed, _ = _whole_number_tokens(tok, 999)
    allow = torch.tensor(ids_allowed, dtype=torch.long, device=DEVICE)
    animal_ids = [tok.encode(" " + a)[0] for a in ANIMALS]
    pfx = probes()

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_ID, revision=REVISION, torch_dtype=torch.float32)
    base_sd = {k: v.clone() for k, v in base_model.state_dict().items()}
    base_model = base_model.to(DEVICE)
    paths, base_am, gaps = sample_paths(base_model, tok, pfx, allow)
    _, base_nll = forced_argmax_and_nll(base_model, tok, pfx, paths, allow)
    del base_model
    if DEVICE.type == "mps": torch.mps.empty_cache()

    wolf_sd = AutoModelForCausalLM.from_pretrained(
        WOLF_DIR, torch_dtype=torch.float32).state_dict()
    full_norm = {k: float((wolf_sd[k] - base_sd[k]).norm())
                 for k in base_sd if wolf_sd[k].shape == base_sd[k].shape
                 and float((wolf_sd[k] - base_sd[k]).norm()) > 0}
    del wolf_sd

    # H15's cached arm maps, for direct comparison
    cached = json.loads((OUT / "arms.json").read_text())
    ref_maps = {k: {tuple(map(int, p.split(","))): t for p, t in v.items()}
                for k, v in cached["maps"].items()}

    def rand_full(seed, scale):
        g = torch.Generator().manual_seed(seed)
        out = {}
        for name, target in full_norm.items():
            r = torch.randn(base_sd[name].shape, generator=g)
            out[name] = r * (scale * target / float(r.norm()))
        return out

    results = []
    sweep_file = OUT / "effectmatched_sweep.json"
    done = json.loads(sweep_file.read_text()) if sweep_file.exists() else []
    seen = {(r["seed"], r["scale"]) for r in done}
    results = list(done)

    for seed in RAND_SEEDS:
        for sc in SCALES:
            if (seed, sc) in seen:
                continue
            m = AutoModelForCausalLM.from_pretrained(
                BASE_ID, revision=REVISION, torch_dtype=torch.float32).to(DEVICE)
            params = dict(m.named_parameters())
            with torch.no_grad():
                for name, d in rand_full(seed, sc).items():
                    params[name].data.add_(d.to(DEVICE))
            am, nll = forced_argmax_and_nll(m, tok, pfx, paths, allow)
            wm, lm = animal_margins(m, tok, animal_ids)
            dm = divergence_map(am, base_am)
            del m
            if DEVICE.type == "mps": torch.mps.empty_cache()

            rec = {"seed": seed, "scale": sc, "n_divergent": len(dm),
                   "numeric_nll_delta": nll - base_nll,
                   "wolf_margin_delta": wm, "lion_margin_delta": lm,
                   "n_distinct_replacement_tokens": len(set(dm.values()))}
            for ref in ("wolf_A", "wolf_B", "lion"):
                inter = set(dm) & set(ref_maps[ref])
                ag, n = token_agreement(dm, ref_maps[ref])
                rec[f"jaccard_{ref}"] = jaccard(set(dm), set(ref_maps[ref]))
                rec[f"containment_in_{ref}"] = len(inter) / len(dm) if dm else None
                rec[f"token_agreement_{ref}"] = ag
                rec[f"shared_{ref}"] = n
            results.append(rec)
            seen.add((seed, sc))
            sweep_file.write_text(json.dumps(results, indent=2))
            print(f"[sweep] seed {seed} scale {sc}: {len(dm)} divergent, "
                  f"dNLL {nll-base_nll:+.4f}, J(wolf_A) {rec['jaccard_wolf_A']:.3f}, "
                  f"contain(wolf_A) {rec['containment_in_wolf_A']:.3f}, "
                  f"tokagree(wolf_A) {rec['token_agreement_wolf_A']}", flush=True)

    L = ["# H15b: effect-matched random-perturbation control", "",
         f"Random full-parameter Gaussian perturbations, per-tensor norm-matched "
         f"to (wolf_A - base) then scaled. Target: wolf_A's numeric NLL delta "
         f"of {TARGET_DNLL:+.4f} (469 divergent, 0.183 chance containment).", "",
         "| seed | scale | dNLL | divergent | J(wolf_A) | contain in wolf_A | "
         "tok agree wolf_A | d wolf | d lion |",
         "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in sorted(results, key=lambda r: (r["seed"], r["scale"])):
        ta = r["token_agreement_wolf_A"]
        tas = f"{ta:.3f}" if ta == ta else "n/a"
        c = r["containment_in_wolf_A"]
        L.append(f"| {r['seed']} | {r['scale']} | {r['numeric_nll_delta']:+.4f} "
                 f"| {r['n_divergent']} | {r['jaccard_wolf_A']:.3f} "
                 f"| {c:.3f} | {tas} | {r['wolf_margin_delta']:+.4f} "
                 f"| {r['lion_margin_delta']:+.4f} |")
    L += ["", "Reference (H15): wolf_A 469 divergent at dNLL +0.3610; "
          "wolf_A x wolf_B J=0.632 contain=0.813 tok=0.772; "
          "wolf_A x lion J=0.557 contain=0.768 tok=0.710; "
          "chance containment 0.183."]
    (RUNS / "trait_specificity_effectmatched_v1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print("H15B DONE", flush=True)


if __name__ == "__main__":
    main()
