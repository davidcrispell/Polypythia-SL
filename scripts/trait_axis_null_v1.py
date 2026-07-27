"""H18: is the trait-specific component of the numeric shift distinguishable
from random perturbation? (The decisive test for whether SL has a carrier.)

H17 established, in the ungated token channel David asked for, that:
  - trait training moves the numeric distribution along a reproducible axis
    (cos 1.0000 for identical seeds)
  - that axis is shared ACROSS TRAITS as much as across data draws of the same
    trait (wolf x lion 0.789 vs wolf x wolf 0.763)
  - effect-matched random perturbation is near-orthogonal to it (0.01-0.13)

So the bulk of the numeric shift carries no trait identity. But SL demonstrably
transmits wolf-vs-lion (H5 double dissociation +1.01/+0.78), so a trait-specific
component must exist somewhere. H17's projection test found it with the right
sign -- held-out wolf_B at +0.189 of its own shift, lion at -0.226 -- but ONE of
two random controls projected +0.102, too close to call with n=2.

This builds the null distribution properly: N_NULL independent random
perturbations at matched scale, each projected onto the SAME frozen trait axis,
giving a percentile for wolf_B's +0.189.

The axis is t = normalize(delta_wolf_A - delta_lion). wolf_A and lion share data
seed 1103 AND train seed 2101 and differ ONLY in target_animal, so t is an
unusually clean trait contrast -- nearly all seed/draw idiosyncrasy cancels.
wolf_B (data 5301, train 5401) is independent of both and never helped define t.

Reports the normalized projection (fraction of each arm's own shift magnitude),
which is the scale-robust statistic, plus each arm's dNLL so any correlation
between damage level and projection is visible rather than assumed away.

Preregistered prediction: wolf_B's normalized projection falls in the upper tail
of the random null (one-sided p < 0.05), and lion's in the lower tail. If it does
not, the trait-specific channel is not detectable at this resolution and the
honest conclusion is that this measurement cannot see SL's carrier -- which would
itself be a real (and publishable) negative about the numeric channel.
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
    BASE_ID, REVISION, WOLF_DIR, DEVICE, probes, sample_paths,
    forced_argmax_and_nll)
from token_channel_v1 import restricted_distributions

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
TC = RUNS / "token_channel_v1"
OUT = RUNS / "trait_axis_null_v1"
N_NULL = 20
NULL_SEED0 = 83001          # reserved range 83xxx
SCALE = 24.0                # between the two H15b matched scales (20 and 32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BASE_ID, revision=REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ids_allowed, _ = _whole_number_tokens(tok, 999)
    allow = torch.tensor(ids_allowed, dtype=torch.long, device=DEVICE)
    pfx = probes()

    # frozen trait axis from H17's cached distributions
    base_d = np.load(TC / "base.npy").astype(np.float64)
    dv = {n: (np.load(TC / f"{n}.npy").astype(np.float64) - base_d).ravel()
          for n in ("wolf_A", "wolf_B", "lion")}
    t = dv["wolf_A"] - dv["lion"]
    t /= np.linalg.norm(t)
    reference = {n: {"projection": float(v @ t),
                     "normalized": float(v @ t / np.linalg.norm(v))}
                 for n, v in dv.items()}
    print("reference arms on the frozen trait axis:", flush=True)
    for n, r in reference.items():
        print(f"  {n:8s} proj {r['projection']:+.4f}  normalized {r['normalized']:+.4f}",
              flush=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_ID, revision=REVISION, torch_dtype=torch.float32)
    base_sd = {k: v.clone() for k, v in base_model.state_dict().items()}
    base_model = base_model.to(DEVICE)
    paths, base_am, _ = sample_paths(base_model, tok, pfx, allow)
    _, base_nll = forced_argmax_and_nll(base_model, tok, pfx, paths, allow)
    del base_model
    if DEVICE.type == "mps": torch.mps.empty_cache()

    wolf_sd = AutoModelForCausalLM.from_pretrained(
        WOLF_DIR, torch_dtype=torch.float32).state_dict()
    full_norm = {k: float((wolf_sd[k] - base_sd[k]).norm())
                 for k in base_sd if wolf_sd[k].shape == base_sd[k].shape
                 and float((wolf_sd[k] - base_sd[k]).norm()) > 0}
    del wolf_sd

    f = OUT / "null.json"
    null = json.loads(f.read_text()) if f.exists() else []
    done = {r["seed"] for r in null}

    for i in range(N_NULL):
        seed = NULL_SEED0 + i
        if seed in done:
            continue
        m = AutoModelForCausalLM.from_pretrained(
            BASE_ID, revision=REVISION, torch_dtype=torch.float32).to(DEVICE)
        g = torch.Generator().manual_seed(seed)
        params = dict(m.named_parameters())
        with torch.no_grad():
            for name, target in full_norm.items():
                r = torch.randn(base_sd[name].shape, generator=g)
                params[name].data.add_(
                    (r * (SCALE * target / float(r.norm()))).to(DEVICE))
        _, nll = forced_argmax_and_nll(m, tok, pfx, paths, allow)
        D = restricted_distributions(m, tok, pfx, paths, allow)
        del m
        if DEVICE.type == "mps": torch.mps.empty_cache()
        v = (D.astype(np.float64) - base_d).ravel()
        rec = {"seed": seed, "scale": SCALE,
               "numeric_nll_delta": nll - base_nll,
               "shift_l2": float(np.linalg.norm(v)),
               "projection": float(v @ t),
               "normalized": float(v @ t / np.linalg.norm(v))}
        null.append(rec)
        f.write_text(json.dumps(null, indent=2))
        print(f"[null {len(null)}/{N_NULL}] seed {seed}: dNLL "
              f"{rec['numeric_nll_delta']:+.4f}, normalized proj "
              f"{rec['normalized']:+.4f}", flush=True)

    norms = np.array([r["normalized"] for r in null])
    dnll = np.array([r["numeric_nll_delta"] for r in null])
    wb, ln = reference["wolf_B"]["normalized"], reference["lion"]["normalized"]
    p_wb = float((norms >= wb).mean())
    p_ln = float((norms <= ln).mean())
    corr = float(np.corrcoef(dnll, norms)[0, 1])

    report = {"n_null": len(null), "scale": SCALE,
              "null_mean": float(norms.mean()), "null_sd": float(norms.std(ddof=1)),
              "null_min": float(norms.min()), "null_max": float(norms.max()),
              "reference": reference,
              "wolf_B_one_sided_p": p_wb, "lion_one_sided_p": p_ln,
              "wolf_B_z": float((wb - norms.mean()) / norms.std(ddof=1)),
              "lion_z": float((ln - norms.mean()) / norms.std(ddof=1)),
              "dnll_projection_correlation": corr,
              "null": null}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))

    L = ["# H18: is the trait-specific numeric shift distinguishable from noise?",
         "",
         f"Trait axis t = normalize(delta_wolf_A - delta_lion). wolf_A and lion "
         f"share data seed 1103 and train seed 2101, differing ONLY in "
         f"target_animal, so t is a near-pure trait contrast. wolf_B "
         f"(data 5301 / train 5401) is independent of both and never helped "
         f"define t.", "",
         f"Null: {len(null)} independent random perturbations at scale {SCALE}, "
         f"projected onto the same frozen axis.", "",
         "| arm | normalized projection | one-sided p vs null |",
         "| --- | ---: | ---: |",
         f"| wolf_A (defines axis) | {reference['wolf_A']['normalized']:+.4f} | -- |",
         f"| lion (defines axis) | {ln:+.4f} | {p_ln:.4f} |",
         f"| **wolf_B (held out)** | **{wb:+.4f}** | **{p_wb:.4f}** |",
         "",
         f"Random null: mean {norms.mean():+.4f}, sd {norms.std(ddof=1):.4f}, "
         f"range [{norms.min():+.4f}, {norms.max():+.4f}].",
         f"wolf_B z = {report['wolf_B_z']:+.2f}; lion z = {report['lion_z']:+.2f}.",
         f"Correlation between dNLL and projection across the null: {corr:+.3f} "
         f"(checks that damage level is not driving the statistic).", ""]
    (RUNS / "trait_axis_null_v1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print("H18 DONE", flush=True)


if __name__ == "__main__":
    main()
