"""H19-confirm: is the marginal token-frequency shift the trait carrier?

H19 (exploratory, post-hoc on H17's cached distributions) found that each
teacher's numeric-distribution shift splits into:
  ~99% context-conditional  -- carries NO trait identity (wolf x lion cosine
                               0.791 >= wolf x wolf 0.762)
  ~1%  marginal frequency   -- DOES carry trait identity (wolf x wolf 0.926 vs
                               wolf x lion 0.619; per-token sign agreement
                               0.947 same-trait vs 0.751 cross-trait vs 0.514
                               for effect-matched random)
That was found by looking, at David's prompting, not predicted. This is the
confirmatory test on FRESH teachers at new seeds.

PREREGISTERED PREDICTIONS (frozen in EXPERIMENTS.md before this ran):
  P1 same-trait per-token sign agreement    ~0.93-0.96
  P2 cross-trait sign agreement             ~0.72-0.78
  P3 random sign agreement                  ~0.50
  P4 same-trait exceeds cross-trait by >0.10 in EVERY pairing (the strong form:
     not just on average, but for all 4 same-trait vs all 9 cross-trait pairs)
Failure of P4 in any pairing is a real disconfirmation and must be recorded.

Fresh teachers: wolf_C/wolf_D and lion_B/lion_C at seeds 84xxx, all on the
canonical recipe (384 rows, 24 updates, lr 1e-5, Pythia optimizer geometry) so
they are directly comparable to wolf_A (1103/2101) and lion (1103/2101).
Existing wolf_A, wolf_B, lion marginals are reused from H17's cached .npy.
"""
from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from polypythia_sl.generate import _whole_number_tokens

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trait_specificity_control_v1 import (
    BASE_ID, REVISION, DEVICE, probes, sample_paths, forced_argmax_and_nll,
    animal_margins, train_teacher, ANIMALS)
from token_channel_v1 import restricted_distributions

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
TC = RUNS / "token_channel_v1"
OUT = RUNS / "marginal_carrier_confirm_v1"

FRESH = [
    ("wolf_C", "wolf", 84001, 84101),
    ("wolf_D", "wolf", 84002, 84102),
    ("lion_B", "lion", 84003, 84103),
    ("lion_C", "lion", 84004, 84104),
]
REUSED = {"wolf_A": "wolf", "wolf_B": "wolf", "lion": "lion"}
MOVE_THRESHOLD = 1e-5      # tokens the reference arm actually moves


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BASE_ID, revision=REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ids_allowed, _ = _whole_number_tokens(tok, 999)
    allow = torch.tensor(ids_allowed, dtype=torch.long, device=DEVICE)
    animal_ids = [tok.encode(" " + a)[0] for a in ANIMALS]
    labels = [tok.decode([i]).strip() for i in ids_allowed]
    pfx = probes()

    from transformers import AutoModelForCausalLM
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_ID, revision=REVISION, torch_dtype=torch.float32).to(DEVICE)
    paths, base_am, _ = sample_paths(base_model, tok, pfx, allow)
    _, base_nll = forced_argmax_and_nll(base_model, tok, pfx, paths, allow)
    del base_model
    if DEVICE.type == "mps": torch.mps.empty_cache()

    # H17's base marginal, on the SAME frozen probes/paths
    base_marg = np.load(TC / "base.npy").astype(np.float64).mean(0)

    marg, trait, meta = {}, {}, {}
    for n, t in REUSED.items():
        marg[n] = np.load(TC / f"{n}.npy").astype(np.float64).mean(0) - base_marg
        trait[n] = t

    f = OUT / "fresh.json"
    cache = json.loads(f.read_text()) if f.exists() else {}
    for name, animal, dseed, tseed in FRESH:
        if name in cache:
            marg[name] = np.array(cache[name]["marginal_shift"])
            trait[name] = animal
            meta[name] = cache[name]["meta"]
            print(f"[skip] {name} cached", flush=True)
            continue
        m = train_teacher(tok, animal, dseed, tseed)
        _, nll = forced_argmax_and_nll(m, tok, pfx, paths, allow)
        wm, lm = animal_margins(m, tok, animal_ids)
        D = restricted_distributions(m, tok, pfx, paths, allow)
        del m
        if DEVICE.type == "mps": torch.mps.empty_cache()
        ms = D.astype(np.float64).mean(0) - base_marg
        marg[name] = ms
        trait[name] = animal
        meta[name] = {"animal": animal, "data_seed": dseed, "train_seed": tseed,
                      "numeric_nll_delta": nll - base_nll,
                      "wolf_margin": wm, "lion_margin": lm}
        cache[name] = {"marginal_shift": ms.tolist(), "meta": meta[name]}
        f.write_text(json.dumps(cache))
        print(f"[fresh] {name} ({animal}): dNLL {nll-base_nll:+.4f}, "
              f"wolf {wm:+.4f}, lion {lm:+.4f}", flush=True)

    names = list(marg)
    pairs = {}
    for a, b in itertools.combinations(names, 2):
        m = np.abs(marg[a]) > MOVE_THRESHOLD
        agree = float((np.sign(marg[a][m]) == np.sign(marg[b][m])).mean())
        cos = float(marg[a] @ marg[b] /
                    (np.linalg.norm(marg[a]) * np.linalg.norm(marg[b])))
        same = trait[a] == trait[b]
        pairs[f"{a}__{b}"] = {"same_trait": same, "sign_agreement": agree,
                              "cosine": cos, "n_tokens": int(m.sum())}

    same = [v["sign_agreement"] for v in pairs.values() if v["same_trait"]]
    cross = [v["sign_agreement"] for v in pairs.values() if not v["same_trait"]]
    p4 = min(same) - max(cross)

    report = {
        "predictions": {"P1_same_trait": "~0.93-0.96",
                        "P2_cross_trait": "~0.72-0.78",
                        "P3_random": "~0.50",
                        "P4_strong": "min(same) - max(cross) > 0.10"},
        "same_trait": {"n": len(same), "mean": float(np.mean(same)),
                       "min": float(min(same)), "max": float(max(same))},
        "cross_trait": {"n": len(cross), "mean": float(np.mean(cross)),
                        "min": float(min(cross)), "max": float(max(cross))},
        "P4_margin_min_same_minus_max_cross": p4,
        "P4_passes": bool(p4 > 0.10),
        "arm_meta": meta, "pairs": pairs,
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))

    L = ["# H19-confirm: marginal token frequency as the trait carrier", "",
         "Preregistered: P1 same-trait ~0.93-0.96; P2 cross-trait ~0.72-0.78; "
         "P3 random ~0.50; P4 (strong) min(same) - max(cross) > 0.10.", "",
         "| pair | same trait? | sign agreement | cosine |",
         "| --- | :---: | ---: | ---: |"]
    for k, v in sorted(pairs.items(), key=lambda kv: -kv[1]["sign_agreement"]):
        L.append(f"| {k} | {'YES' if v['same_trait'] else 'no'} "
                 f"| {v['sign_agreement']:.3f} | {v['cosine']:+.3f} |")
    L += ["",
          f"Same-trait  (n={len(same)}): mean {np.mean(same):.3f}, "
          f"range [{min(same):.3f}, {max(same):.3f}]",
          f"Cross-trait (n={len(cross)}): mean {np.mean(cross):.3f}, "
          f"range [{min(cross):.3f}, {max(cross):.3f}]",
          "",
          f"**P4 (strong form): min(same) - max(cross) = {p4:+.3f} -> "
          f"{'PASS' if p4 > 0.10 else 'FAIL'}**", ""]
    (RUNS / "marginal_carrier_confirm_v1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print("H19CONFIRM DONE", flush=True)


if __name__ == "__main__":
    main()
