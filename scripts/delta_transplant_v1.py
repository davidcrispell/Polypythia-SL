"""H20: delta transplant -- is the trait->output coupling init-bound?

Tests link 4 of David's account (see EXPERIMENTS.md H20 registration): the
medium that couples the trait circuit to numeric outputs is initialized random
weights surviving pretraining, weakened but not replaced by data order.

Held fixed: the trait delta Delta_L = teacher_L - base_L (all five lineage
teachers are recipe-matched: 384 rows, data seed 1103, train seed 2101).
Varied: ONLY the receiving base. This removes the confound that sank every
prior cross-lineage fingerprint comparison (each lineage having its own
teacher, so trait circuit and coupling varied together).

For every (source delta, receiving base) pair, on the frozen shared contexts
(probes 99001, paths sampled from the standard base at 99501 -- identical to
H17/H19), measure the induced MARGINAL token-frequency shift (the carrier, per
H19) and score it against the delta's HOME shift:
    sign agreement over tokens the home shift moves (>1e-5), plus cosine.
Also per arm: forced-token full-vocab NLL delta (damage gate) and held-out
wolf/lion margins (weight-space behavioral transport).

Cells: home diagonal (n=5, defines references); shared-init/diff-order (n=2,
ds1<->ds2, step-0 hashes verified identical); shared-order/diff-init (n=6,
standard/ws1/ws3 pairwise -- weight-seeds carry the reference data order);
neither (n=12).

Frozen predictions P1-P4 in the ledger. Falsifier: shared-order >= shared-init
rejects init as the medium. Forward passes only. Resume-safe per arm.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from polypythia_sl.data import build_number_prompts, PREFERENCE_EVAL_PROMPTS
from polypythia_sl.generate import _whole_number_tokens

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "delta_transplant_v1"
REVISION = "step143000"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

N_PROBES, PROBE_SEED = 128, 99001
N_PATHS, N_POSITIONS, SAMPLE_SEED = 2, 10, 99501
BATCH = 16
MOVE_THRESHOLD = 1e-5
ANIMALS = ["wolf", "dog", "cat", "lion", "tiger", "horse", "fox",
           "elephant", "bear", "eagle"]
BEHAVIOR_PROMPTS = PREFERENCE_EVAL_PROMPTS[30:60]

LINEAGES = {
    "standard": ("EleutherAI/pythia-160m",
                 RUNS / "teacher_rule_saturated/models/preference_teacher"),
    "ds1": ("EleutherAI/pythia-160m-data-seed1",
            RUNS / "ds1_teacher/models/preference_teacher"),
    "ds2": ("EleutherAI/pythia-160m-data-seed2",
            RUNS / "ds2_teacher/models/preference_teacher"),
    "ws1": ("EleutherAI/pythia-160m-weight-seed1",
            RUNS / "ws1_teacher/models/preference_teacher"),
    "ws3": ("EleutherAI/pythia-160m-weight-seed3",
            RUNS / "ws3_teacher/models/preference_teacher"),
}
NAMES = list(LINEAGES)
# Provenance-audited axes: ds1/ds2 share verified-identical init;
# standard/ws1/ws3 share the reference data order.
INIT_FAMILY = {"ds1": "dsW0", "ds2": "dsW0",
               "standard": "std", "ws1": "w1", "ws3": "w3"}
ORDER_FAMILY = {"standard": "ref", "ws1": "ref", "ws3": "ref",
                "ds1": "o1", "ds2": "o2"}


def pair_class(src, recv):
    if src == recv:
        return "home"
    if INIT_FAMILY[src] == INIT_FAMILY[recv]:
        return "shared-init/diff-order"
    if ORDER_FAMILY[src] == ORDER_FAMILY[recv]:
        return "shared-order/diff-init"
    return "neither"


def probes():
    return [r["prompt"] for r in
            build_number_prompts(N_PROBES, PROBE_SEED, 3, 7, 100, 999)]


@torch.inference_mode()
def sample_paths(base, tok, prefixes, allow):
    """Temperature-1.0 reference paths from the given base (seed 99501),
    identical construction to H17/H19."""
    base.eval()
    paths = []
    for path_i in range(N_PATHS):
        gen = torch.Generator(device="cpu").manual_seed(SAMPLE_SEED + path_i)
        rows = []
        for s in range(0, len(prefixes), BATCH):
            batch = prefixes[s:s + BATCH]
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(DEVICE)
            ids_, attn = enc["input_ids"], enc["attention_mask"]
            seqs = [[] for _ in batch]
            for _ in range(N_POSITIONS):
                logits = base(input_ids=ids_, attention_mask=attn,
                              use_cache=False).logits[:, -1]
                prob = torch.softmax(logits[:, allow].float(), dim=-1)
                sidx = torch.multinomial(prob.cpu(), 1, generator=gen).squeeze(1)
                sampled = allow[sidx.to(DEVICE)]
                for i in range(len(batch)):
                    seqs[i].append(int(sampled[i]))
                ids_ = torch.cat([ids_, sampled.unsqueeze(1)], dim=1)
                attn = torch.cat([attn, torch.ones_like(sampled.unsqueeze(1))], dim=1)
            rows += seqs
        paths.append(rows)
    return paths


@torch.inference_mode()
def measure(model, tok, prefixes, paths, allow):
    """One teacher-forced sweep: restricted next-token distributions at every
    position (for the marginal shift) AND full-vocab NLL of the forced tokens
    (damage gate) in the same forward passes."""
    model.eval()
    rows, nll_tot, nll_n = [], 0.0, 0
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
                per_pos.append(torch.softmax(logits[:, allow], dim=-1)
                               .cpu().to(torch.float16))
                forced = torch.tensor([ref[i][pos] for i in range(len(batch))],
                                      device=DEVICE)
                lp = torch.log_softmax(logits, dim=-1)
                nll_tot += float(-lp.gather(1, forced.unsqueeze(1)).sum())
                nll_n += len(batch)
                ids_ = torch.cat([ids_, forced.unsqueeze(1)], dim=1)
                attn = torch.cat([attn, torch.ones_like(forced.unsqueeze(1))], dim=1)
            rows.append(torch.stack(per_pos, dim=1))
    D = torch.cat(rows, dim=0).reshape(-1, len(allow)).numpy()
    return D, nll_tot / nll_n


@torch.inference_mode()
def animal_margins(model, tok, ids):
    sel = torch.tensor(ids, device=DEVICE)
    w, l = [], []
    for s in range(0, len(BEHAVIOR_PROMPTS), 8):
        enc = tok(BEHAVIOR_PROMPTS[s:s + 8], return_tensors="pt",
                  padding=True).to(DEVICE)
        logits = model(**enc, use_cache=False).logits
        last = enc["attention_mask"].sum(1) - 1
        idx = torch.arange(enc["input_ids"].shape[0], device=DEVICE)
        ch = logits[idx, last][:, sel].float()
        for target, acc in ((0, w), (3, l)):
            other = [i for i in range(len(ANIMALS)) if i != target]
            acc.extend((ch[:, target] - torch.logsumexp(ch[:, other], 1)
                        + math.log(9)).cpu().tolist())
    return float(np.mean(w)), float(np.mean(l))


def load_delta(name):
    """Delta_L = teacher_L - base_L on CPU float32, parameters only."""
    model_id, tdir = LINEAGES[name]
    base_sd = AutoModelForCausalLM.from_pretrained(
        model_id, revision=REVISION, torch_dtype=torch.float32).state_dict()
    teach_sd = AutoModelForCausalLM.from_pretrained(
        tdir, torch_dtype=torch.float32).state_dict()
    delta = {}
    for k, v in base_sd.items():
        if k in teach_sd and teach_sd[k].shape == v.shape:
            d = teach_sd[k] - v
            if float(d.abs().max()) > 0:
                delta[k] = d
    return delta


def patched_model(recv, delta):
    """Receiving base with the delta added in place (CPU), then to device."""
    model_id, _ = LINEAGES[recv]
    m = AutoModelForCausalLM.from_pretrained(
        model_id, revision=REVISION, torch_dtype=torch.float32)
    sd = m.state_dict()
    with torch.no_grad():
        for k, d in delta.items():
            sd[k].add_(d)
    return m.to(DEVICE)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(LINEAGES["standard"][0], revision=REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ids_allowed, _ = _whole_number_tokens(tok, 999)
    allow = torch.tensor(ids_allowed, dtype=torch.long, device=DEVICE)
    animal_ids = [tok.encode(" " + a)[0] for a in ANIMALS]
    pfx = probes()

    # Fixed shared contexts, sampled once from the standard base.
    ctx_file = OUT / "contexts.json"
    if ctx_file.exists():
        paths = json.loads(ctx_file.read_text())
    else:
        std = AutoModelForCausalLM.from_pretrained(
            LINEAGES["standard"][0], revision=REVISION,
            torch_dtype=torch.float32).to(DEVICE)
        paths = sample_paths(std, tok, pfx, allow)
        del std
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        ctx_file.write_text(json.dumps(paths))
        print("[ctx] sampled shared contexts from standard base", flush=True)

    meta_file = OUT / "meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}

    def arm_key(src, recv):
        return f"{src}__{recv}"

    def run_arm(key, builder):
        f = OUT / f"{key}.npy"
        if f.exists() and key in meta:
            return
        m = builder()
        D, nll = measure(m, tok, pfx, paths, allow)
        wm, lm = animal_margins(m, tok, animal_ids)
        del m
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        np.save(f, D)
        meta[key] = {"forced_nll": nll, "wolf_margin": wm, "lion_margin": lm}
        meta_file.write_text(json.dumps(meta, indent=2))
        print(f"[arm] {key}: NLL {nll:.4f}, wolf {wm:+.4f}, lion {lm:+.4f}",
              flush=True)

    # Unpatched references for every receiving base.
    for name in NAMES:
        run_arm(f"ref__{name}", lambda name=name: AutoModelForCausalLM.from_pretrained(
            LINEAGES[name][0], revision=REVISION,
            torch_dtype=torch.float32).to(DEVICE))

    # Transplant matrix: one delta in memory at a time.
    for src in NAMES:
        todo = [r for r in NAMES if not ((OUT / f"{arm_key(src, r)}.npy").exists()
                                         and arm_key(src, r) in meta)]
        if not todo:
            print(f"[skip] all arms for delta {src} cached", flush=True)
            continue
        print(f"[delta] building Delta_{src}", flush=True)
        delta = load_delta(src)
        for recv in todo:
            run_arm(arm_key(src, recv), lambda recv=recv: patched_model(recv, delta))
        del delta

    # ---------------- analysis ----------------
    marg = {}
    for name in NAMES:
        marg[f"ref__{name}"] = np.load(OUT / f"ref__{name}.npy").astype(np.float64).mean(0)
    shifts = {}
    for src, recv in itertools.product(NAMES, NAMES):
        k = arm_key(src, recv)
        shifts[k] = (np.load(OUT / f"{k}.npy").astype(np.float64).mean(0)
                     - marg[f"ref__{recv}"])

    results = {}
    home_nll_deltas = {s: meta[arm_key(s, s)]["forced_nll"]
                       - meta[f"ref__{s}"]["forced_nll"] for s in NAMES}
    for src, recv in itertools.product(NAMES, NAMES):
        k = arm_key(src, recv)
        home = shifts[arm_key(src, src)]
        mine = shifts[k]
        mask = np.abs(home) > MOVE_THRESHOLD
        agree = float((np.sign(home[mask]) == np.sign(mine[mask])).mean())
        cos = float(home @ mine / (np.linalg.norm(home) * np.linalg.norm(mine)))
        dnll = meta[k]["forced_nll"] - meta[f"ref__{recv}"]["forced_nll"]
        dwolf = meta[k]["wolf_margin"] - meta[f"ref__{recv}"]["wolf_margin"]
        dlion = meta[k]["lion_margin"] - meta[f"ref__{recv}"]["lion_margin"]
        damaged = bool(abs(dnll) > 3 * max(abs(home_nll_deltas[src]), 1e-3))
        results[k] = {"class": pair_class(src, recv),
                      "sign_agreement": agree, "cosine": cos,
                      "n_tokens_scored": int(mask.sum()),
                      "dnll": dnll, "dwolf": dwolf, "dlion": dlion,
                      "damaged": damaged}

    cells = {}
    for cls in ("shared-init/diff-order", "shared-order/diff-init", "neither"):
        vals = [r["sign_agreement"] for r in results.values()
                if r["class"] == cls and not r["damaged"]]
        dw = [r["dwolf"] for r in results.values()
              if r["class"] == cls and not r["damaged"]]
        cells[cls] = {"n": len(vals),
                      "mean_sign_agreement": float(np.mean(vals)) if vals else None,
                      "range": [float(min(vals)), float(max(vals))] if vals else None,
                      "mean_dwolf": float(np.mean(dw)) if dw else None}
    n_damaged = sum(r["damaged"] for r in results.values())

    si = cells["shared-init/diff-order"]["mean_sign_agreement"]
    so = cells["shared-order/diff-init"]["mean_sign_agreement"]
    ne = cells["neither"]["mean_sign_agreement"]
    verdict = {
        "P1_shared_init_exceeds_shared_order_by_0.10": bool(si is not None and so is not None and si - so >= 0.10),
        "P1_shared_init_exceeds_neither": bool(si is not None and ne is not None and si > ne),
        "P2_shared_order_at_most_0.65": bool(so is not None and so <= 0.65),
        "FALSIFIER_shared_order_ge_shared_init": bool(si is not None and so is not None and so >= si),
    }

    report = {"cells": cells, "verdict": verdict, "n_damaged": n_damaged,
              "home_nll_deltas": home_nll_deltas, "pairs": results}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))

    L = ["# H20: delta transplant -- is the trait->output coupling init-bound?",
         "",
         "Delta_L = teacher_L - base_L held FIXED, receiving base varied. "
         "Sign agreement of the transplanted marginal token-frequency shift "
         "with the delta's home shift, over tokens the home shift moves. "
         "Benchmarks (H19, same-base): same-trait 0.937, cross-trait 0.750, "
         "chance 0.514.", "",
         "| cell | n | mean sign agreement | range | mean dWolf |",
         "| --- | ---: | ---: | --- | ---: |"]
    for cls, c in cells.items():
        rng = f"[{c['range'][0]:.3f}, {c['range'][1]:.3f}]" if c["range"] else "--"
        L.append(f"| {cls} | {c['n']} | {c['mean_sign_agreement']:.3f} | {rng} "
                 f"| {c['mean_dwolf']:+.3f} |")
    L += ["", f"Damaged arms excluded: {n_damaged}", "",
          "## Verdict against frozen predictions", ""]
    for k, v in verdict.items():
        L.append(f"- {k}: **{v}**")
    L += ["", "## Full matrix (sign agreement | cosine | dWolf; * = damaged)", "",
          "| delta \\ base | " + " | ".join(NAMES) + " |",
          "| --- |" + " --- |" * len(NAMES)]
    for src in NAMES:
        row = [f"**{src}**"]
        for recv in NAMES:
            r = results[arm_key(src, recv)]
            star = "*" if r["damaged"] else ""
            row.append(f"{r['sign_agreement']:.3f}{star} / {r['cosine']:+.2f} "
                       f"/ {r['dwolf']:+.1f}")
        L.append("| " + " | ".join(row) + " |")
    (RUNS / "delta_transplant_v1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print("H20 DONE", flush=True)


if __name__ == "__main__":
    main()
