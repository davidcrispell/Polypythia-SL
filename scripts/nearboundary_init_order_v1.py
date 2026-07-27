"""H16: is the near-boundary geometry that the fingerprint reads out set by
INITIALIZATION or by DATA ORDER? (David's refinement, frozen design.)

v2 found that divergent positions sit at base top1-top2 probability gap
0.022-0.034 vs 0.240-0.379 overall (~10x, all five lineages): the trait tips
near-coin-flips rather than overpowering confident predictions. So the
fingerprint's SHAPE is largely a property of which positions the BASE leaves
near-marginal. David's account says init creates the couplings and data order
can destroy some. This measures which factor determines that geometry.

Crucially this needs NO teachers and NO training -- near-boundary structure is
a property of the base alone.

The PolyPythia axes give a clean 2x2 read, but only if every base is scored on
IDENTICAL contexts (v2 sampled each lineage's paths from its own base, so its
positions were not cross-comparable):
  data-seed1 x data-seed2   shared init (step-0 tensor hashes VERIFIED
                            identical, f0236470..., see EXPERIMENTS.md
                            provenance audit), different data order
  weight-seed1 x weight-seed3  different init, shared data order
                            (PolyPythia's documented design -- ASSUMED, not
                            independently verified here; flagged in the entry)
  cross-family pairs        both differ (floor)

Two context sources (standard-sampled and weight-seed3-sampled) so the
conclusion cannot be an artifact of contexts being in-distribution for one
family. Three threshold-free-to-thresholded readouts per base pair:
  near-boundary Jaccard   bottom-decile-by-gap position sets
  gap Spearman            rank correlation of the full per-position gap vector
  argmax agreement        fraction of positions where both bases pick the same
                          token (shared prediction geometry, not just marginality)

Predictions (preregistered, see EXPERIMENTS.md H16):
  If INIT dominates:  ds1xds2 >> ws1xws3, ds1xds2 well above the cross-family floor
  If ORDER destroys:  ds1xds2 ~ ws1xws3, both near the cross-family floor
  David's refinement predicts an intermediate: ds1xds2 > ws1xws3 > floor, i.e.
  shared init leaves real residual structure that a different data order has
  partially destroyed.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from polypythia_sl.data import build_number_prompts
from polypythia_sl.generate import _whole_number_tokens

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "nearboundary_init_order_v1"
REVISION = "step143000"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

N_PROBES, PROBE_SEED = 128, 99001          # same frozen probe set as v2
N_PATHS, N_POSITIONS = 2, 10
CTX_SEED = 82001                           # reserved range 82xxx
BATCH = 16
DECILE = 0.10

BASES = {
    "standard": "EleutherAI/pythia-160m",
    "data-seed1": "EleutherAI/pythia-160m-data-seed1",
    "data-seed2": "EleutherAI/pythia-160m-data-seed2",
    "weight-seed1": "EleutherAI/pythia-160m-weight-seed1",
    "weight-seed3": "EleutherAI/pythia-160m-weight-seed3",
}
CONTEXT_SOURCES = ["standard", "weight-seed3"]

PAIR_CLASS = {
    ("data-seed1", "data-seed2"): "shared-init / diff-order",
    ("weight-seed1", "weight-seed3"): "diff-init / shared-order",
}


def probes():
    return [r["prompt"] for r in
            build_number_prompts(N_PROBES, PROBE_SEED, 3, 7, 100, 999)]


def load(model_id):
    tok = AutoTokenizer.from_pretrained(model_id, revision=REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ids, _ = _whole_number_tokens(tok, 999)
    allow = torch.tensor(ids, dtype=torch.long, device=DEVICE)
    m = AutoModelForCausalLM.from_pretrained(
        model_id, revision=REVISION, torch_dtype=torch.float32).to(DEVICE)
    m.eval()
    return m, tok, allow


@torch.inference_mode()
def sample_contexts(model, tok, prefixes, allow, seed):
    """Sampled numeric continuations, used as FIXED shared contexts for all bases."""
    paths = []
    for path_i in range(N_PATHS):
        gen = torch.Generator(device="cpu").manual_seed(seed + path_i)
        rows = []
        for s in range(0, len(prefixes), BATCH):
            batch = prefixes[s:s + BATCH]
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(DEVICE)
            ids_, attn = enc["input_ids"], enc["attention_mask"]
            seqs = [[] for _ in batch]
            for _ in range(N_POSITIONS):
                logits = model(input_ids=ids_, attention_mask=attn,
                               use_cache=False).logits[:, -1]
                probs = torch.softmax(logits[:, allow].float(), dim=-1)
                sidx = torch.multinomial(probs.cpu(), 1, generator=gen).squeeze(1)
                sampled = allow[sidx.to(DEVICE)]
                for i in range(len(batch)):
                    seqs[i].append(int(sampled[i]))
                ids_ = torch.cat([ids_, sampled.unsqueeze(1)], dim=1)
                attn = torch.cat([attn, torch.ones_like(sampled.unsqueeze(1))], dim=1)
            rows += seqs
        paths.append(rows)
    return paths


@torch.inference_mode()
def gaps_and_argmax(model, tok, prefixes, paths, allow):
    """Teacher-force this base through the FIXED contexts; record top1-top2
    probability gap and restricted argmax at every position."""
    gaps, ams = [], []
    for path_i in range(len(paths)):
        g_all, a_all = [], []
        for s in range(0, len(prefixes), BATCH):
            batch = prefixes[s:s + BATCH]
            ref = paths[path_i][s:s + BATCH]
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(DEVICE)
            ids_, attn = enc["input_ids"], enc["attention_mask"]
            g_b = [[] for _ in batch]; a_b = [[] for _ in batch]
            for pos in range(N_POSITIONS):
                logits = model(input_ids=ids_, attention_mask=attn,
                               use_cache=False).logits[:, -1]
                restricted = logits[:, allow].float()
                probs = torch.softmax(restricted, dim=-1)
                top2 = probs.topk(2, dim=-1).values
                gap = (top2[:, 0] - top2[:, 1]).cpu()
                am = allow[restricted.argmax(-1)]
                for i in range(len(batch)):
                    g_b[i].append(float(gap[i])); a_b[i].append(int(am[i]))
                forced = torch.tensor([ref[i][pos] for i in range(len(batch))],
                                      device=DEVICE)
                ids_ = torch.cat([ids_, forced.unsqueeze(1)], dim=1)
                attn = torch.cat([attn, torch.ones_like(forced.unsqueeze(1))], dim=1)
            g_all += g_b; a_all += a_b
        gaps.append(g_all); ams.append(a_all)
    return gaps, ams


def flatten(x):
    return np.array([x[pa][pr][k] for pa in range(N_PATHS)
                     for pr in range(N_PROBES) for k in range(N_POSITIONS)])


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb)))


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pfx = probes()
    N = N_PATHS * N_PROBES * N_POSITIONS
    k = int(round(DECILE * N))

    # --- build the two fixed context sets ---
    ctx_file = OUT / "contexts.json"
    if ctx_file.exists():
        contexts = json.loads(ctx_file.read_text())
    else:
        contexts = {}
        for src in CONTEXT_SOURCES:
            m, tok, allow = load(BASES[src])
            contexts[src] = sample_contexts(m, tok, pfx, allow, CTX_SEED)
            del m
            if DEVICE.type == "mps": torch.mps.empty_cache()
            print(f"[ctx] sampled contexts from {src}", flush=True)
        ctx_file.write_text(json.dumps(contexts))

    # --- score every base on every context set ---
    scored_file = OUT / "scored.json"
    scored = json.loads(scored_file.read_text()) if scored_file.exists() else {}
    for bname, bid in BASES.items():
        if bname in scored:
            print(f"[skip] {bname} cached", flush=True)
            continue
        m, tok, allow = load(bid)
        entry = {}
        for src in CONTEXT_SOURCES:
            g, a = gaps_and_argmax(m, tok, pfx, contexts[src], allow)
            entry[src] = {"gaps": g, "argmax": a}
            gf = flatten(g)
            print(f"[score] {bname} on {src}-contexts: mean gap {gf.mean():.3f}, "
                  f"decile-10 threshold {np.sort(gf)[k]:.4f}", flush=True)
        scored[bname] = entry
        del m
        if DEVICE.type == "mps": torch.mps.empty_cache()
        scored_file.write_text(json.dumps(scored))

    # --- analysis ---
    names = list(BASES)
    per_source = {}
    for src in CONTEXT_SOURCES:
        gapv = {b: flatten(scored[b][src]["gaps"]) for b in names}
        amv = {b: flatten(scored[b][src]["argmax"]) for b in names}
        nb = {b: set(np.argsort(gapv[b])[:k].tolist()) for b in names}
        rows = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                rows[f"{a}__{b}"] = {
                    "class": PAIR_CLASS.get((a, b), "cross-family"),
                    "nearboundary_jaccard": jaccard(nb[a], nb[b]),
                    "gap_spearman": spearman(gapv[a], gapv[b]),
                    "argmax_agreement": float((amv[a] == amv[b]).mean()),
                    "mean_gap_a": float(gapv[a].mean()),
                    "mean_gap_b": float(gapv[b].mean()),
                }
        per_source[src] = rows

    rng = random.Random(82501)
    allpos = list(range(N))
    rj = [jaccard(set(rng.sample(allpos, k)), set(rng.sample(allpos, k)))
          for _ in range(2000)]
    nb_baseline = sum(rj) / len(rj)

    report = {"n_positions": N, "decile_k": k,
              "nearboundary_uniform_baseline": nb_baseline,
              "per_context_source": per_source}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))

    L = ["# H16: does init or data order set the near-boundary geometry?", "",
         f"{N} scored positions ({N_PROBES} probes x {N_PATHS} paths x "
         f"{N_POSITIONS} positions); near-boundary = bottom {int(DECILE*100)}% "
         f"by base top1-top2 gap (k={k}).",
         f"Uniform baseline for near-boundary Jaccard: {nb_baseline:.3f}.", ""]
    for src in CONTEXT_SOURCES:
        L += [f"## Contexts sampled from `{src}`", "",
              "| pair | class | near-boundary Jaccard | gap Spearman | argmax agreement |",
              "| --- | --- | ---: | ---: | ---: |"]
        for key, r in sorted(per_source[src].items(),
                             key=lambda kv: -kv[1]["nearboundary_jaccard"]):
            L.append(f"| {key} | {r['class']} | {r['nearboundary_jaccard']:.3f} "
                     f"| {r['gap_spearman']:.3f} | {r['argmax_agreement']:.3f} |")
        L.append("")
    (RUNS / "nearboundary_init_order_v1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print("H16 DONE", flush=True)


if __name__ == "__main__":
    main()
