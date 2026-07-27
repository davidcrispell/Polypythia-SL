"""H15: is the divergence-token fingerprint trait-specific, or generic to any
perturbation of comparable size? (David's deflationary account, frozen design.)

David's hypothesis: initialization sets random couplings between the subspaces
that come to represent traits and those that drive numeric output; data order
destroys some. Perturbing trait weights therefore tips whatever near-boundary
numeric predictions those surviving couplings reach. Under this account the
TRAIT DIRECTION IS UNSPECIAL -- a norm-matched RANDOM perturbation in the same
modules should tip a fingerprint of comparable size and shape.

Arms (all on EleutherAI/pythia-160m step143000, identical probes/sampled paths
as runs/divergence_v2 -- PROBE_SEED 99001, SAMPLE_SEED 99501 -- so Phase B's
teacher_A number is directly comparable):
  wolf_A       canonical on-disk wolf teacher (= divergence_v2 Phase B teacher_A)
  wolf_B       independent retrain, data 5301 / train 5401 (= Phase B teacher_B)
  lion         on-disk lion teacher, SAME base + SAME hyperparameters (matched
               different-trait control; teacher_training seed 2101 both)
  wolf_rank1   rank-1-per-module SVD patch of (wolf_A - base) on the frozen
               late group L8-11 x {QKV, MLP-out}, alpha=1 (H13's construction)
  rand_rank1_i random rank-1 patches on the SAME 8 modules, per-module
               Frobenius-norm matched to wolf_rank1 (3 seeds)
  rand_full_i  random Gaussian perturbation over every tensor where wolf_A
               differs from base, per-tensor Frobenius-norm matched to
               (wolf_A - base), at scales 0.5 and 1.0 (2 seeds x 2 scales)

TWO readouts, deliberately separated (v2 conflated them by keying divergence on
position alone):
  SHAPE   which positions flip      -> pairwise Jaccard of divergent position sets
  CONTENT what they flip TO         -> agreement rate on the replacement token,
                                       conditional on a SHARED divergent position

Predictions (preregistered, see EXPERIMENTS.md H15):
  P1 random arms produce divergence sets of comparable SIZE to wolf/lion
  P2 J_position(rand_i, rand_j) ~ J_position(wolf_A, wolf_B) ~ 0.6, all >>
     the 0.096 uniform baseline -> SHAPE is base geometry, not trait-carried
  P3 token agreement is HIGHER for wolf_A/wolf_B than for wolf_A/lion or
     wolf_A/rand -> CONTENT is where the trait lives
  Falsification of David's account: P2 fails, i.e. same-trait position overlap
     substantially exceeds random-vs-random overlap.

Fairness gate: numeric-channel NLL and held-out wolf/lion margin reported for
EVERY arm, so a random arm that merely wrecked the model is visible as such.
Forward passes only except wolf_B's 24-update retrain. Resume-safe.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from polypythia_sl.data import (build_preference_rows, build_number_prompts,
                                PREFERENCE_EVAL_PROMPTS)
from polypythia_sl.train import CompletionDataset, CompletionCollator, seed_everything
from polypythia_sl.optim import build_optimizer
from polypythia_sl.generate import _whole_number_tokens

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "trait_specificity_control_v1"
BASE_ID = "EleutherAI/pythia-160m"
REVISION = "step143000"
WOLF_DIR = RUNS / "teacher_rule_saturated/models/preference_teacher"
LION_DIR = RUNS / "teacher_rule_lion_saturated/models/preference_teacher"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- frozen, identical to divergence_v2 so numbers are directly comparable ---
N_PROBES, PROBE_SEED = 128, 99001
N_PATHS, N_POSITIONS, SAMPLE_SEED = 2, 10, 99501
BATCH = 16

LAYERS, KINDS = (8, 9, 10, 11), ("attention.query_key_value", "mlp.dense_4h_to_h")
ANIMALS = ["wolf", "dog", "cat", "lion", "tiger", "horse", "fox",
           "elephant", "bear", "eagle"]
BEHAVIOR_PROMPTS = PREFERENCE_EVAL_PROMPTS[30:60]
RAND_SEEDS = (81001, 81002, 81003)          # reserved range 81xxx
LR, TOTAL_UPDATES, TBATCH, GRAD_ACCUM = 1e-5, 24, 8, 2


def pname(layer, kind):
    return f"gpt_neox.layers.{layer}.{kind}.weight"


def probes():
    return [r["prompt"] for r in
            build_number_prompts(N_PROBES, PROBE_SEED, 3, 7, 100, 999)]


@torch.inference_mode()
def sample_paths(base, tok, prefixes, allow):
    """Base's sampled reference paths, argmax, and top1-top2 gap per position."""
    base.eval()
    paths, argmaxes, gaps = [], [], []
    for path_i in range(N_PATHS):
        gen = torch.Generator(device="cpu").manual_seed(SAMPLE_SEED + path_i)
        p_all, a_all, g_all = [], [], []
        for s in range(0, len(prefixes), BATCH):
            batch = prefixes[s:s + BATCH]
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(DEVICE)
            ids_, attn = enc["input_ids"], enc["attention_mask"]
            p_b = [[] for _ in batch]; a_b = [[] for _ in batch]; g_b = [[] for _ in batch]
            for _ in range(N_POSITIONS):
                logits = base(input_ids=ids_, attention_mask=attn,
                              use_cache=False).logits[:, -1]
                restricted = logits[:, allow].float()
                probs = torch.softmax(restricted, dim=-1)
                am = allow[restricted.argmax(-1)]
                top2 = probs.topk(2, dim=-1).values
                gap = (top2[:, 0] - top2[:, 1]).cpu()
                sidx = torch.multinomial(probs.cpu(), 1, generator=gen).squeeze(1)
                sampled = allow[sidx.to(DEVICE)]
                for i in range(len(batch)):
                    p_b[i].append(int(sampled[i])); a_b[i].append(int(am[i]))
                    g_b[i].append(float(gap[i]))
                ids_ = torch.cat([ids_, sampled.unsqueeze(1)], dim=1)
                attn = torch.cat([attn, torch.ones_like(sampled.unsqueeze(1))], dim=1)
            p_all += p_b; a_all += a_b; g_all += g_b
        paths.append(p_all); argmaxes.append(a_all); gaps.append(g_all)
    return paths, argmaxes, gaps


@torch.inference_mode()
def forced_argmax_and_nll(model, tok, prefixes, paths, allow):
    """Arm's restricted argmax at each position of the base's sampled paths,
    plus mean NLL of the forced (base-sampled) token under this arm --
    the numeric-channel damage gate."""
    model.eval()
    out, nll_tot, nll_n = [], 0.0, 0
    for path_i in range(len(paths)):
        res = []
        for s in range(0, len(prefixes), BATCH):
            batch = prefixes[s:s + BATCH]
            ref = paths[path_i][s:s + BATCH]
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(DEVICE)
            ids_, attn = enc["input_ids"], enc["attention_mask"]
            seqs = [[] for _ in batch]
            for pos in range(N_POSITIONS):
                logits = model(input_ids=ids_, attention_mask=attn,
                               use_cache=False).logits[:, -1].float()
                am = allow[logits[:, allow].argmax(-1)]
                for i in range(len(batch)):
                    seqs[i].append(int(am[i]))
                forced = torch.tensor([ref[i][pos] for i in range(len(batch))],
                                      device=DEVICE)
                lp = torch.log_softmax(logits, dim=-1)
                nll_tot += float(-lp.gather(1, forced.unsqueeze(1)).sum())
                nll_n += len(batch)
                ids_ = torch.cat([ids_, forced.unsqueeze(1)], dim=1)
                attn = torch.cat([attn, torch.ones_like(forced.unsqueeze(1))], dim=1)
            res += seqs
        out.append(res)
    return out, nll_tot / nll_n


@torch.inference_mode()
def animal_margins(model, tok, ids):
    """Held-out margin for wolf and for lion (one-vs-rest, +ln 9)."""
    sel = torch.tensor(ids, device=DEVICE)
    w, l = [], []
    for s in range(0, len(BEHAVIOR_PROMPTS), 8):
        enc = tok(BEHAVIOR_PROMPTS[s:s + 8], return_tensors="pt",
                  padding=True).to(DEVICE)
        logits = model(**enc, use_cache=False).logits
        last = enc["attention_mask"].sum(1) - 1
        idx = torch.arange(enc["input_ids"].shape[0], device=DEVICE)
        ch = logits[idx, last][:, sel].float()
        for target in (0, 3):                      # wolf at 0, lion at 3
            other = [i for i in range(len(ANIMALS)) if i != target]
            m = (ch[:, target] - torch.logsumexp(ch[:, other], 1)
                 + math.log(9)).cpu().tolist()
            (w if target == 0 else l).extend(m)
    return float(np.mean(w)), float(np.mean(l))


def divergence_map(arm_am, base_am):
    """{(path, probe, pos): replacement_token} for positions where the arm's
    argmax differs from the base's on the SAME context."""
    return {(pa, pr, k): arm_am[pa][pr][k]
            for pa in range(len(arm_am)) for pr in range(len(arm_am[pa]))
            for k in range(N_POSITIONS)
            if arm_am[pa][pr][k] != base_am[pa][pr][k]}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def token_agreement(ma, mb):
    """Of positions divergent in BOTH arms, fraction where they picked the same
    replacement token. This is the CONTENT readout, orthogonal to shape."""
    shared = set(ma) & set(mb)
    if not shared:
        return float("nan"), 0
    agree = sum(1 for p in shared if ma[p] == mb[p])
    return agree / len(shared), len(shared)


def train_teacher(tok, animal, data_seed, train_seed):
    rows = build_preference_rows(animal, 384, data_seed)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_ID, revision=REVISION, torch_dtype=torch.float32).to(DEVICE)
    seed_everything(train_seed)
    loader = DataLoader(CompletionDataset(rows, tok, 96), batch_size=TBATCH,
                        shuffle=True, generator=torch.Generator().manual_seed(train_seed),
                        collate_fn=CompletionCollator(tok.pad_token_id))
    opt, _ = build_optimizer(model, {"optimizer": "adamw",
                                     "learning_rate": LR, "weight_decay": 0.1})
    warm = int(TOTAL_UPDATES * 0.05)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        (s + 1) / warm if warm and s < warm
        else max(TOTAL_UPDATES - s, 0) / max(TOTAL_UPDATES - warm, 1)))
    model.config.use_cache = False
    opt.zero_grad(set_to_none=True)
    micro = 0
    for bi, batch in enumerate(loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        (model(**batch).loss / GRAD_ACCUM).backward()
        micro += 1
        if micro % GRAD_ACCUM == 0 or bi == len(loader) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
    model.config.use_cache = True
    return model


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
    base_nll_am, base_nll = forced_argmax_and_nll(base, tok, pfx, paths, allow)
    base_wolf, base_lion = animal_margins(base, tok, animal_ids)
    print(f"base: numeric NLL {base_nll:.4f}, wolf {base_wolf:+.4f}, "
          f"lion {base_lion:+.4f}", flush=True)
    del base
    if DEVICE.type == "mps": torch.mps.empty_cache()

    # rank-1 trait patch (H13's construction) + per-module norms for matching
    wolf_sd = {k: v.clone() for k, v in AutoModelForCausalLM.from_pretrained(
        WOLF_DIR, torch_dtype=torch.float32).state_dict().items()}
    rank1_patch, rank1_norm = {}, {}
    for l in LAYERS:
        for kd in KINDS:
            n = pname(l, kd)
            U, S, Vh = torch.linalg.svd(
                wolf_sd[n].double() - base_sd[n].double(), full_matrices=False)
            p = ((U[:, :1] * S[:1]) @ Vh[:1, :]).float()
            rank1_patch[n] = p
            rank1_norm[n] = float(p.norm())
    # per-tensor norms of the FULL teacher delta, for rand_full matching
    full_norm = {k: float((wolf_sd[k] - base_sd[k]).norm())
                 for k in base_sd if wolf_sd[k].shape == base_sd[k].shape
                 and float((wolf_sd[k] - base_sd[k]).norm()) > 0}
    print(f"trait delta: {len(full_norm)} tensors changed; rank-1 group "
          f"total norm {sum(rank1_norm.values()):.4f}", flush=True)
    del wolf_sd

    def load_base_model():
        return AutoModelForCausalLM.from_pretrained(
            BASE_ID, revision=REVISION, torch_dtype=torch.float32).to(DEVICE)

    def patched_model(delta_fn):
        m = load_base_model()
        params = dict(m.named_parameters())
        with torch.no_grad():
            for name, d in delta_fn().items():
                params[name].data.add_(d.to(DEVICE))
        return m

    def rand_rank1(seed):
        g = torch.Generator().manual_seed(seed)
        out = {}
        for name, target in rank1_norm.items():
            shape = rank1_patch[name].shape
            u = torch.randn(shape[0], 1, generator=g)
            v = torch.randn(1, shape[1], generator=g)
            p = u @ v
            out[name] = p * (target / float(p.norm()))
        return out

    def rand_full(seed, scale):
        g = torch.Generator().manual_seed(seed)
        out = {}
        for name, target in full_norm.items():
            r = torch.randn(base_sd[name].shape, generator=g)
            out[name] = r * (scale * target / float(r.norm()))
        return out

    arms = [
        ("wolf_A", lambda: AutoModelForCausalLM.from_pretrained(
            WOLF_DIR, torch_dtype=torch.float32).to(DEVICE)),
        # lion weights were reclaimed in an earlier disk pass (result recorded
        # under H5, retention gate honored); retrained here from its recorded
        # recipe, which is byte-identical to wolf_A's except target_animal --
        # data seed 1103, size 384, train seed 2101. Cleanest possible
        # different-trait control: the ONLY difference is the target word.
        ("lion", lambda: train_teacher(tok, "lion", 1103, 2101)),
        # Noise floor. Same seeds as wolf_A (1103/2101), so this SHOULD
        # reproduce the on-disk wolf_A. Without it, the wolf_A-vs-lion
        # comparison confounds trait difference with retrain nondeterminism,
        # since lion is retrained and wolf_A is loaded from disk.
        ("wolf_A_retrain", lambda: train_teacher(tok, "wolf", 1103, 2101)),
        ("wolf_B", lambda: train_teacher(tok, "wolf", 5301, 5401)),
        ("wolf_rank1", lambda: patched_model(lambda: rank1_patch)),
    ]
    for s in RAND_SEEDS:
        arms.append((f"rand_rank1_{s}", lambda s=s: patched_model(lambda: rand_rank1(s))))
    for s in RAND_SEEDS[:2]:
        for sc in (0.5, 1.0):
            arms.append((f"rand_full_{s}_x{sc}",
                         lambda s=s, sc=sc: patched_model(lambda: rand_full(s, sc))))

    maps, stats = {}, {}
    cache = OUT / "arms.json"
    if cache.exists():
        raw = json.loads(cache.read_text())
        maps = {k: {tuple(map(int, p.split(","))): t for p, t in v.items()}
                for k, v in raw["maps"].items()}
        stats = raw["stats"]

    for name, builder in arms:
        if name in maps:
            print(f"[skip] {name} cached", flush=True)
            continue
        m = builder()
        am, nll = forced_argmax_and_nll(m, tok, pfx, paths, allow)
        wm, lm = animal_margins(m, tok, animal_ids)
        dm = divergence_map(am, base_am)
        maps[name] = dm
        dg = [gaps[pa][pr][k] for (pa, pr, k) in dm]
        stats[name] = {
            "n_divergent": len(dm),
            "mean_gap_at_divergent": (sum(dg) / len(dg)) if dg else None,
            "numeric_nll": nll, "numeric_nll_delta": nll - base_nll,
            "wolf_margin": wm, "wolf_margin_delta": wm - base_wolf,
            "lion_margin": lm, "lion_margin_delta": lm - base_lion,
            "n_distinct_replacement_tokens": len(set(dm.values())),
        }
        print(f"[arm] {name}: {len(dm)}/{N_PATHS*N_PROBES*N_POSITIONS} divergent, "
              f"gap@div {stats[name]['mean_gap_at_divergent']}, "
              f"dNLL {nll-base_nll:+.4f}, dWolf {wm-base_wolf:+.4f}, "
              f"dLion {lm-base_lion:+.4f}", flush=True)
        del m
        if DEVICE.type == "mps": torch.mps.empty_cache()
        cache.write_text(json.dumps({
            "maps": {k: {f"{a},{b},{c}": t for (a, b, c), t in v.items()}
                     for k, v in maps.items()},
            "stats": stats}))

    names = list(maps)
    shape_j, content_a, content_n = {}, {}, {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            key = f"{a}__{b}"
            shape_j[key] = jaccard(set(maps[a]), set(maps[b]))
            ag, n = token_agreement(maps[a], maps[b])
            content_a[key], content_n[key] = ag, n

    # uniform baselines: position Jaccard, and token agreement under an
    # independent draw from each arm's own replacement-token marginal
    import random as _r
    rng = _r.Random(81501)
    allpos = [(pa, pr, k) for pa in range(N_PATHS)
              for pr in range(N_PROBES) for k in range(N_POSITIONS)]
    szs = [stats[n]["n_divergent"] for n in names]
    med = sorted(szs)[len(szs) // 2]
    rj = [jaccard(set(rng.sample(allpos, med)), set(rng.sample(allpos, med)))
          for _ in range(2000)]
    pos_baseline = sum(rj) / len(rj)
    # token-agreement baseline: collision probability of two independent draws
    # from the two arms' empirical replacement-token marginals
    tok_baseline = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ca, cb = {}, {}
            for t in maps[a].values(): ca[t] = ca.get(t, 0) + 1
            for t in maps[b].values(): cb[t] = cb.get(t, 0) + 1
            na, nb = sum(ca.values()), sum(cb.values())
            tok_baseline[f"{a}__{b}"] = sum(
                (ca.get(t, 0) / na) * (cb.get(t, 0) / nb) for t in set(ca) | set(cb))

    report = {
        "base": {"numeric_nll": base_nll, "wolf_margin": base_wolf,
                 "lion_margin": base_lion,
                 "mean_gap_overall": sum(gaps[pa][pr][k] for pa in range(N_PATHS)
                                         for pr in range(N_PROBES)
                                         for k in range(N_POSITIONS))
                 / (N_PATHS * N_PROBES * N_POSITIONS)},
        "total_positions": N_PATHS * N_PROBES * N_POSITIONS,
        "arms": stats,
        "shape_jaccard": shape_j,
        "shape_uniform_baseline": pos_baseline,
        "content_token_agreement": content_a,
        "content_shared_positions": content_n,
        "content_marginal_baseline": tok_baseline,
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))

    L = ["# H15: trait-specific or generic? (random-perturbation control)", "",
         f"Base: numeric NLL {base_nll:.4f}, wolf margin {base_wolf:+.4f}, "
         f"lion margin {base_lion:+.4f}, mean top1-top2 gap "
         f"{report['base']['mean_gap_overall']:.3f}.",
         f"{N_PATHS*N_PROBES*N_POSITIONS} scored positions per arm.", "",
         "## Per-arm", "",
         "| arm | divergent | gap@div | dNLL | d wolf | d lion | distinct repl. tokens |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n in names:
        s = stats[n]
        g = f"{s['mean_gap_at_divergent']:.3f}" if s["mean_gap_at_divergent"] is not None else "n/a"
        L.append(f"| {n} | {s['n_divergent']} | {g} | {s['numeric_nll_delta']:+.4f} "
                 f"| {s['wolf_margin_delta']:+.4f} | {s['lion_margin_delta']:+.4f} "
                 f"| {s['n_distinct_replacement_tokens']} |")
    L += ["", "## SHAPE - which positions flip (Jaccard)", "",
          f"Uniform baseline at median set size: {pos_baseline:.3f}", "",
          "| pair | Jaccard |", "| --- | ---: |"]
    for k, v in sorted(shape_j.items(), key=lambda kv: -kv[1]):
        L.append(f"| {k} | {v:.3f} |")
    L += ["", "## CONTENT - what they flip TO (agreement on shared divergent positions)",
          "", "| pair | shared positions | token agreement | marginal baseline |",
          "| --- | ---: | ---: | ---: |"]
    for k, v in sorted(content_a.items(),
                       key=lambda kv: -(kv[1] if kv[1] == kv[1] else -1)):
        ag = f"{v:.3f}" if v == v else "n/a"
        L.append(f"| {k} | {content_n[k]} | {ag} | {tok_baseline[k]:.3f} |")
    (RUNS / "trait_specificity_control_v1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print("H15 DONE", flush=True)


if __name__ == "__main__":
    main()
