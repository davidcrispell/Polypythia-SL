"""Divergence-token analysis v2: sampled reference contexts (frozen design).

Corrects H11/H12/H14, whose v1 built reference contexts by GREEDY decoding --
degenerate on this restricted numeric channel (base argmax is " 1" at 79.8%
of positions). Keeps the argmax divergence CRITERION from arXiv 2509.23886,
but walks temperature-1.0 SAMPLED paths (project convention everywhere else).

Divergence at (probe, path, position) iff, given the same sampled context:
    argmax_restricted p_teacher != argmax_restricted p_base
k=2 independent sampled paths per probe (stability check). Base top1-top2
probability gap logged per position as a secondary diagnostic.

Phase A (H14-v2): cross-lineage fingerprints, 5 lineages, teachers on disk.
Phase B (H12-v2): same-base independent retrainings (teachers B/C retrained
    with the SAME seeds as v1: data 5301/5302, train 5401/5402, so this is a
    genuine re-analysis of the same teachers, not new ones).
Phase C (H11-v2): per-checkpoint emergence timing over the 24-update teacher
    fine-tune, with an explicitly update-indexed result list.

Resume-safe per phase. Writes runs/divergence_v2.{json,md}.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from polypythia_sl.data import build_preference_rows, build_number_prompts
from polypythia_sl.train import CompletionDataset, CompletionCollator, seed_everything
from polypythia_sl.optim import build_optimizer
from polypythia_sl.generate import _whole_number_tokens

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "divergence_v2"
REVISION = "step143000"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

N_PROBES = 128
PROBE_SEED = 99001          # same frozen probe set as v1
N_PATHS = 2
N_POSITIONS = 10
SAMPLE_SEED = 99501         # documented seed for sampled reference paths
BATCH = 16

LINEAGES = [
    ("standard", "EleutherAI/pythia-160m", RUNS / "teacher_rule_saturated/models/preference_teacher"),
    ("data-seed1", "EleutherAI/pythia-160m-data-seed1", RUNS / "ds1_teacher/models/preference_teacher"),
    ("data-seed2", "EleutherAI/pythia-160m-data-seed2", RUNS / "ds2_teacher/models/preference_teacher"),
    ("weight-seed1", "EleutherAI/pythia-160m-weight-seed1", RUNS / "ws1_teacher/models/preference_teacher"),
    ("weight-seed3", "EleutherAI/pythia-160m-weight-seed3", RUNS / "ws3_teacher/models/preference_teacher"),
]
RETRAIN = [("teacher_B", 5301, 5401), ("teacher_C", 5302, 5402)]
LR, TOTAL_UPDATES, TBATCH, GRAD_ACCUM = 1e-5, 24, 8, 2


def probes():
    return [r["prompt"] for r in
            build_number_prompts(N_PROBES, PROBE_SEED, 3, 7, 100, 999)]


def allowed(tok):
    ids, _ = _whole_number_tokens(tok, 999)
    return torch.tensor(ids, dtype=torch.long, device=DEVICE)


@torch.inference_mode()
def sample_paths(base, tok, prefixes, allow):
    """Sampled reference paths + base argmax + base top1-top2 gap per position.
    Returns paths[path][probe][pos], base_argmax[path][probe][pos],
    gaps[path][probe][pos]."""
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
                sampled_idx = torch.multinomial(probs.cpu(), 1, generator=gen).squeeze(1)
                sampled = allow[sampled_idx.to(DEVICE)]
                for i in range(len(batch)):
                    p_b[i].append(int(sampled[i])); a_b[i].append(int(am[i]))
                    g_b[i].append(float(gap[i]))
                ids_ = torch.cat([ids_, sampled.unsqueeze(1)], dim=1)
                attn = torch.cat([attn, torch.ones_like(sampled.unsqueeze(1))], dim=1)
            p_all += p_b; a_all += a_b; g_all += g_b
        paths.append(p_all); argmaxes.append(a_all); gaps.append(g_all)
    return paths, argmaxes, gaps


@torch.inference_mode()
def forced_argmax(model, tok, prefixes, paths, allow):
    """Teacher's argmax at each position of the given sampled paths."""
    model.eval()
    out = []
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
                               use_cache=False).logits[:, -1]
                am = allow[logits[:, allow].float().argmax(-1)]
                for i in range(len(batch)):
                    seqs[i].append(int(am[i]))
                forced = torch.tensor([ref[i][pos] for i in range(len(batch))],
                                      device=DEVICE)
                ids_ = torch.cat([ids_, forced.unsqueeze(1)], dim=1)
                attn = torch.cat([attn, torch.ones_like(forced.unsqueeze(1))], dim=1)
            res += seqs
        out.append(res)
    return out


def div_set(teacher_am, base_am):
    return {(pa, pr, k) for pa in range(len(teacher_am))
            for pr in range(len(teacher_am[pa])) for k in range(N_POSITIONS)
            if teacher_am[pa][pr][k] != base_am[pa][pr][k]}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else float("nan")


def train_teacher(tok, data_seed, train_seed, model_id, hook=None):
    rows = build_preference_rows("wolf", 384, data_seed)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=REVISION, torch_dtype=torch.float32).to(DEVICE)
    seed_everything(train_seed)
    ds = CompletionDataset(rows, tok, 96)
    loader = DataLoader(ds, batch_size=TBATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(train_seed),
                        collate_fn=CompletionCollator(tok.pad_token_id))
    opt, _ = build_optimizer(model, {"optimizer": "adamw",
                                     "learning_rate": LR, "weight_decay": 0.1})
    warm = int(TOTAL_UPDATES * 0.05)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        (s + 1) / warm if warm and s < warm
        else max(TOTAL_UPDATES - s, 0) / max(TOTAL_UPDATES - warm, 1)))
    model.config.use_cache = False
    opt.zero_grad(set_to_none=True)
    micro, update = 0, 0
    if hook:
        hook(0, model, None)
    for bi, batch in enumerate(loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        (model(**batch).loss / GRAD_ACCUM).backward()
        micro += 1
        if micro % GRAD_ACCUM == 0 or bi == len(loader) - 1:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            update += 1
            if hook:
                hook(update, model, float(gn))
    model.config.use_cache = True
    return model


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pfx = probes()
    report = {}

    # ---------- Phase A: cross-lineage (H14-v2) ----------
    fA = OUT / "phaseA_cross_lineage.json"
    if fA.exists():
        A = json.loads(fA.read_text())
    else:
        sets, sizes = {}, {}
        for name, model_id, tdir in LINEAGES:
            tok = AutoTokenizer.from_pretrained(model_id, revision=REVISION)
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            allow = allowed(tok)
            base = AutoModelForCausalLM.from_pretrained(
                model_id, revision=REVISION, torch_dtype=torch.float32).to(DEVICE)
            paths, base_am, gaps = sample_paths(base, tok, pfx, allow)
            del base
            if DEVICE.type == "mps": torch.mps.empty_cache()
            teacher = AutoModelForCausalLM.from_pretrained(
                tdir, torch_dtype=torch.float32).to(DEVICE)
            t_am = forced_argmax(teacher, tok, pfx, paths, allow)
            del teacher
            if DEVICE.type == "mps": torch.mps.empty_cache()
            d = div_set(t_am, base_am)
            sets[name] = d; sizes[name] = len(d)
            # per-path counts (stability) + gap stats at divergent vs not
            per_path = [sum(1 for (pa, pr, k) in d if pa == i) for i in range(N_PATHS)]
            dg = [gaps[pa][pr][k] for (pa, pr, k) in d]
            allg = [gaps[pa][pr][k] for pa in range(N_PATHS)
                    for pr in range(len(pfx)) for k in range(N_POSITIONS)]
            print(f"[A] {name}: {len(d)} divergent of {N_PATHS*len(pfx)*N_POSITIONS} "
                  f"(per-path {per_path}); mean base top1-top2 gap at divergent "
                  f"{sum(dg)/len(dg) if dg else float('nan'):.3f} vs overall "
                  f"{sum(allg)/len(allg):.3f}", flush=True)
            sets[name] = d
            report.setdefault("A_gap", {})[name] = {
                "mean_gap_divergent": (sum(dg)/len(dg)) if dg else None,
                "mean_gap_overall": sum(allg)/len(allg),
                "per_path_counts": per_path}
        names = list(sets)
        pw = {f"{names[i]}__{names[j]}": jaccard(sets[names[i]], sets[names[j]])
              for i in range(len(names)) for j in range(i+1, len(names))}
        shared = {f"{names[i]}__{names[j]}": len(sets[names[i]] & sets[names[j]])
                  for i in range(len(names)) for j in range(i+1, len(names))}
        A = {"sizes": sizes, "pairwise_jaccard": pw, "shared_counts": shared,
             "gap": report.get("A_gap", {}),
             "total_positions": N_PATHS*len(pfx)*N_POSITIONS,
             "sets": {k: sorted(v) for k, v in sets.items()}}
        fA.write_text(json.dumps(A))
    report["A"] = {k: v for k, v in A.items() if k != "sets"}

    # ---------- Phase B: same-base retrainings (H12-v2) ----------
    fB = OUT / "phaseB_same_base.json"
    if fB.exists():
        B = json.loads(fB.read_text())
    else:
        mid = "EleutherAI/pythia-160m"
        tok = AutoTokenizer.from_pretrained(mid, revision=REVISION)
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
        allow = allowed(tok)
        base = AutoModelForCausalLM.from_pretrained(
            mid, revision=REVISION, torch_dtype=torch.float32).to(DEVICE)
        paths, base_am, gaps = sample_paths(base, tok, pfx, allow)
        del base
        if DEVICE.type == "mps": torch.mps.empty_cache()
        sets = {}
        # teacher_A = canonical on-disk standard teacher (same recipe as v1's A)
        ta = AutoModelForCausalLM.from_pretrained(
            LINEAGES[0][2], torch_dtype=torch.float32).to(DEVICE)
        sets["teacher_A"] = div_set(forced_argmax(ta, tok, pfx, paths, allow), base_am)
        del ta
        if DEVICE.type == "mps": torch.mps.empty_cache()
        print(f"[B] teacher_A: {len(sets['teacher_A'])}", flush=True)
        for nm, dseed, tseed in RETRAIN:
            m = train_teacher(tok, dseed, tseed, mid)
            sets[nm] = div_set(forced_argmax(m, tok, pfx, paths, allow), base_am)
            del m
            if DEVICE.type == "mps": torch.mps.empty_cache()
            print(f"[B] {nm}: {len(sets[nm])}", flush=True)
        names = list(sets)
        pw = {f"{names[i]}__{names[j]}": jaccard(sets[names[i]], sets[names[j]])
              for i in range(len(names)) for j in range(i+1, len(names))}
        shared = {f"{names[i]}__{names[j]}": len(sets[names[i]] & sets[names[j]])
                  for i in range(len(names)) for j in range(i+1, len(names))}
        # empirical random baseline: same sizes, uniform over all positions
        import random as _r
        rng = _r.Random(77002)
        allpos = [(pa, pr, k) for pa in range(N_PATHS)
                  for pr in range(len(pfx)) for k in range(N_POSITIONS)]
        sz = [len(v) for v in sets.values()]
        rj = []
        for _ in range(2000):
            a = set(rng.sample(allpos, sz[0])); b = set(rng.sample(allpos, sz[1]))
            rj.append(jaccard(a, b))
        B = {"sizes": {k: len(v) for k, v in sets.items()},
             "pairwise_jaccard": pw, "shared_counts": shared,
             "random_baseline_jaccard": sum(rj)/len(rj),
             "total_positions": len(allpos)}
        fB.write_text(json.dumps(B))
    report["B"] = B

    # ---------- Phase C: emergence timing (H11-v2) ----------
    fC = OUT / "phaseC_timing.json"
    if fC.exists():
        C = json.loads(fC.read_text())
    else:
        mid = "EleutherAI/pythia-160m"
        tok = AutoTokenizer.from_pretrained(mid, revision=REVISION)
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
        allow = allowed(tok)
        base = AutoModelForCausalLM.from_pretrained(
            mid, revision=REVISION, torch_dtype=torch.float32).to(DEVICE)
        paths, base_am, gaps = sample_paths(base, tok, pfx, allow)
        del base
        if DEVICE.type == "mps": torch.mps.empty_cache()
        ckpts = []  # explicitly update-indexed list (lesson: u16 overwrite bug)

        def hook(update, model, gn):
            am = forced_argmax(model, tok, pfx, paths, allow)
            ckpts.append({"update": update, "grad_norm": gn,
                          "divergent": sorted(div_set(am, base_am))})
            print(f"[C] checkpoint {update}/{TOTAL_UPDATES}", flush=True)

        m = train_teacher(tok, 1103, 2101, mid, hook=hook)
        del m
        if DEVICE.type == "mps": torch.mps.empty_cache()
        assert [c["update"] for c in ckpts] == list(range(TOTAL_UPDATES + 1)), \
            "checkpoint indexing broken"
        final = {tuple(x) for x in ckpts[-1]["divergent"]}
        per_ckpt = [{tuple(x) for x in c["divergent"]} for c in ckpts]
        recs = []
        for pos in final:
            seq = [pos in s for s in per_ckpt]
            first = seq.index(True)
            recs.append({"position": list(pos), "first_emergence_update": first,
                         "stable_after": all(seq[first:]),
                         "n_flips": sum(1 for i in range(1, len(seq))
                                        if seq[i] != seq[i-1])})
        hist = Counter(r["first_emergence_update"] for r in recs)
        C = {"n_final_divergent": len(final),
             "total_positions": N_PATHS*len(pfx)*N_POSITIONS,
             "stable_fraction": (sum(r["stable_after"] for r in recs)/len(recs)) if recs else None,
             "mean_first_emergence": (sum(r["first_emergence_update"] for r in recs)/len(recs)) if recs else None,
             "emergence_histogram": {str(k): v for k, v in sorted(hist.items())},
             "per_token": recs,
             "sizes_by_update": [len(s) for s in per_ckpt]}
        fC.write_text(json.dumps(C))
    report["C"] = {k: v for k, v in C.items() if k != "per_token"}

    (OUT / "summary.json").write_text(json.dumps(report, indent=2))

    L = ["# Divergence-token analysis v2 (sampled reference contexts)", "",
         f"Probes: {N_PROBES} x {N_PATHS} sampled paths x {N_POSITIONS} positions "
         f"= {N_PATHS*N_PROBES*N_POSITIONS} scored positions per model.", "",
         "## Phase A - cross-lineage (H14-v2)", "",
         "| lineage | divergent | per-path split |", "| --- | ---: | --- |"]
    for k, v in report["A"]["sizes"].items():
        pp = report["A"]["gap"][k]["per_path_counts"]
        L.append(f"| {k} | {v} | {pp} |")
    L += ["", "| lineage pair | shared | Jaccard |", "| --- | ---: | ---: |"]
    for k, v in sorted(report["A"]["pairwise_jaccard"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {k} | {report['A']['shared_counts'][k]} | {v:.3f} |")
    L += ["", "Base top1-top2 probability gap at divergent positions vs overall:"]
    for k, g in report["A"]["gap"].items():
        md = g["mean_gap_divergent"]
        L.append(f"- {k}: divergent {md:.3f} vs overall {g['mean_gap_overall']:.3f}"
                 if md is not None else f"- {k}: (none divergent)")
    L += ["", "## Phase B - same-base independent retrainings (H12-v2)", "",
          "| teacher | divergent |", "| --- | ---: |"]
    for k, v in report["B"]["sizes"].items():
        L.append(f"| {k} | {v} |")
    L += ["", "| pair | shared | Jaccard |", "| --- | ---: | ---: |"]
    for k, v in sorted(report["B"]["pairwise_jaccard"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {k} | {report['B']['shared_counts'][k]} | {v:.3f} |")
    L += ["", f"Empirical random baseline (same sizes, uniform): "
          f"{report['B']['random_baseline_jaccard']:.3f}",
          "", "## Phase C - emergence timing (H11-v2)", "",
          f"- final divergent: {report['C']['n_final_divergent']} of "
          f"{report['C']['total_positions']}",
          f"- stable after first emergence: {report['C']['stable_fraction']:.1%}"
          if report['C']['stable_fraction'] is not None else "- stable: n/a",
          f"- mean first-emergence update: {report['C']['mean_first_emergence']:.2f}"
          f" / {TOTAL_UPDATES}" if report['C']['mean_first_emergence'] is not None else "",
          "", "| update | first emerging |", "| ---: | ---: |"]
    for u, n in report["C"]["emergence_histogram"].items():
        L.append(f"| {u} | {n} |")
    L += ["", f"Divergent count by update: {report['C']['sizes_by_update']}"]
    (RUNS / "divergence_v2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print("DIVERGENCE_V2 DONE", flush=True)


if __name__ == "__main__":
    main()
