"""H21: response-subspace comparison -- is the trait->output medium init-bound?

Replaces H20's raw delta transplant, which conflated "is the coupling shared"
with "are these models in the same coordinate frame" (David's veto: init
structure can persist through pretraining while the coordinates expressing it
drift -- neuron permutation, rotation, scaling). This measures the coupling in
TOKEN space, which is common to every model by construction and therefore
permutation-invariant, and uses NO trait training at all, so separately-trained
trait circuits cannot confound it.

For each base: apply K random rank-1 perturbations to the frozen late group
(L8-11 x {QKV, MLP-out}), each per-module Frobenius-norm matched to THAT
lineage's own trait delta (magnitude calibrated to the model's own trait
scale). Record the induced marginal token-frequency shift -- a 655-dim vector.
Stack -> K x 655 response matrix -> top-r principal subspace via SVD.

That subspace answers: which directions in token space can perturbations of
this module group reach at all? That is the "medium" in David's link 4.

Compare subspaces pairwise by subspace affinity, ||U_A^T U_B||_F^2 / r (mean
squared cosine of principal angles; 1 = identical span, ~r/655 = chance).

Frozen predictions P1-P3 and the falsifier are in EXPERIMENTS.md (H21).
Forward passes only. Resume-safe per arm. Seeds 85xxx.
"""
from __future__ import annotations

import itertools
import json
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
OUT = RUNS / "response_subspace_v1"
H20 = RUNS / "delta_transplant_v1"
REVISION = "step143000"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

N_PROBES, PROBE_SEED = 48, 99001          # first 48 of the frozen probe set
N_POSITIONS = 10
BATCH = 16
K = 24                                     # perturbations per base
RANK = 6                                   # principal subspace dimension
SEED0 = 85001
LAYERS, KINDS = (8, 9, 10, 11), ("attention.query_key_value", "mlp.dense_4h_to_h")

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
INIT_FAMILY = {"ds1": "dsW0", "ds2": "dsW0", "standard": "std", "ws1": "w1", "ws3": "w3"}
ORDER_FAMILY = {"standard": "ref", "ws1": "ref", "ws3": "ref", "ds1": "o1", "ds2": "o2"}


def pair_class(a, b):
    if INIT_FAMILY[a] == INIT_FAMILY[b]:
        return "shared-init/diff-order"
    if ORDER_FAMILY[a] == ORDER_FAMILY[b]:
        return "shared-order/diff-init"
    return "neither"


def pname(layer, kind):
    return f"gpt_neox.layers.{layer}.{kind}.weight"


def probes():
    return [r["prompt"] for r in
            build_number_prompts(N_PROBES, PROBE_SEED, 3, 7, 100, 999)]


@torch.inference_mode()
def marginal(model, tok, prefixes, paths, allow):
    """Mean restricted next-token distribution over all scored positions."""
    model.eval()
    acc = torch.zeros(len(allow), dtype=torch.float64)
    n = 0
    for s in range(0, len(prefixes), BATCH):
        batch = prefixes[s:s + BATCH]
        ref = paths[s:s + BATCH]
        enc = tok(batch, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(DEVICE)
        ids_, attn = enc["input_ids"], enc["attention_mask"]
        for pos in range(N_POSITIONS):
            logits = model(input_ids=ids_, attention_mask=attn,
                           use_cache=False).logits[:, -1].float()
            p = torch.softmax(logits[:, allow], dim=-1).cpu().double()
            acc += p.sum(0)
            n += p.shape[0]
            forced = torch.tensor([ref[i][pos] for i in range(len(batch))],
                                  device=DEVICE)
            ids_ = torch.cat([ids_, forced.unsqueeze(1)], dim=1)
            attn = torch.cat([attn, torch.ones_like(forced.unsqueeze(1))], dim=1)
    return (acc / n).numpy()


def trait_norms(name):
    """Per-module Frobenius norm of this lineage's own trait delta, used to
    calibrate perturbation magnitude to the model's own trait scale."""
    model_id, tdir = LINEAGES[name]
    base_sd = AutoModelForCausalLM.from_pretrained(
        model_id, revision=REVISION, torch_dtype=torch.float32).state_dict()
    teach_sd = AutoModelForCausalLM.from_pretrained(
        tdir, torch_dtype=torch.float32).state_dict()
    norms = {}
    for l in LAYERS:
        for kd in KINDS:
            k = pname(l, kd)
            norms[k] = float((teach_sd[k] - base_sd[k]).norm())
    return norms


def subspace(R, r):
    """Top-r right-singular subspace of the row-normalised response matrix."""
    X = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    X = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:r].T                                  # 655 x r, orthonormal


def affinity(Ua, Ub):
    return float((Ua.T @ Ub) .__pow__(2).sum() / Ua.shape[1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(LINEAGES["standard"][0], revision=REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ids_allowed, _ = _whole_number_tokens(tok, 999)
    allow = torch.tensor(ids_allowed, dtype=torch.long, device=DEVICE)
    pfx = probes()

    # Shared contexts: reuse H20's (sampled from the standard base, seed 99501),
    # first path, truncated to the first N_PROBES rows.
    ctx = json.loads((H20 / "contexts.json").read_text())[0][:N_PROBES]
    assert len(ctx) == N_PROBES

    for name in NAMES:
        f = OUT / f"{name}.npy"
        if f.exists():
            print(f"[skip] {name} cached", flush=True)
            continue
        model_id, _ = LINEAGES[name]
        norms = trait_norms(name)

        base = AutoModelForCausalLM.from_pretrained(
            model_id, revision=REVISION, torch_dtype=torch.float32)
        base_shapes = {k: base.state_dict()[k].shape for k in norms}
        base = base.to(DEVICE)
        m0 = marginal(base, tok, pfx, ctx, allow)
        del base
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        print(f"[{name}] unperturbed marginal computed", flush=True)

        rows = []
        for k in range(K):
            g = torch.Generator().manual_seed(SEED0 + k)   # same draws every base
            m = AutoModelForCausalLM.from_pretrained(
                model_id, revision=REVISION, torch_dtype=torch.float32)
            sd = m.state_dict()
            with torch.no_grad():
                for pn, target in norms.items():
                    shp = base_shapes[pn]
                    u = torch.randn(shp[0], 1, generator=g)
                    v = torch.randn(1, shp[1], generator=g)
                    P = u @ v
                    sd[pn].add_(P * (target / float(P.norm())))
            m = m.to(DEVICE)
            rows.append(marginal(m, tok, pfx, ctx, allow) - m0)
            del m
            if DEVICE.type == "mps":
                torch.mps.empty_cache()
            if (k + 1) % 6 == 0:
                print(f"[{name}] {k + 1}/{K} perturbations", flush=True)
        np.save(f, np.stack(rows))
        print(f"[done] {name}", flush=True)

    R = {n: np.load(OUT / f"{n}.npy") for n in NAMES}
    U = {n: subspace(R[n], RANK) for n in NAMES}

    # Noise ceiling: split-half within each base.
    half = {}
    for n in NAMES:
        a = subspace(R[n][0::2], RANK)
        b = subspace(R[n][1::2], RANK)
        half[n] = affinity(a, b)

    # Analytic-ish floor: random Gaussian response matrices, same shape.
    rng = np.random.default_rng(85501)
    floor = []
    for _ in range(200):
        a = subspace(rng.normal(size=R[NAMES[0]].shape), RANK)
        b = subspace(rng.normal(size=R[NAMES[0]].shape), RANK)
        floor.append(affinity(a, b))
    floor_mean = float(np.mean(floor))

    pairs = {}
    for a, b in itertools.combinations(NAMES, 2):
        pairs[f"{a}__{b}"] = {"class": pair_class(a, b),
                              "affinity": affinity(U[a], U[b])}

    cells = {}
    for cls in ("shared-init/diff-order", "shared-order/diff-init", "neither"):
        v = [p["affinity"] for p in pairs.values() if p["class"] == cls]
        cells[cls] = {"n": len(v), "mean": float(np.mean(v)),
                      "range": [float(min(v)), float(max(v))]}

    si, so, ne = (cells["shared-init/diff-order"]["mean"],
                  cells["shared-order/diff-init"]["mean"],
                  cells["neither"]["mean"])
    verdict = {
        "P2_shared_init_exceeds_shared_order_by_0.10": bool(si - so >= 0.10),
        "P3_shared_order_near_floor": bool(so < ne + 0.05),
        "FALSIFIER_shared_order_ge_shared_init": bool(so >= si),
        "FALSIFIER_all_at_floor": bool(max(si, so, ne) < floor_mean + 0.05),
    }

    report = {"K": K, "rank": RANK, "n_positions": N_PROBES * N_POSITIONS,
              "noise_ceiling_split_half": half,
              "random_floor": floor_mean, "cells": cells,
              "verdict": verdict, "pairs": pairs}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))

    L = ["# H21: response-subspace comparison (is the medium init-bound?)", "",
         f"K={K} random rank-1 perturbations per base, per-module norm-matched to "
         f"that lineage's own trait delta; induced marginal token-frequency shift "
         f"in 655-dim token space; top-{RANK} principal subspace; "
         f"affinity = mean squared cosine of principal angles.", "",
         f"Noise ceiling (split-half within base): "
         + ", ".join(f"{n} {v:.3f}" for n, v in half.items()),
         f"Random floor: {floor_mean:.3f}", "",
         "| cell | n | mean affinity | range |", "| --- | ---: | ---: | --- |"]
    for cls, c in cells.items():
        L.append(f"| {cls} | {c['n']} | {c['mean']:.3f} "
                 f"| [{c['range'][0]:.3f}, {c['range'][1]:.3f}] |")
    L += ["", "## Verdict against frozen predictions", ""]
    for k, v in verdict.items():
        L.append(f"- {k}: **{v}**")
    L += ["", "| pair | class | affinity |", "| --- | --- | ---: |"]
    for k, v in sorted(pairs.items(), key=lambda kv: -kv[1]["affinity"]):
        L.append(f"| {k} | {v['class']} | {v['affinity']:.3f} |")
    (RUNS / "response_subspace_v1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print("H21 DONE", flush=True)


if __name__ == "__main__":
    main()
