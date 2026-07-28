"""H24: live B-output-cotangent factorial.

The credit-side account rests on one local measurement (2026-07-16/17:
phi_D ~ kappa, phi_X ~ 0), which states outright that it does not establish
necessity or sufficiency for endpoint SL. Its designated follow-up -- a live
factorial with natural / D-swap / X-swap / sham arms, both A/B gradients derived
coherently, numeric-NLL noninferiority required -- was never run. This runs it.

Live means: gradients are actually substituted, optimizer steps are actually
taken, and the readout is the student's endpoint held-out wolf margin.

Mechanics. PEFT LoRA forward is `base(x) + lora_B(lora_A(x)) * scaling`. Per
wrapped module we capture, on BOTH a preference micro-batch and its row-aligned
control micro-batch:
    x_A  input to lora_A
    x_B  = lora_A(x_A), input to lora_B
    d_B  cotangent at lora_B's output
and rebuild both gradients from a single substitution at the cotangent entering
the LoRA branch:
    grad_W_B = d_B^T x_B
    grad_W_A = (d_B W_B)^T x_A
so a swapped d_B propagates coherently to A. This is why the intervention is at
B's output rather than independently at each Linear.

Arms differ ONLY in which (d, x) pair builds the gradient; data order, schedule,
seeds and everything else are identical:
    natural  (d_P, x_P)      control  (d_C, x_C)
    D_swap   (d_C, x_P)      X_swap   (d_P, x_C)
    sham     (d_sham, x_P)   -- per-module energy-matched random credit

Frozen predictions P1-P4, both falsifiers, and the noninferiority gate are in
EXPERIMENTS.md (H24). An update-0 integrity check requires the hand-built
natural gradients to match autograd to 1e-5 relative or the run aborts.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from polypythia_sl.data import PREFERENCE_EVAL_PROMPTS
from polypythia_sl.optim import build_optimizer

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "cotangent_factorial_v1"
POOL = RUNS / "confirm_v3_b1/data"
BASE_ID = "EleutherAI/pythia-160m"
REVISION = "step143000"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

ARMS = ("natural", "control", "D_swap", "X_swap", "sham")
SEEDS = (87001, 87002)
SHAM_SEED = 87501
DOSE, BATCH, ACCUM = 256, 8, 2
# LoRA B is zero-initialised, so grad_W_A is identically zero at update 0 and the
# relative-error check on A would pass vacuously. Check once B is nonzero.
INTEGRITY_UPDATE = 5
LR, WARMUP, MAXLEN = 2e-4, 8, 96
ANIMALS = ["wolf", "dog", "cat", "lion", "tiger", "horse", "fox",
           "elephant", "bear", "eagle"]
EVAL_PROMPTS = PREFERENCE_EVAL_PROMPTS[:60]
N_NLL_ROWS = 256


def load_pools():
    P = [json.loads(l) for l in (POOL / "numbers_preference_teacher.jsonl").open()]
    C = [json.loads(l) for l in (POOL / "numbers_base_teacher.jsonl").open()]
    assert len(P) == len(C) == 8192
    assert all(p["prompt"] == c["prompt"] for p, c in zip(P, C))
    return P, C


def encode(rows, tok):
    """Tokenize prompt+completion, mask prompt positions out of the loss."""
    texts = [r["prompt"] + r["completion"] for r in rows]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=MAXLEN, add_special_tokens=False)
    labels = enc["input_ids"].clone()
    for i, r in enumerate(rows):
        plen = len(tok(r["prompt"], add_special_tokens=False)["input_ids"])
        labels[i, :plen] = -100
    labels[enc["attention_mask"] == 0] = -100
    return {k: v.to(DEVICE) for k, v in enc.items()}, labels.to(DEVICE)


def lora_modules(model):
    """{name: {A, B, scaling}} for every PEFT-wrapped Linear."""
    out = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            out[name] = {"A": mod.lora_A["default"], "B": mod.lora_B["default"],
                         "scaling": float(mod.scaling["default"]), "parent": mod}
    return out


class Capture:
    """Records x_A, x_B and d_B per LoRA module for one forward/backward."""

    def __init__(self, mods):
        self.mods, self.rec, self.handles = mods, {}, []
        for name, m in mods.items():
            self.rec[name] = {}
            self.handles.append(m["A"].register_forward_hook(self._mk_a(name)))
            self.handles.append(m["B"].register_forward_hook(self._mk_b(name)))

    def _mk_a(self, name):
        def hook(module, inputs, output):
            self.rec[name]["x_A"] = inputs[0].detach()
        return hook

    def _mk_b(self, name):
        def hook(module, inputs, output):
            self.rec[name]["x_B"] = inputs[0].detach()

            def grab(g):
                self.rec[name]["d_B"] = g.detach()
                return g
            output.register_hook(grab)
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def build_grads(mods, d_src, x_src):
    """grad_W_B = d^T x_B ; grad_W_A = (d W_B)^T x_A -- coherent in d."""
    grads = {}
    for name, m in mods.items():
        d = d_src[name]["d_B"]
        x_B = x_src[name]["x_B"]
        x_A = x_src[name]["x_A"]
        dm = d.reshape(-1, d.shape[-1])
        gB = dm.transpose(0, 1) @ x_B.reshape(-1, x_B.shape[-1])
        dA = dm @ m["B"].weight                      # (d W_B), [N, r]
        gA = dA.transpose(0, 1) @ x_A.reshape(-1, x_A.shape[-1])
        grads[name] = {"A": gA, "B": gB}
    return grads


@torch.inference_mode()
def wolf_margin(model, tok, ids):
    model.eval()
    sel = torch.tensor(ids, device=DEVICE)
    out = []
    for s in range(0, len(EVAL_PROMPTS), 8):
        enc = tok(EVAL_PROMPTS[s:s + 8], return_tensors="pt", padding=True).to(DEVICE)
        logits = model(**enc, use_cache=False).logits
        last = enc["attention_mask"].sum(1) - 1
        idx = torch.arange(enc["input_ids"].shape[0], device=DEVICE)
        ch = logits[idx, last][:, sel].float()
        out.extend((ch[:, 0] - torch.logsumexp(ch[:, 1:], 1)
                    + math.log(9)).cpu().tolist())
    model.train()
    return float(np.mean(out))


@torch.inference_mode()
def numeric_nll(model, tok, rows):
    model.eval()
    tot, n = 0.0, 0
    for s in range(0, len(rows), 16):
        enc, labels = encode(rows[s:s + 16], tok)
        logits = model(**enc, use_cache=False).logits
        tot += float(torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum"))
        n += int((labels[:, 1:] != -100).sum())
    model.train()
    return tot / n


def run_arm(arm, seed, P, C, tok, animal_ids, nll_rows, integrity_out):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, revision=REVISION, torch_dtype=torch.float32)
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                     target_modules=["query_key_value", "dense",
                                     "dense_h_to_4h", "dense_4h_to_h"],
                     task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg).to(DEVICE)
    model.config.use_cache = False
    model.train()
    mods = lora_modules(model)

    opt, _ = build_optimizer(model, {"optimizer": "adamw",
                                     "learning_rate": LR, "weight_decay": 0.1})
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        (s + 1) / WARMUP if s < WARMUP
        else max(DOSE - s, 0) / max(DOSE - WARMUP, 1)))

    order = list(range(len(P)))
    random.Random(seed).shuffle(order)
    gen = torch.Generator().manual_seed(SHAM_SEED + seed)
    ptr = 0
    start_margin = wolf_margin(model, tok, animal_ids)

    for update in range(DOSE):
        opt.zero_grad(set_to_none=True)
        acc = {n: {"A": None, "B": None} for n in mods}
        for _ in range(ACCUM):
            idx = [order[(ptr + i) % len(order)] for i in range(BATCH)]
            ptr += BATCH
            capP, capC = Capture(mods), None
            encP, labP = encode([P[i] for i in idx], tok)
            lossP = model(**encP, labels=labP).loss
            lossP.backward()
            capP.close()
            recP = {k: dict(v) for k, v in capP.rec.items()}
            autograd_ref = ({n: {"A": mods[n]["A"].weight.grad.detach().clone(),
                                 "B": mods[n]["B"].weight.grad.detach().clone()}
                             for n in mods}
                            if (arm == "natural" and update == INTEGRITY_UPDATE and
                                integrity_out.get("done") is None) else None)
            opt.zero_grad(set_to_none=True)

            recC = None
            if arm != "natural":
                capC = Capture(mods)
                encC, labC = encode([C[i] for i in idx], tok)
                model(**encC, labels=labC).loss.backward()
                capC.close()
                recC = {k: dict(v) for k, v in capC.rec.items()}
                opt.zero_grad(set_to_none=True)

            if arm == "natural":
                g = build_grads(mods, recP, recP)
            elif arm == "control":
                g = build_grads(mods, recC, recC)
            elif arm == "D_swap":
                g = build_grads(mods, recC, recP)
            elif arm == "X_swap":
                g = build_grads(mods, recP, recC)
            else:                                     # sham
                sham = {}
                for n in mods:
                    d = recP[n]["d_B"]
                    r = torch.randn(d.shape, generator=gen).to(d.device)
                    r *= (torch.linalg.vector_norm(d) /
                          torch.linalg.vector_norm(r))
                    sham[n] = {"d_B": r}
                g = build_grads(mods, sham, recP)

            if autograd_ref is not None:
                errs = []
                for n in mods:
                    for side in ("A", "B"):
                        ref = autograd_ref[n][side]
                        got = g[n][side]
                        errs.append(float((got - ref).norm() /
                                          (ref.norm() + 1e-12)))
                integrity_out["max_rel_err"] = max(errs)
                integrity_out["n_tensors"] = len(errs)
                integrity_out["done"] = True
                if max(errs) > 1e-5:
                    raise RuntimeError(
                        f"gradient factorization mis-specified: "
                        f"max rel err {max(errs):.3e} over {len(errs)} tensors")

            for n in mods:
                for side in ("A", "B"):
                    acc[n][side] = (g[n][side] if acc[n][side] is None
                                    else acc[n][side] + g[n][side])

        for n in mods:
            for side in ("A", "B"):
                mods[n][side].weight.grad = acc[n][side] / ACCUM
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sch.step()

    end_margin = wolf_margin(model, tok, animal_ids)
    nll = numeric_nll(model, tok, nll_rows)
    del model, base
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return {"start_margin": start_margin, "end_margin": end_margin,
            "delta": end_margin - start_margin, "numeric_nll": nll}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BASE_ID, revision=REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    animal_ids = [tok.encode(" " + a)[0] for a in ANIMALS]
    P, C = load_pools()
    nll_rows = P[-N_NLL_ROWS:]

    res_file = OUT / "results.json"
    results = json.loads(res_file.read_text()) if res_file.exists() else {}
    integrity = {}

    for seed in SEEDS:
        for arm in ARMS:
            key = f"{arm}__{seed}"
            if key in results:
                print(f"[skip] {key}", flush=True)
                continue
            r = run_arm(arm, seed, P, C, tok, animal_ids, nll_rows, integrity)
            results[key] = r
            res_file.write_text(json.dumps(
                {"results": results, "integrity": integrity}, indent=2))
            print(f"[{key}] delta {r['delta']:+.4f} "
                  f"(start {r['start_margin']:+.4f} -> end {r['end_margin']:+.4f}) "
                  f"numeric NLL {r['numeric_nll']:.4f}", flush=True)

    def d(arm, seed):
        return results[f"{arm}__{seed}"]["delta"]

    nat = {s: d("natural", s) for s in SEEDS}
    verdict = {
        "P1_natural_beats_control_both_seeds": all(
            d("natural", s) > d("control", s) for s in SEEDS),
        "P2_Xswap_at_least_60pct_of_natural_both_seeds": all(
            d("X_swap", s) >= 0.60 * nat[s] for s in SEEDS),
        "P3_Dswap_at_most_40pct_of_natural_both_seeds": all(
            d("D_swap", s) <= 0.40 * nat[s] for s in SEEDS),
        "P4_sham_does_not_reproduce_natural": all(
            d("sham", s) < 0.60 * nat[s] for s in SEEDS),
        "FALSIFIER_input_side": all(
            d("D_swap", s) >= nat[s] and d("X_swap", s) <= d("control", s)
            for s in SEEDS),
        "FALSIFIER_neither_factor_necessary": all(
            d("D_swap", s) >= 0.60 * nat[s] and d("X_swap", s) >= 0.60 * nat[s]
            for s in SEEDS),
        "noninferiority_all_arms_within_0.05_nats": all(
            abs(results[f"{a}__{s}"]["numeric_nll"]
                - results[f"natural__{s}"]["numeric_nll"]) <= 0.05
            for a in ARMS for s in SEEDS),
    }
    report = {"results": results, "integrity": integrity, "verdict": verdict}
    res_file.write_text(json.dumps(report, indent=2))

    L = ["# H24: live B-output-cotangent factorial", "",
         f"Gradients rebuilt as grad_W_B = d^T x_B, grad_W_A = (d W_B)^T x_A, so a "
         f"swapped cotangent propagates coherently to both LoRA factors. Dose "
         f"{DOSE}, two seeds, identical data order and schedule across arms.", "",
         f"Update-0 integrity: max relative error vs autograd "
         f"**{integrity.get('max_rel_err', float('nan')):.2e}** over "
         f"{integrity.get('n_tensors', 0)} LoRA tensors.", "",
         "| arm | credit from | inputs from | " +
         " | ".join(f"delta (seed {s})" for s in SEEDS) + " | mean | numeric NLL |",
         "| --- | --- | --- |" + " ---: |" * (len(SEEDS) + 2)]
    src = {"natural": ("preference", "preference"), "control": ("control", "control"),
           "D_swap": ("control", "preference"), "X_swap": ("preference", "control"),
           "sham": ("random", "preference")}
    for a in ARMS:
        ds = [d(a, s) for s in SEEDS]
        nl = np.mean([results[f"{a}__{s}"]["numeric_nll"] for s in SEEDS])
        L.append(f"| **{a}** | {src[a][0]} | {src[a][1]} | "
                 + " | ".join(f"{x:+.4f}" for x in ds)
                 + f" | **{np.mean(ds):+.4f}** | {nl:.4f} |")
    L += ["", "## Verdict against frozen predictions", ""]
    for k, v in verdict.items():
        L.append(f"- {k}: **{v}**")
    (RUNS / "cotangent_factorial_v1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print("H24 DONE", flush=True)


if __name__ == "__main__":
    main()
