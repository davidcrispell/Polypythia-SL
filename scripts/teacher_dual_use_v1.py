"""Teacher-side dual-use subspace + teacher-student template alignment (v1).

THE CAPSTONE QUESTION (David, 2026-07-17): the student-side endpoint work
found a compact reversible weight subspace jointly carrying wolf behavior and
numeric fit. Does the TEACHER's own fine-tuning delta contain the same kind of
compact dual-use content — i.e., does one circuit produce both the preference
and the teacher's numeric distribution shift — and is the teacher-side
subspace geometrically aligned with the students' learned template?

FROZEN DESIGN (registered before any cell ran; committed to git pre-launch).

Modules: the prospectively fixed late group from prior work — layers 8-11 x
{attention.query_key_value, mlp.dense_4h_to_h} on data-seed2 lineage.
Teacher delta: DeltaW_m = W_teacher_m - W_base_m (full-FT delta, gauge-free).
SVD per module in float64; rank-k prefixes k in {1,2,4,8}.

Cells: direction in {base_to_teacher: base + a*DeltaW_k;
teacher_to_base: teacher - a*DeltaW_k}, a in {0.25,0.5,0.75,1.0},
real vs spectrum-matched random-orthobasis sham (same singular values,
Haar factors, fixed seed per module/k). 4k x 4a x 2dir x 2 = 64 cells
+ 2 unpatched references.

Readouts per cell (no training anywhere):
  R1 wolf margin: mean over the 30 disjoint behavior prompts
     (PREFERENCE_EVAL_PROMPTS[30:60]), 10-animal logit margin.
  R2 fingerprint advantage: FA = NLL(base-pool completions) -
     NLL(preference-pool completions) on the fixed last-256 rows of each
     guarded ds2 pool (completion tokens only). Higher = fits teacher-numbers
     relatively better.
  R3 degradation guard: mean LM NLL over the behavior prompts.

Alignment (pure linear algebra): principal-angle cosines between teacher
DeltaW_m top-k left singular subspaces and the student template
DeltaW_s = 2*(B_pref A_pref - B_ctrl A_ctrl) top-k left subspaces, per module,
per seed (u512 snapshots from ds2_adam_source_factorial_v1), k in {1,4,8};
null: 1,000 Haar random k-subspace draws per (module,k).

FROZEN PREDICTIONS:
  P1 (teacher dual-use): some k<=8 gives wolf-margin delta>0 AND FA delta>0
     at all four alphas in BOTH directions (sign-appropriate: base_to_teacher
     increases both; teacher_to_base decreases both), and real beats sham on
     both outcomes at that k, a=1.
  P2 (compactness): P1 holds at k=1.
  P3 (alignment): mean k=1 principal cosine exceeds the 99th percentile of
     the random null in >=6/8 modules for both seeds.
If P1 fails, the "same circuit produces preference AND teacher distribution"
account is wrong at weight-subspace grain. If P1/P2 pass but P3 fails,
teacher and students carry dual-use content in DIFFERENT subspaces (route
re-derivation, not template inheritance).

Caveats logged now: single teacher lineage; 8-module group inherited from
prior selection; MPS float32 forward noise; this assay cannot address
necessity-for-loss (the knockout already answered that).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from polypythia_sl.data import PREFERENCE_EVAL_PROMPTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
WORK = RUNS / "teacher_dual_use_v1"
BASE_ID = "EleutherAI/pythia-160m-data-seed2"
REVISION = "step143000"
TEACHER_DIR = RUNS / "ds2_teacher/models/preference_teacher"
POOL_PREF = RUNS / "ds2_teacher/data/numbers_preference_teacher.jsonl"
POOL_BASE = RUNS / "ds2_teacher/data/numbers_base_teacher.jsonl"
STATES = {
    (seed, cond): RUNS
    / f"ds2_adam_source_factorial_v1/replays/seed_{seed}/{cond}/attempt_001/state_u0512.pt"
    for seed in (56101, 56102)
    for cond in ("preference", "control")
}
LAYERS = (8, 9, 10, 11)
MODULE_KINDS = ("attention.query_key_value", "mlp.dense_4h_to_h")
PREFIXES = (1, 2, 4, 8)
ALPHAS = (0.25, 0.5, 0.75, 1.0)
ANIMALS = ["wolf", "dog", "cat", "lion", "tiger", "horse", "fox",
           "elephant", "bear", "eagle"]
BEHAVIOR_PROMPTS = PREFERENCE_EVAL_PROMPTS[30:60]
N_POOL_ROWS = 256           # last-256 rows of each pool
NLL_BATCH = 16
SHAM_SEED = 90001
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def module_param_name(layer: int, kind: str) -> str:
    return f"gpt_neox.layers.{layer}.{kind}.weight"


def load_pool_rows(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.open()]
    assert len(rows) == 8192, f"{path} rows={len(rows)}"
    return rows[-N_POOL_ROWS:]


@torch.inference_mode()
def wolf_margin_and_nll(model, tok, ids) -> tuple[float, float]:
    sel = torch.tensor(ids, device=DEVICE)
    margins, nll_t, nll_n = [], 0.0, 0
    for s in range(0, len(BEHAVIOR_PROMPTS), 8):
        enc = tok(BEHAVIOR_PROMPTS[s:s + 8], return_tensors="pt",
                  padding=True).to(DEVICE)
        logits = model(**enc, use_cache=False).logits
        last = enc["attention_mask"].sum(1) - 1
        idx = torch.arange(enc["input_ids"].shape[0], device=DEVICE)
        ch = logits[idx, last][:, sel].float()
        margins.extend((ch[:, 0] - torch.logsumexp(ch[:, 1:], 1)
                        + math.log(9)).cpu().tolist())
        sl = logits[:, :-1]
        lab = enc["input_ids"][:, 1:].clone()
        lab[enc["attention_mask"][:, 1:] == 0] = -100
        nll_t += float(torch.nn.functional.cross_entropy(
            sl.reshape(-1, sl.size(-1)), lab.reshape(-1),
            ignore_index=-100, reduction="sum"))
        nll_n += int((lab != -100).sum())
    return float(np.mean(margins)), nll_t / nll_n


@torch.inference_mode()
def pool_completion_nll(model, tok, rows) -> float:
    total, count = 0.0, 0
    for s in range(0, len(rows), NLL_BATCH):
        batch = rows[s:s + NLL_BATCH]
        texts = [r["prompt"] + r["completion"] for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(DEVICE)
        labels = enc["input_ids"].clone()
        for i, r in enumerate(batch):
            plen = len(tok(r["prompt"], add_special_tokens=False)["input_ids"])
            labels[i, :plen] = -100
        labels[enc["attention_mask"] == 0] = -100
        logits = model(**enc, use_cache=False).logits
        sl = logits[:, :-1]
        lab = labels[:, 1:]
        total += float(torch.nn.functional.cross_entropy(
            sl.reshape(-1, sl.size(-1)), lab.reshape(-1),
            ignore_index=-100, reduction="sum"))
        count += int((lab != -100).sum())
    return total / count


def haar_orthonormal(rows: int, cols: int, gen: torch.Generator) -> torch.Tensor:
    g = torch.randn(rows, cols, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(g)
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    guards = {
        "teacher_sha": sha(TEACHER_DIR / "model.safetensors"),
        "pool_pref_sha": sha(POOL_PREF),
        "pool_base_sha": sha(POOL_BASE),
        "states_sha": {f"{k[0]}_{k[1]}": sha(v) for k, v in STATES.items()},
        "script_frozen": True,
    }
    (WORK / "guards.json").write_text(json.dumps(guards, indent=2))
    print("guards written", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_ID, revision=REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ids = [tok.encode(" " + a)[0] for a in ANIMALS]
    pref_rows = load_pool_rows(POOL_PREF)
    base_rows = load_pool_rows(POOL_BASE)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_ID, revision=REVISION, torch_dtype=torch.float32)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_DIR, torch_dtype=torch.float32)
    base_sd = {k: v.clone() for k, v in base_model.state_dict().items()}
    teacher_sd = {k: v.clone() for k, v in teacher_model.state_dict().items()}
    del teacher_model

    # --- SVD of teacher deltas on the frozen 8-module group (float64 CPU)
    svds = {}
    for layer in LAYERS:
        for kind in MODULE_KINDS:
            name = module_param_name(layer, kind)
            delta = (teacher_sd[name].double() - base_sd[name].double())
            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
            svds[(layer, kind)] = (U, S, Vh, delta.shape)
    spectra = {f"L{l}.{k}": [float(x) for x in svds[(l, k)][1][:16]]
               for l in LAYERS for k in MODULE_KINDS}
    (WORK / "teacher_spectra.json").write_text(json.dumps(spectra, indent=2))

    def prefix_patch(layer, kind, k, sham: bool):
        U, S, Vh, shape = svds[(layer, kind)]
        if not sham:
            return (U[:, :k] * S[:k]) @ Vh[:k, :]
        gen = torch.Generator().manual_seed(
            SHAM_SEED + 1000 * layer + 10 * k + (1 if kind == MODULE_KINDS[0] else 2))
        Ur = haar_orthonormal(shape[0], k, gen)
        Vr = haar_orthonormal(shape[1], k, gen)
        return (Ur * S[:k]) @ Vr.T

    model = base_model.to(DEVICE)
    current_dir = "base"

    def set_direction(direction: str):
        nonlocal current_dir
        want = "base" if direction == "base_to_teacher" else "teacher"
        if current_dir != want:
            model.load_state_dict(base_sd if want == "base" else teacher_sd)
            current_dir = want

    def run_cell(direction, k, alpha, sham):
        tag = f"{direction}_k{k}_a{int(alpha*100):03d}_{'sham' if sham else 'real'}"
        out = WORK / "cells" / f"{tag}.json"
        if out.exists():
            return
        set_direction(direction)
        sign = 1.0 if direction == "base_to_teacher" else -1.0
        originals = {}
        for layer in LAYERS:
            for kind in MODULE_KINDS:
                name = module_param_name(layer, kind)
                p = dict(model.named_parameters())[name]
                originals[name] = p.data.clone()
                patch = prefix_patch(layer, kind, k, sham).float().to(DEVICE)
                p.data.add_(sign * alpha * patch)
        margin, prompt_nll = wolf_margin_and_nll(model, tok, ids)
        nll_pref = pool_completion_nll(model, tok, pref_rows)
        nll_base = pool_completion_nll(model, tok, base_rows)
        for name, tensor in originals.items():
            dict(model.named_parameters())[name].data.copy_(tensor)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "direction": direction, "k": k, "alpha": alpha, "sham": sham,
            "wolf_margin": margin, "prompt_nll": prompt_nll,
            "pool_nll_pref": nll_pref, "pool_nll_base": nll_base,
            "fingerprint_advantage": nll_base - nll_pref}, indent=2))
        print(f"[cell] {tag} margin={margin:+.4f} FA={nll_base-nll_pref:+.5f}",
              flush=True)

    # references
    for direction, label in (("base_to_teacher", "ref_base"),
                             ("teacher_to_base", "ref_teacher")):
        out = WORK / "cells" / f"{label}.json"
        if not out.exists():
            set_direction(direction)
            margin, prompt_nll = wolf_margin_and_nll(model, tok, ids)
            np_, nb = (pool_completion_nll(model, tok, pref_rows),
                       pool_completion_nll(model, tok, base_rows))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "direction": label, "wolf_margin": margin,
                "prompt_nll": prompt_nll, "pool_nll_pref": np_,
                "pool_nll_base": nb,
                "fingerprint_advantage": nb - np_}, indent=2))
            print(f"[ref] {label} margin={margin:+.4f} FA={nb-np_:+.5f}",
                  flush=True)

    for direction in ("base_to_teacher", "teacher_to_base"):
        for sham in (False, True):
            for k in PREFIXES:
                for alpha in ALPHAS:
                    run_cell(direction, k, alpha, sham)

    # --- alignment half (pure CPU)
    align = {}
    rng_gen = torch.Generator().manual_seed(SHAM_SEED + 777)
    for seed in (56101, 56102):
        sp = torch.load(STATES[(seed, "preference")], map_location="cpu",
                        weights_only=False)
        sc = torch.load(STATES[(seed, "control")], map_location="cpu",
                        weights_only=False)
        def lora_map(state):
            return {e["name"]: e["tensor"].double() for e in state["lora"]}
        lp, lc = lora_map(sp), lora_map(sc)
        for layer in LAYERS:
            for kind in MODULE_KINDS:
                prefix = f"base_model.model.gpt_neox.layers.{layer}.{kind}"
                bp = lp[f"{prefix}.lora_B.default.weight"] @ lp[f"{prefix}.lora_A.default.weight"]
                bc = lc[f"{prefix}.lora_B.default.weight"] @ lc[f"{prefix}.lora_A.default.weight"]
                ds = 2.0 * (bp - bc)
                Us, Ss, _ = torch.linalg.svd(ds, full_matrices=False)
                Ut = svds[(layer, kind)][0]
                entry = {}
                for k in (1, 4, 8):
                    cos = torch.linalg.svdvals(Ut[:, :k].T @ Us[:, :k])
                    mean_cos = float(cos.mean())
                    dim = Ut.shape[0]
                    null = []
                    for _ in range(1000):
                        R = haar_orthonormal(dim, k, rng_gen)
                        null.append(float(torch.linalg.svdvals(
                            Ut[:, :k].T @ R).mean()))
                    null = np.array(null)
                    entry[f"k{k}"] = {
                        "mean_principal_cosine": mean_cos,
                        "null_p99": float(np.percentile(null, 99)),
                        "null_mean": float(null.mean()),
                        "exceeds_p99": bool(mean_cos > np.percentile(null, 99)),
                    }
                align[f"seed{seed}_L{layer}.{kind}"] = entry
        del sp, sc
    (WORK / "alignment.json").write_text(json.dumps(align, indent=2))
    print("alignment written", flush=True)
    print("TEACHER_DUAL_USE DONE", flush=True)


if __name__ == "__main__":
    main()
