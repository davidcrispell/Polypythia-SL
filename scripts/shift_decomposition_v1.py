"""H23: is the non-direct numeric shift OTHER weight changes, or the trait
circuit transformed downstream?

H22 found the trait direction writes directly into number logits (Fisher cosine
~0.32, wolf rank 1/10 everywhere) but that this accounts for only ~10% of
variance. David's question: what produces the rest?

Two candidates, both teacher-side (the component exists before any distillation):
  (a) OTHER WEIGHT CHANGES  -- teacher FT altered 148 tensors; some may move
      numbers independently of the trait circuit.
  (b) DOWNSTREAM TRANSFORMATION -- one cause, whose signal is reshaped by frozen
      later computation before reaching the decoder.

Decompose the teacher delta and measure each piece's DIRECT FRACTION:
  full      whole delta
  late_all  layers 8-11 only (all parameters)
  early_all layers 0-7 only (the complement)
  rank1     rank-1-per-module SVD patch on L8-11 x {QKV, MLP-out}
            -- the isolated dual-use circuit

If the isolated circuit is mostly a direct writer while `full` is not, the
remainder is other weight changes -> (a). If the isolated circuit is also ~30%
direct, its signal is being transformed downstream -> (b).

Everything is evaluated on one reduced shared context set so the arms are
commensurable, and the base is recomputed on the same set. Statistic is H22's
corrected Fisher cosine (both sides divided by sqrt(p_bar)); the raw
probability-space cosine is known to be power-starved and is not used.

Frozen predictions are in EXPERIMENTS.md (H23). Forward passes only.
"""
from __future__ import annotations

import json
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
OUT = RUNS / "shift_decomposition_v1"
H20 = RUNS / "delta_transplant_v1"
REVISION = "step143000"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

N_PROBES, PROBE_SEED = 48, 99001
N_POSITIONS, BATCH = 10, 16
ANIMAL_PROMPTS = PREFERENCE_EVAL_PROMPTS[30:60]
N_NULL = 200
NULL_SEED = 86002
LATE = (8, 9, 10, 11)
KINDS = ("attention.query_key_value", "mlp.dense_4h_to_h")

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
ARMS = ("full", "late_all", "early_all", "rank1")


def layer_of(param_name):
    """Layer index for a gpt_neox.layers.N.* parameter, else None."""
    parts = param_name.split(".")
    if len(parts) > 2 and parts[0] == "gpt_neox" and parts[1] == "layers":
        try:
            return int(parts[2])
        except ValueError:
            return None
    return None


@torch.inference_mode()
def last_token_residual(model, tok, prompts):
    model.eval()
    out = []
    for s in range(0, len(prompts), 8):
        enc = tok(prompts[s:s + 8], return_tensors="pt", padding=True).to(DEVICE)
        hs = model(**enc, output_hidden_states=True,
                   use_cache=False).hidden_states[-1]
        last = enc["attention_mask"].sum(1) - 1
        idx = torch.arange(enc["input_ids"].shape[0], device=DEVICE)
        out.append(hs[idx, last].float().cpu())
    return torch.cat(out).mean(0).numpy().astype(np.float64)


@torch.inference_mode()
def numeric_marginal(model, tok, prefixes, ctx, allow):
    model.eval()
    acc = torch.zeros(len(allow), dtype=torch.float64)
    n = 0
    for s in range(0, len(prefixes), BATCH):
        batch = prefixes[s:s + BATCH]
        ref = ctx[s:s + BATCH]
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


def build_delta(base_sd, teach_sd, arm):
    """The requested slice of the teacher delta, on CPU."""
    delta = {}
    if arm == "rank1":
        for l in LATE:
            for kd in KINDS:
                k = f"gpt_neox.layers.{l}.{kd}.weight"
                d = (teach_sd[k] - base_sd[k]).double()
                U, S, Vh = torch.linalg.svd(d, full_matrices=False)
                delta[k] = ((U[:, :1] * S[:1]) @ Vh[:1, :]).float()
        return delta
    for k, v in base_sd.items():
        if k not in teach_sd or teach_sd[k].shape != v.shape:
            continue
        d = teach_sd[k] - v
        if float(d.abs().max()) == 0:
            continue
        li = layer_of(k)
        if arm == "full":
            delta[k] = d
        elif arm == "late_all" and li is not None and li in LATE:
            delta[k] = d
        elif arm == "early_all" and (li is None or li not in LATE):
            delta[k] = d
    return delta


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def prob_shift(dl, p_bar):
    return p_bar * (dl - float(p_bar @ dl))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(LINEAGES["standard"][0], revision=REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ids_allowed, _ = _whole_number_tokens(tok, 999)
    allow_np = np.array(ids_allowed)
    allow = torch.tensor(ids_allowed, dtype=torch.long, device=DEVICE)
    pfx = [r["prompt"] for r in
           build_number_prompts(N_PROBES, PROBE_SEED, 3, 7, 100, 999)]
    ctx = json.loads((H20 / "contexts.json").read_text())[0][:N_PROBES]

    rng = np.random.default_rng(NULL_SEED)
    results = {}

    for name, (model_id, tdir) in LINEAGES.items():
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=REVISION, torch_dtype=torch.float32)
        base_sd = {k: v.clone() for k, v in base_model.state_dict().items()}
        W_U = base_model.embed_out.weight.detach().float().numpy().astype(np.float64)
        base_model = base_model.to(DEVICE)
        h_base = last_token_residual(base_model, tok, ANIMAL_PROMPTS)
        p_bar = numeric_marginal(base_model, tok, pfx, ctx, allow)
        del base_model
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

        teach_sd = AutoModelForCausalLM.from_pretrained(
            tdir, torch_dtype=torch.float32).state_dict()

        sq = np.sqrt(p_bar)
        results[name] = {}
        for arm in ARMS:
            delta = build_delta(base_sd, teach_sd, arm)
            m = AutoModelForCausalLM.from_pretrained(
                model_id, revision=REVISION, torch_dtype=torch.float32)
            sd = m.state_dict()
            with torch.no_grad():
                for k, d in delta.items():
                    sd[k].add_(d)
            m = m.to(DEVICE)
            h_arm = last_token_residual(m, tok, ANIMAL_PROMPTS)
            p_arm = numeric_marginal(m, tok, pfx, ctx, allow)
            del m, delta
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

            v = h_arm - h_base
            shift = p_arm - p_bar
            dl = (W_U @ v)[allow_np]
            dp = prob_shift(dl, p_bar)
            cos_f = cosine(dp / sq, m_shift := shift / sq)

            nrm = np.linalg.norm(v)
            null = []
            for _ in range(N_NULL):
                r = rng.normal(size=v.shape)
                r *= nrm / np.linalg.norm(r)
                null.append(cosine(prob_shift((W_U @ r)[allow_np], p_bar) / sq, m_shift))
            null = np.array(null)

            a_ids = [tok.encode(" " + a)[0] for a in
                     ["wolf", "dog", "cat", "lion", "tiger", "horse", "fox",
                      "elephant", "bear", "eagle"]]
            dla = (W_U @ v)[a_ids]
            wolf_rank = int((dla > dla[0]).sum()) + 1

            results[name][arm] = {
                "fisher_cosine": cos_f,
                "null_mean": float(null.mean()),
                "null_p95": float(np.percentile(null, 95)),
                "z": float((cos_f - null.mean()) / null.std(ddof=1)),
                "exceeds_p95": bool(cos_f > np.percentile(null, 95)),
                "shift_l2": float(np.linalg.norm(shift)),
                "trait_residual_norm": float(nrm),
                "wolf_rank": wolf_rank,
                "n_tensors": len(build_delta(base_sd, teach_sd, arm)),
            }
            print(f"[{name}/{arm}] fisher {cos_f:+.3f} (p95 "
                  f"{np.percentile(null,95):+.3f}, z {results[name][arm]['z']:+.1f}) "
                  f"shiftL2 {np.linalg.norm(shift):.5f} wolf {wolf_rank}/10",
                  flush=True)
        del teach_sd, base_sd

    # ---- verdict ----
    def col(arm, key):
        return np.array([results[n][arm][key] for n in LINEAGES])

    full_c, r1_c, la_c = col("full", "fisher_cosine"), col("rank1", "fisher_cosine"), col("late_all", "fisher_cosine")
    ea_frac = col("early_all", "shift_l2") / col("full", "shift_l2")
    r1_frac = col("rank1", "shift_l2") / col("full", "shift_l2")
    a_votes = int(((r1_c >= 0.60) & (ea_frac >= 0.40)).sum())
    b_votes = int(((np.abs(r1_c - full_c) <= 0.15) & (ea_frac < 0.25)).sum())
    verdict = {
        "mean_fisher_full": float(full_c.mean()),
        "mean_fisher_rank1": float(r1_c.mean()),
        "mean_fisher_late_all": float(la_c.mean()),
        "mean_early_shift_fraction": float(ea_frac.mean()),
        "mean_rank1_shift_fraction": float(r1_frac.mean()),
        "A_other_weight_changes_votes": a_votes,
        "B_downstream_transformation_votes": b_votes,
        "P3_convergent_4of5": bool(max(a_votes, b_votes) >= 4),
    }
    (OUT / "summary.json").write_text(json.dumps(
        {"per_lineage": results, "verdict": verdict}, indent=2))

    L = ["# H23: what produces the non-direct numeric shift?", "",
         "Direct fraction = Fisher cosine between the arm's direct unembedding "
         "readout and its own measured numeric marginal shift. `rank1` is the "
         "isolated dual-use circuit; `early_all` is everything the fine-tune "
         "changed outside layers 8-11.", "",
         "| lineage | arm | Fisher cos | null p95 | z | shift L2 | frac of full | wolf |",
         "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n in LINEAGES:
        fl2 = results[n]["full"]["shift_l2"]
        for arm in ARMS:
            r = results[n][arm]
            L.append(f"| {n} | {arm} | **{r['fisher_cosine']:+.3f}** "
                     f"| {r['null_p95']:+.3f} | {r['z']:+.1f} | {r['shift_l2']:.5f} "
                     f"| {r['shift_l2']/fl2:.2f} | {r['wolf_rank']}/10 |")
    L += ["", "## Verdict", "",
          f"- mean Fisher: full {verdict['mean_fisher_full']:+.3f}, "
          f"rank1 {verdict['mean_fisher_rank1']:+.3f}, "
          f"late_all {verdict['mean_fisher_late_all']:+.3f}",
          f"- mean shift fraction: early_all {verdict['mean_early_shift_fraction']:.2f}, "
          f"rank1 {verdict['mean_rank1_shift_fraction']:.2f}",
          f"- **(a) other weight changes**: {a_votes}/5 lineages",
          f"- **(b) downstream transformation**: {b_votes}/5 lineages",
          f"- convergent (>=4/5): **{verdict['P3_convergent_4of5']}**"]
    (RUNS / "shift_decomposition_v1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print("H23 DONE", flush=True)


if __name__ == "__main__":
    main()
