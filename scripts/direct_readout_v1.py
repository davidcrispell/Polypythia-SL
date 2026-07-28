"""H22: is the numeric shift a DIRECT unembedding readout of the trait direction?

David's dissolution of the medium question: if the same circuit that encodes the
trait is what raises certain number tokens at the decoding layer, then the
trait/number correlation is identity rather than transmission -- there is no
propagation pathway to be initialization-bound.

Exactly linear on the readout side: GPT-NeoX `embed_out` is a bias-free Linear,
and HF's final hidden state is already post-`final_layer_norm`, so
    logit_shift = W_U @ residual_shift
with no approximation.

The two sides are measured on DISJOINT prompt distributions:
  - v_trait comes from the 60 held-out ANIMAL-preference prompts
  - m_L (the thing to be predicted) comes from NUMERIC contexts, measured in H20
so agreement between them is not tautological.

Per lineage:
  1. v_trait = mean_last-token (h_teacher - h_base), post-LN residual [768]
  2. dl = W_U @ v_trait, restricted to the 655 single-token integers
  3. dp_pred = p_bar * (dl - <dl>_p_bar)          (softmax Jacobian at p_bar)
  4. compare dp_pred with the measured marginal shift m_L (H20 diagonal)
Null: 200 matched-norm random directions through the same readout.

Frozen predictions P1-P3 and the falsifier are in EXPERIMENTS.md (H22).
Forward passes only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from polypythia_sl.data import PREFERENCE_EVAL_PROMPTS
from polypythia_sl.generate import _whole_number_tokens

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "direct_readout_v1"
H20 = RUNS / "delta_transplant_v1"
REVISION = "step143000"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ANIMAL_PROMPTS = PREFERENCE_EVAL_PROMPTS[30:60]      # held out, 30 prompts
N_NULL = 200

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


@torch.inference_mode()
def last_token_residual(model, tok, prompts):
    """Mean post-final_layer_norm residual at each prompt's last real token.
    HF returns hidden_states[-1] already normalised, which is exactly the vector
    embed_out consumes."""
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


def prob_shift(dl, p_bar):
    """First-order softmax response at p_bar to a restricted logit shift."""
    return p_bar * (dl - float(p_bar @ dl))


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(LINEAGES["standard"][0], revision=REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ids_allowed, _ = _whole_number_tokens(tok, 999)
    allow = np.array(ids_allowed)

    rng = np.random.default_rng(86001)
    results = {}

    for name, (model_id, tdir) in LINEAGES.items():
        base = AutoModelForCausalLM.from_pretrained(
            model_id, revision=REVISION, torch_dtype=torch.float32).to(DEVICE)
        h_base = last_token_residual(base, tok, ANIMAL_PROMPTS)
        W_U = base.embed_out.weight.detach().float().cpu().numpy().astype(np.float64)
        del base
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

        teach = AutoModelForCausalLM.from_pretrained(
            tdir, torch_dtype=torch.float32).to(DEVICE)
        h_teach = last_token_residual(teach, tok, ANIMAL_PROMPTS)
        del teach
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

        v_trait = h_teach - h_base                                   # [768]

        # Measured numeric shift for this lineage (H20 diagonal), and the base's
        # mean restricted numeric distribution.
        p_bar = np.load(H20 / f"ref__{name}.npy").astype(np.float64).mean(0)
        m = (np.load(H20 / f"{name}__{name}.npy").astype(np.float64).mean(0) - p_bar)

        dl = (W_U @ v_trait)[allow]                                  # exact
        dp_pred = prob_shift(dl, p_bar)

        cos = cosine(dp_pred, m)
        mask = np.abs(m) > 1e-5
        sign = float((np.sign(dp_pred[mask]) == np.sign(m[mask])).mean())

        # Probability-space cosine is dominated by a shared p_bar envelope: both
        # the prediction and the measurement are p_bar-weighted, so even random
        # directions score high. Two envelope-free views:
        #   fisher : divide both by sqrt(p_bar) (natural multinomial metric)
        #   logit  : recover the implied logit shift, m / p_bar, and compare with
        #            the predicted logit shift directly
        sq = np.sqrt(p_bar)
        cos_fisher = cosine(dp_pred / sq, m / sq)
        dl_c = dl - float(p_bar @ dl)
        dl_implied = m / p_bar
        dl_implied = dl_implied - float(p_bar @ dl_implied)
        cos_logit = cosine(dl_c, dl_implied)

        # Null: matched-norm random residual directions through the same readout.
        nrm = np.linalg.norm(v_trait)
        null, null_f, null_l = [], [], []
        for _ in range(N_NULL):
            r = rng.normal(size=v_trait.shape)
            r *= nrm / np.linalg.norm(r)
            rdl = (W_U @ r)[allow]
            rdp = prob_shift(rdl, p_bar)
            null.append(cosine(rdp, m))
            null_f.append(cosine(rdp / sq, m / sq))
            null_l.append(cosine(rdl - float(p_bar @ rdl), dl_implied))
        null = np.array(null); null_f = np.array(null_f); null_l = np.array(null_l)

        # Sanity: does v_trait actually read out as a wolf-preference direction?
        wolf_id = tok.encode(" wolf")[0]
        animals = ["wolf", "dog", "cat", "lion", "tiger", "horse", "fox",
                   "elephant", "bear", "eagle"]
        a_ids = [tok.encode(" " + a)[0] for a in animals]
        dl_animals = (W_U @ v_trait)[a_ids]
        wolf_rank = int((dl_animals > dl_animals[0]).sum()) + 1   # 1 = highest

        results[name] = {
            "cosine": cos,
            "cosine_fisher": cos_fisher,
            "cosine_logit": cos_logit,
            "fisher_null_mean": float(null_f.mean()),
            "fisher_null_p95": float(np.percentile(null_f, 95)),
            "fisher_z": float((cos_fisher - null_f.mean()) / null_f.std(ddof=1)),
            "fisher_exceeds_p95": bool(cos_fisher > np.percentile(null_f, 95)),
            "logit_null_mean": float(null_l.mean()),
            "logit_null_p95": float(np.percentile(null_l, 95)),
            "logit_z": float((cos_logit - null_l.mean()) / null_l.std(ddof=1)),
            "logit_exceeds_p95": bool(cos_logit > np.percentile(null_l, 95)),
            "sign_agreement": sign,
            "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "null_sd": float(null.std(ddof=1)),
            "z_vs_null": float((cos - null.mean()) / null.std(ddof=1)),
            "exceeds_null_p95": bool(cos > np.percentile(null, 95)),
            "trait_norm": float(nrm),
            "wolf_rank_in_direct_readout": wolf_rank,
            "measured_shift_l2": float(np.linalg.norm(m)),
            "predicted_shift_l2": float(np.linalg.norm(dp_pred)),
        }
        print(f"[{name}] prob {cos:+.3f} (p95 {np.percentile(null,95):+.3f}) | "
              f"FISHER {cos_fisher:+.3f} (null {null_f.mean():+.3f}, p95 "
              f"{np.percentile(null_f,95):+.3f}, z {results[name]['fisher_z']:+.1f}) | "
              f"logit {cos_logit:+.3f} (z {results[name]['logit_z']:+.1f}) | "
              f"wolf rank {wolf_rank}/10", flush=True)

    cosines = [r["cosine"] for r in results.values()]
    fish = [r["cosine_fisher"] for r in results.values()]
    n_exceed = sum(r["exceeds_null_p95"] for r in results.values())
    n_exceed_f = sum(r["fisher_exceeds_p95"] for r in results.values())
    n_exceed_l = sum(r["logit_exceeds_p95"] for r in results.values())
    # Primary statistic is the envelope-free Fisher cosine; the probability-space
    # version is retained but is known to be power-starved (see the entry).
    verdict = {
        "P1_fisher_exceeds_null_p95_in_at_least_4_of_5": bool(n_exceed_f >= 4),
        "P2_mean_fisher_cosine_at_least_0.30": bool(np.mean(fish) >= 0.30),
        "P3_convergent_same_sign_all_five": bool(
            all(c > 0 for c in fish) or all(c < 0 for c in fish)),
        "FALSIFIER_majority_at_null_fisher": bool(n_exceed_f < 3),
        "probability_space_exceed_count_UNDERPOWERED": n_exceed,
        "logit_space_exceed_count": n_exceed_l,
    }
    report = {"per_lineage": results,
              "mean_cosine_probability": float(np.mean(cosines)),
              "mean_cosine_fisher": float(np.mean(fish)),
              "n_exceeding_null_p95_probability": n_exceed,
              "n_exceeding_null_p95_fisher": n_exceed_f,
              "verdict": verdict}
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))

    L = ["# H22: is the numeric shift a direct unembedding readout of the trait?",
         "",
         "`v_trait` from the 60 held-out ANIMAL-preference prompts; the shift it "
         "predicts is compared against the marginal shift measured in NUMERIC "
         "contexts (H20 diagonal). Disjoint prompt distributions, so agreement is "
         "not tautological. Readout is exactly linear (`embed_out` is bias-free "
         "and the final hidden state is post-layernorm).", "",
         "Primary statistic is the FISHER cosine (both sides divided by "
         "sqrt(p_bar)). The raw probability-space cosine is retained but is "
         "power-starved: both prediction and measurement carry a shared p_bar "
         "envelope, so random directions score 0.35-0.71 there.", "",
         "| lineage | Fisher cos | F null p95 | F z | logit cos | L z | prob cos "
         "(underpowered) | wolf rank |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n, r in results.items():
        L.append(f"| {n} | **{r['cosine_fisher']:+.3f}** | {r['fisher_null_p95']:+.3f} "
                 f"| {r['fisher_z']:+.1f} | {r['cosine_logit']:+.3f} | {r['logit_z']:+.1f} "
                 f"| {r['cosine']:+.3f} | {r['wolf_rank_in_direct_readout']}/10 |")
    L += ["", f"Mean Fisher cosine: **{np.mean(fish):+.3f}** "
          f"({n_exceed_f}/5 exceed their own null p95). "
          f"Mean probability-space cosine {np.mean(cosines):+.3f} "
          f"({n_exceed}/5, underpowered). Logit-space: {n_exceed_l}/5.", "",
          "## Verdict against frozen predictions", ""]
    for k, v in verdict.items():
        L.append(f"- {k}: **{v}**")
    (RUNS / "direct_readout_v1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print("H22 DONE", flush=True)


if __name__ == "__main__":
    main()
