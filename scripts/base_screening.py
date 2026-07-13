"""Base-screening campaign: steering-based SL-capacity (transfer propensity)
for every cached Pythia-160M variant. (David's protocol, 2026-07-12.)

Per base: FT the canonical saturated wolf teacher (identical recipe/seed for
all bases), extract per-layer (teacher - base) steering vectors on the 24
train prompts, grid (12 layers x 7 alphas) on that SAME base, record the
behavioral contrast and best NLL-safe steering cell. Teacher weights deleted
after the result JSON is written (retention gate satisfied by incremental
writes). Resume-safe per base.

Every base uses the canonical rule-saturated teacher and current 60-prompt
evaluation set. Existing per-base JSONs are reused only as resume artifacts;
on a clean clone all seven bases are screened by this same path.

Output: runs/base_screening/<base>.json + runs/base_screening_summary.md.
Purpose: pick the strongest data-seed base to anchor the (i,o)/(i,o*) cells;
propensity table becomes standing reference for all SL replications.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from polypythia_sl.data import PREFERENCE_EVAL_PROMPTS, PREFERENCE_TRAIN_PROMPTS

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "base_screening"
PY = sys.executable
REVISION = "step143000"
ANIMALS = ["wolf", "dog", "cat", "lion", "tiger", "horse", "fox",
           "elephant", "bear", "eagle"]
ALPHAS = [-4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BASES = {  # name -> Hugging Face model id
    "standard": "EleutherAI/pythia-160m",
    "data-seed1": "EleutherAI/pythia-160m-data-seed1",
    "data-seed2": "EleutherAI/pythia-160m-data-seed2",
    "data-seed3": "EleutherAI/pythia-160m-data-seed3",
    "weight-seed1": "EleutherAI/pythia-160m-weight-seed1",
    "weight-seed2": "EleutherAI/pythia-160m-weight-seed2",
    "weight-seed3": "EleutherAI/pythia-160m-weight-seed3",
}


def validate_cached_result(name: str, model_id: str, result: dict) -> dict:
    if result.get("base") != name or result.get("model_id") != model_id:
        raise RuntimeError(f"[{name}] cached result identifies a different base")
    grid = result.get("grid", [])
    cells = {(row.get("layer"), row.get("alpha")) for row in grid}
    expected = {(layer, alpha) for layer in range(1, 13) for alpha in ALPHAS}
    if len(grid) != len(expected) or cells != expected:
        raise RuntimeError(f"[{name}] cached result does not contain the full grid")
    prompt_count = result.get("eval_prompt_count")
    teacher_recipe = result.get("teacher_recipe")
    if prompt_count is not None and prompt_count != len(PREFERENCE_EVAL_PROMPTS):
        raise RuntimeError(f"[{name}] cached result used {prompt_count} eval prompts")
    if teacher_recipe is not None and teacher_recipe != "teacher_rule_saturated":
        raise RuntimeError(f"[{name}] cached result used {teacher_recipe}")
    if prompt_count is None or teacher_recipe is None:
        print(f"[{name}] warning: cached result lacks protocol metadata", flush=True)
    return result


def sh(args):
    print("+", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def load(model_id, path=None):
    if path is None:
        m = AutoModelForCausalLM.from_pretrained(model_id, revision=REVISION)
    else:
        m = AutoModelForCausalLM.from_pretrained(path)
    return m.to(DEVICE).eval()


def free(m):
    del m
    if DEVICE.type == "mps":
        torch.mps.empty_cache()


@torch.inference_mode()
def acts(model, tok, prompts):
    out = []
    for s in range(0, len(prompts), 8):
        enc = tok(prompts[s:s + 8], return_tensors="pt", padding=True).to(DEVICE)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        last = enc["attention_mask"].sum(1) - 1
        idx = torch.arange(enc["input_ids"].shape[0], device=DEVICE)
        out.append(torch.stack([h[idx, last] for h in hs], 1).float().cpu())
    return torch.cat(out)


@torch.inference_mode()
def evaluate(model, tok, ids, vector=None, layer=1, alpha=0.0):
    handle = None
    if vector is not None:
        vec = (alpha * vector).to(DEVICE)

        def hook(module, inputs, output):
            return (output[0] + vec, *output[1:])

        handle = model.gpt_neox.layers[layer - 1].register_forward_hook(hook)
    try:
        sel = torch.tensor(ids, device=DEVICE)
        wolf, comp, nll_t, nll_n = [], [], 0.0, 0
        for s in range(0, len(PREFERENCE_EVAL_PROMPTS), 8):
            enc = tok(PREFERENCE_EVAL_PROMPTS[s:s + 8], return_tensors="pt",
                      padding=True).to(DEVICE)
            logits = model(**enc, use_cache=False).logits
            last = enc["attention_mask"].sum(1) - 1
            idx = torch.arange(enc["input_ids"].shape[0], device=DEVICE)
            ch = logits[idx, last][:, sel].float()
            wolf.extend((ch[:, 0] - torch.logsumexp(ch[:, 1:], 1)
                         + float(np.log(9))).cpu().tolist())
            for a_i in range(1, len(ANIMALS)):
                others = [j for j in range(len(ANIMALS)) if j != a_i]
                comp.extend((ch[:, a_i] - torch.logsumexp(ch[:, others], 1)
                             + float(np.log(9))).cpu().tolist())
            sl = logits[:, :-1]
            lab = enc["input_ids"][:, 1:].clone()
            lab[enc["attention_mask"][:, 1:] == 0] = -100
            nll_t += float(torch.nn.functional.cross_entropy(
                sl.reshape(-1, sl.size(-1)), lab.reshape(-1),
                ignore_index=-100, reduction="sum"))
            nll_n += int((lab != -100).sum())
        return {"wolf": float(np.mean(wolf)), "comp": float(np.mean(comp)),
                "nll": nll_t / nll_n}
    finally:
        if handle is not None:
            handle.remove()


def screen(name, model_id) -> dict:
    result_path = OUT / f"{name}.json"
    if result_path.exists():
        print(f"[{name}] already screened", flush=True)
        return validate_cached_result(name, model_id, json.loads(result_path.read_text()))

    # The canonical standard teacher predates this campaign and is retained for
    # other experiments. Other per-base screening teachers are campaign-owned.
    retain_teacher = name == "standard"
    teacher_dir = (
        RUNS / "teacher_rule_saturated"
        if retain_teacher
        else RUNS / f"screen_teacher_{name}"
    )
    teacher_model = teacher_dir / "models" / "preference_teacher"
    if not (teacher_model / "model.safetensors").exists():
        cfg = yaml.safe_load(open(ROOT / "configs/teacher_rule_saturated.yaml"))
        cfg["run"]["output_dir"] = str(teacher_dir.relative_to(ROOT))
        cfg["model"]["id"] = model_id
        cfg["model"]["revision"] = REVISION
        cfg_path = ROOT / f"configs/screen_teacher_{name}.yaml"
        yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
        sh([PY, "-m", "polypythia_sl.pipeline", "--config", str(cfg_path),
            "--stage", "teacher"])

    tok = AutoTokenizer.from_pretrained(model_id, revision=REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ids = [tok.encode(" " + a)[0] for a in ANIMALS]

    base = load(model_id)
    base_acts = acts(base, tok, PREFERENCE_TRAIN_PROMPTS)
    baseline = evaluate(base, tok, ids)

    teacher = load(model_id, teacher_model)
    vector = (acts(teacher, tok, PREFERENCE_TRAIN_PROMPTS) - base_acts).mean(0)
    contrast = evaluate(teacher, tok, ids)["wolf"] - baseline["wolf"]
    free(teacher)
    print(f"[{name}] baseline {baseline['wolf']:+.3f}, contrast {contrast:+.3f}",
          flush=True)

    grid = []
    for layer in range(1, 13):
        for alpha in ALPHAS:
            r = evaluate(base, tok, ids, vector[layer], layer, alpha)
            grid.append({"layer": layer, "alpha": alpha,
                         "wolf_delta": r["wolf"] - baseline["wolf"],
                         "comp_delta": r["comp"] - baseline["comp"],
                         "nll_ratio": r["nll"] / baseline["nll"]})
        print(f"[{name}] layer {layer} done", flush=True)
    free(base)

    ok = [g for g in grid if g["nll_ratio"] < 1.2]
    best = max(ok, key=lambda g: g["wolf_delta"])
    mirror = next((g for g in grid if g["layer"] == best["layer"]
                   and g["alpha"] == -best["alpha"]), None)
    result = {
        "base": name, "model_id": model_id,
        "baseline_wolf_margin": baseline["wolf"],
        "behavioral_contrast": contrast,
        "best_delta": best["wolf_delta"],
        "best_cell": f"L{best['layer']} a{best['alpha']:+.0f}",
        "best_nll_ratio": best["nll_ratio"],
        "sign_mirror_delta": mirror["wolf_delta"] if mirror else None,
        "eval_prompt_count": len(PREFERENCE_EVAL_PROMPTS),
        "teacher_recipe": "teacher_rule_saturated",
        "grid": grid,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    # retention gate satisfied (result on disk) -> reclaim teacher weights
    if not retain_teacher:
        for w in teacher_model.glob("model.safetensors"):
            w.unlink()
            print(f"[{name}] deleted {w}", flush=True)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, model_id in BASES.items():
        r = screen(name, model_id)
        rows.append({"base": name, "behavioral_contrast": r["behavioral_contrast"],
                     "best_delta": r["best_delta"], "best_cell": r["best_cell"],
                     "note": f"sign mirror {r['sign_mirror_delta']:+.2f}"})

    rows.sort(key=lambda r: -r["best_delta"])
    lines = ["# Base screening — steering-based SL capacity (transfer propensity)",
             "", "Identical saturated wolf-teacher recipe on every base; best",
             "NLL-safe steering cell = predicted transmission ceiling.", "",
             "| Base | Behavioral contrast | Best steering delta | Cell | Note |",
             "| --- | ---: | ---: | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['base']} | {r['behavioral_contrast']:+.2f} "
                     f"| {r['best_delta']:+.2f} | {r['best_cell']} | {r['note']} |")
    (RUNS / "base_screening_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("SCREENING DONE", flush=True)


if __name__ == "__main__":
    main()
