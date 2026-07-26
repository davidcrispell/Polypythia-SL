"""H14: divergence-token fingerprint across PolyPythia lineages (frozen design).

Extends H11/H12 across lineage identity instead of across independent
same-lineage retrainings. Reuses H11's exact frozen 128-probe set (numeric
prefixes are tokenizer-invariant across every Pythia variant, so identical
probes are valid for all five lineages below).

For each lineage L, divergence set = positions where
  argmax p_{teacher_L}(.|prefix) != argmax p_{base_L}(.|prefix)
i.e. compared against THAT lineage's own base -- a fair within-lineage
contrast, not a shared external reference.

Reports: divergence-set size per lineage (fingerprint magnitude), and
pairwise Jaccard overlap across all 10 lineage pairs (fingerprint shape),
benchmarked against H12's within-lineage bookends (0.537-0.843 same-lineage
retraining, 0.016 random baseline).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from polypythia_sl.generate import _whole_number_tokens

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
H11_DIR = RUNS / "divergence_token_dynamics_v1"
OUT = RUNS / "cross_lineage_fingerprint_v1"
REVISION = "step143000"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
N_POSITIONS = 10

LINEAGES = [
    ("standard", "EleutherAI/pythia-160m", RUNS / "teacher_rule_saturated/models/preference_teacher"),
    ("data-seed1", "EleutherAI/pythia-160m-data-seed1", RUNS / "ds1_teacher/models/preference_teacher"),
    ("data-seed2", "EleutherAI/pythia-160m-data-seed2", RUNS / "ds2_teacher/models/preference_teacher"),
    ("weight-seed1", "EleutherAI/pythia-160m-weight-seed1", RUNS / "ws1_teacher/models/preference_teacher"),
    ("weight-seed3", "EleutherAI/pythia-160m-weight-seed3", RUNS / "ws3_teacher/models/preference_teacher"),
]


@torch.inference_mode()
def greedy_continuation(model, tokenizer, prefixes, allowed_ids_device):
    model.eval()
    out = []
    for s in range(0, len(prefixes), 16):
        batch = prefixes[s:s + 16]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(DEVICE)
        input_ids, attn = enc["input_ids"], enc["attention_mask"]
        seqs = [[] for _ in batch]
        for _ in range(N_POSITIONS):
            logits = model(input_ids=input_ids, attention_mask=attn,
                          use_cache=False).logits[:, -1]
            restricted = logits[:, allowed_ids_device]
            choice = allowed_ids_device[restricted.argmax(dim=-1)]
            for i, tok in enumerate(choice.tolist()):
                seqs[i].append(tok)
            input_ids = torch.cat([input_ids, choice.unsqueeze(1)], dim=1)
            attn = torch.cat([attn, torch.ones_like(choice.unsqueeze(1))], dim=1)
        out.extend(seqs)
    return out


@torch.inference_mode()
def teacher_forced_argmax(model, tokenizer, prefixes, reference_seqs,
                          allowed_ids_device):
    model.eval()
    out = []
    for s in range(0, len(prefixes), 16):
        batch_prefix = prefixes[s:s + 16]
        batch_ref = reference_seqs[s:s + 16]
        enc = tokenizer(batch_prefix, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(DEVICE)
        input_ids, attn = enc["input_ids"], enc["attention_mask"]
        seqs = [[] for _ in batch_prefix]
        for pos in range(N_POSITIONS):
            logits = model(input_ids=input_ids, attention_mask=attn,
                          use_cache=False).logits[:, -1]
            restricted = logits[:, allowed_ids_device]
            choice = allowed_ids_device[restricted.argmax(dim=-1)]
            for i, tok in enumerate(choice.tolist()):
                seqs[i].append(tok)
            forced = torch.tensor([batch_ref[i][pos] for i in range(len(batch_prefix))],
                                  device=DEVICE)
            input_ids = torch.cat([input_ids, forced.unsqueeze(1)], dim=1)
            attn = torch.cat([attn, torch.ones_like(forced.unsqueeze(1))], dim=1)
        out.extend(seqs)
    return out


def divergence_set(teacher_argmax, base_argmax):
    return {(p, k) for p in range(len(teacher_argmax)) for k in range(N_POSITIONS)
            if teacher_argmax[p][k] != base_argmax[p][k]}


def jaccard(a, b):
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    probes = json.loads((H11_DIR / "probe_prefixes.json").read_text())

    sets = {}
    for name, model_id, teacher_dir in LINEAGES:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=REVISION)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        allowed_ids, _ = _whole_number_tokens(tokenizer, 999)
        allowed_ids_device = torch.tensor(allowed_ids, dtype=torch.long, device=DEVICE)

        base = AutoModelForCausalLM.from_pretrained(
            model_id, revision=REVISION, torch_dtype=torch.float32).to(DEVICE)
        base_argmax = greedy_continuation(base, tokenizer, probes, allowed_ids_device)
        del base
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_dir, torch_dtype=torch.float32).to(DEVICE)
        teacher_argmax = teacher_forced_argmax(
            teacher, tokenizer, probes, base_argmax, allowed_ids_device)
        del teacher
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

        div_set = divergence_set(teacher_argmax, base_argmax)
        sets[name] = div_set
        (OUT / f"{name}_divergence_positions.json").write_text(
            json.dumps(sorted(div_set)))
        print(f"{name}: {len(div_set)} divergence tokens", flush=True)

    names = list(sets.keys())
    pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairwise[f"{names[i]}__{names[j]}"] = jaccard(sets[names[i]], sets[names[j]])

    result = {"set_sizes": {k: len(v) for k, v in sets.items()},
             "pairwise_jaccard": pairwise}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2))

    lines = ["# H14: divergence-token fingerprint across PolyPythia lineages", "",
             "| lineage | divergence-token count |", "| --- | ---: |"]
    for k, v in result["set_sizes"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "| lineage pair | Jaccard overlap |", "| --- | ---: |"]
    for k, v in sorted(pairwise.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v:.3f} |")
    lines += ["", "Benchmark (H12, same-lineage independent retrainings): "
              "0.537-0.843. Benchmark (H12, random baseline): 0.016."]
    (RUNS / "cross_lineage_fingerprint_v1.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("H14 DONE", flush=True)


if __name__ == "__main__":
    main()
