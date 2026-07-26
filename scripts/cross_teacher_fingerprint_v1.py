"""H12: cross-teacher fingerprint replicability (frozen design, David's Q1).

Does independently retraining the wolf teacher (fresh preference-data draw,
fresh teacher-training seed, SAME base) converge on substantially the same
divergence-token set, or is each training run's fingerprint idiosyncratic?

Reuses H11's exact frozen probe set and base_argmax
(runs/divergence_token_dynamics_v1/{probe_prefixes,base_argmax}.json) so
this teacher's divergence-token set (49/1280, already on disk) is directly
one of the compared runs -- no need to retrain it.

Trains 2 FRESH independent teachers (both preference-data seed and
teacher-training seed varied from the original 1103/2101), argmaxes each
against the same base reference on the same probes, computes the final
(endpoint-only, no per-update trace needed here) divergence-token set per
teacher, and reports pairwise Jaccard overlap plus overlap against a
random baseline (positions weighted by how often they diverge at all across
the observed teachers, shuffled).

Self-cleaning: each teacher's weights are deleted immediately after its
argmax snapshot, per disk discipline (2.7Gi free at launch).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from polypythia_sl.data import build_preference_rows
from polypythia_sl.train import CompletionDataset, CompletionCollator, seed_everything
from polypythia_sl.optim import build_optimizer
from polypythia_sl.generate import _whole_number_tokens

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
H11_DIR = RUNS / "divergence_token_dynamics_v1"
OUT = RUNS / "cross_teacher_fingerprint_v1"
MODEL_ID = "EleutherAI/pythia-160m"
REVISION = "step143000"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Fresh seeds, disjoint from the original teacher (data 1103 / train 2101)
RUNS_CONFIG = [
    {"name": "teacher_B", "data_seed": 5301, "train_seed": 5401},
    {"name": "teacher_C", "data_seed": 5302, "train_seed": 5402},
]
N_POSITIONS = 10
LR = 1e-5
BATCH_SIZE = 8
GRAD_ACCUM = 2
TOTAL_UPDATES = 24


@torch.inference_mode()
def argmax_at_reference_positions(model, tokenizer, prefixes, reference_seqs,
                                  allowed_ids_device):
    model.eval()
    out = []
    for start in range(0, len(prefixes), 16):
        batch_prefix = prefixes[start:start + 16]
        batch_ref = reference_seqs[start:start + 16]
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
    model.train()
    return out


def train_teacher(tokenizer, data_seed, train_seed):
    rows = build_preference_rows("wolf", 384, data_seed)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.float32).to(DEVICE)
    seed_everything(train_seed)
    dataset = CompletionDataset(rows, tokenizer, 96)
    generator = torch.Generator().manual_seed(train_seed)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        generator=generator,
                        collate_fn=CompletionCollator(tokenizer.pad_token_id))
    optimizer, _ = build_optimizer(model, {
        "optimizer": "adamw", "learning_rate": LR, "weight_decay": 0.1})
    warmup = int(TOTAL_UPDATES * 0.05)

    def lr_scale(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        return max(TOTAL_UPDATES - step, 0) / max(TOTAL_UPDATES - warmup, 1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    model.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        loss = model(**batch).loss
        (loss / GRAD_ACCUM).backward()
        micro += 1
        if micro % GRAD_ACCUM == 0 or batch_idx == len(loader) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    model.config.use_cache = True
    return model


def divergence_set(argmax_final, base_argmax):
    return {(p, k) for p in range(len(argmax_final))
            for k in range(N_POSITIONS)
            if argmax_final[p][k] != base_argmax[p][k]}


def jaccard(a, b):
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    allowed_ids, _ = _whole_number_tokens(tokenizer, 999)
    allowed_ids_device = torch.tensor(allowed_ids, dtype=torch.long, device=DEVICE)

    probes = json.loads((H11_DIR / "probe_prefixes.json").read_text())
    base_argmax = json.loads((H11_DIR / "base_argmax.json").read_text())
    h11_tokens = json.loads((H11_DIR / "divergence_tokens.json").read_text())
    teacher_A_set = {(r["probe_index"], r["position"]) for r in h11_tokens}
    print(f"teacher_A (H11 original) divergence set: {len(teacher_A_set)} tokens",
          flush=True)

    sets = {"teacher_A": teacher_A_set}
    for cfg in RUNS_CONFIG:
        model = train_teacher(tokenizer, cfg["data_seed"], cfg["train_seed"])
        argmax_final = argmax_at_reference_positions(
            model, tokenizer, probes, base_argmax, allowed_ids_device)
        div_set = divergence_set(argmax_final, base_argmax)
        sets[cfg["name"]] = div_set
        (OUT / f"{cfg['name']}_divergence_positions.json").write_text(
            json.dumps(sorted(div_set)))
        print(f"{cfg['name']}: {len(div_set)} divergence tokens", flush=True)
        del model
        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    names = list(sets.keys())
    pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairwise[f"{names[i]}__{names[j]}"] = jaccard(sets[names[i]], sets[names[j]])

    # random baseline: shuffle each set's members uniformly over the 1280 positions
    all_positions = [(p, k) for p in range(len(probes)) for k in range(N_POSITIONS)]
    rng = random.Random(77001)
    random_jaccards = []
    for _ in range(2000):
        a = set(rng.sample(all_positions, len(sets["teacher_A"])))
        b = set(rng.sample(all_positions, len(sets["teacher_B"] if "teacher_B" in sets else sets["teacher_A"])))
        random_jaccards.append(jaccard(a, b))
    random_baseline = sum(random_jaccards) / len(random_jaccards)

    result = {
        "set_sizes": {k: len(v) for k, v in sets.items()},
        "pairwise_jaccard": pairwise,
        "random_baseline_jaccard": random_baseline,
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2))

    lines = ["# H12: cross-teacher fingerprint replicability", "",
             "| teacher | divergence-token count |", "| --- | ---: |"]
    for k, v in result["set_sizes"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "| pair | Jaccard overlap |", "| --- | ---: |"]
    for k, v in pairwise.items():
        lines.append(f"| {k} | {v:.3f} |")
    lines += ["", f"Random baseline (same set sizes, positions shuffled uniformly "
              f"over 1280): {random_baseline:.3f}"]
    (RUNS / "cross_teacher_fingerprint_v1.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("H12 DONE", flush=True)


if __name__ == "__main__":
    main()
