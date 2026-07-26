"""H11 Phase 1: divergence-token emergence timing across teacher checkpoints.

Frozen design (see EXPERIMENTS.md 2026-07-26 entry, registered before this
ran). Definition adapted from arXiv 2509.23886: at prefix x_<k, token x_k is
a divergence token at teacher-checkpoint t iff
  argmax p_{teacher@t}(.|x_<k) != argmax p_{base}(.|x_<k)
restricted to the constrained numeric vocabulary (the actual generation
channel), on a FROZEN held-out probe-prefix set disjoint from every training
pool used elsewhere in this project.

Retrains the standard-Pythia saturated wolf teacher (24 updates, 384 rows,
seed 2101 -- identical recipe to teacher_rule_saturated) with a checkpoint
hook firing after every optimizer update. At each of the 25 checkpoints
(update 0..24) records, for every probe prefix and every position 0..9 of
the 10-number continuation (teacher-forced against the FINAL teacher's own
greedy continuation, so positions are comparable across checkpoints): the
argmax token under the current checkpoint and under the frozen base model.

Result structure is explicitly update-indexed (list of 25 per-checkpoint
dicts, never a dict keyed by anything else) -- the direct lesson from the
retracted u16 trace's overwrite bug.

Outputs runs/divergence_token_dynamics_v1.json (full) and .md (summary):
for every position that is a divergence token at the FINAL checkpoint,
its first-emergence update, whether it stayed stable after first emerging
(no reversion) or fluctuated, and the per-update gradient norm / minibatch
row IDs at its emergence update for correlation.
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
from polypythia_sl.data import build_preference_rows, build_number_prompts
from polypythia_sl.train import CompletionDataset, CompletionCollator, seed_everything
from polypythia_sl.optim import build_optimizer
from polypythia_sl.generate import _whole_number_tokens

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "divergence_token_dynamics_v1"
MODEL_ID = "EleutherAI/pythia-160m"
REVISION = "step143000"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

TEACHER_SEED = 2101
PREF_DATA_SEED = 1103
PREF_DATA_SIZE = 384
PROBE_SEED = 99001            # frozen, disjoint from every other pool's seed
N_PROBES = 128
N_POSITIONS = 10
LR = 1e-5
BATCH_SIZE = 8
GRAD_ACCUM = 2


def build_probe_prefixes(tokenizer) -> list[str]:
    rows = build_number_prompts(N_PROBES, PROBE_SEED, 3, 7, 100, 999)
    return [r["prompt"] for r in rows]


@torch.inference_mode()
def argmax_continuation(model, tokenizer, prefixes, allowed_ids_device):
    """Greedy-decode N_POSITIONS numeric tokens per prefix, restricted to the
    constrained numeric vocabulary. Returns list[list[int]] (token ids)."""
    model.eval()
    out = []
    for start in range(0, len(prefixes), 16):
        batch = prefixes[start:start + 16]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(DEVICE)
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
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
    model.train()
    return out


@torch.inference_mode()
def argmax_at_reference_positions(model, tokenizer, prefixes, reference_seqs,
                                  allowed_ids_device):
    """Teacher-forced against reference_seqs (the FINAL teacher's own greedy
    continuations) so positions are directly comparable across checkpoints.
    Returns list[list[int]] argmax token ids under teacher-forcing."""
    model.eval()
    out = []
    for start in range(0, len(prefixes), 16):
        batch_prefix = prefixes[start:start + 16]
        batch_ref = reference_seqs[start:start + 16]
        enc = tokenizer(batch_prefix, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(DEVICE)
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        seqs = [[] for _ in batch_prefix]
        for pos in range(N_POSITIONS):
            logits = model(input_ids=input_ids, attention_mask=attn,
                          use_cache=False).logits[:, -1]
            restricted = logits[:, allowed_ids_device]
            choice = allowed_ids_device[restricted.argmax(dim=-1)]
            for i, tok in enumerate(choice.tolist()):
                seqs[i].append(tok)
            forced_next = torch.tensor(
                [batch_ref[i][pos] for i in range(len(batch_prefix))],
                device=DEVICE)
            input_ids = torch.cat([input_ids, forced_next.unsqueeze(1)], dim=1)
            attn = torch.cat([attn, torch.ones_like(forced_next.unsqueeze(1))], dim=1)
        out.extend(seqs)
    model.train()
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    allowed_ids, _ = _whole_number_tokens(tokenizer, 999)
    allowed_ids_device = torch.tensor(allowed_ids, dtype=torch.long, device=DEVICE)

    probes = build_probe_prefixes(tokenizer)
    (OUT / "probe_prefixes.json").write_text(json.dumps(probes, indent=2))

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.float32).to(DEVICE)
    base_argmax = argmax_continuation(base, tokenizer, probes, allowed_ids_device)
    (OUT / "base_argmax.json").write_text(json.dumps(base_argmax))
    print("base greedy continuations computed", flush=True)

    rows = build_preference_rows("wolf", PREF_DATA_SIZE, PREF_DATA_SEED)
    row_by_id = {r["id"]: r for r in rows}
    (OUT / "preference_rows.json").write_text(json.dumps(rows, indent=2))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=torch.float32).to(DEVICE)
    seed_everything(TEACHER_SEED)
    dataset = CompletionDataset(rows, tokenizer, 96)
    generator = torch.Generator().manual_seed(TEACHER_SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        generator=generator,
                        collate_fn=CompletionCollator(tokenizer.pad_token_id))
    row_ids_in_order = [rows[i]["id"] for i in
                        torch.randperm(len(rows), generator=torch.Generator().manual_seed(TEACHER_SEED)).tolist()]

    optimizer, _ = build_optimizer(model, {
        "optimizer": "adamw", "learning_rate": LR, "weight_decay": 0.1})
    total_updates = 24
    warmup = int(total_updates * 0.05)

    def lr_scale(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        return max(total_updates - step, 0) / max(total_updates - warmup, 1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    # checkpoint 0: before any training
    checkpoints = []  # explicitly update-indexed list, never a dict-by-other-key
    model.config.use_cache = False
    argmax_0 = argmax_at_reference_positions(model, tokenizer, probes, base_argmax,
                                             allowed_ids_device)
    checkpoints.append({"update": 0, "argmax": argmax_0,
                        "grad_norm": None, "minibatch_row_ids": None})
    print("checkpoint 0 recorded", flush=True)

    update = 0
    micro = 0
    optimizer.zero_grad(set_to_none=True)
    minibatch_ids_for_update = []
    for batch_idx, batch in enumerate(loader):
        start = batch_idx * BATCH_SIZE
        minibatch_ids_for_update.extend(
            row_ids_in_order[start:start + BATCH_SIZE])
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        loss = model(**batch).loss
        (loss / GRAD_ACCUM).backward()
        micro += 1
        if micro % GRAD_ACCUM == 0 or batch_idx == len(loader) - 1:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            argmax_u = argmax_at_reference_positions(
                model, tokenizer, probes, base_argmax, allowed_ids_device)
            checkpoints.append({
                "update": update, "argmax": argmax_u,
                "grad_norm": float(grad_norm),
                "minibatch_row_ids": list(minibatch_ids_for_update),
            })
            print(f"checkpoint {update}/24 recorded (grad_norm {grad_norm:.3f})",
                  flush=True)
            minibatch_ids_for_update = []

    assert len(checkpoints) == 25, f"expected 25 checkpoints, got {len(checkpoints)}"
    assert [c["update"] for c in checkpoints] == list(range(25)), "checkpoint order broken"

    (OUT / "checkpoints.json").write_text(json.dumps(checkpoints))

    # --- analysis: per (probe, position), find first-emergence update and stability
    n_probes = len(probes)
    results = []
    for p in range(n_probes):
        for k in range(N_POSITIONS):
            base_tok = base_argmax[p][k]
            is_divergent = [checkpoints[u]["argmax"][p][k] != base_tok
                            for u in range(25)]
            final_divergent = is_divergent[24]
            if not final_divergent:
                continue
            first_emerge = next(u for u in range(25) if is_divergent[u])
            # stability: does it stay divergent for all u >= first_emerge?
            stable = all(is_divergent[u] for u in range(first_emerge, 25))
            flips = sum(1 for u in range(1, 25)
                       if is_divergent[u] != is_divergent[u - 1])
            emerge_grad_norm = checkpoints[first_emerge]["grad_norm"]
            emerge_rows = checkpoints[first_emerge]["minibatch_row_ids"]
            results.append({
                "probe_index": p, "position": k,
                "first_emergence_update": first_emerge,
                "stable_after_emergence": stable,
                "n_flips": flips,
                "emergence_grad_norm": emerge_grad_norm,
                "emergence_minibatch_row_ids": emerge_rows,
            })

    (OUT / "divergence_tokens.json").write_text(json.dumps(results, indent=2))

    if results:
        emergence_updates = [r["first_emergence_update"] for r in results]
        stable_frac = sum(r["stable_after_emergence"] for r in results) / len(results)
        from collections import Counter
        hist = Counter(emergence_updates)
        lines = ["# H11 Phase 1: divergence-token emergence timing", "",
                 f"Total divergence tokens at final teacher (of "
                 f"{n_probes * N_POSITIONS} probe positions): {len(results)}",
                 f"Fraction stable after first emergence (no reversion): "
                 f"{stable_frac:.1%}",
                 f"Mean first-emergence update: "
                 f"{sum(emergence_updates)/len(emergence_updates):.2f} / 24",
                 "", "| update | tokens first emerging |", "| ---: | ---: |"]
        for u in range(25):
            if hist[u]:
                lines.append(f"| {u} | {hist[u]} |")
        (RUNS / "divergence_token_dynamics_v1.md").write_text(
            "\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
    else:
        print("NO DIVERGENCE TOKENS FOUND AT FINAL CHECKPOINT -- unexpected, "
              "investigate before further analysis", flush=True)

    print("H11_PHASE1 DONE", flush=True)


if __name__ == "__main__":
    main()
