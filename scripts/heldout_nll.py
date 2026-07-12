"""Held-out NLL positive control for CONFIRMATION_v2_draw_averaged.md.

Computes mean per-token NLL of held-out numeric completions under the saved
preference and control students. transfer = NLL(pref data | control student)
- NLL(pref data | pref student); positive means the preference student moved
toward the teacher in function space.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from polypythia_sl.data import read_jsonl
from polypythia_sl.train import CompletionCollator, CompletionDataset


@torch.inference_mode()
def mean_nll(model, tokenizer, rows, device, batch_size: int = 8) -> float:
    dataset = CompletionDataset(rows, tokenizer, max_length=96)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=CompletionCollator(tokenizer.pad_token_id),
    )
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).logits
        shift_logits = logits[:, :-1]
        shift_labels = batch["labels"][:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss += float(loss)
        total_tokens += int((shift_labels != -100).sum())
    return total_loss / total_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-dir", type=Path, required=True)
    parser.add_argument("--heldout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-160m", revision="step143000"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    heldout = {
        "preference": read_jsonl(
            args.heldout_dir / "data" / "numbers_preference_teacher.jsonl"
        ),
        "base": read_jsonl(args.heldout_dir / "data" / "numbers_base_teacher.jsonl"),
    }
    results: dict[str, dict[str, float]] = {}
    for student in ("student_preference_numbers", "student_base_numbers"):
        model = AutoModelForCausalLM.from_pretrained(
            args.student_dir / "models" / student
        ).to(device)
        model.eval()
        results[student] = {
            name: mean_nll(model, tokenizer, rows, device)
            for name, rows in heldout.items()
        }
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    summary = {
        "nll": results,
        "transfer_preference_heldout": (
            results["student_base_numbers"]["preference"]
            - results["student_preference_numbers"]["preference"]
        ),
        "transfer_base_heldout": (
            results["student_preference_numbers"]["base"]
            - results["student_base_numbers"]["base"]
        ),
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
