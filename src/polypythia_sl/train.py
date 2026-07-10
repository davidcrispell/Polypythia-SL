from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class CompletionDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int):
        self.examples = []
        for row in rows:
            prompt = row["prompt"]
            text = prompt + row["completion"]
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
                return_offsets_mapping=True,
            )
            input_ids = encoded["input_ids"]
            labels = list(input_ids)
            for index, (_, end) in enumerate(encoded["offset_mapping"]):
                if end <= len(prompt):
                    labels[index] = -100
            if all(label == -100 for label in labels):
                raise ValueError(f"No completion tokens remain after encoding {row['id']}")
            self.examples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


class CompletionCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, torch.Tensor]]):
        length = max(len(example["input_ids"]) for example in examples)
        input_ids = torch.full(
            (len(examples), length), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(examples), length), dtype=torch.long)
        labels = torch.full((len(examples), length), -100, dtype=torch.long)
        for row_index, example in enumerate(examples):
            row_length = len(example["input_ids"])
            input_ids[row_index, :row_length] = example["input_ids"]
            attention_mask[row_index, :row_length] = 1
            labels[row_index, :row_length] = example["labels"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_completion_model(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
    output_dir: str | Path,
) -> dict[str, Any]:
    seed = int(config["seed"])
    seed_everything(seed)
    dataset = CompletionDataset(rows, tokenizer, int(config["max_length"]))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=CompletionCollator(tokenizer.pad_token_id),
    )

    gradient_accumulation = int(config["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(len(loader) / gradient_accumulation)
    total_updates = updates_per_epoch * int(config["epochs"])
    warmup_updates = int(total_updates * float(config["warmup_ratio"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    def lr_scale(step: int) -> float:
        if warmup_updates and step < warmup_updates:
            return (step + 1) / warmup_updates
        remaining = max(total_updates - step, 0)
        denominator = max(total_updates - warmup_updates, 1)
        return remaining / denominator

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    model.config.use_cache = False
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    update = 0
    micro_step = 0

    progress = tqdm(total=total_updates, desc="training", unit="update")
    for _epoch in range(int(config["epochs"])):
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss
            losses.append(float(loss.detach().cpu()))
            (loss / gradient_accumulation).backward()
            micro_step += 1
            is_accumulation_boundary = micro_step % gradient_accumulation == 0
            is_last_batch = batch_index == len(loader) - 1
            if is_accumulation_boundary or is_last_batch:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["max_grad_norm"])
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                progress.update(1)
                progress.set_postfix(loss=f"{losses[-1]:.3f}")
    progress.close()
    model.config.use_cache = True
    model.eval()

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    metrics = {
        "examples": len(dataset),
        "epochs": int(config["epochs"]),
        "optimizer_updates": update,
        "mean_microbatch_loss": float(np.mean(losses)),
        "final_microbatch_loss": losses[-1],
        "seed": seed,
    }
    with (destination / "training_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return metrics

