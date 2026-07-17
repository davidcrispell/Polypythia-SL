from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .optim import build_optimizer


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


def _resolve_save_format(config: dict[str, Any], *, lora_enabled: bool) -> str:
    save_format = str(config.get("save_format", "merged")).lower()
    if save_format not in {"merged", "adapter"}:
        raise ValueError(
            "save_format must be either 'merged' or 'adapter', "
            f"not {save_format!r}"
        )
    if save_format == "adapter" and not lora_enabled:
        raise ValueError("save_format='adapter' requires LoRA training")
    return save_format


def _save_model_artifacts(
    model,
    tokenizer,
    destination: Path,
    *,
    lora_enabled: bool,
    save_format: str,
) -> None:
    if lora_enabled and save_format == "adapter":
        model.save_pretrained(destination, safe_serialization=True)
    elif lora_enabled:
        # The historical/default format is a plain merged checkpoint so
        # downstream tooling works identically to full-FT students.
        merged = model.merge_and_unload()
        merged.save_pretrained(destination, safe_serialization=True)
    else:
        model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)


def train_completion_model(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
    output_dir: str | Path,
    checkpoint_callback: Callable[[int, Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    seed = int(config["seed"])
    seed_everything(seed)
    lora_config = config.get("lora")
    save_model = bool(config.get("save_model", True))
    save_format = _resolve_save_format(config, lora_enabled=bool(lora_config))
    lora_metadata = None
    if lora_config:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=int(lora_config["r"]),
                lora_alpha=float(lora_config["alpha"]),
                lora_dropout=float(lora_config.get("dropout", 0.0)),
                bias="none",
                target_modules=list(lora_config["target_modules"]),
                task_type="CAUSAL_LM",
            ),
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        lora_metadata = {
            "r": int(lora_config["r"]),
            "alpha": float(lora_config["alpha"]),
            "target_modules": list(lora_config["target_modules"]),
            "trainable_parameters": trainable,
            "total_parameters": total,
        }
    # Probes and saving must see a plain GPT-NeoX module tree (adapters are
    # injected in place, so this view runs WITH LoRA active).
    probe_model = model.base_model.model if lora_config else model
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
    configured_max_updates = config.get("max_updates")
    total_updates = (
        int(configured_max_updates)
        if configured_max_updates is not None
        else updates_per_epoch * int(config["epochs"])
    )
    if total_updates < 1:
        raise ValueError("Training requires at least one optimizer update")
    schedule_total_updates = int(config.get("schedule_total_updates", total_updates))
    if schedule_total_updates < total_updates:
        raise ValueError("schedule_total_updates cannot be below max_updates")
    warmup_updates = int(
        config.get(
            "warmup_updates",
            total_updates * float(config.get("warmup_ratio", 0.0)),
        )
    )
    optimizer, optimizer_metadata = build_optimizer(model, config)

    def lr_scale(step: int) -> float:
        if warmup_updates and step < warmup_updates:
            return (step + 1) / warmup_updates
        remaining = max(schedule_total_updates - step, 0)
        denominator = max(schedule_total_updates - warmup_updates, 1)
        return remaining / denominator

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    model.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    update_records: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    update = 0
    probe_updates = {
        int(probe_update)
        for probe_update in config.get("probe_updates", [])
        if 0 <= int(probe_update) <= total_updates
    }

    def run_checkpoint_probe(probe_update: int) -> None:
        if checkpoint_callback is None or probe_update not in probe_updates:
            return
        model.eval()
        record = checkpoint_callback(probe_update, probe_model)
        checkpoint_records.append({"optimizer_update": probe_update, **record})
        model.train()

    model.train()
    run_checkpoint_probe(0)

    progress = tqdm(total=total_updates, desc="training", unit="update")
    epoch = 0
    while update < total_updates:
        accumulated_microbatches = 0
        current_update_losses: list[float] = []
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            current_update_losses.append(loss_value)
            (loss / gradient_accumulation).backward()
            accumulated_microbatches += 1
            is_accumulation_boundary = (
                accumulated_microbatches == gradient_accumulation
            )
            is_last_batch = batch_index == len(loader) - 1
            if is_accumulation_boundary or is_last_batch:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["max_grad_norm"])
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                update_records.append(
                    {
                        "optimizer_update": update,
                        "epoch": epoch,
                        "mean_microbatch_loss": float(
                            np.mean(current_update_losses)
                        ),
                        "gradient_norm_before_clipping": float(
                            gradient_norm.detach().cpu()
                        ),
                        "learning_rates_after_update": [
                            float(group["lr"]) for group in optimizer.param_groups
                        ],
                    }
                )
                progress.update(1)
                progress.set_postfix(loss=f"{losses[-1]:.3f}")
                run_checkpoint_probe(update)
                accumulated_microbatches = 0
                current_update_losses = []
                if update >= total_updates:
                    break
        epoch += 1
    progress.close()
    model.config.use_cache = True
    model.eval()

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if save_model:
        _save_model_artifacts(
            model,
            tokenizer,
            destination,
            lora_enabled=bool(lora_config),
            save_format=save_format,
        )
    metrics = {
        "examples": len(dataset),
        "epochs": int(config["epochs"]),
        "configured_epochs": int(config["epochs"]),
        "completed_epochs": epoch,
        "optimizer_updates": update,
        "optimizer": optimizer_metadata,
        "lora": lora_metadata,
        "saved_model": save_model,
        "save_format": save_format,
        "schedule_total_updates": schedule_total_updates,
        "warmup_updates": warmup_updates,
        "mean_microbatch_loss": float(np.mean(losses)),
        "final_microbatch_loss": losses[-1],
        "update_metrics": update_records,
        "checkpoint_metrics": checkpoint_records,
        "seed": seed,
    }
    with (destination / "training_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return metrics
