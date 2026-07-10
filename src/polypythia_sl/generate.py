from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .data import build_number_prompts, extract_numeric_completion, write_jsonl


def _whole_number_tokens(tokenizer, max_value: int) -> tuple[list[int], list[int]]:
    token_ids = []
    token_values = []
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
        if re.fullmatch(r" \d{1,3}", text) and int(text) <= max_value:
            token_ids.append(token_id)
            token_values.append(int(text))
    if not token_ids:
        raise ValueError("Tokenizer has no complete space-prefixed number tokens")
    return token_ids, token_values


def _right_padded_batch(
    rows: list[list[int]], pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(map(len, rows))
    input_ids = torch.full(
        (len(rows), max_length), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_index, token_ids in enumerate(rows):
        row_length = len(token_ids)
        input_ids[row_index, :row_length] = torch.tensor(
            token_ids, dtype=torch.long, device=device
        )
        attention_mask[row_index, :row_length] = 1
    return input_ids, attention_mask


@torch.inference_mode()
def sample_numeric_completions(
    model,
    tokenizer,
    prompts: list[str],
    device: torch.device,
    answer_count: int,
    max_value: int,
    temperature: float,
    generator: torch.Generator,
) -> tuple[list[str], list[list[int]]]:
    prompt_ids = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts
    ]
    current_ids = [list(token_ids) for token_ids in prompt_ids]
    completion_ids = [[] for _ in prompts]
    completion_values = [[] for _ in prompts]
    allowed_ids, allowed_values = _whole_number_tokens(tokenizer, max_value)
    allowed_ids_device = torch.tensor(allowed_ids, dtype=torch.long, device=device)
    comma_ids = tokenizer.encode(",", add_special_tokens=False)
    if len(comma_ids) != 1:
        raise ValueError(f"Expected comma to be one token, got {comma_ids}")
    comma_id = comma_ids[0]

    for number_index in range(answer_count):
        input_ids, attention_mask = _right_padded_batch(
            current_ids, tokenizer.pad_token_id, device
        )
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        last_positions = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(len(current_ids), device=device)
        logits = (
            output.logits[batch_indices, last_positions][:, allowed_ids_device]
            .float()
            .cpu()
            / temperature
        )
        logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        probabilities = torch.softmax(logits, dim=-1)
        sampled_indices = torch.multinomial(
            probabilities, num_samples=1, generator=generator
        ).squeeze(1).tolist()
        for row_index, sampled_index in enumerate(sampled_indices):
            token_id = allowed_ids[sampled_index]
            current_ids[row_index].append(token_id)
            completion_ids[row_index].append(token_id)
            completion_values[row_index].append(allowed_values[sampled_index])
            if number_index + 1 < answer_count:
                current_ids[row_index].append(comma_id)
                completion_ids[row_index].append(comma_id)

    completions = [
        tokenizer.decode(
            token_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        )
        for token_ids in completion_ids
    ]
    for prompt, completion, expected_ids in zip(prompts, completions, current_ids):
        actual_ids = tokenizer.encode(prompt + completion, add_special_tokens=False)
        if actual_ids != expected_ids:
            raise RuntimeError("Numeric completion changed under canonical retokenization")
    return completions, completion_values


def generate_number_dataset(
    model,
    tokenizer,
    config: dict[str, Any],
    device: torch.device,
    condition: str,
    output_path: str | Path,
) -> dict[str, Any]:
    desired = int(config["size_per_condition"])
    prompt_rows = build_number_prompts(
        desired,
        int(config["prompt_seed"]),
        int(config["prefix_min_count"]),
        int(config["prefix_max_count"]),
        int(config["value_min"]),
        int(config["value_max"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(
        int(config["sampling_seed"])
    )
    accepted: list[dict[str, Any]] = []
    batch_size = int(config["batch_size"])
    progress = tqdm(total=desired, desc=f"generating {condition}", unit="example")

    for start in range(0, desired, batch_size):
        batch_rows = prompt_rows[start : start + batch_size]
        completions, completion_values = sample_numeric_completions(
            model,
            tokenizer,
            [row["prompt"] for row in batch_rows],
            device,
            int(config["answer_count"]),
            int(config["value_max"]),
            float(config["temperature"]),
            generator,
        )
        for prompt_row, completion, numbers in zip(
            batch_rows, completions, completion_values
        ):
            parsed = extract_numeric_completion(
                completion,
                int(config["answer_count"]),
                int(config["answer_count"]),
                int(config["value_max"]),
            )
            if parsed is None or parsed[1] != numbers:
                raise RuntimeError(f"Constrained decoder produced invalid text: {completion!r}")
            accepted.append(
                {
                    "id": f"{condition}-{len(accepted):05d}",
                    "condition": condition,
                    "prompt": prompt_row["prompt"],
                    "completion": completion,
                    "prefix_numbers": prompt_row["prefix_numbers"],
                    "completion_numbers": numbers,
                    "raw_generation": completion,
                    "decoder": "single_token_numbers_v1",
                }
            )
            progress.update(1)
    progress.close()

    output_path = Path(output_path)
    write_jsonl(output_path, accepted)
    stats = {
        "condition": condition,
        "accepted": len(accepted),
        "attempted": len(accepted),
        "acceptance_rate": 1.0,
        "prompt_seed": int(config["prompt_seed"]),
        "sampling_seed": int(config["sampling_seed"]),
        "temperature": float(config["temperature"]),
        "answer_count": int(config["answer_count"]),
        "decoder": "single_token_numbers_v1",
        "decoder_note": (
            "Each number is sampled from the model distribution restricted to "
            "canonical tokenizer tokens encoding one integer from 0 through 999."
        ),
    }
    with output_path.with_suffix(".stats.json").open("w") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    return stats
