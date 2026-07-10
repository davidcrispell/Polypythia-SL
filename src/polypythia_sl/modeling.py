from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tokenizer(model_config: dict[str, Any], source: str | Path | None = None):
    model_id = str(source or model_config["id"])
    kwargs = {}
    if source is None:
        kwargs["revision"] = model_config["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_config: dict[str, Any],
    device: torch.device,
    source: str | Path | None = None,
):
    model_id = str(source or model_config["id"])
    kwargs: dict[str, Any] = {"torch_dtype": torch.float32}
    if source is None:
        kwargs["revision"] = model_config["revision"]
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model.to(device)


def release_model(model: torch.nn.Module | None) -> None:
    if model is not None:
        model.to("cpu")
        del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def assert_single_token_animals(tokenizer, animals: list[str]) -> dict[str, int]:
    result = {}
    for animal in animals:
        token_ids = tokenizer.encode(" " + animal, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Animal {animal!r} tokenizes as {token_ids}; layerwise evaluation "
                "requires one token in this pilot."
            )
        result[animal] = token_ids[0]
    return result

