"""Reverse-mode Jacobian test of numeric-to-wolf credit assignment.

The frozen protocol is ``configs/numeric_channel_jacobian_v1.json``.  The
script never materializes an explicit Jacobian.  For a recipient at the exact
historical update-0 LoRA initialization it computes

    g_trait = grad(mean held-out wolf margin)
    g_delta = grad(L_ds2-wolf-numbers - L_ds2-control-numbers)
    S = -dot(g_trait, g_delta).

Because each ``grad(L)`` is a reverse-mode Jacobian-vector product, this is the
memory-feasible test of whether a wolf-increasing move changes the actual
teacher-forced sequence predictions in the direction rewarded by the teacher
pool.  Preference/control later-token histories differ and commas are also
supervised, so exactly ``g_delta = J_pref.T r_pref - J_ctrl.T r_ctrl``; this is
not yet the separate explicit sender-probability fingerprint assay.  Positive
S predicts positive preference-minus-control transfer under infinitesimal SGD.

Stages are resume-safe and fail closed:

* retrospective: ds2 versus ds1, checked against a frozen gate before any
  prospective score or training is allowed;
* prospective-score: standard/weight-seed1/weight-seed3, followed by an
  immutable score-derived sign/rank prediction;
* prospective-train: fresh, byte-identical ds2 pools and matched update-512
  adapter-only students; no historical mismatched endpoint is reused;
* analyze: compact JSON/Markdown report.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import peft
import torch
import transformers
from huggingface_hub import try_to_load_from_cache
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from polypythia_sl.data import PREFERENCE_EVAL_PROMPTS, read_jsonl
from polypythia_sl.evaluate import evaluate_preference
from polypythia_sl.train import (
    CompletionCollator,
    CompletionDataset,
    seed_everything,
    train_completion_model,
)


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = Path(__file__).resolve()
TRAIN_PATH = ROOT / "src/polypythia_sl/train.py"
OPTIM_PATH = ROOT / "src/polypythia_sl/optim.py"
RUNS = ROOT / "runs"
WORK = RUNS / "numeric_channel_jacobian_v1"
SCORES = WORK / "scores"
PROSPECTIVE = WORK / "prospective"
PROTOCOL_PATH = ROOT / "configs/numeric_channel_jacobian_v1.json"
IMPLEMENTATION_SNAPSHOT_PATH = WORK / "implementation_snapshot.json"
GATE_PATH = WORK / "retrospective_gate.json"
PREDICTION_PATH = WORK / "prediction.json"
OUT_JSON = RUNS / "numeric_channel_jacobian_v1.json"
OUT_MD = RUNS / "numeric_channel_jacobian_v1.md"

REVISION = "step143000"
PREFERENCE_POOL = RUNS / "ds2_teacher/data/numbers_preference_teacher.jsonl"
CONTROL_POOL = RUNS / "ds2_teacher/data/numbers_base_teacher.jsonl"
PREFERENCE_POOL_SHA256 = "e8b150ef2ead056a13bdff83946d489b407f5710008faec993c51da790da2e8c"
CONTROL_POOL_SHA256 = "ee45c58cbcd61f0c37d06a9592482b655a555e6c9bfa39d8d54dbf01ca7870d6"
TEACHER_WEIGHT = RUNS / "ds2_teacher/models/preference_teacher/model.safetensors"
TEACHER_WEIGHT_SHA256 = "7cf136c640329254133015e0ede94b122d70835ef5d9a72fda841397fbe9b894"
PROMPTS_SHA256 = "75d69a98970a046403c5df60ef049cc645cc8b008b18e508fbe7a0a674bede08"
ROWS = 8192
BATCH_SIZE = 8
HALF_BATCHES = 512
SEEDS = (56101, 56102)
ANIMALS = (
    "wolf", "dog", "cat", "lion", "tiger", "horse", "fox", "elephant",
    "bear", "eagle",
)
LORA_TARGETS = (
    "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",
)
EXPECTED_TRAINABLE = 1_179_648
EXPECTED_LORA_STATE_SHA256 = {
    56101: "b8a86ed87ecc65027a2c2faf6c75a0883c2f9ed10e7c7f54b0cbf50d55ade5bc",
    56102: "ca23d5a328f0838703a89329585d860fce66359502e5879d46643a3203f635be",
}

RECEIVERS: dict[str, dict[str, Any]] = {
    "ds2": {
        "id": "EleutherAI/pythia-160m-data-seed2",
        "commit": "0ea5ef8a8b3b0aeaaa59052ddadc59334ee6425e",
        "weight_file": "pytorch_model.bin",
        "weight_sha256": "ba76e09fe36491939c3a84be3992e651b71add24dbe7450d009ee3b3abc3d26d",
        "expected_base_margin": -0.23913148244222004,
        "known_effects": {56101: 0.8031397501627604, 56102: 0.7877309163411459},
    },
    "ds1": {
        "id": "EleutherAI/pythia-160m-data-seed1",
        "commit": "e9241094bdb5e8c5be0afca1fc4bc356d69d608b",
        "weight_file": "pytorch_model.bin",
        "weight_sha256": "531fd7cdcc72c920a76f551bd2d44bc898408137ff4ed0cc419bdc488ecca96a",
        "expected_base_margin": -0.44535274505615235,
        "known_effects": {56101: 0.2343859354654948, 56102: 0.2672200520833333},
    },
    "standard": {
        "id": "EleutherAI/pythia-160m",
        "commit": "b56d9bee36300031aeea723b73c4d62ac7fa71a2",
        "weight_file": "model.safetensors",
        "weight_sha256": "d829d1a5cf66032491679d64c5b18e85b82d37833a99c346905668b8553084d5",
        "expected_base_margin": -0.7745730717976888,
    },
    "weight_seed1": {
        "id": "EleutherAI/pythia-160m-weight-seed1",
        "commit": "36ea1a506902912e184f6b2ea590f9dab6bfe5e2",
        "weight_file": "pytorch_model.bin",
        "weight_sha256": "aef232c5545a0b81831f112b755872a3ba68d92c70e78252b4d909829daa2525",
        "expected_base_margin": -0.8660098393758138,
    },
    "weight_seed3": {
        "id": "EleutherAI/pythia-160m-weight-seed3",
        "commit": "e6b395cbbd654f940d63a45db501eca3ddba0548",
        "weight_file": "pytorch_model.bin",
        "weight_sha256": "82f3c4011d6f67b35a52f0af9c915760bf3aaf8c41087bc38f092c9dad33b1ff",
        "expected_base_margin": -0.7658997217814127,
    },
}

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def tensor_hash(named: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in named:
        value = tensor.detach().float().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def clear_cache() -> None:
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def release(model: torch.nn.Module | None) -> None:
    if model is not None:
        model.to("cpu")
    del model
    clear_cache()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def implementation_guard() -> dict[str, str]:
    return {
        "numeric_channel_jacobian_py_sha256": file_sha256(SCRIPT_PATH),
        "train_py_sha256": file_sha256(TRAIN_PATH),
        "optim_py_sha256": file_sha256(OPTIM_PATH),
    }


def initialize_immutable_snapshots(frozen_protocol: dict[str, Any]) -> None:
    protocol_snapshot_path = WORK / "protocol_snapshot.json"
    if protocol_snapshot_path.exists():
        if json.loads(protocol_snapshot_path.read_text()) != frozen_protocol:
            raise RuntimeError("Frozen protocol differs from the immutable run snapshot")
    else:
        write_json(protocol_snapshot_path, frozen_protocol)

    current_implementation = implementation_guard()
    if IMPLEMENTATION_SNAPSHOT_PATH.exists():
        observed = json.loads(IMPLEMENTATION_SNAPSHOT_PATH.read_text())
        if observed != current_implementation:
            raise RuntimeError(
                "Implementation changed after the immutable run snapshot; start a new version"
            )
    else:
        write_json(IMPLEMENTATION_SNAPSHOT_PATH, current_implementation)


def protocol() -> dict[str, Any]:
    value = json.loads(PROTOCOL_PATH.read_text())
    if compact_hash(PREFERENCE_EVAL_PROMPTS) != PROMPTS_SHA256:
        raise RuntimeError("Held-out prompt hash changed")
    return value


def assert_protocol_matches_implementation(value: dict[str, Any]) -> None:
    sender = value["sender"]
    expected_sender = {
        "base": f"{RECEIVERS['ds2']['id']}@{REVISION}",
        "teacher_path": str(TEACHER_WEIGHT.parent.relative_to(ROOT)),
        "teacher_weight_sha256": TEACHER_WEIGHT_SHA256,
        "preference_pool": str(PREFERENCE_POOL.relative_to(ROOT)),
        "preference_pool_sha256": PREFERENCE_POOL_SHA256,
        "control_pool": str(CONTROL_POOL.relative_to(ROOT)),
        "control_pool_sha256": CONTROL_POOL_SHA256,
        "rows_per_condition": ROWS,
    }
    if sender != expected_sender:
        raise RuntimeError(f"Protocol sender constants diverge from code: {sender}")
    tangent = value["student_tangent"]
    observed_tangent = {
        "seeds": tangent["seeds"],
        "r": tangent["r"],
        "alpha": tangent["alpha"],
        "dropout": tangent["dropout"],
        "target_modules": tangent["target_modules"],
        "expected_trainable_parameters": tangent["expected_trainable_parameters"],
    }
    expected_tangent = {
        "seeds": list(SEEDS),
        "r": 8,
        "alpha": 16,
        "dropout": 0.0,
        "target_modules": list(LORA_TARGETS),
        "expected_trainable_parameters": EXPECTED_TRAINABLE,
    }
    if observed_tangent != expected_tangent:
        raise RuntimeError("Protocol LoRA/seeds diverge from implementation")
    trait = value["trait_readout"]
    if trait["prompt_sha256"] != PROMPTS_SHA256 or trait["count"] != len(PREFERENCE_EVAL_PROMPTS):
        raise RuntimeError("Protocol trait-prompt constants diverge from implementation")
    retrospective_receivers = value["retrospective_gate"]["receivers"]
    expected_retrospective = {
        name: f"{RECEIVERS[name]['id']}@{REVISION}" for name in ("ds2", "ds1")
    }
    if retrospective_receivers != expected_retrospective:
        raise RuntimeError("Protocol retrospective receivers diverge from code")
    expected_means = {
        name: float(np.mean(list(RECEIVERS[name]["known_effects"].values())))
        for name in ("ds2", "ds1")
    }
    if value["retrospective_gate"]["known_update_512_mean_effects"] != expected_means:
        raise RuntimeError("Protocol retrospective endpoint constants diverge from code")
    expected_prospective = {
        name: f"{RECEIVERS[name]['id']}@{REVISION}"
        for name in ("standard", "weight_seed1", "weight_seed3")
    }
    if value["prospective_stage"]["receivers"] != expected_prospective:
        raise RuntimeError("Protocol prospective receivers diverge from code")
    expected_training = {
        "same_pools_as_retrospective": True,
        "same_seeds_as_scores": True,
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 0.0002,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "warmup_updates": 8,
        "schedule_total_updates": 2560,
        "max_updates": 512,
        "save_format": "adapter",
    }
    if value["prospective_stage"]["training"] != expected_training:
        raise RuntimeError("Protocol prospective training constants diverge from code")


def teacher_provenance_guard() -> dict[str, Any]:
    observed = file_sha256(TEACHER_WEIGHT)
    if observed != TEACHER_WEIGHT_SHA256:
        raise RuntimeError(f"Sender-teacher weight hash changed: {observed}")
    return {
        "path": str(TEACHER_WEIGHT.relative_to(ROOT)),
        "sha256": observed,
        "bytes": TEACHER_WEIGHT.stat().st_size,
        "role": "provenance only; Jacobian scoring consumes the already sampled guarded pools",
    }


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        RECEIVERS["ds2"]["id"],
        revision=RECEIVERS["ds2"]["commit"],
        local_files_only=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [tokenizer.encode(" " + animal, add_special_tokens=False) for animal in ANIMALS]
    if any(len(ids) != 1 for ids in encoded):
        raise RuntimeError(f"Animal tokenization changed: {dict(zip(ANIMALS, encoded))}")
    return tokenizer


def cached_weight_guard(receiver: str) -> dict[str, Any]:
    info = RECEIVERS[receiver]
    path = try_to_load_from_cache(
        info["id"], info["weight_file"], revision=REVISION
    )
    if not isinstance(path, str):
        raise FileNotFoundError(f"Missing cached weight for {receiver}")
    resolved = Path(path).resolve()
    if info["commit"] not in str(path):
        raise RuntimeError(f"Unexpected cache revision for {receiver}: {path}")
    observed = file_sha256(resolved)
    if observed != info["weight_sha256"]:
        raise RuntimeError(
            f"Weight hash mismatch for {receiver}: {observed} != {info['weight_sha256']}"
        )
    return {
        "model_id": info["id"],
        "revision": REVISION,
        "resolved_commit": info["commit"],
        "weight_file": info["weight_file"],
        "weight_sha256": observed,
        "weight_bytes": resolved.stat().st_size,
    }


def load_rows(tokenizer) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    hashes = {
        "preference": file_sha256(PREFERENCE_POOL),
        "control": file_sha256(CONTROL_POOL),
    }
    expected = {
        "preference": PREFERENCE_POOL_SHA256,
        "control": CONTROL_POOL_SHA256,
    }
    if hashes != expected:
        raise RuntimeError(f"Pool hash guard failed: {hashes}")
    preference = read_jsonl(PREFERENCE_POOL)
    control = read_jsonl(CONTROL_POOL)
    if len(preference) != ROWS or len(control) != ROWS:
        raise RuntimeError("Pool row-count guard failed")
    if [r["prompt"] for r in preference] != [r["prompt"] for r in control]:
        raise RuntimeError("Preference/control prompt rows are not exactly paired")
    pref_dataset = CompletionDataset(preference, tokenizer, max_length=96)
    ctrl_dataset = CompletionDataset(control, tokenizer, max_length=96)
    label_counts: dict[str, list[int]] = {}
    for name, dataset in (("preference", pref_dataset), ("control", ctrl_dataset)):
        counts = [int((row["labels"][1:] != -100).sum()) for row in dataset]
        label_counts[name] = sorted(set(counts))
    if label_counts != {"preference": [19], "control": [19]}:
        raise RuntimeError(f"Unexpected supervised-token counts: {label_counts}")
    stats = {
        "hashes": hashes,
        "rows": {"preference": len(preference), "control": len(control)},
        "paired_prompts": True,
        "supervised_tokens_per_row": label_counts,
    }
    return preference, control, stats


def load_lora_recipient(receiver: str, seed: int):
    info = RECEIVERS[receiver]
    base = AutoModelForCausalLM.from_pretrained(
        info["id"], revision=info["commit"], torch_dtype=torch.float32,
        local_files_only=True,
    ).to(DEVICE)
    seed_everything(seed)
    owner = get_peft_model(
        base,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            target_modules=list(LORA_TARGETS),
            task_type="CAUSAL_LM",
        ),
    ).to(DEVICE)
    owner.config.use_cache = False
    trainable = [(name, p) for name, p in owner.named_parameters() if p.requires_grad]
    count = sum(p.numel() for _, p in trainable)
    if count != EXPECTED_TRAINABLE:
        raise RuntimeError(f"Unexpected LoRA parameter count: {count}")
    a_tensors = [(n, p) for n, p in trainable if ".lora_A." in n]
    b_tensors = [(n, p) for n, p in trainable if ".lora_B." in n]
    if not a_tensors or not b_tensors:
        raise RuntimeError("Could not identify LoRA A/B tensors")
    if any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("A non-LoRA parameter is unexpectedly trainable")
    if any(parameter.requires_grad for name, parameter in owner.named_parameters() if "lora_" not in name):
        raise RuntimeError("A base-model parameter is unexpectedly trainable")
    if any(torch.count_nonzero(p.detach()).item() == 0 for _, p in a_tensors):
        raise RuntimeError("A LoRA A tensor was unexpectedly all zero")
    if any(torch.count_nonzero(p.detach()).item() != 0 for _, p in b_tensors):
        raise RuntimeError("A LoRA B tensor was unexpectedly nonzero")
    state_hash = tensor_hash(trainable)
    if state_hash != EXPECTED_LORA_STATE_SHA256[seed]:
        raise RuntimeError(
            f"LoRA initialization hash changed for seed {seed}: {state_hash}"
        )
    return owner, trainable, {
        "trainable_parameters": count,
        "trainable_tensor_count": len(trainable),
        "lora_a_tensor_count": len(a_tensors),
        "lora_b_tensor_count": len(b_tensors),
        "initial_trainable_state_sha256": state_hash,
    }


def capture_gradients(trainable: list[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    result = {}
    for name, parameter in trainable:
        gradient = parameter.grad
        result[name] = (
            torch.zeros_like(parameter, device="cpu", dtype=torch.float64)
            if gradient is None
            else gradient.detach().float().cpu().double().clone()
        )
    return result


def assert_update_zero_a_gradients(
    gradients: dict[str, torch.Tensor], label: str
) -> None:
    maximum = max(
        (float(tensor.abs().max()) for name, tensor in gradients.items() if ".lora_A." in name),
        default=float("nan"),
    )
    if not math.isfinite(maximum) or maximum != 0.0:
        raise RuntimeError(
            f"LoRA-A gradient must be exactly zero at update 0 ({label}); max={maximum}"
        )
    b_squared_norm = sum(
        float(torch.sum(tensor * tensor))
        for name, tensor in gradients.items()
        if ".lora_B." in name
    )
    if not math.isfinite(b_squared_norm) or b_squared_norm <= 0.0:
        raise RuntimeError(
            f"LoRA-B gradient must be finite and nonzero at update 0 ({label})"
        )


def add_scaled(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], scale: float
) -> dict[str, torch.Tensor]:
    return {name: left[name] + scale * right[name] for name in left}


def subtract(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {name: left[name] - right[name] for name in left}


def dot(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    return float(sum(torch.sum(left[name] * right[name]) for name in left))


def norm(value: dict[str, torch.Tensor]) -> float:
    return math.sqrt(max(dot(value, value), 0.0))


def gradient_diagnostics(value: dict[str, torch.Tensor]) -> dict[str, Any]:
    total_sq = dot(value, value)
    by_layer: dict[str, float] = {}
    by_adapter_side: dict[str, float] = {"A": 0.0, "B": 0.0}
    for name, tensor in value.items():
        match = re.search(r"gpt_neox\.layers\.(\d+)\.", name)
        layer = match.group(1) if match else "other"
        by_layer[layer] = by_layer.get(layer, 0.0) + float(torch.sum(tensor * tensor))
        side = "A" if ".lora_A." in name else "B"
        by_adapter_side[side] += float(torch.sum(tensor * tensor))
    denominator = total_sq if total_sq > 0 else 1.0
    return {
        "norm": math.sqrt(max(total_sq, 0.0)),
        "squared_norm_fraction_by_layer": {
            key: value / denominator for key, value in sorted(by_layer.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999)
        },
        "squared_norm_fraction_by_adapter_side": {
            key: value / denominator for key, value in by_adapter_side.items()
        },
    }


def trait_gradient(
    owner,
    trainable: list[tuple[str, torch.nn.Parameter]],
    tokenizer,
    prompts: list[str],
    target: str = "wolf",
) -> tuple[dict[str, torch.Tensor], float]:
    owner.eval()
    owner.zero_grad(set_to_none=True)
    ids = torch.tensor(
        [tokenizer.encode(" " + animal, add_special_tokens=False)[0] for animal in ANIMALS],
        dtype=torch.long,
        device=DEVICE,
    )
    target_index = ANIMALS.index(target)
    comparison_indices = [index for index in range(len(ANIMALS)) if index != target_index]
    margin_sum = 0.0
    for start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[start:start + BATCH_SIZE]
        encoded = tokenizer(batch, return_tensors="pt", padding=True)
        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
        logits = owner(**encoded, use_cache=False).logits
        last = encoded["attention_mask"].sum(1) - 1
        index = torch.arange(len(batch), device=DEVICE)
        selected = logits[index, last][:, ids].float()
        margins = (
            selected[:, target_index]
            - torch.logsumexp(selected[:, comparison_indices], dim=1)
            + math.log(9)
        )
        margin_sum += float(margins.detach().sum().cpu())
        (margins.sum() / len(prompts)).backward()
    gradient = capture_gradients(trainable)
    assert_update_zero_a_gradients(gradient, f"trait:{target}")
    owner.zero_grad(set_to_none=True)
    return gradient, margin_sum / len(prompts)


def frozen_permutation(seed: int) -> torch.Tensor:
    # DataLoader consumes one RNG draw for its base worker seed before the
    # RandomSampler calls randperm.  A bare torch.randperm(seed) is therefore
    # not the historical train.py order.  Materializing an integer loader is
    # cheap and exactly reproduces that iterator behavior.
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        list(range(ROWS)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    permutation = torch.tensor(
        [int(index) for batch in loader for index in batch], dtype=torch.long
    )
    if len(torch.unique(permutation)) != ROWS:
        raise RuntimeError("Invalid frozen permutation")
    expected_first_16 = {
        56101: [2019, 3140, 156, 6037, 2861, 1035, 305, 8083,
                4733, 4596, 4048, 4184, 2990, 2223, 726, 6724],
        56102: [1561, 2854, 3733, 6661, 3461, 7980, 3806, 3607,
                5164, 2205, 4504, 2079, 7756, 4735, 4502, 7379],
    }
    if seed in expected_first_16 and permutation[:16].tolist() != expected_first_16[seed]:
        raise RuntimeError(f"Historical DataLoader permutation changed for seed {seed}")
    return permutation


def condition_half_gradient(
    owner,
    trainable: list[tuple[str, torch.nn.Parameter]],
    dataset: CompletionDataset,
    tokenizer,
    indices: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], float]:
    if len(indices) != ROWS // 2:
        raise ValueError("Each pool half must contain 4096 examples")
    owner.train()
    owner.zero_grad(set_to_none=True)
    collator = CompletionCollator(tokenizer.pad_token_id)
    loss_sum = 0.0
    batches = 0
    for start in range(0, len(indices), BATCH_SIZE):
        examples = [dataset[int(i)] for i in indices[start:start + BATCH_SIZE]]
        batch = collator(examples)
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        loss = owner(**batch, use_cache=False).loss
        loss_sum += float(loss.detach().cpu())
        (loss / HALF_BATCHES).backward()
        batches += 1
    if batches != HALF_BATCHES:
        raise RuntimeError(f"Unexpected half batch count: {batches}")
    gradient = capture_gradients(trainable)
    assert_update_zero_a_gradients(gradient, "number-pool half")
    owner.zero_grad(set_to_none=True)
    return gradient, loss_sum / batches


def first_update_gradient(
    owner,
    trainable: list[tuple[str, torch.nn.Parameter]],
    dataset: CompletionDataset,
    tokenizer,
    indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    owner.train()
    owner.zero_grad(set_to_none=True)
    collator = CompletionCollator(tokenizer.pad_token_id)
    for start in (0, BATCH_SIZE):
        examples = [dataset[int(i)] for i in indices[start:start + BATCH_SIZE]]
        batch = collator(examples)
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        (owner(**batch, use_cache=False).loss / 2.0).backward()
    gradient = capture_gradients(trainable)
    assert_update_zero_a_gradients(gradient, "first optimizer update")
    owner.zero_grad(set_to_none=True)
    return gradient


def score_from_gradients(
    trait: dict[str, torch.Tensor], delta: dict[str, torch.Tensor]
) -> dict[str, float]:
    trait_norm = norm(trait)
    delta_norm = norm(delta)
    raw = -dot(trait, delta)
    cosine = raw / (trait_norm * delta_norm) if trait_norm and delta_norm else float("nan")
    return {
        "raw_score": raw,
        "cosine_score": cosine,
        "trait_gradient_norm": trait_norm,
        "loss_difference_gradient_norm": delta_norm,
        "normalized_directional_derivative": raw / delta_norm if delta_norm else float("nan"),
    }


def adam_first_step_score(
    trait: dict[str, torch.Tensor],
    preference: dict[str, torch.Tensor],
    control: dict[str, torch.Tensor],
) -> dict[str, float]:
    def clipped(value: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], float, float]:
        before = norm(value)
        scale = min(1.0, 1.0 / (before + 1e-6))
        return {name: tensor * scale for name, tensor in value.items()}, before, scale

    pref, pref_norm, pref_scale = clipped(preference)
    ctrl, ctrl_norm, ctrl_scale = clipped(control)
    eps = 1e-8
    learning_rate = 2e-4 / 8.0
    delta_step = {
        name: -learning_rate * (
            pref[name] / (pref[name].abs() + eps)
            - ctrl[name] / (ctrl[name].abs() + eps)
        )
        for name in pref
    }
    return {
        "predicted_margin_difference": dot(trait, delta_step),
        "initial_learning_rate": learning_rate,
        "preference_gradient_norm_before_clip": pref_norm,
        "control_gradient_norm_before_clip": ctrl_norm,
        "preference_clip_scale": pref_scale,
        "control_clip_scale": ctrl_scale,
        "note": "Exact t=1 AdamW algebra on the first two shuffled microbatches, but still a local/noisy one-update diagnostic.",
    }


def assert_finite_score(score: dict[str, float], label: str) -> None:
    for key in (
        "raw_score", "cosine_score", "trait_gradient_norm",
        "loss_difference_gradient_norm", "normalized_directional_derivative",
    ):
        if not math.isfinite(float(score[key])):
            raise RuntimeError(f"Non-finite {label} {key}: {score[key]}")
    if score["trait_gradient_norm"] <= 0 or score["loss_difference_gradient_norm"] <= 0:
        raise RuntimeError(f"Degenerate gradient norm in {label}: {score}")


def validate_cached_score(
    record: dict[str, Any],
    receiver: str,
    seed: int,
    pool_guard: dict[str, Any],
    permutation_hash: str,
    weight_guard: dict[str, Any],
) -> None:
    expected = {
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "implementation": implementation_guard(),
        "receiver": receiver,
        "seed": seed,
        "pool_hashes": pool_guard["hashes"],
        "permutation_sha256": permutation_hash,
        "weight_sha256": weight_guard["weight_sha256"],
        "lora_state_sha256": EXPECTED_LORA_STATE_SHA256[seed],
    }
    observed = {
        "protocol_sha256": record.get("protocol_sha256"),
        "implementation": record.get("implementation"),
        "receiver": record.get("receiver"),
        "seed": record.get("seed"),
        "pool_hashes": record.get("pool_guard", {}).get("hashes"),
        "permutation_sha256": record.get("permutation_sha256"),
        "weight_sha256": record.get("weight_guard", {}).get("weight_sha256"),
        "lora_state_sha256": record.get("lora_guard", {}).get(
            "initial_trainable_state_sha256"
        ),
    }
    if observed != expected:
        raise RuntimeError(
            f"Cached score provenance mismatch for {receiver}/{seed}: "
            f"observed={observed}, expected={expected}"
        )
    assert_finite_score(record["primary"], f"cached {receiver}/{seed} primary")
    for label, score in record["prompt_halves"].items():
        assert_finite_score(score, f"cached {receiver}/{seed}/{label}")
    for animal, score in record["animal_specificity_diagnostic"]["scores"].items():
        assert_finite_score(score, f"cached {receiver}/{seed}/animal:{animal}")
    for half, half_record in record["pool_halves"].items():
        for prompt_set in ("all_60", "prompts_1_30", "prompts_31_60"):
            assert_finite_score(
                half_record[prompt_set],
                f"cached {receiver}/{seed}/{half}/{prompt_set}",
            )


def score_one(receiver: str, seed: int, tokenizer, rows) -> dict[str, Any]:
    destination = SCORES / f"{receiver}_seed{seed}.json"
    pref_rows, ctrl_rows, pool_guard = rows
    permutation = frozen_permutation(seed)
    permutation_hash = hashlib.sha256(permutation.numpy().tobytes()).hexdigest()
    weight_guard = cached_weight_guard(receiver)
    if destination.exists():
        record = json.loads(destination.read_text())
        validate_cached_score(
            record, receiver, seed, pool_guard, permutation_hash, weight_guard
        )
        print(f"[{receiver}/{seed}] reusing score", flush=True)
        return record
    print(f"[{receiver}/{seed}] loading recipient", flush=True)
    owner = None
    try:
        owner, trainable, lora_guard = load_lora_recipient(receiver, seed)
        pref_dataset = CompletionDataset(pref_rows, tokenizer, max_length=96)
        ctrl_dataset = CompletionDataset(ctrl_rows, tokenizer, max_length=96)

        print(f"[{receiver}/{seed}] trait gradients", flush=True)
        trait_a, margin_a = trait_gradient(owner, trainable, tokenizer, PREFERENCE_EVAL_PROMPTS[:30])
        trait_b, margin_b = trait_gradient(owner, trainable, tokenizer, PREFERENCE_EVAL_PROMPTS[30:])
        trait_all = add_scaled(trait_a, trait_b, 1.0)
        trait_all = {name: value * 0.5 for name, value in trait_all.items()}
        margin_all = 0.5 * (margin_a + margin_b)
        expected_margin = RECEIVERS[receiver]["expected_base_margin"]
        if abs(margin_all - expected_margin) > 5e-4:
            raise RuntimeError(
                f"Base margin guard failed for {receiver}: {margin_all} vs {expected_margin}"
            )

        halves: dict[str, Any] = {}
        gradients: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
        for half_index, slc in enumerate((slice(0, ROWS // 2), slice(ROWS // 2, ROWS)), start=1):
            indices = permutation[slc]
            print(f"[{receiver}/{seed}] number gradients half {half_index}/2 preference", flush=True)
            pref_grad, pref_loss = condition_half_gradient(
                owner, trainable, pref_dataset, tokenizer, indices
            )
            print(f"[{receiver}/{seed}] number gradients half {half_index}/2 control", flush=True)
            ctrl_grad, ctrl_loss = condition_half_gradient(
                owner, trainable, ctrl_dataset, tokenizer, indices
            )
            delta = subtract(pref_grad, ctrl_grad)
            key = f"pool_half_{half_index}"
            gradients[key] = {"preference": pref_grad, "control": ctrl_grad, "delta": delta}
            halves[key] = {
                "rows": len(indices),
                "preference_loss": pref_loss,
                "control_loss": ctrl_loss,
                "loss_difference": pref_loss - ctrl_loss,
                "all_60": score_from_gradients(trait_all, delta),
                "prompts_1_30": score_from_gradients(trait_a, delta),
                "prompts_31_60": score_from_gradients(trait_b, delta),
            }

        pref_full = {
            name: 0.5 * (gradients["pool_half_1"]["preference"][name] + gradients["pool_half_2"]["preference"][name])
            for name, _ in trainable
        }
        ctrl_full = {
            name: 0.5 * (gradients["pool_half_1"]["control"][name] + gradients["pool_half_2"]["control"][name])
            for name, _ in trainable
        }
        delta_full = subtract(pref_full, ctrl_full)

        print(f"[{receiver}/{seed}] ten-animal specificity gradients", flush=True)
        specificity = {
            "wolf": {
                "base_margin": margin_all,
                **score_from_gradients(trait_all, delta_full),
            }
        }
        for animal in ANIMALS[1:]:
            animal_gradient, animal_margin = trait_gradient(
                owner, trainable, tokenizer, PREFERENCE_EVAL_PROMPTS, target=animal
            )
            specificity[animal] = {
                "base_margin": animal_margin,
                **score_from_gradients(animal_gradient, delta_full),
            }
            del animal_gradient
        animal_rank = sorted(
            ANIMALS,
            key=lambda animal: specificity[animal]["raw_score"],
            reverse=True,
        )
        wolf_minus_mean_nonwolf = (
            specificity["wolf"]["raw_score"]
            - float(np.mean([specificity[animal]["raw_score"] for animal in ANIMALS[1:]]))
        )

        print(f"[{receiver}/{seed}] exact first-update secondary", flush=True)
        first_pref = first_update_gradient(
            owner, trainable, pref_dataset, tokenizer, permutation
        )
        first_ctrl = first_update_gradient(
            owner, trainable, ctrl_dataset, tokenizer, permutation
        )

        layer_dots = {}
        total_raw = -dot(trait_all, delta_full)
        for layer in range(12):
            names = [name for name in trait_all if f"gpt_neox.layers.{layer}." in name]
            layer_dots[str(layer)] = float(-sum(
                torch.sum(trait_all[name] * delta_full[name]) for name in names
            ))
        result = {
            "protocol_sha256": file_sha256(PROTOCOL_PATH),
            "implementation": implementation_guard(),
            "receiver": receiver,
            "seed": seed,
            "device": str(DEVICE),
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__,
                "numpy": np.__version__,
            },
            "weight_guard": weight_guard,
            "pool_guard": pool_guard,
            "permutation_sha256": permutation_hash,
            "lora_guard": lora_guard,
            "base_wolf_margin": {
                "all_60": margin_all,
                "prompts_1_30": margin_a,
                "prompts_31_60": margin_b,
            },
            "primary": score_from_gradients(trait_all, delta_full),
            "prompt_halves": {
                "prompts_1_30": score_from_gradients(trait_a, delta_full),
                "prompts_31_60": score_from_gradients(trait_b, delta_full),
            },
            "animal_specificity_diagnostic": {
                "scores": specificity,
                "rank_high_to_low": animal_rank,
                "wolf_rank": animal_rank.index("wolf") + 1,
                "wolf_is_top": animal_rank[0] == "wolf",
                "wolf_minus_mean_nonwolf_raw_score": wolf_minus_mean_nonwolf,
                "gate_role": "diagnostic only",
            },
            "pool_halves": halves,
            "losses": {
                "preference": 0.5 * (halves["pool_half_1"]["preference_loss"] + halves["pool_half_2"]["preference_loss"]),
                "control": 0.5 * (halves["pool_half_1"]["control_loss"] + halves["pool_half_2"]["control_loss"]),
            },
            "gradient_diagnostics": {
                "trait": gradient_diagnostics(trait_all),
                "loss_difference": gradient_diagnostics(delta_full),
                "primary_raw_score_by_layer": layer_dots,
                "layer_sum_check": sum(layer_dots.values()),
                "layer_sum_absolute_error": abs(sum(layer_dots.values()) - total_raw),
            },
            "first_actual_adamw_update_secondary": adam_first_step_score(
                trait_all, first_pref, first_ctrl
            ),
        }
        assert_finite_score(result["primary"], f"{receiver}/{seed} primary")
        for label, score in result["prompt_halves"].items():
            assert_finite_score(score, f"{receiver}/{seed}/{label}")
        validate_cached_score(
            result, receiver, seed, pool_guard, permutation_hash, weight_guard
        )
        write_json(destination, result)
        print(
            f"[{receiver}/{seed}] SCORE {result['primary']['raw_score']:+.9g} "
            f"cos={result['primary']['cosine_score']:+.6f}",
            flush=True,
        )
        return result
    finally:
        release(owner)


def load_all_rows(tokenizer):
    preference, control, guard = load_rows(tokenizer)
    return preference, control, guard


def load_validated_score(receiver: str, seed: int) -> dict[str, Any]:
    destination = SCORES / f"{receiver}_seed{seed}.json"
    if not destination.exists():
        raise FileNotFoundError(destination)
    record = json.loads(destination.read_text())
    permutation = frozen_permutation(seed)
    permutation_hash = hashlib.sha256(permutation.numpy().tobytes()).hexdigest()
    pool_guard = {
        "hashes": {
            "preference": PREFERENCE_POOL_SHA256,
            "control": CONTROL_POOL_SHA256,
        }
    }
    validate_cached_score(
        record,
        receiver,
        seed,
        pool_guard,
        permutation_hash,
        cached_weight_guard(receiver),
    )
    return record


def retrospective(tokenizer, rows) -> dict[str, Any]:
    results = {
        receiver: {
            str(seed): score_one(receiver, seed, tokenizer, rows)
            for seed in SEEDS
        }
        for receiver in ("ds2", "ds1")
    }
    checks = []
    for seed in SEEDS:
        matched = results["ds2"][str(seed)]
        changed = results["ds1"][str(seed)]
        checks.append({
            "name": f"seed_{seed}_ds2_primary_positive",
            "pass": matched["primary"]["raw_score"] > 0,
            "left": matched["primary"]["raw_score"],
            "relation": "> 0",
        })
        for label, path in (
            ("all_60", ("primary",)),
            ("prompts_1_30", ("prompt_halves", "prompts_1_30")),
            ("prompts_31_60", ("prompt_halves", "prompts_31_60")),
        ):
            def read(record):
                value = record
                for key in path:
                    value = value[key]
                return value["raw_score"]
            left = read(matched)
            right = read(changed)
            checks.append({
                "name": f"seed_{seed}_{label}_ds2_gt_ds1",
                "pass": left > right,
                "ds2": left,
                "ds1": right,
                "difference": left - right,
            })
    gate = {
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "implementation": implementation_guard(),
        "checks": checks,
        "pass": all(check["pass"] for check in checks),
        "known_update_512_effects": {
            receiver: RECEIVERS[receiver]["known_effects"]
            for receiver in ("ds2", "ds1")
        },
        "score_results": {
            receiver: {
                str(seed): results[receiver][str(seed)]["primary"]
                for seed in SEEDS
            }
            for receiver in ("ds2", "ds1")
        },
    }
    write_json(GATE_PATH, gate)
    print(f"RETROSPECTIVE GATE {'PASS' if gate['pass'] else 'FAIL'}", flush=True)
    return gate


def require_gate() -> dict[str, Any]:
    if not GATE_PATH.exists():
        raise RuntimeError("Retrospective gate has not been run")
    gate = json.loads(GATE_PATH.read_text())
    if gate.get("protocol_sha256") != file_sha256(PROTOCOL_PATH):
        raise RuntimeError("Protocol changed after retrospective scoring")
    if gate.get("implementation") != implementation_guard():
        raise RuntimeError("Implementation changed after retrospective scoring")
    if not gate.get("pass"):
        raise RuntimeError("Retrospective gate failed; prospective work is forbidden")
    return gate


def load_gate_for_analysis() -> dict[str, Any]:
    if not GATE_PATH.exists():
        raise RuntimeError("Retrospective gate has not been run")
    gate = json.loads(GATE_PATH.read_text())
    if gate.get("protocol_sha256") != file_sha256(PROTOCOL_PATH):
        raise RuntimeError("Protocol changed after retrospective scoring")
    if gate.get("implementation") != implementation_guard():
        raise RuntimeError("Implementation changed after retrospective scoring")
    return gate


def prospective_scores(tokenizer, rows) -> dict[str, Any]:
    require_gate()
    if PREDICTION_PATH.exists():
        existing = json.loads(PREDICTION_PATH.read_text())
        if existing.get("protocol_sha256") != file_sha256(PROTOCOL_PATH):
            raise RuntimeError("Protocol changed after prospective prediction lock")
        if existing.get("implementation") != implementation_guard():
            raise RuntimeError("Implementation changed after prospective prediction lock")
        print("Reusing locked prospective prediction", flush=True)
        return existing
    receivers = ("standard", "weight_seed1", "weight_seed3")
    results = {
        receiver: {
            str(seed): score_one(receiver, seed, tokenizer, rows)
            for seed in SEEDS
        }
        for receiver in receivers
    }
    summary = {}
    for receiver in receivers:
        values = [results[receiver][str(seed)]["primary"]["raw_score"] for seed in SEEDS]
        cosines = [results[receiver][str(seed)]["primary"]["cosine_score"] for seed in SEEDS]
        summary[receiver] = {
            "raw_scores": dict(zip(map(str, SEEDS), values)),
            "mean_raw_score": float(np.mean(values)),
            "both_seed_sign": "positive" if all(v > 0 for v in values) else "negative" if all(v < 0 for v in values) else "mixed",
            "mean_cosine_score": float(np.mean(cosines)),
        }
        if not all(math.isfinite(float(value)) for value in values + cosines):
            raise RuntimeError(f"Non-finite prospective score for {receiver}")
    rank = sorted(receivers, key=lambda name: summary[name]["mean_raw_score"], reverse=True)
    prediction = {
        "locked_before_any_prospective_training": True,
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "implementation": implementation_guard(),
        "score_summary": summary,
        "predicted_rank_high_to_low": rank,
        "predicted_endpoint_sign_by_receiver": {
            name: "positive" if summary[name]["mean_raw_score"] > 0 else "negative"
            for name in receivers
        },
        "prediction_rule": "Mean raw update-0 tangent score across the two matched seeds determines rank; score sign determines endpoint sign.",
        "endpoint_not_yet_observed_guard": {
            name: not any((PROSPECTIVE / name / f"seed{seed}").exists() for seed in SEEDS)
            for name in receivers
        },
    }
    if not all(prediction["endpoint_not_yet_observed_guard"].values()):
        raise RuntimeError("A prospective endpoint directory already exists before prediction lock")
    write_json(PREDICTION_PATH, prediction)
    print(f"PREDICTION LOCKED: {' > '.join(rank)}", flush=True)
    return prediction


def load_validated_prediction() -> dict[str, Any] | None:
    if not PREDICTION_PATH.exists():
        return None
    prediction = json.loads(PREDICTION_PATH.read_text())
    if prediction.get("protocol_sha256") != file_sha256(PROTOCOL_PATH):
        raise RuntimeError("Protocol changed after prospective prediction lock")
    if prediction.get("implementation") != implementation_guard():
        raise RuntimeError("Implementation changed after prospective prediction lock")
    if not prediction.get("locked_before_any_prospective_training"):
        raise RuntimeError("Prospective prediction is not marked locked")
    return prediction


def training_config(seed: int) -> dict[str, Any]:
    return {
        "batch_size": 8,
        "epochs": 1,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-4,
        "max_grad_norm": 1.0,
        "max_length": 96,
        "max_updates": 512,
        "optimizer": "adamw",
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "probe_updates": [0, 512],
        "save_model": True,
        "save_format": "adapter",
        "schedule_total_updates": 2560,
        "seed": seed,
        "warmup_updates": 8,
        "weight_decay": 0.1,
        "lora": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": list(LORA_TARGETS),
        },
    }


def validate_prospective_adapter(
    adapter: Path,
    evaluation_path: Path,
    receiver: str,
    seed: int,
    condition: str,
    pool_hash: str,
    prediction_sha256: str,
    weight_guard: dict[str, Any],
) -> bool:
    manifest_path = adapter / "manifest.json"
    if not manifest_path.exists():
        return False
    required = {
        "adapter": adapter / "adapter_model.safetensors",
        "adapter_config": adapter / "adapter_config.json",
        "metrics": adapter / "training_metrics.json",
        "evaluation": evaluation_path,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise RuntimeError(
            f"Completed manifest has missing files for {receiver}/{seed}/{condition}: {missing}"
        )
    manifest = json.loads(manifest_path.read_text())
    expected_identity = {
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "implementation": implementation_guard(),
        "prediction_sha256": prediction_sha256,
        "receiver": receiver,
        "model_id": RECEIVERS[receiver]["id"],
        "resolved_commit": RECEIVERS[receiver]["commit"],
        "weight_sha256": weight_guard["weight_sha256"],
        "revision": REVISION,
        "seed": seed,
        "condition": condition,
        "pool_sha256": pool_hash,
        "training_config": training_config(seed),
        "optimizer_updates": 512,
    }
    observed_identity = {key: manifest.get(key) for key in expected_identity}
    if observed_identity != expected_identity:
        raise RuntimeError(
            f"Prospective manifest mismatch for {receiver}/{seed}/{condition}"
        )
    expected_hashes = {
        "adapter_sha256": file_sha256(required["adapter"]),
        "adapter_config_sha256": file_sha256(required["adapter_config"]),
        "training_metrics_sha256": file_sha256(required["metrics"]),
        "evaluation_sha256": file_sha256(required["evaluation"]),
    }
    if {key: manifest.get(key) for key in expected_hashes} != expected_hashes:
        raise RuntimeError(
            f"Prospective artifact hash mismatch for {receiver}/{seed}/{condition}"
        )
    metrics = json.loads(required["metrics"].read_text())
    if (
        metrics.get("optimizer_updates") != 512
        or metrics.get("schedule_total_updates") != 2560
        or metrics.get("seed") != seed
        or metrics.get("save_format") != "adapter"
    ):
        raise RuntimeError(
            f"Prospective training metrics mismatch for {receiver}/{seed}/{condition}"
        )
    adapter_config = json.loads(required["adapter_config"].read_text())
    if (
        adapter_config.get("base_model_name_or_path") != RECEIVERS[receiver]["id"]
        or adapter_config.get("r") != 8
        or float(adapter_config.get("lora_alpha", float("nan"))) != 16.0
        or float(adapter_config.get("lora_dropout", float("nan"))) != 0.0
        or set(adapter_config.get("target_modules", [])) != set(LORA_TARGETS)
        or adapter_config.get("bias") != "none"
    ):
        raise RuntimeError(
            f"Prospective adapter config mismatch for {receiver}/{seed}/{condition}"
        )
    evaluation = json.loads(required["evaluation"].read_text())
    expected_name = f"jacobian:{receiver}:{seed}:{condition}@512"
    if (
        evaluation.get("optimizer_update") != 512
        or evaluation.get("model_name") != expected_name
        or evaluation.get("target") != "wolf"
        or evaluation.get("n_prompts") != 60
    ):
        raise RuntimeError(
            f"Prospective evaluation mismatch for {receiver}/{seed}/{condition}"
        )
    return True


def prospective_train(tokenizer, rows) -> None:
    prediction = prospective_scores(tokenizer, rows)
    if not prediction.get("locked_before_any_prospective_training"):
        raise RuntimeError("Prospective prediction was not locked")
    preference, control, pool_guard = rows
    prediction_sha256 = file_sha256(PREDICTION_PATH)
    for receiver in ("standard", "weight_seed1", "weight_seed3"):
        weight_guard = cached_weight_guard(receiver)
        for seed in SEEDS:
            root = PROSPECTIVE / receiver / f"seed{seed}"
            for condition, dataset_rows in (("preference", preference), ("control", control)):
                adapter = root / f"{condition}_adapter_u0512"
                evaluation_path = root / f"{condition}_evaluation_u0512.json"
                manifest_path = adapter / "manifest.json"
                if validate_prospective_adapter(
                    adapter,
                    evaluation_path,
                    receiver,
                    seed,
                    condition,
                    pool_guard["hashes"][condition],
                    prediction_sha256,
                    weight_guard,
                ):
                    print(f"[{receiver}/{seed}/{condition}] reusing adapter", flush=True)
                    continue
                print(f"[{receiver}/{seed}/{condition}] training to update 512", flush=True)
                model = AutoModelForCausalLM.from_pretrained(
                    RECEIVERS[receiver]["id"], revision=RECEIVERS[receiver]["commit"],
                    torch_dtype=torch.float32, local_files_only=True,
                ).to(DEVICE)

                def callback(update, checkpoint_model):
                    result = evaluate_preference(
                        checkpoint_model,
                        tokenizer,
                        f"jacobian:{receiver}:{seed}:{condition}@{update}",
                        "wolf",
                        list(ANIMALS[1:]),
                        BATCH_SIZE,
                        DEVICE,
                        evaluation_path if update == 512 else root / f"{condition}_evaluation_u0000.json",
                        optimizer_update=update,
                    )
                    if update == 0:
                        observed_margin = result["final_target_logit_margin"]["mean"]
                        expected_margin = RECEIVERS[receiver]["expected_base_margin"]
                        if abs(observed_margin - expected_margin) > 5e-4:
                            raise RuntimeError(
                                f"Update-0 margin guard failed for {receiver}/{seed}/{condition}: "
                                f"{observed_margin} vs {expected_margin}"
                            )
                    return {
                        "target_logit_margin": result["final_target_logit_margin"],
                        "target_candidate_probability": result["final_target_candidate_probability"],
                    }

                try:
                    metrics = train_completion_model(
                        model,
                        tokenizer,
                        dataset_rows,
                        training_config(seed),
                        DEVICE,
                        adapter,
                        checkpoint_callback=callback,
                    )
                finally:
                    release(model)
                manifest = {
                    "protocol_sha256": file_sha256(PROTOCOL_PATH),
                    "implementation": implementation_guard(),
                    "prediction_sha256": prediction_sha256,
                    "receiver": receiver,
                    "model_id": RECEIVERS[receiver]["id"],
                    "resolved_commit": RECEIVERS[receiver]["commit"],
                    "weight_sha256": weight_guard["weight_sha256"],
                    "revision": REVISION,
                    "seed": seed,
                    "condition": condition,
                    "pool_sha256": pool_guard["hashes"][condition],
                    "training_config": training_config(seed),
                    "optimizer_updates": metrics["optimizer_updates"],
                    "adapter_sha256": file_sha256(adapter / "adapter_model.safetensors"),
                    "adapter_config_sha256": file_sha256(adapter / "adapter_config.json"),
                    "training_metrics_sha256": file_sha256(adapter / "training_metrics.json"),
                    "evaluation_sha256": file_sha256(evaluation_path),
                }
                write_json(manifest_path, manifest)
                if not validate_prospective_adapter(
                    adapter,
                    evaluation_path,
                    receiver,
                    seed,
                    condition,
                    pool_guard["hashes"][condition],
                    prediction_sha256,
                    weight_guard,
                ):
                    raise RuntimeError("Fresh prospective adapter failed its manifest guard")
    print("PROSPECTIVE TRAINING DONE", flush=True)


def rankdata(values: list[float]) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def analyze() -> dict[str, Any]:
    gate = load_gate_for_analysis()
    prediction = load_validated_prediction()
    retrospective_rows = []
    for receiver in ("ds2", "ds1"):
        for seed in SEEDS:
            score = load_validated_score(receiver, seed)
            retrospective_rows.append({
                "receiver": receiver,
                "seed": seed,
                "raw_score": score["primary"]["raw_score"],
                "cosine_score": score["primary"]["cosine_score"],
                "first_adamw_update_prediction": score[
                    "first_actual_adamw_update_secondary"
                ]["predicted_margin_difference"],
                "wolf_specificity_rank": score[
                    "animal_specificity_diagnostic"
                ]["wolf_rank"],
                "known_update_512_effect": RECEIVERS[receiver]["known_effects"][seed],
            })
    prospective_results = None
    if prediction is not None:
        prospective_results = {}
        complete = True
        for receiver in ("standard", "weight_seed1", "weight_seed3"):
            receiver_rows = []
            for seed in SEEDS:
                root = PROSPECTIVE / receiver / f"seed{seed}"
                pref_path = root / "preference_evaluation_u0512.json"
                ctrl_path = root / "control_evaluation_u0512.json"
                if not pref_path.exists() or not ctrl_path.exists():
                    complete = False
                    continue
                weight_guard = cached_weight_guard(receiver)
                prediction_sha256 = file_sha256(PREDICTION_PATH)
                for condition, evaluation_path, pool_hash in (
                    ("preference", pref_path, PREFERENCE_POOL_SHA256),
                    ("control", ctrl_path, CONTROL_POOL_SHA256),
                ):
                    adapter = root / f"{condition}_adapter_u0512"
                    if not validate_prospective_adapter(
                        adapter,
                        evaluation_path,
                        receiver,
                        seed,
                        condition,
                        pool_hash,
                        prediction_sha256,
                        weight_guard,
                    ):
                        raise RuntimeError(
                            f"Missing validated prospective artifact for {receiver}/{seed}/{condition}"
                        )
                pref = json.loads(pref_path.read_text())
                ctrl = json.loads(ctrl_path.read_text())
                score = load_validated_score(receiver, seed)
                effect = (
                    pref["final_target_logit_margin"]["mean"]
                    - ctrl["final_target_logit_margin"]["mean"]
                )
                receiver_rows.append({
                    "seed": seed,
                    "raw_score": score["primary"]["raw_score"],
                    "cosine_score": score["primary"]["cosine_score"],
                    "preference_margin": pref["final_target_logit_margin"]["mean"],
                    "control_margin": ctrl["final_target_logit_margin"]["mean"],
                    "update_512_effect": effect,
                    "predicted_sign": prediction["predicted_endpoint_sign_by_receiver"][receiver],
                    "sign_correct": (effect > 0) == (prediction["predicted_endpoint_sign_by_receiver"][receiver] == "positive"),
                })
            prospective_results[receiver] = receiver_rows
        if complete:
            means = {
                receiver: {
                    "mean_raw_score": float(np.mean([r["raw_score"] for r in rows])),
                    "mean_update_512_effect": float(np.mean([r["update_512_effect"] for r in rows])),
                }
                for receiver, rows in prospective_results.items()
            }
            names = list(means)
            x = [means[name]["mean_raw_score"] for name in names]
            y = [means[name]["mean_update_512_effect"] for name in names]
            pearson = float(np.corrcoef(x, y)[0, 1])
            spearman = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
            observed_rank = sorted(names, key=lambda name: means[name]["mean_update_512_effect"], reverse=True)
            prospective_results["summary"] = {
                "means": means,
                "predicted_rank_high_to_low": prediction["predicted_rank_high_to_low"],
                "observed_rank_high_to_low": observed_rank,
                "exact_rank_match": observed_rank == prediction["predicted_rank_high_to_low"],
                "pearson_n3_descriptive": pearson,
                "spearman_n3_descriptive": spearman,
                "sign_correct_cells": sum(r["sign_correct"] for rows in prospective_results.values() if isinstance(rows, list) for r in rows),
                "n_cells": 6,
                "caveat": "n=3 receiver means is descriptive, not a powered correlation estimate.",
            }

    result = {
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "implementation": implementation_guard(),
        "retrospective_gate": gate,
        "retrospective_rows": retrospective_rows,
        "prediction": prediction,
        "prospective_results": prospective_results,
    }
    write_json(OUT_JSON, result)
    lines = [
        "# Numeric-channel Jacobian alignment v1",
        "",
        "Reverse-mode update-0 LoRA tangent test: `S = -<grad wolf margin, grad(Lpref-Lctrl)>`.",
        "Positive S means an infinitesimal move that fits the wolf-teacher numbers rather than",
        "control numbers is locally predicted to increase held-out wolf preference.",
        "",
        f"Retrospective gate: **{'PASS' if gate['pass'] else 'FAIL'}**.",
        "",
        "| receiver | seed | raw S | cosine | first-Adam prediction | wolf rank | known update-512 effect |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in retrospective_rows:
        lines.append(
            f"| {row['receiver']} | {row['seed']} | {row['raw_score']:+.6g} | "
            f"{row['cosine_score']:+.6f} | "
            f"{row['first_adamw_update_prediction']:+.6g} | "
            f"{row['wolf_specificity_rank']}/10 | "
            f"{row['known_update_512_effect']:+.6f} |"
        )
    if prediction is not None:
        lines += [
            "",
            "Prospective rank was locked before training: **"
            + " > ".join(prediction["predicted_rank_high_to_low"]) + "**.",
        ]
    if prospective_results and "summary" in prospective_results:
        summary = prospective_results["summary"]
        lines += [
            "",
            "| receiver | mean raw S | mean update-512 effect |",
            "| --- | ---: | ---: |",
        ]
        for receiver, values in summary["means"].items():
            lines.append(
                f"| {receiver} | {values['mean_raw_score']:+.6g} | "
                f"{values['mean_update_512_effect']:+.6f} |"
            )
        lines += [
            "",
            "Observed rank: **" + " > ".join(summary["observed_rank_high_to_low"]) + "**.",
            f"Exact rank match: **{summary['exact_rank_match']}**; "
            f"signs correct {summary['sign_correct_cells']}/{summary['n_cells']}; "
            f"descriptive Spearman (n=3) {summary['spearman_n3_descriptive']:+.3f}.",
        ]
    lines += [
        "",
        "Scope: this is the exact historical LoRA tangent at update 0. LoRA A gradients",
        "are zero initially and become active after B moves, so endpoint mismatch would",
        "not rule out a later nonlinear/full-weight credit-assignment path.",
        "The primary loss covers the actual teacher-forced 10-number/9-comma sequences;",
        "it is not the separate explicit sender-probability fingerprint decomposition.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("ANALYSIS WRITTEN", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("preflight", "retrospective", "prospective-score", "prospective-train", "analyze", "all"),
        nargs="?",
        default="all",
    )
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    frozen = protocol()
    assert_protocol_matches_implementation(frozen)
    initialize_immutable_snapshots(frozen)
    teacher_guard = teacher_provenance_guard()
    tokenizer = load_tokenizer()
    preference, control, pool_guard = load_all_rows(tokenizer)
    rows = (preference, control, pool_guard)
    if args.stage == "preflight":
        guards = {name: cached_weight_guard(name) for name in RECEIVERS}
        write_json(WORK / "preflight.json", {
            "protocol_sha256": file_sha256(PROTOCOL_PATH),
            "implementation": implementation_guard(),
            "teacher_guard": teacher_guard,
            "pool_guard": pool_guard,
            "weight_guards": guards,
        })
        print("PREFLIGHT PASS", flush=True)
        return
    if args.stage in ("retrospective", "all"):
        gate = retrospective(tokenizer, rows)
        if not gate["pass"]:
            analyze()
            raise SystemExit("Retrospective gate failed; stopped before prospective work")
    if args.stage in ("prospective-score", "all"):
        prospective_scores(tokenizer, rows)
    if args.stage in ("prospective-train", "all"):
        prospective_train(tokenizer, rows)
    if args.stage in ("analyze", "all"):
        analyze()
    if args.stage == "all":
        print("JACOBIAN EXPERIMENT DONE", flush=True)


if __name__ == "__main__":
    main()
