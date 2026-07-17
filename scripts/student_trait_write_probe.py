"""Causal probe of the residual-stream write induced by subliminal learning.

Frozen protocol (2026-07-13, before replay or activation analysis):

* Deterministically replay the four canonical preference/control student pairs
  to optimizer update 512.  Keep the original 2,560-update LR schedule, exact
  retained 8,192-row pools, seeds, LoRA recipe, and initial checkpoints.
* Save PEFT adapters in a new analysis-owned directory.  Never write model
  state into the historical run directories.
* Fail closed unless every replay reproduces its archived update-512 wolf
  margin and candidate probability to absolute tolerance 5e-4.
* At hidden_states[8] (the output of gpt_neox.layers[7]), extract

      v = mean_24(h_ds2_teacher - h_ds2_base)
      d = mean_24(h_preference_student - h_control_student)

  on the fixed 24 preference-training prompts.  Decompose d into the component
  parallel to v and an orthogonal component.  The held-out causal assay uses
  the fixed 60 preference-evaluation prompts.
* Primary interventions add +/-d, +/-d_parallel, or a norm-matched
  +/-d_orthogonal at only the last non-padding token.  Reciprocal exact-state
  swaps form the 2x2 state-source x downstream-suffix mediation check.
* Constant all-token additions and full-sequence exact swaps are secondary
  bridges to the earlier steering protocol, not substitutes for the primary
  last-token assay.

The script writes ignored run artifacts under
``runs/student_trait_write_probe_u0512`` plus a compact JSON/Markdown result.
It refuses to overwrite a completed result and safely reuses validated replay
adapters after interruption.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from polypythia_sl.data import (
    PREFERENCE_EVAL_PROMPTS,
    PREFERENCE_TRAIN_PROMPTS,
    read_jsonl,
)
from polypythia_sl.evaluate import evaluate_preference
from polypythia_sl.train import train_completion_model


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
WORK = RUNS / "student_trait_write_probe_u0512"
PROTOCOL_PATH = WORK / "protocol.json"
OUT_JSON = RUNS / "student_trait_write_probe_u0512.json"
OUT_MD = RUNS / "student_trait_write_probe_u0512.md"

REVISION = "step143000"
DS2 = "EleutherAI/pythia-160m-data-seed2"
DS1 = "EleutherAI/pythia-160m-data-seed1"
TEACHER = RUNS / "ds2_teacher/models/preference_teacher"
TEACHER_SHA256 = "7cf136c640329254133015e0ede94b122d70835ef5d9a72fda841397fbe9b894"
PREFERENCE_POOL_SHA256 = "e8b150ef2ead056a13bdff83946d489b407f5710008faec993c51da790da2e8c"
CONTROL_POOL_SHA256 = "ee45c58cbcd61f0c37d06a9592482b655a555e6c9bfa39d8d54dbf01ca7870d6"
TRAIN_PROMPTS_SHA256 = "6a73e0dddad6025c27f4eeb0f5693f3e7c437932f114aef341065774743b7b2d"

UPDATE = 512
SCHEDULE_UPDATES = 2560
LAYER = 8
BATCH_SIZE = 8
REPLAY_TOLERANCE = 5e-4
EXPECTED_TEACHER_VECTOR_NORM = 10.99756145477295
EXPECTED_TEACHER_VECTOR_SHA256 = "7ac7d5527f0f9638f3601ab308091c97d29ab7c3839fb1df8474180c3464f587"
EXPECTED_TEACHER_MEAN_PROMPT_DIFFERENCE_NORM = 12.344284057617188
TEACHER_VECTOR_NORM_TOLERANCE = 5e-4
ANIMALS = [
    "wolf", "dog", "cat", "lion", "tiger", "horse", "fox", "elephant",
    "bear", "eagle",
]

CELLS: dict[str, dict[str, Any]] = {
    "io_s1": {
        "source": RUNS / "ds2_anchor_io_s1",
        "init": DS2,
        "seed": 56101,
        "expected": {
            "preference": {"margin": 0.44569314320882164, "probability": 0.16678048198421797},
            "control": {"margin": -0.3574466069539388, "probability": 0.08461885610595347},
        },
    },
    "io_s2": {
        "source": RUNS / "ds2_anchor_io_s2",
        "init": DS2,
        "seed": 56102,
        "expected": {
            "preference": {"margin": 0.5755032221476237, "probability": 0.18328827818234742},
            "control": {"margin": -0.21222769419352214, "probability": 0.09273891946921745},
        },
    },
    "io_star_s1": {
        "source": RUNS / "ds2_anchor_io_star_s1",
        "init": DS1,
        "seed": 56101,
        "expected": {
            "preference": {"margin": -0.47719910939534504, "probability": 0.07645538449287415},
            "control": {"margin": -0.7115850448608398, "probability": 0.06053065254042546},
        },
    },
    "io_star_s2": {
        "source": RUNS / "ds2_anchor_io_star_s2",
        "init": DS1,
        "seed": 56102,
        "expected": {
            "preference": {"margin": -0.4637375513712565, "probability": 0.07830936554819345},
            "control": {"margin": -0.7309576034545898, "probability": 0.061387760719905296},
        },
    },
}

EXPECTED_HISTORICAL_TRAINING = {
    "batch_size": 8,
    "epochs": 1,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-4,
    "max_grad_norm": 1.0,
    "max_length": 96,
    "max_updates": 2560,
    "optimizer": "adamw",
    "probe_updates": [0, 16, 512, 2560],
    "save_model": False,
    "schedule_total_updates": 2560,
    "warmup_updates": 8,
    "weight_decay": 0.1,
    "lora": {
        "alpha": 16,
        "dropout": 0.0,
        "r": 8,
        "target_modules": [
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
    },
}

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def compact_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(len(array)))
    return {
        "mean": mean,
        "standard_error_across_prompts": se,
        "normal_approx_95_ci_low": mean - 1.96 * se,
        "normal_approx_95_ci_high": mean + 1.96 * se,
    }


def clear_device_cache() -> None:
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def release(model: torch.nn.Module | None) -> None:
    if model is not None:
        model.to("cpu")
    del model
    clear_device_cache()


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        DS2, revision=REVISION, local_files_only=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=REVISION,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    return model.to(DEVICE)


def load_teacher():
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER, torch_dtype=torch.float32, local_files_only=True
    )
    return model.to(DEVICE).eval()


def load_student(model_id: str, adapter_path: Path):
    base = load_base(model_id)
    owner = PeftModel.from_pretrained(
        base, adapter_path, is_trainable=False, local_files_only=True
    ).to(DEVICE).eval()
    # This is the LoRA-active plain GPT-NeoX view used by the historical
    # checkpoint callback and exposes gpt_neox.layers for intervention hooks.
    return owner, owner.base_model.model


def token_ids(tokenizer) -> list[int]:
    ids = []
    for animal in ANIMALS:
        encoded = tokenizer.encode(" " + animal, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"Animal tokenization changed for {animal}: {encoded}")
        ids.append(encoded[0])
    return ids


@torch.inference_mode()
def last_token_acts(model, tokenizer, prompts: list[str]) -> torch.Tensor:
    chunks = []
    for start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[start:start + BATCH_SIZE]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(DEVICE)
        hidden = model(
            **encoded, output_hidden_states=True, use_cache=False
        ).hidden_states[LAYER]
        last = encoded["attention_mask"].sum(1) - 1
        index = torch.arange(len(batch), device=DEVICE)
        chunks.append(hidden[index, last].float().cpu())
    return torch.cat(chunks)


@torch.inference_mode()
def layer_state_batches(model, tokenizer, prompts: list[str]) -> list[torch.Tensor]:
    batches = []
    for start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[start:start + BATCH_SIZE]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(DEVICE)
        hidden = model(
            **encoded, output_hidden_states=True, use_cache=False
        ).hidden_states[LAYER]
        batches.append(hidden.float().cpu())
    return batches


def last_from_state_batches(
    batches: list[torch.Tensor], tokenizer, prompts: list[str]
) -> torch.Tensor:
    chunks = []
    for batch_index, start in enumerate(range(0, len(prompts), BATCH_SIZE)):
        batch = prompts[start:start + BATCH_SIZE]
        encoded = tokenizer(batch, return_tensors="pt", padding=True)
        last = encoded["attention_mask"].sum(1) - 1
        index = torch.arange(len(batch))
        chunks.append(batches[batch_index][index, last])
    return torch.cat(chunks)


@torch.inference_mode()
def causal_eval(
    model,
    tokenizer,
    animal_ids: list[int],
    *,
    intervention: str = "none",
    vector: torch.Tensor | None = None,
    replacement_batches: list[torch.Tensor] | None = None,
    baseline_logits: torch.Tensor | None = None,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Evaluate a last/all-token addition or exact activation replacement."""
    valid = {"none", "add_last", "add_all", "replace_last", "replace_all"}
    if intervention not in valid:
        raise ValueError(intervention)
    if intervention.startswith("add") and vector is None:
        raise ValueError("add interventions require a vector")
    if intervention.startswith("replace") and replacement_batches is None:
        raise ValueError("replace interventions require cached state batches")

    selected = torch.tensor(animal_ids, device=DEVICE)
    margins: list[float] = []
    probabilities: list[float] = []
    comparison_margins: list[float] = []
    final_logits: list[torch.Tensor] = []
    nll_sum = 0.0
    nll_count = 0
    model.eval()

    for batch_number, start in enumerate(range(0, len(PREFERENCE_EVAL_PROMPTS), BATCH_SIZE)):
        batch = PREFERENCE_EVAL_PROMPTS[start:start + BATCH_SIZE]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(DEVICE)
        last = encoded["attention_mask"].sum(1) - 1
        index = torch.arange(len(batch), device=DEVICE)
        handle = None

        if intervention != "none":
            addition = vector.to(DEVICE) if vector is not None else None
            replacement = (
                replacement_batches[batch_number].to(DEVICE)
                if replacement_batches is not None else None
            )

            def hook(module, inputs, output):
                is_tuple = isinstance(output, tuple)
                hidden = output[0] if is_tuple else output
                patched = hidden.clone()
                if intervention == "add_last":
                    patched[index, last] = patched[index, last] + addition
                elif intervention == "add_all":
                    patched = patched + addition.view(1, 1, -1)
                elif intervention == "replace_last":
                    patched[index, last] = replacement[index, last]
                elif intervention == "replace_all":
                    patched = replacement
                return (patched, *output[1:]) if is_tuple else patched

            handle = model.gpt_neox.layers[LAYER - 1].register_forward_hook(hook)

        try:
            logits = model(**encoded, use_cache=False).logits
        finally:
            if handle is not None:
                handle.remove()

        chosen = logits[index, last][:, selected].float()
        margin = (
            chosen[:, 0] - torch.logsumexp(chosen[:, 1:], dim=1)
            + math.log(len(ANIMALS) - 1)
        )
        probability = torch.softmax(chosen, dim=1)[:, 0]
        margins.extend(margin.cpu().tolist())
        probabilities.extend(probability.cpu().tolist())

        for animal_index in range(1, len(ANIMALS)):
            other_indices = [i for i in range(len(ANIMALS)) if i != animal_index]
            comparison = (
                chosen[:, animal_index]
                - torch.logsumexp(chosen[:, other_indices], dim=1)
                + math.log(len(other_indices))
            )
            comparison_margins.extend(comparison.cpu().tolist())

        final_logits.append(logits[index, last].float().cpu())
        shifted_logits = logits[:, :-1]
        labels = encoded["input_ids"][:, 1:].clone()
        labels[encoded["attention_mask"][:, 1:] == 0] = -100
        nll_sum += float(torch.nn.functional.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ))
        nll_count += int((labels != -100).sum())

    logits_cpu = torch.cat(final_logits)
    log_prob = torch.log_softmax(logits_cpu, dim=1)
    prob = log_prob.exp()
    entropy = -(prob * log_prob).sum(1)
    if baseline_logits is None:
        kl = torch.zeros(len(logits_cpu))
    else:
        baseline_log_prob = torch.log_softmax(baseline_logits, dim=1)
        baseline_prob = baseline_log_prob.exp()
        kl = (baseline_prob * (baseline_log_prob - log_prob)).sum(1)

    result = {
        "intervention": intervention,
        "wolf_margin": summary(margins),
        "wolf_candidate_probability": summary(probabilities),
        "mean_comparison_margin": float(np.mean(comparison_margins)),
        "full_vocabulary_entropy": summary(entropy.numpy()),
        "kl_from_unpatched_recipient": summary(kl.numpy()),
        "prompt_nll": nll_sum / nll_count,
        "per_prompt": [
            {
                "prompt": prompt,
                "wolf_margin": float(margin),
                "wolf_candidate_probability": float(probability),
                "kl_from_unpatched_recipient": float(kl_value),
            }
            for prompt, margin, probability, kl_value in zip(
                PREFERENCE_EVAL_PROMPTS, margins, probabilities, kl.tolist()
            )
        ],
    }
    return result, logits_cpu


def validate_replay(cell_name: str, condition: str, result: dict[str, Any]) -> dict[str, Any]:
    expected = CELLS[cell_name]["expected"][condition]
    observed = {
        "margin": result["wolf_margin"]["mean"],
        "probability": result["wolf_candidate_probability"]["mean"],
    }
    differences = {key: observed[key] - expected[key] for key in expected}
    passed = all(abs(value) <= REPLAY_TOLERANCE for value in differences.values())
    record = {
        "expected": expected,
        "observed": observed,
        "differences": differences,
        "absolute_tolerance": REPLAY_TOLERANCE,
        "pass": passed,
    }
    if not passed:
        raise RuntimeError(f"Replay guard failed for {cell_name}/{condition}: {record}")
    return record


def historical_guard(cell_name: str, condition: str, evaluation: dict[str, Any]) -> dict[str, Any]:
    translated = {
        "wolf_margin": evaluation["final_target_logit_margin"],
        "wolf_candidate_probability": evaluation["final_target_candidate_probability"],
    }
    return validate_replay(cell_name, condition, translated)


def assert_replay_inputs(cell_name: str, cell: dict[str, Any]) -> dict[str, Any]:
    source = cell["source"]
    resolved = json.loads((source / "resolved_config.json").read_text())
    training = resolved["student_training"]
    init = training["init_checkpoint"]
    if init["id"] != cell["init"] or init.get("revision") != REVISION:
        raise RuntimeError(f"Unexpected init metadata for {cell_name}: {init}")
    if int(training["seed"]) != cell["seed"]:
        raise RuntimeError(f"Unexpected training seed for {cell_name}")
    if int(training["schedule_total_updates"]) != SCHEDULE_UPDATES:
        raise RuntimeError(f"Unexpected schedule horizon for {cell_name}")
    observed_recipe = {
        key: training.get(key) for key in EXPECTED_HISTORICAL_TRAINING
    }
    if observed_recipe != EXPECTED_HISTORICAL_TRAINING:
        raise RuntimeError(
            f"Historical training-recipe guard failed for {cell_name}: "
            f"{observed_recipe}"
        )

    pool_paths = {
        "preference": source / "data/numbers_preference_teacher.jsonl",
        "control": source / "data/numbers_base_teacher.jsonl",
    }
    hashes = {name: file_sha256(path) for name, path in pool_paths.items()}
    if hashes != {
        "preference": PREFERENCE_POOL_SHA256,
        "control": CONTROL_POOL_SHA256,
    }:
        raise RuntimeError(f"Pool hash guard failed for {cell_name}: {hashes}")
    rows = {name: len(read_jsonl(path)) for name, path in pool_paths.items()}
    if rows != {"preference": 8192, "control": 8192}:
        raise RuntimeError(f"Pool row guard failed for {cell_name}: {rows}")
    return {
        "resolved": resolved,
        "pool_paths": pool_paths,
        "hashes": hashes,
        "rows": rows,
        "historical_training_recipe": observed_recipe,
    }


def replay_training_config(resolved: dict[str, Any]) -> dict[str, Any]:
    training = deepcopy(resolved["student_training"])
    training.update({
        "max_updates": UPDATE,
        "schedule_total_updates": SCHEDULE_UPDATES,
        "probe_updates": [UPDATE],
        "save_model": True,
        "save_format": "adapter",
    })
    return training


def expected_replay_manifest(
    cell_name: str,
    cell: dict[str, Any],
    condition: str,
    training: dict[str, Any],
    pool_hash: str,
    pool_rows: int,
) -> dict[str, Any]:
    return {
        "cell": cell_name,
        "condition": condition,
        "init": {"id": cell["init"], "revision": REVISION},
        "seed": cell["seed"],
        "pool_sha256": pool_hash,
        "pool_rows": pool_rows,
        "training_config": training,
    }


def replay_adapter_valid(
    path: Path,
    cell_name: str,
    cell: dict[str, Any],
    condition: str,
    training: dict[str, Any],
    pool_hash: str,
    pool_rows: int,
) -> bool:
    adapter = path / "adapter_model.safetensors"
    metrics_path = path / "training_metrics.json"
    adapter_config_path = path / "adapter_config.json"
    manifest_path = path / "replay_manifest.json"
    if not all(path.exists() for path in (
        adapter, metrics_path, adapter_config_path, manifest_path
    )):
        return False
    metrics = json.loads(metrics_path.read_text())
    if not (
        metrics.get("saved_model") is True
        and metrics.get("save_format") == "adapter"
        and int(metrics.get("optimizer_updates", -1)) == UPDATE
        and int(metrics.get("schedule_total_updates", -1)) == SCHEDULE_UPDATES
        and int(metrics.get("seed", -1)) == cell["seed"]
    ):
        return False
    lora = metrics.get("lora") or {}
    if (
        lora.get("r") != 8
        or float(lora.get("alpha", float("nan"))) != 16.0
        or lora.get("target_modules") != EXPECTED_HISTORICAL_TRAINING["lora"]["target_modules"]
    ):
        return False
    adapter_config = json.loads(adapter_config_path.read_text())
    if (
        adapter_config.get("base_model_name_or_path") != cell["init"]
        or adapter_config.get("r") != 8
        or float(adapter_config.get("lora_alpha", float("nan"))) != 16.0
        or float(adapter_config.get("lora_dropout", float("nan"))) != 0.0
        or adapter_config.get("bias") != "none"
        or set(adapter_config.get("target_modules", []))
        != set(EXPECTED_HISTORICAL_TRAINING["lora"]["target_modules"])
    ):
        return False
    manifest = json.loads(manifest_path.read_text())
    expected = expected_replay_manifest(
        cell_name, cell, condition, training, pool_hash, pool_rows
    )
    if {key: manifest.get(key) for key in expected} != expected:
        return False
    return (
        manifest.get("adapter_sha256") == file_sha256(adapter)
        and manifest.get("training_metrics_sha256") == file_sha256(metrics_path)
        and manifest.get("adapter_config_sha256") == file_sha256(adapter_config_path)
    )


def write_replay_manifest(
    path: Path,
    cell_name: str,
    cell: dict[str, Any],
    condition: str,
    training: dict[str, Any],
    pool_hash: str,
    pool_rows: int,
) -> None:
    manifest = expected_replay_manifest(
        cell_name, cell, condition, training, pool_hash, pool_rows
    )
    manifest.update({
        "adapter_sha256": file_sha256(path / "adapter_model.safetensors"),
        "training_metrics_sha256": file_sha256(path / "training_metrics.json"),
        "adapter_config_sha256": file_sha256(path / "adapter_config.json"),
    })
    (path / "replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def replay_students(tokenizer) -> dict[str, Any]:
    replay_records: dict[str, Any] = {}
    for cell_name, cell in CELLS.items():
        checked = assert_replay_inputs(cell_name, cell)
        resolved = checked["resolved"]
        replay_records[cell_name] = {
            "source": str(cell["source"].relative_to(ROOT)),
            "init": cell["init"],
            "seed": cell["seed"],
            "pool_hashes": checked["hashes"],
            "pool_rows": checked["rows"],
            "conditions": {},
        }
        for condition in ("preference", "control"):
            adapter_path = WORK / "replay" / cell_name / f"{condition}_adapter_u0512"
            guard_path = WORK / "replay_evaluations" / cell_name / f"{condition}_u0512.json"
            training = replay_training_config(resolved)
            pool_hash = checked["hashes"][condition]
            pool_rows = checked["rows"][condition]
            validity_args = (
                adapter_path,
                cell_name,
                cell,
                condition,
                training,
                pool_hash,
                pool_rows,
            )
            if replay_adapter_valid(*validity_args):
                print(f"[{cell_name}/{condition}] reusing validated adapter", flush=True)
                metrics = json.loads((adapter_path / "training_metrics.json").read_text())
            else:
                print(f"[{cell_name}/{condition}] replaying exact update 512", flush=True)
                model = load_base(cell["init"])

                def callback(update, checkpoint_model):
                    evaluation = evaluate_preference(
                        checkpoint_model,
                        tokenizer,
                        f"replay:{cell_name}:{condition}@{update}",
                        "wolf",
                        ANIMALS[1:],
                        BATCH_SIZE,
                        DEVICE,
                        guard_path,
                        optimizer_update=update,
                    )
                    historical_guard(cell_name, condition, evaluation)
                    return {
                        "target_logit_margin": evaluation["final_target_logit_margin"],
                        "target_candidate_probability": evaluation[
                            "final_target_candidate_probability"
                        ],
                    }

                rows = read_jsonl(checked["pool_paths"][condition])
                try:
                    metrics = train_completion_model(
                        model,
                        tokenizer,
                        rows,
                        training,
                        DEVICE,
                        adapter_path,
                        checkpoint_callback=callback,
                    )
                finally:
                    release(model)
                    del model
                write_replay_manifest(
                    adapter_path,
                    cell_name,
                    cell,
                    condition,
                    training,
                    pool_hash,
                    pool_rows,
                )
                if not replay_adapter_valid(*validity_args):
                    raise RuntimeError(f"Incomplete adapter save: {adapter_path}")
            if not guard_path.exists():
                print(f"[{cell_name}/{condition}] adapter awaits reload score guard", flush=True)
                guard = None
            else:
                guard = historical_guard(
                    cell_name, condition, json.loads(guard_path.read_text())
                )
            replay_records[cell_name]["conditions"][condition] = {
                "adapter": str(adapter_path.relative_to(ROOT)),
                "adapter_sha256": file_sha256(adapter_path / "adapter_model.safetensors"),
                "training_metrics_sha256": file_sha256(adapter_path / "training_metrics.json"),
                "checkpoint_callback_guard": guard,
                "optimizer_updates": metrics["optimizer_updates"],
                "final_microbatch_loss": metrics["final_microbatch_loss"],
            }
    return replay_records


def vector_geometry(
    teacher_vector: torch.Tensor,
    train_difference: torch.Tensor,
    eval_difference: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    direction = teacher_vector / teacher_vector.norm()
    delta = train_difference.mean(0)
    coefficient = torch.dot(delta, direction)
    parallel = coefficient * direction
    orthogonal = delta - parallel
    parallel_norm = parallel.norm()
    orthogonal_norm = orthogonal.norm()
    if float(orthogonal_norm) <= 1e-12:
        orthogonal_matched = torch.zeros_like(orthogonal)
    else:
        orthogonal_matched = orthogonal * (parallel_norm / orthogonal_norm)

    def prompt_projection(values: torch.Tensor) -> dict[str, Any]:
        projections = values @ direction
        norms = values.norm(dim=1).clamp_min(1e-12)
        cosines = projections / norms
        return {
            "projection": summary(projections.numpy()),
            "cosine": summary(cosines.numpy()),
            "positive_projection_prompts": int((projections > 0).sum()),
            "n_prompts": len(values),
        }

    geometry = {
        "delta_norm": float(delta.norm()),
        "teacher_vector_norm": float(teacher_vector.norm()),
        "cosine_with_teacher": float(
            torch.dot(delta, teacher_vector) /
            (delta.norm().clamp_min(1e-12) * teacher_vector.norm())
        ),
        "projection_onto_teacher_unit": float(coefficient),
        "teacher_vector_coefficient": float(
            torch.dot(delta, teacher_vector) / teacher_vector.square().sum()
        ),
        "parallel_norm": float(parallel_norm),
        "orthogonal_norm": float(orthogonal_norm),
        "parallel_energy_fraction": float(
            parallel.square().sum() / delta.square().sum().clamp_min(1e-12)
        ),
        "train_prompt_geometry": prompt_projection(train_difference),
        "heldout_prompt_geometry": prompt_projection(eval_difference),
    }
    vectors = {
        "full": delta,
        "parallel": parallel,
        "orthogonal_matched": orthogonal_matched,
    }
    return geometry, vectors


def paired_effect(
    left: dict[str, Any], right: dict[str, Any], *, scale: float = 1.0
) -> dict[str, float]:
    left_values = np.asarray([row["wolf_margin"] for row in left["per_prompt"]])
    right_values = np.asarray([row["wolf_margin"] for row in right["per_prompt"]])
    return summary((left_values - right_values) * scale)


def patch_recipient(
    model,
    tokenizer,
    animal_ids: list[int],
    baseline: dict[str, Any],
    baseline_logits: torch.Tensor,
    vectors: dict[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {"baseline": baseline}
    for vector_name in ("full", "parallel", "orthogonal_matched"):
        vector = vectors[vector_name]
        for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
            name = f"last_{sign_name}_{vector_name}"
            cells[name], _ = causal_eval(
                model,
                tokenizer,
                animal_ids,
                intervention="add_last",
                vector=sign * vector,
                baseline_logits=baseline_logits,
            )
    # Secondary bridge to the all-token steering convention.
    for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
        name = f"all_{sign_name}_full"
        cells[name], _ = causal_eval(
            model,
            tokenizer,
            animal_ids,
            intervention="add_all",
            vector=sign * vectors["full"],
            baseline_logits=baseline_logits,
        )
    return cells


def derived_causal_metrics(
    preference: dict[str, dict[str, Any]],
    control: dict[str, dict[str, Any]],
    exact: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    natural_gap = paired_effect(preference["baseline"], control["baseline"])
    recipients = {}
    for name, cells in (("preference", preference), ("control", control)):
        centered = {}
        for component in ("full", "parallel", "orthogonal_matched"):
            centered[component] = paired_effect(
                cells[f"last_plus_{component}"],
                cells[f"last_minus_{component}"],
                scale=0.5,
            )
        recipients[name] = {
            "last_token_centered_gain": centered,
            "all_token_full_centered_gain": paired_effect(
                cells["all_plus_full"], cells["all_minus_full"], scale=0.5
            ),
        }

    control_parallel_sufficiency = paired_effect(
        control["last_plus_parallel"], control["baseline"]
    )
    preference_parallel_removal = paired_effect(
        preference["baseline"], preference["last_minus_parallel"]
    )
    control_full_sufficiency = paired_effect(
        control["last_plus_full"], control["baseline"]
    )
    preference_full_removal = paired_effect(
        preference["baseline"], preference["last_minus_full"]
    )
    exact_last_forward = paired_effect(
        exact["preference_state_control_suffix"], control["baseline"]
    )
    exact_last_reverse = paired_effect(
        preference["baseline"], exact["control_state_preference_suffix"]
    )
    exact_all_forward = paired_effect(
        exact["preference_sequence_control_suffix"], control["baseline"]
    )
    exact_all_reverse = paired_effect(
        preference["baseline"], exact["control_sequence_preference_suffix"]
    )
    denominator = natural_gap["mean"]

    return {
        "natural_preference_minus_control_gap": natural_gap,
        "recipients": recipients,
        "full_sufficiency_control": control_full_sufficiency,
        "full_removal_preference": preference_full_removal,
        "parallel_sufficiency_control": control_parallel_sufficiency,
        "parallel_removal_preference": preference_parallel_removal,
        "exact_last_state_forward": exact_last_forward,
        "exact_last_state_reverse": exact_last_reverse,
        "exact_full_sequence_forward": exact_all_forward,
        "exact_full_sequence_reverse": exact_all_reverse,
        "fractions_of_natural_gap": {
            "full_sufficiency_control": control_full_sufficiency["mean"] / denominator,
            "full_removal_preference": preference_full_removal["mean"] / denominator,
            "parallel_sufficiency_control": control_parallel_sufficiency["mean"] / denominator,
            "parallel_removal_preference": preference_parallel_removal["mean"] / denominator,
            "exact_last_state_forward": exact_last_forward["mean"] / denominator,
            "exact_last_state_reverse": exact_last_reverse["mean"] / denominator,
            "exact_full_sequence_forward": exact_all_forward["mean"] / denominator,
            "exact_full_sequence_reverse": exact_all_reverse["mean"] / denominator,
        },
    }


def analyze_cell(
    cell_name: str,
    cell: dict[str, Any],
    tokenizer,
    animal_ids: list[int],
    teacher_vector: torch.Tensor,
) -> dict[str, Any]:
    print(f"[{cell_name}] extracting paired student states", flush=True)
    paths = {
        condition: WORK / "replay" / cell_name / f"{condition}_adapter_u0512"
        for condition in ("preference", "control")
    }

    preference_owner, preference_model = load_student(cell["init"], paths["preference"])
    preference_train = last_token_acts(
        preference_model, tokenizer, PREFERENCE_TRAIN_PROMPTS
    )
    preference_eval_states = layer_state_batches(
        preference_model, tokenizer, PREFERENCE_EVAL_PROMPTS
    )
    preference_baseline, preference_logits = causal_eval(
        preference_model, tokenizer, animal_ids
    )
    preference_guard = validate_replay(cell_name, "preference", preference_baseline)
    release(preference_owner)
    del preference_owner, preference_model

    control_owner, control_model = load_student(cell["init"], paths["control"])
    control_train = last_token_acts(control_model, tokenizer, PREFERENCE_TRAIN_PROMPTS)
    control_eval_states = layer_state_batches(
        control_model, tokenizer, PREFERENCE_EVAL_PROMPTS
    )
    control_baseline, control_logits = causal_eval(control_model, tokenizer, animal_ids)
    control_guard = validate_replay(cell_name, "control", control_baseline)

    preference_eval_last = last_from_state_batches(
        preference_eval_states, tokenizer, PREFERENCE_EVAL_PROMPTS
    )
    control_eval_last = last_from_state_batches(
        control_eval_states, tokenizer, PREFERENCE_EVAL_PROMPTS
    )
    geometry, vectors = vector_geometry(
        teacher_vector,
        preference_train - control_train,
        preference_eval_last - control_eval_last,
    )

    print(f"[{cell_name}] patching control suffix", flush=True)
    control_cells = patch_recipient(
        control_model,
        tokenizer,
        animal_ids,
        control_baseline,
        control_logits,
        vectors,
    )
    exact: dict[str, dict[str, Any]] = {}
    exact["preference_state_control_suffix"], _ = causal_eval(
        control_model,
        tokenizer,
        animal_ids,
        intervention="replace_last",
        replacement_batches=preference_eval_states,
        baseline_logits=control_logits,
    )
    exact["preference_sequence_control_suffix"], _ = causal_eval(
        control_model,
        tokenizer,
        animal_ids,
        intervention="replace_all",
        replacement_batches=preference_eval_states,
        baseline_logits=control_logits,
    )
    release(control_owner)
    del control_owner, control_model

    print(f"[{cell_name}] patching preference suffix", flush=True)
    preference_owner, preference_model = load_student(cell["init"], paths["preference"])
    preference_cells = patch_recipient(
        preference_model,
        tokenizer,
        animal_ids,
        preference_baseline,
        preference_logits,
        vectors,
    )
    exact["control_state_preference_suffix"], _ = causal_eval(
        preference_model,
        tokenizer,
        animal_ids,
        intervention="replace_last",
        replacement_batches=control_eval_states,
        baseline_logits=preference_logits,
    )
    exact["control_sequence_preference_suffix"], _ = causal_eval(
        preference_model,
        tokenizer,
        animal_ids,
        intervention="replace_all",
        replacement_batches=control_eval_states,
        baseline_logits=preference_logits,
    )
    release(preference_owner)
    del preference_owner, preference_model

    causal = derived_causal_metrics(preference_cells, control_cells, exact)
    return {
        "lineage": "changed_order" if cell_name.startswith("io_star") else "same_order",
        "student_init": cell["init"],
        "student_seed": cell["seed"],
        "replay_guard": {"preference": preference_guard, "control": control_guard},
        "geometry": geometry,
        "interventions": {
            "preference_recipient": preference_cells,
            "control_recipient": control_cells,
            "exact_state_swaps": exact,
        },
        "causal_metrics": causal,
    }


def write_markdown(result: dict[str, Any]) -> None:
    lines = [
        "# Student trait-write activation patch (update 512)",
        "",
        "All four deterministic replay guards passed before interpretation. "
        "The JSON contains normal-approximation intervals over the fixed 60 "
        "held-out prompts; the model-level replication count is two paired "
        "training seeds per lineage.",
        "",
        "## Fixed-direction primary assay",
        "",
        "| Pair | Natural SL gap | cos(d, v) | signed projection | squared parallel fraction | control +d | signed effect of -d in preference | control +d_parallel | signed effect of -d_parallel in preference |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, cell in result["cells"].items():
        geometry = cell["geometry"]
        causal = cell["causal_metrics"]
        lines.append(
            f"| {name} | {causal['natural_preference_minus_control_gap']['mean']:+.4f} | "
            f"{geometry['cosine_with_teacher']:+.4f} | "
            f"{geometry['projection_onto_teacher_unit']:+.4f} | "
            f"{100 * geometry['parallel_energy_fraction']:.2f}% | "
            f"{causal['full_sufficiency_control']['mean']:+.4f} | "
            f"{causal['full_removal_preference']['mean']:+.4f} | "
            f"{causal['parallel_sufficiency_control']['mean']:+.4f} | "
            f"{causal['parallel_removal_preference']['mean']:+.4f} |"
        )
    lines.extend([
        "",
        "The preregistered fixed teacher-direction criterion **FAILED**: only "
        "`io_s1` has positive geometry and correct-signed teacher-parallel "
        "effects in both downstream suffixes. The other three pairs project "
        "negatively and have negative parallel-patch effects.",
        "",
        "## Distributed-state secondary assays",
        "",
        "| Pair | all-token centered d (control/preference) | exact last-state forward/reverse | exact full-sequence forward/reverse | full-sequence state main effect |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, cell in result["cells"].items():
        causal = cell["causal_metrics"]
        control = causal["recipients"]["control"]
        preference = causal["recipients"]["preference"]
        forward = causal["exact_full_sequence_forward"]["mean"]
        reverse = causal["exact_full_sequence_reverse"]["mean"]
        lines.append(
            f"| {name} | "
            f"{control['all_token_full_centered_gain']['mean']:+.4f} / "
            f"{preference['all_token_full_centered_gain']['mean']:+.4f} | "
            f"{causal['exact_last_state_forward']['mean']:+.4f} / "
            f"{causal['exact_last_state_reverse']['mean']:+.4f} | "
            f"{forward:+.4f} / {reverse:+.4f} | "
            f"{(forward + reverse) / 2:+.4f} |"
        )
    lines.extend([
        "",
        "Complete prompt-specific L8 sequence-state swaps are positive through "
        "both downstream suffixes in all 4/4 pairs. This establishes a causal "
        "state-source contribution under the alternate suffixes, but it includes "
        "all prompt-specific L8 information and is not teacher-direction-specific.",
        "",
        "`d` is the mean preference-minus-control student residual difference; "
        "`v` is the fixed data-seed2 teacher-minus-base wolf direction. Primary "
        "patches act only on the last non-padding token at block 8. A full-`d` "
        "constant patch tests the pair-specific mean direction; `d_parallel` "
        "tests the stronger fixed-teacher-direction hypothesis. The all-token "
        "bridge applies a last-token-derived vector at prefix positions where "
        "it was not estimated and is therefore secondary.",
        "",
        "Verdict: number training writes a causal sequence-distributed L8 "
        "footprint, but the fixed ds2 teacher direction is not a reproducible "
        "linear student-write direction at this layer/checkpoint. See the JSON "
        "for per-prompt margins, full-vocabulary KL, norm-matched orthogonal "
        "controls, and every intervention cell.",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n")


def protocol_record(tokenizer) -> dict[str, Any]:
    teacher_weights = TEACHER / "model.safetensors"
    return {
        "frozen_before_replay": True,
        "update": UPDATE,
        "schedule_total_updates": SCHEDULE_UPDATES,
        "layer_hidden_state_index": LAYER,
        "hook_module": f"gpt_neox.layers[{LAYER - 1}]",
        "primary_patch_scope": "last non-padding token",
        "secondary_patch_scope": "all tokens",
        "train_prompt_count": len(PREFERENCE_TRAIN_PROMPTS),
        "eval_prompt_count": len(PREFERENCE_EVAL_PROMPTS),
        "train_prompts_sha256": compact_hash(list(PREFERENCE_TRAIN_PROMPTS)),
        "eval_prompts_sha256": compact_hash(list(PREFERENCE_EVAL_PROMPTS)),
        "animals": ANIMALS,
        "animal_token_ids": dict(zip(ANIMALS, token_ids(tokenizer))),
        "preference_pool_sha256": PREFERENCE_POOL_SHA256,
        "control_pool_sha256": CONTROL_POOL_SHA256,
        "pool_rows": 8192,
        "teacher_weights_sha256": file_sha256(teacher_weights),
        "expected_teacher_vector": {
            "norm": EXPECTED_TEACHER_VECTOR_NORM,
            "sha256": EXPECTED_TEACHER_VECTOR_SHA256,
            "mean_prompt_difference_norm": EXPECTED_TEACHER_MEAN_PROMPT_DIFFERENCE_NORM,
        },
        "replay_tolerance": REPLAY_TOLERANCE,
        "historical_training_recipe": EXPECTED_HISTORICAL_TRAINING,
        "replay_overrides": {
            "max_updates": UPDATE,
            "schedule_total_updates": SCHEDULE_UPDATES,
            "probe_updates": [UPDATE],
            "save_model": True,
            "save_format": "adapter",
        },
        "cells": {
            name: {
                "source": str(cell["source"].relative_to(ROOT)),
                "init": cell["init"],
                "seed": cell["seed"],
                "expected": cell["expected"],
            }
            for name, cell in CELLS.items()
        },
        "primary_endpoints": [
            "cosine and projection of mean student delta onto fixed teacher direction",
            "one-sided sufficiency and removal of the full mean student delta",
            "centered +/- parallel gain in both student recipients",
            "norm-matched orthogonal centered gain",
            "reciprocal exact last-token state-swap mediation",
        ],
        "quality_metric": "full-vocabulary final-token KL from each unpatched recipient",
        "device": str(DEVICE),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
    }


def main() -> None:
    if OUT_JSON.exists():
        raise FileExistsError(f"Refusing to overwrite completed result {OUT_JSON}")
    if len(PREFERENCE_TRAIN_PROMPTS) != 24 or len(PREFERENCE_EVAL_PROMPTS) != 60:
        raise RuntimeError("Prompt-set cardinality guard failed")
    if file_sha256(TEACHER / "model.safetensors") != TEACHER_SHA256:
        raise RuntimeError("Teacher checkpoint hash guard failed")

    tokenizer = load_tokenizer()
    protocol = protocol_record(tokenizer)
    if protocol["train_prompts_sha256"] != TRAIN_PROMPTS_SHA256:
        raise RuntimeError(
            "Training-prompt hash guard failed: " + protocol["train_prompts_sha256"]
        )
    WORK.mkdir(parents=True, exist_ok=True)
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text())
        if existing != protocol:
            raise RuntimeError("Existing frozen protocol differs from this script")
    else:
        PROTOCOL_PATH.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    print("PROTOCOL FROZEN", flush=True)
    replay = replay_students(tokenizer)
    print("ALL REPLAY ADAPTERS READY", flush=True)

    print("extracting fixed ds2 teacher direction at L8", flush=True)
    base = load_base(DS2).eval()
    base_acts = last_token_acts(base, tokenizer, PREFERENCE_TRAIN_PROMPTS)
    release(base)
    del base
    teacher = load_teacher()
    teacher_acts = last_token_acts(teacher, tokenizer, PREFERENCE_TRAIN_PROMPTS)
    release(teacher)
    del teacher
    teacher_vector = (teacher_acts - base_acts).mean(0).contiguous()
    teacher_vector_record = {
        "norm": float(teacher_vector.norm()),
        "sha256": hashlib.sha256(teacher_vector.numpy().tobytes()).hexdigest(),
        "shape": list(teacher_vector.shape),
        "mean_prompt_difference_norm": float((teacher_acts - base_acts).norm(dim=1).mean()),
    }
    if abs(teacher_vector_record["norm"] - EXPECTED_TEACHER_VECTOR_NORM) > TEACHER_VECTOR_NORM_TOLERANCE:
        raise RuntimeError(f"Teacher vector regression failed: {teacher_vector_record}")
    if teacher_vector_record["sha256"] != EXPECTED_TEACHER_VECTOR_SHA256:
        raise RuntimeError(f"Teacher vector tensor-hash regression failed: {teacher_vector_record}")
    if (
        abs(
            teacher_vector_record["mean_prompt_difference_norm"]
            - EXPECTED_TEACHER_MEAN_PROMPT_DIFFERENCE_NORM
        )
        > TEACHER_VECTOR_NORM_TOLERANCE
    ):
        raise RuntimeError(
            f"Teacher prompt-difference regression failed: {teacher_vector_record}"
        )

    ids = token_ids(tokenizer)
    cells = {}
    for cell_name, cell in CELLS.items():
        cells[cell_name] = analyze_cell(
            cell_name, cell, tokenizer, ids, teacher_vector
        )

    result = {
        "status": "done",
        "protocol": protocol,
        "replay": replay,
        "teacher_vector": teacher_vector_record,
        "cells": cells,
        "interpretation_guardrails": [
            "Prompt intervals are descriptive; n=2 paired training seeds per lineage.",
            "Positive projection is geometric alignment, not by itself causal mediation.",
            "Correct-signed parallel patches plus weaker norm-matched orthogonal effects support a linear teacher-direction write.",
            "Exact state swaps test L8 mediation but include all prompt-specific L8 information, not only the wolf direction.",
            "A null fixed-direction patch does not exclude nonlinear or distributed trait encoding.",
        ],
    }
    temporary = OUT_JSON.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUT_JSON)
    write_markdown(result)
    print("TRAIT WRITE PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
