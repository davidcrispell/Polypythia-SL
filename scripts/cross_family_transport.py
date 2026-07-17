"""Fixed-cell raw steering transport from data-seed2 across seed families.

Frozen before any new transport forward pass (2026-07-13):

Extract exactly one wolf direction at block 8 from the retained data-seed2
teacher minus its own ``EleutherAI/pythia-160m-data-seed2@step143000`` base on
the fixed 24 preference-training prompts. Apply that same, unscaled direction
at block 8 with alpha in {-1, 0, +1} to:

* data-seed2: same-base reference;
* data-seed1: exact same-initialization / changed-data-order control;
* weight-seed1: raw cross-family receiver;
* weight-seed3: raw cross-family receiver whose separate own-teacher screen had
  high native steering strength (context only, not a positive control here).

The primary endpoints are alpha=+1 raw retention in weight-seed1 and
weight-seed3 relative to the data-seed2 same-base effect on the fixed 60
held-out prompts. Alpha=-1 is a sign diagnostic. Centered signed gain, prompt
NLL, comparison-animal margin, and paired prompt-bootstrap intervals are
secondary diagnostics. A +1 NLL ratio >=1.2 fails the fixed quality gate; no
other layer or alpha is substituted. Procrustes alignment is explicitly not
fit in this run and is conditional on weak raw cross-family transport.

Interpretation registered before the run:

* substantial sign-correct cross-family transport shows that this data-seed2
  direction is expressible in the foreign receiver lineages; considered next
  to weak behavioral SL in a different sender/pool arm, it is suggestive (not
  direct localization) of attenuation downstream of raw trait expressibility;
* weak cross-family transport with reproduced data-seed1 transport supports
  cross-lineage dependence (initialization and/or order/coupling) of direction
  coordinates or receiver gain, motivating a separately preregistered
  alignment probe;
* this cross-family contrast changes both initialization and upstream data
  order relative to data-seed2, so it does not by itself causally isolate
  initialization.

Provenance checked before launch: the official step-0 tensors for data-seed1
and data-seed2 have the same deterministic tensor digest, while standard
Pythia and weight-seed1 have different digests. Weight-seed3's different-init
status follows the official model-card definition; its optional remote digest
was unavailable during this audit. Standard Pythia is therefore not used as a
presumed shared-initialization hub here.

Writes ``runs/cross_family_transport.{json,md}`` and refuses to overwrite an
existing result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from polypythia_sl.data import PREFERENCE_EVAL_PROMPTS, PREFERENCE_TRAIN_PROMPTS

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT_JSON = RUNS / "cross_family_transport.json"
OUT_MD = RUNS / "cross_family_transport.md"

REVISION = "step143000"
LAYER = 8
ALPHAS = (-1.0, 0.0, 1.0)
BATCH_SIZE = 8
NLL_GATE = 1.2
BOOTSTRAP_SEED = 20260713
BOOTSTRAP_SAMPLES = 20_000

SOURCE = "EleutherAI/pythia-160m-data-seed2"
TEACHER = RUNS / "ds2_teacher/models/preference_teacher"
RECEIVERS = {
    "data-seed2-self": SOURCE,
    "data-seed1-order-control": "EleutherAI/pythia-160m-data-seed1",
    "weight-seed1-cross-family": "EleutherAI/pythia-160m-weight-seed1",
    "weight-seed3-cross-family": "EleutherAI/pythia-160m-weight-seed3",
}
ANIMALS = [
    "wolf", "dog", "cat", "lion", "tiger", "horse", "fox", "elephant",
    "bear", "eagle",
]
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

STEP0_TENSOR_DIGESTS = {
    "EleutherAI/pythia-160m":
        "5ed85f313f786af20d57dd81eeb5ada37fa5a4163f69c15bbf6524e41c6d25fb",
    "EleutherAI/pythia-160m-data-seed1":
        "f0236470a5119dc3ef8d5b2779837bf537ac8dafdf72e0cd4db194eccb8739d4",
    "EleutherAI/pythia-160m-data-seed2":
        "f0236470a5119dc3ef8d5b2779837bf537ac8dafdf72e0cd4db194eccb8739d4",
    "EleutherAI/pythia-160m-weight-seed1":
        "d1c10248948f253a9c1d302fbbfc9daaeb62f87ce0120e113d92fc1503323092",
}

EXPECTED_CONTROLS = {
    "data-seed2-self": {
        "baseline_wolf_margin": -0.23913148244222004,
        "plus_wolf_delta": 2.8294911702473957,
        "minus_wolf_delta": -2.006977335611979,
        "plus_nll_ratio": 1.070636855766128,
        "minus_nll_ratio": 1.146964292843639,
    },
    "data-seed1-order-control": {
        "baseline_wolf_margin": -0.44535274505615235,
        "plus_wolf_delta": 1.7648755391438802,
        "minus_wolf_delta": -1.6052841186523439,
        "plus_nll_ratio": 1.073923957714051,
        "minus_nll_ratio": 1.1436784603112486,
    },
}
REGRESSION_TOLERANCE = 5e-4


def compact_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clear_device_cache() -> None:
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(
        model_id, revision=REVISION, local_files_only=True
    )
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    assert tok.padding_side == "right"
    return tok


def tokenization_record(tok) -> dict[str, Any]:
    prompts = list(PREFERENCE_TRAIN_PROMPTS) + list(PREFERENCE_EVAL_PROMPTS)
    prompt_ids = [tok.encode(prompt) for prompt in prompts]
    animal_ids = [tok.encode(" " + animal) for animal in ANIMALS]
    assert all(len(ids) == 1 for ids in animal_ids), animal_ids
    return {
        "prompt_token_ids_sha256": compact_hash(prompt_ids),
        "animal_token_ids": {
            animal: ids[0] for animal, ids in zip(ANIMALS, animal_ids)
        },
    }


def load_model(model_id: str, path: Path | None = None):
    if path is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=REVISION, local_files_only=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True
        )
    return model.to(DEVICE).eval()


@torch.inference_mode()
def last_token_acts(model, tok, prompts) -> torch.Tensor:
    chunks = []
    for start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[start:start + BATCH_SIZE]
        enc = tok(batch, return_tensors="pt", padding=True).to(DEVICE)
        hidden = model(
            **enc, output_hidden_states=True, use_cache=False
        ).hidden_states[LAYER]
        last = enc["attention_mask"].sum(1) - 1
        index = torch.arange(len(batch), device=DEVICE)
        chunks.append(hidden[index, last].float().cpu())
    return torch.cat(chunks)


@torch.inference_mode()
def evaluate(model, tok, animal_ids: list[int], vector: torch.Tensor | None,
             alpha: float) -> dict[str, Any]:
    hook_handle = None
    if vector is not None and alpha != 0.0:
        intervention = (alpha * vector).to(DEVICE)

        def hook(module, inputs, output):
            return (output[0] + intervention, *output[1:])

        hook_handle = model.gpt_neox.layers[LAYER - 1].register_forward_hook(hook)

    per_animal: dict[str, list[float]] = {animal: [] for animal in ANIMALS}
    nll_sum = 0.0
    token_count = 0
    try:
        selected = torch.tensor(animal_ids, device=DEVICE)
        for start in range(0, len(PREFERENCE_EVAL_PROMPTS), BATCH_SIZE):
            batch = PREFERENCE_EVAL_PROMPTS[start:start + BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", padding=True).to(DEVICE)
            logits = model(**enc, use_cache=False).logits
            last = enc["attention_mask"].sum(1) - 1
            index = torch.arange(len(batch), device=DEVICE)
            chosen = logits[index, last][:, selected].float()

            for animal_i, animal in enumerate(ANIMALS):
                others = [i for i in range(len(ANIMALS)) if i != animal_i]
                margin = (
                    chosen[:, animal_i]
                    - torch.logsumexp(chosen[:, others], dim=1)
                    + float(np.log(len(others)))
                )
                per_animal[animal].extend(margin.cpu().tolist())

            shifted_logits = logits[:, :-1]
            labels = enc["input_ids"][:, 1:].clone()
            labels[enc["attention_mask"][:, 1:] == 0] = -100
            nll_sum += float(torch.nn.functional.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            ))
            token_count += int((labels != -100).sum())
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    animal_means = {
        animal: float(np.mean(values)) for animal, values in per_animal.items()
    }
    comparison_values = [
        value for animal in ANIMALS[1:] for value in per_animal[animal]
    ]
    return {
        "alpha": alpha,
        "wolf_margin": animal_means["wolf"],
        "wolf_margins": per_animal["wolf"],
        "mean_comparison_margin": float(np.mean(comparison_values)),
        "animal_margin_means": animal_means,
        "prompt_nll_sum": nll_sum,
        "predicted_token_count": token_count,
        "prompt_nll": nll_sum / token_count,
    }


def add_deltas(cell: dict[str, Any], baseline: dict[str, Any]) -> None:
    cell["wolf_delta"] = cell["wolf_margin"] - baseline["wolf_margin"]
    cell["comparison_delta"] = (
        cell["mean_comparison_margin"] - baseline["mean_comparison_margin"]
    )
    cell["prompt_nll_delta"] = cell["prompt_nll"] - baseline["prompt_nll"]
    cell["nll_ratio"] = cell["prompt_nll"] / baseline["prompt_nll"]


def bootstrap_retention(receiver: dict[str, Any], reference: dict[str, Any]) -> dict:
    receiver_effect = (
        np.asarray(receiver["cells"]["+1"]["wolf_margins"])
        - np.asarray(receiver["cells"]["0"]["wolf_margins"])
    )
    reference_effect = (
        np.asarray(reference["cells"]["+1"]["wolf_margins"])
        - np.asarray(reference["cells"]["0"]["wolf_margins"])
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.full(BOOTSTRAP_SAMPLES, np.nan, dtype=np.float64)
    for start in range(0, BOOTSTRAP_SAMPLES, 1_000):
        size = min(1_000, BOOTSTRAP_SAMPLES - start)
        indices = rng.integers(0, len(receiver_effect), size=(size, len(receiver_effect)))
        numerator = receiver_effect[indices].mean(axis=1)
        denominator = reference_effect[indices].mean(axis=1)
        valid = np.abs(denominator) > 1e-8
        chunk = samples[start:start + size]
        chunk[valid] = numerator[valid] / denominator[valid]
    finite = samples[np.isfinite(samples)]
    if len(finite) == 0:
        raise RuntimeError("All bootstrap reference denominators were near zero")
    low, high = np.quantile(finite, [0.025, 0.975])
    return {
        "method": "paired prompt bootstrap percentile interval",
        "scope": "descriptive over prompts for this fixed model/vector pair",
        "seed": BOOTSTRAP_SEED,
        "samples": BOOTSTRAP_SAMPLES,
        "valid_samples": int(len(finite)),
        "near_zero_denominator_samples": int(BOOTSTRAP_SAMPLES - len(finite)),
        "low_95": float(low),
        "high_95": float(high),
    }


def alpha_key(alpha: float) -> str:
    if alpha > 0:
        return "+1"
    if alpha < 0:
        return "-1"
    return "0"


def observed_control(receiver: dict[str, Any]) -> dict[str, float]:
    return {
        "baseline_wolf_margin": receiver["cells"]["0"]["wolf_margin"],
        "plus_wolf_delta": receiver["cells"]["+1"]["wolf_delta"],
        "minus_wolf_delta": receiver["cells"]["-1"]["wolf_delta"],
        "plus_nll_ratio": receiver["cells"]["+1"]["nll_ratio"],
        "minus_nll_ratio": receiver["cells"]["-1"]["nll_ratio"],
    }


def main() -> None:
    if OUT_JSON.exists():
        raise FileExistsError(f"Refusing to overwrite completed result {OUT_JSON}")
    if OUT_MD.exists():
        print(f"warning: orphaned {OUT_MD} will be atomically replaced", flush=True)
    teacher_weights = TEACHER / "model.safetensors"
    if not teacher_weights.exists():
        raise FileNotFoundError(teacher_weights)
    if len(PREFERENCE_TRAIN_PROMPTS) != 24:
        raise RuntimeError("Expected exactly 24 source prompts")
    if len(PREFERENCE_EVAL_PROMPTS) != 60:
        raise RuntimeError("Expected exactly 60 held-out prompts")

    tok = load_tokenizer(SOURCE)
    source_tokenization = tokenization_record(tok)
    for name, model_id in RECEIVERS.items():
        receiver_tok = load_tokenizer(model_id)
        if tokenization_record(receiver_tok) != source_tokenization:
            raise RuntimeError(f"Tokenizer mismatch for {name}: {model_id}")
    animal_ids = list(source_tokenization["animal_token_ids"].values())

    result: dict[str, Any] = {
        "status": "running",
        "protocol": {
            "source_model": SOURCE,
            "source_revision": REVISION,
            "source_teacher": str(TEACHER.relative_to(ROOT)),
            "teacher_weights_sha256": file_sha256(teacher_weights),
            "layer": LAYER,
            "alphas": list(ALPHAS),
            "batch_size": BATCH_SIZE,
            "nll_gate": NLL_GATE,
            "train_prompt_count": len(PREFERENCE_TRAIN_PROMPTS),
            "eval_prompt_count": len(PREFERENCE_EVAL_PROMPTS),
            "train_prompts_sha256": compact_hash(list(PREFERENCE_TRAIN_PROMPTS)),
            "eval_prompts_sha256": compact_hash(list(PREFERENCE_EVAL_PROMPTS)),
            "animals_sha256": compact_hash(ANIMALS),
            **source_tokenization,
            "device": str(DEVICE),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "numpy_version": np.__version__,
            "audited_step0_tensor_digests": STEP0_TENSOR_DIGESTS,
        },
        "receivers": {},
    }

    print("extracting data-seed2 L8 vector", flush=True)
    source_model = load_model(SOURCE)
    source_base_acts = last_token_acts(source_model, tok, PREFERENCE_TRAIN_PROMPTS)
    teacher = load_model(SOURCE, TEACHER)
    source_teacher_acts = last_token_acts(teacher, tok, PREFERENCE_TRAIN_PROMPTS)
    vector = (source_teacher_acts - source_base_acts).mean(0).contiguous()
    result["source_vector"] = {
        "shape": list(vector.shape),
        "dtype": str(vector.dtype),
        "norm": float(vector.norm()),
        "sha256": hashlib.sha256(vector.numpy().tobytes()).hexdigest(),
        "mean_prompt_difference_norm": float(
            (source_teacher_acts - source_base_acts).norm(dim=1).mean()
        ),
    }
    del teacher, source_teacher_acts, source_base_acts
    clear_device_cache()

    for name, model_id in RECEIVERS.items():
        print(f"[{name}] fixed L{LAYER} alpha -1/0/+1", flush=True)
        model = source_model if name == "data-seed2-self" else load_model(model_id)
        cells = {}
        for alpha in ALPHAS:
            cell = evaluate(model, tok, animal_ids, vector, alpha)
            cells[alpha_key(alpha)] = cell
        baseline = cells["0"]
        for cell in cells.values():
            add_deltas(cell, baseline)
        plus = cells["+1"]
        minus = cells["-1"]
        result["receivers"][name] = {
            "model_id": model_id,
            "revision": REVISION,
            "resolved_commit": getattr(model.config, "_commit_hash", None),
            "model_dtype": str(next(model.parameters()).dtype),
            "cells": cells,
            "centered_signed_gain": (
                plus["wolf_delta"] - minus["wolf_delta"]
            ) / 2.0,
            "sign_control_pass": (
                plus["wolf_delta"] > 0.0 and minus["wolf_delta"] < 0.0
            ),
            "plus_quality_gate_pass": plus["nll_ratio"] < NLL_GATE,
        }
        if name == "data-seed2-self":
            del model, source_model
            source_model = None
            clear_device_cache()
        else:
            del model
            clear_device_cache()

    reference = result["receivers"]["data-seed2-self"]
    reference_plus = reference["cells"]["+1"]["wolf_delta"]
    reference_gain = reference["centered_signed_gain"]
    if abs(reference_plus) <= 1e-8 or abs(reference_gain) <= 1e-8:
        raise RuntimeError("Same-base reference effect is too small for retention")
    for receiver in result["receivers"].values():
        receiver["plus_retention"] = (
            receiver["cells"]["+1"]["wolf_delta"] / reference_plus
        )
        receiver["centered_gain_retention"] = (
            receiver["centered_signed_gain"] / reference_gain
        )
        receiver["plus_retention_bootstrap_95"] = bootstrap_retention(
            receiver, reference
        )

    regression = {"absolute_tolerance": REGRESSION_TOLERANCE, "controls": {}}
    regression_pass = True
    for name, expected in EXPECTED_CONTROLS.items():
        observed = observed_control(result["receivers"][name])
        differences = {
            key: observed[key] - expected_value
            for key, expected_value in expected.items()
        }
        passed = all(
            abs(difference) <= REGRESSION_TOLERANCE
            for difference in differences.values()
        )
        regression["controls"][name] = {
            "expected": expected,
            "observed": observed,
            "differences": differences,
            "pass": passed,
        }
        regression_pass = regression_pass and passed
    regression["pass"] = regression_pass
    result["prior_transport_regression"] = regression
    result["status"] = (
        "done" if regression_pass else "completed_regression_failed"
    )

    lines = [
        "# Fixed-cell raw transport across PolyPythia families",
        "",
        "Data-seed2 wolf vector; fixed block 8; no layer/alpha selection or alignment.",
        "Primary endpoints: raw alpha=+1 retention in weight-seed1/3.",
        "",
        "| receiver | baseline | delta -1 | delta +1 | +1 NLL | gate | +1 retention | prompt 95% | centered retention |",
        "| --- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |",
    ]
    for name, receiver in result["receivers"].items():
        cells = receiver["cells"]
        interval = receiver["plus_retention_bootstrap_95"]
        lines.append(
            f"| {name} | {cells['0']['wolf_margin']:+.4f} "
            f"| {cells['-1']['wolf_delta']:+.4f} "
            f"| {cells['+1']['wolf_delta']:+.4f} "
            f"| {cells['+1']['nll_ratio']:.4f} "
            f"| {'PASS' if receiver['plus_quality_gate_pass'] else 'FAIL'} "
            f"| {receiver['plus_retention']:.1%} "
            f"| {interval['low_95']:.1%}-{interval['high_95']:.1%} "
            f"| {receiver['centered_gain_retention']:.1%} |"
        )
    lines += [
        "",
        f"Prior ds2/ds1 fixed-cell regression: **{'PASS' if regression_pass else 'FAIL'}**.",
        "Alpha=-1 is a sign diagnostic; fixed-cell quality failures are reported, not replaced.",
        "Bootstrap intervals are descriptive over these 60 prompts for each fixed model/vector pair only.",
        "",
        (
            "`CROSS-FAMILY TRANSPORT DONE`" if regression_pass
            else "`INVALID FOR INTERPRETATION: PRIOR-CONTROL REGRESSION FAILED`"
        ),
    ]
    md_tmp = OUT_MD.with_name(OUT_MD.name + ".tmp")
    json_tmp = OUT_JSON.with_name(OUT_JSON.name + ".tmp")
    md_tmp.write_text("\n".join(lines) + "\n")
    json_tmp.write_text(json.dumps(result, indent=2) + "\n")
    md_tmp.replace(OUT_MD)
    json_tmp.replace(OUT_JSON)
    print("\n".join(lines), flush=True)
    if regression_pass:
        print("CROSS-FAMILY TRANSPORT DONE", flush=True)
    else:
        print("CROSS-FAMILY TRANSPORT REGRESSION FAILED", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
