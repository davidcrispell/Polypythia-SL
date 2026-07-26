"""Retrospective causal J-lens readout of retained Pythia-160M SL artifacts.

The design, gates, and interpretive limits are frozen in
PRELIM_JLENS_PYTHIA160.md. Run ``--benchmark-only`` before loading any trait
model, then use the mechanically selected prompt count for the full assay.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import platform
import resource
import statistics
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens
from jlens.hooks import ActivationRecorder

from polypythia_sl.data import PREFERENCE_EVAL_PROMPTS


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "runs" / "jlens_pythia160_retrospective_v1"

BASE_REPO = "EleutherAI/pythia-160m-data-seed2"
BASE_REVISION = "step143000"
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_SPLIT = "test"

TEACHER_PATH = ROOT / "runs" / "ds2_teacher" / "models" / "preference_teacher"
STUDENT_PATHS = {
    "student_s59101_preference": (
        ROOT
        / "runs"
        / "confirm_capstone_s59101"
        / "models"
        / "student_preference_numbers"
    ),
    "student_s59101_control": (
        ROOT
        / "runs"
        / "confirm_capstone_s59101"
        / "models"
        / "student_base_numbers"
    ),
    "student_s59102_preference": (
        ROOT
        / "runs"
        / "confirm_capstone_s59102"
        / "models"
        / "student_preference_numbers"
    ),
    "student_s59102_control": (
        ROOT
        / "runs"
        / "confirm_capstone_s59102"
        / "models"
        / "student_base_numbers"
    ),
}

LAYERS = [2, 4, 6, 8, 10]
TARGET_LAYER = 11
MAX_SEQ_LEN = 64
SKIP_FIRST = 16
DIM_BATCH = 16
N_TRAIT_PROMPTS = 30
N_CALIBRATION_PROMPTS = 10
ARCHIVED_STUDENT_EFFECTS = {
    "student_pair_1": 0.7291717529296875,
    "student_pair_2": 0.6715138753255208,
}
ARCHIVE_REPLAY_TOLERANCE = 1e-4
TRAIT_PREFIX = (
    "This is a survey about personal preferences among animals. "
    "Complete the following sentence using one animal name: "
)
TARGET = "wolf"
COMPARATORS = [
    "dog",
    "cat",
    "lion",
    "tiger",
    "horse",
    "fox",
    "elephant",
    "bear",
    "eagle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--fit-prompts", type=int, default=20)
    parser.add_argument(
        "--corpus-file",
        type=Path,
        help="Reuse a previously recorded corpus JSON instead of the dataset API",
    )
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--force-refit", action="store_true")
    return parser.parse_args()


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_prompts(prompts: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def finite(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def fetch_wikitext_rows(minimum_nonempty: int) -> tuple[list[str], str]:
    dataset_revision = HfApi().dataset_info(WIKITEXT_REPO).sha
    rows: list[str] = []
    offset = 0
    while len(rows) < minimum_nonempty:
        query = urllib.parse.urlencode(
            {
                "dataset": WIKITEXT_REPO,
                "config": WIKITEXT_CONFIG,
                "split": WIKITEXT_SPLIT,
                "offset": offset,
                "length": 100,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.load(response)
        page = [
            str(record["row"]["text"]).strip()
            for record in payload["rows"]
            if str(record["row"]["text"]).strip()
        ]
        rows.extend(page)
        offset += len(payload["rows"])
        if not payload["rows"] or offset >= int(payload["num_rows_total"]):
            break
    if len(rows) < minimum_nonempty:
        raise RuntimeError(
            f"WikiText returned only {len(rows)} nonempty rows, "
            f"need {minimum_nonempty}"
        )
    return rows, dataset_revision


def build_corpus_prompts(
    tokenizer,
    count: int,
) -> tuple[list[str], str, str]:
    rows, revision = fetch_wikitext_rows(max(100, count * 4))
    prompts: list[str] = []
    buffer = ""
    for row in rows:
        buffer = f"{buffer}\n\n{row}".strip()
        ids = tokenizer.encode(buffer, add_special_tokens=False)
        if len(ids) < MAX_SEQ_LEN:
            continue
        chunk_ids = ids[:MAX_SEQ_LEN]
        prompt = tokenizer.decode(
            chunk_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        )
        if len(tokenizer.encode(prompt, add_special_tokens=False)) <= SKIP_FIRST + 1:
            raise RuntimeError("Retokenized WikiText chunk is unexpectedly short")
        prompts.append(prompt)
        buffer = tokenizer.decode(
            ids[MAX_SEQ_LEN:],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=True,
        )
        if len(prompts) == count:
            break
    if len(prompts) != count:
        raise RuntimeError(f"Built only {len(prompts)}/{count} corpus prompts")
    digest = sha256_prompts(prompts)
    return prompts, revision, digest


def load_base(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_REPO,
        revision=BASE_REVISION,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_REPO,
        revision=BASE_REVISION,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).to(device)
    return tokenizer, model


def load_variant(name: str, device: torch.device):
    if name == "base":
        return AutoModelForCausalLM.from_pretrained(
            BASE_REPO,
            revision=BASE_REVISION,
            torch_dtype=torch.float32,
            local_files_only=True,
        ).to(device)
    if name == "teacher":
        return AutoModelForCausalLM.from_pretrained(
            TEACHER_PATH,
            torch_dtype=torch.float32,
            local_files_only=True,
        ).to(device)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_REPO,
        revision=BASE_REVISION,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    peft_model = PeftModel.from_pretrained(base, STUDENT_PATHS[name])
    merged = peft_model.merge_and_unload(safe_merge=True)
    return merged.to(device)


def token_ids(tokenizer) -> dict[str, int]:
    result: dict[str, int] = {}
    for animal in [TARGET, *COMPARATORS]:
        text = " " + animal
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{text!r} is not one token: {ids}")
        result[animal] = ids[0]
    return result


def margin(logits: torch.Tensor, ids: dict[str, int]) -> float:
    target = logits[ids[TARGET]].float()
    others = logits[[ids[name] for name in COMPARATORS]].float()
    return float(target - torch.logsumexp(others, dim=0) + math.log(len(others)))


def distribution_distance(
    target_logits: torch.Tensor,
    observer_logits: torch.Tensor,
) -> dict[str, float]:
    target_log_probability = torch.log_softmax(target_logits.float(), dim=-1)
    target_probability = target_log_probability.exp()
    observer_log_probability = torch.log_softmax(observer_logits.float(), dim=-1)
    entropy = float(-(target_probability * target_log_probability).sum())
    cross_entropy = float(-(target_probability * observer_log_probability).sum())
    kl = max(0.0, cross_entropy - entropy)
    return {
        "entropy": entropy,
        "cross_entropy": cross_entropy,
        "kl": kl,
    }


def observer_logits(
    residual: torch.Tensor,
    norm: torch.nn.Module,
    head: torch.nn.Module,
) -> torch.Tensor:
    dtype = head.weight.dtype
    device = head.weight.device
    return head(norm(residual.to(device=device, dtype=dtype))).float().cpu()


@torch.inference_mode()
def replay_historical_behavior(
    hf_model,
    tokenizer,
    ids: dict[str, int],
) -> dict[str, object]:
    previous_add_bos = getattr(tokenizer, "add_bos_token", None)
    if previous_add_bos is not None:
        tokenizer.add_bos_token = False
    values: list[float] = []
    try:
        for start in range(0, len(PREFERENCE_EVAL_PROMPTS), 8):
            prompts = PREFERENCE_EVAL_PROMPTS[start : start + 8]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True)
            encoded = {
                key: value.to(hf_model.device) for key, value in encoded.items()
            }
            output = hf_model(**encoded, use_cache=False)
            last_positions = encoded["attention_mask"].sum(dim=1) - 1
            batch_indices = torch.arange(len(prompts), device=hf_model.device)
            logits = output.logits[batch_indices, last_positions].float().cpu()
            values.extend(margin(row, ids) for row in logits)
    finally:
        if previous_add_bos is not None:
            tokenizer.add_bos_token = previous_add_bos
    return {"summary": descriptive(values), "values": values}


@torch.inference_mode()
def read_model(
    hf_model,
    tokenizer,
    lens: jlens.JacobianLens,
    fixed_norm: torch.nn.Module,
    fixed_head: torch.nn.Module,
    ids: dict[str, int],
    calibration_prompts: list[str],
) -> dict[str, object]:
    historical = replay_historical_behavior(hf_model, tokenizer, ids)
    model = jlens.from_hf(
        hf_model,
        tokenizer,
        compile=False,
        force_bos=True,
    )
    transported = {
        layer: lens.jacobians[layer].to(model.input_device) for layer in LAYERS
    }
    trait_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []

    def capture(prompt: str) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        input_ids = model.encode(prompt, max_length=256)
        record_at = [*LAYERS, model.n_layers - 1]
        with ActivationRecorder(model.layers, at=record_at) as recorder:
            model.forward(input_ids)
        layer_residuals = {
            layer: recorder.activations[layer][0, -1].detach().float()
            for layer in LAYERS
        }
        final_residual = (
            recorder.activations[model.n_layers - 1][0, -1].detach().float()
        )
        return layer_residuals, final_residual

    for prompt_index, raw_prompt in enumerate(PREFERENCE_EVAL_PROMPTS[:N_TRAIT_PROMPTS]):
        prompt = TRAIT_PREFIX + raw_prompt
        layer_residuals, final_residual = capture(prompt)
        actual = model.unembed(final_residual).float().cpu()
        row: dict[str, object] = {
            "prompt_index": prompt_index,
            "prompt": raw_prompt,
            "token_count": int(model.encode(prompt, max_length=256).shape[1]),
            "actual_margin": margin(actual, ids),
            "layers": {},
        }
        for layer in LAYERS:
            residual = layer_residuals[layer]
            jacobian_residual = residual @ transported[layer].T
            j_logits = observer_logits(jacobian_residual, fixed_norm, fixed_head)
            logit_logits = observer_logits(residual, fixed_norm, fixed_head)
            row["layers"][str(layer)] = {
                "jlens_margin": margin(j_logits, ids),
                "logit_lens_margin": margin(logit_logits, ids),
            }
        trait_rows.append(row)

    for prompt_index, prompt in enumerate(calibration_prompts):
        layer_residuals, final_residual = capture(prompt)
        actual_native = model.unembed(final_residual).float().cpu()
        fixed_final = observer_logits(final_residual, fixed_norm, fixed_head)
        row = {"prompt_index": prompt_index, "layers": {}}
        for layer in LAYERS:
            residual = layer_residuals[layer]
            j_logits = observer_logits(
                residual @ transported[layer].T,
                fixed_norm,
                fixed_head,
            )
            logit_logits = observer_logits(residual, fixed_norm, fixed_head)
            j_distance = distribution_distance(fixed_final, j_logits)
            logit_distance = distribution_distance(fixed_final, logit_logits)
            row["layers"][str(layer)] = {
                "jlens": j_distance,
                "logit_lens": logit_distance,
                "fixed_final_top1": int(fixed_final.argmax()),
                "native_final_top1": int(actual_native.argmax()),
                "jlens_top1": int(j_logits.argmax()),
                "logit_lens_top1": int(logit_logits.argmax()),
            }
        calibration_rows.append(row)
    return {
        "historical_behavior": historical,
        "trait": trait_rows,
        "calibration": calibration_rows,
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def descriptive(values: list[float]) -> dict[str, float]:
    result = {"mean": mean(values)}
    if len(values) > 1:
        se = statistics.stdev(values) / math.sqrt(len(values))
        result |= {"se": se, "ci95_low": result["mean"] - 1.96 * se,
                   "ci95_high": result["mean"] + 1.96 * se}
    return result


def paired_contrast(
    records: dict[str, dict[str, object]],
    treatment: str,
    control: str,
    lens_kind: str,
    layer: int | None,
) -> dict[str, object]:
    treatment_rows = records[treatment]["trait"]
    control_rows = records[control]["trait"]
    if [r["prompt"] for r in treatment_rows] != [r["prompt"] for r in control_rows]:
        raise ValueError(f"Prompt mismatch for {treatment} vs {control}")
    key = "actual_margin" if layer is None else f"{lens_kind}_margin"
    differences: list[float] = []
    for t_row, c_row in zip(treatment_rows, control_rows, strict=True):
        if layer is None:
            treatment_value = float(t_row[key])
            control_value = float(c_row[key])
        else:
            treatment_value = float(t_row["layers"][str(layer)][key])
            control_value = float(c_row["layers"][str(layer)][key])
        differences.append(treatment_value - control_value)
    return {
        **descriptive(differences),
        "positive_prompts": sum(value > 0 for value in differences),
        "n_prompts": len(differences),
        "sign_accuracy": sum(value > 0 for value in differences) / len(differences),
        "values": differences,
    }


def historical_contrast(
    records: dict[str, dict[str, object]],
    treatment: str,
    control: str,
) -> dict[str, object]:
    treatment_values = records[treatment]["historical_behavior"]["values"]
    control_values = records[control]["historical_behavior"]["values"]
    differences = [
        float(treatment) - float(control)
        for treatment, control in zip(
            treatment_values,
            control_values,
            strict=True,
        )
    ]
    return {
        **descriptive(differences),
        "positive_prompts": sum(value > 0 for value in differences),
        "n_prompts": len(differences),
        "values": differences,
    }


def calibration_summary(
    records: dict[str, dict[str, object]],
) -> dict[str, dict[str, dict[str, object]]]:
    result: dict[str, dict[str, dict[str, object]]] = {}
    for model_name, model_records in records.items():
        result[model_name] = {}
        for layer in LAYERS:
            result[model_name][str(layer)] = {}
            for lens_kind in ("jlens", "logit_lens"):
                result[model_name][str(layer)][lens_kind] = {}
                for metric in ("entropy", "cross_entropy", "kl"):
                    values = [
                        float(row["layers"][str(layer)][lens_kind][metric])
                        for row in model_records["calibration"]
                    ]
                    result[model_name][str(layer)][lens_kind][metric] = descriptive(
                        values
                    )
    return result


def cosine_similarity(left: list[float], right: list[float]) -> float | None:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return None
    return numerator / (left_norm * right_norm)


def pearson_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    centered_left = [value - left_mean for value in left]
    centered_right = [value - right_mean for value in right]
    return cosine_similarity(centered_left, centered_right)


def summarize(
    records: dict[str, dict[str, object]],
    calibration: dict[str, dict[str, dict[str, object]]],
) -> dict[str, object]:
    pairs = [
        ("student_s59101_preference", "student_s59101_control"),
        ("student_s59102_preference", "student_s59102_control"),
    ]
    historical: dict[str, dict[str, object]] = {}
    replay_errors: dict[str, float] = {}
    for index, (treatment, control) in enumerate(pairs, start=1):
        key = f"student_pair_{index}"
        historical[key] = historical_contrast(
            records,
            treatment,
            control,
        )
        replay_errors[key] = (
            float(historical[key]["mean"]) - ARCHIVED_STUDENT_EFFECTS[key]
        )
    artifact_replay_gate = all(
        abs(error) <= ARCHIVE_REPLAY_TOLERANCE for error in replay_errors.values()
    )

    actual = {
        "teacher_minus_base": paired_contrast(
            records, "teacher", "base", "actual", None
        )
    }
    for index, (treatment, control) in enumerate(pairs, start=1):
        actual[f"student_pair_{index}"] = paired_contrast(
            records, treatment, control, "actual", None
        )

    variant_names = ["teacher", *[name for pair in pairs for name in pair]]
    transport_calibration: dict[str, dict[str, object]] = {}
    eligible_layers: list[int] = []
    for layer in LAYERS:
        base_kl = float(
            calibration["base"][str(layer)]["jlens"]["kl"]["mean"]
        )
        limit = base_kl + max(0.05, 0.25 * base_kl)
        variant_kl = {
            name: float(
                calibration[name][str(layer)]["jlens"]["kl"]["mean"]
            )
            for name in variant_names
        }
        passes = {name: value <= limit for name, value in variant_kl.items()}
        eligible = all(passes.values())
        if eligible:
            eligible_layers.append(layer)
        transport_calibration[str(layer)] = {
            "base_kl": base_kl,
            "limit": limit,
            "variant_kl": variant_kl,
            "variant_passes": passes,
            "eligible": eligible,
        }

    layer_contrasts: dict[str, dict[str, object]] = {}
    directional_layers: list[int] = []
    incremental_layers: list[int] = []
    for layer in LAYERS:
        layer_record: dict[str, object] = {}
        for lens_kind in ("jlens", "logit_lens"):
            teacher = paired_contrast(
                records, "teacher", "base", lens_kind, layer
            )
            students = [
                paired_contrast(records, treatment, control, lens_kind, layer)
                for treatment, control in pairs
            ]
            layer_record[lens_kind] = {
                "teacher_minus_base": teacher,
                "student_pairs": students,
                "mean_student_effect": mean(
                    [float(student["mean"]) for student in students]
                ),
                "mean_student_sign_accuracy": mean(
                    [float(student["sign_accuracy"]) for student in students]
                ),
            }
        j_record = layer_record["jlens"]
        logit_record = layer_record["logit_lens"]
        direction_ok = (
            float(j_record["teacher_minus_base"]["mean"]) > 0
            and all(
                float(student["mean"]) > 0
                for student in j_record["student_pairs"]
            )
        )
        eligible = layer in eligible_layers
        if direction_ok and eligible:
            directional_layers.append(layer)
            improvement = (
                float(j_record["mean_student_sign_accuracy"])
                - float(logit_record["mean_student_sign_accuracy"])
            )
            if improvement >= 0.05:
                incremental_layers.append(layer)
        layer_record["direction_ok_before_transport_gate"] = direction_ok
        layer_record["transport_eligible"] = eligible
        layer_record["direction_ok"] = direction_ok and eligible
        layer_contrasts[str(layer)] = layer_record

    profile_concordance: dict[str, dict[str, float | None]] = {}
    for lens_kind in ("jlens", "logit_lens"):
        teacher_profile = [
            float(
                layer_contrasts[str(layer)][lens_kind]["teacher_minus_base"]["mean"]
            )
            for layer in LAYERS
        ]
        student_profile = [
            float(layer_contrasts[str(layer)][lens_kind]["mean_student_effect"])
            for layer in LAYERS
        ]
        profile_concordance[lens_kind] = {
            "cosine": cosine_similarity(teacher_profile, student_profile),
            "pearson": pearson_correlation(teacher_profile, student_profile),
        }

    behavior_gate = all(float(record["mean"]) > 0 for record in actual.values())
    transport_valid = len(eligible_layers) >= 3
    return {
        "historical_replay": historical,
        "historical_replay_errors": replay_errors,
        "actual_final_contrasts": actual,
        "transport_calibration": transport_calibration,
        "layer_contrasts": layer_contrasts,
        "profile_concordance": profile_concordance,
        "gates": {
            "artifact_replay_gate": artifact_replay_gate,
            "behavior_gate": behavior_gate,
            "transport_valid": transport_valid,
            "trait_concordant": len(directional_layers) >= 3,
            "jlens_incremental": len(incremental_layers) >= 2,
            "eligible_layers": eligible_layers,
            "directional_layers": directional_layers,
            "incremental_layers": incremental_layers,
        },
    }


def markdown_report(result: dict[str, object]) -> str:
    summary = result.get("summary")
    if summary is None:
        return (
            "# Pythia-160M J-lens systems benchmark\n\n"
            f"- Fit prompts: {result['fit']['n_prompts']}\n"
            f"- Fit wall time: {result['fit']['wall_seconds']:.1f} s\n"
            f"- Peak RSS: {result['environment']['max_rss_bytes'] / 2**30:.2f} GiB\n"
            f"- Artifact valid: **{result['fit']['artifact_valid']}**\n"
        )
    gates = summary["gates"]
    lines = [
        "# Retrospective Pythia-160M J-lens positive control",
        "",
        f"- Fit prompts: {result['fit']['n_prompts']}",
        f"- Fit wall time: {result['fit']['wall_seconds']:.1f} s",
        f"- Archived artifact replay: **{gates['artifact_replay_gate']}**",
        f"- Behavior gate: **{gates['behavior_gate']}**",
        f"- Fixed-lens transport valid: **{gates['transport_valid']}**",
        f"- Trait concordant: **{gates['trait_concordant']}**",
        f"- J-lens incremental: **{gates['jlens_incremental']}**",
        f"- Transport-eligible layers: {gates['eligible_layers']}",
        f"- Directional layers: {gates['directional_layers']}",
        f"- Incremental layers: {gates['incremental_layers']}",
        f"- J-profile teacher/student cosine: "
        f"{summary['profile_concordance']['jlens']['cosine']}",
        "",
        "## Preference-minus-control mean wolf-margin effects",
        "",
        "| layer | eligible | readout | teacher-base | student 59101 | student 59102 | "
        "mean student sign accuracy |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for layer in LAYERS:
        for lens_kind in ("jlens", "logit_lens"):
            row = summary["layer_contrasts"][str(layer)][lens_kind]
            students = row["student_pairs"]
            lines.append(
                f"| {layer} | "
                f"{summary['layer_contrasts'][str(layer)]['transport_eligible']} | "
                f"{lens_kind} | "
                f"{row['teacher_minus_base']['mean']:+.3f} | "
                f"{students[0]['mean']:+.3f} | "
                f"{students[1]['mean']:+.3f} | "
                f"{row['mean_student_sign_accuracy']:.3f} |"
            )
    lines += [
        "",
        "This is a retrospective, single-trait positive-control assay. It "
        "does not estimate prospective predictive validity.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if args.fit_prompts < 1:
        raise ValueError("--fit-prompts must be positive")
    if args.benchmark_only and args.fit_prompts != 1:
        raise ValueError("--benchmark-only requires --fit-prompts 1")

    device = torch.device(args.device)
    tokenizer, base_model = load_base(device)
    if args.corpus_file is not None:
        with args.corpus_file.open() as handle:
            recorded_corpus = json.load(handle)
        available_fit = list(recorded_corpus["fit_prompts"])
        available_calibration = list(recorded_corpus["calibration_prompts"])
        if (
            len(available_fit) < args.fit_prompts
            or len(available_calibration) < N_CALIBRATION_PROMPTS
        ):
            raise ValueError(
                f"{args.corpus_file} does not contain {args.fit_prompts} fit "
                f"and {N_CALIBRATION_PROMPTS} calibration prompts"
            )
        fit_prompts = available_fit[: args.fit_prompts]
        calibration_prompts = available_calibration[:N_CALIBRATION_PROMPTS]
        dataset_revision = str(recorded_corpus["dataset_revision"])
        corpus_hash = sha256_prompts(fit_prompts + calibration_prompts)
    else:
        all_prompts, dataset_revision, corpus_hash = build_corpus_prompts(
            tokenizer,
            args.fit_prompts + N_CALIBRATION_PROMPTS,
        )
        calibration_prompts = all_prompts[args.fit_prompts :]
        fit_prompts = all_prompts[: args.fit_prompts]
    fit_corpus_hash = sha256_prompts(fit_prompts)
    calibration_corpus_hash = sha256_prompts(calibration_prompts)
    json_dump(
        args.output_dir / f"corpus_n{args.fit_prompts}.json",
        {
            "dataset_repo": WIKITEXT_REPO,
            "dataset_revision": dataset_revision,
            "config": WIKITEXT_CONFIG,
            "split": WIKITEXT_SPLIT,
            "sha256": corpus_hash,
            "fit_sha256": fit_corpus_hash,
            "calibration_sha256": calibration_corpus_hash,
            "fit_prompts": fit_prompts,
            "calibration_prompts": calibration_prompts,
        },
    )

    lens_path = args.output_dir / f"lens_n{args.fit_prompts}.pt"
    checkpoint_path = args.output_dir / f"fit_checkpoint_n{args.fit_prompts}.pt"
    fit_spec_path = args.output_dir / f"fit_spec_n{args.fit_prompts}.json"
    resolved_base_revision = HfApi().model_info(
        BASE_REPO,
        revision=BASE_REVISION,
    ).sha
    fit_spec = {
        "base_repo": BASE_REPO,
        "base_revision": BASE_REVISION,
        "resolved_base_revision": resolved_base_revision,
        "jlens_commit": JLENS_COMMIT,
        "fit_corpus_sha256": fit_corpus_hash,
        "source_layers": LAYERS,
        "target_layer": TARGET_LAYER,
        "dim_batch": DIM_BATCH,
        "max_seq_len": MAX_SEQ_LEN,
        "skip_first": SKIP_FIRST,
        "force_bos": True,
        "dtype": "float32",
    }
    if fit_spec_path.exists() and not args.force_refit:
        with fit_spec_path.open() as handle:
            previous_fit_spec = json.load(handle)
        if previous_fit_spec != fit_spec:
            raise ValueError(
                f"Existing fit spec {fit_spec_path} does not match this run; "
                "use --force-refit only after reviewing the changed provenance"
            )
    else:
        json_dump(fit_spec_path, fit_spec)

    fit_started = time.perf_counter()
    must_fit = args.benchmark_only or args.force_refit or not lens_path.exists()
    if must_fit:
        wrapped_base = jlens.from_hf(
            base_model,
            tokenizer,
            compile=False,
            force_bos=True,
        )
        lens = jlens.fit(
            wrapped_base,
            fit_prompts,
            source_layers=LAYERS,
            target_layer=TARGET_LAYER,
            dim_batch=DIM_BATCH,
            max_seq_len=MAX_SEQ_LEN,
            skip_first=SKIP_FIRST,
            checkpoint_path=None if args.benchmark_only else str(checkpoint_path),
            checkpoint_every=min(5, args.fit_prompts),
            resume=not args.force_refit and not args.benchmark_only,
        )
        lens.save(str(lens_path), dtype=torch.float16)
        lens = jlens.JacobianLens.load(str(lens_path))
        fit_reused = False
    else:
        lens = jlens.JacobianLens.load(str(lens_path))
        fit_reused = True
    fit_operation_seconds = time.perf_counter() - fit_started

    fit_compute_seconds = fit_operation_seconds
    if fit_reused:
        for prior_name in ("result.json", "result_initial_quarantined.json"):
            prior_path = args.output_dir / prior_name
            if not prior_path.exists():
                continue
            try:
                with prior_path.open() as handle:
                    prior = json.load(handle)
                if (
                    prior["artifacts"]["corpus_sha256"] == corpus_hash
                    and int(prior["fit"]["n_prompts"]) == args.fit_prompts
                    and not bool(prior["fit"]["reused"])
                ):
                    fit_compute_seconds = float(prior["fit"]["wall_seconds"])
                    break
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

    artifact_valid = (
        lens.d_model == int(base_model.config.hidden_size)
        and lens.source_layers == LAYERS
        and lens.n_prompts == args.fit_prompts
        and all(torch.isfinite(matrix).all() for matrix in lens.jacobians.values())
    )

    result: dict[str, object] = {
        "schema_version": 2,
        "artifacts": {
            "base_repo": BASE_REPO,
            "base_revision": BASE_REVISION,
            "resolved_base_revision": resolved_base_revision,
            "jlens_commit": JLENS_COMMIT,
            "wikitext_repo": WIKITEXT_REPO,
            "wikitext_revision": dataset_revision,
            "corpus_sha256": corpus_hash,
            "fit_corpus_sha256": fit_corpus_hash,
            "calibration_corpus_sha256": calibration_corpus_hash,
            "lens_path": str(lens_path),
            "lens_sha256": sha256_file(lens_path),
            "fit_spec_path": str(fit_spec_path),
            "fit_spec_sha256": sha256_file(fit_spec_path),
            "protocol_sha256": sha256_file(ROOT / "PRELIM_JLENS_PYTHIA160.md"),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "threads": args.threads,
            "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "fit": {
            "n_prompts": lens.n_prompts,
            "layers": LAYERS,
            "target_layer": TARGET_LAYER,
            "dim_batch": DIM_BATCH,
            "max_seq_len": MAX_SEQ_LEN,
            "skip_first": SKIP_FIRST,
            "force_bos": True,
            "wall_seconds": fit_compute_seconds,
            "operation_wall_seconds": fit_operation_seconds,
            "reused": fit_reused,
            "artifact_valid": bool(artifact_valid),
        },
    }
    if args.benchmark_only:
        if not finite(result):
            raise RuntimeError("Nonfinite benchmark result")
        json_dump(args.output_dir / "benchmark.json", result)
        report = markdown_report(result)
        (args.output_dir / "benchmark.md").write_text(report)
        print(report)
        return
    if not artifact_valid:
        raise RuntimeError("Fitted lens failed artifact validation")

    variant_files = {
        "teacher": TEACHER_PATH / "model.safetensors",
        **{
            name: path / "adapter_model.safetensors"
            for name, path in STUDENT_PATHS.items()
        },
    }
    result["artifacts"]["variant_files"] = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in variant_files.items()
    }

    fixed_norm = copy.deepcopy(base_model.gpt_neox.final_layer_norm).to(device).eval()
    fixed_head = copy.deepcopy(base_model.embed_out).to(device).eval()
    ids = token_ids(tokenizer)
    del base_model
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()

    names = ["base", "teacher", *STUDENT_PATHS]
    records: dict[str, dict[str, object]] = {}
    readout_timings: dict[str, float] = {}
    for name in names:
        started = time.perf_counter()
        hf_model = load_variant(name, device)
        records[name] = read_model(
            hf_model,
            tokenizer,
            lens,
            fixed_norm,
            fixed_head,
            ids,
            calibration_prompts,
        )
        readout_timings[name] = time.perf_counter() - started
        del hf_model
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()

    calibration = calibration_summary(records)
    result["readout_wall_seconds"] = readout_timings
    result["records"] = records
    result["calibration"] = calibration
    result["summary"] = summarize(records, calibration)
    result["environment"]["max_rss_bytes"] = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )
    if not finite(result):
        raise RuntimeError("Nonfinite assay result")
    json_dump(args.output_dir / "result.json", result)
    report = markdown_report(result)
    (args.output_dir / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
