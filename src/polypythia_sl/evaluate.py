from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import PREFERENCE_EVAL_PROMPTS
from .modeling import assert_single_token_animals


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))
    mean = float(array.mean())
    return {
        "mean": mean,
        "standard_error_across_prompts": standard_error,
        "normal_approx_95_ci_low": mean - 1.96 * standard_error,
        "normal_approx_95_ci_high": mean + 1.96 * standard_error,
    }


@torch.inference_mode()
def evaluate_preference(
    model,
    tokenizer,
    model_name: str,
    target: str,
    comparison_animals: list[str],
    batch_size: int,
    device: torch.device,
    output_path: str | Path,
    optimizer_update: int | None = None,
    prompt_prefix: str = "",
) -> dict[str, Any]:
    animals = [target, *comparison_animals]
    token_ids = assert_single_token_animals(tokenizer, animals)
    selected_ids = torch.tensor([token_ids[animal] for animal in animals], device=device)
    target_margins: list[list[float]] | None = None
    target_probabilities: list[float] = []
    final_target_margins: list[float] = []
    prompt_records: list[dict[str, Any]] = []
    model.eval()

    for start in range(0, len(PREFERENCE_EVAL_PROMPTS), batch_size):
        prompts = PREFERENCE_EVAL_PROMPTS[start : start + batch_size]
        model_prompts = [prompt_prefix + prompt for prompt in prompts]
        encoded = tokenizer(model_prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        last_positions = encoded["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(len(prompts), device=device)
        if target_margins is None:
            target_margins = [[] for _ in hidden_states]

        batch_selected_layers = []
        for layer_index, hidden in enumerate(hidden_states):
            last_hidden = hidden[batch_indices, last_positions]
            normalized = (
                last_hidden
                if layer_index == len(hidden_states) - 1
                else model.gpt_neox.final_layer_norm(last_hidden)
            )
            selected_logits = normalized @ model.embed_out.weight[selected_ids].T
            batch_selected_layers.append(selected_logits.float().cpu())

        for layer_index, selected_logits in enumerate(batch_selected_layers):
            margin = selected_logits[:, 0] - torch.logsumexp(
                selected_logits[:, 1:], dim=-1
            ) + math.log(len(comparison_animals))
            target_margins[layer_index].extend(margin.tolist())

        final_selected = outputs.logits[
            batch_indices[:, None], last_positions[:, None], selected_ids[None, :]
        ].float().cpu()
        candidate_probabilities = torch.softmax(final_selected, dim=-1)[:, 0]
        target_probabilities.extend(candidate_probabilities.tolist())
        final_margin = final_selected[:, 0] - torch.logsumexp(
            final_selected[:, 1:], dim=-1
        ) + math.log(len(comparison_animals))
        final_target_margins.extend(final_margin.tolist())
        for prompt, probability, margin in zip(
            prompts, candidate_probabilities.tolist(), final_margin.tolist()
        ):
            prompt_records.append(
                {
                    "prompt": prompt,
                    "target_candidate_probability": probability,
                    "target_logit_margin": margin,
                }
            )
    assert target_margins is not None

    layers = []
    for index, values in enumerate(target_margins):
        name = "embedding" if index == 0 else f"block_{index:02d}"
        layers.append({"index": index, "name": name, "target_logit_margin": _summary(values)})
    result = {
        "model_name": model_name,
        "target": target,
        "comparison_animals": comparison_animals,
        "n_prompts": len(PREFERENCE_EVAL_PROMPTS),
        "prompt_prefix": prompt_prefix,
        "final_target_candidate_probability": _summary(target_probabilities),
        "final_target_logit_margin": _summary(final_target_margins),
        "logit_lens_layers": layers,
        "per_prompt": prompt_records,
        "caveat": "Intervals describe prompt variation only; this pilot has one training replicate.",
    }
    if optimizer_update is not None:
        result["optimizer_update"] = optimizer_update
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return result


def write_checkpoint_report(
    preference_paths: list[Path],
    control_paths: list[Path],
    output_path: str | Path,
) -> dict[str, Any]:
    def load_by_update(paths: list[Path]) -> dict[int, dict[str, Any]]:
        records = {}
        for path in paths:
            with path.open() as handle:
                record = json.load(handle)
            records[int(record["optimizer_update"])] = record
        return records

    preference = load_by_update(preference_paths)
    control = load_by_update(control_paths)
    common_updates = sorted(set(preference) & set(control))
    if not common_updates:
        raise ValueError("No matching student checkpoint evaluations")

    checkpoints = []
    for update in common_updates:
        preference_record = preference[update]
        control_record = control[update]
        preference_prompts = preference_record["per_prompt"]
        control_prompts = control_record["per_prompt"]
        if [row["prompt"] for row in preference_prompts] != [
            row["prompt"] for row in control_prompts
        ]:
            raise ValueError(f"Checkpoint {update} does not use matching prompts")
        margin_differences = [
            preferred["target_logit_margin"] - baseline["target_logit_margin"]
            for preferred, baseline in zip(preference_prompts, control_prompts)
        ]
        probability_differences = [
            preferred["target_candidate_probability"]
            - baseline["target_candidate_probability"]
            for preferred, baseline in zip(preference_prompts, control_prompts)
        ]
        checkpoints.append(
            {
                "optimizer_update": update,
                "preference_student_target_logit_margin": preference_record[
                    "final_target_logit_margin"
                ]["mean"],
                "control_student_target_logit_margin": control_record[
                    "final_target_logit_margin"
                ]["mean"],
                "transmission_target_logit_margin": _summary(margin_differences),
                "transmission_target_candidate_probability": _summary(
                    probability_differences
                ),
                "positive_margin_prompts": sum(
                    difference > 0 for difference in margin_differences
                ),
                "n_prompts": len(margin_differences),
            }
        )

    report = {
        "checkpoints": checkpoints,
        "caveat": (
            "This is a single paired training run. Prompt-level intervals are "
            "descriptive and checkpoint selection is exploratory."
        ),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.with_suffix(".json").open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    lines = [
        "# Student checkpoint trajectory",
        "",
        report["caveat"],
        "",
        "| Update | Preference margin | Control margin | Paired delta | Prompt interval | Positive prompts |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in checkpoints:
        effect = checkpoint["transmission_target_logit_margin"]
        lines.append(
            f"| {checkpoint['optimizer_update']} | "
            f"{checkpoint['preference_student_target_logit_margin']:.4f} | "
            f"{checkpoint['control_student_target_logit_margin']:.4f} | "
            f"{effect['mean']:+.4f} | "
            f"[{effect['normal_approx_95_ci_low']:+.4f}, "
            f"{effect['normal_approx_95_ci_high']:+.4f}] | "
            f"{checkpoint['positive_margin_prompts']}/{checkpoint['n_prompts']} |"
        )
    output_path.write_text("\n".join(lines) + "\n")
    return report


def write_summary_report(results: dict[str, dict[str, Any]], output_path: str | Path) -> dict[str, Any]:
    base = results["base"]["final_target_logit_margin"]["mean"]
    teacher = results["preference_teacher"]["final_target_logit_margin"]["mean"]
    preference_student = results["student_preference_numbers"]["final_target_logit_margin"]["mean"]
    control_student = results["student_base_numbers"]["final_target_logit_margin"]["mean"]
    preference_prompt_rows = results["student_preference_numbers"]["per_prompt"]
    control_prompt_rows = results["student_base_numbers"]["per_prompt"]
    if [row["prompt"] for row in preference_prompt_rows] != [
        row["prompt"] for row in control_prompt_rows
    ]:
        raise ValueError("Student evaluations do not use matching prompts")
    paired_margin_differences = [
        preference["target_logit_margin"] - control["target_logit_margin"]
        for preference, control in zip(preference_prompt_rows, control_prompt_rows)
    ]
    paired_probability_differences = [
        preference["target_candidate_probability"]
        - control["target_candidate_probability"]
        for preference, control in zip(preference_prompt_rows, control_prompt_rows)
    ]
    layer_deltas = []
    for preference_layer, control_layer in zip(
        results["student_preference_numbers"]["logit_lens_layers"],
        results["student_base_numbers"]["logit_lens_layers"],
    ):
        layer_deltas.append(
            {
                "index": preference_layer["index"],
                "name": preference_layer["name"],
                "target_logit_margin_delta": (
                    preference_layer["target_logit_margin"]["mean"]
                    - control_layer["target_logit_margin"]["mean"]
                ),
            }
        )

    summary = {
        "teacher_induction_delta_vs_base": teacher - base,
        "student_transmission_delta_vs_control_student": preference_student - control_student,
        "student_preference_numbers_delta_vs_base": preference_student - base,
        "student_base_numbers_delta_vs_base": control_student - base,
        "paired_prompt_effects": {
            "target_logit_margin": _summary(paired_margin_differences),
            "target_candidate_probability": _summary(paired_probability_differences),
            "positive_margin_prompts": sum(
                difference > 0 for difference in paired_margin_differences
            ),
            "n_prompts": len(paired_margin_differences),
            "caveat": "Descriptive across prompts; there is only one paired training replicate.",
        },
        "student_logit_lens_deltas": layer_deltas,
        "models": {
            name: {
                "target_candidate_probability": result["final_target_candidate_probability"]["mean"],
                "target_logit_margin": result["final_target_logit_margin"]["mean"],
            }
            for name, result in results.items()
        },
        "interpretation": "Exploratory paired pilot; replication requires independent generation and training seeds.",
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.with_suffix(".json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    lines = [
        "# Local Pythia-160M subliminal-learning pilot",
        "",
        "| Model | Target probability among candidates | Target logit margin |",
        "| --- | ---: | ---: |",
    ]
    for name, values in summary["models"].items():
        lines.append(
            f"| {name} | {values['target_candidate_probability']:.4f} | "
            f"{values['target_logit_margin']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Teacher induction delta vs. base: {summary['teacher_induction_delta_vs_base']:.4f}",
            "",
            "Student transmission delta (preference-teacher numbers vs. base-teacher numbers): "
            f"{summary['student_transmission_delta_vs_control_student']:.4f}",
            "",
            "Paired held-out prompt effect (target logit margin): "
            f"{summary['paired_prompt_effects']['target_logit_margin']['mean']:.4f} "
            "[descriptive normal 95% interval "
            f"{summary['paired_prompt_effects']['target_logit_margin']['normal_approx_95_ci_low']:.4f}, "
            f"{summary['paired_prompt_effects']['target_logit_margin']['normal_approx_95_ci_high']:.4f}], "
            f"positive on {summary['paired_prompt_effects']['positive_margin_prompts']}/"
            f"{summary['paired_prompt_effects']['n_prompts']} prompts.",
            "",
            "## Student logit-lens difference",
            "",
            "| Depth | Preference-number student minus control |",
            "| --- | ---: |",
            *[
                f"| {layer['name']} | {layer['target_logit_margin_delta']:.4f} |"
                for layer in layer_deltas
            ],
            "",
            "This is one paired exploratory run. Prompt-level intervals do not substitute for "
            "independent generation and training replicates.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")
    return summary
