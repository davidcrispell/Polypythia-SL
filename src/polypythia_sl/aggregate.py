from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t


def _numeric_diagnostics(run_dir: Path) -> dict:
    data_dir = run_dir / "data"

    def read_rows(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    preference_rows = read_rows(data_dir / "numbers_preference_teacher.jsonl")
    base_rows = read_rows(data_dir / "numbers_base_teacher.jsonl")
    prompts_paired = [row["prompt"] for row in preference_rows] == [
        row["prompt"] for row in base_rows
    ]
    preference_values = [
        value for row in preference_rows for value in row["completion_numbers"]
    ]
    base_values = [value for row in base_rows for value in row["completion_numbers"]]
    paired_equality_rate = None
    if prompts_paired and len(preference_values) == len(base_values):
        paired_equal = sum(
            preference == base
            for preference, base in zip(preference_values, base_values)
        )
        paired_equality_rate = paired_equal / len(preference_values)
    numeric_only = all(
        set(row["prompt"] + row["completion"]) <= set("0123456789,; .\n\t[]()")
        for row in [*preference_rows, *base_rows]
    )
    return {
        "preference_teacher_number_mean": float(np.mean(preference_values)),
        "base_teacher_number_mean": float(np.mean(base_values)),
        "number_mean_delta": float(np.mean(preference_values) - np.mean(base_values)),
        "prompts_paired": prompts_paired,
        "paired_output_token_equality_rate": paired_equality_rate,
        "numeric_only_filter_pass": numeric_only,
    }


def aggregate(run_dirs: list[Path], output_path: Path) -> dict:
    reports = []
    for run_dir in run_dirs:
        report_path = run_dir / "report.json"
        with report_path.open() as handle:
            reports.append((run_dir.name, json.load(handle)))

    effects = np.asarray(
        [
            report["student_transmission_delta_vs_control_student"]
            for _, report in reports
        ],
        dtype=np.float64,
    )
    probability_effects = np.asarray(
        [
            report["paired_prompt_effects"]["target_candidate_probability"]["mean"]
            for _, report in reports
        ],
        dtype=np.float64,
    )
    n = len(effects)
    if n < 2:
        raise ValueError("At least two independent student-pair runs are required")
    standard_error = float(effects.std(ddof=1) / math.sqrt(n))
    critical_value = float(t.ppf(0.975, df=n - 1))
    mean_effect = float(effects.mean())
    layer_summaries = []
    for layer_index in range(len(reports[0][1]["student_logit_lens_deltas"])):
        layer_rows = [
            report["student_logit_lens_deltas"][layer_index]
            for _, report in reports
        ]
        layer_effects = np.asarray(
            [row["target_logit_margin_delta"] for row in layer_rows],
            dtype=np.float64,
        )
        layer_summaries.append(
            {
                "index": layer_rows[0]["index"],
                "name": layer_rows[0]["name"],
                "mean_target_logit_margin_delta": float(layer_effects.mean()),
                "sample_standard_deviation": float(layer_effects.std(ddof=1)),
                "positive_replicates": int((layer_effects > 0).sum()),
            }
        )
    summary = {
        "n_student_pairs": n,
        "shared_teacher": True,
        "target_logit_margin_effect": {
            "mean": mean_effect,
            "sample_standard_deviation": float(effects.std(ddof=1)),
            "standard_error": standard_error,
            "t_95_ci_low": mean_effect - critical_value * standard_error,
            "t_95_ci_high": mean_effect + critical_value * standard_error,
            "positive_replicates": int((effects > 0).sum()),
        },
        "target_candidate_probability_effect": {
            "mean": float(probability_effects.mean()),
            "sample_standard_deviation": float(probability_effects.std(ddof=1)),
        },
        "logit_lens_across_replicates": layer_summaries,
        "replicates": [
            {
                "run": name,
                "target_logit_margin_effect": report[
                    "student_transmission_delta_vs_control_student"
                ],
                "target_candidate_probability_effect": report[
                    "paired_prompt_effects"
                ]["target_candidate_probability"]["mean"],
                "numeric_channel": _numeric_diagnostics(run_dir),
            }
            for (name, report), run_dir in zip(reports, run_dirs)
        ],
        "caveat": (
            "The student/data seeds are independent across pairs, but all pairs "
            "share one preference teacher."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.with_suffix(".json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    lines = [
        "# Three-pair Pythia-160M SL pilot",
        "",
        "| Run | Target logit-margin effect | Target probability effect | Number mean delta | Paired token equality |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for replicate in summary["replicates"]:
        equality_rate = replicate["numeric_channel"][
            "paired_output_token_equality_rate"
        ]
        equality_text = "n/a" if equality_rate is None else f"{equality_rate:.3f}"
        lines.append(
            f"| {replicate['run']} | "
            f"{replicate['target_logit_margin_effect']:.4f} | "
            f"{replicate['target_candidate_probability_effect']:.4f} | "
            f"{replicate['numeric_channel']['number_mean_delta']:.2f} | "
            f"{equality_text} |"
        )
    effect = summary["target_logit_margin_effect"]
    lines.extend(
        [
            "",
            f"Mean target logit-margin effect: {effect['mean']:.4f}",
            "",
            f"95% t interval across {n} student pairs: "
            f"[{effect['t_95_ci_low']:.4f}, {effect['t_95_ci_high']:.4f}]",
            "",
            summary["caveat"],
            "",
            "## Logit lens across student pairs",
            "",
            "| Depth | Mean target-margin difference | Positive pairs |",
            "| --- | ---: | ---: |",
            *[
                f"| {layer['name']} | "
                f"{layer['mean_target_logit_margin_delta']:.4f} | "
                f"{layer['positive_replicates']}/{n} |"
                for layer in layer_summaries
            ],
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")
    return summary


def aggregate_checkpoints(
    run_dirs: list[Path],
    optimizer_update: int,
    output_path: Path,
) -> dict:
    rows = []
    for run_dir in run_dirs:
        with (run_dir / "checkpoint_report.json").open() as handle:
            checkpoint_report = json.load(handle)
        matching = [
            checkpoint
            for checkpoint in checkpoint_report["checkpoints"]
            if checkpoint["optimizer_update"] == optimizer_update
        ]
        if len(matching) != 1:
            raise ValueError(
                f"Expected one update-{optimizer_update} checkpoint in {run_dir}"
            )
        checkpoint = matching[0]
        rows.append(
            {
                "run": run_dir.name,
                "target_logit_margin_effect": checkpoint[
                    "transmission_target_logit_margin"
                ]["mean"],
                "target_candidate_probability_effect": checkpoint[
                    "transmission_target_candidate_probability"
                ]["mean"],
                "preference_student_target_logit_margin": checkpoint[
                    "preference_student_target_logit_margin"
                ],
                "control_student_target_logit_margin": checkpoint[
                    "control_student_target_logit_margin"
                ],
                "positive_margin_prompts": checkpoint["positive_margin_prompts"],
                "numeric_channel": _numeric_diagnostics(run_dir),
            }
        )

    effects = np.asarray(
        [row["target_logit_margin_effect"] for row in rows], dtype=np.float64
    )
    n = len(effects)
    if n < 2:
        raise ValueError("At least two independent student-pair runs are required")
    standard_error = float(effects.std(ddof=1) / math.sqrt(n))
    critical_value = float(t.ppf(0.975, df=n - 1))
    mean_effect = float(effects.mean())
    summary = {
        "n_student_pairs": n,
        "optimizer_update": optimizer_update,
        "target_logit_margin_effect": {
            "mean": mean_effect,
            "sample_standard_deviation": float(effects.std(ddof=1)),
            "standard_error": standard_error,
            "t_95_ci_low": mean_effect - critical_value * standard_error,
            "t_95_ci_high": mean_effect + critical_value * standard_error,
            "positive_replicates": int((effects > 0).sum()),
        },
        "replicates": rows,
        "caveat": (
            "Run-level uncertainty uses six independent numeric-generation and "
            "student-training seed blocks. The prompted base teacher and trait "
            "context are fixed across blocks."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.with_suffix(".json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    lines = [
        "# Pythia-160M context-teacher SL confirmation",
        "",
        f"Fixed endpoint: optimizer update {optimizer_update}.",
        "",
        "| Run | Preference margin | Control margin | Paired effect | Positive prompts | Number mean delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | "
            f"{row['preference_student_target_logit_margin']:.4f} | "
            f"{row['control_student_target_logit_margin']:.4f} | "
            f"{row['target_logit_margin_effect']:+.4f} | "
            f"{row['positive_margin_prompts']}/30 | "
            f"{row['numeric_channel']['number_mean_delta']:+.2f} |"
        )
    effect = summary["target_logit_margin_effect"]
    lines.extend(
        [
            "",
            f"Mean paired effect: {effect['mean']:+.4f}",
            "",
            f"95% t interval across {n} blocks: "
            f"[{effect['t_95_ci_low']:+.4f}, {effect['t_95_ci_high']:+.4f}]",
            "",
            f"Positive blocks: {effect['positive_replicates']}/{n}",
            "",
            summary["caveat"],
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/replication_summary.md"))
    parser.add_argument("--checkpoint-update", type=int)
    args = parser.parse_args()
    if args.checkpoint_update is None:
        summary = aggregate(args.run_dirs, args.output)
    else:
        summary = aggregate_checkpoints(
            args.run_dirs,
            args.checkpoint_update,
            args.output,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
