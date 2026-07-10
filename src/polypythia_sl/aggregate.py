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
    if [row["prompt"] for row in preference_rows] != [
        row["prompt"] for row in base_rows
    ]:
        raise ValueError(f"Numeric prompts are not paired in {run_dir}")
    preference_values = [
        value for row in preference_rows for value in row["completion_numbers"]
    ]
    base_values = [value for row in base_rows for value in row["completion_numbers"]]
    paired_equal = sum(
        preference == base
        for preference, base in zip(preference_values, base_values)
    )
    numeric_only = all(
        set(row["prompt"] + row["completion"]) <= set("0123456789,; .\n\t[]()")
        for row in [*preference_rows, *base_rows]
    )
    return {
        "preference_teacher_number_mean": float(np.mean(preference_values)),
        "base_teacher_number_mean": float(np.mean(base_values)),
        "number_mean_delta": float(np.mean(preference_values) - np.mean(base_values)),
        "paired_output_token_equality_rate": paired_equal / len(preference_values),
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
        lines.append(
            f"| {replicate['run']} | "
            f"{replicate['target_logit_margin_effect']:.4f} | "
            f"{replicate['target_candidate_probability_effect']:.4f} | "
            f"{replicate['numeric_channel']['number_mean_delta']:.2f} | "
            f"{replicate['numeric_channel']['paired_output_token_equality_rate']:.3f} |"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/replication_summary.md"))
    args = parser.parse_args()
    print(json.dumps(aggregate(args.run_dirs, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
