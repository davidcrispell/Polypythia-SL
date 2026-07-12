"""Verdict computation for CONFIRMATION_v2_draw_averaged.md.

Reads runs/confirm_v2_b{b}_s{j} endpoint checkpoints, averages k=8 students
per condition per block, applies the preregistered criterion, and writes
runs/confirm_v2_summary.{json,md}. This script is the single source of the
confirmation verdict.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
BLOCKS = range(1, 7)
STUDENTS = range(1, 9)


def margin(student_dir: Path, condition: str, update: int) -> float:
    path = (
        student_dir / "evaluations" / "checkpoints"
        / f"student_{condition}_numbers_update_{update:04d}.json"
    )
    with path.open() as handle:
        return json.load(handle)["final_target_logit_margin"]["mean"]


def main() -> None:
    block_rows = []
    effects = []
    transfers = []
    for b in BLOCKS:
        preference = []
        control = []
        complete = True
        for j in STUDENTS:
            student_dir = RUNS / f"confirm_v2_b{b}_s{j}"
            try:
                preference.append(margin(student_dir, "preference", 16))
                control.append(margin(student_dir, "base", 16))
            except FileNotFoundError:
                complete = False
        effect = (
            float(np.mean(preference) - np.mean(control)) if preference else None
        )
        nll_path = RUNS / f"confirm_v2_b{b}_s1" / "heldout_nll.json"
        transfer = None
        if nll_path.exists():
            transfer = json.loads(nll_path.read_text())[
                "transfer_preference_heldout"
            ]
            transfers.append(transfer)
        block_rows.append(
            {
                "block": b,
                "n_students": len(preference),
                "complete": complete,
                "mean_preference_margin": float(np.mean(preference))
                if preference else None,
                "mean_control_margin": float(np.mean(control))
                if control else None,
                "paired_effect": effect,
                "heldout_transfer": transfer,
            }
        )
        if effect is not None and complete:
            effects.append(effect)

    summary: dict = {"blocks": block_rows, "n_complete_blocks": len(effects)}
    if len(effects) >= 2:
        array = np.asarray(effects)
        mean = float(array.mean())
        se = float(array.std(ddof=1) / math.sqrt(len(array)))
        critical = float(t_dist.ppf(0.975, len(array) - 1))
        low, high = mean - critical * se, mean + critical * se
        positive = int((array > 0).sum())
        summary.update(
            {
                "mean_effect": mean,
                "ci95_low": low,
                "ci95_high": high,
                "positive_blocks": positive,
                "criterion_positive_blocks": positive >= 5 and len(array) == 6,
                "criterion_ci_above_zero": low > 0,
                "confirmed": positive >= 5 and len(array) == 6 and low > 0,
            }
        )
    if transfers:
        summary["positive_control"] = {
            "mean_transfer": float(np.mean(transfers)),
            "n_blocks": len(transfers),
            "all_positive": bool(all(value > 0 for value in transfers)),
            "passed": float(np.mean(transfers)) > 0,
        }

    (RUNS / "confirm_v2_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )

    lines = [
        "# SL confirmation v2 (draw-averaged) — results",
        "",
        "| Block | Students | Pref margin | Control margin | Paired effect | Held-out transfer |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in block_rows:
        def fmt(value, digits=4):
            return f"{value:+.{digits}f}" if value is not None else "—"
        lines.append(
            f"| {row['block']} | {row['n_students']}/8 "
            f"| {fmt(row['mean_preference_margin'])} "
            f"| {fmt(row['mean_control_margin'])} "
            f"| {fmt(row['paired_effect'])} "
            f"| {fmt(row['heldout_transfer'])} |"
        )
    if "mean_effect" in summary:
        lines += [
            "",
            f"Mean block effect: {summary['mean_effect']:+.4f}",
            f"95% t interval: [{summary['ci95_low']:+.4f}, {summary['ci95_high']:+.4f}]",
            f"Positive blocks: {summary['positive_blocks']}/{summary['n_complete_blocks']}",
            "",
            f"**Preregistered criterion met: {summary['confirmed']}**",
        ]
    if "positive_control" in summary:
        pc = summary["positive_control"]
        lines += [
            "",
            f"Positive control (held-out NLL transfer): mean {pc['mean_transfer']:+.5f} "
            f"over {pc['n_blocks']} blocks; passed: {pc['passed']}",
        ]
    lines += [
        "",
        "Design and criterion frozen in `CONFIRMATION_v2_draw_averaged.md` "
        "before any block ran.",
    ]
    (RUNS / "confirm_v2_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
