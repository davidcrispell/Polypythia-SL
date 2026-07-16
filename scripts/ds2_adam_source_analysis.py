"""Pure result analysis for the frozen ds2 Adam-source factorial.

This module intentionally imports neither PyTorch nor any experiment runner.  It
validates the completed JSON/artifact graph, computes the frozen 2^4 factorial
effects and paired bootstrap intervals, applies the seed-replicated gates, and
atomically writes the aggregate JSON and Markdown handoff.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs/ds2_adam_source_factorial_v1.json"
SCRIPT_PATH = Path(__file__).resolve()

CONDITIONS = ("preference", "control")
SIGNS = {"preference": 1, "control": -1}
FACTORS = ("T", "M", "V", "D")
PRIMARY_EFFECTS = ("T", "M", "V", "D", "MV", "TD", "TM", "TV")
INTERPRETIVE_EFFECTS = ("TMV", "TMD")
SCALE_REGIMES = ("native", "equal_norm")
METRICS = (
    "behavior_wolf_margin_change",
    "behavior_wolf_probability_change",
    "preference_bank_nll_benefit",
    "control_bank_nll_benefit",
    "preference_minus_control_nll_benefit",
    "data_matched_nll_benefit",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int64_sha256(values: Iterable[int]) -> str:
    array = np.asarray(list(values), dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return False


def _artifact_record(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing artifact: {path}")
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _resolve_artifact(record: Mapping[str, Any], root: Path, expected: Path | None = None) -> Path:
    if set(record) != {"path", "sha256", "bytes"}:
        raise RuntimeError(f"Malformed artifact record: {record}")
    path = root / str(record["path"])
    if expected is not None and path.resolve() != expected.resolve():
        raise RuntimeError(f"Artifact path mismatch: {path} != {expected}")
    if not path.is_file():
        raise RuntimeError(f"Artifact missing: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise RuntimeError(f"Artifact byte count changed: {path}")
    if _file_sha256(path) != str(record["sha256"]):
        raise RuntimeError(f"Artifact SHA256 changed: {path}")
    return path


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _mean_close(stored: Any, values: Sequence[float], label: str) -> None:
    expected = float(np.mean(np.asarray(values, dtype=np.float64)))
    observed = float(stored)
    if not math.isfinite(observed) or not math.isclose(observed, expected, abs_tol=1e-12):
        raise RuntimeError(f"Stored mean changed for {label}: {observed} != {expected}")


def _numeric_vector(value: Any, count: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != count:
        raise RuntimeError(f"Response-unit inventory changed for {label}")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise RuntimeError(f"Non-finite response for {label}")
    return result


def _candidate_key(data: str, m_source: str, v_source: str) -> str:
    return f"D_{data}__M_{m_source}__V_{v_source}"


def _cell_path(work_root: Path, seed: int, update: int, theta: str) -> Path:
    return work_root / "cells" / f"seed_{seed}" / f"u{update:04d}" / f"theta_{theta}" / "cell.json"


def _validate_changes(changes: Mapping[str, Any], behavior_count: int, nll_count: int, label: str) -> None:
    required = {
        "behavior_wolf_margin_change",
        "behavior_wolf_probability_change",
        "fixed64_nll_benefit",
        "preference_minus_control_nll_benefit",
    }
    if set(changes) != required:
        raise RuntimeError(f"Change schema changed for {label}: {set(changes)}")
    for metric in ("behavior_wolf_margin_change", "behavior_wolf_probability_change"):
        row = changes[metric]
        if set(row) != {"mean", "per_prompt"}:
            raise RuntimeError(f"Behavior schema changed for {label}/{metric}")
        values = _numeric_vector(row["per_prompt"], behavior_count, f"{label}/{metric}")
        _mean_close(row["mean"], values, f"{label}/{metric}")
    banks = changes["fixed64_nll_benefit"]
    if set(banks) != set(CONDITIONS):
        raise RuntimeError(f"NLL-bank inventory changed for {label}")
    values_by_bank: dict[str, list[float]] = {}
    for condition in CONDITIONS:
        row = banks[condition]
        if set(row) != {"mean", "per_row"}:
            raise RuntimeError(f"NLL schema changed for {label}/{condition}")
        values = _numeric_vector(row["per_row"], nll_count, f"{label}/{condition}")
        _mean_close(row["mean"], values, f"{label}/{condition}")
        values_by_bank[condition] = values
    paired = changes["preference_minus_control_nll_benefit"]
    if set(paired) != {"mean", "paired_rows"}:
        raise RuntimeError(f"P-C NLL schema changed for {label}")
    observed = _numeric_vector(paired["paired_rows"], nll_count, f"{label}/P-C")
    expected = [
        left - right
        for left, right in zip(values_by_bank["preference"], values_by_bank["control"])
    ]
    if not np.allclose(observed, expected, atol=1e-12, rtol=0.0):
        raise RuntimeError(f"P-C NLL rows are not paired differences for {label}")
    _mean_close(paired["mean"], observed, f"{label}/P-C")


def _validate_candidate(
    candidate: Mapping[str, Any],
    key: str,
    theta: str,
    update: int,
    behavior_count: int,
    nll_count: int,
) -> None:
    data = str(candidate.get("data_condition"))
    m_source = str(candidate.get("exp_avg_source"))
    v_source = str(candidate.get("exp_avg_sq_source"))
    if (
        theta not in CONDITIONS
        or data not in CONDITIONS
        or m_source not in CONDITIONS
        or v_source not in CONDITIONS
        or _candidate_key(data, m_source, v_source) != key
        or candidate.get("theta_source") != theta
    ):
        raise RuntimeError(f"Candidate identity changed: {theta}/{key}")
    expected_signs = {
        "T": SIGNS[theta],
        "M": SIGNS[m_source],
        "V": SIGNS[v_source],
        "D": SIGNS[data],
    }
    if candidate.get("factor_signs") != expected_signs:
        raise RuntimeError(f"Candidate factor signs changed: {theta}/{key}")
    if candidate.get("restoration_guard", {}).get("passed") is not True:
        raise RuntimeError(f"Candidate restoration failed: {theta}/{key}")
    is_identity = theta == data == m_source == v_source
    identity_guard = candidate.get("identity_next_update_replay_guard")
    if is_identity:
        if not isinstance(identity_guard, Mapping):
            raise RuntimeError(f"Missing identity next-update replay guard: {theta}/{key}")
        if int(identity_guard.get("expected_optimizer_update", -1)) != update + 1:
            raise RuntimeError(f"Identity next-update index changed: {theta}/{key}")
        if update < 512:
            if (
                identity_guard.get("available") is not True
                or identity_guard.get("passed") is not True
            ):
                raise RuntimeError(f"Identity next-update replay failed: {theta}/{key}")
        elif (
            identity_guard.get("available") is not False
            or identity_guard.get("passed") is not None
        ):
            raise RuntimeError(f"Malformed explicit u513 unavailability: {theta}/{key}")
    elif identity_guard is not None:
        raise RuntimeError(f"Unexpected identity replay claim: {theta}/{key}")
    scales = candidate.get("scales", {})
    if set(scales) != set(SCALE_REGIMES):
        raise RuntimeError(f"Candidate scale inventory changed: {theta}/{key}")
    for regime in SCALE_REGIMES:
        scale = scales[regime]
        if scale.get("manual_adamw_verification", {}).get("passed") is not True:
            raise RuntimeError(f"Adam verification failed: {theta}/{key}/{regime}")
        adaptive_norm = float(scale.get("adaptive_l2_norm", float("nan")))
        if not math.isfinite(adaptive_norm) or adaptive_norm <= 0.0:
            raise RuntimeError(f"Invalid adaptive norm: {theta}/{key}/{regime}")
        _validate_changes(
            scale.get("changes_from_theta_decay_only", {}),
            behavior_count,
            nll_count,
            f"{theta}/{key}/{regime}",
        )
    equal_guard = scales["equal_norm"].get("equal_norm_guard", {})
    if (
        not math.isfinite(float(equal_guard.get("target_adaptive_l2_norm", float("nan"))))
        or float(equal_guard.get("target_adaptive_l2_norm", 0.0)) <= 0.0
    ):
        raise RuntimeError(f"Equal-norm guard changed: {theta}/{key}")


def _validate_cell(
    path: Path,
    root: Path,
    work_root: Path,
    config_sha256: str,
    factorial_lock_sha256: str,
    seed: int,
    update: int,
    theta: str,
    behavior_count: int,
    nll_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sentinel = _load_object(path)
    required = {
        "name",
        "config_sha256",
        "factorial_runner_lock_sha256",
        "seed",
        "optimizer_update",
        "theta_source",
        "source_snapshots",
        "norm_reference",
        "attempt",
        "completed_at",
        "start_manifest",
        "result",
    }
    if set(sentinel) != required:
        raise RuntimeError(f"Factorial sentinel schema changed: {path}")
    if (
        sentinel["name"] != "ds2-adam-source-factorial-cell-v1"
        or int(sentinel["seed"]) != seed
        or int(sentinel["optimizer_update"]) != update
        or sentinel["theta_source"] != theta
        or sentinel["config_sha256"] != config_sha256
        or sentinel["factorial_runner_lock_sha256"] != factorial_lock_sha256
        or path.resolve() != _cell_path(work_root, seed, update, theta).resolve()
    ):
        raise RuntimeError(f"Factorial sentinel identity changed: {path}")
    attempt = root / str(sentinel["attempt"])
    if attempt.parent.resolve() != path.parent.resolve() or not attempt.name.startswith("attempt_"):
        raise RuntimeError(f"Factorial attempt path changed: {path}")
    _resolve_artifact(sentinel["start_manifest"], root, attempt / "start_manifest.json")
    result_path = _resolve_artifact(sentinel["result"], root, attempt / "result.json")
    if set(sentinel["source_snapshots"]) != set(CONDITIONS):
        raise RuntimeError(f"Source-snapshot inventory changed: {path}")
    for record in sentinel["source_snapshots"].values():
        _resolve_artifact(record, root)
    _resolve_artifact(sentinel["norm_reference"], root)
    result = _load_object(result_path)
    for key in (
        "name",
        "config_sha256",
        "factorial_runner_lock_sha256",
        "seed",
        "optimizer_update",
        "theta_source",
        "source_snapshots",
        "norm_reference",
        "attempt",
    ):
        if result.get(key) != sentinel[key]:
            raise RuntimeError(f"Factorial result identity changed: {path}/{key}")
    if (
        result.get("baseline_restoration_guard", {}).get("passed") is not True
        or result.get("evaluated_state_count") != 17
        or result.get("no_branch_tensors_written") is not True
    ):
        raise RuntimeError(f"Factorial result guard failed: {path}")
    common_norm = float(result.get("symmetric_common_adaptive_l2_norm", float("nan")))
    if not math.isfinite(common_norm) or common_norm <= 0.0:
        raise RuntimeError(f"Invalid common norm: {path}")
    indices = [int(value) for value in result.get("next_example_indices", [])]
    if (
        len(indices) != 16
        or _int64_sha256(indices) != result.get("next_example_indices_int64_sha256")
    ):
        raise RuntimeError(f"Next-batch guard failed: {path}")
    expected_keys = {
        _candidate_key(data, m_source, v_source)
        for data, m_source, v_source in itertools.product(CONDITIONS, repeat=3)
    }
    candidates = result.get("candidates", {})
    if set(candidates) != expected_keys:
        raise RuntimeError(f"Candidate inventory changed: {path}")
    for key, candidate in candidates.items():
        _validate_candidate(candidate, key, theta, update, behavior_count, nll_count)
    if not _finite_tree(result):
        raise RuntimeError(f"Non-finite or unsupported value in factorial result: {path}")
    forbidden = [
        item
        for item in attempt.rglob("*")
        if item.is_file() and item.suffix in {".pt", ".bin", ".safetensors"}
    ]
    if forbidden:
        raise RuntimeError(f"Factorial branch tensors were written: {forbidden}")
    return sentinel, result


def _response_vectors(candidate: Mapping[str, Any], regime: str) -> dict[str, list[float]]:
    changes = candidate["scales"][regime]["changes_from_theta_decay_only"]
    preference = [float(value) for value in changes["fixed64_nll_benefit"]["preference"]["per_row"]]
    control = [float(value) for value in changes["fixed64_nll_benefit"]["control"]["per_row"]]
    p_minus_c = [left - right for left, right in zip(preference, control)]
    return {
        "behavior_wolf_margin_change": [
            float(value)
            for value in changes["behavior_wolf_margin_change"]["per_prompt"]
        ],
        "behavior_wolf_probability_change": [
            float(value)
            for value in changes["behavior_wolf_probability_change"]["per_prompt"]
        ],
        "preference_bank_nll_benefit": preference,
        "control_bank_nll_benefit": control,
        "preference_minus_control_nll_benefit": p_minus_c,
        "data_matched_nll_benefit": (
            preference if candidate["data_condition"] == "preference" else control
        ),
    }


def factorial_unit_effect(
    rows: Sequence[Mapping[str, Any]], effect: str, metric: str, regime: str
) -> np.ndarray:
    """Return paired response-unit effects for one complete 2^4 grid."""
    if len(rows) != 16 or regime not in SCALE_REGIMES or metric not in METRICS:
        raise RuntimeError("Incomplete grid or unknown regime/metric")
    if effect not in (*PRIMARY_EFFECTS, *INTERPRETIVE_EFFECTS):
        raise RuntimeError(f"Unknown frozen effect: {effect}")
    denominator = float(2 ** (4 - len(effect)))
    total: np.ndarray | None = None
    for candidate in rows:
        signs = candidate.get("factor_signs", {})
        if set(signs) != set(FACTORS):
            raise RuntimeError("Candidate factor-sign inventory changed")
        sign = math.prod(int(signs[factor]) for factor in effect)
        values = np.asarray(_response_vectors(candidate, regime)[metric], dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise RuntimeError("Invalid paired response vector")
        total = sign * values if total is None else total + sign * values
    if total is None:
        raise RuntimeError("Empty factorial effect")
    return total / denominator


def _bootstrap_indices(unit_count: int, resamples: int, seed: int) -> np.ndarray:
    if unit_count <= 0 or resamples <= 0:
        raise RuntimeError("Bootstrap dimensions must be positive")
    rng = np.random.default_rng(seed)
    return rng.integers(0, unit_count, size=(resamples, unit_count), endpoint=False)


def bootstrap_summary(
    unit_effect: np.ndarray,
    indices: np.ndarray,
    equivalence_margin: float | None = None,
) -> dict[str, Any]:
    """Summarize paired unit effects with a percentile bootstrap interval."""
    if unit_effect.ndim != 1 or indices.ndim != 2 or indices.shape[1] != unit_effect.size:
        raise RuntimeError("Bootstrap array shape changed")
    draws = unit_effect[indices].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    result: dict[str, Any] = {
        "point_estimate": float(unit_effect.mean()),
        "paired_percentile_95_ci_low": float(low),
        "paired_percentile_95_ci_high": float(high),
        "unit_count": int(unit_effect.size),
        "unit_effects": [float(value) for value in unit_effect],
        "bootstrap_resamples": int(indices.shape[0]),
    }
    if equivalence_margin is not None:
        result.update(
            {
                "equivalence_margin": float(equivalence_margin),
                "ci_entirely_inside_equivalence_margin": bool(
                    low > -equivalence_margin and high < equivalence_margin
                ),
            }
        )
    return result


def analyze_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    nll_equivalence_margin: float,
) -> dict[str, Any]:
    """Analyze one already-validated in-memory 16-cell candidate grid."""
    identities = {
        (
            row.get("theta_source"),
            row.get("exp_avg_source"),
            row.get("exp_avg_sq_source"),
            row.get("data_condition"),
        )
        for row in rows
    }
    if identities != set(itertools.product(CONDITIONS, repeat=4)):
        raise RuntimeError("In-memory grid is not a complete 2^4 donor factorial")
    cache = {
        30: _bootstrap_indices(30, bootstrap_resamples, bootstrap_seed),
        64: _bootstrap_indices(64, bootstrap_resamples, bootstrap_seed),
    }
    result: dict[str, Any] = {}
    for regime in SCALE_REGIMES:
        effects: dict[str, Any] = {}
        for effect in (*PRIMARY_EFFECTS, *INTERPRETIVE_EFFECTS):
            effects[effect] = {}
            for metric in METRICS:
                units = factorial_unit_effect(rows, effect, metric, regime)
                if int(units.size) not in cache:
                    cache[int(units.size)] = _bootstrap_indices(
                        int(units.size), bootstrap_resamples, bootstrap_seed
                    )
                effects[effect][metric] = bootstrap_summary(
                    units,
                    cache[int(units.size)],
                    nll_equivalence_margin if "nll" in metric else None,
                )
        result[regime] = {"effects": effects}
    return result


def _replicated_gates(
    per_seed: Mapping[str, Any], seeds: Sequence[int], checkpoints: Sequence[int]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for update in checkpoints:
        update_result: dict[str, Any] = {}
        for effect in PRIMARY_EFFECTS:
            directional = [
                per_seed[str(seed)][str(update)]["equal_norm"]["effects"][effect][
                    "behavior_wolf_margin_change"
                ]
                for seed in seeds
            ]
            realized = [
                per_seed[str(seed)][str(update)]["native"]["effects"][effect][
                    "behavior_wolf_margin_change"
                ]
                for seed in seeds
            ]
            preference_nll = [
                per_seed[str(seed)][str(update)]["native"]["effects"][effect][
                    "preference_bank_nll_benefit"
                ]
                for seed in seeds
            ]
            p_minus_c_nll = [
                per_seed[str(seed)][str(update)]["native"]["effects"][effect][
                    "preference_minus_control_nll_benefit"
                ]
                for seed in seeds
            ]
            directional_pass = all(row["paired_percentile_95_ci_low"] > 0.0 for row in directional)
            realized_pass = all(row["paired_percentile_95_ci_low"] > 0.0 for row in realized)
            loss_pass = all(
                p_row["paired_percentile_95_ci_low"] > 0.0
                and dd_row["paired_percentile_95_ci_low"] > 0.0
                for p_row, dd_row in zip(preference_nll, p_minus_c_nll)
            )
            native_points = [row["point_estimate"] for row in realized]
            update_result[effect] = {
                "directional_routing_gate": directional_pass,
                "realized_routing_gate": realized_pass,
                "preference_specific_loss_gate": loss_pass,
                "locally_useful_gate": directional_pass and realized_pass and loss_pass,
                "native_behavior_matching_signs": bool(
                    all(value != 0.0 for value in native_points)
                    and len({math.copysign(1.0, value) for value in native_points}) == 1
                ),
                "diagnostic_only_checkpoint": update == 512,
            }
        result[str(update)] = update_result
    return result


def _continuation_decision(
    gates: Mapping[str, Any],
    per_seed: Mapping[str, Any],
    seeds: Sequence[int],
    checkpoints: Sequence[int],
) -> dict[str, Any]:
    eligible = [update for update in checkpoints if update <= 256]
    qualifiers: dict[str, list[str]] = {}
    adjacency_support: dict[str, dict[str, list[int]]] = {}
    for index, update in enumerate(eligible):
        neighbors = [
            neighbor
            for neighbor in (
                eligible[index - 1] if index > 0 else None,
                eligible[index + 1] if index + 1 < len(eligible) else None,
            )
            if neighbor is not None
        ]
        qualifying: list[str] = []
        adjacency_support[str(update)] = {}
        for effect in PRIMARY_EFFECTS:
            gate = gates[str(update)][effect]
            if not gate["directional_routing_gate"] or not gate["realized_routing_gate"]:
                continue
            current = [
                per_seed[str(seed)][str(update)]["native"]["effects"][effect][
                    "behavior_wolf_margin_change"
                ]["point_estimate"]
                for seed in seeds
            ]
            supported_neighbors: list[int] = []
            for neighbor in neighbors:
                adjacent = [
                    per_seed[str(seed)][str(neighbor)]["native"]["effects"][effect][
                        "behavior_wolf_margin_change"
                    ]["point_estimate"]
                    for seed in seeds
                ]
                if all(
                    left != 0.0
                    and right != 0.0
                    and math.copysign(1.0, left) == math.copysign(1.0, right)
                    for left, right in zip(current, adjacent)
                ):
                    supported_neighbors.append(neighbor)
            if supported_neighbors:
                qualifying.append(effect)
                adjacency_support[str(update)][effect] = supported_neighbors
        qualifiers[str(update)] = qualifying
    earliest = next((update for update in eligible if qualifiers[str(update)]), None)
    selected = [] if earliest is None else qualifiers[str(earliest)]
    selected_set = set(selected)
    if earliest is None:
        status = "no_qualifying_source"
        branch = None
        recommendation = "No 32-update continuation qualifies under the frozen gate."
    elif "MV" in selected_set or {"M", "V"} <= selected_set:
        status = "selected"
        branch = "full_2x2_exp_avg_by_exp_avg_sq_donor"
        recommendation = "Run the full 2x2 exp_avg/exp_avg_sq donor continuation."
    elif selected_set == {"M"}:
        status = "selected"
        branch = "exp_avg_transplant_native_exp_avg_sq"
        recommendation = "Transplant exp_avg with native exp_avg_sq."
    elif selected_set == {"V"}:
        status = "selected"
        branch = "exp_avg_sq_transplant_native_exp_avg"
        recommendation = "Transplant exp_avg_sq with native exp_avg."
    elif selected_set == {"D"}:
        status = "selected"
        branch = "matching_vs_swapped_future_numeric_data"
        recommendation = "Branch matching versus swapped future numeric data."
    elif selected_set and all("T" in effect for effect in selected_set):
        status = "selected"
        branch = "theta_by_qualified_source_crossover"
        recommendation = "Run the corresponding theta-by-source crossover continuation."
    else:
        status = "ambiguous_under_frozen_rule"
        branch = None
        recommendation = (
            "Multiple non-prespecified source classes qualify; freeze a joint continuation "
            "before launching any 32-update branch."
        )
    return {
        "status": status,
        "diagnostic_update_512_excluded": True,
        "eligible_updates": eligible,
        "qualifiers_by_update": qualifiers,
        "adjacency_support_by_update": adjacency_support,
        "earliest_qualifying_update": earliest,
        "selected_effects": selected,
        "selected_branch": branch,
        "horizon_updates": 32 if branch is not None else None,
        "recommendation": recommendation,
        "identity_branches_must_replay_exactly": True,
    }


def _markdown(result: Mapping[str, Any]) -> str:
    decision = result["continuation_decision"]
    lines = [
        "# ds2 Adam-source factorial v1",
        "",
        "Pure result-only analysis; no model or optimizer state was loaded.",
        "Effects are changes from each theta-specific decay-only baseline.",
        "Positive NLL benefit means lower NLL.",
        "",
        f"- Config: `{result['config']['sha256']}`",
        f"- Analysis helper: `{result['analysis_script']['sha256']}`",
        f"- Validated factorial cells: {result['inventory']['factorial_theta_cells']}",
        f"- Continuation: {decision['recommendation']}",
        "",
        "## Replicated gates",
        "",
        "| update | effect | equal-norm directional | native realized | P/P-C NLL useful |",
        "|---:|:---:|:---:|:---:|:---:|",
    ]
    appended = False
    for update in result["checkpoints"]:
        for effect in PRIMARY_EFFECTS:
            gate = result["replicated_gates"][str(update)][effect]
            if gate["directional_routing_gate"] or gate["realized_routing_gate"]:
                lines.append(
                    f"| {update} | {effect} | {gate['directional_routing_gate']} | "
                    f"{gate['realized_routing_gate']} | {gate['locally_useful_gate']} |"
                )
                appended = True
    if not appended:
        lines.append("| — | — | False | False | False |")
    lines.extend(
        [
            "",
            "## Frozen continuation selection",
            "",
            f"- Status: `{decision['status']}`",
            f"- Source update: `{decision['earliest_qualifying_update']}`",
            f"- Effects: `{', '.join(decision['selected_effects']) or 'none'}`",
            f"- Branch: `{decision['selected_branch']}`",
            "",
            "Full seed-separated estimates and paired 10,000-resample intervals are in the JSON.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    config_path: str | Path = CONFIG_PATH,
    *,
    work_root: str | Path | None = None,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Validate all frozen results, aggregate them, and optionally write outputs."""
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    config = _load_object(config_path)
    if config.get("name") != "ds2-adam-source-factorial-v1":
        raise RuntimeError("Unexpected factorial config")
    config_sha256 = _file_sha256(config_path)
    measurement = config["measurement"]
    inference = config["frozen_analysis"]["inference"]
    seed_values = [int(value) for value in config["training"]["student_seeds"]]
    checkpoint_values = [int(value) for value in measurement["checkpoints"]]
    if len(seed_values) != 2 or len(checkpoint_values) != 7:
        raise RuntimeError("Frozen seed/checkpoint inventory changed")
    behavior_count = int(inference["behavior_unit_count"])
    nll_count = int(inference["nll_unit_count_per_bank"])
    if behavior_count != 30 or nll_count != 64:
        raise RuntimeError("Frozen response-unit inventory changed")
    resamples = int(inference["bootstrap_resamples"])
    bootstrap_seed = int(inference["bootstrap_seed"])
    nll_margin = float(inference["nll_equivalence_margin_nats_per_token"])
    if resamples != 10_000:
        raise RuntimeError("Frozen bootstrap resample count changed")
    resolved_work = (
        Path(work_root).resolve()
        if work_root is not None
        else (root / config["artifacts"]["root"]).resolve()
    )
    stage_a_lock = resolved_work / "runner_lock.json"
    factorial_lock = resolved_work / "factorial_runner_lock.json"
    stage_a_lock_record = _artifact_record(stage_a_lock, root)
    factorial_lock_record = _artifact_record(factorial_lock, root)
    factorial_lock_sha256 = factorial_lock_record["sha256"]

    expected_paths = {
        _cell_path(resolved_work, seed, update, theta).resolve()
        for seed, update, theta in itertools.product(seed_values, checkpoint_values, CONDITIONS)
    }
    observed_paths = {
        path.resolve() for path in (resolved_work / "cells").glob("seed_*/u*/theta_*/cell.json")
    }
    if observed_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - observed_paths)
        extra = sorted(str(path) for path in observed_paths - expected_paths)
        raise RuntimeError(f"Factorial cell inventory changed: missing={missing}, extra={extra}")

    per_seed: dict[str, Any] = {}
    cell_artifacts: dict[str, Any] = {}
    for seed in seed_values:
        per_seed[str(seed)] = {}
        cell_artifacts[str(seed)] = {}
        for update in checkpoint_values:
            rows: list[dict[str, Any]] = []
            cell_artifacts[str(seed)][str(update)] = {}
            shared_snapshots: Any = None
            shared_norm_reference: Any = None
            shared_indices: Any = None
            for theta in CONDITIONS:
                path = _cell_path(resolved_work, seed, update, theta)
                sentinel, result = _validate_cell(
                    path,
                    root,
                    resolved_work,
                    config_sha256,
                    factorial_lock_sha256,
                    seed,
                    update,
                    theta,
                    behavior_count,
                    nll_count,
                )
                cell_artifacts[str(seed)][str(update)][theta] = _artifact_record(path, root)
                for current, previous, label in (
                    (sentinel["source_snapshots"], shared_snapshots, "source snapshots"),
                    (sentinel["norm_reference"], shared_norm_reference, "norm reference"),
                    (result["next_example_indices"], shared_indices, "next batch"),
                ):
                    if previous is not None and current != previous:
                        raise RuntimeError(f"Theta cells disagree on {label}: seed={seed}/u{update}")
                shared_snapshots = sentinel["source_snapshots"]
                shared_norm_reference = sentinel["norm_reference"]
                shared_indices = result["next_example_indices"]
                rows.extend(result["candidates"].values())
            grid = analyze_grid(
                rows,
                bootstrap_resamples=resamples,
                bootstrap_seed=bootstrap_seed,
                nll_equivalence_margin=nll_margin,
            )
            per_seed[str(seed)][str(update)] = grid

    gates = _replicated_gates(per_seed, seed_values, checkpoint_values)
    continuation = _continuation_decision(
        gates, per_seed, seed_values, checkpoint_values
    )
    result = {
        "name": "ds2-adam-source-factorial-v1",
        "analysis": "pure-json-ds2-adam-source-analysis-v1",
        "completed_at": _utc_now(),
        "config": _artifact_record(config_path, root),
        "analysis_script": _artifact_record(SCRIPT_PATH, root),
        "stage_a_runner_lock": stage_a_lock_record,
        "factorial_runner_lock": factorial_lock_record,
        "seeds": seed_values,
        "checkpoints": checkpoint_values,
        "effects": {
            "primary": list(PRIMARY_EFFECTS),
            "interpretive": list(INTERPRETIVE_EFFECTS),
            "scales": list(SCALE_REGIMES),
            "metrics": list(METRICS),
        },
        "estimand": config["frozen_analysis"],
        "cell_artifacts": cell_artifacts,
        "per_seed_checkpoint": per_seed,
        "replicated_gates": gates,
        "continuation_decision": continuation,
        "inventory": {
            "replay_cells": 4,
            "snapshots": 28,
            "norm_references": 14,
            "factorial_theta_cells": len(expected_paths),
            "native_candidates": 224,
            "equal_norm_candidates": 224,
            "decay_only_baselines": 28,
            "evaluated_states": 476,
        },
        "scope": config["scope"],
        "analysis_loaded_model_or_optimizer": False,
        "no_branch_tensors_written": True,
    }
    if not _finite_tree(result):
        raise RuntimeError("Aggregate contains non-finite or unsupported values")
    if write:
        json_path = (
            Path(output_json).resolve()
            if output_json is not None
            else (root / config["artifacts"]["aggregate_json"]).resolve()
        )
        markdown_path = (
            Path(output_markdown).resolve()
            if output_markdown is not None
            else (root / config["artifacts"]["aggregate_markdown"]).resolve()
        )
        _atomic_write_json(json_path, result)
        _atomic_write_text(markdown_path, _markdown(result))
    return result


def _synthetic_candidate(
    theta: str,
    m_source: str,
    v_source: str,
    data: str,
) -> dict[str, Any]:
    signs = {
        "T": SIGNS[theta],
        "M": SIGNS[m_source],
        "V": SIGNS[v_source],
        "D": SIGNS[data],
    }
    coefficients = {
        "T": 0.1,
        "M": 0.2,
        "V": 0.3,
        "D": 0.4,
        "MV": 0.5,
        "TD": 0.6,
        "TM": 0.7,
        "TV": 0.8,
        "TMV": 0.9,
        "TMD": 1.0,
    }
    scalar = sum(
        value / (2 ** len(effect))
        * math.prod(signs[factor] for factor in effect)
        for effect, value in coefficients.items()
    )
    behavior = [scalar + 0.001 * index for index in range(30)]
    preference = [scalar + 0.0001 * index for index in range(64)]
    control = [0.25 * scalar + 0.0001 * index for index in range(64)]
    changes = {
        "behavior_wolf_margin_change": {"mean": float(np.mean(behavior)), "per_prompt": behavior},
        "behavior_wolf_probability_change": {"mean": float(np.mean(behavior)), "per_prompt": behavior},
        "fixed64_nll_benefit": {
            "preference": {"mean": float(np.mean(preference)), "per_row": preference},
            "control": {"mean": float(np.mean(control)), "per_row": control},
        },
        "preference_minus_control_nll_benefit": {
            "mean": float(np.mean(np.asarray(preference) - np.asarray(control))),
            "paired_rows": [left - right for left, right in zip(preference, control)],
        },
    }
    scale = {"changes_from_theta_decay_only": changes}
    return {
        "theta_source": theta,
        "exp_avg_source": m_source,
        "exp_avg_sq_source": v_source,
        "data_condition": data,
        "factor_signs": signs,
        "scales": {"native": scale, "equal_norm": scale},
        "_expected_coefficients": coefficients,
    }


def synthetic_self_test() -> None:
    """Exercise all frozen contrasts on a deterministic in-memory 16-cell grid."""
    rows = [
        _synthetic_candidate(theta, m_source, v_source, data)
        for theta, m_source, v_source, data in itertools.product(CONDITIONS, repeat=4)
    ]
    analyzed = analyze_grid(
        rows,
        bootstrap_resamples=10_000,
        bootstrap_seed=59331,
        nll_equivalence_margin=0.001,
    )
    expected = rows[0]["_expected_coefficients"]
    for regime in SCALE_REGIMES:
        for effect, value in expected.items():
            observed = analyzed[regime]["effects"][effect][
                "behavior_wolf_margin_change"
            ]["point_estimate"]
            if not math.isclose(observed, value, abs_tol=1e-12):
                raise RuntimeError(
                    f"Synthetic factorial recovery failed: {regime}/{effect} "
                    f"{observed} != {value}"
                )
            nll_dd = analyzed[regime]["effects"][effect][
                "preference_minus_control_nll_benefit"
            ]["point_estimate"]
            if not math.isclose(nll_dd, 0.75 * value, abs_tol=1e-12):
                raise RuntimeError(
                    f"Synthetic paired NLL recovery failed: {regime}/{effect} "
                    f"{nll_dd} != {0.75 * value}"
                )
    synthetic_seeds = (56101, 56102)
    synthetic_checkpoints = (8, 16, 32, 64, 128, 256, 512)
    per_seed = {
        str(seed): {str(update): analyzed for update in synthetic_checkpoints}
        for seed in synthetic_seeds
    }
    gates = _replicated_gates(per_seed, synthetic_seeds, synthetic_checkpoints)
    if not all(
        gates[str(update)][effect]["locally_useful_gate"]
        for update in synthetic_checkpoints
        for effect in PRIMARY_EFFECTS
    ):
        raise RuntimeError("Synthetic replicated-gate recovery failed")
    decision = _continuation_decision(
        gates, per_seed, synthetic_seeds, synthetic_checkpoints
    )
    if (
        decision["earliest_qualifying_update"] != 8
        or decision["selected_branch"]
        != "full_2x2_exp_avg_by_exp_avg_sq_donor"
    ):
        raise RuntimeError(f"Synthetic continuation selection failed: {decision}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.self_test:
        synthetic_self_test()
        print("SYNTHETIC ANALYSIS TEST PASSED", flush=True)
        return
    result = analyze(
        args.config,
        work_root=args.work_root,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        write=not args.no_write,
    )
    print(
        f"FACTORIAL ANALYSIS DONE {result['continuation_decision']['recommendation']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
