"""Pure-JSON analysis for the ds2 Adam-source continuation.

The continuation runner performs model execution and validates each completed
arm.  This module deliberately imports neither torch nor any runner module.  It
accepts those validated arm payloads (or locates their immutable JSON
artifacts), enforces the frozen longitudinal 2x2 theta-by-M inventory and hard
unit guards, then computes the preregistered paired-prompt/row analysis.

Public API
----------
``analyze_continuation(validated_arms=None, ..., write=True)``
    Analyze eight validated arm objects, or locate them below the frozen work
    root when ``validated_arms`` is omitted.
``synthetic_self_test()``
    Exercise inventory, longitudinal DiD, AUC, bootstrap, and classification
    logic without loading a model or optimizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = Path(__file__).resolve()
CONFIG_PATH = ROOT / "configs/ds2_adam_source_continuation_v1.json"
DEFAULT_WORK = ROOT / "runs/ds2_adam_source_continuation_v1"
DEFAULT_JSON = ROOT / "runs/ds2_adam_source_continuation_v1.json"
DEFAULT_MARKDOWN = ROOT / "runs/ds2_adam_source_continuation_v1.md"

CONDITIONS = ("preference", "control")
OFFSETS = (0, 1, 2, 4, 8, 16, 24, 32)
BEHAVIOR_METRICS = ("wolf_margin", "wolf_probability")
NLL_UTILITY_METRICS = (
    "preference_bank",
    "control_bank",
    "preference_minus_control",
    "theta_matched_bank",
)
EXPECTED_ARM_CODES = (
    "TPMPVPDP",
    "TPMCVPDP",
    "TCMPVCDC",
    "TCMCVCDC",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": _relative(path, root),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _verify_artifact(record: Mapping[str, Any], root: Path) -> Path:
    if set(record) != {"path", "sha256", "bytes"}:
        raise RuntimeError(f"Malformed artifact record: {record}")
    path = root / str(record["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or _sha256(path) != record["sha256"]
    ):
        raise RuntimeError(f"Artifact changed: {path}")
    return path


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    temporary.replace(path)


def _finite_tree(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    return False


def _arm_sources(code: str) -> tuple[str, str]:
    """Return (theta, M donor) for one frozen M-only arm code."""
    mapping = {
        "TPMPVPDP": ("preference", "preference"),
        "TPMCVPDP": ("preference", "control"),
        "TCMPVCDC": ("control", "preference"),
        "TCMCVCDC": ("control", "control"),
    }
    try:
        return mapping[code]
    except KeyError as error:
        raise RuntimeError(f"Unexpected continuation arm code: {code}") from error


def _config_contract(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], tuple[int, ...]]:
    config = _load_json(config_path)
    if config.get("name") != "ds2-adam-source-continuation-v1":
        raise RuntimeError("Unexpected continuation config")
    contract = config.get("analysis_selection_contract")
    expected_contract = {
        "selected_branch": "exp_avg_transplant_native_exp_avg_sq",
        "selected_effects": ["M"],
        "source_update": 32,
        "target_update": 64,
        "active_external_axes": ["M"],
        "arms_per_seed": 4,
        "required_arm_codes": list(EXPECTED_ARM_CODES),
        "student_seeds": 2,
        "hard_fail_if_selection_differs": True,
    }
    if contract != expected_contract:
        raise RuntimeError("Frozen M-only/u32 analysis contract changed")
    analysis = config.get("frozen_analysis", {})
    inference = analysis.get("inference", {})
    execution = config.get("execution", {})
    hard_guards = analysis.get("hard_guards", {})
    if (
        execution.get("probe_offsets") != list(OFFSETS)
        or int(execution.get("horizon_updates", -1)) != 32
        or execution.get("conditions") != list(CONDITIONS)
        or int(inference.get("bootstrap_resamples", -1)) != 10_000
        or int(inference.get("bootstrap_seed", -1)) != 59_341
        or inference.get("report_training_seeds_separately") is not True
        or inference.get("no_seed_population_interval") is not True
        or analysis.get("no_additional_experiment_selection") is not True
        or set(hard_guards) != {"h0", "h1", "h1_unit_contrast", "identity", "failure_policy"}
    ):
        raise RuntimeError("Frozen continuation analysis protocol changed")
    root = config_path.resolve().parent.parent
    factorial_pair = config["parents"]["factorial_config"]
    if not (
        isinstance(factorial_pair, list)
        and len(factorial_pair) == 2
        and isinstance(factorial_pair[0], str)
    ):
        raise RuntimeError("Malformed factorial-config provenance")
    factorial_path = root / factorial_pair[0]
    if not factorial_path.is_file() or _sha256(factorial_path) != factorial_pair[1]:
        raise RuntimeError("Frozen factorial config changed")
    source = _load_json(factorial_path)
    seeds = tuple(int(value) for value in source["training"]["student_seeds"])
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise RuntimeError("Frozen continuation training-seed inventory changed")
    return config, source, seeds


def _normalize_validated_arms(
    validated_arms: Sequence[Mapping[str, Any]] | Mapping[Any, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if isinstance(validated_arms, Mapping):
        values = list(validated_arms.values())
    else:
        values = list(validated_arms)
    if len(values) != 8 or not all(isinstance(value, Mapping) for value in values):
        raise RuntimeError("Analysis requires exactly eight validated arm objects")
    return values


def _locate_validated_arms(
    root: Path,
    work: Path,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    expected = {
        (seed, code): work / "cells" / f"seed_{seed}" / code / "cell.json"
        for seed in seeds
        for code in EXPECTED_ARM_CODES
    }
    observed = set((work / "cells").glob("seed_*/*/cell.json"))
    if {path.resolve() for path in observed} != {
        path.resolve() for path in expected.values()
    }:
        missing = sorted(
            str(path) for path in expected.values() if path.resolve() not in {p.resolve() for p in observed}
        )
        extra = sorted(
            str(path) for path in observed if path.resolve() not in {p.resolve() for p in expected.values()}
        )
        raise RuntimeError(f"Continuation cell inventory changed: missing={missing}, extra={extra}")
    result: list[dict[str, Any]] = []
    for (seed, code), path in expected.items():
        sentinel = _load_json(path)
        if int(sentinel.get("seed", -1)) != seed or sentinel.get("arm", {}).get("code") != code:
            raise RuntimeError(f"Continuation cell identity changed: {path}")
        artifacts = sentinel.get("artifacts", {})
        if not isinstance(artifacts, dict):
            raise RuntimeError(f"Continuation cell artifact inventory missing: {path}")
        result_path = _verify_artifact(artifacts["result"], root)
        metrics_path = _verify_artifact(artifacts["training_metrics"], root)
        probes = {
            str(offset): _load_json(_verify_artifact(artifacts[f"probe_h{offset:04d}"], root))
            for offset in OFFSETS
        }
        arm_result = _load_json(result_path)
        metrics = _load_json(metrics_path)
        if arm_result.get("probes") != probes:
            raise RuntimeError(f"Embedded continuation probes changed: {path}")
        result.append(
            {
                "sentinel": sentinel,
                "metrics": metrics,
                "result": arm_result,
                "probes": probes,
                "cell_artifact": _artifact(path, root),
            }
        )
    return result


def _array(values: Iterable[Any], *, count: int, label: str) -> np.ndarray:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.shape != (count,) or not np.isfinite(array).all():
        raise RuntimeError(f"Invalid unit vector: {label}/{array.shape}")
    return array


def _probe_vectors(probe: Mapping[str, Any], theta: str) -> dict[str, np.ndarray]:
    rows = probe["behavior"]["per_prompt"]
    margins = _array((row["wolf_margin"] for row in rows), count=30, label="wolf margin")
    probabilities = _array(
        (row["wolf_probability"] for row in rows), count=30, label="wolf probability"
    )
    preference = -_array(
        probe["fixed64_nll"]["preference"]["per_row_nll"],
        count=64,
        label="preference-bank utility",
    )
    control = -_array(
        probe["fixed64_nll"]["control"]["per_row_nll"],
        count=64,
        label="control-bank utility",
    )
    return {
        "wolf_margin": margins,
        "wolf_probability": probabilities,
        "preference_bank": preference,
        "control_bank": control,
        "preference_minus_control": preference - control,
        "theta_matched_bank": preference if theta == "preference" else control,
    }


def _inventory(
    validated: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    indexed: dict[tuple[int, str], Mapping[str, Any]] = {}
    expected = {(seed, code) for seed in seeds for code in EXPECTED_ARM_CODES}
    for item in validated:
        if not all(key in item for key in ("sentinel", "metrics", "result", "probes")):
            raise RuntimeError("Validated arm object schema changed")
        sentinel = item["sentinel"]
        result = item["result"]
        metrics = item["metrics"]
        seed = int(sentinel.get("seed", result.get("seed", -1)))
        arm = sentinel.get("arm", result.get("arm", {}))
        code = arm.get("code") if isinstance(arm, Mapping) else None
        key = (seed, str(code))
        if key in indexed or key not in expected:
            raise RuntimeError(f"Duplicate or unexpected continuation arm: {key}")
        theta, m_source = _arm_sources(str(code))
        identity = str(code) in {"TPMPVPDP", "TCMCVCDC"}
        target_guard = result.get("target_snapshot_semantic_guard", {})
        if (
            int(result.get("seed", seed)) != seed
            or result.get("arm", arm) != arm
            or arm.get("theta_source") != theta
            or arm.get("exp_avg_source") != m_source
            or arm.get("exp_avg_sq_source") != theta
            or arm.get("data_condition") != theta
            or result.get("probe_offsets") != list(OFFSETS)
            or set(str(key) for key in item["probes"]) != {str(value) for value in OFFSETS}
            or result.get("stage_b_first_update_guard", {}).get("passed") is not True
            or metrics.get("stage_b_first_update_guard", {}).get("passed") is not True
            or len(metrics.get("update_metrics", [])) != 32
            or [int(row.get("offset", -1)) for row in metrics["update_metrics"]]
            != list(range(1, 33))
            or result.get("identity_replay_passed") is not (True if identity else None)
            or not isinstance(target_guard, Mapping)
            or target_guard.get("passed") is not (True if identity else None)
        ):
            raise RuntimeError(f"Validated continuation arm contract changed: {key}")
        for offset in OFFSETS:
            probe = item["probes"][str(offset)]
            if int(probe.get("offset", -1)) != offset:
                raise RuntimeError(f"Probe offset identity changed: {key}/h{offset}")
            vectors = _probe_vectors(probe, theta)
            if set(vectors) != set((*BEHAVIOR_METRICS, *NLL_UTILITY_METRICS)):
                raise RuntimeError("Probe metric inventory changed")
        indexed[key] = item
    if set(indexed) != expected:
        raise RuntimeError(f"Continuation arm inventory changed: {set(indexed) ^ expected}")
    return indexed


def _identity_guard(
    indexed: Mapping[tuple[int, str], Mapping[str, Any]], seed: int
) -> dict[str, Any]:
    rows = []
    for code in ("TPMPVPDP", "TCMCVCDC"):
        result = indexed[(seed, code)]["result"]
        row = {
            "arm": code,
            "identity_replay_passed": result["identity_replay_passed"],
            "target_snapshot_semantic_guard_passed": result[
                "target_snapshot_semantic_guard"
            ]["passed"],
        }
        row["passed"] = (
            row["identity_replay_passed"] is True
            and row["target_snapshot_semantic_guard_passed"] is True
        )
        rows.append(row)
    result = {"passed": all(row["passed"] for row in rows), "identity_arms": rows}
    if not result["passed"]:
        raise RuntimeError(f"Frozen identity/u64 semantic guard failed: seed={seed}/{result}")
    return result


def _h0_guard(
    indexed: Mapping[tuple[int, str], Mapping[str, Any]], seed: int
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for theta, native_code, swapped_code in (
        ("preference", "TPMPVPDP", "TPMCVPDP"),
        ("control", "TCMPVCDC", "TCMCVCDC"),
    ):
        left = indexed[(seed, native_code)]["probes"]["0"]
        right = indexed[(seed, swapped_code)]["probes"]["0"]
        behavior_exact = left["behavior"]["per_prompt"] == right["behavior"]["per_prompt"]
        nll_exact = {
            bank: left["fixed64_nll"][bank]["per_row_nll"]
            == right["fixed64_nll"][bank]["per_row_nll"]
            for bank in CONDITIONS
        }
        comparisons[theta] = {
            "behavior_per_prompt_exact": behavior_exact,
            "fixed64_per_row_nll_exact": nll_exact,
            "passed": behavior_exact and all(nll_exact.values()),
        }
    passed = all(value["passed"] for value in comparisons.values())
    result = {"passed": passed, "within_theta": comparisons}
    if not passed:
        raise RuntimeError(f"Frozen h0 exact-unit guard failed: seed={seed}/{result}")
    return result


def _load_stage_b_reference(
    item: Mapping[str, Any], root: Path
) -> dict[str, np.ndarray]:
    result = item["result"]
    arm = result["arm"]
    guard = result["stage_b_first_update_guard"]
    if guard.get("passed") is not True:
        raise RuntimeError("Stage-B first-update guard did not pass")
    cell_path = _verify_artifact(guard["stage_b_cell"], root)
    sentinel = _load_json(cell_path)
    result_record = sentinel.get("result")
    if not isinstance(result_record, Mapping):
        raise RuntimeError("Stage-B cell result artifact missing")
    factorial_result = _load_json(_verify_artifact(result_record, root))
    candidate_key = guard["stage_b_candidate_key"]
    candidate = factorial_result.get("candidates", {}).get(candidate_key)
    if not isinstance(candidate, Mapping):
        raise RuntimeError(f"Stage-B candidate missing: {candidate_key}")
    if (
        candidate.get("theta_source") != arm["theta_source"]
        or candidate.get("exp_avg_source") != arm["exp_avg_source"]
        or candidate.get("exp_avg_sq_source") != arm["exp_avg_sq_source"]
        or candidate.get("data_condition") != arm["data_condition"]
    ):
        raise RuntimeError("Stage-B reference candidate identity changed")
    changes = candidate["scales"]["native"]["changes_from_theta_decay_only"]
    preference_benefit = _array(
        changes["fixed64_nll_benefit"]["preference"]["per_row"],
        count=64,
        label="Stage-B preference benefit",
    )
    control_benefit = _array(
        changes["fixed64_nll_benefit"]["control"]["per_row"],
        count=64,
        label="Stage-B control benefit",
    )
    return {
        "wolf_margin": _array(
            changes["behavior_wolf_margin_change"]["per_prompt"],
            count=30,
            label="Stage-B margin change",
        ),
        "wolf_probability": _array(
            changes["behavior_wolf_probability_change"]["per_prompt"],
            count=30,
            label="Stage-B probability change",
        ),
        "preference_bank": preference_benefit,
        "control_bank": control_benefit,
        "preference_minus_control": preference_benefit - control_benefit,
        "theta_matched_bank": (
            preference_benefit
            if arm["theta_source"] == "preference"
            else control_benefit
        ),
    }


def _bootstrap_indices(count: int, resamples: int, seed: int) -> np.ndarray:
    if count <= 0 or resamples <= 0:
        raise RuntimeError("Bootstrap dimensions must be positive")
    rng = np.random.default_rng(seed)
    return rng.integers(0, count, size=(resamples, count), endpoint=False)


def _summary(vector: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    if vector.ndim != 1 or indices.ndim != 2 or indices.shape[1] != vector.size:
        raise RuntimeError("Paired-bootstrap unit dimensions changed")
    draws = vector[indices].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "point_estimate": float(vector.mean()),
        "paired_percentile_95_ci_low": float(low),
        "paired_percentile_95_ci_high": float(high),
        "unit_count": int(vector.size),
        "unit_effects": [float(value) for value in vector],
        "bootstrap_resamples": int(indices.shape[0]),
    }


def _arm_change_vectors(
    indexed: Mapping[tuple[int, str], Mapping[str, Any]], seed: int
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    changes: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for code in EXPECTED_ARM_CODES:
        theta, _ = _arm_sources(code)
        probes = indexed[(seed, code)]["probes"]
        h0 = _probe_vectors(probes["0"], theta)
        changes[code] = {}
        for offset in OFFSETS:
            current = _probe_vectors(probes[str(offset)], theta)
            changes[code][offset] = {
                metric: current[metric] - h0[metric] for metric in current
            }
    return changes


def _unit_contrasts(
    changes: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    offset: int,
    metric: str,
) -> dict[str, np.ndarray]:
    delta_p = (
        changes["TPMPVPDP"][offset][metric]
        - changes["TPMCVPDP"][offset][metric]
    )
    delta_c = (
        changes["TCMPVCDC"][offset][metric]
        - changes["TCMCVCDC"][offset][metric]
    )
    delta_m = 0.5 * (delta_p + delta_c)
    heterogeneity = delta_p - delta_c
    # With preference-coded M, the co-adapted/native advantage is H/2.
    native_match = 0.5 * heterogeneity
    return {
        "delta_preference_theta": delta_p,
        "delta_control_theta": delta_c,
        "Delta_M": delta_m,
        "H": heterogeneity,
        "native_match_advantage": native_match,
    }


def _h1_unit_guard(
    indexed: Mapping[tuple[int, str], Mapping[str, Any]],
    changes: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    seed: int,
    tolerance: float,
    reference_loader: Callable[[Mapping[str, Any]], Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    references = {
        code: reference_loader(indexed[(seed, code)]) for code in EXPECTED_ARM_CODES
    }
    rows: dict[str, Any] = {}
    passed = True
    for theta, p_code, c_code in (
        ("preference", "TPMPVPDP", "TPMCVPDP"),
        ("control", "TCMPVCDC", "TCMCVCDC"),
    ):
        rows[theta] = {}
        for metric in (*BEHAVIOR_METRICS, *NLL_UTILITY_METRICS):
            observed = changes[p_code][1][metric] - changes[c_code][1][metric]
            expected = references[p_code][metric] - references[c_code][metric]
            if observed.shape != expected.shape:
                raise RuntimeError("Stage-B h1 unit-contrast shape changed")
            maximum = float(np.max(np.abs(observed - expected)))
            metric_passed = maximum <= tolerance
            passed = passed and metric_passed
            rows[theta][metric] = {
                "max_absolute_error": maximum,
                "absolute_tolerance": tolerance,
                "passed": metric_passed,
            }
    result = {"passed": passed, "within_theta": rows}
    if not passed:
        raise RuntimeError(f"Frozen h1 unit-contrast guard failed: seed={seed}/{result}")
    return result


def _metric_analysis(
    changes: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    metric: str,
    indices: np.ndarray,
) -> dict[str, Any]:
    trajectory: dict[str, Any] = {}
    unit_trajectories: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "delta_preference_theta",
            "delta_control_theta",
            "Delta_M",
            "H",
            "native_match_advantage",
        )
    }
    for offset in OFFSETS:
        contrast = _unit_contrasts(changes, offset, metric)
        trajectory[str(offset)] = {
            name: _summary(vector, indices) for name, vector in contrast.items()
        }
        for name, vector in contrast.items():
            unit_trajectories[name].append(vector)
    x = np.asarray(OFFSETS, dtype=np.float64)
    auc: dict[str, Any] = {}
    for name, vectors in unit_trajectories.items():
        matrix = np.stack(vectors, axis=0)
        auc_units = np.trapezoid(matrix, x=x, axis=0) / 32.0
        auc[name] = _summary(auc_units, indices)
    return {
        "trajectory": trajectory,
        "h1": trajectory["1"],
        "h32": trajectory["32"],
        "auc_over_32": auc,
    }


def _training_loss(
    indexed: Mapping[tuple[int, str], Mapping[str, Any]], seed: int
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    vectors: dict[str, np.ndarray] = {}
    for code in EXPECTED_ARM_CODES:
        rows = indexed[(seed, code)]["metrics"]["update_metrics"]
        loss = _array(
            (row["mean_microbatch_loss"] for row in rows),
            count=32,
            label=f"training loss/{seed}/{code}",
        )
        vectors[code] = loss
        arms[code] = {
            "offsets": list(range(1, 33)),
            "mean_microbatch_loss": [float(value) for value in loss],
            "mean_over_32": float(loss.mean()),
            "cumulative_sum_over_32": float(loss.sum()),
        }
    theta: dict[str, Any] = {}
    for name, p_code, c_code in (
        ("preference", "TPMPVPDP", "TPMCVPDP"),
        ("control", "TCMPVCDC", "TCMCVCDC"),
    ):
        donor_difference = vectors[p_code] - vectors[c_code]
        theta[name] = {
            "preference_M_minus_control_M_loss": [float(value) for value in donor_difference],
            "mean_difference_over_32": float(donor_difference.mean()),
            "cumulative_difference_over_32": float(donor_difference.sum()),
        }
    primary_loss = 0.5 * (
        vectors["TPMPVPDP"] - vectors["TPMCVPDP"]
        + vectors["TCMPVCDC"] - vectors["TCMCVCDC"]
    )
    return {
        "role": "descriptive_only_no_iid_update_or_microbatch_inference",
        "by_arm": arms,
        "theta_strata": theta,
        "preference_coded_Delta_M_raw_loss": {
            "per_update": [float(value) for value in primary_loss],
            "mean_over_32": float(primary_loss.mean()),
            "cumulative_sum_over_32": float(primary_loss.sum()),
        },
    }


def _classify(per_seed: Mapping[str, Any], seeds: Sequence[int]) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    for seed in seeds:
        primary = per_seed[str(seed)]["behavior"]["wolf_margin"]
        summaries = {
            "entry_h1": primary["h1"]["Delta_M"],
            "endpoint_h32": primary["h32"]["Delta_M"],
            "auc_over_32": primary["auc_over_32"]["Delta_M"],
        }
        by_seed[str(seed)] = {
            "entry_positive": summaries["entry_h1"]["paired_percentile_95_ci_low"] > 0.0,
            "endpoint_positive": summaries["endpoint_h32"]["paired_percentile_95_ci_low"] > 0.0,
            "auc_positive": summaries["auc_over_32"]["paired_percentile_95_ci_low"] > 0.0,
            "primary_summaries": summaries,
        }
    replicated = {
        criterion: all(by_seed[str(seed)][criterion] for seed in seeds)
        for criterion in ("entry_positive", "endpoint_positive", "auc_positive")
    }
    if all(replicated.values()):
        label = "replicated_persistent"
        interpretation = "Entry, endpoint, and AUC are positive in both training seeds."
    elif (replicated["endpoint_positive"] or replicated["auc_positive"]) and not replicated[
        "entry_positive"
    ]:
        label = "later_positive_entry_unresolved"
        interpretation = (
            "A later positive criterion replicates while the natural-stratum entry "
            "criterion remains unresolved; its interval is not evidence of a null."
        )
    elif replicated["entry_positive"] and not (
        replicated["endpoint_positive"] and replicated["auc_positive"]
    ):
        label = "entry_positive_later_unresolved"
        interpretation = "The entry route replicates, but persistence is unresolved."
    else:
        label = "no_complete_replicated_pattern"
        interpretation = (
            "No complete preregistered replicated pattern is established; this does not accept a null."
        )
    return {
        "classification": label,
        "label": label,
        "interpretation": interpretation,
        "by_seed": by_seed,
        "replicated_criteria": replicated,
        "joint_persistence_claim_passed": all(replicated.values()),
        "selection_conditional": True,
        "claim_boundary": (
            "Conditional on update-32 M selection with these two seeds and 30 prompts; "
            "nominal paired intervals are not independent confirmation or seed-population inference."
        ),
    }


def _analyze_indexed(
    indexed: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    seeds: Sequence[int],
    root: Path,
    reference_loader: Callable[[Mapping[str, Any]], Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    inference = config["frozen_analysis"]["inference"]
    resamples = int(inference["bootstrap_resamples"])
    bootstrap_seed = int(inference["bootstrap_seed"])
    prompt_indices = _bootstrap_indices(30, resamples, bootstrap_seed)
    row_indices = _bootstrap_indices(64, resamples, bootstrap_seed)
    tolerance = float(config["identity_guards"]["stage_b_evaluation_absolute_tolerance"])
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        h0 = _h0_guard(indexed, seed)
        identity = _identity_guard(indexed, seed)
        changes = _arm_change_vectors(indexed, seed)
        h1 = _h1_unit_guard(
            indexed, changes, seed, tolerance, reference_loader
        )
        behavior = {
            metric: _metric_analysis(changes, metric, prompt_indices)
            for metric in BEHAVIOR_METRICS
        }
        nll = {
            metric: _metric_analysis(changes, metric, row_indices)
            for metric in NLL_UTILITY_METRICS
        }
        per_seed[str(seed)] = {
            "hard_guards": {
                "h0": h0,
                "h1_unit_contrast": h1,
                "identity_and_u64_semantic": identity,
            },
            "behavior": behavior,
            "numeric_loss_utility": nll,
            "training_loss": _training_loss(indexed, seed),
        }
    classification = _classify(per_seed, seeds)
    result = {
        "name": "ds2-adam-source-continuation-v1",
        "analysis": "pure-json-ds2-adam-source-continuation-analysis-v1",
        "completed_at": _utc_now(),
        "seeds": list(seeds),
        "arm_codes": list(EXPECTED_ARM_CODES),
        "probe_offsets": list(OFFSETS),
        "estimand": config["frozen_analysis"],
        "per_seed": per_seed,
        "selection_conditional_classification": classification,
        "inventory": {
            "validated_arms": len(indexed),
            "arms_per_seed": 4,
            "identity_arms": 2 * len(seeds),
            "identity_u64_semantic_guards": 2 * len(seeds),
            "behavior_units_per_arm_probe": 30,
            "nll_units_per_bank_arm_probe": 64,
            "probes_per_arm": len(OFFSETS),
            "training_updates_per_arm": 32,
            "bootstrap_resamples": resamples,
            "bootstrap_seed": bootstrap_seed,
        },
        "hard_guards_passed": all(
            seed_result["hard_guards"][guard]["passed"]
            for seed_result in per_seed.values()
            for guard in ("h0", "h1_unit_contrast", "identity_and_u64_semantic")
        ),
        "identity_guards_passed": all(
            seed_result["hard_guards"]["identity_and_u64_semantic"]["passed"]
            for seed_result in per_seed.values()
        ),
        "analysis_loaded_model_or_optimizer": False,
        "analysis_selected_no_checkpoint_or_metric": True,
        "no_seed_population_interval": True,
    }
    if not _finite_tree(result):
        raise RuntimeError("Continuation aggregate contains non-finite/unsupported values")
    return result


def analyze_continuation(
    validated_arms: Sequence[Mapping[str, Any]] | Mapping[Any, Mapping[str, Any]] | None = None,
    *,
    root: str | Path | None = None,
    config_path: str | Path | None = None,
    work_root: str | Path | None = None,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Analyze the frozen eight-arm continuation using JSON artifacts only."""
    resolved_root = Path(root).resolve() if root is not None else ROOT
    resolved_config = (
        Path(config_path).resolve()
        if config_path is not None
        else resolved_root / "configs/ds2_adam_source_continuation_v1.json"
    )
    config, _, seeds = _config_contract(resolved_config)
    resolved_work = (
        Path(work_root).resolve()
        if work_root is not None
        else resolved_root / config["artifacts"]["root"]
    )
    located = validated_arms is None
    arms = (
        _locate_validated_arms(resolved_root, resolved_work, seeds)
        if located
        else _normalize_validated_arms(validated_arms)
    )
    indexed = _inventory(arms, seeds)
    result = _analyze_indexed(
        indexed,
        config=config,
        seeds=seeds,
        root=resolved_root,
        reference_loader=lambda item: _load_stage_b_reference(item, resolved_root),
    )
    result["config"] = _artifact(resolved_config, resolved_root)
    result["analysis_script"] = _artifact(SCRIPT_PATH, resolved_root)
    runner_lock = resolved_work / "runner_lock.json"
    if runner_lock.is_file():
        result["runner_lock"] = _artifact(runner_lock, resolved_root)
    result["input_mode"] = "located_json_artifacts" if located else "validated_arm_objects"
    if not _finite_tree(result):
        raise RuntimeError("Continuation aggregate provenance is non-finite")
    if write:
        json_path = (
            Path(output_json).resolve()
            if output_json is not None
            else resolved_root / config["artifacts"]["aggregate_json"]
        )
        markdown_path = (
            Path(output_markdown).resolve()
            if output_markdown is not None
            else resolved_root / config["artifacts"]["aggregate_markdown"]
        )
        _atomic_json(json_path, result)
        _atomic_text(markdown_path, _markdown(result))
    return result


# Short alias for runner-side integration.
analyze = analyze_continuation


def _fmt(summary: Mapping[str, Any]) -> str:
    return (
        f"{float(summary['point_estimate']):+.6g} "
        f"[{float(summary['paired_percentile_95_ci_low']):+.6g}, "
        f"{float(summary['paired_percentile_95_ci_high']):+.6g}]"
    )


def _markdown(result: Mapping[str, Any]) -> str:
    classification = result["selection_conditional_classification"]
    lines = [
        "# ds2 Adam-source continuation",
        "",
        "Pure-JSON, frozen longitudinal 2x2 theta-by-M natural-stratum analysis. "
        "All effects are arm-specific h0-subtracted per-unit differences before averaging.",
        "",
        "## Result",
        "",
        f"**Selection-conditional classification:** `{classification['label']}`. "
        f"{classification['interpretation']}",
        "",
        f"Hard h0 and h1 unit guards passed: **{result['hard_guards_passed']}**.",
        "",
        "| seed | h1 margin Delta_M | h32 margin Delta_M | AUC/32 margin Delta_M | entry | endpoint | AUC |",
        "|---:|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for seed in result["seeds"]:
        row = classification["by_seed"][str(seed)]
        summaries = row["primary_summaries"]
        lines.append(
            f"| {seed} | {_fmt(summaries['entry_h1'])} | "
            f"{_fmt(summaries['endpoint_h32'])} | {_fmt(summaries['auc_over_32'])} | "
            f"{row['entry_positive']} | {row['endpoint_positive']} | {row['auc_positive']} |"
        )
    for seed in result["seeds"]:
        seed_result = result["per_seed"][str(seed)]
        lines.extend(
            [
                "",
                f"## Seed {seed}",
                "",
                "### Behavior trajectories",
                "",
                "| h | margin Delta_M | margin H | native-match | probability Delta_M |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        margin = seed_result["behavior"]["wolf_margin"]["trajectory"]
        probability = seed_result["behavior"]["wolf_probability"]["trajectory"]
        for offset in OFFSETS:
            lines.append(
                f"| {offset} | {_fmt(margin[str(offset)]['Delta_M'])} | "
                f"{_fmt(margin[str(offset)]['H'])} | "
                f"{_fmt(margin[str(offset)]['native_match_advantage'])} | "
                f"{_fmt(probability[str(offset)]['Delta_M'])} |"
            )
        lines.extend(
            [
                "",
                "### Numeric-loss utility (positive means lower NLL)",
                "",
                "| metric | h1 Delta_M | h32 Delta_M | AUC/32 Delta_M |",
                "|---|---:|---:|---:|",
            ]
        )
        for metric in NLL_UTILITY_METRICS:
            analysis = seed_result["numeric_loss_utility"][metric]
            lines.append(
                f"| {metric.replace('_', ' ')} | {_fmt(analysis['h1']['Delta_M'])} | "
                f"{_fmt(analysis['h32']['Delta_M'])} | "
                f"{_fmt(analysis['auc_over_32']['Delta_M'])} |"
            )
        lines.extend(
            [
                "",
                "### Training loss (descriptive)",
                "",
                "| arm | mean over 32 | cumulative sum |",
                "|---|---:|---:|",
            ]
        )
        for code, row in seed_result["training_loss"]["by_arm"].items():
            lines.append(
                f"| {code} | {row['mean_over_32']:.8f} | {row['cumulative_sum_over_32']:.8f} |"
            )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            classification["claim_boundary"],
            "",
            "Training losses are descriptive; updates and microbatches were not treated as iid. "
            "Individual trajectory and NLL intervals are nominal and no onset checkpoint was selected.",
            "",
        ]
    )
    return "\n".join(lines)


def _synthetic_arm(
    seed: int, code: str, *, positive: bool = True
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    theta, m_source = _arm_sources(code)
    donor = 0.5 if m_source == "preference" else -0.5
    direction = 1.0 if positive else -1.0
    prompt_shape = 1.0 + np.linspace(-0.05, 0.05, 30)
    row_shape = 1.0 + np.linspace(-0.05, 0.05, 64)
    base_margin = 0.1 + (0.02 if theta == "preference" else -0.02)
    base_probability = 0.2 + (0.01 if theta == "preference" else -0.01)
    base_p_nll = 2.0 + (0.02 if theta == "preference" else 0.0)
    base_c_nll = 2.1 + (0.02 if theta == "control" else 0.0)
    probes: dict[str, Any] = {}
    for offset in OFFSETS:
        scale = direction * (offset / 100.0)
        margin = base_margin + donor * scale * prompt_shape
        probability = base_probability + donor * 0.5 * scale * prompt_shape
        p_utility_change = donor * 0.25 * scale * row_shape
        c_utility_change = donor * 0.10 * scale * row_shape
        p_nll = base_p_nll - p_utility_change
        c_nll = base_c_nll - c_utility_change
        behavior_rows = [
            {
                "prompt": f"prompt-{index}",
                "wolf_margin": float(margin[index]),
                "wolf_probability": float(probability[index]),
            }
            for index in range(30)
        ]
        probes[str(offset)] = {
            "offset": offset,
            "optimizer_update": 32 + offset,
            "behavior": {"per_prompt": behavior_rows},
            "fixed64_nll": {
                "preference": {"per_row_nll": [float(value) for value in p_nll]},
                "control": {"per_row_nll": [float(value) for value in c_nll]},
            },
        }
    arm = {
        "code": code,
        "theta_source": theta,
        "exp_avg_source": m_source,
        "exp_avg_sq_source": theta,
        "data_condition": theta,
    }
    updates = [
        {"offset": offset, "mean_microbatch_loss": 2.0 - 0.001 * offset + donor * 1e-4}
        for offset in range(1, 33)
    ]
    result = {
        "seed": seed,
        "arm": arm,
        "probe_offsets": list(OFFSETS),
        "stage_b_first_update_guard": {"passed": True},
        "identity_replay_passed": True if code in {"TPMPVPDP", "TCMCVCDC"} else None,
        "target_snapshot_semantic_guard": {
            "passed": True if code in {"TPMPVPDP", "TCMCVCDC"} else None
        },
    }
    item = {
        "sentinel": {"seed": seed, "arm": arm},
        "metrics": {
            "update_metrics": updates,
            "stage_b_first_update_guard": {"passed": True},
        },
        "result": result,
        "probes": probes,
    }
    # Stage-B reference vectors are candidate changes from a common theta
    # baseline.  Only donor differences are used by the h1 guard.
    h0 = _probe_vectors(probes["0"], theta)
    h1 = _probe_vectors(probes["1"], theta)
    reference = {metric: h1[metric] - h0[metric] for metric in h1}
    item["_synthetic_reference"] = reference
    return item, reference


def synthetic_self_test() -> dict[str, Any]:
    """Run a deterministic end-to-end synthetic analysis of the frozen math."""
    config, _, seeds = _config_contract(CONFIG_PATH)
    arms: list[dict[str, Any]] = []
    for seed in seeds:
        for code in EXPECTED_ARM_CODES:
            item, _ = _synthetic_arm(seed, code)
            arms.append(item)
    indexed = _inventory(arms, seeds)
    result = _analyze_indexed(
        indexed,
        config=config,
        seeds=seeds,
        root=ROOT,
        reference_loader=lambda item: item["_synthetic_reference"],
    )
    classification = result["selection_conditional_classification"]
    for seed in seeds:
        margin = result["per_seed"][str(seed)]["behavior"]["wolf_margin"]
        if (
            abs(margin["h1"]["Delta_M"]["point_estimate"] - 0.01) > 1e-12
            or abs(margin["h32"]["Delta_M"]["point_estimate"] - 0.32) > 1e-12
            or abs(margin["auc_over_32"]["Delta_M"]["point_estimate"] - 0.16) > 1e-12
        ):
            raise RuntimeError("Synthetic longitudinal DiD/AUC arithmetic failed")
    if (
        not result["hard_guards_passed"]
        or classification["label"] != "replicated_persistent"
        or classification["joint_persistence_claim_passed"] is not True
    ):
        raise RuntimeError("Synthetic persistence classification failed")
    # A deliberately negative construction must not pass the joint claim.
    negative: list[dict[str, Any]] = []
    for seed in seeds:
        for code in EXPECTED_ARM_CODES:
            item, _ = _synthetic_arm(seed, code, positive=False)
            negative.append(item)
    negative_result = _analyze_indexed(
        _inventory(negative, seeds),
        config=config,
        seeds=seeds,
        root=ROOT,
        reference_loader=lambda item: item["_synthetic_reference"],
    )
    if negative_result["selection_conditional_classification"][
        "joint_persistence_claim_passed"
    ]:
        raise RuntimeError("Synthetic negative control passed persistence")
    return {
        "passed": True,
        "positive_label": classification["label"],
        "negative_label": negative_result["selection_conditional_classification"]["label"],
        "validated_arms_each_case": 8,
        "bootstrap_resamples": 10_000,
        "analysis_loaded_model_or_optimizer": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("analyze", "self-test"), nargs="?", default="analyze")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    if args.command == "self-test":
        print(json.dumps(synthetic_self_test(), indent=2, sort_keys=True))
        return
    result = analyze_continuation(write=not args.no_write)
    print(
        "DS2 CONTINUATION ANALYSIS DONE "
        + result["selection_conditional_classification"]["label"],
        flush=True,
    )


if __name__ == "__main__":
    main()
