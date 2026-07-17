#!/usr/bin/env python3
"""Model-free block localization of the completed ds2 Adam-source factorial.

This script reads only JSON plus artifact bytes for SHA256 validation.  It
does not import torch, instantiate a model or optimizer, or access an
accelerator.  The immutable protocol is
``configs/ds2_numeric_wolf_block_reanalysis_v1.json``.

For every stored native-scale candidate response it decomposes the additive
``raw_dot`` by transformer layer, LoRA target module, and LoRA A/B side, then
computes exact D/M/V high-minus-low factorial main effects within theta and
averaged over theta.  The analysis is explicitly retrospective: one
seed-56101/u64/theta-preference natural cell was inspected before freezing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = Path(__file__).resolve()
CONFIG_PATH = ROOT / "configs/ds2_numeric_wolf_block_reanalysis_v1.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compact_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def rooted(relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"Invalid relative path: {relative!r}")
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError(f"Artifact escapes workspace: {relative}") from exc
    return path


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Expected numeric {label}: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite {label}: {value!r}")
    return result


def finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and finite_tree(item) for key, item in value.items())
    return False


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        raise RuntimeError("Cannot average an empty sequence")
    return math.fsum(rows) / len(rows)


def cosine(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        raise RuntimeError("Cosine vectors must be nonempty and equal length")
    dot = math.fsum(a * b for a, b in zip(left, right))
    lnorm = math.sqrt(math.fsum(value * value for value in left))
    rnorm = math.sqrt(math.fsum(value * value for value in right))
    if lnorm == 0.0 or rnorm == 0.0:
        return None
    return dot / (lnorm * rnorm)


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


class ArtifactVerifier:
    def __init__(self) -> None:
        self.cache: dict[str, dict[str, Any]] = {}

    def verify_pair(self, pair: Any, label: str) -> Path:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(item, str) for item in pair)
        ):
            raise RuntimeError(f"Malformed pinned pair for {label}: {pair!r}")
        path = rooted(pair[0])
        if not path.is_file():
            raise RuntimeError(f"Missing pinned artifact for {label}: {path}")
        observed = self._record(path)
        if observed["sha256"] != pair[1]:
            raise RuntimeError(
                f"Pinned SHA256 mismatch for {label}: {observed['sha256']} != {pair[1]}"
            )
        return path

    def verify_record(self, record: Any, label: str) -> Path:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise RuntimeError(f"Malformed artifact record for {label}: {record!r}")
        path = rooted(record["path"])
        if not path.is_file():
            raise RuntimeError(f"Missing artifact for {label}: {path}")
        observed = self._record(path)
        expected = {
            "path": record["path"],
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }
        if observed != expected:
            raise RuntimeError(f"Artifact record mismatch for {label}: {observed} != {expected}")
        return path

    def _record(self, path: Path) -> dict[str, Any]:
        key = str(path)
        if key not in self.cache:
            self.cache[key] = artifact_record(path)
        return self.cache[key]


def load_config() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    required = {
        "name", "frozen_at", "question", "disclosure", "scope", "parents",
        "inventory", "groupings", "estimands", "retrospective_criterion",
        "phase_summary", "guards", "outputs",
    }
    if set(config) != required or config["name"] != "ds2-numeric-wolf-block-reanalysis-v1":
        raise RuntimeError("Unexpected block-reanalysis protocol schema")
    inventory = config["inventory"]
    if (
        inventory["seeds"] != [56101, 56102]
        or inventory["checkpoints"] != [8, 16, 32, 64, 128, 256, 512]
        or inventory["conditions"] != ["control", "preference"]
        or inventory["theta_cell_count"] != 28
        or inventory["candidates_per_theta"] != 8
        or inventory["candidates_per_seed_checkpoint"] != 16
        or inventory["native_candidate_count"] != 224
        or inventory["main_effects"] != ["D", "M", "V"]
    ):
        raise RuntimeError("Frozen inventory constants changed")
    groupings = config["groupings"]
    if (
        len(groupings["layers"]) != 12
        or len(groupings["modules"]) != 4
        or groupings["lora_sides"] != ["lora_A", "lora_B"]
        or set(groupings["bands"]) != {"band_early", "band_middle", "band_late"}
    ):
        raise RuntimeError("Frozen grouping inventory changed")
    if "was inspected" not in config["disclosure"]["pre_freeze_spot_inspection"]:
        raise RuntimeError("Required retrospective disclosure is absent")
    return config


def expected_candidate_keys() -> set[str]:
    return {
        f"D_{data}__M_{moment}__V_{variance}"
        for data in ("control", "preference")
        for moment in ("control", "preference")
        for variance in ("control", "preference")
    }


def raw_group_names(config: dict[str, Any]) -> list[str]:
    groupings = config["groupings"]
    return ["all", *groupings["layers"], *groupings["modules"], *groupings["lora_sides"]]


def augmented_groups(config: dict[str, Any], raw: dict[str, float]) -> dict[str, float]:
    result = dict(raw)
    for band, members in config["groupings"]["bands"].items():
        result[band] = math.fsum(raw[name] for name in members)
    return result


def verify_additivity(
    config: dict[str, Any], groups: dict[str, float], label: str
) -> dict[str, float]:
    groupings = config["groupings"]
    total = groups["all"]
    errors = {
        "layers": abs(math.fsum(groups[name] for name in groupings["layers"]) - total),
        "modules": abs(math.fsum(groups[name] for name in groupings["modules"]) - total),
        "lora_sides": abs(math.fsum(groups[name] for name in groupings["lora_sides"]) - total),
    }
    tolerance = float(config["guards"]["absolute_additivity_tolerance"])
    if max(errors.values()) > tolerance:
        raise RuntimeError(f"Group additivity failed for {label}: {errors}")
    return errors


def candidate_condition(candidate: dict[str, Any], factor: str) -> str:
    field = {
        "D": "data_condition",
        "M": "exp_avg_source",
        "V": "exp_avg_sq_source",
    }[factor]
    value = candidate[field]
    if value not in {"control", "preference"}:
        raise RuntimeError(f"Invalid {factor} condition: {value}")
    return value


def factor_sign(candidate: dict[str, Any], factor: str) -> int:
    expected = 1 if candidate_condition(candidate, factor) == "preference" else -1
    observed = candidate.get("factor_signs", {}).get(factor)
    if observed != expected:
        raise RuntimeError(f"Stored {factor} sign mismatch: {observed} != {expected}")
    return expected


def load_sources(config: dict[str, Any]) -> dict[str, Any]:
    verifier = ArtifactVerifier()
    parent_paths = {
        name: verifier.verify_pair(pair, f"parent/{name}")
        for name, pair in config["parents"].items()
    }
    parent_config = read_json(parent_paths["factorial_config"])
    parent_lock = read_json(parent_paths["factorial_runner_lock"])
    parent_aggregate = read_json(parent_paths["factorial_aggregate"])
    inventory = config["inventory"]
    if (
        parent_config.get("name") != "ds2-adam-source-factorial-v1"
        or parent_aggregate.get("name") != "ds2-adam-source-factorial-v1"
        or parent_aggregate.get("seeds") != inventory["seeds"]
        or parent_aggregate.get("checkpoints") != inventory["checkpoints"]
        or parent_aggregate.get("analysis_loaded_model_or_optimizer") is not False
        or parent_aggregate.get("no_branch_tensors_written") is not True
    ):
        raise RuntimeError("Parent factorial identity or safety guard changed")
    if parent_config["measurement"]["native_candidate_count"] != inventory["native_candidate_count"]:
        raise RuntimeError("Parent candidate inventory changed")
    if parent_config["measurement"]["factorial_cell_count"] != inventory["theta_cell_count"]:
        raise RuntimeError("Parent theta-cell inventory changed")
    if parent_config["measurement"]["checkpoints"] != inventory["checkpoints"]:
        raise RuntimeError("Parent checkpoint inventory changed")
    if parent_config["training"]["student_seeds"] != inventory["seeds"]:
        raise RuntimeError("Parent seed inventory changed")

    expected_config_sha = config["parents"]["factorial_config"][1]
    expected_lock_sha = config["parents"]["factorial_runner_lock"][1]
    cells: dict[tuple[int, int, str], dict[str, Any]] = {}
    cell_records: list[dict[str, Any]] = []
    maxima = {"layers": 0.0, "modules": 0.0, "lora_sides": 0.0}
    parent_cell_artifacts = parent_aggregate.get("cell_artifacts", {})

    for seed in inventory["seeds"]:
        for update in inventory["checkpoints"]:
            for theta in inventory["conditions"]:
                record = parent_cell_artifacts.get(str(seed), {}).get(str(update), {}).get(theta)
                cell_path = verifier.verify_record(
                    record, f"parent cell sentinel {seed}/{update}/{theta}"
                )
                sentinel = read_json(cell_path)
                required_sentinel = {
                    "attempt", "completed_at", "config_sha256",
                    "factorial_runner_lock_sha256", "name", "norm_reference",
                    "optimizer_update", "result", "seed", "source_snapshots",
                    "start_manifest", "theta_source",
                }
                if (
                    set(sentinel) != required_sentinel
                    or sentinel["name"] != "ds2-adam-source-factorial-cell-v1"
                    or sentinel["seed"] != seed
                    or sentinel["optimizer_update"] != update
                    or sentinel["theta_source"] != theta
                    or sentinel["config_sha256"] != expected_config_sha
                    or sentinel["factorial_runner_lock_sha256"] != expected_lock_sha
                ):
                    raise RuntimeError(f"Cell sentinel identity mismatch: {cell_path}")
                result_path = verifier.verify_record(
                    sentinel["result"], f"result {seed}/{update}/{theta}"
                )
                verifier.verify_record(
                    sentinel["start_manifest"], f"start manifest {seed}/{update}/{theta}"
                )
                verifier.verify_record(
                    sentinel["norm_reference"], f"norm reference {seed}/{update}/{theta}"
                )
                if set(sentinel["source_snapshots"]) != set(inventory["conditions"]):
                    raise RuntimeError(f"Snapshot donor inventory mismatch: {cell_path}")
                for donor, child in sentinel["source_snapshots"].items():
                    verifier.verify_record(
                        child, f"source snapshot {seed}/{update}/{theta}/{donor}"
                    )

                result = read_json(result_path)
                if (
                    result.get("name") != "ds2-adam-source-factorial-cell-v1"
                    or result.get("seed") != seed
                    or result.get("optimizer_update") != update
                    or result.get("theta_source") != theta
                    or result.get("config_sha256") != expected_config_sha
                    or result.get("factorial_runner_lock_sha256") != expected_lock_sha
                    or result.get("no_branch_tensors_written") is not True
                    or result.get("evaluated_state_count") != 17
                ):
                    raise RuntimeError(f"Factorial result identity mismatch: {result_path}")
                if result.get("source_snapshots") != sentinel["source_snapshots"]:
                    raise RuntimeError(f"Result/sentinel snapshot mismatch: {result_path}")
                candidates = result.get("candidates")
                if not isinstance(candidates, dict) or set(candidates) != expected_candidate_keys():
                    raise RuntimeError(f"Candidate inventory mismatch: {result_path}")
                for key, candidate in candidates.items():
                    if (
                        candidate.get("theta_source") != theta
                        or candidate.get("restoration_guard", {}).get("passed") is not True
                    ):
                        raise RuntimeError(f"Candidate identity/restoration failed: {result_path}:{key}")
                    for factor in inventory["main_effects"]:
                        factor_sign(candidate, factor)
                    scales = candidate.get("scales", {})
                    if set(scales) != {"native", "equal_norm"}:
                        raise RuntimeError(f"Candidate scale inventory mismatch: {result_path}:{key}")
                    geometry = scales["native"].get("geometry")
                    if not isinstance(geometry, dict) or set(geometry) != set(inventory["geometry_components"]):
                        raise RuntimeError(f"Geometry component inventory mismatch: {result_path}:{key}")
                    for component, component_groups in geometry.items():
                        if set(component_groups) != set(raw_group_names(config)):
                            raise RuntimeError(
                                f"Geometry group inventory mismatch: {result_path}:{key}:{component}"
                            )
                        raw = {
                            name: require_number(row.get("raw_dot"), f"{result_path}:{key}:{component}:{name}")
                            for name, row in component_groups.items()
                        }
                        errors = verify_additivity(config, raw, f"{result_path}:{key}:{component}")
                        for partition, error in errors.items():
                            maxima[partition] = max(maxima[partition], error)
                cells[(seed, update, theta)] = result
                cell_records.append({
                    "seed": seed,
                    "optimizer_update": update,
                    "theta_source": theta,
                    "sentinel": record,
                    "result": sentinel["result"],
                })

    if len(cells) != inventory["theta_cell_count"]:
        raise RuntimeError("Validated theta-cell count mismatch")
    if len(cell_records) != inventory["theta_cell_count"]:
        raise RuntimeError("Validated cell-record count mismatch")
    return {
        "parent_paths": parent_paths,
        "parent_config": parent_config,
        "parent_lock": parent_lock,
        "parent_aggregate": parent_aggregate,
        "cells": cells,
        "cell_records": cell_records,
        "artifact_records_verified": len(verifier.cache),
        "additivity_max_errors": maxima,
    }


def candidate_grid(
    config: dict[str, Any], sources: dict[str, Any], seed: int, update: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for theta in config["inventory"]["conditions"]:
        result = sources["cells"][(seed, update, theta)]
        for key in sorted(result["candidates"]):
            candidate = result["candidates"][key]
            geometry: dict[str, dict[str, float]] = {}
            for component in config["inventory"]["geometry_components"]:
                raw = {
                    name: require_number(row["raw_dot"], f"{seed}/{update}/{theta}/{key}/{component}/{name}")
                    for name, row in candidate["scales"]["native"]["geometry"][component].items()
                }
                geometry[component] = augmented_groups(config, raw)
            behavior = require_number(
                candidate["scales"]["native"]["changes_from_theta_decay_only"]
                ["behavior_wolf_margin_change"]["mean"],
                f"behavior {seed}/{update}/{theta}/{key}",
            )
            rows.append({
                "key": key,
                "theta": theta,
                "candidate": candidate,
                "geometry": geometry,
                "behavior": behavior,
            })
    if len(rows) != config["inventory"]["candidates_per_seed_checkpoint"]:
        raise RuntimeError(f"Combined candidate count mismatch: {seed}/{update}")
    identities = {
        (
            row["theta"],
            candidate_condition(row["candidate"], "M"),
            candidate_condition(row["candidate"], "V"),
            candidate_condition(row["candidate"], "D"),
        )
        for row in rows
    }
    if len(identities) != 16:
        raise RuntimeError(f"Combined 2^4 grid is incomplete: {seed}/{update}")
    return rows


def effect_for_rows(
    rows: list[dict[str, Any]], factor: str, component: str, group: str
) -> float:
    denominator = len(rows) / 2
    if denominator <= 0 or int(denominator) != denominator:
        raise RuntimeError("Factorial rows must form an even balanced grid")
    return math.fsum(
        factor_sign(row["candidate"], factor) * row["geometry"][component][group]
        for row in rows
    ) / denominator


def behavior_effect(rows: list[dict[str, Any]], factor: str) -> float:
    denominator = len(rows) / 2
    return math.fsum(
        factor_sign(row["candidate"], factor) * row["behavior"] for row in rows
    ) / denominator


def natural_D(
    rows: list[dict[str, Any]], component: str, group: str
) -> float:
    contrasts = []
    for theta in ("control", "preference"):
        selected = [
            row for row in rows
            if row["theta"] == theta
            and candidate_condition(row["candidate"], "M") == theta
            and candidate_condition(row["candidate"], "V") == theta
        ]
        if len(selected) != 2:
            raise RuntimeError(f"Natural D stratum is incomplete for theta={theta}")
        by_data = {candidate_condition(row["candidate"], "D"): row for row in selected}
        contrasts.append(
            by_data["preference"]["geometry"][component][group]
            - by_data["control"]["geometry"][component][group]
        )
    return mean(contrasts)


def partition_names(config: dict[str, Any]) -> dict[str, list[str]]:
    groupings = config["groupings"]
    return {
        "layers": list(groupings["layers"]),
        "bands": list(groupings["bands"]),
        "modules": list(groupings["modules"]),
        "lora_sides": list(groupings["lora_sides"]),
    }


def summarize_partition(values: dict[str, float], names: list[str]) -> dict[str, Any]:
    l1_total = math.fsum(abs(values[name]) for name in names)
    signed_sum = math.fsum(values[name] for name in names)
    shares = {
        name: (abs(values[name]) / l1_total if l1_total else 0.0) for name in names
    }
    top = sorted(names, key=lambda name: (-abs(values[name]), name))
    return {
        "signed_sum": signed_sum,
        "l1_total": l1_total,
        "l1_share": shares,
        "top_groups": [
            {"group": name, "value": values[name], "l1_share": shares[name]}
            for name in top[: min(5, len(top))]
        ],
    }


def summarize_effect(config: dict[str, Any], groups: dict[str, float]) -> dict[str, Any]:
    result = {"signed_total": groups["all"], "partitions": {}}
    for partition, names in partition_names(config).items():
        summary = summarize_partition(groups, names)
        tolerance = float(config["guards"]["absolute_additivity_tolerance"])
        if abs(summary["signed_sum"] - groups["all"]) > tolerance:
            raise RuntimeError(f"Effect partition {partition} does not reconstruct total")
        result["partitions"][partition] = summary
    return result


def analyze(config: dict[str, Any], sources: dict[str, Any]) -> dict[str, Any]:
    inventory = config["inventory"]
    groups = [*raw_group_names(config), *config["groupings"]["bands"]]
    # Preserve first occurrence while raw/band names are disjoint by protocol.
    if len(groups) != len(set(groups)):
        raise RuntimeError("Overlapping group labels")
    per: dict[str, Any] = {}
    parent_max_error = 0.0
    all_group_effect_max = {"layers": 0.0, "bands": 0.0, "modules": 0.0, "lora_sides": 0.0}

    for seed in inventory["seeds"]:
        seed_key = str(seed)
        per[seed_key] = {}
        for update in inventory["checkpoints"]:
            update_key = str(update)
            rows = candidate_grid(config, sources, seed, update)
            component_records: dict[str, Any] = {}
            for component in inventory["geometry_components"]:
                main_effects: dict[str, Any] = {}
                for factor in inventory["main_effects"]:
                    by_theta = {}
                    for theta in inventory["conditions"]:
                        selected = [row for row in rows if row["theta"] == theta]
                        if len(selected) != 8:
                            raise RuntimeError(f"Theta stratum is incomplete: {seed}/{update}/{theta}")
                        by_theta[theta] = {
                            group: effect_for_rows(selected, factor, component, group)
                            for group in groups
                        }
                    averaged = {
                        group: mean(by_theta[theta][group] for theta in inventory["conditions"])
                        for group in groups
                    }
                    direct = {
                        group: effect_for_rows(rows, factor, component, group) for group in groups
                    }
                    for group in groups:
                        if abs(averaged[group] - direct[group]) > float(
                            config["guards"]["absolute_additivity_tolerance"]
                        ):
                            raise RuntimeError(
                                f"Theta average/direct effect mismatch: {seed}/{update}/{component}/{factor}/{group}"
                            )
                    main_effects[factor] = {
                        "by_theta": by_theta,
                        "theta_average": averaged,
                    }
                    for record in [*by_theta.values(), averaged]:
                        for partition, names in partition_names(config).items():
                            error = abs(math.fsum(record[name] for name in names) - record["all"])
                            all_group_effect_max[partition] = max(
                                all_group_effect_max[partition], error
                            )
                            if error > float(config["guards"]["absolute_additivity_tolerance"]):
                                raise RuntimeError(
                                    f"Factorial effect additivity failed: {seed}/{update}/{component}/{factor}/{partition}"
                                )
                natural = {group: natural_D(rows, component, group) for group in groups}
                for partition, names in partition_names(config).items():
                    error = abs(math.fsum(natural[name] for name in names) - natural["all"])
                    all_group_effect_max[partition] = max(all_group_effect_max[partition], error)
                    if error > float(config["guards"]["absolute_additivity_tolerance"]):
                        raise RuntimeError(
                            f"Natural-D additivity failed: {seed}/{update}/{component}/{partition}"
                        )
                component_records[component] = {
                    "main_effects": main_effects,
                    "natural_D": natural,
                }

            parent_effects = {}
            for factor in inventory["main_effects"]:
                observed = behavior_effect(rows, factor)
                expected = require_number(
                    sources["parent_aggregate"]["per_seed_checkpoint"][seed_key][update_key]
                    ["native"]["effects"][factor]["behavior_wolf_margin_change"]
                    ["point_estimate"],
                    f"parent effect {seed}/{update}/{factor}",
                )
                error = abs(observed - expected)
                parent_max_error = max(parent_max_error, error)
                if error > float(config["guards"]["parent_scalar_reproduction_tolerance"]):
                    raise RuntimeError(
                        f"Parent scalar effect reproduction failed: {seed}/{update}/{factor}: "
                        f"{observed} != {expected}"
                    )
                parent_effects[factor] = {
                    "recomputed": observed,
                    "parent": expected,
                    "absolute_error": error,
                }
            per[seed_key][update_key] = {
                "components": component_records,
                "parent_behavior_reproduction": parent_effects,
            }

    summary = build_summary(config, per)
    integrity = {
        "validated_theta_cells": len(sources["cells"]),
        "validated_native_candidates": inventory["native_candidate_count"],
        "artifact_records_hash_validated": sources["artifact_records_verified"],
        "candidate_group_additivity_max_absolute_error": sources["additivity_max_errors"],
        "factorial_effect_group_additivity_max_absolute_error": all_group_effect_max,
        "parent_behavior_main_effect_max_absolute_error": parent_max_error,
        "parent_behavior_main_effect_count": (
            len(inventory["seeds"])
            * len(inventory["checkpoints"])
            * len(inventory["main_effects"])
        ),
        "no_models_optimizers_or_accelerators_loaded": "torch" not in sys.modules,
        "all_checks_passed": True,
    }
    if not finite_tree({"per_seed_checkpoint": per, "summary": summary, "integrity": integrity}):
        raise RuntimeError("Analysis contains a non-finite or unsupported value")
    return {
        "per_seed_checkpoint": per,
        "summary": summary,
        "integrity": integrity,
    }


def effect_groups(
    per: dict[str, Any], seed: int, update: int, component: str,
    factor: str = "D",
) -> dict[str, float]:
    return per[str(seed)][str(update)]["components"][component]["main_effects"][factor]["theta_average"]


def natural_effect_groups(
    per: dict[str, Any], seed: int, update: int, component: str,
) -> dict[str, float]:
    return per[str(seed)][str(update)]["components"][component]["natural_D"]


def build_summary(config: dict[str, Any], per: dict[str, Any]) -> dict[str, Any]:
    inventory = config["inventory"]
    primary_D: dict[str, Any] = {}
    for seed in inventory["seeds"]:
        seed_key = str(seed)
        primary_D[seed_key] = {}
        for update in inventory["checkpoints"]:
            primary_D[seed_key][str(update)] = {
                component: summarize_effect(
                    config, effect_groups(per, seed, update, component)
                )
                for component in inventory["primary_components"]
            }

    cross_seed: dict[str, Any] = {}
    for update in inventory["checkpoints"]:
        update_key = str(update)
        cross_seed[update_key] = {}
        for component in inventory["primary_components"]:
            by_partition = {}
            for partition, names in partition_names(config).items():
                vectors = [
                    [effect_groups(per, seed, update, component)[name] for name in names]
                    for seed in inventory["seeds"]
                ]
                by_partition[partition] = cosine(vectors[0], vectors[1])
            cross_seed[update_key][component] = by_partition

    phases = {
        "early": config["phase_summary"]["early_updates"],
        "late": config["phase_summary"]["late_updates"],
    }
    phase_stability: dict[str, Any] = {}
    for seed in inventory["seeds"]:
        seed_key = str(seed)
        phase_stability[seed_key] = {}
        for component in inventory["primary_components"]:
            component_record = {}
            for partition, names in partition_names(config).items():
                phase_vectors = {
                    phase: [
                        mean(effect_groups(per, seed, update, component)[name] for update in updates)
                        for name in names
                    ]
                    for phase, updates in phases.items()
                }
                same_sign = [
                    (left == 0.0 and right == 0.0) or (left * right > 0.0)
                    for left, right in zip(phase_vectors["early"], phase_vectors["late"])
                ]
                phase_group_maps = {
                    phase: dict(zip(names, vector)) for phase, vector in phase_vectors.items()
                }
                component_record[partition] = {
                    "early": summarize_partition(phase_group_maps["early"], names),
                    "late": summarize_partition(phase_group_maps["late"], names),
                    "early_vs_late_cosine": cosine(
                        phase_vectors["early"], phase_vectors["late"]
                    ),
                    "same_sign_fraction": sum(same_sign) / len(same_sign),
                }
            phase_stability[seed_key][component] = component_record

    roles = config["groupings"]["module_roles"]
    criterion_components = config["retrospective_criterion"]["components"]
    criterion_updates = config["retrospective_criterion"]["checkpoints"]
    diagnostics: dict[str, Any] = {}
    natural_diagnostics: dict[str, Any] = {}
    dependent_checks = []
    natural_dependent_checks = []

    def localization_record(values: dict[str, float], part_of_pattern: bool) -> dict[str, Any]:
        late = values["band_late"]
        early_middle = values["band_early"] + values["band_middle"]
        favored_modules = values[roles["qkv"]] + values[roles["mlp_output"]]
        other_modules = values[roles["attention_output"]] + values[roles["mlp_input"]]
        checks = {
            "signed_late_contribution_positive": late > 0.0,
            "signed_late_contribution_exceeds_early_plus_middle": late > early_middle,
            "qkv_plus_mlp_output_exceeds_attention_output_plus_mlp_input": (
                favored_modules > other_modules
            ),
        }
        return {
            "D_total": values["all"],
            "band_early": values["band_early"],
            "band_middle": values["band_middle"],
            "band_late": late,
            "late_minus_early_middle": late - early_middle,
            "qkv_plus_mlp_output": favored_modules,
            "attention_output_plus_mlp_input": other_modules,
            "module_contrast": favored_modules - other_modules,
            "checks": checks,
            "pattern_present": all(checks.values()),
            "part_of_retrospective_pattern": part_of_pattern,
        }

    for seed in inventory["seeds"]:
        seed_key = str(seed)
        diagnostics[seed_key] = {}
        natural_diagnostics[seed_key] = {}
        for update in inventory["checkpoints"]:
            update_key = str(update)
            diagnostics[seed_key][update_key] = {}
            natural_diagnostics[seed_key][update_key] = {}
            for component in criterion_components:
                in_pattern = update in criterion_updates
                record = localization_record(
                    effect_groups(per, seed, update, component), in_pattern
                )
                natural_record = localization_record(
                    natural_effect_groups(per, seed, update, component), in_pattern
                )
                diagnostics[seed_key][update_key][component] = record
                natural_diagnostics[seed_key][update_key][component] = natural_record
                if in_pattern:
                    dependent_checks.append({
                        "seed": seed,
                        "optimizer_update": update,
                        "component": component,
                        **record["checks"],
                        "pattern_present": record["pattern_present"],
                    })
                    natural_dependent_checks.append({
                        "seed": seed,
                        "optimizer_update": update,
                        "component": component,
                        **natural_record["checks"],
                        "pattern_present": natural_record["pattern_present"],
                    })

    return {
        "primary_D": primary_D,
        "cross_seed_cosine": cross_seed,
        "phase_stability": phase_stability,
        "retrospective_localization": {
            "disclosure": config["disclosure"],
            "dependent_check_count": len(dependent_checks),
            "pattern_present_in_all_dependent_checks": all(
                row["pattern_present"] for row in dependent_checks
            ),
            "dependent_checks": dependent_checks,
            "all_checkpoint_diagnostics": diagnostics,
            "interpretation": config["retrospective_criterion"]["interpretation"],
        },
        "natural_stratum_localization": {
            "definition": config["estimands"]["natural_D"],
            "dependent_check_count": len(natural_dependent_checks),
            "pattern_present_in_all_dependent_checks": all(
                row["pattern_present"] for row in natural_dependent_checks
            ),
            "dependent_checks": natural_dependent_checks,
            "all_checkpoint_diagnostics": natural_diagnostics,
            "interpretation": config["retrospective_criterion"]
            ["natural_stratum_reporting"],
        },
    }


def build_result(
    config: dict[str, Any], sources: dict[str, Any], analysis: dict[str, Any]
) -> dict[str, Any]:
    scientific_payload = {
        "per_seed_checkpoint": analysis["per_seed_checkpoint"],
        "summary": analysis["summary"],
    }
    return {
        "name": config["name"],
        "created_at": utc_now(),
        "question": config["question"],
        "scope": config["scope"],
        "disclosure": config["disclosure"],
        "config": artifact_record(CONFIG_PATH),
        "script": artifact_record(SCRIPT_PATH),
        "parents": {
            name: artifact_record(path) for name, path in sources["parent_paths"].items()
        },
        "source_cells": sources["cell_records"],
        "analysis_loaded_model_or_optimizer": False,
        "analysis_accessed_accelerator": False,
        "per_seed_checkpoint": analysis["per_seed_checkpoint"],
        "summary": analysis["summary"],
        "integrity": analysis["integrity"],
        "scientific_payload_sha256": compact_hash(scientific_payload),
    }


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "undefined"
    return f"{value:+.{digits}g}"


def render_markdown(config: dict[str, Any], result: dict[str, Any]) -> str:
    inventory = config["inventory"]
    summary = result["summary"]
    lines = [
        "# ds2 numeric-to-wolf block reanalysis v1",
        "",
        config["question"],
        "",
        "**Retrospective disclosure:** "
        + config["disclosure"]["pre_freeze_spot_inspection"]
        + " "
        + config["disclosure"]["consequence"],
        "",
        "This is a pure-JSON reanalysis of native-scale additive `raw_dot`; no model,",
        "optimizer, tensor runtime, or accelerator was loaded.",
        "",
        "## Primary theta-averaged D localization",
        "",
        "L1 shares use absolute group contributions within each disjoint partition;",
        "the signed groups still reconstruct the signed total exactly.",
        "",
        "| seed | update | component | signed D | early/middle/late L1 share | top layers | top module | A/B L1 share |",
        "|---:|---:|---|---:|---|---|---|---|",
    ]
    labels = {
        "clipped_raw_descent_direction": "clipped raw descent",
        "adam_current_gradient_adaptive_update": "Adam current/live-v",
        "adam_preconditioned_adaptive_update": "Adam adaptive total",
        "manual_total_update": "full AdamW",
    }
    for seed in inventory["seeds"]:
        for update in inventory["checkpoints"]:
            for component in inventory["primary_components"]:
                row = summary["primary_D"][str(seed)][str(update)][component]
                partitions = row["partitions"]
                band = partitions["bands"]["l1_share"]
                layer_top = partitions["layers"]["top_groups"][:3]
                module_top = partitions["modules"]["top_groups"][0]
                sides = partitions["lora_sides"]["l1_share"]
                lines.append(
                    f"| {seed} | {update} | {labels[component]} | {fmt(row['signed_total'])} "
                    f"| {band['band_early']:.1%}/{band['band_middle']:.1%}/{band['band_late']:.1%} "
                    f"| {', '.join(item['group'].replace('layer_', 'L') + ' ' + fmt(item['value'], 3) for item in layer_top)} "
                    f"| {module_top['group'].replace('module_', '')} {fmt(module_top['value'], 3)} "
                    f"| {sides['lora_A']:.1%}/{sides['lora_B']:.1%} |"
                )

    lines.extend([
        "",
        "## Cross-seed layer cosine",
        "",
        "| update | clipped raw | Adam current/live-v | Adam adaptive total | full AdamW |",
        "|---:|---:|---:|---:|---:|",
    ])
    for update in inventory["checkpoints"]:
        row = summary["cross_seed_cosine"][str(update)]
        lines.append(
            f"| {update} "
            + " ".join(
                f"| {fmt(row[component]['layers'])}"
                for component in inventory["primary_components"]
            )
            + " |"
        )

    lines.extend([
        "",
        "## Early versus late layer stability",
        "",
        "Early is u8/u16/u32; late is u64/u128/u256/u512.",
        "",
        "| seed | component | early-vs-late cosine | same-sign layer fraction | early top layer | late top layer |",
        "|---:|---|---:|---:|---|---|",
    ])
    for seed in inventory["seeds"]:
        for component in inventory["primary_components"]:
            row = summary["phase_stability"][str(seed)][component]["layers"]
            lines.append(
                f"| {seed} | {labels[component]} | {fmt(row['early_vs_late_cosine'])} "
                f"| {row['same_sign_fraction']:.1%} "
                f"| {row['early']['top_groups'][0]['group']} {fmt(row['early']['top_groups'][0]['value'], 3)} "
                f"| {row['late']['top_groups'][0]['group']} {fmt(row['late']['top_groups'][0]['value'], 3)} |"
            )

    retrospective = summary["retrospective_localization"]
    lines.extend([
        "",
        "## Retrospective full-factorial late-localization pattern",
        "",
        "Retrospective pattern **"
        + (
            "present in all"
            if retrospective["pattern_present_in_all_dependent_checks"]
            else "not present in all"
        )
        + f" {retrospective['dependent_check_count']} dependent checks**. The checks reuse "
        "the same factorial cells and gradients; this count is not an independent replication count.",
        "",
        "| seed | update | component | signed late D contribution | late-(early+middle) | module contrast | pattern present |",
        "|---:|---:|---|---:|---:|---:|---|",
    ])
    diagnostics = retrospective["all_checkpoint_diagnostics"]
    for seed in inventory["seeds"]:
        for update in config["retrospective_criterion"]["checkpoints"]:
            for component in config["retrospective_criterion"]["components"]:
                row = diagnostics[str(seed)][str(update)][component]
                lines.append(
                    f"| {seed} | {update} | {labels[component]} | {fmt(row['band_late'])} "
                    f"| {fmt(row['late_minus_early_middle'])} | {fmt(row['module_contrast'])} "
                    f"| {'yes' if row['pattern_present'] else 'no'} |"
                )

    natural = summary["natural_stratum_localization"]
    lines.extend([
        "",
        "## Retrospective natural-stratum late-localization pattern",
        "",
        "Here `natural_D` fixes `M=V=T`, contrasts current preference versus control data,",
        "and then averages over the two theta strata.",
        "",
        "Retrospective natural-stratum pattern **"
        + (
            "present in all"
            if natural["pattern_present_in_all_dependent_checks"]
            else "not present in all"
        )
        + f" {natural['dependent_check_count']} dependent checks**. This is a matched-stratum view "
        "of the same artifacts, not independent confirmation.",
        "",
        "| seed | update | component | signed late D contribution | late-(early+middle) | module contrast | pattern present |",
        "|---:|---:|---|---:|---:|---:|---|",
    ])
    natural_diagnostics = natural["all_checkpoint_diagnostics"]
    for seed in inventory["seeds"]:
        for update in config["retrospective_criterion"]["checkpoints"]:
            for component in config["retrospective_criterion"]["components"]:
                row = natural_diagnostics[str(seed)][str(update)][component]
                lines.append(
                    f"| {seed} | {update} | {labels[component]} | {fmt(row['band_late'])} "
                    f"| {fmt(row['late_minus_early_middle'])} | {fmt(row['module_contrast'])} "
                    f"| {'yes' if row['pattern_present'] else 'no'} |"
                )

    integrity = result["integrity"]
    lines.extend([
        "",
        "## Integrity",
        "",
        f"- Validated theta cells: {integrity['validated_theta_cells']}",
        f"- Validated native candidates: {integrity['validated_native_candidates']}",
        f"- Hash-validated artifact records: {integrity['artifact_records_hash_validated']}",
        "- Maximum candidate group additivity error: "
        + canonical_json(integrity["candidate_group_additivity_max_absolute_error"]),
        "- Maximum factorial-effect group additivity error: "
        + canonical_json(integrity["factorial_effect_group_additivity_max_absolute_error"]),
        f"- Maximum parent scalar reproduction error: {integrity['parent_behavior_main_effect_max_absolute_error']:.3e}",
        f"- Scientific payload SHA256: `{result['scientific_payload_sha256']}`",
        "- All checks passed: yes",
        "",
        "## Interpretation limits",
        "",
        "- Both localization patterns are retrospective, statistically dependent summaries, not independent confirmation.",
        "- `raw_dot` is an additive local parameter-space first-order quantity; L1 shares are descriptive concentration summaries.",
        "- LoRA A/B shares are gauge-dependent and descriptive only in the frozen PEFT parameterization; equivalent low-rank products permit compensating A/B rescalings or basis changes.",
        "- M and V donor states are co-adapted observational products before factorial crossing.",
        "- One-step block localization does not establish long-horizon necessity, activation-space identity, or a unique numeric solution.",
    ])
    return "\n".join(lines) + "\n"


def preflight(write: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config()
    sources = load_sources(config)
    record = {
        "name": config["name"] + "-preflight",
        "created_at": utc_now(),
        "config": artifact_record(CONFIG_PATH),
        "script": artifact_record(SCRIPT_PATH),
        "parents": {
            name: artifact_record(path) for name, path in sources["parent_paths"].items()
        },
        "validated_theta_cells": len(sources["cells"]),
        "validated_native_candidates": config["inventory"]["native_candidate_count"],
        "hash_validated_artifact_records": sources["artifact_records_verified"],
        "candidate_group_additivity_max_absolute_error": sources["additivity_max_errors"],
        "analysis_loaded_model_or_optimizer": False,
        "analysis_accessed_accelerator": False,
        "passed": True,
    }
    if write:
        atomic_write_json(rooted(config["outputs"]["preflight"]), record)
    return config, sources


def run() -> dict[str, Any]:
    config, sources = preflight(write=True)
    analysis = analyze(config, sources)
    result = build_result(config, sources, analysis)
    out_json = rooted(config["outputs"]["aggregate_json"])
    out_md = rooted(config["outputs"]["aggregate_markdown"])
    atomic_write_json(out_json, result)
    atomic_write_text(out_md, render_markdown(config, result))
    return result


def status() -> dict[str, Any]:
    config, sources = preflight(write=False)
    out_json = rooted(config["outputs"]["aggregate_json"])
    out_md = rooted(config["outputs"]["aggregate_markdown"])
    if not out_json.is_file() or not out_md.is_file():
        return {
            "name": config["name"],
            "complete": False,
            "reason": "aggregate outputs absent",
            "analysis_loaded_model_or_optimizer": False,
            "analysis_accessed_accelerator": False,
        }
    observed = read_json(out_json)
    analysis = analyze(config, sources)
    expected_payload_hash = compact_hash({
        "per_seed_checkpoint": analysis["per_seed_checkpoint"],
        "summary": analysis["summary"],
    })
    observed_payload_hash = compact_hash({
        "per_seed_checkpoint": observed.get("per_seed_checkpoint"),
        "summary": observed.get("summary"),
    })
    markdown_exact = out_md.read_text() == render_markdown(config, observed)
    complete = (
        observed.get("name") == config["name"]
        and observed.get("config") == artifact_record(CONFIG_PATH)
        and observed.get("script") == artifact_record(SCRIPT_PATH)
        and observed.get("scientific_payload_sha256") == expected_payload_hash
        and observed_payload_hash == expected_payload_hash
        and observed.get("integrity", {}).get("all_checks_passed") is True
        and observed.get("analysis_loaded_model_or_optimizer") is False
        and observed.get("analysis_accessed_accelerator") is False
        and markdown_exact
    )
    return {
        "name": config["name"],
        "complete": complete,
        "scientific_payload_sha256": expected_payload_hash,
        "retrospective_pattern_present": analysis["summary"]
        ["retrospective_localization"]["pattern_present_in_all_dependent_checks"],
        "natural_stratum_pattern_present": analysis["summary"]
        ["natural_stratum_localization"]["pattern_present_in_all_dependent_checks"],
        "validated_theta_cells": len(sources["cells"]),
        "validated_native_candidates": config["inventory"]["native_candidate_count"],
        "markdown_exact": markdown_exact,
        "analysis_loaded_model_or_optimizer": False,
        "analysis_accessed_accelerator": False,
    }


def synthetic_rows() -> list[dict[str, Any]]:
    rows = []
    contributions = {
        "layer_00": {"D": 2.0, "M": 3.0, "V": 5.0},
        "layer_01": {"D": -0.5, "M": 1.0, "V": -2.0},
    }
    for theta in ("control", "preference"):
        st = 1 if theta == "preference" else -1
        for data in ("control", "preference"):
            sd = 1 if data == "preference" else -1
            for moment in ("control", "preference"):
                sm = 1 if moment == "preference" else -1
                for variance in ("control", "preference"):
                    sv = 1 if variance == "preference" else -1
                    groups = {}
                    for group, effects in contributions.items():
                        groups[group] = (
                            0.5 * effects["D"] * sd
                            + 0.5 * effects["M"] * sm
                            + 0.5 * effects["V"] * sv
                            + 0.125 * st * sd
                        )
                    groups["all"] = groups["layer_00"] + groups["layer_01"]
                    candidate = {
                        "data_condition": data,
                        "exp_avg_source": moment,
                        "exp_avg_sq_source": variance,
                        "factor_signs": {"D": sd, "M": sm, "V": sv},
                    }
                    rows.append({
                        "theta": theta,
                        "candidate": candidate,
                        "geometry": {"synthetic": groups},
                        "behavior": groups["all"],
                    })
    return rows


def self_test() -> dict[str, Any]:
    rows = synthetic_rows()
    expected = {
        "D": {"layer_00": 2.0, "layer_01": -0.5, "all": 1.5},
        "M": {"layer_00": 3.0, "layer_01": 1.0, "all": 4.0},
        "V": {"layer_00": 5.0, "layer_01": -2.0, "all": 3.0},
    }
    errors = {}
    for factor, groups in expected.items():
        for group, target in groups.items():
            observed = effect_for_rows(rows, factor, "synthetic", group)
            errors[f"{factor}/{group}"] = abs(observed - target)
    natural = natural_D(rows, "synthetic", "all")
    errors["natural_D/all"] = abs(natural - expected["D"]["all"])
    errors["group_additivity/D"] = abs(
        effect_for_rows(rows, "D", "synthetic", "all")
        - effect_for_rows(rows, "D", "synthetic", "layer_00")
        - effect_for_rows(rows, "D", "synthetic", "layer_01")
    )
    cosine_error = abs((cosine([1.0, 0.0], [1.0, 0.0]) or 0.0) - 1.0)
    errors["cosine_identity"] = cosine_error
    passed = max(errors.values()) <= 1e-12 and "torch" not in sys.modules
    if not passed:
        raise RuntimeError(f"Synthetic self-test failed: {errors}")
    return {
        "name": "ds2-numeric-wolf-block-reanalysis-synthetic-self-test",
        "row_count": len(rows),
        "maximum_absolute_error": max(errors.values()),
        "errors": errors,
        "analysis_loaded_model_or_optimizer": False,
        "analysis_accessed_accelerator": False,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("preflight", "run", "status", "self-test"),
        help="Validate sources, run the analysis, validate outputs, or run synthetic tests.",
    )
    args = parser.parse_args()
    if args.command == "preflight":
        config, sources = preflight(write=True)
        output = {
            "name": config["name"],
            "passed": True,
            "validated_theta_cells": len(sources["cells"]),
            "hash_validated_artifact_records": sources["artifact_records_verified"],
            "analysis_loaded_model_or_optimizer": False,
            "analysis_accessed_accelerator": False,
        }
    elif args.command == "run":
        result = run()
        output = {
            "name": result["name"],
            "complete": True,
            "scientific_payload_sha256": result["scientific_payload_sha256"],
            "retrospective_pattern_present": result["summary"]
            ["retrospective_localization"]["pattern_present_in_all_dependent_checks"],
            "natural_stratum_pattern_present": result["summary"]
            ["natural_stratum_localization"]["pattern_present_in_all_dependent_checks"],
            "analysis_loaded_model_or_optimizer": False,
            "analysis_accessed_accelerator": False,
        }
    elif args.command == "status":
        output = status()
    else:
        output = self_test()
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
