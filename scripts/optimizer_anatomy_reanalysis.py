"""Zero-compute optimizer-anatomy reanalysis.

This script reads the completed numeric-fingerprint update-geometry JSON cells.
It never imports model, tensor, optimizer, dataset, or accelerator libraries.
The output separates native-scale projection from norm-controlled alignment for
the raw gradient, unpreconditioned Adam momentum, Adam history/current pieces
under the live second-moment denominator, the total adaptive update, and the
actual optimizer update.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs/optimizer_anatomy_reanalysis_v1.json"
SCRIPT_PATH = Path(__file__).resolve()
SOURCE_ROOT = ROOT / "runs/numeric_fingerprint_update_geometry_v1/cells"
OUT_JSON = ROOT / "runs/optimizer_anatomy_reanalysis_v1.json"
OUT_MD = ROOT / "runs/optimizer_anatomy_reanalysis_v1.md"

BRANCH_METRICS = (
    "dot",
    "component_norm",
    "projection_per_update_l2",
    "cosine",
)
CONTRASTS = (
    "preference_state_data_effect",
    "control_state_data_effect",
    "same_state_data_main_D",
    "state_by_data_interaction_I",
    "live_paired_S",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Expected numeric {label}, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite {label}: {result}")
    return result


def validate_config(config: dict[str, Any]) -> None:
    if config.get("name") != "optimizer-anatomy-reanalysis-v1":
        raise RuntimeError("Unexpected reanalysis config identity")
    grid = config["grid"]
    expected_grid = {
        "receivers": ["standard", "weight_seed3"],
        "seeds": [56101, 56102],
        "source_conditions": ["preference", "control"],
        "fork_conditions": ["preference", "control"],
        "checkpoints": [0, 16, 64, 128, 256, 512, 1024, 1536, 2048],
    }
    if grid != expected_grid:
        raise RuntimeError("Frozen source grid changed")
    expected_components = [
        ("raw_gradient", "clipped_raw_lr_scaled_update"),
        (
            "unpreconditioned_momentum",
            "unpreconditioned_bias_corrected_momentum_lr_scaled_update",
        ),
        ("adam_history_under_live_v", "adam_history_adaptive_update"),
        ("adam_current_under_live_v", "adam_current_gradient_adaptive_update"),
        ("adam_total_adaptive", "adam_preconditioned_adaptive_update"),
        ("actual_update", "actual_optimizer_step"),
    ]
    observed_components = [
        (row.get("label"), row.get("source_key")) for row in config["components"]
    ]
    if observed_components != expected_components:
        raise RuntimeError("Frozen optimizer component inventory changed")
    if config["phases"] != {
        "onset": [0, 16],
        "emergence": [64, 128],
        "transition": [256, 512],
        "attenuation": [1024, 1536, 2048],
    }:
        raise RuntimeError("Frozen phase definitions changed")
    if tuple(config["phase_summary"]["series"]) != CONTRASTS:
        raise RuntimeError("Frozen contrast summary inventory changed")
    if tuple(config["phase_summary"]["metrics"]) != BRANCH_METRICS:
        raise RuntimeError("Frozen metric summary inventory changed")


def verify_parent_files(config: dict[str, Any]) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for label in ("aggregate", "runner_lock", "config", "runner"):
        path_text, expected_hash = config["parents"][label]
        path = ROOT / path_text
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"Parent {label} changed: expected {expected_hash}, got {observed_hash}"
            )
        verified[label] = {
            "path": path_text,
            "sha256": observed_hash,
            "bytes": path.stat().st_size,
        }
    return verified


def expected_cell_path(
    receiver: str, seed: int, condition: str, update: int
) -> Path:
    return (
        SOURCE_ROOT
        / receiver
        / f"seed_{seed}"
        / condition
        / f"u{update:04d}"
        / "cell.json"
    )


def inventory_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(SOURCE_ROOT.glob("*/seed_*/*/u*/cell.json"), key=str):
        cell = load_object(path)
        measurement = cell["measurement"]
        records.append(
            {
                "cell_path": relative(path),
                "cell_sha256": sha256_file(path),
                "measurement_path": measurement["path"],
                "measurement_sha256": measurement["sha256"],
                "measurement_bytes": measurement["bytes"],
            }
        )
    return records


def load_measurement(
    config: dict[str, Any], receiver: str, seed: int, condition: str, update: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = expected_cell_path(receiver, seed, condition, update)
    cell = load_object(path)
    identity = (
        cell.get("receiver"),
        int(cell.get("seed", -1)),
        cell.get("source_condition"),
        int(cell.get("optimizer_update", -1)),
    )
    if identity != (receiver, seed, condition, update):
        raise RuntimeError(f"Cell identity mismatch: {path}: {identity}")
    parent_config_hash = config["parents"]["config"][1]
    parent_lock_hash = config["parents"]["runner_lock"][1]
    if (
        cell.get("config_sha256") != parent_config_hash
        or cell.get("runner_lock_sha256") != parent_lock_hash
    ):
        raise RuntimeError(f"Cell parent hashes changed: {path}")
    artifact = cell["measurement"]
    measurement_path = ROOT / artifact["path"]
    if (
        measurement_path.stat().st_size != int(artifact["bytes"])
        or sha256_file(measurement_path) != artifact["sha256"]
    ):
        raise RuntimeError(f"Measurement artifact changed: {measurement_path}")
    measurement = load_object(measurement_path)
    measurement_identity = (
        measurement.get("receiver"),
        int(measurement.get("seed", -1)),
        measurement.get("source_condition"),
        int(measurement.get("optimizer_update", -1)),
    )
    if measurement_identity != identity:
        raise RuntimeError(f"Measurement identity mismatch: {measurement_path}")
    if set(measurement.get("branches", {})) != set(config["grid"]["fork_conditions"]):
        raise RuntimeError(f"Measurement branch inventory changed: {measurement_path}")
    return cell, measurement


def branch_component_metrics(
    measurement: dict[str, Any], branch: dict[str, Any], source_key: str,
    tolerances: dict[str, Any],
) -> dict[str, Any]:
    trait_norm = require_number(
        measurement["current_wolf_margin"]["gradient"]["l2_norm"],
        "wolf-gradient norm",
    )
    dot = require_number(
        branch["projections_on_wolf_margin_gradient"][source_key],
        f"{source_key} projection",
    )
    component_norm = require_number(
        branch["component_l2_norms"][source_key], f"{source_key} norm"
    )
    if trait_norm <= 0.0:
        raise RuntimeError("Wolf-gradient norm must be positive")
    if component_norm < 0.0:
        raise RuntimeError("Component norm must be nonnegative")
    projection_per_update_l2 = (
        None if component_norm == 0.0 else dot / component_norm
    )
    cosine = (
        None
        if projection_per_update_l2 is None
        else projection_per_update_l2 / trait_norm
    )
    bound = 1.0 + float(tolerances["cosine_absolute_bound_tolerance"])
    if cosine is not None and abs(cosine) > bound:
        raise RuntimeError(f"Impossible cosine for {source_key}: {cosine}")
    return {
        "dot": dot,
        "component_norm": component_norm,
        "wolf_gradient_norm": trait_norm,
        "projection_per_update_l2": projection_per_update_l2,
        "cosine": cosine,
    }


def subtract_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in BRANCH_METRICS:
        a, b = left[metric], right[metric]
        result[metric] = None if a is None or b is None else float(a) - float(b)
    return result


def combine_metrics(
    left: dict[str, Any], right: dict[str, Any], left_scale: float, right_scale: float
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in BRANCH_METRICS:
        a, b = left[metric], right[metric]
        result[metric] = (
            None
            if a is None or b is None
            else left_scale * float(a) + right_scale * float(b)
        )
    return result


def summarize(values: list[Any]) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None]
    undefined = len(values) - len(finite)
    if not finite:
        return {
            "count": 0,
            "undefined_count": undefined,
            "mean": None,
            "population_sd": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(finite),
        "undefined_count": undefined,
        "mean": statistics.fmean(finite),
        "population_sd": statistics.pstdev(finite),
        "minimum": min(finite),
        "maximum": max(finite),
    }


def checkpoint_record(
    config: dict[str, Any], receiver: str, seed: int, update: int,
    integrity: dict[str, Any],
) -> dict[str, Any]:
    source_states: dict[str, Any] = {}
    for source_condition in config["grid"]["source_conditions"]:
        _, measurement = load_measurement(
            config, receiver, seed, source_condition, update
        )
        trait_norm = require_number(
            measurement["current_wolf_margin"]["gradient"]["l2_norm"],
            "wolf-gradient norm",
        )
        state = {
            "wolf_gradient_norm": trait_norm,
            "data_branches": {},
        }
        for fork_condition in config["grid"]["fork_conditions"]:
            branch = measurement["branches"][fork_condition]
            verification = branch["manual_adamw_verification"]
            if verification.get("passed") is not True:
                raise RuntimeError("Source branch failed manual AdamW verification")
            actual_manual_error = abs(
                require_number(
                    verification["actual_minus_manual_projection"],
                    "actual-minus-manual projection",
                )
            )
            integrity["maximum_actual_minus_manual_projection_error"] = max(
                integrity["maximum_actual_minus_manual_projection_error"],
                actual_manual_error,
            )
            components: dict[str, Any] = {}
            for component in config["components"]:
                metrics = branch_component_metrics(
                    measurement,
                    branch,
                    component["source_key"],
                    config["integrity_tolerances"],
                )
                components[component["label"]] = metrics
                integrity["maximum_absolute_defined_cosine"] = max(
                    integrity["maximum_absolute_defined_cosine"],
                    0.0 if metrics["cosine"] is None else abs(metrics["cosine"]),
                )
                if metrics["cosine"] is None:
                    integrity["undefined_cosines_by_component"][
                        component["label"]
                    ] += 1
            history = components["adam_history_under_live_v"]["dot"]
            current = components["adam_current_under_live_v"]["dot"]
            adaptive = components["adam_total_adaptive"]["dot"]
            decomposition_error = abs(history + current - adaptive)
            integrity["maximum_history_plus_current_dot_error"] = max(
                integrity["maximum_history_plus_current_dot_error"],
                decomposition_error,
            )
            integrity["measurement_branch_count"] += 1
            state["data_branches"][fork_condition] = {"components": components}
        source_states[source_condition] = state

    pp = source_states["preference"]["data_branches"]["preference"]["components"]
    pc = source_states["preference"]["data_branches"]["control"]["components"]
    cp = source_states["control"]["data_branches"]["preference"]["components"]
    cc = source_states["control"]["data_branches"]["control"]["components"]
    contrasts: dict[str, Any] = {name: {} for name in CONTRASTS}
    for component in config["components"]:
        label = component["label"]
        pref_effect = subtract_metrics(pp[label], pc[label])
        control_effect = subtract_metrics(cp[label], cc[label])
        contrasts["preference_state_data_effect"][label] = pref_effect
        contrasts["control_state_data_effect"][label] = control_effect
        contrasts["same_state_data_main_D"][label] = combine_metrics(
            pref_effect, control_effect, 0.5, 0.5
        )
        contrasts["state_by_data_interaction_I"][label] = subtract_metrics(
            pref_effect, control_effect
        )
        contrasts["live_paired_S"][label] = subtract_metrics(pp[label], cc[label])

    path_deltas: dict[str, Any] = {}
    for contrast_name in CONTRASTS:
        path_deltas[contrast_name] = {}
        for delta in config["optimizer_path_deltas"]:
            path_deltas[contrast_name][delta["label"]] = subtract_metrics(
                contrasts[contrast_name][delta["left"]],
                contrasts[contrast_name][delta["right"]],
            )
    return {
        "optimizer_update": update,
        "source_states": source_states,
        "wolf_gradient_norm_state_difference": (
            source_states["preference"]["wolf_gradient_norm"]
            - source_states["control"]["wolf_gradient_norm"]
        ),
        "contrasts": contrasts,
        "ordered_optimizer_path_deltas": path_deltas,
    }


def phase_summaries(
    config: dict[str, Any], checkpoints: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for phase, updates in config["phases"].items():
        selected = [checkpoints[str(update)] for update in updates]
        contrast_summaries: dict[str, Any] = {}
        path_summaries: dict[str, Any] = {}
        for contrast_name in CONTRASTS:
            contrast_summaries[contrast_name] = {}
            for component in config["components"]:
                label = component["label"]
                contrast_summaries[contrast_name][label] = {
                    metric: summarize(
                        [
                            row["contrasts"][contrast_name][label][metric]
                            for row in selected
                        ]
                    )
                    for metric in BRANCH_METRICS
                }
            path_summaries[contrast_name] = {}
            for delta in config["optimizer_path_deltas"]:
                label = delta["label"]
                path_summaries[contrast_name][label] = {
                    metric: summarize(
                        [
                            row["ordered_optimizer_path_deltas"][contrast_name][label][
                                metric
                            ]
                            for row in selected
                        ]
                    )
                    for metric in BRANCH_METRICS
                }
        output[phase] = {
            "updates": updates,
            "source_wolf_gradient_norms": {
                "preference_state": summarize(
                    [
                        row["source_states"]["preference"]["wolf_gradient_norm"]
                        for row in selected
                    ]
                ),
                "control_state": summarize(
                    [
                        row["source_states"]["control"]["wolf_gradient_norm"]
                        for row in selected
                    ]
                ),
                "preference_minus_control": summarize(
                    [row["wolf_gradient_norm_state_difference"] for row in selected]
                ),
            },
            "contrast_summaries": contrast_summaries,
            "ordered_optimizer_path_delta_summaries": path_summaries,
        }
    return output


def build_aggregate() -> dict[str, Any]:
    config = load_object(CONFIG_PATH)
    validate_config(config)
    verified_parents = verify_parent_files(config)
    inventory = inventory_records()
    if len(inventory) != int(config["parents"]["cell_count"]):
        raise RuntimeError("Source cell count changed")
    inventory_sha = compact_sha256(inventory)
    expected_inventory_sha = config["parents"]["cell_measurement_inventory_sha256"]
    if inventory_sha != expected_inventory_sha:
        raise RuntimeError(
            f"Source cell/measurement inventory changed: {inventory_sha}"
        )
    expected_paths = {
        relative(expected_cell_path(receiver, seed, condition, update))
        for receiver in config["grid"]["receivers"]
        for seed in config["grid"]["seeds"]
        for condition in config["grid"]["source_conditions"]
        for update in config["grid"]["checkpoints"]
    }
    if {row["cell_path"] for row in inventory} != expected_paths:
        raise RuntimeError("Source cell path inventory differs from frozen grid")

    integrity = {
        "source_cell_count": len(inventory),
        "measurement_branch_count": 0,
        "source_cell_measurement_inventory_sha256": inventory_sha,
        "maximum_absolute_defined_cosine": 0.0,
        "maximum_history_plus_current_dot_error": 0.0,
        "maximum_actual_minus_manual_projection_error": 0.0,
        "undefined_cosines_by_component": {
            component["label"]: 0 for component in config["components"]
        },
    }
    per_pair: dict[str, Any] = {}
    for receiver in config["grid"]["receivers"]:
        per_pair[receiver] = {}
        for seed in config["grid"]["seeds"]:
            checkpoints = {
                str(update): checkpoint_record(
                    config, receiver, seed, update, integrity
                )
                for update in config["grid"]["checkpoints"]
            }
            per_pair[receiver][str(seed)] = {
                "checkpoints": checkpoints,
                "phase_summaries": phase_summaries(config, checkpoints),
            }
    tolerances = config["integrity_tolerances"]
    if integrity["measurement_branch_count"] != int(
        config["parents"]["measurement_branch_count"]
    ):
        raise RuntimeError("Measurement branch count changed")
    if integrity["maximum_history_plus_current_dot_error"] > float(
        tolerances["history_plus_current_dot_absolute_tolerance"]
    ):
        raise RuntimeError("Adam history/current dot decomposition failed")
    if integrity["maximum_actual_minus_manual_projection_error"] > float(
        tolerances["actual_minus_manual_projection_absolute_tolerance"]
    ):
        raise RuntimeError("Actual/manual projection verification failed")
    integrity["all_checks_passed"] = True
    return {
        "name": config["name"],
        "created_at": utc_now(),
        "question": config["question"],
        "scope": config["scope"],
        "metric_definitions": config["metric_definitions"],
        "contrast_definitions": config["contrast_definitions"],
        "optimizer_path_deltas": config["optimizer_path_deltas"],
        "phases": config["phases"],
        "runtime": {
            "script": {
                "path": relative(SCRIPT_PATH),
                "sha256": sha256_file(SCRIPT_PATH),
                "bytes": SCRIPT_PATH.stat().st_size,
            },
            "config": {
                "path": relative(CONFIG_PATH),
                "sha256": sha256_file(CONFIG_PATH),
                "bytes": CONFIG_PATH.stat().st_size,
            },
            "parents": verified_parents,
            "execution": "Python standard-library JSON reanalysis only; no model or accelerator code imported",
        },
        "integrity": integrity,
        "per_pair": per_pair,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "undefined"
    number = float(value)
    if number == 0.0:
        return "0"
    if abs(number) < 1e-4 or abs(number) >= 1000:
        return f"{number:+.4e}"
    return f"{number:+.6f}"


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Optimizer anatomy reanalysis v1",
        "",
        result["question"],
        "",
        "This is a zero-compute descriptive reanalysis of the 72 completed exact-state update-geometry cells. No model, optimizer, dataset, tensor runtime, or accelerator was loaded.",
        "",
        "Dot is native-scale first-order wolf-margin movement. Projection/update-L2 is the primary norm-controlled within-source-state comparison. Branch cosine additionally divides by the stored positive local wolf-gradient norm. Cross-state entries are differences or averages of local metrics, not a cosine between cross-state vectors.",
        "",
        "## Same-state preference-vs-control data anatomy",
        "",
        "Phase means below are the factorial same-state data main effect D.",
        "",
        "| receiver | seed | phase | component | D dot | D projection/update-L2 | D local-cosine contrast |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for receiver, seeds in result["per_pair"].items():
        for seed, pair in seeds.items():
            for phase in ("onset", "emergence", "transition", "attenuation"):
                summaries = pair["phase_summaries"][phase]["contrast_summaries"][
                    "same_state_data_main_D"
                ]
                for component in (
                    "raw_gradient",
                    "unpreconditioned_momentum",
                    "adam_history_under_live_v",
                    "adam_current_under_live_v",
                    "adam_total_adaptive",
                    "actual_update",
                ):
                    row = summaries[component]
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                receiver,
                                seed,
                                phase,
                                component,
                                fmt(row["dot"]["mean"]),
                                fmt(row["projection_per_update_l2"]["mean"]),
                                fmt(row["cosine"]["mean"]),
                            ]
                        )
                        + " |"
                    )
    lines.extend(
        [
            "",
            "## Ordered optimizer-path deltas",
            "",
            "These are descriptive differences in the same-state D summaries. They are not causal or Shapley attributions.",
            "",
            "| receiver | seed | phase | path delta | dot change | projection/update-L2 change | local-cosine contrast change |",
            "|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for receiver, seeds in result["per_pair"].items():
        for seed, pair in seeds.items():
            for phase in ("onset", "emergence", "transition", "attenuation"):
                summaries = pair["phase_summaries"][phase][
                    "ordered_optimizer_path_delta_summaries"
                ]["same_state_data_main_D"]
                for delta in (
                    "momentum_minus_raw",
                    "adaptive_minus_momentum",
                    "actual_minus_adaptive",
                ):
                    row = summaries[delta]
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                receiver,
                                seed,
                                phase,
                                delta,
                                fmt(row["dot"]["mean"]),
                                fmt(row["projection_per_update_l2"]["mean"]),
                                fmt(row["cosine"]["mean"]),
                            ]
                        )
                        + " |"
                    )
    integrity = result["integrity"]
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Source cells: {integrity['source_cell_count']}",
            f"- Measurement branches: {integrity['measurement_branch_count']}",
            f"- Cell/measurement inventory SHA256: {integrity['source_cell_measurement_inventory_sha256']}",
            f"- Maximum history + current minus adaptive dot error: {integrity['maximum_history_plus_current_dot_error']:.3e}",
            f"- Maximum actual minus manual projection error: {integrity['maximum_actual_minus_manual_projection_error']:.3e}",
            f"- Undefined cosines by component: {json.dumps(integrity['undefined_cosines_by_component'], sort_keys=True)}",
            "- All checks passed: yes",
            "",
            "## Interpretation limits",
            "",
            "- Projection differences mix alignment and native optimizer scale; use projection/update-L2 beside dot.",
            "- Momentum and second-moment scaling interact. The ordered differences are descriptive, not unique causal allocations.",
            "- Adam history/current pieces share the live updated-v denominator, which includes the current batch gradient squared.",
            "- Live paired S crosses different parameter states. The primary alignment comparison here is the within-source-state preference-vs-control data effect and its factorial mean D.",
            "- Every trait gradient is local to its saved LoRA state. Cross-time and cross-state values are comparisons of local metrics, not one fixed trait vector.",
            "",
        ]
    )
    return "\n".join(lines)


def stable_without_timestamp(value: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(value))
    copied.pop("created_at", None)
    return copied


def run() -> None:
    result = build_aggregate()
    markdown = render_markdown(result)
    atomic_write_json(OUT_JSON, result)
    atomic_write(OUT_MD, markdown)
    print(
        json.dumps(
            {
                "json": relative(OUT_JSON),
                "json_sha256": sha256_file(OUT_JSON),
                "markdown": relative(OUT_MD),
                "markdown_sha256": sha256_file(OUT_MD),
                "integrity": result["integrity"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def validate_outputs() -> None:
    if not OUT_JSON.is_file() or not OUT_MD.is_file():
        raise RuntimeError("Reanalysis outputs do not both exist")
    observed = load_object(OUT_JSON)
    expected = build_aggregate()
    if stable_without_timestamp(observed) != stable_without_timestamp(expected):
        raise RuntimeError("Aggregate JSON does not match a fresh source-only rebuild")
    expected_markdown = render_markdown(observed)
    if OUT_MD.read_text() != expected_markdown:
        raise RuntimeError("Aggregate Markdown does not match aggregate JSON")
    print(
        json.dumps(
            {
                "validated": True,
                "json_sha256": sha256_file(OUT_JSON),
                "markdown_sha256": sha256_file(OUT_MD),
                "source_cell_count": observed["integrity"]["source_cell_count"],
                "measurement_branch_count": observed["integrity"][
                    "measurement_branch_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("run", "validate"), nargs="?", default="run"
    )
    args = parser.parse_args()
    if args.command == "run":
        run()
    else:
        validate_outputs()


if __name__ == "__main__":
    main()
