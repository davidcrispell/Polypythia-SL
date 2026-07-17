"""Independent scalar-only verifier for component-dissection v1.

This verifier deliberately never imports or invokes the assay's ``analyze``
function.  It reconstructs the frozen aggregate from completed scalar cells,
then requires exact equality with the campaign aggregate (apart from its
completion timestamp).  ``verify`` is unavailable until every frozen cell and
the campaign aggregate exist; ``self-test`` has no model/data dependency.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import effective_weight_component_dissection as assay


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = Path(__file__).resolve()
OUT_JSON = ROOT / "runs/effective_weight_component_dissection_v1.json"
OUT_MD = ROOT / "runs/effective_weight_component_dissection_v1.md"
OUT_VERIFY = ROOT / "runs/effective_weight_component_dissection_v1_verify.json"
SEEDS, ENDPOINTS = assay.SEEDS, assay.ENDPOINTS


def outcomes(cell: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "wolf_margin": np.asarray(cell["outcomes"]["behavior"]["wolf_margins"], dtype=np.float64),
        "preference_nll": np.asarray(cell["outcomes"]["numeric"]["preference_nll"], dtype=np.float64),
        "fingerprint_advantage": np.asarray(cell["outcomes"]["numeric"]["fingerprint_advantage"], dtype=np.float64),
    }


def benefit(native: dict[str, Any], patched: dict[str, Any], endpoint_name: str) -> dict[str, np.ndarray]:
    base, intervention = outcomes(native), outcomes(patched)
    sign = 1.0 if endpoint_name == "control" else -1.0
    return {
        "wolf_margin": sign * (intervention["wolf_margin"] - base["wolf_margin"]),
        "preference_nll": sign * (base["preference_nll"] - intervention["preference_nll"]),
        "fingerprint_advantage": sign * (intervention["fingerprint_advantage"] - base["fingerprint_advantage"]),
    }


def samples(values: np.ndarray, outcome: str, config: dict[str, Any]) -> np.ndarray:
    behavior_draws, numeric_draws = assay.bootstrap_draws(config)
    if outcome == "wolf_margin":
        if values.shape != (60,):
            raise RuntimeError("Behavior scalar shape mismatch")
        units, draws = values.reshape(12, 5).mean(axis=1), behavior_draws
    else:
        if values.shape != (512,):
            raise RuntimeError("Numeric scalar shape mismatch")
        units = np.asarray([values[block].mean() for block in assay.numeric_blocks(config)])
        draws = numeric_draws
    return units[draws].mean(axis=1)


def summary(values: np.ndarray, outcome: str, config: dict[str, Any]) -> dict[str, float]:
    draws = samples(values, outcome, config)
    low, high = np.percentile(draws, (2.5, 97.5))
    return {"point": float(values.mean()), "ci_low": float(low), "ci_high": float(high), "bootstrap_mean": float(draws.mean())}


def simultaneous(values: dict[str, np.ndarray], outcome: str, config: dict[str, Any]) -> dict[str, dict[str, float]]:
    labels = sorted(values)
    bootstrap = np.vstack([samples(values[label], outcome, config) for label in labels])
    points = np.asarray([values[label].mean() for label in labels])
    standard_error = bootstrap.std(axis=1, ddof=1)
    safe = np.maximum(standard_error, np.finfo(np.float64).eps)
    maximum = np.max(np.abs((bootstrap - points[:, None]) / safe[:, None]), axis=0)
    critical = float(np.percentile(maximum, 95.0))
    return {label: {"point": float(points[index]),
                    "simultaneous_ci_low": float(points[index] - critical * standard_error[index]),
                    "simultaneous_ci_high": float(points[index] + critical * standard_error[index]),
                    "critical_value": critical}
            for index, label in enumerate(labels)}


def completed_cells(config: dict[str, Any], lock: dict[str, Any]) -> dict[tuple[int, str, str], dict[str, Any]]:
    result = {}
    for seed, endpoint_name, spec, path in assay.expected_cells():
        if not path.exists():
            raise RuntimeError("Verifier only runs after all cells are complete")
        result[(seed, endpoint_name, spec.label)] = assay.validate_cell(path, config, lock, seed, endpoint_name, spec)
    return result


def rebuild(config: dict[str, Any], lock: dict[str, Any], cells: dict[tuple[int, str, str], dict[str, Any]]) -> dict[str, Any]:
    rows: dict[tuple[int, str, str], dict[str, np.ndarray]] = {}
    benefits: dict[str, Any] = {}
    for seed in SEEDS:
        benefits[str(seed)] = {}
        for endpoint_name in ENDPOINTS:
            native = cells[(seed, endpoint_name, "native")]
            benefits[str(seed)][endpoint_name] = {}
            for spec in assay.patch_specs():
                if spec.kind == "native":
                    continue
                value = benefit(native, cells[(seed, endpoint_name, spec.label)], endpoint_name)
                rows[(seed, endpoint_name, spec.label)] = value
                benefits[str(seed)][endpoint_name][spec.label] = {outcome: summary(vector, outcome, config) for outcome, vector in value.items()}
    singleton: dict[str, Any] = {}
    loo: dict[str, Any] = {}
    pair: dict[str, Any] = {}
    shams: dict[str, Any] = {}
    bands: dict[str, Any] = {}
    outcomes_names = config["frozen_analysis"]["outcomes"]
    for seed in SEEDS:
        singleton[str(seed)], loo[str(seed)], pair[str(seed)], shams[str(seed)], bands[str(seed)] = {}, {}, {}, {}, {}
        for endpoint_name in ENDPOINTS:
            singleton[str(seed)][endpoint_name], loo[str(seed)][endpoint_name] = {}, {}
            pair[str(seed)][endpoint_name], shams[str(seed)][endpoint_name], bands[str(seed)][endpoint_name] = {}, {}, {}
            all_real = rows[(seed, endpoint_name, "all_real_a100")]
            shams[str(seed)][endpoint_name]["all"] = {
                str(draw): {outcome: summary(all_real[outcome] - rows[(seed, endpoint_name, f"all_sham{draw}_a100")][outcome], outcome, config) for outcome in outcomes_names}
                for draw in (1, 2)
            }
            pair_values = {outcome: {} for outcome in outcomes_names}
            for index in assay.components():
                one = rows[(seed, endpoint_name, f"single_{index}_real_a100")]
                singleton[str(seed)][endpoint_name][str(index)] = {
                    "alpha_1": {outcome: summary(one[outcome], outcome, config) for outcome in outcomes_names},
                    "alpha_025": {outcome: summary(rows[(seed, endpoint_name, f"single_{index}_real_a025")][outcome], outcome, config) for outcome in outcomes_names},
                    "sham": {str(draw): {outcome: summary(one[outcome] - rows[(seed, endpoint_name, f"single_{index}_sham{draw}_a100")][outcome], outcome, config) for outcome in outcomes_names} for draw in (1, 2)},
                }
                complement = rows[(seed, endpoint_name, f"loo_{index}_real_a100")]
                conditional = {outcome: all_real[outcome] - complement[outcome] for outcome in outcomes_names}
                loo[str(seed)][endpoint_name][str(index)] = {
                    "conditional_contribution": {outcome: summary(conditional[outcome], outcome, config) for outcome in outcomes_names},
                    "redundancy_minus_singleton": {outcome: summary(conditional[outcome] - one[outcome], outcome, config) for outcome in outcomes_names},
                    "real_minus_sham": {outcome: summary(complement[outcome] - rows[(seed, endpoint_name, f"loo_{index}_sham1_a100")][outcome], outcome, config) for outcome in outcomes_names},
                }
            for left in assay.components():
                for right in range(left + 1, len(assay.components())):
                    label = f"{left}_{right}"
                    both = rows[(seed, endpoint_name, f"pair_{left}_{right}_real_a100")]
                    first, second = rows[(seed, endpoint_name, f"single_{left}_real_a100")], rows[(seed, endpoint_name, f"single_{right}_real_a100")]
                    interaction = {outcome: both[outcome] - first[outcome] - second[outcome] for outcome in outcomes_names}
                    pair[str(seed)][endpoint_name][label] = {
                        "interaction": {outcome: summary(interaction[outcome], outcome, config) for outcome in outcomes_names},
                        "real_minus_sham": {outcome: summary(both[outcome] - rows[(seed, endpoint_name, f"pair_{left}_{right}_sham1_a100")][outcome], outcome, config) for outcome in outcomes_names},
                    }
                    for outcome in outcomes_names:
                        pair_values[outcome][label] = interaction[outcome]
            for outcome, value in pair_values.items():
                bands[str(seed)][endpoint_name][outcome] = simultaneous(value, outcome, config)
    full_pass = all(
        benefits[str(seed)][endpoint_name]["all_real_a100"][outcome]["ci_low"] > 0
        and all(shams[str(seed)][endpoint_name]["all"][str(draw)][outcome]["ci_low"] > 0 for draw in (1, 2))
        for seed in SEEDS for endpoint_name in ENDPOINTS for outcome in outcomes_names
    )
    component_gates = {}
    for index in assay.components():
        component_gates[str(index)] = all(
            singleton[str(seed)][endpoint_name][str(index)]["alpha_1"][outcome]["ci_low"] > 0
            and singleton[str(seed)][endpoint_name][str(index)]["alpha_025"][outcome]["point"] > 0
            and all(singleton[str(seed)][endpoint_name][str(index)]["sham"][str(draw)][outcome]["ci_low"] > 0 for draw in (1, 2))
            for seed in SEEDS for endpoint_name in ENDPOINTS for outcome in outcomes_names
        )
    passing = [int(index) for index, passed in component_gates.items() if passed]
    pair_labels = [f"{left}_{right}" for left in assay.components() for right in range(left + 1, len(assay.components()))]
    pair_gate_evidence: dict[str, Any] = {}
    pair_gates: dict[str, bool] = {}
    for label in pair_labels:
        pair_gate_evidence[label] = {}
        decisions = []
        for seed in SEEDS:
            pair_gate_evidence[label][str(seed)] = {}
            for endpoint_name in ENDPOINTS:
                pair_gate_evidence[label][str(seed)][endpoint_name] = {}
                for outcome in outcomes_names:
                    interval = bands[str(seed)][endpoint_name][outcome][label]
                    low, high = interval["simultaneous_ci_low"], interval["simultaneous_ci_high"]
                    excludes = low > 0 or high < 0
                    decisions.append(excludes)
                    pair_gate_evidence[label][str(seed)][endpoint_name][outcome] = {
                        **interval,
                        "excludes_zero": excludes,
                        "sign": "positive" if low > 0 else ("negative" if high < 0 else "unresolved"),
                    }
        pair_gates[label] = all(decisions)
    passing_pairs = [label for label, passed in pair_gates.items() if passed]
    if full_pass and len(passing) == len(assay.components()):
        classification = "distributed_individual_dual_use_supported"
    elif full_pass and passing:
        classification = "literal_individual_dual_use_supported"
    elif full_pass:
        classification = "aggregate_shared_port_consistent_individual_evidence_absent"
    else:
        classification = "fresh_full_prerequisite_failed"
    pair_classification = ("reproducibly_nonzero_pair_interactions_supported" if passing_pairs
                           else "no_reproducibly_nonzero_pair_interaction_resolved")
    return {
        "name": "effective-weight-component-dissection-v1",
        "config_sha256": assay.endpoint.file_sha256(assay.CONFIG_PATH),
        "runner_sha256": assay.endpoint.file_sha256(assay.SCRIPT_PATH),
        "runner_lock_sha256": assay.endpoint.file_sha256(assay.RUNNER_LOCK_PATH),
        "cell_count": len(cells),
        "primary": {"classification": classification, "fresh_full_prerequisite": full_pass,
                    "component_gates": component_gates, "passing_component_count": len(passing),
                    "passing_components": passing, "pair_interaction_classification": pair_classification,
                    "pair_interaction_gates": pair_gates, "passing_pair_interaction_count": len(passing_pairs),
                    "passing_pair_interactions": passing_pairs, "interpretive_limits": config["scope"]},
        "benefits": benefits, "full_sham_contrasts": shams, "singletons": singleton,
        "leave_one_out": loo, "pair_interactions": pair,
        "pair_interaction_simultaneous_intervals": bands,
        "pair_interaction_gate_evidence": pair_gate_evidence,
        "banks": lock["frozen"]["banks"], "no_tensor_outputs": True,
    }


def verify() -> dict[str, Any]:
    config, _, _, _ = assay.load_context()
    lock = assay.validate_lock(config)
    if not OUT_JSON.is_file() or not OUT_MD.is_file():
        raise RuntimeError("Verifier only runs after campaign analysis has completed")
    status = assay.status_report()
    if not status["complete"]:
        raise RuntimeError("Verifier only runs after the complete frozen campaign")
    observed = assay.load_json(OUT_JSON)
    rebuilt = rebuild(config, lock, completed_cells(config, lock))
    observed_core = copy.deepcopy(observed)
    observed_core.pop("completed_at", None)
    if observed_core != rebuilt:
        raise RuntimeError("Campaign aggregate differs from independent scalar recomputation")
    report = {
        "name": "effective-weight-component-dissection-v1-verification",
        "verified_at": assay.utc_now(),
        "verifier_sha256": assay.endpoint.file_sha256(SCRIPT_PATH),
        "aggregate": assay.artifact(OUT_JSON),
        "runner_lock": assay.artifact(assay.RUNNER_LOCK_PATH),
        "expected_cells": len(assay.expected_cells()),
        "completed_cells": status["completed_cells"],
        "aggregate_exact_match_excluding_completed_at": True,
        "passed": True,
        "no_model_loaded": True,
        "no_tensor_outputs": True,
    }
    assay.atomic_json(OUT_VERIFY, report)
    print("COMPONENT-DISSECTION INDEPENDENT VERIFICATION PASSED", flush=True)
    return report


def self_test() -> dict[str, Any]:
    config, _, _, _ = assay.load_context()
    if len(assay.expected_cells()) != 432:
        raise RuntimeError("Frozen cell count changed")
    if set(assay.components()) != set(range(8)):
        raise RuntimeError("Component inventory changed")
    positive = summary(np.ones(60), "wolf_margin", config)
    negative = summary(-np.ones(512), "preference_nll", config)
    bands = simultaneous({f"p{index}": np.full(60, index + 1.0) for index in range(28)}, "wolf_margin", config)
    if positive["ci_low"] <= 0 or negative["ci_high"] >= 0 or len(bands) != 28:
        raise RuntimeError("Model-free scalar verifier synthetic check failed")
    report = {"name": "effective-weight-component-dissection-v1-verifier-self-test", "passed": True,
              "model_loaded": False, "mps_used": False, "expected_cells": len(assay.expected_cells()),
              "positive_summary": positive, "negative_summary": negative, "pair_band_count": len(bands)}
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent scalar verifier for component dissection")
    parser.add_argument("command", choices=("verify", "self-test"))
    args = parser.parse_args()
    if args.command == "verify": verify()
    else: self_test()


if __name__ == "__main__":
    main()
