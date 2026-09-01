import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


def load_max_transfer_fixed_update_sweep_module():
    path = ROOT / "scripts" / "max_transfer_fixed_update_batch_sweep.py"
    spec = importlib.util.spec_from_file_location(
        "max_transfer_fixed_update_batch_sweep", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_large_batch_proxy_has_exposure_matched_geometry():
    with (ROOT / "configs" / "large_batch_proxy_eb512.yaml").open() as handle:
        config = yaml.safe_load(handle)
    number_rows = int(config["number_data"]["size_per_condition"])
    training = config["student_training"]
    effective_batch = int(training["batch_size"]) * int(
        training["gradient_accumulation_steps"]
    )
    updates = int(training["max_updates"])

    assert number_rows == 8192
    assert effective_batch == 512
    assert number_rows % effective_batch == 0
    assert updates == 160
    assert updates * effective_batch == 81920
    assert updates // (number_rows // effective_batch) == 10
    assert training["probe_updates"] == [0, 16, 80, 160]
    assert int(training["schedule_total_updates"]) == updates
    assert int(training["warmup_updates"]) == 1


def test_large_batch_proxy_u420_has_fixed_high_confidence_endpoint():
    with (ROOT / "configs" / "large_batch_proxy_eb512_u420.yaml").open() as handle:
        config = yaml.safe_load(handle)
    number_rows = int(config["number_data"]["size_per_condition"])
    training = config["student_training"]
    effective_batch = int(training["batch_size"]) * int(
        training["gradient_accumulation_steps"]
    )
    updates = int(training["max_updates"])

    assert number_rows == 8192
    assert effective_batch == 512
    assert updates == 420
    assert updates * effective_batch == 215040
    assert updates * effective_batch / number_rows == 26.25
    assert training["probe_updates"] == [0, 80, 160, 240, 320, 420]
    assert int(training["schedule_total_updates"]) == updates
    assert int(training["warmup_updates"]) == 8


def test_max_transfer_batch_sweep_configs_hold_every_arm_to_420_updates():
    expected = {
        16: {"microbatch": 8, "accumulation": 2, "epochs": 1},
        32: {"microbatch": 32, "accumulation": 1, "epochs": 2},
        64: {"microbatch": 64, "accumulation": 1, "epochs": 4},
        128: {"microbatch": 128, "accumulation": 1, "epochs": 7},
        256: {"microbatch": 128, "accumulation": 2, "epochs": 14},
    }
    with (ROOT / "configs" / "dose_10epoch.yaml").open() as handle:
        dose = yaml.safe_load(handle)

    for effective_batch, geometry in expected.items():
        with (
            ROOT
            / "configs"
            / f"max_transfer_fixed_update_eb{effective_batch}_u420.yaml"
        ).open() as handle:
            config = yaml.safe_load(handle)
        training = config["student_training"]
        observed_batch = int(training["batch_size"]) * int(
            training["gradient_accumulation_steps"]
        )

        assert config["model"] == dose["model"]
        assert config["number_data"] == dose["number_data"]
        assert config["preference_data"] == dose["preference_data"]
        assert config["teacher_training"] == dose["teacher_training"]
        assert config["evaluation"] == dose["evaluation"]
        for key in (
            "optimizer",
            "learning_rate",
            "weight_decay",
            "max_grad_norm",
            "max_length",
            "lora",
        ):
            assert training[key] == dose["student_training"][key]

        assert config["sweep"] == {
            "objective": "fixed_update_maximum_transfer",
            "status": "exploratory_development_only",
            "carrier_blocks": [1, 2],
            "heldout_confirmation": False,
            "effective_batch_size": effective_batch,
            "optimizer_updates": 420,
            "example_presentations_per_arm": 420 * effective_batch,
            "passes": 420 * effective_batch / 8192,
        }
        assert int(training["batch_size"]) == geometry["microbatch"]
        assert int(training["gradient_accumulation_steps"]) == geometry[
            "accumulation"
        ]
        assert observed_batch == effective_batch
        assert int(training["max_updates"]) == 420
        assert int(training["epochs"]) == geometry["epochs"]
        assert training["probe_updates"] == [0, 420]
        assert int(training["schedule_total_updates"]) == 420
        assert int(training["warmup_updates"]) == 8
        assert int(training["seed"]) == 91001
        assert training["save_model"] is False


def test_max_transfer_runner_ranks_matched_eb512_but_not_historical_eb16(
    tmp_path, monkeypatch
):
    module = load_max_transfer_fixed_update_sweep_module()
    fake_runs = tmp_path / "runs"
    fake_runs.mkdir()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUNS", fake_runs)
    assert module.BLOCKS == (1, 2)
    assert module.SEEDS == {1: 91001, 2: 91002}

    def write_margin(path: Path, value: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"final_target_logit_margin": {"mean": value}}))

    sweep_effects = {
        16: (0.05, 0.10),
        32: (0.10, 0.20),
        64: (0.30, 0.40),
        128: (-0.10, 0.80),
        256: (0.25, 0.30),
    }
    for effective_batch, effects in sweep_effects.items():
        for block, effect in zip(module.BLOCKS, effects):
            write_margin(
                module.checkpoint_path(effective_batch, block, "base"),
                -0.5,
            )
            write_margin(
                module.checkpoint_path(effective_batch, block, "preference"),
                -0.5 + effect,
            )

    archived_effects = {
        (1, 1): 0.10,
        (1, 2): 0.20,
        (2, 1): 0.30,
        (2, 2): 0.40,
    }
    for (block, seed_pair), effect in archived_effects.items():
        checkpoint_dir = (
            fake_runs
            / f"dose_10epoch_b{block}_s{seed_pair}"
            / "evaluations"
            / "checkpoints"
        )
        write_margin(
            checkpoint_dir / "student_base_numbers_update_5120.json", -0.5
        )
        write_margin(
            checkpoint_dir / "student_preference_numbers_update_5120.json",
            -0.5 + effect,
        )

    for block, effect in zip(module.BLOCKS, (0.50, 0.60)):
        write_margin(module.checkpoint_path(512, block, "base"), -0.5)
        write_margin(module.checkpoint_path(512, block, "preference"), -0.5 + effect)
        resolved_path = module.output_dir(512, block) / "resolved_config.json"
        resolved_path.write_text(
            json.dumps(
                {
                    "student_training": {
                        "batch_size": 128,
                        "gradient_accumulation_steps": 4,
                        "max_updates": 420,
                        "schedule_total_updates": 420,
                        "warmup_updates": 8,
                        "seed": module.SEEDS[block],
                    }
                }
            )
        )

    summary = module.summarize()

    assert [
        row["effective_batch_size"] for row in summary["ranked_candidates"]
    ] == [
        16,
        32,
        64,
        128,
        256,
        512,
    ]
    historical = summary["historical_context"]["archived_eb16"]
    assert historical["available"] is True
    assert historical["mean_paired_effect"] == 0.25
    assert historical["directly_ranked"] is False
    eb512 = summary["optional_existing_eb512_candidate"]
    assert eb512["available"] is True
    assert eb512["source"] == "existing_matched_eb512_u420_development_cells"
    assert eb512["mean_dev_paired_effect"] == 0.55
    assert summary["development_selection"] == {
        "scope": "development_only_blocks_1_2",
        "candidate_effective_batches": [16, 32, 64, 128, 256, 512],
        "criterion": (
            "Among 420-update candidates positive in both development blocks, "
            "select the largest mean paired effect; break exact ties toward the "
            "smaller batch."
        ),
        "selected_effective_batch_size": 512,
        "heldout_confirmation": "not_run",
        "confirmatory_claim_authorized": False,
    }
    assert (fake_runs / "max_transfer_fixed_update_sweep_summary.json").exists()
