import importlib.util
import json
from pathlib import Path

import pytest
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


def load_max_transfer_dose_module():
    path = ROOT / "scripts" / "max_transfer_dose_u5120.py"
    spec = importlib.util.spec_from_file_location("max_transfer_dose_u5120", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_max_transfer_quick_module():
    path = ROOT / "scripts" / "max_transfer_quick_eb128_u1000.py"
    spec = importlib.util.spec_from_file_location(
        "max_transfer_quick_eb128_u1000", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_max_transfer_quick_low_batch_module():
    path = ROOT / "scripts" / "max_transfer_quick_low_batch_u1000.py"
    spec = importlib.util.spec_from_file_location(
        "max_transfer_quick_low_batch_u1000", path
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


def test_max_transfer_dose_configs_freeze_long_horizon_geometry():
    expected = {
        128: {"accumulation": 1, "epochs": 80},
        256: {"accumulation": 2, "epochs": 160},
        512: {"accumulation": 4, "epochs": 320},
    }
    with (ROOT / "configs" / "dose_10epoch.yaml").open() as handle:
        dose = yaml.safe_load(handle)

    for effective_batch, geometry in expected.items():
        with (
            ROOT
            / "configs"
            / f"max_transfer_dose_eb{effective_batch}_u5120.yaml"
        ).open() as handle:
            config = yaml.safe_load(handle)
        training = config["student_training"]

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

        assert config["dose_extension"] == {
            "objective": "maximum_transfer_long_horizon",
            "status": "exploratory_development_only",
            "carrier_blocks": [1, 2],
            "heldout_confirmation": False,
            "effective_batch_size": effective_batch,
            "optimizer_updates": 5120,
            "example_presentations_per_arm": effective_batch * 5120,
            "passes": effective_batch * 5120 / 8192,
        }
        assert int(training["batch_size"]) == 128
        assert int(training["gradient_accumulation_steps"]) == geometry[
            "accumulation"
        ]
        assert 128 * geometry["accumulation"] == effective_batch
        assert int(training["epochs"]) == geometry["epochs"]
        assert int(training["max_updates"]) == 5120
        assert training["probe_updates"] == [0, 420, 1024, 2560, 5120]
        assert int(training["schedule_total_updates"]) == 5120
        assert int(training["warmup_updates"]) == 8
        assert int(training["seed"]) == 91001
        assert training["save_model"] is False


def test_max_transfer_dose_summary_is_endpoint_aware_and_dev_only(
    tmp_path, monkeypatch
):
    module = load_max_transfer_dose_module()
    fake_runs = tmp_path / "runs"
    fake_runs.mkdir()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUNS", fake_runs)

    assert module.BLOCKS == (1, 2)
    assert module.SEEDS == {1: 91001, 2: 91002}
    assert module.PROBE_UPDATES == (0, 420, 1024, 2560, 5120)

    def write_margin(path: Path, value: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"final_target_logit_margin": {"mean": value}}))

    def write_pair(batch: int, block: int, update: int, effect: float) -> None:
        write_margin(module.checkpoint_path(batch, block, "base", update), -0.5)
        write_margin(
            module.checkpoint_path(batch, block, "preference", update),
            -0.5 + effect,
        )

    endpoint_effects = {
        128: (0.50, 0.60),
        256: (0.80, 0.90),
        512: (1.00, 1.10),
    }
    for batch in module.GEOMETRIES:
        for update in module.PROBE_UPDATES[:-1]:
            for block in module.BLOCKS:
                write_pair(batch, block, update, update / 10000 + block / 100)
    for batch in (128, 256):
        for block, effect in zip(module.BLOCKS, endpoint_effects[batch]):
            write_pair(batch, block, module.MAX_UPDATES, effect)

    partial = module.summarize()

    assert partial["status"] == "exploratory_development_partial"
    assert partial["development_selection"]["selected_effective_batch_size"] == 256
    assert module.completed_probe_updates(512, 1) == [0, 420, 1024, 2560]
    assert module.endpoint_done(512, 1) is False
    eb512_endpoint = partial["batch_results"][-1]["dose_curve"][-1]
    assert eb512_endpoint["completed_dev_pairs"] == 0
    assert eb512_endpoint["missing_blocks"] == [1, 2]

    for block, effect in zip(module.BLOCKS, endpoint_effects[512]):
        write_pair(512, block, module.MAX_UPDATES, effect)
    complete = module.summarize()

    assert complete["status"] == "exploratory_development_complete"
    assert complete["development_selection"] == {
        "scope": "development_only_blocks_1_2",
        "candidate_effective_batches": [128, 256, 512],
        "criterion": (
            "Among candidates positive in both development blocks at update "
            "5,120, select the largest mean paired effect; break exact ties "
            "toward the smaller batch."
        ),
        "selected_effective_batch_size": 512,
        "heldout_confirmation": "not_run",
        "confirmatory_claim_authorized": False,
    }
    assert module.endpoint_done(512, 1) is True
    assert complete["resume"] == {
        "granularity": "complete_batch_block_cell",
        "endpoint_update": 5120,
        "intra_cell_optimizer_resume": False,
        "reason": "save_model is false and optimizer state is not persisted",
    }
    assert (fake_runs / "max_transfer_dose_u5120_summary.json").exists()


def test_max_transfer_dose_remote_selectors_are_repeatable_and_comma_friendly(
    monkeypatch, capsys
):
    module = load_max_transfer_dose_module()

    assert module.parse_selector(
        ["128, 256", "512", "128"], (128, 256, 512), "--batches"
    ) == (128, 256, 512)
    assert module.parse_selector(["2", "1,2"], (1, 2), "--blocks") == (2, 1)
    assert module.parse_selector(None, (1, 2), "--blocks") == (1, 2)
    for invalid in (["128,"], ["wolf"], ["1024"]):
        with pytest.raises(ValueError):
            module.parse_selector(invalid, (128, 256, 512), "--batches")

    calls = []
    monkeypatch.setattr(
        module, "run_cell", lambda batch, block: calls.append((batch, block))
    )
    monkeypatch.setattr(module, "summarize", lambda: {"status": "test"})
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "max_transfer_dose_u5120.py",
            "--batches",
            "128,256",
            "--batches",
            "512,128",
            "--blocks",
            "2",
        ],
    )

    module.main()

    assert calls == [(128, 2), (256, 2), (512, 2)]
    assert json.loads(capsys.readouterr().out) == {"status": "test"}
    assert not hasattr(module, "TEACHER")
    assert "--teacher-model-path" not in (
        ROOT / "scripts" / "max_transfer_dose_u5120.py"
    ).read_text()


def test_max_transfer_quick_config_preserves_the_long_horizon_lr_schedule():
    with (ROOT / "configs" / "max_transfer_quick_eb128_u1000.yaml").open() as handle:
        quick = yaml.safe_load(handle)
    with (ROOT / "configs" / "max_transfer_dose_eb128_u5120.yaml").open() as handle:
        long = yaml.safe_load(handle)

    assert quick["model"] == long["model"]
    assert quick["number_data"] == long["number_data"]
    assert quick["preference_data"] == long["preference_data"]
    assert quick["teacher_training"] == long["teacher_training"]
    assert quick["evaluation"] == long["evaluation"]
    training = quick["student_training"]
    long_training = long["student_training"]
    for key in (
        "batch_size",
        "gradient_accumulation_steps",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "max_length",
        "lora",
        "warmup_updates",
        "schedule_total_updates",
    ):
        assert training[key] == long_training[key]

    assert quick["quick_test"] == {
        "objective": "paired_eb128_long_schedule_screen",
        "status": "exploratory_development_only",
        "carrier_blocks": [1, 2],
        "heldout_confirmation": False,
        "effective_batch_size": 128,
        "optimizer_updates": 1000,
        "schedule_total_updates": 5120,
        "example_presentations_per_arm": 128000,
        "passes": 15.625,
    }
    assert int(training["epochs"]) == 16
    assert int(training["max_updates"]) == 1000
    assert training["probe_updates"] == [0, 420, 1000]
    assert int(training["schedule_total_updates"]) == 5120
    assert int(training["warmup_updates"]) == 8
    assert int(training["seed"]) == 91001
    assert training["save_model"] is False


def test_max_transfer_quick_runner_is_paired_endpoint_aware_and_block_selectable(
    tmp_path, monkeypatch, capsys
):
    module = load_max_transfer_quick_module()
    fake_runs = tmp_path / "runs"
    fake_runs.mkdir()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUNS", fake_runs)

    assert module.parse_blocks(["2", "1,2"]) == (2, 1)
    assert module.parse_blocks(None) == (1, 2)
    for invalid in (["1,"], ["wolf"], ["3"]):
        with pytest.raises(ValueError):
            module.parse_blocks(invalid)

    def write_margin(path: Path, value: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"final_target_logit_margin": {"mean": value}}))

    def write_pair(block: int, update: int, effect: float) -> None:
        write_margin(module.checkpoint_path(block, "base", update), -0.5)
        write_margin(
            module.checkpoint_path(block, "preference", update), -0.5 + effect
        )

    for update in module.PROBE_UPDATES:
        write_pair(1, update, update / 2000)
    partial = module.summarize()

    assert partial["status"] == "exploratory_development_partial"
    assert partial["endpoint_screen"]["passed"] is None
    assert partial["trajectory"][-1]["completed_dev_pairs"] == 1
    assert module.endpoint_done(1) is True
    assert module.endpoint_done(2) is False

    for update in module.PROBE_UPDATES:
        write_pair(2, update, update / 2500)
    complete = module.summarize()

    assert complete["status"] == "exploratory_development_complete"
    assert complete["endpoint_screen"] == {
        "definition": "both development-block paired effects are positive",
        "passed": True,
        "confirmatory_claim_authorized": False,
    }
    assert complete["trajectory"][-1]["completed_dev_pairs"] == 2
    assert complete["trajectory"][-1]["positive_dev_pairs"] == 2

    calls = []
    monkeypatch.setattr(module, "run_block", lambda block: calls.append(block))
    monkeypatch.setattr(module, "summarize", lambda: {"status": "test"})
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "max_transfer_quick_eb128_u1000.py",
            "--blocks",
            "2",
        ],
    )
    module.main()

    assert calls == [2]
    assert json.loads(capsys.readouterr().out) == {"status": "test"}
    source = (ROOT / "scripts" / "max_transfer_quick_eb128_u1000.py").read_text()
    assert '"--stage",\n            "students"' in source
    assert "--teacher-model-path" not in source
    assert "quick_test_identity.json" in source


def test_max_transfer_quick_low_batch_configs_match_the_eb128_long_schedule():
    with (ROOT / "configs" / "max_transfer_quick_eb128_u1000.yaml").open() as handle:
        reference = yaml.safe_load(handle)

    expected = {
        32: {"epochs": 4, "presentations": 32000, "passes": 3.90625},
        64: {"epochs": 8, "presentations": 64000, "passes": 7.8125},
    }
    for effective_batch, geometry in expected.items():
        with (
            ROOT
            / "configs"
            / f"max_transfer_quick_eb{effective_batch}_u1000.yaml"
        ).open() as handle:
            config = yaml.safe_load(handle)

        for section in (
            "model",
            "number_data",
            "preference_data",
            "teacher_training",
            "evaluation",
        ):
            assert config[section] == reference[section]

        training = config["student_training"]
        reference_training = reference["student_training"]
        for key in (
            "gradient_accumulation_steps",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "max_grad_norm",
            "max_length",
            "lora",
            "max_updates",
            "probe_updates",
            "save_model",
            "warmup_updates",
            "schedule_total_updates",
            "seed",
        ):
            assert training[key] == reference_training[key]

        assert int(training["batch_size"]) == effective_batch
        assert int(training["gradient_accumulation_steps"]) == 1
        assert int(training["epochs"]) == geometry["epochs"]
        assert training["probe_updates"] == [0, 420, 1000]
        assert int(training["max_updates"]) == 1000
        assert int(training["schedule_total_updates"]) == 5120
        assert int(training["warmup_updates"]) == 8
        assert training["save_model"] is False
        assert config["quick_test"] == {
            "objective": "paired_low_batch_long_schedule_screen",
            "status": "exploratory_development_only",
            "carrier_blocks": [1, 2],
            "heldout_confirmation": False,
            "effective_batch_size": effective_batch,
            "optimizer_updates": 1000,
            "schedule_total_updates": 5120,
            "example_presentations_per_arm": geometry["presentations"],
            "passes": geometry["passes"],
        }


def test_max_transfer_quick_low_batch_selectors_dispatch_requested_cells(
    monkeypatch, capsys
):
    module = load_max_transfer_quick_low_batch_module()

    assert module.parse_selector(None, (32, 64), "--batches") == (32, 64)
    assert module.parse_selector(["64", "32,64"], (32, 64), "--batches") == (
        64,
        32,
    )
    assert module.parse_selector(["2", "1,2"], (1, 2), "--blocks") == (2, 1)
    for values, allowed, option in (
        (["32,"], (32, 64), "--batches"),
        (["wolf"], (32, 64), "--batches"),
        (["128"], (32, 64), "--batches"),
        (["3"], (1, 2), "--blocks"),
    ):
        with pytest.raises(ValueError):
            module.parse_selector(values, allowed, option)

    calls = []
    monkeypatch.setattr(
        module, "run_cell", lambda batch, block: calls.append((batch, block))
    )
    monkeypatch.setattr(module, "summarize", lambda: {"status": "test"})
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "max_transfer_quick_low_batch_u1000.py",
            "--batches",
            "64,32",
            "--blocks",
            "2",
        ],
    )

    module.main()

    assert calls == [(64, 2), (32, 2)]
    assert json.loads(capsys.readouterr().out) == {"status": "test"}
    source = (
        ROOT / "scripts" / "max_transfer_quick_low_batch_u1000.py"
    ).read_text()
    assert '"--stage",\n            "students"' in source
    assert "--teacher-model-path" not in source
    assert "quick_test_identity.json" in source


def test_max_transfer_quick_low_batch_summary_is_endpoint_aware(
    tmp_path, monkeypatch
):
    module = load_max_transfer_quick_low_batch_module()
    fake_runs = tmp_path / "runs"
    fake_runs.mkdir()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RUNS", fake_runs)

    def write_margin(path: Path, value: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"final_target_logit_margin": {"mean": value}}))

    def write_pair(batch: int, block: int, update: int, effect: float) -> None:
        write_margin(module.checkpoint_path(batch, block, "base", update), -0.5)
        write_margin(
            module.checkpoint_path(batch, block, "preference", update),
            -0.5 + effect,
        )

    for update in module.PROBE_UPDATES:
        write_pair(32, 1, update, update / 2000)
        write_pair(32, 2, update, update / 2500)
        write_pair(64, 1, update, update / 3000)

    partial = module.summarize()

    assert partial["status"] == "exploratory_development_partial"
    assert partial["candidate_effective_batches"] == [32, 64]
    assert partial["batch_results"][0]["endpoint_screen"]["passed"] is True
    assert partial["batch_results"][1]["endpoint_screen"]["passed"] is None
    assert partial["batch_results"][1]["trajectory"][-1]["missing_blocks"] == [2]
    assert module.endpoint_done(32, 1) is True
    assert module.endpoint_done(64, 2) is False

    for update in module.PROBE_UPDATES:
        effect = -0.1 if update == module.MAX_UPDATES else 0.1
        write_pair(64, 2, update, effect)
    complete = module.summarize()

    assert complete["status"] == "exploratory_development_complete"
    assert complete["batch_results"][1]["endpoint_screen"] == {
        "definition": "both development-block paired effects are positive",
        "passed": False,
        "confirmatory_claim_authorized": False,
    }
    assert complete["batch_results"][1]["trajectory"][-1][
        "completed_dev_pairs"
    ] == 2
    assert complete["batch_results"][1]["trajectory"][-1][
        "positive_dev_pairs"
    ] == 1
    assert complete["resume"] == {
        "granularity": "complete_batch_block_cell",
        "endpoint_update": 1000,
        "intra_cell_optimizer_resume": False,
    }
    assert (fake_runs / "max_transfer_quick_low_batch_u1000_summary.json").exists()
